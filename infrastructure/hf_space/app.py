import json
import logging
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# language_machine is in the same directory as this file
from language_machine.generate import generate_text
from language_machine.transformer.transformer_lm import TransformerLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REPO = os.environ.get("MODEL_REPO", "czuo/lm_v0")
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "weights/checkpoint_final.pt")
TOKENIZER_FILE = os.environ.get("TOKENIZER_FILE", "weights/tokenizer.json")


def load_tokenizer(path: str):
    with open(path) as f:
        raw = json.load(f)
    vocab = {int(k): bytes(v) for k, v in raw["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in raw["merges"]]
    return vocab, merges


def load_model(checkpoint_path: str, tokenizer_vocab_size: int):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    embed_w = model_state["embedding_layer.W"]
    ckpt_vocab_size, d_model = embed_w.shape

    if ckpt_vocab_size != tokenizer_vocab_size:
        model_state["embedding_layer.W"] = model_state["embedding_layer.W"][:tokenizer_vocab_size]
        model_state["lm_head.W"] = model_state["lm_head.W"][:, :tokenizer_vocab_size]

    num_layers = sum(
        1 for k in model_state
        if k.startswith("transformer_blocks.") and k.endswith(".norm1.weight")
    )
    num_heads = ckpt.get("num_heads", max(1, d_model // 64))
    context_length = ckpt.get("context_length", 256)

    model = TransformerLM(
        vocab_size=tokenizer_vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=0,
    )
    model.load_state_dict(model_state)
    model.context_length = context_length
    model.eval()
    return model


logger.info(f"Downloading model files from {MODEL_REPO}...")
_checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
_tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename=TOKENIZER_FILE)

vocab, merges = load_tokenizer(_tokenizer_path)
model = load_model(_checkpoint_path, len(vocab))
logger.info(f"Model ready: {sum(p.numel() for p in model.parameters()):,} params")

# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Language Machine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest):
    try:
        text = generate_text(
            model=model,
            vocab=vocab,
            merges=merges,
            prompt=body.prompt,
            max_new_tokens=min(body.max_new_tokens, 500),
            temperature=body.temperature,
            top_p=body.top_p,
            device="cpu",
        )
        return GenerateResponse(text=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
