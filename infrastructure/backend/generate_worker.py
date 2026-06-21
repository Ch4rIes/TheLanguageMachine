import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ASSIGNMENT_DIR = Path(__file__).parent.parent.parent / "core"

_executor = ThreadPoolExecutor(max_workers=1)


def _ensure_path() -> None:
    """Add core to sys.path if needed."""
    p = str(ASSIGNMENT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path) as f:
        raw = json.load(f)
    vocab = {int(k): bytes(v) for k, v in raw["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in raw["merges"]]
    return vocab, merges


def _run_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    # Lazy import: only runs inside ThreadPoolExecutor, avoids module-load errors
    # when language_machine is not pip-installed (its __init__.py calls
    # importlib.metadata.version which requires package metadata).
    _ensure_path()
    from language_machine.generate import generate_text  # noqa: PLC0415
    from language_machine.transformer.transformer_lm import TransformerLM  # noqa: PLC0415

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    vocab, merges = _load_tokenizer(tokenizer_path)
    tokenizer_vocab_size = len(vocab)

    # Infer model hyperparams from state dict shapes
    embed_w = model_state["embedding_layer.W"]
    ckpt_vocab_size, d_model = embed_w.shape

    # If the checkpoint vocab is larger than the tokenizer (e.g. config defaulted to
    # 32000 but tokenizer only has 10000 tokens), truncate so generation stays in range.
    if ckpt_vocab_size != tokenizer_vocab_size:
        model_state["embedding_layer.W"] = model_state["embedding_layer.W"][:tokenizer_vocab_size]
        model_state["lm_head.W"] = model_state["lm_head.W"][:, :tokenizer_vocab_size]
    vocab_size = tokenizer_vocab_size

    num_layers = sum(
        1 for k in model_state if k.startswith("transformer_blocks.") and k.endswith(".norm1.weight")
    )

    num_heads = ckpt.get("num_heads", max(1, d_model // 64))

    # d_ff is ignored by SwiGLUFeedForward (it recomputes from d_model), so any value works
    context_length = ckpt.get("context_length", 256)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=0,
    )
    model.load_state_dict(model_state)
    model.context_length = context_length
    model.eval()

    return generate_text(
        model=model,
        vocab=vocab,
        merges=merges,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device="cpu",
    )


async def generate_async(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _run_inference,
        checkpoint_path,
        tokenizer_path,
        prompt,
        max_new_tokens,
        temperature,
        top_p,
    )
