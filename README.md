# The Language Machine

A full-stack language model system(adapted from Stanford's CS336) built from first principles — no `transformers`, no `torch.nn.Transformer`. Every component, from the attention kernel to the optimizer to the BPE algorithm, is implemented from scratch.

The system spans three layers: a **core** model library, a **training infrastructure** with scheduling and checkpointing, and an **experimentation stack** for launching, monitoring, and comparing runs through a live dashboard.

---

## Architecture

```
language-machine/
├── core/               # Model library & training engine
└── infrastructure/     # Experimentation & monitoring stack
```

---

## `core/` — Model Library

The `language_machine` Python package. Every layer is hand-rolled.

### Transformer
- Multi-head self-attention with **Rotary Positional Embeddings (RoPE)**
- **RMSNorm** for pre-normalization
- **SwiGLU** position-wise feed-forward network
- Full autoregressive **TransformerLM**

### Tokenizer
- **Byte-Pair Encoding (BPE)** trained from scratch
- Parallel pre-tokenization across file chunks
- CLI for training, encoding, decoding, and inspection

### Training Infrastructure
- **AdamW** optimizer with decoupled weight decay
- **Cosine annealing** LR schedule with linear warmup
- Gradient norm clipping
- Distributed checkpointing (save/resume)
- Streaming data loader for memory-efficient training
- **W&B** integration and **SLURM/submitit** support for cluster jobs

### Setup

Requires [`uv`](https://github.com/astral-sh/uv).

```sh
cd core
uv run pytest                                        # run test suite
uv run python -m language_machine.tokenizer_cli --help
uv run python -m language_machine.training_loop      # train a model
```

Training data (TinyStories, OpenWebText) is not tracked in git. See `core/README.md` for download instructions.

---

## `infrastructure/` — Experimentation Stack

A full-stack interface for managing training runs end to end.
<img width="830" height="853" alt="PNG image" src="https://github.com/user-attachments/assets/77f043c1-349f-4b75-bc85-edc9734558bc" />

### Backend (FastAPI)
- Launch and terminate training jobs
- Stream live training metrics from log files
- Tokenize text and run autoregressive generation against any checkpoint
- Persistent experiment store with run history

### Frontend (React + TypeScript)
- Live loss curve visualization
- Side-by-side experiment comparison
- Checkpoint selector for generation
- Tokenizer playground

### Start

```sh
cd infrastructure
./start.sh    # starts backend + frontend concurrently
```

---

## Design Philosophy

The goal is full transparency across the stack. No black-box abstractions — if something runs, you can read exactly why. The architecture mirrors production LLM systems at a scale that fits on a single machine or a small cluster, making it a useful substrate for research and experimentation.
