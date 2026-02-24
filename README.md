# The Language Machine

A full-stack language model system built from scratch — covering transformer architecture, training infrastructure, tokenization, and an experimentation dashboard.

## Structure

```
.
├── assignment1-basics/   # Core model implementation
└── dashboard/            # Experimentation & monitoring stack
```

### `assignment1-basics/` — Model Core

The `language_machine` Python package implements every component from first principles:

- **Transformer** — multi-head attention (with RoPE), RMSNorm, SwiGLU feed-forward, full transformer LM
- **Tokenizer** — BPE tokenizer with parallel pre-tokenization (`tokenizer/`)
- **Training utilities** — AdamW optimizer, cosine LR schedule, gradient clipping, checkpointing, data loader
- **Training loop** — end-to-end training with W&B logging and SLURM/submitit support
- **Generation** — autoregressive text generation with temperature/top-k sampling

#### Setup

Requires [`uv`](https://github.com/astral-sh/uv).

```sh
cd assignment1-basics
uv run pytest          # run tests
uv run python -m language_machine.tokenizer_cli --help
uv run python -m language_machine.training_loop
```

Data (TinyStories, OpenWebText sample) is not tracked — see `assignment1-basics/README.md` for download instructions.

### `dashboard/` — Experimentation Stack

A full-stack experiment management interface:

- **Backend** — FastAPI server for launching/monitoring training runs, tokenization, and text generation
- **Frontend** — React + TypeScript UI for experiment tracking and live metrics
- **Process manager** — handles concurrent training job lifecycle

```sh
cd dashboard
./start.sh
```

## Philosophy

Every component — from the attention kernel to the optimizer to the tokenization algorithm — is implemented from scratch. No `transformers`, no `torch.nn.Transformer`. Just tensors and math.
