from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import psutil
import torch

from language_machine.training_utils.adamw import AdamW
from language_machine.training_utils.cross_entropy import cross_entropy
from language_machine.transformer.transformer_lm import TransformerLM

ProfileMode = Literal["forward", "forward_backward", "forward_backward_optimizer"]


@dataclass
class StepProfileConfig:
    device: str = "cpu"
    batch_size: int = 8
    warmup_steps: int = 5
    profile_steps: int = 20
    vocab_size: int = 10000
    context_length: int = 128
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024
    theta: float = 10000.0
    lr: float = 1e-3
    weight_decay: float = 0.01


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _validate_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested, but torch.backends.mps is not available")
    return torch.device(device)


def _gpu_memory_mb(device: str) -> dict[str, float | None]:
    if device.startswith("cuda") and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
    return {
        "allocated_mb": None,
        "reserved_mb": None,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
    }


def collect_hardware(device: str) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cpu": {
            "model": platform.processor() or "unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_gb": round(memory.total / 1024**3, 2),
        },
        "gpu": None,
    }

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        hardware["gpu"] = {
            "backend": "cuda",
            "name": props.name,
            "index": idx,
            "total_memory_mb": round(props.total_memory / 1024**2, 1),
            "cuda_version": torch.version.cuda,
            "capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    elif device == "mps" and torch.backends.mps.is_available():
        hardware["gpu"] = {
            "backend": "mps",
            "name": "Apple GPU",
            "total_memory_mb": None,
            "cuda_version": None,
            "capability": None,
        }

    return hardware


def _make_batch(config: StepProfileConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        dtype=torch.long,
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


def _make_model(config: StepProfileConfig, device: torch.device) -> TransformerLM:
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        theta=config.theta,
    )
    model.context_length = config.context_length
    return model.to(device)


def _run_step(
    mode: ProfileMode,
    model: TransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float | None:
    optimizer.zero_grad(set_to_none=True)

    if mode == "forward":
        with torch.no_grad():
            model(inputs)
        return None

    logits = model(inputs)
    loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    if mode == "forward_backward_optimizer":
        optimizer.step()

    return float(loss.item())


def _summarize(times_ms: list[float], tokens_per_step: int) -> dict[str, float]:
    sorted_times = sorted(times_ms)
    p95_index = max(0, min(len(sorted_times) - 1, int(len(sorted_times) * 0.95) - 1))
    mean_ms = statistics.fmean(times_ms)
    return {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(times_ms),
        "p95_ms": sorted_times[p95_index],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "tokens_per_sec": tokens_per_step / (mean_ms / 1000),
    }


def profile_mode(config: StepProfileConfig, mode: ProfileMode) -> dict[str, Any]:
    device = _validate_device(config.device)
    model = _make_model(config, device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    inputs, targets = _make_batch(config, device)
    tokens_per_step = config.batch_size * config.context_length
    parameter_count = sum(p.numel() for p in model.parameters())

    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(config.warmup_steps):
        _run_step(mode, model, optimizer, inputs, targets)
    _synchronize(config.device)

    times_ms: list[float] = []
    losses: list[float] = []

    for _ in range(config.profile_steps):
        _synchronize(config.device)
        start = time.perf_counter()
        loss = _run_step(mode, model, optimizer, inputs, targets)
        _synchronize(config.device)
        times_ms.append((time.perf_counter() - start) * 1000)
        if loss is not None:
            losses.append(loss)

    result = {
        "mode": mode,
        "steps": config.profile_steps,
        "parameter_count": parameter_count,
        "tokens_per_step": tokens_per_step,
        **_summarize(times_ms, tokens_per_step),
        **_gpu_memory_mb(config.device),
    }
    if losses:
        result["last_loss"] = losses[-1]

    return result


def run_step_profile(config: StepProfileConfig) -> dict[str, Any]:
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if config.batch_size < 1 or config.context_length < 1:
        raise ValueError("batch_size and context_length must be positive")
    if config.warmup_steps < 0 or config.profile_steps < 1:
        raise ValueError("warmup_steps must be >= 0 and profile_steps must be >= 1")

    modes: list[ProfileMode] = ["forward", "forward_backward", "forward_backward_optimizer"]
    results = [profile_mode(config, mode) for mode in modes]

    return {
        "config": asdict(config),
        "hardware": collect_hardware(config.device),
        "results": results,
    }
