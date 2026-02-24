from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1024
    theta: float = 10000.0


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    warmup_iters: int = 100
    cosine_cycle_iters: int = 10000
    min_lr_ratio: float = 0.1  # min_lr = lr * min_lr_ratio


@dataclass
class TrainingConfig:
    # Data
    train_data_path: str = ""
    val_data_path: Optional[str] = None

    # Training
    batch_size: int = 32
    max_iters: int = 10000
    grad_clip_norm: float = 1.0

    # Logging & Checkpointing
    log_interval: int = 10
    val_interval: int = 100
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cpu"

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


def load_config(path: str) -> TrainingConfig:
    """Load config from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config = TrainingConfig()

    # Top-level fields
    for key in [
        "train_data_path",
        "val_data_path",
        "batch_size",
        "max_iters",
        "grad_clip_norm",
        "log_interval",
        "val_interval",
        "checkpoint_interval",
        "checkpoint_dir",
        "device",
    ]:
        if key in data:
            setattr(config, key, data[key])

    # Model config
    if "model" in data:
        config.model = ModelConfig(**data["model"])

    # Optimizer config
    if "optimizer" in data:
        opt_data = data["optimizer"]
        if "betas" in opt_data:
            opt_data["betas"] = tuple(opt_data["betas"])
        config.optimizer = OptimizerConfig(**opt_data)

    # Scheduler config
    if "scheduler" in data:
        config.scheduler = SchedulerConfig(**data["scheduler"])

    return config


def save_config(config: TrainingConfig, path: str) -> None:
    """Save config to YAML file."""
    data = {
        "train_data_path": config.train_data_path,
        "val_data_path": config.val_data_path,
        "batch_size": config.batch_size,
        "max_iters": config.max_iters,
        "grad_clip_norm": config.grad_clip_norm,
        "log_interval": config.log_interval,
        "val_interval": config.val_interval,
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_dir": config.checkpoint_dir,
        "device": config.device,
        "model": {
            "vocab_size": config.model.vocab_size,
            "context_length": config.model.context_length,
            "num_layers": config.model.num_layers,
            "d_model": config.model.d_model,
            "num_heads": config.model.num_heads,
            "d_ff": config.model.d_ff,
            "theta": config.model.theta,
        },
        "optimizer": {
            "lr": config.optimizer.lr,
            "betas": list(config.optimizer.betas),
            "eps": config.optimizer.eps,
            "weight_decay": config.optimizer.weight_decay,
        },
        "scheduler": {
            "warmup_iters": config.scheduler.warmup_iters,
            "cosine_cycle_iters": config.scheduler.cosine_cycle_iters,
            "min_lr_ratio": config.scheduler.min_lr_ratio,
        },
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
