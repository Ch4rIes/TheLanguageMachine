import json
import os
import time
from pathlib import Path
from typing import IO, Optional

import numpy as np
import torch

from language_machine.config import TrainingConfig, load_config
from language_machine.training_utils.adamw import AdamW
from language_machine.training_utils.checkpointing import load_checkpoint, save_checkpoint
from language_machine.training_utils.cross_entropy import cross_entropy
from language_machine.training_utils.dataloader import get_batch
from language_machine.training_utils.gradient_clipping import clip_gradient
from language_machine.training_utils.lr_schedule import get_lr_cosine_schedule
from language_machine.transformer.transformer_lm import TransformerLM


def load_dataset(path: str) -> np.ndarray:
    """Load tokenized dataset from file (memory-mapped for large files)."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 20,
) -> float:
    """Evaluate model on dataset, return average loss."""
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)

        # Reshape for cross_entropy: (batch * seq_len, vocab_size) and (batch * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = cross_entropy(logits_flat, targets_flat)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def train(config: TrainingConfig, resume_from: Optional[str] = None, metrics_file: Optional[str] = None) -> None:
    """
    Main training loop.

    Args:
        config: Training configuration
        resume_from: Optional path to checkpoint to resume from
        metrics_file: Optional path to JSONL file for logging metrics
    """
    device = config.device

    metrics_fh: Optional[IO] = None
    if metrics_file:
        Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = open(metrics_file, "a")

    def log_metric(record: dict) -> None:
        if metrics_fh is not None:
            metrics_fh.write(json.dumps(record) + "\n")
            metrics_fh.flush()

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading training data from {config.train_data_path}")
    train_data = load_dataset(config.train_data_path)
    print(f"Training data: {len(train_data):,} tokens")

    val_data = None
    if config.val_data_path:
        print(f"Loading validation data from {config.val_data_path}")
        val_data = load_dataset(config.val_data_path)
        print(f"Validation data: {len(val_data):,} tokens")

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        theta=config.model.theta,
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # Compute scheduler parameters
    max_lr = config.optimizer.lr
    min_lr = max_lr * config.scheduler.min_lr_ratio

    print(f"Starting training for {config.max_iters} iterations...")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Context length: {config.model.context_length}")
    print(f"  Learning rate: {max_lr} -> {min_lr}")
    print(f"  Device: {device}")
    print("-" * 50)

    model.train()

    for iteration in range(start_iter, config.max_iters):
        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            t=iteration,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=config.scheduler.warmup_iters,
            cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
        )
        set_lr(optimizer, lr)

        # Sample batch
        inputs, targets = get_batch(
            train_data,
            config.batch_size,
            config.model.context_length,
            device,
        )

        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)

        # Reshape for cross_entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute loss
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        clip_gradient(model.parameters(), config.grad_clip_norm)

        # Optimizer step
        optimizer.step()

        # Logging
        if iteration % config.log_interval == 0:
            print(f"iter {iteration:6d} | loss {loss.item():.4f} | lr {lr:.2e}")
            log_metric({"iteration": iteration, "train_loss": loss.item(), "lr": lr, "timestamp": time.time()})

        # Validation
        if val_data is not None and iteration % config.val_interval == 0 and iteration > 0:
            val_loss = evaluate(
                model,
                val_data,
                config.batch_size,
                config.model.context_length,
                device,
            )
            print(f"iter {iteration:6d} | val_loss {val_loss:.4f}")
            log_metric({"iteration": iteration, "val_loss": val_loss, "timestamp": time.time()})

        # Checkpointing
        if iteration % config.checkpoint_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, config.max_iters, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")

    if metrics_fh is not None:
        metrics_fh.close()


def train_from_yaml(config_path: str, resume_from: Optional[str] = None, metrics_file: Optional[str] = None) -> None:
    """Load config from YAML and start training."""
    config = load_config(config_path)
    train(config, resume_from, metrics_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--metrics-file", type=str, default=None, help="Path to JSONL file for metrics logging")
    args = parser.parse_args()

    train_from_yaml(args.config, args.resume, args.metrics_file)
