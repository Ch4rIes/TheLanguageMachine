import os
from typing import IO, BinaryIO, Union

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Save model, optimizer, and iteration to a checkpoint file.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: Current training iteration
        out: Path or file-like object to save to
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        src: Path or file-like object to load from
        model: The model to restore state to
        optimizer: The optimizer to restore state to

    Returns:
        The iteration number saved in the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
