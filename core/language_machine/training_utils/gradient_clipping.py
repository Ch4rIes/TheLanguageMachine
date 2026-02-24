from collections.abc import Iterable

import torch


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: Collection of trainable parameters
        max_l2_norm: Maximum allowed l2 norm for gradients

    Modifies parameter.grad in-place.
    """
    eps = 1e-6

    # Collect all gradients and compute total l2 norm
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)

    if not grads:
        return

    # Compute total l2 norm: sqrt(sum of squared norms)
    total_norm_sq = sum(g.pow(2).sum() for g in grads)
    total_norm = total_norm_sq.sqrt()

    # Clip if necessary
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
