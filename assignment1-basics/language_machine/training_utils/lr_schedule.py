import math


def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Args:
        t: Current iteration
        max_learning_rate: alpha_max, maximum learning rate
        min_learning_rate: alpha_min, minimum/final learning rate
        warmup_iters: T_w, number of warmup iterations
        cosine_cycle_iters: T_c, number of cosine annealing iterations

    Returns:
        Learning rate at iteration t
    """
    # Warm-up: linear increase from 0 to max_learning_rate
    if t < warmup_iters:
        return (t / warmup_iters) * max_learning_rate

    # Post-annealing: constant at min_learning_rate
    if t > cosine_cycle_iters:
        return min_learning_rate

    # Cosine annealing: smooth decay from max to min
    progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
