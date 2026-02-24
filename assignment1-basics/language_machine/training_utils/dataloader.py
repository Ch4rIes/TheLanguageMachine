import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Sample random sequences from dataset for language modeling.
    """
    # Maximum valid starting index (need context_length + 1 tokens: input + 1 target)
    max_start = len(dataset) - context_length - 1

    # Random starting positions
    starting_indices = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = np.array([dataset[i : i + context_length] for i in starting_indices])
    targets = np.array([dataset[i + 1 : i + context_length + 1] for i in starting_indices])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return (inputs, targets)
