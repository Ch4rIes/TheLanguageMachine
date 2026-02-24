import torch
from torch import Tensor

from language_machine.transformer.utils.softmax import softmax


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits
        targets: (batch_size,) target token indices

    Returns:
        Scalar tensor with mean cross-entropy loss
    """
    # Softmax to get probabilities
    probs = softmax(inputs, dim=-1)

    # Gather the probabilities of the correct classes
    batch_size = inputs.shape[0]
    correct_probs = probs[torch.arange(batch_size), targets]

    # Negative log likelihood, averaged
    return -torch.log(correct_probs).mean()
