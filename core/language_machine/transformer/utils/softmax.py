from torch import Tensor


def softmax(t: Tensor, dim: int) -> Tensor:
    """
    Normalize along dimension `dim` of tensor t
    """
    t_shifted = t - t.max(dim=dim, keepdim=True).values
    exp_t = t_shifted.exp()
    return exp_t / exp_t.sum(dim=dim, keepdim=True)
