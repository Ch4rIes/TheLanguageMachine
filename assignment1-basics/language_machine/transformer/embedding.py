import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3, b=3)

    def forward(self, X: Tensor) -> Tensor:
        return self.W[X]
