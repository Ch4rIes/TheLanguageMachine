import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.W
