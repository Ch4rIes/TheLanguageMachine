import torch
from torch import Tensor, nn


class SwiGLUFeedForward(torch.nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        d_ff = ((int(d_model * 8 / 3) >> 6) + 1) << 6
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        std = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.W3, mean=0, std=std, a=-3 * std, b=3 * std)

    @staticmethod
    def SiLU(x):
        return x * torch.sigmoid(x)

    def forward(self, X: Tensor) -> Tensor:
        # Silu
        return (SwiGLUFeedForward.SiLU(X @ self.W1) * (X @ self.W3)) @ self.W2
