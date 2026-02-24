import torch
from torch import Tensor, nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (dim_indices / d_k))  # shape: (d_k/2,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_seq_len, d_k/2)

        # Precompute cos and sin and register as buffers
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
            token_positions: (batch, seq_len) or (seq_len,) - position indices
        Returns:
            rotated x with same shape
        """
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        # Split x into pairs: x[..., 0::2] and x[..., 1::2]
        x_even = x[..., 0::2]  # (..., d_k/2)
        x_odd = x[..., 1::2]  # (..., d_k/2)

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # (..., d_k)

        return x_rotated
