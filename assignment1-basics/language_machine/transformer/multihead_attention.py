import torch
from torch import Tensor, nn

from .linear import Linear
from .rope import RoPE
from .utils.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHead(nn.Module):
    def __init__(
        self, d_model: int, num_heads, max_seq_len: int = 1024 * 16, theta: float = 10000.0, use_rope: bool = True
    ):
        super().__init__()
        self.d_k = d_model // num_heads  # a prior on head complexity of the model
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape

        # QKV projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into heads, transpose for attention calculation
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # RoPE (optional)
        if self.use_rope:
            positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Merge heads: (batch, num_heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # Output projection
        return self.o_proj(attn_output)
