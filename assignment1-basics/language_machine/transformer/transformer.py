import torch
from torch import Tensor, nn

from language_machine.transformer.multihead_attention import MultiHead
from language_machine.transformer.norm import RMSLNorm
from language_machine.transformer.position_wise_feed_forward import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 1024 * 16,
        theta: float = 10000.0,
        use_rope=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSLNorm(d_model)
        self.multihead_attention_layer = MultiHead(
            d_model, num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=use_rope
        )
        self.position_wise_feed_forward_layer = SwiGLUFeedForward(d_model, device, dtype)
        self.norm2 = RMSLNorm(d_model)

    def forward(self, x) -> Tensor:
        x = self.multihead_attention_layer(self.norm1(x)) + x
        x = self.position_wise_feed_forward_layer(self.norm2(x)) + x
        return x
