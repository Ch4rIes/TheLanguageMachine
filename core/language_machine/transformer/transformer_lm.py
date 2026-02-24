import torch
from torch import nn

from language_machine.transformer.embedding import Embedding
from language_machine.transformer.linear import Linear
from language_machine.transformer.norm import RMSLNorm
from language_machine.transformer.transformer import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, d_model)

        # initialize transformer blocks using ModuleList so params are registered
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=theta)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSLNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        """
        produce next token logits
        """
        x = self.embedding_layer(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = self.ln_final(x)
        next_token_logits = self.lm_head(x)

        return next_token_logits
