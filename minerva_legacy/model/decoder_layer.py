import torch
import torch.nn as nn
from ..modules import (
    CausalSelfAttention,
    HighwayResidual,
    SwiGLU,
    ReZeroResidual,
)


class DecoderLayer(nn.Module):
    """Transformer decoder layer integrating the *green check* components.

    Composition:
        LN → Self-Attention → HighwayResidual
        LN → SwiGLU FFN      → ReZeroResidual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        expansion: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.highway = HighwayResidual(hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, expansion, dropout)
        self.rezero = ReZeroResidual()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + highway residual
        h_attn = self.attn(self.norm1(x))
        x = self.highway(x, h_attn)

        # Feed-forward + ReZero residual
        h_ffn = self.ffn(self.norm2(x))
        x = self.rezero(x, h_ffn)
        return x 