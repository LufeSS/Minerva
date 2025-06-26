import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """Feed-forward block using SwiGLU activation (Shazeer, 2020).

    Formula (channel-wise):
        y = (a * SiLU(b)) W_o
    where (a, b) = Linear(x).chunk(2, dim=-1).

    Notes
    -----
    * We follow the convention hidden_dim → expansion_dim*2 via a single Linear.
    * The module keeps output dimension identical to input.
    """

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expanded_dim = hidden_dim * expansion

        # One projection produces both a and b (gating) vectors
        self.w_in = nn.Linear(hidden_dim, self.expanded_dim * 2, bias=True)
        self.w_out = nn.Linear(self.expanded_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Initialize following Transformer defaults
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        if self.w_in.bias is not None:
            nn.init.zeros_(self.w_in.bias)
        if self.w_out.bias is not None:
            nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq, dim)
        a, b = self.w_in(x).chunk(2, dim=-1)
        y = a * F.silu(b)
        y = self.w_out(self.dropout(y))
        return y 