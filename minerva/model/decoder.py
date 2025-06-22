import math
from typing import Optional

import torch
import torch.nn as nn

from ..modules import ReZeroResidual  # for projection sharing later
from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """Language-model decoder with *green* components.

    This is a GPT-style decoder (no encoder cross-attention) with causal masking.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 4,
        hidden_dim: int = 512,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # Sinusoidal positional embeddings (can be swapped with Rotary later)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Output projection shares weights with token embedding
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def _positional_encoding(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        pos_emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return pos_emb

    def forward(
        self, input_ids: torch.Tensor, *, past_kv: Optional[None] = None
    ) -> torch.Tensor:
        """Args:
            input_ids: ``(batch, seq_len)`` token indices.
            past_kv: reserved for KV-cache interface (not yet used).
        Returns:
            logits: ``(batch, seq_len, vocab_size)``
        """
        device = input_ids.device
        batch, seq_len = input_ids.shape

        x = self.token_emb(input_ids) + self._positional_encoding(seq_len, device)[None, :, :]
        x = x.to(self.token_emb.weight.dtype)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits 