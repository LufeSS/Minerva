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

        # Pre-compute and cache positional encodings up to ``max_seq_len`` on CPU.
        self.register_buffer(
            "pos_emb_cache",
            self._positional_encoding(max_seq_len, device=torch.device("cpu")),
            persistent=False,
        )

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
        """Compute sinusoidal positional embeddings on the given *device*."""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)

    def _get_positional_slice(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return positional embeddings of length *seq_len* on *device*.

        Extends the cached tensor if a longer sequence is requested during
        fine-tuning/inference.
        """
        if seq_len > self.pos_emb_cache.size(0):
            # Extend cache lazily on CPU then reuse.
            extra = self._positional_encoding(seq_len - self.pos_emb_cache.size(0), device=torch.device("cpu"))
            self.pos_emb_cache = torch.cat([self.pos_emb_cache, extra], dim=0)
        return self.pos_emb_cache[:seq_len].to(device)

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

        pos_emb = self._get_positional_slice(seq_len, device)
        x = self.token_emb(input_ids) + pos_emb.unsqueeze(0)
        x = x.to(self.token_emb.weight.dtype)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits 