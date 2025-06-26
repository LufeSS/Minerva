import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """Causal (autoregressive) self-attention with optional dropout.

    Uses PyTorch 2.* ``scaled_dot_product_attention`` which dispatches to FlashAttention-2
    or efficient implementations when available on the current GPU.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape ``(batch, seq_len, hidden_dim)``
        Returns:
            Tensor of the same shape.
        """
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each is (B, T, H, Hd)

        # Move head dim into batch for efficient attention op (B, H, T, Hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p, is_causal=True
        )  # (B, H, T, Hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        return self.o_proj(attn_output) 