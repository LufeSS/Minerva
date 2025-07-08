from __future__ import annotations

"""Minerva-X Transformer – KSV core with Rotary Positional Embeddings (RoPE).

This file re-implements a *subset* of ``ksv.model.KSVTransformer`` but swaps
learned absolute position embeddings for parameter-free RoPE.  Other building
blocks (Butterfly-Givens FFN, Nyström attention, ACT, future-token head) are
unchanged and are re-used from the existing ``ksv.layers`` package to avoid
code duplication.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from butterfly_givens import ButterflyGivens
from ksv.layers import NystromAttention, MiniCausalSelfAttention  # reuse
from ksv.model import RMSNorm, KSVHead, FutureTokenProjector  # type: ignore

# ---------------------------------------------------------------------------
# Rotary positional embedding helper                                          
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Implements RoPE as in GPT-NeoX (S. Liu 2021).

    Produces per-position sine / cosine tables and applies the rotation to
    query/key tensors.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # cache to avoid recomputing large sin/cos
        self._cache: Tuple[int, torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    def _get_sin_cos(self, seq_len: int, device: torch.device):
        if self._cache is not None and self._cache[0] >= seq_len and self._cache[1].device == device:
            cached_len, sin_cached, cos_cached = self._cache
            return sin_cached[:seq_len], cos_cached[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))  # (T, dim/2)
        sin, cos = freqs.sin(), freqs.cos()  # each (T, dim/2)
        # interleave to match even/odd structure
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # (T, dim)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        self._cache = (seq_len, sin, cos)
        return sin, cos

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half(x: torch.Tensor):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    # ------------------------------------------------------------------
    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, seq_dim: int = -2):
        """Rotate ``q`` and ``k`` in-place along *sequence* dimension.

        Args:
            q, k: tensors with shape ``(..., seq_len, dim)`` where ``dim`` is
               divisible by 2.
            seq_dim: which axis is the *sequence length*.
        Returns: rotated ``q``, ``k`` (same shapes, same dtype).
        """
        # Move seq_len to second to last dim for broadcasting if needed
        if seq_dim != -2:
            raise NotImplementedError("seq_dim other than -2 not yet supported")
        seq_len = q.shape[-2]
        sin, cos = self._get_sin_cos(seq_len, q.device)
        # reshape for broadcasting across batch/heads: (1,…,T,1)
        while sin.ndim < q.ndim:
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        q2 = (q * cos) + (self._rotate_half(q) * sin)
        k2 = (k * cos) + (self._rotate_half(k) * sin)
        return q2, k2

# ---------------------------------------------------------------------------
# Transformer block identical except we inject Rotary in attention ----------
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, num_landmarks: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.attn = NystromAttention(embed_dim, num_heads, num_landmarks, dropout)
        self.rotary = RotaryEmbedding(embed_dim // num_heads)
        self.ln2 = RMSNorm(embed_dim)
        self.ffn_lin = ButterflyGivens(embed_dim, bias=False, init="random")
        self.ffn_drop = nn.Dropout(dropout)
        self.gate_proj = ButterflyGivens(embed_dim, bias=True, init="identity")
        self.alpha = nn.Parameter(torch.ones(embed_dim))

    # ------------------------------------------------------------------
    def _gate(self, x):
        z = self.gate_proj(x)
        base = (torch.atan(z) + math.pi / 2) / math.pi
        return base * (1 + 2 * self.alpha) - self.alpha

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # Compute q, k, v inside attention, but we need to inject RoPE.
        # Easier: temporarily access internal linear projections.
        # Copy of attention forward with extra rotation.
        B, T, _ = x.shape
        q = self.attn.q(self.ln1(x))
        k = self.attn.k(self.ln1(x))
        v = self.attn.v(self.ln1(x))
        # Reshape to B,H,T,Hd for rotation
        H = self.attn.heads
        Hd = self.attn.hd
        q = q.view(B, T, H, Hd).transpose(1, 2)  # B,H,T,Hd
        k = k.view(B, T, H, Hd).transpose(1, 2)
        q, k = self.rotary.apply_rotary(q, k, seq_dim=-2)  # RoPE applied
        # Merge back and continue Nyström path
        q = q.transpose(1, 2).contiguous().view(B, T, H * Hd)
        k = k.transpose(1, 2).contiguous().view(B, T, H * Hd)
        # Replace the existing linear outputs inside attn for this call
        out = self.attn.forward_with_qkv(q, k, v, mask)  # we'll add method
        h = out
        g = self._gate(x)
        x = h * g + x * (1 - g)
        h2 = self.ffn_drop(self.ffn_lin(self.ln2(x)))
        return x + h2

# ---------------------------------------------------------------------------
# Need to extend NystromAttention with helper that accepts pre-computed q,k,v.
# We'll monkey-patch a small method onto the original class so that we don’t
# duplicate its logic.
# ---------------------------------------------------------------------------

def _forward_with_qkv(self: NystromAttention, q_lin, k_lin, v_lin, mask=None):
    """Same as NystromAttention.forward but skips initial projections."""
    B, T, _ = q_lin.shape
    q = self._reshape(q_lin, B, T)
    k = self._reshape(k_lin, B, T)
    v = self._reshape(v_lin, B, T)
    m = min(self.lm, T)
    ql, kl = q[:, :, :m, :], k[:, :, :m, :]
    scale = 1.0 / math.sqrt(self.hd)
    k1 = F.softmax(torch.einsum("bhid,bhjd->bhij", q, kl) * scale, -1)
    k2 = F.softmax(torch.einsum("bhid,bhjd->bhij", ql, kl) * scale, -1)
    k3 = F.softmax(torch.einsum("bhid,bhjd->bhij", ql, k) * scale, -1)
    k2_inv = torch.linalg.pinv(k2.float()).to(k1.dtype)
    attn = k1 @ k2_inv @ k3
    if mask is not None:
        attn = attn.masked_fill(~mask[:, None], -1e9)
        attn = F.softmax(attn, -1)
    attn = self.drop(attn)
    o = torch.einsum("bhij,bhjd->bhid", attn, v)
    return self.o(self._merge(o, B, T))

# attach only once
if not hasattr(NystromAttention, "forward_with_qkv"):
    setattr(NystromAttention, "forward_with_qkv", _forward_with_qkv)

# ---------------------------------------------------------------------------
# Full Transformer (name: MinervaXTransformer) ------------------------------
# ---------------------------------------------------------------------------
class MinervaXTransformer(nn.Module):
    """KSV architecture + RoPE; ready for latent autoregressive generation."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        num_landmarks: int = 32,
        block_depth: int = 1,
        max_ponder_steps: int = 3,
        dropout: float = 0.1,
        act_epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_ponder_steps = max_ponder_steps
        self.act_epsilon = act_epsilon

        # Embeddings ---------------------------------------------------
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.time_emb = nn.Embedding(max_ponder_steps, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack of blocks ---------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, num_landmarks, dropout)
                for _ in range(block_depth)
            ]
        )

        # ACT ----------------------------------------------------------
        self.gate_pred = nn.Linear(embed_dim, 1)
        self.act_slope_raw = nn.Parameter(torch.tensor(0.0))
        self.token_slope_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GLU(dim=-1),
            nn.Linear(embed_dim, 1),
        )

        # Heads --------------------------------------------------------
        self.output_head = KSVHead(embed_dim, embed_dim, vocab_size)
        self.num_future_tokens = 3
        self.future_proj = nn.ModuleList([
            FutureTokenProjector(embed_dim, dropout) for _ in range(self.num_future_tokens)
        ])
        self.future_attn = MiniCausalSelfAttention(embed_dim, 1, dropout)
        self.future_head = KSVHead(embed_dim, embed_dim, vocab_size)

        # mask cache ---------------------------------------------------
        self._mask_cache: dict[Tuple[int, torch.device], torch.Tensor] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(logits: torch.Tensor):
        log_p = F.log_softmax(logits, -1)
        p = log_p.exp()
        return -(p * log_p).sum(-1)

    # ------------------------------------------------------------------
    def _causal_mask(self, T: int, device: torch.device):
        key = (T, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return self._mask_cache[key]

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, *, return_future: bool = False):
        B, T = ids.shape
        device = ids.device

        x = self.tok_emb(ids)
        x = self.dropout(x)

        mask = self._causal_mask(T, device)[None]

        # ---- ACT initial state -------------------------------------
        logits_prev = self.output_head(x)
        H_prev = self._entropy(logits_prev)
        R = torch.ones_like(H_prev)

        c_w = torch.zeros_like(x)
        c_g = torch.zeros(B, T, 1, device=device)
        max_g = torch.full_like(c_g, -1e9)

        # ---- Ponder steps ------------------------------------------
        for s in range(self.max_ponder_steps):
            h_in = x + self.time_emb(torch.tensor(s, device=device))
            h = h_in
            for blk in self.blocks:
                h = blk(h, mask)

            g_t = self.gate_pred(h)
            max_new = torch.maximum(max_g, g_t)
            scale = torch.exp(max_g - max_new)
            c_w = c_w * scale + h * torch.exp(g_t - max_new)
            c_g = c_g * scale + torch.exp(g_t - max_new)
            max_g = max_new
            x = c_w / (c_g + 1e-9)

            logits_t = self.output_head(x)
            H_t = self._entropy(logits_t)
            a_raw = H_t / (H_prev + self.act_epsilon)

            gamma = 1.0 + F.softplus(self.act_slope_raw)
            phi = 1.0 + 9.0 * torch.sigmoid(self.token_slope_net(h)).squeeze(-1)
            a_t = torch.sigmoid(gamma * phi * a_raw)
            if s == self.max_ponder_steps - 1:
                a_t = torch.ones_like(a_t)

            R = R * (1 - a_t)
            H_prev = H_t

        logits_final = logits_t

        if not return_future:
            return logits_final

        # Future-token branch --------------------------------------
        latents = [proj(x) for proj in self.future_proj]  # list B,T,D
        fut = torch.stack(latents, 2)  # B,T,K,D
        fut = fut + self.future_attn(fut)
        fut_logits = self.future_head(fut.view(B, T * self.num_future_tokens, self.embed_dim))
        fut_logits = fut_logits.view(B, T, self.num_future_tokens, -1)
        return logits_final, fut_logits 