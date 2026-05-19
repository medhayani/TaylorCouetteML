"""Set Transformer (Lee et al. 2019) — ISAB context encoder.

ISAB(X) = MAB(X, MAB(I_m, X))   with learnable inducing points I_m.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multi-head Attention Block: Y = LayerNorm(X + MultiHead(X, Y_kv, Y_kv))."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        H, _ = self.attn(X, Y, Y, need_weights=False)
        H = self.norm1(X + H)
        return self.norm2(H + self.ff(H))


class ISAB(nn.Module):
    """Induced Set Attention Block."""

    def __init__(self, d_model: int, num_heads: int = 8, num_inducing: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, d_model) / math.sqrt(d_model))
        self.mab1 = MAB(d_model, num_heads, dropout)
        self.mab2 = MAB(d_model, num_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, d_model)
        B = X.size(0)
        I = self.inducing.expand(B, -1, -1)
        H = self.mab1(I, X)        # (B, m, d)
        return self.mab2(X, H)     # (B, N, d)


class SetTransformerEncoder(nn.Module):
    """Encode context vector into a permutation-invariant latent z_ctx.

    Input  : (B, in_dim) — flat context (log10E + 23 branch features).
    Output : (B, d_model) — pooled latent.
    """

    def __init__(self, in_dim: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 8, num_inducing: int = 16, dropout: float = 0.1):
        super().__init__()
        # Each input scalar is treated as a set element of dim 1 lifted to d_model.
        self.elem_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [ISAB(d_model, num_heads, num_inducing, dropout) for _ in range(num_layers)]
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.pool_attn = MAB(d_model, num_heads, dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (B, in_dim) -> set of in_dim scalars, each lifted to d_model.
        if ctx.dim() == 2:
            X = self.elem_proj(ctx.unsqueeze(-1))      # (B, in_dim, d_model)
        else:
            X = self.elem_proj(ctx)                     # (B, N, d_model)
        for blk in self.blocks:
            X = blk(X)
        B = X.size(0)
        Q = self.pool_query.expand(B, -1, -1)
        z = self.pool_attn(Q, X).squeeze(1)
        return self.out_norm(z)
