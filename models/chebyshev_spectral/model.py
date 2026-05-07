"""Chebyshev Spectral Regressor.

Decomposes each branch's curve T_a(k) on the Chebyshev polynomial basis of
the first kind on [-1, 1] (after normalising k_phys -> k_norm in [-1, 1]):

    T_a(k_norm) = sum_{n=0}^{N} c_n(xi) * T_n(k_norm)

A small Transformer encodes the context xi = (log10 E, branch features) and
outputs the N+1 coefficients c_n. Smoothness is guaranteed by the truncated
spectral series; convergence on smooth functions is geometric in N.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


def chebyshev_T(N: int, x: torch.Tensor) -> torch.Tensor:
    """Compute Chebyshev polynomials of the first kind T_0,...,T_N at x.

    Args:
        N: max degree (inclusive).
        x: (B, T) values in [-1, 1].
    Returns:
        T: (B, T, N+1) with T[..., n] = T_n(x).
    """
    B, T = x.shape
    out = torch.zeros(B, T, N + 1, device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if N >= 1:
        out[..., 1] = x
    for n in range(1, N):
        out[..., n + 1] = 2.0 * x * out[..., n] - out[..., n - 1]
    return out


class ChebyshevSpectralRegressor(nn.Module):
    def __init__(self, ctx_dim: int = 24, n_modes: int = 16,
                 d_model: int = 256, num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.05):
        super().__init__()
        self.n_modes = n_modes

        # Treat context as a set of ctx_dim scalars; embed each, then attend.
        self.elem_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_modes + 1),
        )

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Args:
            k_norm: (B, T) in [-1, 1]
            ctx:    (B, ctx_dim)
        Returns:
            ta:     (B, T) reconstructed from N+1 Chebyshev coefficients.
        """
        B = ctx.size(0)
        # Build a token sequence: [CLS] + each ctx scalar lifted to d_model
        tok = self.elem_proj(ctx.unsqueeze(-1))       # (B, ctx_dim, d_model)
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x = torch.cat([cls, tok], dim=1)              # (B, 1+ctx_dim, d_model)
        h = self.encoder(x)
        z = h[:, 0]                                    # CLS token, (B, d_model)
        c = self.head(z)                               # (B, N+1) coefficients

        Tn = chebyshev_T(self.n_modes, k_norm)          # (B, T, N+1)
        ta = torch.einsum("btn,bn->bt", Tn, c)
        return ta

    def coefficients(self, ctx: torch.Tensor) -> torch.Tensor:
        """Return the predicted Chebyshev coefficients (B, N+1) for diagnostic plots."""
        B = ctx.size(0)
        tok = self.elem_proj(ctx.unsqueeze(-1))
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tok], dim=1)
        h = self.encoder(x)
        return self.head(h[:, 0])
