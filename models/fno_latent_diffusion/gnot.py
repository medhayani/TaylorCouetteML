"""GNOT — Geometry-Aware Neural Operator.

Pipeline:
    gamma(s) -> CrossAttn(gamma, z_ctx) -> v^(0)
    v^(0) -> [FNOBlock]^L -> u(s)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .spectral_blocks import FNOBlock


def fourier_features_1d(s: torch.Tensor, K: int = 32) -> torch.Tensor:
    """gamma(s) = [sin(2 pi k s), cos(2 pi k s)]_{k=1..K}.

    Args:
        s: (B, T) coordinates in [0, 1].
    Returns:
        (B, T, 2K)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
    k = torch.arange(1, K + 1, device=s.device, dtype=s.dtype)  # (K,)
    arg = 2.0 * math.pi * s.unsqueeze(-1) * k                    # (B, T, K)
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)   # (B, T, 2K)


class GNOTOperator(nn.Module):
    """Geometry-aware FNO conditioned on z_ctx via cross-attention."""

    def __init__(
        self,
        d_ctx: int,
        d_model: int = 192,
        num_blocks: int = 6,
        num_heads: int = 8,
        num_fourier_modes: int = 16,
        fourier_features: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fourier_features = fourier_features
        self.embed = nn.Linear(2 * fourier_features, d_model)
        self.ctx_proj = nn.Linear(d_ctx, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm0 = nn.LayerNorm(d_model)
        self.fno = nn.Sequential(
            *[FNOBlock(d_model, num_modes=num_fourier_modes, dropout=dropout)
              for _ in range(num_blocks)]
        )
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=d_model)

    def forward(self, s: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        """Args:
            s:     (B, T) coordinates in [0, 1].
            z_ctx: (B, d_ctx) context latent.
        Returns:
            u: (B, d_model, T) operator features.
        """
        gamma = fourier_features_1d(s, K=self.fourier_features)   # (B, T, 2K)
        v = self.embed(gamma)                                      # (B, T, d_model)

        ctx = self.ctx_proj(z_ctx).unsqueeze(1)                    # (B, 1, d_model)
        v_attn, _ = self.cross_attn(v, ctx, ctx, need_weights=False)
        v = self.norm0(v + v_attn)                                 # (B, T, d_model)

        v = v.transpose(1, 2)                                      # (B, d_model, T)
        v = self.fno(v)
        return self.out_norm(v)
