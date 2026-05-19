"""Auxiliary heads on top of GNOT operator features u(s).

- SwitchHead   : per-point switch probability + scalar (center, width).
- PINOHead     : predicts local curvature scalar kappa(z_ctx) for the PINO loss.
- SpectralHead : passes-through (loss is computed on the surrogate output).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwitchHead(nn.Module):
    def __init__(self, d_op: int, d_ctx: int):
        super().__init__()
        self.point_classifier = nn.Sequential(
            nn.Conv1d(d_op, d_op, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_op, 1, kernel_size=1),
        )
        self.scalars = nn.Sequential(
            nn.Linear(d_ctx + d_op, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # (center, width)
        )

    def forward(self, u: torch.Tensor, z_ctx: torch.Tensor) -> dict:
        # u: (B, d_op, T)  z_ctx: (B, d_ctx)
        prob_logits = self.point_classifier(u).squeeze(1)        # (B, T)
        u_pool = u.mean(dim=-1)                                   # (B, d_op)
        scalars = self.scalars(torch.cat([u_pool, z_ctx], dim=-1))
        center, width = scalars[..., 0], scalars[..., 1]
        return {"prob_logits": prob_logits, "center": center, "width": width}


class PINOHead(nn.Module):
    def __init__(self, d_ctx: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_ctx, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.net(z_ctx).squeeze(-1)   # (B,)


class SpectralHead(nn.Module):
    """Identity head — the spectral loss is computed directly on the predicted curve."""

    def forward(self, ta_pred: torch.Tensor) -> torch.Tensor:
        return ta_pred
