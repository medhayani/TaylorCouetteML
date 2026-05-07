"""Geometry agent — left/right slopes, width, micro-shifts."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)
