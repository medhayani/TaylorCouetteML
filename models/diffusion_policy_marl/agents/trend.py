"""Trend agent — global slope / intercept / curvature corrections (NEW in PRO)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrendAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)
