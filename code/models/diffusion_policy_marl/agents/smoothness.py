"""Smoothness agent — outputs a 1D field of length T (anti-roughness corrections)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmoothnessAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 49, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)
