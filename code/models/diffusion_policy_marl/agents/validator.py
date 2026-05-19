"""Validator agent — outputs gates in [0,1] per other agent (veto / weighting)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValidatorAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return torch.sigmoid(self.head(h))
