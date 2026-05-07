"""Hierarchical manager (Feudal HRL). Sets a sub-goal vector every K steps."""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalManager(nn.Module):
    def __init__(self, state_dim: int, subgoal_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.subgoal_dim = subgoal_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, subgoal_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
