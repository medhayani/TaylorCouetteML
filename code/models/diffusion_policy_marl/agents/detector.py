"""Detector agent — locates the switch (center shift, width refine, confidence)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DetectorAgent(nn.Module):
    """Encoder + 2 heads: action mean / log_std for SAC-like sampling.

    For HYDRA-MARL the actual policy is a DiffusionPolicy wrapping this encoder.
    Action: (delta_center, delta_width, alpha_confidence).
    """

    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 256):
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
