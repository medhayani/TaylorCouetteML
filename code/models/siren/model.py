"""SIREN — Sinusoidal Implicit Neural Representation.

Sitzmann et al. "Implicit Neural Representations with Periodic Activation
Functions", NeurIPS 2020.

We map (k_norm in [-1, 1], context xi) -> Ta_norm via an MLP with sinusoidal
activations. FiLM conditioning by xi modulates the activations per branch
without breaking the implicit-representation property.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Linear -> sin(omega * x). Initialisation per Sitzmann et al."""

    def __init__(self, in_f: int, out_f: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.omega_0 = omega_0
        self.is_first = is_first
        self._init()

    def _init(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scales+shifts the coordinate features
    by a learned (gamma, beta) computed from the branch context vector."""

    def __init__(self, ctx_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, feat_dim)
        self.beta = nn.Linear(ctx_dim, feat_dim)

    def forward(self, h, z_ctx):
        return self.gamma(z_ctx) * h + self.beta(z_ctx)


class SIRENRegressor(nn.Module):
    def __init__(self, ctx_dim: int = 24, hidden: int = 256, depth: int = 6,
                 omega_0_first: float = 30.0, omega_0_hidden: float = 30.0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.hidden = hidden

        # Context encoder (small MLP -> latent z)
        self.ctx_enc = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128),
        )
        self.ctx_dim_out = 128

        # SIREN trunk on (k_norm,) — input dim 1
        self.first = SineLayer(1, hidden, is_first=True, omega_0=omega_0_first)
        self.films = nn.ModuleList([FiLM(self.ctx_dim_out, hidden) for _ in range(depth)])
        self.body = nn.ModuleList(
            [SineLayer(hidden, hidden, is_first=False, omega_0=omega_0_hidden)
             for _ in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / omega_0_hidden
            self.out.weight.uniform_(-bound, bound)

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Args:
            k_norm: (B, T)  in [-1, 1]
            ctx:    (B, ctx_dim)
        Returns:
            ta:     (B, T)
        """
        B, T = k_norm.shape
        z = self.ctx_enc(ctx)                       # (B, 128)
        z_per_pt = z.unsqueeze(1).expand(B, T, -1)  # (B, T, 128)

        x = k_norm.unsqueeze(-1)                    # (B, T, 1)
        h = self.first(x)
        for film, sine in zip(self.films, self.body):
            h = film(h, z_per_pt)
            h = sine(h)
        return self.out(h).squeeze(-1)              # (B, T)
