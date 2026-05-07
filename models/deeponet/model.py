"""DeepONet — branch + trunk operator learning.

Lu, Jin, Pang, Zhang, Karniadakis, "Learning nonlinear operators via DeepONet",
Nature Mach. Intell. 3, 218 (2021).

Branch encodes the input parameters (E, branch features) into a latent vector
b in R^p; trunk encodes the spatial coordinate k into t in R^p; the prediction
is the dot product T_a(k) = <b, t> + b_0.

We add a Fourier coordinate embedding on the trunk side to capture
high-frequency content in the curves.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def fourier_features(k: torch.Tensor, K: int = 16, omega_max: float = 16.0) -> torch.Tensor:
    """gamma(k) = [k, sin(2pi w_i k), cos(2pi w_i k)]_{i=1..K}.

    Args:
        k: (B, T) values in any range.
        K: number of frequency bands.
        omega_max: top frequency (in radians/length-unit of normalised k).
    Returns: (B, T, 1 + 2K)
    """
    if k.dim() == 1:
        k = k.unsqueeze(0)
    B, T = k.shape
    log_min, log_max = 0.0, math.log(omega_max)
    omegas = torch.exp(torch.linspace(log_min, log_max, K, device=k.device))
    arg = 2.0 * math.pi * k.unsqueeze(-1) * omegas             # (B, T, K)
    feat = torch.cat([k.unsqueeze(-1), torch.sin(arg), torch.cos(arg)], dim=-1)
    return feat                                                  # (B, T, 1+2K)


class _MLP(nn.Module):
    def __init__(self, in_dim, layers, out_dim, act="gelu", dropout: float = 0.0):
        super().__init__()
        Act = nn.GELU if act == "gelu" else nn.ReLU
        mods = []
        d = in_dim
        for h in layers:
            mods += [nn.Linear(d, h), Act(),
                       nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            d = h
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, ctx_dim: int = 24, branch_layers=(256, 256, 256, 256),
                 trunk_layers=(256, 256, 256, 256), latent_dim: int = 128,
                 fourier_bands: int = 16, omega_max: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.fourier_bands = fourier_bands
        self.omega_max = omega_max
        # Branch network: ctx -> R^p
        self.branch = _MLP(ctx_dim, branch_layers, latent_dim, dropout=dropout)
        # Trunk network: gamma(k) -> R^p
        self.trunk = _MLP(1 + 2 * fourier_bands, trunk_layers, latent_dim, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Args:
            k_norm: (B, T)  in [-1, 1] (already normalised within branch)
            ctx:    (B, ctx_dim)
        Returns:
            ta:     (B, T)
        """
        b = self.branch(ctx)                                   # (B, p)
        t = self.trunk(fourier_features(k_norm,
                                          K=self.fourier_bands,
                                          omega_max=self.omega_max))
        # t: (B, T, p) ; b: (B, p) -> (B, T) via einsum
        ta = torch.einsum("bp,btp->bt", b, t) + self.bias
        return ta
