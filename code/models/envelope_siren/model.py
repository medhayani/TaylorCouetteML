"""Multi-Mode Envelope SIREN.

The marginal-stability problem  Ta(k, E)  arises from a Floquet eigenvalue
system. Each Floquet mode m gives a smooth threshold curve  Ta_m(k, E);
the observed neutral curve is the lower envelope

        Ta(k, E)  =  min_{m=1..M}  Ta_m(k, E),

which is piecewise-smooth with cusps at the parameter values where the
active mode switches (mode-crossings). A single smooth network cannot
represent this — it tries to interpolate through every cusp and either
rounds them off or oscillates near them (Gibbs-like ringing).

This architecture mirrors the physics:
    * a shared context encoder z = phi_ctx(xi);
    * M independent SIREN "mode" branches predicting Ta_m(k, z);
    * a hard min envelope across modes.

The hard min is piecewise-differentiable; gradients flow back through
the argmin index at each point, so each mode learns the curve only on
the (k, xi) regions where it is active.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Linear -> sin(omega * x), Sitzmann et al. NeurIPS 2020."""

    def __init__(self, in_f: int, out_f: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.omega_0 = omega_0
        self.is_first = is_first
        with torch.no_grad():
            bound = (1.0 / in_f) if is_first else (math.sqrt(6.0 / in_f) / omega_0)
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma(z) * h + beta(z)."""

    def __init__(self, ctx_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, feat_dim)
        self.beta = nn.Linear(ctx_dim, feat_dim)

    def forward(self, h, z_ctx):
        return self.gamma(z_ctx) * h + self.beta(z_ctx)


class _ContextEncoder(nn.Module):
    def __init__(self, ctx_dim: int, latent: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, latent),
        )

    def forward(self, x):
        return self.net(x)


class _ModeSIREN(nn.Module):
    """Single SIREN trunk operating on k_norm with FiLM conditioning."""

    def __init__(self, ctx_latent: int, hidden: int, depth: int,
                 omega_first: float = 30.0, omega_hidden: float = 30.0):
        super().__init__()
        self.first = SineLayer(1, hidden, is_first=True, omega_0=omega_first)
        self.films = nn.ModuleList([FiLM(ctx_latent, hidden) for _ in range(depth)])
        self.body = nn.ModuleList(
            [SineLayer(hidden, hidden, is_first=False, omega_0=omega_hidden)
             for _ in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / omega_hidden
            self.out.weight.uniform_(-bound, bound)

    def forward(self, k_norm: torch.Tensor, z_per_pt: torch.Tensor) -> torch.Tensor:
        x = k_norm.unsqueeze(-1)
        h = self.first(x)
        for film, sine in zip(self.films, self.body):
            h = film(h, z_per_pt)
            h = sine(h)
        return self.out(h).squeeze(-1)


class MultiModeEnvelopeSIREN(nn.Module):
    """Predicts M smooth modes; output is their lower envelope (min).

    Args:
        ctx_dim:   input context vector size.
        n_modes:   number M of competing modes (default 4 — covers most
                    Floquet branch crossings observed).
        hidden:    SIREN hidden width per mode.
        depth:     number of FiLM+SineLayer pairs per mode.
    """

    def __init__(self, ctx_dim: int = 24, n_modes: int = 4,
                 hidden: int = 192, depth: int = 5,
                 omega_first: float = 30.0, omega_hidden: float = 30.0):
        super().__init__()
        self.n_modes = n_modes
        self.ctx_enc = _ContextEncoder(ctx_dim, latent=128)
        self.modes = nn.ModuleList([
            _ModeSIREN(128, hidden, depth, omega_first, omega_hidden)
            for _ in range(n_modes)
        ])
        # Symmetry-breaking initial offsets to prevent collapse onto a single mode.
        self.mode_offset = nn.Parameter(torch.linspace(-0.1, 0.4, n_modes))

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor,
                return_modes: bool = False):
        """Args:
            k_norm: (B, T) in [-1, 1].
            ctx:    (B, ctx_dim).
            return_modes: if True, also return the per-mode curves and argmin.
        Returns:
            env:    (B, T) the lower envelope.
            (Y, idx): per-mode curves (B, M, T) and argmin (B, T) when requested.
        """
        B, T = k_norm.shape
        z = self.ctx_enc(ctx)                          # (B, latent)
        z_per_pt = z.unsqueeze(1).expand(B, T, -1)     # (B, T, latent)
        outs = []
        for m, smod in enumerate(self.modes):
            outs.append(smod(k_norm, z_per_pt) + self.mode_offset[m])
        Y = torch.stack(outs, dim=1)                   # (B, M, T)
        env, idx = Y.min(dim=1)                         # (B, T)
        if return_modes:
            return env, Y, idx
        return env
