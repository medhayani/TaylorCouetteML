"""Conditional latent diffusion decoder for Ta(s).

Forward (cosine schedule):
    q(T_a^t | T_a^0) = N(sqrt(abar_t) T_a^0, (1 - abar_t) I)

Reverse (parameterized by epsilon network):
    eps_theta(T_a^t, t, u, z_ctx)

Loss: simple MSE on epsilon.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    f = torch.cos(((x / num_timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    abar = f / f[0]
    betas = 1.0 - (abar[1:] / abar[:-1])
    return betas.clamp(min=1e-5, max=0.999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class EpsNet(nn.Module):
    """Small UNet1D-style epsilon predictor conditioned on (t, u, z_ctx)."""

    def __init__(self, d_op: int, d_ctx: int, d_t: int = 128, hidden: int = 128):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_t),
            nn.Linear(d_t, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj = nn.Linear(d_ctx, hidden)
        self.in_conv = nn.Conv1d(1 + d_op, hidden, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                u: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        # x_t: (B, T)  | u: (B, d_op, T)  | z_ctx: (B, d_ctx)  | t: (B,)
        h = torch.cat([x_t.unsqueeze(1), u], dim=1)         # (B, 1+d_op, T)
        h = self.in_conv(h)                                  # (B, hidden, T)
        cond = (self.t_embed(t) + self.ctx_proj(z_ctx)).unsqueeze(-1)
        h = h + cond
        h = self.mid(h)
        return self.out_conv(h).squeeze(1)                   # (B, T)


class LatentDiffusionDecoder(nn.Module):
    def __init__(
        self,
        d_op: int,
        d_ctx: int,
        num_timesteps: int = 1000,
        beta_schedule_s: float = 0.008,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = cosine_beta_schedule(num_timesteps, s=beta_schedule_s)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", abar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(abar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - abar))
        self.eps_net = EpsNet(d_op=d_op, d_ctx=d_ctx)

    # ------- forward (corruption) -------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_bar[t].unsqueeze(-1)
        soa = self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1)
        return sa * x0 + soa * noise

    # ------- training loss -------
    def loss(self, x0: torch.Tensor, u: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        pred = self.eps_net(x_t, t, u, z_ctx)
        return F.mse_loss(pred, noise)

    # ------- DDIM sampler (eta=0) -------
    @torch.no_grad()
    def sample(self, u: torch.Tensor, z_ctx: torch.Tensor,
               num_steps: int = 50,
               x0_clip: tuple = (-0.5, 1.5)) -> torch.Tensor:
        """DDIM eta=0 sampler with x0 clipping to stabilise high-noise steps.

        x0_clip clamps the implied clean signal each step to keep the
        trajectory bounded. Default range covers Ta_norm in [0,1] with margin.
        """
        B, _, T = u.shape
        x = torch.randn(B, T, device=u.device)
        step_ids = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1).long()
        for i in range(num_steps):
            t_now = step_ids[i].to(u.device).expand(B)
            t_next = step_ids[i + 1].to(u.device).expand(B)
            eps = self.eps_net(x, t_now, u, z_ctx)

            a_now = self.alphas_bar[t_now].unsqueeze(-1)
            a_next = self.alphas_bar[t_next].unsqueeze(-1)
            x0_pred = (x - torch.sqrt(1 - a_now) * eps) / torch.sqrt(a_now).clamp(min=1e-6)
            if x0_clip is not None:
                x0_pred = x0_pred.clamp(min=float(x0_clip[0]), max=float(x0_clip[1]))
            # recompute eps consistent with the clipped x0
            eps = (x - torch.sqrt(a_now) * x0_pred) / torch.sqrt(1 - a_now).clamp(min=1e-6)
            x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps
        return x
