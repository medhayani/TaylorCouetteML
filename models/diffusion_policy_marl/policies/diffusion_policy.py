"""Per-agent conditional diffusion policy with advantage-weighted training.

L_i^diff = E [ exp(beta_w * A_i^COMA) * || eps - eps_theta(a^t, t, s, m) ||^2 ]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neptune_pro.diffusion import cosine_beta_schedule, SinusoidalTimeEmbedding


class _ActionEpsNet(nn.Module):
    """MLP epsilon predictor for a small action vector."""

    def __init__(self, action_dim: int, state_dim: int, msg_dim: int,
                 d_t: int = 64, hidden: int = 256):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_t),
            nn.Linear(d_t, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(action_dim + state_dim + msg_dim, hidden)
        self.body = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, a_t: torch.Tensor, t: torch.Tensor,
                state: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([a_t, state, msg], dim=-1))
        h = h + self.t_embed(t)
        return self.out(self.body(h))


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        msg_dim: int,
        num_timesteps: int = 100,
        beta_schedule: str = "linear",
        advantage_weight: float = 3.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.advantage_weight = advantage_weight

        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, num_timesteps)
        else:
            betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_bar", abar)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_one_minus_abar", torch.sqrt(1.0 - abar))

        self.eps_net = _ActionEpsNet(action_dim, state_dim, msg_dim)

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(a0)
        sa = self.sqrt_abar[t].unsqueeze(-1)
        soa = self.sqrt_one_minus_abar[t].unsqueeze(-1)
        return sa * a0 + soa * noise

    def loss(self, a0: torch.Tensor, state: torch.Tensor, msg: torch.Tensor,
             advantage: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = a0.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=a0.device)
        noise = torch.randn_like(a0)
        a_t = self.q_sample(a0, t, noise=noise)
        pred = self.eps_net(a_t, t, state, msg)
        per_sample = ((pred - noise) ** 2).mean(dim=-1)              # (B,)
        if advantage is not None:
            w = torch.exp(self.advantage_weight * advantage).clamp(max=100.0)
            return (w * per_sample).mean()
        return per_sample.mean()

    @torch.no_grad()
    def sample(self, state: torch.Tensor, msg: torch.Tensor,
               num_steps: int = 20) -> torch.Tensor:
        B = state.size(0)
        a = torch.randn(B, self.action_dim, device=state.device)
        step_ids = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1).long()
        for i in range(num_steps):
            t_now = step_ids[i].to(state.device).expand(B)
            t_next = step_ids[i + 1].to(state.device).expand(B)
            eps = self.eps_net(a, t_now, state, msg)
            a_now = self.alphas_bar[t_now].unsqueeze(-1)
            a_next = self.alphas_bar[t_next].unsqueeze(-1)
            a0 = (a - torch.sqrt(1 - a_now) * eps) / torch.sqrt(a_now).clamp(min=1e-6)
            a = torch.sqrt(a_next) * a0 + torch.sqrt(1 - a_next) * eps
        return a
