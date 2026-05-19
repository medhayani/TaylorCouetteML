"""SAC_PRO — Soft Actor-Critic with our custom feature extractor.

Pure-PyTorch implementation (no Stable-Baselines3 dependency).
Continuous action, double Q critic, target networks, automatic entropy tuning.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, layers: List[int], out_dim: int) -> nn.Sequential:
    mods = []
    d = in_dim
    for h in layers:
        mods += [nn.Linear(d, h), nn.GELU()]
        d = h
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class GaussianActor(nn.Module):
    """Squashed Gaussian actor (tanh)."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, feature_dim: int, action_dim: int, hidden_layers: List[int]):
        super().__init__()
        self.body = _mlp(feature_dim, hidden_layers[:-1], hidden_layers[-1])
        self.mean = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std = nn.Linear(hidden_layers[-1], action_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(feat)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        z = mu + std * eps
        a = torch.tanh(z)
        log_p = (-0.5 * ((z - mu) / std).pow(2) - log_std
                 - 0.5 * float(torch.log(torch.tensor(2 * torch.pi))))
        log_p = log_p.sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, log_p


class TwinCritic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_layers: List[int]):
        super().__init__()
        self.q1 = _mlp(feature_dim + action_dim, hidden_layers, 1)
        self.q2 = _mlp(feature_dim + action_dim, hidden_layers, 1)

    def forward(self, feat: torch.Tensor, action: torch.Tensor):
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACPro(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int,
                 actor_layers: List[int], critic_layers: List[int],
                 gamma: float = 0.99, tau: float = 0.005,
                 init_alpha: float = 0.1):
        super().__init__()
        self.actor = GaussianActor(feature_dim, action_dim, actor_layers)
        self.critic = TwinCritic(feature_dim, action_dim, critic_layers)
        self.target_critic = TwinCritic(feature_dim, action_dim, critic_layers)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_alpha)))))
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def critic_loss(self, feat, action, reward, next_feat, done):
        with torch.no_grad():
            next_a, next_logp = self.actor(next_feat)
            tq1, tq2 = self.target_critic(next_feat, next_a)
            target = reward + (1.0 - done) * self.gamma * (
                torch.min(tq1, tq2) - self.alpha.detach() * next_logp)
        q1, q2 = self.critic(feat, action)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def actor_and_alpha_loss(self, feat):
        a, logp = self.actor(feat)
        q1, q2 = self.critic(feat, a)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * logp - q).mean()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        return actor_loss, alpha_loss

    @torch.no_grad()
    def soft_update(self):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.lerp_(p.data, self.tau)
