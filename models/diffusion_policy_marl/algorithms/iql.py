"""Implicit Q-Learning critic (Kostrikov et al. 2022).

V via expectile regression of Q :
    L_V = E [ L2_tau ( Q(s,a) - V(s) ) ],   L2_tau(u) = | tau - 1[u<0] | u^2
Q via TD :
    L_Q = E [ ( r + gamma V(s') - Q(s,a) ) ^ 2 ]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class IQLCritic(nn.Module):
    """V-network and joint Q-network (Q over flattened action vector)."""

    def __init__(self, state_dim: int, action_dims: List[int],
                 hidden: int = 256, expectile_tau: float = 0.7,
                 discount: float = 0.99, target_update_rate: float = 0.005):
        super().__init__()
        self.flat_action_dim = sum(action_dims)
        self.tau = expectile_tau
        self.gamma = discount
        self.target_update_rate = target_update_rate

        self.V = _MLP(state_dim, hidden, 1)
        self.Q = _MLP(state_dim + self.flat_action_dim, hidden, 1)
        self.Q_target = _MLP(state_dim + self.flat_action_dim, hidden, 1)
        self.Q_target.load_state_dict(self.Q.state_dict())
        for p in self.Q_target.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _flatten_actions(actions: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(actions, dim=-1)

    def value_loss(self, state: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            q = self.Q_target(torch.cat([state, self._flatten_actions(actions)], dim=-1))
        v = self.V(state)
        return expectile_loss(q - v, tau=self.tau)

    def q_loss(self, state: torch.Tensor, actions: List[torch.Tensor],
               reward: torch.Tensor, next_state: torch.Tensor,
               done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            v_next = self.V(next_state)
            target = reward + self.gamma * (1.0 - done) * v_next
        q = self.Q(torch.cat([state, self._flatten_actions(actions)], dim=-1))
        return torch.nn.functional.mse_loss(q, target)

    @torch.no_grad()
    def soft_update(self) -> None:
        for tp, p in zip(self.Q_target.parameters(), self.Q.parameters()):
            tp.data.lerp_(p.data, self.target_update_rate)

    def advantage(self, state: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        q = self.Q(torch.cat([state, self._flatten_actions(actions)], dim=-1))
        v = self.V(state)
        return q - v
