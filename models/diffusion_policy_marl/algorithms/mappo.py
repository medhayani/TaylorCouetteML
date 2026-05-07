"""MAPPO clipped surrogate loss.

L_i = -E [ min( rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i ) ]
"""

from __future__ import annotations

import torch


def mappo_clip_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float = 0.2,
    entropy: torch.Tensor = None,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    rho = torch.exp(log_prob_new - log_prob_old)
    a_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    s1 = rho * a_norm
    s2 = torch.clamp(rho, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * a_norm
    loss = -torch.min(s1, s2).mean()
    if entropy is not None:
        loss = loss - entropy_coef * entropy.mean()
    return loss
