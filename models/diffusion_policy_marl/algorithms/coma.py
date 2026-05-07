"""COMA — Counterfactual Multi-Agent advantage.

A_i = Q(s, a) - sum_{a'_i} pi(a'_i | s, m) Q(s, a'_i, a_-i)
For continuous actions we Monte-Carlo this expectation.
"""

from __future__ import annotations

from typing import Callable, List

import torch


@torch.no_grad()
def compute_coma_advantage(
    state: torch.Tensor,
    actions: List[torch.Tensor],
    agent_idx: int,
    q_fn: Callable[[torch.Tensor, List[torch.Tensor]], torch.Tensor],
    sample_action_fn: Callable[[int, torch.Tensor], torch.Tensor],
    num_samples: int = 16,
) -> torch.Tensor:
    """Args:
        state:           (B, state_dim)
        actions:         list of N action tensors (current joint action).
        agent_idx:       index i to compute counterfactual for.
        q_fn:            (state, [a_1..a_N]) -> Q^joint  (B,)
        sample_action_fn: (agent_idx, state) -> sampled action (B, action_dim_i)
        num_samples:     Monte Carlo samples for the counterfactual baseline.
    Returns:
        A_i: (B,)
    """
    q_actual = q_fn(state, actions)
    baseline_acc = torch.zeros_like(q_actual)
    for _ in range(num_samples):
        a_alt = list(actions)
        a_alt[agent_idx] = sample_action_fn(agent_idx, state)
        baseline_acc = baseline_acc + q_fn(state, a_alt)
    baseline = baseline_acc / float(num_samples)
    return q_actual - baseline
