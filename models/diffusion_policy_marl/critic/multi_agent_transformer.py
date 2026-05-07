"""Multi-Agent Transformer (MAT) — centralized critic.

Q^joint(s, a_1, ..., a_N) via a transformer over (state_token, action_tokens).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiAgentTransformerCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.num_agents = len(action_dims)
        self.causal = causal
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_projs = nn.ModuleList([nn.Linear(d, d_model) for d in action_dims])
        self.role_embed = nn.Parameter(torch.randn(1 + self.num_agents, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, state: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        """Args:
            state:   (B, state_dim)
            actions: list of N tensors, each (B, action_dim_i)
        Returns:
            Q^joint: (B,)
        """
        B = state.size(0)
        s_tok = self.state_proj(state).unsqueeze(1)                                  # (B, 1, d)
        a_toks = [proj(a).unsqueeze(1) for proj, a in zip(self.action_projs, actions)]
        x = torch.cat([s_tok] + a_toks, dim=1)                                       # (B, 1+N, d)
        x = x + self.role_embed.unsqueeze(0)
        mask = self._causal_mask(x.size(1), x.device) if self.causal else None
        h = self.transformer(x, mask=mask)
        # Pool: mean across tokens (state included).
        return self.out(h.mean(dim=1)).squeeze(-1)
