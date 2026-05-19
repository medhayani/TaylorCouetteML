"""Multi-Agent Decision Transformer (offline pre-training).

Tokens:
    (R_1, s_1, a_1^1..a_1^N,  R_2, s_2, a_2^1..a_2^N, ...)
Loss: autoregressive log-likelihood of joint actions.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiAgentDecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        context_length: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dims = action_dims
        self.num_agents = len(action_dims)
        self.context_length = context_length

        self.state_proj = nn.Linear(state_dim, d_model)
        self.return_proj = nn.Linear(1, d_model)
        self.action_in = nn.ModuleList([nn.Linear(d, d_model) for d in action_dims])
        self.action_out = nn.ModuleList([nn.Linear(d_model, d) for d in action_dims])

        self.pos_embed = nn.Parameter(
            torch.randn(1, context_length * (2 + self.num_agents), d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, returns: torch.Tensor, states: torch.Tensor,
                actions: List[torch.Tensor]) -> List[torch.Tensor]:
        """Args:
            returns: (B, L, 1)        return-to-go per timestep
            states:  (B, L, state_dim)
            actions: list of N tensors (B, L, action_dim_i)
        Returns:
            list of N predicted action logits / continuous outputs (B, L, action_dim_i)
        """
        B, L, _ = states.shape
        toks = []
        toks.append(self.return_proj(returns))                    # (B, L, d)
        toks.append(self.state_proj(states))                      # (B, L, d)
        for i, a in enumerate(actions):
            toks.append(self.action_in[i](a))                      # (B, L, d)
        # Interleave : R_t, s_t, a^1_t, ..., a^N_t  per timestep.
        T_per_step = 2 + self.num_agents
        x = torch.stack(toks, dim=2).reshape(B, L * T_per_step, -1)   # (B, L*T_per_step, d)
        x = x + self.pos_embed[:, : x.size(1), :]
        mask = self._causal_mask(x.size(1), x.device)
        h = self.transformer(x, mask=mask)                            # (B, L*Tps, d)

        h = h.reshape(B, L, T_per_step, -1)
        # The token preceding each a^i_t is its conditioning context (state + previous actions).
        # We read the action prediction from the corresponding action position (next-token-style).
        preds = []
        for i in range(self.num_agents):
            # predict a^i_t from h at index (1 + i) per step
            preds.append(self.action_out[i](h[:, :, 1 + i, :]))
        return preds
