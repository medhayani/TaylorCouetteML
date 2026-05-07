"""MARL_PRO — 3-agent SAC system with cross-agent attention message passing.

Replaces code4's "freeze previous agent" sequential paradigm with a
proper joint cooperative architecture:
    1. Each agent has its own SARL_PRO-style feature extractor.
    2. Cross-agent attention exchanges messages over K=3 rounds.
    3. Each agent has its own SAC policy + critic, but the inputs are
       enriched by the messages from the other two agents.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from sarl_pro.feature_extractor import SARLProFeatureExtractor
from sarl_pro.sac_pro import GaussianActor, TwinCritic


class CrossAgentAttention(nn.Module):
    """Multi-round cross-attention message passing between agents."""

    def __init__(self, dim: int, num_rounds: int = 3, num_heads: int = 8,
                 dropout: float = 0.05):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_rounds)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N_agents, dim)
        for attn, norm in zip(self.layers, self.norms):
            out, _ = attn(h, h, h, need_weights=False)
            h = norm(h + out)
        return h


class AgentSAC(nn.Module):
    """SAC actor-critic for one agent (no feature extractor — uses pre-extracted)."""

    def __init__(self, feature_dim: int, action_dim: int,
                 actor_layers: List[int], critic_layers: List[int],
                 init_alpha: float = 0.1):
        super().__init__()
        self.actor = GaussianActor(feature_dim, action_dim, actor_layers)
        self.critic = TwinCritic(feature_dim, action_dim, critic_layers)
        self.target_critic = TwinCritic(feature_dim, action_dim, critic_layers)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        import math
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.target_entropy = -float(action_dim)


class MARLProSystem(nn.Module):
    """3 agents (Localizer / Shape / Geometry) with cross-agent attention."""

    AGENT_NAMES = ("localizer", "shape", "geometry")
    AGENT_ACTION_DIMS = (2, 2, 4)

    def __init__(self, obs_seq_dim: int, obs_seq_T: int, static_dim: int, cfg: dict):
        super().__init__()
        af = cfg["agent_features"]
        self.extractors = nn.ModuleList([
            SARLProFeatureExtractor(obs_seq_dim, obs_seq_T, static_dim, af)
            for _ in range(3)
        ])
        feat_dim = self.extractors[0].out_dim

        self.cross_attn = CrossAgentAttention(
            dim=feat_dim,
            num_rounds=cfg["cross_agent_attention"]["num_rounds"],
            num_heads=cfg["cross_agent_attention"]["num_heads"],
        ) if cfg.get("cross_agent_attention", {}).get("enabled", True) else None

        self.agents = nn.ModuleList([
            AgentSAC(feat_dim, ad,
                     cfg["actor_layers"], cfg["critic_layers"])
            for ad in self.AGENT_ACTION_DIMS
        ])

    def encode(self, obs_seq: torch.Tensor, static_vec: torch.Tensor) -> torch.Tensor:
        """Returns post-attention features for each agent: (B, 3, feat_dim)."""
        feats = [ext(obs_seq, static_vec) for ext in self.extractors]
        h = torch.stack(feats, dim=1)        # (B, 3, feat_dim)
        if self.cross_attn is not None:
            h = self.cross_attn(h)
        return h

    def sample_actions(self, obs_seq: torch.Tensor,
                        static_vec: torch.Tensor) -> List[torch.Tensor]:
        h = self.encode(obs_seq, static_vec)
        actions = []
        for i, ag in enumerate(self.agents):
            a, _ = ag.actor(h[:, i, :])
            actions.append(a)
        return actions
