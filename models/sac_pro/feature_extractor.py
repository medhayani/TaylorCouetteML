"""Custom feature extractor for SARL_PRO.

Conv1D + multi-head attention pooling on the obs_seq, MLP on static_vec,
fusion trunk to produce a single (B, fusion_hidden) feature vector for both
the actor and critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SARLProFeatureExtractor(nn.Module):
    def __init__(self, obs_seq_dim: int, obs_seq_T: int, static_dim: int, cfg: dict):
        super().__init__()
        seq_h = cfg["seq_hidden"]
        seq_o = cfg["seq_out"]
        st_h = cfg["static_hidden"]
        fus_h = cfg["fusion_hidden"]
        n_heads = cfg.get("num_attn_heads", 8)
        dropout = cfg.get("dropout", 0.05)

        # 1) sequence encoder (Conv1D stack)
        self.seq_proj = nn.Conv1d(obs_seq_dim, seq_h, 1)
        self.seq_blocks = nn.Sequential(
            nn.Conv1d(seq_h, seq_h, 5, padding=2), nn.GELU(),
            nn.Conv1d(seq_h, seq_h, 5, padding=2), nn.GELU(),
            nn.Conv1d(seq_h, seq_h, 3, padding=1), nn.GELU(),
        )
        # 2) attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, seq_h) * 0.02)
        self.pool_attn = nn.MultiheadAttention(seq_h, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.seq_out_proj = nn.Linear(seq_h, seq_o)

        # 3) static encoder
        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, st_h), nn.LayerNorm(st_h), nn.GELU(),
            nn.Linear(st_h, st_h), nn.GELU(),
        )

        # 4) fusion
        self.fusion = nn.Sequential(
            nn.Linear(seq_o + st_h, fus_h), nn.LayerNorm(fus_h), nn.GELU(),
            nn.Linear(fus_h, fus_h), nn.GELU(),
        )
        self.out_dim = fus_h

    def forward(self, obs_seq: torch.Tensor, static_vec: torch.Tensor) -> torch.Tensor:
        # obs_seq: (B, T, F) -> conv expects (B, F, T)
        h = obs_seq.transpose(1, 2)
        h = self.seq_proj(h)
        h = self.seq_blocks(h).transpose(1, 2)             # (B, T, seq_h)
        B = h.size(0)
        Q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(Q, h, h, need_weights=False)
        seq_feat = self.seq_out_proj(pooled.squeeze(1))    # (B, seq_o)
        st_feat = self.static_enc(static_vec)              # (B, st_h)
        return self.fusion(torch.cat([seq_feat, st_feat], dim=-1))
