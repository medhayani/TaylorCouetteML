"""Graph Attention Layer (Velickovic et al. 2018) and K-round communication.

h_i^(k+1) = sigma( W_s h_i^(k) + sum_j alpha_ij^(k) W_n h_j^(k) ).
alpha_ij = softmax_j ( LeakyReLU( a^T [W_s h_i || W_n h_j] ) )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.W_s = nn.Linear(in_dim, out_dim, bias=False)
        self.W_n = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(num_heads, 2 * out_dim // num_heads))
        self.dropout = dropout

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Args:
            h: (B, N, in_dim)  — N agents, B batch.
        Returns:
            (B, N, out_dim)
        """
        B, N, _ = h.shape
        H = self.num_heads
        Hs = self.W_s(h).view(B, N, H, -1)             # (B, N, H, dh)
        Hn = self.W_n(h).view(B, N, H, -1)             # (B, N, H, dh)

        # pairwise concat for attention scoring
        Hi = Hs.unsqueeze(2).expand(-1, -1, N, -1, -1)   # (B, N, N, H, dh)
        Hj = Hn.unsqueeze(1).expand(-1, N, -1, -1, -1)   # (B, N, N, H, dh)
        cat = torch.cat([Hi, Hj], dim=-1)                # (B, N, N, H, 2dh)
        e = F.leaky_relu((cat * self.attn).sum(dim=-1), negative_slope=0.2)  # (B, N, N, H)
        alpha = F.softmax(e, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # weighted sum over neighbors j
        out = torch.einsum("bijh,bjhd->bihd", alpha, Hn)  # (B, N, H, dh)
        return F.elu(out.reshape(B, N, -1))


class GATCommunication(nn.Module):
    """K rounds of stacked GAT layers + L1 regularizer hook on attention sparsity."""

    def __init__(self, dim: int, num_rounds: int = 3, num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(dim, dim, num_heads=num_heads) for _ in range(num_rounds)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = h + layer(h)
        return h
