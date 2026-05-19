"""SSST_PRO — Switch-Segment Sparse Transformer (PyTorch, scaled up).

Architecture:
    - point-wise embedding for the local features (s, k, etc.) plus context features
    - trunk: stack of TrunkBlocks (Conv1D + MultiHeadAttention + FFN, residual)
    - mixture-of-experts (MoE) on top of trunk: K experts (depth-wise dilated
      convolutions), top-k gating
    - heads: scalar regression for Ta_norm + auxiliary switch / center / width

This is a faithful PyTorch re-implementation of the SSST architecture in code4
(see 05_StepD5_..._v5.py), with the parameters scaled up by ~2x as defined in
configs/sizes.yaml under `ssst_pro:`.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertConv1D(nn.Module):
    """One expert: depth-wise dilated Conv1D + GELU + 1x1 projection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.dconv = nn.Conv1d(channels, channels, kernel_size,
                                 dilation=dilation, padding=pad, groups=channels)
        self.pconv = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(F.gelu(self.pconv(self.dconv(x)))))


class MoEGating(nn.Module):
    """Top-k gating router. Inputs: pooled trunk features. Outputs: weights (B, K)."""

    def __init__(self, d_model: int, num_experts: int, top_k: int,
                 hidden: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, d_model)  -> logits (B, num_experts) -> top-k softmax
        logits = self.net(x)
        topk_v, topk_i = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_v, dim=-1)                      # (B, K)
        return weights, topk_i


class TrunkBlock(nn.Module):
    """One trunk block: Conv1D + MHA + FFN + residual."""

    def __init__(self, d_model: int, num_heads: int, kernel: int = 5,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2,
                               groups=d_model // 8 if d_model % 8 == 0 else 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # Conv branch
        h = x.transpose(1, 2)           # (B, d, T)
        h = self.conv(h).transpose(1, 2)
        x = self.norm1(x + h)
        # Attention branch
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm2(x + a)
        # FFN branch
        x = self.norm3(x + self.ff(x))
        return x


class SSSTProSurrogate(nn.Module):
    """SSST_PRO main model.

    Inputs:
        ctx: (B, ctx_dim)          static context (log10E + 23 branch features)
        s:   (B, T)                normalized abscissa
    Outputs:
        ta_pred:    (B, T)         predicted Ta_norm
        switch_logits: (B, T)
        switch_center: (B,)
        switch_width:  (B,)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        kernel = cfg.get("trunk_kernel", 5)
        ff_mult = cfg.get("ff_mult", 4)
        dropout = cfg.get("dropout", 0.10)
        num_blocks = cfg["num_blocks"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k_experts"]
        kernel_sizes = list(cfg["expert_kernel_sizes"])[:num_experts]
        dilations = list(cfg["expert_dilations"])[:num_experts]

        ctx_dim = cfg.get("ctx_dim", 24)

        # 1) Embedding of (s, ctx) at every position
        self.s_embed = nn.Sequential(nn.Linear(1, d_model // 2), nn.GELU(),
                                      nn.Linear(d_model // 2, d_model))
        self.ctx_proj = nn.Linear(ctx_dim, d_model)

        # 2) Trunk
        self.trunk = nn.ModuleList(
            [TrunkBlock(d_model, num_heads, kernel, ff_mult, dropout)
             for _ in range(num_blocks)]
        )

        # 3) Experts (depth-wise dilated convolutions)
        self.experts = nn.ModuleList(
            [ExpertConv1D(d_model, k, dl, dropout)
             for k, dl in zip(kernel_sizes, dilations)]
        )
        self.gate = MoEGating(d_model, num_experts, top_k,
                                hidden=cfg.get("gate_hidden", 256))

        # 4) Heads
        self.head_main = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_switch_pt = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model // 2, 1, 1),
        )
        self.head_switch_scalar = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 2),  # (center, width)
        )

    def forward(self, ctx: torch.Tensor, s: torch.Tensor) -> dict:
        B, T = s.shape
        # 1) embed s and ctx
        h = self.s_embed(s.unsqueeze(-1))            # (B, T, d_model)
        c = self.ctx_proj(ctx).unsqueeze(1)            # (B, 1, d_model)
        h = h + c

        # 2) trunk
        for blk in self.trunk:
            h = blk(h)                                 # (B, T, d_model)

        # 3) MoE
        h_BCT = h.transpose(1, 2)                      # (B, d_model, T)
        pooled = h.mean(dim=1)                          # (B, d_model)
        weights, topk_i = self.gate(pooled)
        # gather expert outputs only for chosen ones (loop over top_k experts)
        moe_out = torch.zeros_like(h_BCT)
        for k_pos in range(weights.size(-1)):
            idx_k = topk_i[:, k_pos]                   # (B,) expert id
            w_k = weights[:, k_pos].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            # apply each unique expert to the matching subset of the batch
            for e_id in idx_k.unique():
                mask = (idx_k == e_id)
                if mask.any():
                    inp = h_BCT[mask]
                    out = self.experts[int(e_id)](inp)
                    moe_out[mask] += w_k[mask] * out
        h_BCT = h_BCT + moe_out
        h = h_BCT.transpose(1, 2)                       # (B, T, d_model)

        # 4) heads
        ta_pred = self.head_main(h).squeeze(-1)         # (B, T)
        switch_logits = self.head_switch_pt(h_BCT).squeeze(1)
        sw_scalars = self.head_switch_scalar(pooled)
        center, width = sw_scalars[..., 0], sw_scalars[..., 1]
        return {"ta_pred": ta_pred,
                "switch_logits": switch_logits,
                "switch_center": center,
                "switch_width": width,
                "moe_weights": weights}

    def compute_loss(self, batch: dict) -> dict:
        out = self.forward(batch["ctx"], batch["s"])
        ta_true = batch["ta_true"]
        l_data = F.mse_loss(out["ta_pred"], ta_true)
        l_grad = F.mse_loss(out["ta_pred"][:, 1:] - out["ta_pred"][:, :-1],
                             ta_true[:, 1:] - ta_true[:, :-1])
        l_curv = F.mse_loss(
            out["ta_pred"][:, 2:] - 2 * out["ta_pred"][:, 1:-1] + out["ta_pred"][:, :-2],
            ta_true[:, 2:] - 2 * ta_true[:, 1:-1] + ta_true[:, :-2])
        l_switch = F.binary_cross_entropy_with_logits(
            out["switch_logits"],
            batch.get("switch_label",
                      torch.zeros_like(out["switch_logits"])))
        l_extrema = F.huber_loss(
            out["switch_center"],
            batch.get("switch_center", torch.zeros_like(out["switch_center"])))
        l_spec = (torch.fft.rfft(out["ta_pred"], dim=-1).abs()[..., 16:] ** 2).mean()
        w = self.cfg["losses"]
        total = (w["data"] * l_data + w["grad"] * l_grad + w["curv"] * l_curv
                 + w["switch"] * l_switch + w["extrema"] * l_extrema
                 + w["spectral"] * l_spec)
        return {"loss": total, "loss_data": l_data.detach(),
                "loss_grad": l_grad.detach(), "loss_curv": l_curv.detach(),
                "loss_switch": l_switch.detach(), "loss_spectral": l_spec.detach()}

    @torch.no_grad()
    def predict(self, ctx: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.forward(ctx, s)["ta_pred"]
