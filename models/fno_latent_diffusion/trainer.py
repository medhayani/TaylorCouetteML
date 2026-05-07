"""NEPTUNE_PRO surrogate.

Architecture identical in structure to code6's NEPTUNE but with much larger
hyperparameters (d_model 256/384, 5 ISAB layers, 8 FNO blocks, 1000 diffusion
timesteps, 32 fourier modes/64 features). Expected ~25M params per member.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .set_transformer import SetTransformerEncoder
from .gnot import GNOTOperator
from .diffusion import LatentDiffusionDecoder
from .auxiliary_heads import SwitchHead, PINOHead
from .losses import diffusion_loss, switch_loss, pino_loss, spectral_loss


class NeptuneProSurrogate(nn.Module):
    """One PRO ensemble member."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        c = cfg["context"]
        op = cfg["operator"]
        df = cfg["diffusion"]
        self.encoder = SetTransformerEncoder(
            in_dim=c["in_dim"], d_model=c["d_model"],
            num_layers=c["num_isab_layers"], num_heads=c["num_heads"],
            num_inducing=c["num_inducing_points"], dropout=c["dropout"],
        )
        self.operator = GNOTOperator(
            d_ctx=c["d_model"], d_model=op["d_model"],
            num_blocks=op["num_blocks"], num_heads=op["num_heads"],
            num_fourier_modes=op["num_fourier_modes"],
            fourier_features=op["fourier_features"], dropout=op["dropout"],
        )
        self.decoder = LatentDiffusionDecoder(
            d_op=op["d_model"], d_ctx=c["d_model"],
            num_timesteps=df["num_timesteps"],
            beta_schedule_s=df.get("beta_schedule_s", 0.008),
        )
        self.switch_head = SwitchHead(d_op=op["d_model"], d_ctx=c["d_model"])
        self.pino_head = PINOHead(d_ctx=c["d_model"])

    def forward(self, ctx: torch.Tensor, s: torch.Tensor) -> dict:
        z_ctx = self.encoder(ctx)
        u = self.operator(s, z_ctx)
        return {"z_ctx": z_ctx, "u": u}

    def compute_loss(self, batch: dict) -> dict:
        ctx, s = batch["ctx"], batch["s"]
        ta_true = batch["ta_true"]
        out = self.forward(ctx, s)
        z_ctx, u = out["z_ctx"], out["u"]
        l_diff = self.decoder.loss(ta_true, u, z_ctx)
        sw = self.switch_head(u, z_ctx)
        l_switch = switch_loss(
            prob_logits=sw["prob_logits"],
            switch_label=batch.get("switch_label", torch.zeros_like(sw["prob_logits"])),
            center_pred=sw["center"],
            center_true=batch.get("switch_center", torch.zeros_like(sw["center"])),
            width_pred=sw["width"],
            width_true=batch.get("switch_width", torch.zeros_like(sw["width"])),
        )
        kappa = self.pino_head(z_ctx)
        l_pino = pino_loss(ta_pred=ta_true, s=s, kappa_pred=kappa)
        l_spec = spectral_loss(ta_true,
                                k_max=self.cfg["operator"]["num_fourier_modes"])
        w = self.cfg["losses"]
        total = (w["diffusion"] * l_diff + w["switch"] * l_switch
                 + w["pino"] * l_pino + w["spectral"] * l_spec)
        return {"loss": total,
                "loss_diff": l_diff.detach(), "loss_switch": l_switch.detach(),
                "loss_pino": l_pino.detach(), "loss_spectral": l_spec.detach()}

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, s: torch.Tensor,
               num_steps: Optional[int] = None) -> torch.Tensor:
        out = self.forward(ctx, s)
        return self.decoder.sample(
            out["u"], out["z_ctx"],
            num_steps=num_steps or self.cfg["diffusion"]["inference"]["num_steps"],
        )
