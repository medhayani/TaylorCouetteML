"""NEPTUNE_PRO — bigger NEPTUNE (~25M params/member, 5× larger than code6)."""

from .set_transformer import SetTransformerEncoder, ISAB, MAB
from .spectral_blocks import SpectralConv1D, FNOBlock
from .gnot import GNOTOperator
from .diffusion import LatentDiffusionDecoder, cosine_beta_schedule
from .auxiliary_heads import SwitchHead, PINOHead, SpectralHead
from .losses import diffusion_loss, switch_loss, pino_loss, spectral_loss
from .deep_ensemble import DeepEnsemble, BayesianLinear
from .trainer import NeptuneProSurrogate

__all__ = [
    "SetTransformerEncoder", "ISAB", "MAB",
    "SpectralConv1D", "FNOBlock", "GNOTOperator",
    "LatentDiffusionDecoder", "cosine_beta_schedule",
    "SwitchHead", "PINOHead", "SpectralHead",
    "diffusion_loss", "switch_loss", "pino_loss", "spectral_loss",
    "DeepEnsemble", "BayesianLinear", "NeptuneProSurrogate",
]
