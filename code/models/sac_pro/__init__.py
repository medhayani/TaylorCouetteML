"""SARL_PRO — pure-PyTorch SAC with bigger networks (~30M params)."""

from .feature_extractor import SARLProFeatureExtractor
from .sac_pro import SACPro

__all__ = ["SARLProFeatureExtractor", "SACPro"]
