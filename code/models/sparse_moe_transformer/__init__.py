"""SSST_PRO — pure-PyTorch port of code4's Switch-Segment Sparse Transformer,
scaled up: d_model 384 (vs 224), 10 trunk blocks (vs 6), 16 experts top-4
(vs 10/3), 8 modes (vs 5). Expected ~80M params.
"""

from .ssst_model import SSSTProSurrogate, ExpertConv1D, MoEGating, TrunkBlock

__all__ = ["SSSTProSurrogate", "ExpertConv1D", "MoEGating", "TrunkBlock"]
