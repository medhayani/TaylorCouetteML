"""MARL_PRO — 3-agent SAC (Localizer / Shape / Geometry) with cross-agent attention.

Bigger nets than code4 MARL, plus attention layer between agents that
exchanges latent features (not just freezing the previous agent).
"""

from .marl_model import MARLProSystem, CrossAgentAttention, AgentSAC

__all__ = ["MARLProSystem", "CrossAgentAttention", "AgentSAC"]
