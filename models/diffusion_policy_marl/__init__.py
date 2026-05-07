"""HYDRA-MARL_PRO — 7 agents (5 + Trend + Asymmetry), deeper critic, more rounds."""

from .agents.detector import DetectorAgent
from .agents.shape import ShapeAgent
from .agents.geometry import GeometryAgent
from .agents.smoothness import SmoothnessAgent
from .agents.validator import ValidatorAgent
from .agents.manager import HierarchicalManager
from .agents.trend import TrendAgent
from .agents.asymmetry import AsymmetryAgent
from .communication.gat import GraphAttentionLayer, GATCommunication
from .critic.multi_agent_transformer import MultiAgentTransformerCritic
from .policies.diffusion_policy import DiffusionPolicy
from .algorithms.iql import IQLCritic
from .algorithms.coma import compute_coma_advantage
from .algorithms.mappo import mappo_clip_loss
from .algorithms.madt import MultiAgentDecisionTransformer
from .algorithms.pbt import PopulationBasedTrainer
from .env.refinement_env import RefinementEnv

__all__ = [
    "DetectorAgent",
    "ShapeAgent",
    "GeometryAgent",
    "SmoothnessAgent",
    "ValidatorAgent",
    "HierarchicalManager",
    "TrendAgent",
    "AsymmetryAgent",
    "GraphAttentionLayer",
    "GATCommunication",
    "MultiAgentTransformerCritic",
    "DiffusionPolicy",
    "IQLCritic",
    "compute_coma_advantage",
    "mappo_clip_loss",
    "MultiAgentDecisionTransformer",
    "PopulationBasedTrainer",
    "RefinementEnv",
]
