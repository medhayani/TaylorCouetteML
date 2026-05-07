"""Print parameter count of every PRO model with the production config."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neptune_pro.trainer import NeptuneProSurrogate
from ssst_pro.ssst_model import SSSTProSurrogate
from sarl_pro.feature_extractor import SARLProFeatureExtractor
from sarl_pro.sac_pro import SACPro
from marl_pro.marl_model import MARLProSystem

from hydra_marl_pro.agents.detector import DetectorAgent
from hydra_marl_pro.agents.shape import ShapeAgent
from hydra_marl_pro.agents.geometry import GeometryAgent
from hydra_marl_pro.agents.smoothness import SmoothnessAgent
from hydra_marl_pro.agents.validator import ValidatorAgent
from hydra_marl_pro.agents.manager import HierarchicalManager
from hydra_marl_pro.agents.trend import TrendAgent
from hydra_marl_pro.agents.asymmetry import AsymmetryAgent
from hydra_marl_pro.communication.gat import GATCommunication
from hydra_marl_pro.critic.multi_agent_transformer import MultiAgentTransformerCritic
from hydra_marl_pro.policies.diffusion_policy import DiffusionPolicy


def n_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))

    print("=" * 70)
    print(f"{'Model':<40} {'Params (M)':>12}  Notes")
    print("-" * 70)

    # NEPTUNE_PRO (one member)
    n_cfg = cfg["neptune_pro"]
    np_model = NeptuneProSurrogate(n_cfg)
    print(f"{'NEPTUNE_PRO (1 member)':<40} {n_params(np_model):>12.2f}  M={n_cfg['ensemble']['num_members']}")
    total_n = n_params(np_model) * n_cfg["ensemble"]["num_members"]
    print(f"{'NEPTUNE_PRO (full ensemble)':<40} {total_n:>12.2f}")

    # SSST_PRO
    s_cfg = cfg["ssst_pro"]
    s_cfg.setdefault("ctx_dim", 24)
    s_model = SSSTProSurrogate(s_cfg)
    print(f"{'SSST_PRO':<40} {n_params(s_model):>12.2f}")

    # SARL_PRO
    sa = cfg["sarl_pro"]
    ext = SARLProFeatureExtractor(obs_seq_dim=9, obs_seq_T=49, static_dim=23,
                                    cfg=sa["feature_extractor"])
    sac = SACPro(feature_dim=ext.out_dim, action_dim=4,
                  actor_layers=sa["actor_layers"], critic_layers=sa["critic_layers"])
    print(f"{'SARL_PRO (extractor + SAC)':<40} {(n_params(ext) + n_params(sac)):>12.2f}")

    # MARL_PRO
    m_cfg = cfg["marl_pro"]
    m_cfg["agent_features"].setdefault("seq_out", m_cfg["agent_features"]["seq_hidden"])
    marl = MARLProSystem(obs_seq_dim=9, obs_seq_T=49, static_dim=23, cfg=m_cfg)
    print(f"{'MARL_PRO (3 agents + xattn)':<40} {n_params(marl):>12.2f}")

    # HYDRA_PRO (sum across all components)
    h = cfg["hydra_pro"]
    state_dim = 49 * 9 + 23 + 49        # = 513
    parts = []
    parts.append(("Detector",   DetectorAgent(state_dim, h["agents"]["detector"]["action_dim"],
                                                h["agents"]["detector"]["hidden_dim"])))
    parts.append(("Shape",      ShapeAgent(state_dim,    h["agents"]["shape"]["action_dim"],
                                              h["agents"]["shape"]["hidden_dim"])))
    parts.append(("Geometry",   GeometryAgent(state_dim, h["agents"]["geometry"]["action_dim"],
                                                h["agents"]["geometry"]["hidden_dim"])))
    parts.append(("Smoothness", SmoothnessAgent(state_dim, h["agents"]["smoothness"]["action_dim"],
                                                  h["agents"]["smoothness"]["hidden_dim"])))
    parts.append(("Validator",  ValidatorAgent(state_dim, h["agents"]["validator"]["action_dim"],
                                                  h["agents"]["validator"]["hidden_dim"])))
    parts.append(("Trend",      TrendAgent(state_dim, h["agents"]["trend"]["action_dim"],
                                              h["agents"]["trend"]["hidden_dim"])))
    parts.append(("Asymmetry",  AsymmetryAgent(state_dim, h["agents"]["asymmetry"]["action_dim"],
                                                 h["agents"]["asymmetry"]["hidden_dim"])))
    parts.append(("Manager",    HierarchicalManager(state_dim,
                                                      h["manager"]["subgoal_dim"],
                                                      h["manager"]["hidden_dim"])))
    parts.append(("GAT",        GATCommunication(dim=state_dim,
                                                    num_rounds=h["communication"]["num_rounds"],
                                                    num_heads=1)))
    a_dims = [h["agents"][k]["action_dim"] for k in
              ["detector", "shape", "geometry", "smoothness", "trend", "asymmetry", "validator"]]
    parts.append(("MAT critic", MultiAgentTransformerCritic(state_dim=state_dim,
                                                              action_dims=a_dims,
                                                              d_model=h["critic"]["d_model"],
                                                              num_layers=h["critic"]["num_layers"],
                                                              num_heads=h["critic"]["num_heads"])))
    for i, (k, ad) in enumerate([("detector", a_dims[0]), ("shape", a_dims[1]),
                                    ("geometry", a_dims[2]), ("smoothness", a_dims[3]),
                                    ("trend", a_dims[4]), ("asymmetry", a_dims[5])]):
        parts.append((f"DiffPol[{k}]",
                       DiffusionPolicy(action_dim=ad, state_dim=state_dim,
                                          msg_dim=state_dim,
                                          num_timesteps=h["diffusion_policy"]["num_timesteps"])))

    total_h = 0.0
    for name, mod in parts:
        p = n_params(mod); total_h += p
        print(f"{'  HYDRA_PRO/' + name:<40} {p:>12.2f}")
    print(f"{'HYDRA_PRO (total)':<40} {total_h:>12.2f}")
    print("=" * 70)
    print()
    grand = total_n + n_params(s_model) + n_params(ext) + n_params(sac) + n_params(marl) + total_h
    print(f"{'TOTAL code7 PRO':<40} {grand:>12.2f}")


if __name__ == "__main__":
    main()
