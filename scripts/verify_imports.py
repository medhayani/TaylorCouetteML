"""Verify that code7 PRO modules import + forward correctly."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


MODULES = [
    "common", "common.seeds", "common.io_utils", "common.interp_utils",
    "common.normalizers", "common.logging_utils",
    "data_pipeline", "data_pipeline.canonical", "data_pipeline.dataset",
    # NEPTUNE_PRO
    "neptune_pro", "neptune_pro.set_transformer", "neptune_pro.spectral_blocks",
    "neptune_pro.gnot", "neptune_pro.diffusion", "neptune_pro.auxiliary_heads",
    "neptune_pro.losses", "neptune_pro.deep_ensemble", "neptune_pro.trainer",
    # HYDRA_PRO
    "hydra_marl_pro", "hydra_marl_pro.agents.detector",
    "hydra_marl_pro.agents.shape", "hydra_marl_pro.agents.geometry",
    "hydra_marl_pro.agents.smoothness", "hydra_marl_pro.agents.validator",
    "hydra_marl_pro.agents.manager",
    "hydra_marl_pro.agents.trend", "hydra_marl_pro.agents.asymmetry",
    "hydra_marl_pro.communication.gat",
    "hydra_marl_pro.critic.multi_agent_transformer",
    "hydra_marl_pro.policies.diffusion_policy",
    "hydra_marl_pro.algorithms.iql", "hydra_marl_pro.algorithms.coma",
    "hydra_marl_pro.algorithms.mappo", "hydra_marl_pro.algorithms.madt",
    "hydra_marl_pro.algorithms.pbt",
    "hydra_marl_pro.env.refinement_env",
    # SSST / SARL / MARL PRO
    "ssst_pro", "ssst_pro.ssst_model",
    "sarl_pro", "sarl_pro.feature_extractor", "sarl_pro.sac_pro",
    "marl_pro", "marl_pro.marl_model",
]


def stage1_imports():
    n_ok = n_fail = 0
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  [OK]   {name}"); n_ok += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {type(e).__name__}: {e}")
            traceback.print_exc(); n_fail += 1
    print(f"\n  Total: {n_ok} OK / {n_fail} FAIL")
    return n_fail


def stage2_forwards():
    import torch
    failures = 0

    def _try(label, fn):
        nonlocal failures
        try:
            fn(); print(f"  [OK]   {label}")
        except Exception as e:
            print(f"  [FAIL] {label} -> {type(e).__name__}: {e}")
            traceback.print_exc(); failures += 1

    # ---- NEPTUNE_PRO ----
    from neptune_pro.trainer import NeptuneProSurrogate
    cfg_n = {
        "context": {"in_dim": 24, "d_model": 64, "num_isab_layers": 2,
                     "num_inducing_points": 8, "num_heads": 4, "dropout": 0.0},
        "operator": {"in_dim": 1, "d_model": 64, "num_blocks": 2, "num_heads": 4,
                      "num_fourier_modes": 8, "fourier_features": 8, "dropout": 0.0},
        "diffusion": {"num_timesteps": 50, "beta_schedule_s": 0.008,
                       "inference": {"method": "ddim", "num_steps": 4, "eta": 0.0}},
        "losses": {"diffusion": 1.0, "switch": 0.5, "pino": 0.1, "spectral": 0.05},
    }

    def _neptune_pro():
        m = NeptuneProSurrogate(cfg_n)
        ctx = torch.randn(2, 24); s = torch.linspace(0, 1, 16).unsqueeze(0).expand(2, -1)
        out = m(ctx, s)
        assert out["u"].shape == (2, 64, 16)
        batch = {"ctx": ctx, "s": s, "ta_true": torch.randn(2, 16),
                  "switch_label": torch.zeros(2, 16),
                  "switch_center": torch.zeros(2), "switch_width": torch.zeros(2)}
        loss = m.compute_loss(batch)
        assert loss["loss"].dim() == 0
        sample = m.sample(ctx, s, num_steps=4)
        assert sample.shape == (2, 16)
    _try("NEPTUNE_PRO surrogate forward + loss + sample", _neptune_pro)

    # ---- HYDRA_PRO ----
    from hydra_marl_pro.agents.trend import TrendAgent
    from hydra_marl_pro.agents.asymmetry import AsymmetryAgent
    from hydra_marl_pro.critic.multi_agent_transformer import MultiAgentTransformerCritic

    def _hydra_pro_new_agents():
        t = TrendAgent(32, 3, 64); a = AsymmetryAgent(32, 4, 64)
        x = torch.randn(2, 32)
        assert t(x).shape == (2, 64)
        assert a(x).shape == (2, 64)

    def _hydra_pro_critic_7agents():
        # 7 agents with action dims [3,4,5,49,3,4,7]
        m = MultiAgentTransformerCritic(state_dim=32,
                                          action_dims=[3, 4, 5, 49, 3, 4, 7],
                                          d_model=64, num_layers=2, num_heads=4)
        s = torch.randn(2, 32)
        actions = [torch.randn(2, d) for d in [3, 4, 5, 49, 3, 4, 7]]
        q = m(s, actions); assert q.shape == (2,)
    _try("HYDRA_PRO new agents (Trend, Asymmetry)", _hydra_pro_new_agents)
    _try("HYDRA_PRO critic with 7 agents", _hydra_pro_critic_7agents)

    # ---- SSST_PRO ----
    from ssst_pro.ssst_model import SSSTProSurrogate

    def _ssst_pro():
        cfg = {"d_model": 64, "num_blocks": 2, "num_heads": 4, "trunk_kernel": 5,
                "ff_mult": 2, "dropout": 0.0,
                "num_modes": 4, "num_experts": 4, "top_k_experts": 2,
                "expert_kernel_sizes": [3, 5, 7, 9],
                "expert_dilations":    [1, 2, 3, 4],
                "gate_hidden": 64,
                "ctx_dim": 24,
                "losses": {"data": 1, "grad": 0.5, "curv": 0.3, "switch": 0.4,
                            "extrema": 0.1, "spectral": 0.05}}
        m = SSSTProSurrogate(cfg)
        ctx = torch.randn(2, 24); s = torch.linspace(0, 1, 16).unsqueeze(0).expand(2, -1)
        out = m(ctx, s)
        assert out["ta_pred"].shape == (2, 16)
        assert out["switch_logits"].shape == (2, 16)
        loss = m.compute_loss({"ctx": ctx, "s": s, "ta_true": torch.randn(2, 16),
                                "switch_label": torch.zeros(2, 16),
                                "switch_center": torch.zeros(2)})
        assert loss["loss"].dim() == 0
    _try("SSST_PRO forward + loss", _ssst_pro)

    # ---- SARL_PRO ----
    from sarl_pro.feature_extractor import SARLProFeatureExtractor
    from sarl_pro.sac_pro import SACPro

    def _sarl_pro():
        cfg = {"seq_hidden": 32, "seq_out": 48, "static_hidden": 32,
                "fusion_hidden": 64, "num_attn_heads": 4}
        ext = SARLProFeatureExtractor(obs_seq_dim=9, obs_seq_T=16,
                                        static_dim=23, cfg=cfg)
        obs = torch.randn(2, 16, 9); st = torch.randn(2, 23)
        feat = ext(obs, st); assert feat.shape == (2, 64)

        sac = SACPro(feature_dim=64, action_dim=4,
                       actor_layers=[64, 64], critic_layers=[64, 64])
        a, logp = sac.actor(feat); assert a.shape == (2, 4) and logp.shape == (2,)
        l_c = sac.critic_loss(feat, a, torch.randn(2), feat, torch.zeros(2))
        l_a, l_al = sac.actor_and_alpha_loss(feat)
        assert l_c.dim() == 0 and l_a.dim() == 0 and l_al.dim() == 0
    _try("SARL_PRO feature extractor + SAC", _sarl_pro)

    # ---- MARL_PRO ----
    from marl_pro.marl_model import MARLProSystem

    def _marl_pro():
        cfg = {"num_agents": 3,
                "agent_features": {"seq_hidden": 32, "seq_out": 32,
                                     "static_hidden": 24, "fusion_hidden": 48,
                                     "num_attn_heads": 4},
                "actor_layers": [64, 64], "critic_layers": [64, 64],
                "cross_agent_attention": {"enabled": True, "num_rounds": 2,
                                            "num_heads": 4}}
        m = MARLProSystem(obs_seq_dim=9, obs_seq_T=16, static_dim=23, cfg=cfg)
        obs = torch.randn(2, 16, 9); st = torch.randn(2, 23)
        h = m.encode(obs, st); assert h.shape == (2, 3, 48)
        actions = m.sample_actions(obs, st)
        assert len(actions) == 3
        for i, ad in enumerate(MARLProSystem.AGENT_ACTION_DIMS):
            assert actions[i].shape == (2, ad)
    _try("MARL_PRO 3-agent SAC system + cross-agent attention", _marl_pro)

    return failures


if __name__ == "__main__":
    print("=" * 70); print("STAGE 1 — module imports"); print("=" * 70)
    n_imp = stage1_imports()
    print(); print("=" * 70); print("STAGE 2 — forward passes"); print("=" * 70)
    n_fwd = stage2_forwards()
    total = n_imp + n_fwd
    print(); print("=" * 70); print(f"GRAND TOTAL: {total} failures"); print("=" * 70)
    sys.exit(1 if total > 0 else 0)
