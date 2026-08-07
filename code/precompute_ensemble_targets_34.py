"""Pre-compute the 34-model ensemble median Ta_norm targets for every branch.

For each branch in branch_functional_descriptors.csv the curve is resampled
at 101 equally-spaced k points (the same grid the training scripts use) and
every trained supervised surrogate of the full ensemble predicts Ta_norm
there. The median across the 34 predictions per (branch, k) point is the
teacher target used to train the distilled student (DIST v2).

Models included (34):
    12 precision V2  (3 seeds x 4 archis: SIREN, DeepONet, Chebyshev,
                      Multi-Mode Envelope SIREN)
     5 CSON V1       (5 seeds)
     5 CNP V1        (5 seeds, zero-shot mode)
     7 STAR V1       (7 seeds)
     3 DISTIL V1     (3 seeds, STAR architecture)
     1 SSST V2       (Sparse-MoE Transformer)
     1 NEPTUNE V2    (FNO + Latent Diffusion, deterministic seed)

The loading code mirrors predict_3models_v8_rl_aug.py, so the median
computed here is exactly the MED-NN baseline of the production pipeline.

Output: data/processed/ensemble_median_targets_34.npz with arrays
    E (n_branches,)
    branch_local_id (n_branches,)
    ta_norm_median (n_branches, 101)
    k_norm (101,)
plus a JSON manifest of the ensemble composition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
for p in (str(CODE_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_pipeline.dataset import NeptuneProfileDataset
from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.sparse_moe_transformer.ssst_model import SSSTProSurrogate
from models.fno_latent_diffusion.trainer import NeptuneProSurrogate
from models.cson.model import CSON
from models.cnp_tc.model import CNP_TC
from models.star.model import STAR


PRECISION_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def find_configs():
    for cand in (ROOT / "configs" / "sizes.yaml",
                 CODE_DIR / "configs" / "sizes.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError("configs/sizes.yaml not found")


def load_all_surrogates(runs, device, train_in_dim):
    models = []          # (name, callable-tag, model)

    print("Loading 12 precision V2 models...")
    for sd in sorted([d for d in (runs / "precision_v2").iterdir()
                      if d.is_dir() and d.name.startswith("seed_")]):
        m1 = SIRENRegressor(ctx_dim=23, hidden=384, depth=8).to(device)
        m1.load_state_dict(torch.load(sd / "siren" / "best.pt",
                                      map_location=device, weights_only=False)["state_dict"])
        m1.eval(); models.append((f"precision/{sd.name}/siren", "plain", m1))
        m2 = DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4,
                      latent_dim=192, fourier_bands=24).to(device)
        m2.load_state_dict(torch.load(sd / "deeponet" / "best.pt",
                                      map_location=device, weights_only=False)["state_dict"])
        m2.eval(); models.append((f"precision/{sd.name}/deeponet", "plain", m2))
        m3 = ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32).to(device)
        m3.load_state_dict(torch.load(sd / "chebyshev" / "best.pt",
                                      map_location=device, weights_only=False)["state_dict"])
        m3.eval(); models.append((f"precision/{sd.name}/chebyshev", "plain", m3))
        m4 = MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7).to(device)
        m4.load_state_dict(torch.load(sd / "envelope_siren" / "best.pt",
                                      map_location=device, weights_only=False)["state_dict"])
        m4.eval(); models.append((f"precision/{sd.name}/envelope_siren", "plain", m4))

    print("Loading 5 CSON V1 seeds...")
    for sd in sorted([d for d in (runs / "cson_v1").iterdir()
                      if d.is_dir() and d.name.startswith("seed_")]):
        m = CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5,
                 num_heads=8, dropout=0.05, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt",
                                     map_location=device, weights_only=False)["state_dict"])
        m.eval(); models.append((f"cson/{sd.name}", "plain", m))

    print("Loading 5 CNP V1 seeds (zero-shot)...")
    for sd in sorted([d for d in (runs / "cnp_v1").iterdir()
                      if d.is_dir() and d.name.startswith("seed_")]):
        m = CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256,
                   n_enc_layers=6, n_dec_layers=3, n_heads=8,
                   dropout=0.05, n_E_freq=32, n_k_freq=16,
                   max_ctx_points=32).to(device)
        m.load_state_dict(torch.load(sd / "best.pt",
                                     map_location=device, weights_only=False)["state_dict"])
        m.eval(); models.append((f"cnp/{sd.name}", "cnp", m))

    print("Loading 7 STAR V1 seeds...")
    for sd in sorted([d for d in (runs / "star_v1").iterdir()
                      if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                 n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                 use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt",
                                     map_location=device, weights_only=False)["state_dict"])
        m.eval(); models.append((f"star/{sd.name}", "star", m))

    print("Loading 3 DISTIL V1 seeds...")
    for sd in sorted([d for d in (runs / "distil_v1").iterdir()
                      if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                 n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                 use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt",
                                     map_location=device, weights_only=False)["state_dict"])
        m.eval(); models.append((f"distil/{sd.name}", "star", m))

    print("Loading SSST V2 and NEPTUNE V2...")
    full_cfg = yaml.safe_load(find_configs().read_text(encoding="utf-8"))
    cfg_ssst = full_cfg["ssst_pro"]; cfg_ssst["ctx_dim"] = train_in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    ssst.load_state_dict(torch.load(runs / "ssst_v2" / "best.pt", map_location=device,
                                    weights_only=False)["state_dict"])
    ssst.eval()
    cfg_n = full_cfg["neptune_pro"]; cfg_n["context"]["in_dim"] = train_in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    neptune.load_state_dict(torch.load(runs / "neptune_v2" / "member_00" / "best.pt",
                                       map_location=device, weights_only=False)["state_dict"])
    neptune.eval()
    return models, ssst, neptune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neptune_steps", type=int, default=120)
    ap.add_argument("--out_name", type=str, default="ensemble_median_targets_34.npz")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    runs = ROOT / "data" / "runs"
    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv")
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))

    n_resampled = 101
    s_grid = np.linspace(0.0, 1.0, n_resampled, dtype=np.float32)
    k_norm_np = 2.0 * s_grid - 1.0
    k_t = torch.from_numpy(k_norm_np).unsqueeze(0).to(device)
    s_t = torch.from_numpy(s_grid).unsqueeze(0).to(device)

    prec_norm = np.load(runs / "precision_v2" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6

    anc_raw_all = np.stack([
        desc["k_left"].values,
        desc["k_right"].values,
        np.log10(desc["Ta_min"].clip(lower=1e-6).values),
        np.log10(desc["Ta_max"].clip(lower=1e-6).values),
    ], axis=1).astype(np.float32)
    anc_mean = anc_raw_all.mean(axis=0)
    anc_std = anc_raw_all.std(axis=0) + 1e-6

    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                    ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    models, ssst, neptune = load_all_surrogates(runs, device, train_ds.in_dim)
    n_transformer = len(models)
    print(f"Models loaded: {n_transformer} + SSST + NEPTUNE = {n_transformer + 2}")

    n_branches = len(desc)
    print(f"Processing {n_branches} branches at {n_resampled} k points each...")

    ta_norm_median = np.zeros((n_branches, n_resampled), dtype=np.float32)
    n_members = np.zeros(n_branches, dtype=np.int16)
    E_arr = np.zeros(n_branches, dtype=np.float32)
    branch_arr = np.zeros(n_branches, dtype=np.int32)

    empty_ck = torch.zeros(1, 32, device=device)
    empty_cta = torch.zeros(1, 32, device=device)
    empty_mask = torch.zeros(1, 32, device=device)

    for idx, (_, row) in enumerate(desc.iterrows()):
        E_val = float(row["E"]); b_id = int(row["branch_local_id"])
        E_arr[idx] = E_val; branch_arr[idx] = b_id

        ctx_p = []
        for col in PRECISION_FEATURES:
            v = row.get(col, np.nan)
            try: ctx_p.append(float(v))
            except (TypeError, ValueError): ctx_p.append(0.0)
        ctx_p_arr = np.asarray(ctx_p, dtype=np.float32)
        ctx_p_arr = (ctx_p_arr - prec_ctx_mean) / prec_ctx_std
        ctx_p_arr = np.clip(np.nan_to_num(ctx_p_arr, nan=0.0,
                                          posinf=0.0, neginf=0.0), -5.0, 5.0)
        ctx_t = torch.from_numpy(ctx_p_arr).unsqueeze(0).to(device)
        anc_raw = np.array([float(row["k_left"]), float(row["k_right"]),
                            np.log10(max(float(row["Ta_min"]), 1e-6)),
                            np.log10(max(float(row["Ta_max"]), 1e-6))], dtype=np.float32)
        anc_norm = np.clip(np.nan_to_num((anc_raw - anc_mean) / anc_std,
                                         nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
        anc_t = torch.from_numpy(anc_norm).unsqueeze(0).to(device)
        logE_t = torch.tensor([np.log10(max(E_val, 1e-30))],
                              dtype=torch.float32, device=device)

        preds = []
        with torch.no_grad():
            for name, tag, m in models:
                if tag == "plain":
                    preds.append(m(k_t, ctx_t).squeeze(0).cpu().numpy())
                elif tag == "cnp":
                    preds.append(m(k_t, ctx_t, anc_t, logE_t,
                                   empty_ck, empty_cta, empty_mask).squeeze(0).cpu().numpy())
                else:  # star / distil
                    preds.append(m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy())
            if (E_val, b_id) in keys_to_idx:
                i = keys_to_idx[(E_val, b_id)]
                ctx_n_t = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
                preds.append(ssst.predict(ctx_n_t, s_t).squeeze(0).cpu().numpy())
                ta_n = neptune.sample(ctx_n_t, s_t, num_steps=args.neptune_steps)
                preds.append(ta_n.squeeze(0).cpu().numpy())

        stack = np.stack(preds, axis=0)
        n_members[idx] = stack.shape[0]
        ta_norm_median[idx] = np.clip(np.median(stack, axis=0), 0.0, 1.0)

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{n_branches} done", flush=True)

    out_path = ROOT / "data" / "processed" / args.out_name
    np.savez(out_path,
             E=E_arr, branch_local_id=branch_arr,
             ta_norm_median=ta_norm_median, k_norm=k_norm_np)
    manifest = {
        "n_models": int(n_transformer + 2),
        "composition": {
            "precision_v2": 12, "cson_v1": 5, "cnp_v1": 5, "star_v1": 7,
            "distil_v1": 3, "ssst_v2": 1, "neptune_v2": 1,
        },
        "members": [name for name, _, _ in models] + ["ssst_v2", "neptune_v2/member_00"],
        "neptune_steps": args.neptune_steps,
        "branches_with_full_34": int((n_members == n_transformer + 2).sum()),
        "branches_total": int(n_branches),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Array shape: {ta_norm_median.shape}, dtype: {ta_norm_median.dtype}")
    print(f"Range: [{ta_norm_median.min():.4f}, {ta_norm_median.max():.4f}]")
    print(f"Branches with full 34 members: {manifest['branches_with_full_34']}"
          f"/{n_branches}")


if __name__ == "__main__":
    main()
