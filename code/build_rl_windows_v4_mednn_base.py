"""Rebuild RL switch windows with MED-NN as the base prediction.

Input:
    rl_switch_windows_{train,val,test}.npz       (v3, base = D5)
    All 34 supervised surrogate checkpoints under data/runs/.

Output:
    rl_switch_windows_v4_{train,val,test}.npz    (base = MED-NN)

Procedure:
    For every (window_id, E, branch_local_id) in the v3 windows:
        1. Compute the absolute s axis from   s = center_pred + half * local_grid.
        2. Map to physical k:                 k = k_left + s * (k_right - k_left).
        3. Convert to k_norm in [-1, 1]:      k_norm = 2*s - 1.
        4. Run all 34 surrogates at k_norm and average via median.
        5. Renormalise back to ta_norm in [0, 1] (clip), use as new y_pred.
        6. Recompute obs_seq features that depend on y_pred:
              dy/ds, d2y/ds2, |dy|, |d2y|     (use central differences on
              the new y_pred over the window grid).
        7. Keep s_rel, switch_prob, switch_focus, curv_focus from v3.
        8. Save everything else identically.

This is intended to be used as the base for the v3 RL refiner training:
the v3 SARL_v2/MARL_v2 learned to refine a poor D5 base; with a clearly
better MED-NN base the refiner can focus its capacity on the residual
near the switch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT_REPO = Path("C:/Users/hayan/Desktop/Code_Final_IA7/_TCML_check")
sys.path.insert(0, str(ROOT_REPO))

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


def load_all_supervised(runs, device, train_in_dim):
    prec_models = {}
    for sd in sorted([d for d in (runs / "precision_v2").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        ms = {}
        m1 = SIRENRegressor(ctx_dim=23, hidden=384, depth=8).to(device)
        m1.load_state_dict(torch.load(sd / "siren" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m1.eval(); ms["SIREN"] = m1
        m2 = DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4,
                          latent_dim=192, fourier_bands=24).to(device)
        m2.load_state_dict(torch.load(sd / "deeponet" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m2.eval(); ms["DeepONet"] = m2
        m3 = ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32).to(device)
        m3.load_state_dict(torch.load(sd / "chebyshev" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m3.eval(); ms["Chebyshev"] = m3
        m4 = MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7).to(device)
        m4.load_state_dict(torch.load(sd / "envelope_siren" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m4.eval(); ms["EnvelopeSIREN"] = m4
        prec_models[sd.name] = ms

    cson_models = []
    for sd in sorted([d for d in (runs / "cson_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5,
                    num_heads=8, dropout=0.05, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); cson_models.append(m)

    cnp_models = []
    for sd in sorted([d for d in (runs / "cnp_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6,
                      n_dec_layers=3, n_heads=8, dropout=0.05,
                      n_E_freq=32, n_k_freq=16, max_ctx_points=32).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); cnp_models.append(m)

    star_models = []
    for sd in sorted([d for d in (runs / "star_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                    n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                    use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); star_models.append(m)

    distil_models = []
    for sd in sorted([d for d in (runs / "distil_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                    n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                    use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); distil_models.append(m)

    full_cfg = yaml.safe_load((ROOT_REPO / "configs" / "sizes.yaml")
                                  .read_text(encoding="utf-8"))
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
    return prec_models, cson_models, cnp_models, star_models, distil_models, ssst, neptune


@torch.no_grad()
def predict_median_at_k(k_phys: np.ndarray, row, log10E: float,
                          prec_models, cson_models, cnp_models,
                          star_models, distil_models, ssst, neptune,
                          full_ds, keys_to_idx,
                          prec_ctx_mean, prec_ctx_std,
                          anc_mean, anc_std,
                          neptune_steps: int, device) -> np.ndarray:
    """Return MED-NN normalised prediction (length T) at given physical k array."""
    k_left, k_right = float(row["k_left"]), float(row["k_right"])
    Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
    amp = max(Ta_max - Ta_min, 1e-12)
    # s in [0,1], k_norm in [-1,1]
    s = (k_phys - k_left) / (k_right - k_left)
    k_norm = (2.0 * s - 1.0).astype(np.float32)
    k_t = torch.from_numpy(k_norm).unsqueeze(0).to(device)

    # precision context
    ctx_p = []
    for col in PRECISION_FEATURES:
        v = row.get(col, np.nan)
        try: ctx_p.append(float(v))
        except (TypeError, ValueError): ctx_p.append(0.0)
    ctx_p_arr = np.asarray(ctx_p, dtype=np.float32)
    ctx_p_arr = (ctx_p_arr - prec_ctx_mean) / prec_ctx_std
    ctx_p_arr = np.clip(np.nan_to_num(ctx_p_arr, nan=0.0, posinf=0.0,
                                          neginf=0.0), -5.0, 5.0)
    ctx_t = torch.from_numpy(ctx_p_arr).unsqueeze(0).to(device)

    anc_raw = np.array([k_left, k_right,
                          np.log10(max(Ta_min, 1e-6)),
                          np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
    anc_norm = np.clip(np.nan_to_num((anc_raw - anc_mean) / anc_std,
                                          nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
    anc_t = torch.from_numpy(anc_norm).unsqueeze(0).to(device)
    logE_t = torch.tensor([log10E], dtype=torch.float32, device=device)

    preds = []
    for seed, ms in prec_models.items():
        for name, m in ms.items():
            ta = m(k_t, ctx_t).squeeze(0).cpu().numpy()
            preds.append(ta)
    for m in cson_models:
        ta = m(k_t, ctx_t).squeeze(0).cpu().numpy()
        preds.append(ta)
    empty_ck = torch.zeros(1, 32, device=device)
    empty_cta = torch.zeros(1, 32, device=device)
    empty_mask = torch.zeros(1, 32, device=device)
    for m in cnp_models:
        ta = m(k_t, ctx_t, anc_t, logE_t,
                  empty_ck, empty_cta, empty_mask).squeeze(0).cpu().numpy()
        preds.append(ta)
    for m in star_models:
        ta = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
        preds.append(ta)
    for m in distil_models:
        ta = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
        preds.append(ta)
    E_val = float(row["E"]); b_id = int(row["branch_local_id"])
    if (E_val, b_id) in keys_to_idx:
        i = keys_to_idx[(E_val, b_id)]
        ctx_n_t = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
        s_n = s.astype(np.float32)
        s_n_t = torch.from_numpy(s_n).unsqueeze(0).to(device)
        ta_s = ssst.predict(ctx_n_t, s_n_t).squeeze(0).cpu().numpy()
        preds.append(ta_s)
        ta_n = neptune.sample(ctx_n_t, s_n_t, num_steps=neptune_steps)
        ta_n = ta_n.squeeze(0).cpu().numpy()
        preds.append(ta_n)
    stack = np.stack(preds, axis=0)
    return np.clip(np.median(stack, axis=0), 0.0, 1.0).astype(np.float32)


def recompute_obs_seq_from_y(y_pred: np.ndarray, obs_seq_old: np.ndarray) -> np.ndarray:
    """obs_seq features (per window):
        col 0 = s_rel (kept from old)
        col 1 = y_pred (replaced)
        col 2 = dy_pred (recomputed via central differences)
        col 3 = d2y_pred (recomputed)
        col 4 = abs_dy_pred (recomputed)
        col 5 = abs_d2y_pred (recomputed)
        col 6 = switch_prob (kept from old)
        col 7 = switch_focus (kept from old)
        col 8 = curv_focus (kept from old)
    """
    N, T, F = obs_seq_old.shape
    assert F == 9, f"Expected 9 features, got {F}"
    out = obs_seq_old.copy()
    out[..., 1] = y_pred
    # central differences along the time axis; ds = 1/T (uniform local grid)
    ds = 2.0 / (T - 1)  # local_grid spans [-1, 1]
    dy = np.gradient(y_pred, ds, axis=-1)
    d2y = np.gradient(dy, ds, axis=-1)
    out[..., 2] = dy
    out[..., 3] = d2y
    out[..., 4] = np.abs(dy)
    out[..., 5] = np.abs(d2y)
    return out.astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                     default=Path("C:/Users/hayan/Desktop/Code_Final_IA7/Livraison/v2/Input/rl_windows_v4"))
    ap.add_argument("--in_root", type=Path,
                     default=Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/"
                                    "07_StepF__Export_RL_Windows_Pro/02_outputs/"
                                    "switch_rl_dataset_pro_v3_norm_fixed"))
    ap.add_argument("--neptune_steps", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    runs = ROOT_REPO / "data" / "runs"
    args.out_root.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT_REPO / "data" / "processed"
                          / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    desc_index = {(float(r["E"]), int(r["branch_local_id"])): r
                       for _, r in desc.iterrows()}

    prec_norm = np.load(runs / "precision_v2" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6
    anc_arr = np.stack([
        desc["k_left"].values, desc["k_right"].values,
        np.log10(desc["Ta_min"].clip(lower=1e-6).values),
        np.log10(desc["Ta_max"].clip(lower=1e-6).values),
    ], axis=1).astype(np.float32)
    anc_mean = anc_arr.mean(axis=0); anc_std = anc_arr.std(axis=0) + 1e-6

    profile_csv = ROOT_REPO / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    print("Loading 34 supervised surrogates ...", flush=True)
    (prec_models, cson_models, cnp_models, star_models, distil_models,
       ssst, neptune) = load_all_supervised(runs, device, train_ds.in_dim)
    n_sup = (sum(len(ms) for ms in prec_models.values())
              + len(cson_models) + len(cnp_models)
              + len(star_models) + len(distil_models) + 2)
    print(f"  -> {n_sup} surrogates loaded", flush=True)

    for split in ["train", "val", "test"]:
        in_npz = args.in_root / f"rl_switch_windows_{split}.npz"
        out_npz = args.out_root / f"rl_switch_windows_v4_{split}.npz"
        if not in_npz.exists():
            print(f"skip missing {in_npz}", flush=True); continue
        d = np.load(in_npz, allow_pickle=False)
        N = d["y_pred"].shape[0]
        T = d["y_pred"].shape[1]
        print(f"\n=== {split}: {N} windows, T={T} ===", flush=True)

        new_y_pred = np.zeros((N, T), dtype=np.float32)
        skipped = 0
        for i in range(N):
            E_val = float(d["E"][i]); b_id = int(d["branch_local_id"][i])
            centre = float(d["center_pred"][i])
            half = float(d["window_half_width"][i])
            local_grid = d["local_grid"][i].astype(np.float64)
            # absolute s in branch normalised [0, 1]
            s_abs = centre + half * local_grid
            # locate branch row
            key = None
            for (e_k, b_k), row in desc_index.items():
                if abs(e_k - E_val) < 1e-9 and b_k == b_id:
                    key = (e_k, b_k, row); break
            if key is None:
                # fallback: closest E with same b_id
                for (e_k, b_k), row in desc_index.items():
                    if b_k == b_id and abs(e_k - E_val) / max(E_val, 1e-30) < 1e-3:
                        key = (e_k, b_k, row); break
            if key is None:
                new_y_pred[i] = d["y_pred"][i]
                skipped += 1; continue
            _, _, row = key
            k_left = float(row["k_left"]); k_right = float(row["k_right"])
            k_phys = (k_left + s_abs * (k_right - k_left)).astype(np.float64)
            log10E = float(np.log10(max(E_val, 1e-30)))
            y_med = predict_median_at_k(
                k_phys, row, log10E,
                prec_models, cson_models, cnp_models, star_models, distil_models,
                ssst, neptune, full_ds, keys_to_idx,
                prec_ctx_mean, prec_ctx_std, anc_mean, anc_std,
                neptune_steps=args.neptune_steps, device=device)
            new_y_pred[i] = y_med
            if (i + 1) % 25 == 0 or i == N - 1:
                print(f"  window {i + 1}/{N}", flush=True)
        print(f"  skipped (no branch match): {skipped}", flush=True)

        new_obs = recompute_obs_seq_from_y(new_y_pred, d["obs_seq"].astype(np.float32))
        mae_new = np.mean(np.abs(new_y_pred - d["y_true"]))
        mae_old = np.mean(np.abs(d["y_pred"] - d["y_true"]))
        print(f"  baseline MAE old (D5)   = {mae_old:.5f}", flush=True)
        print(f"  baseline MAE new (MED)  = {mae_new:.5f}", flush=True)

        # Save with all old keys + new y_pred / obs_seq
        out_dict = {k: d[k] for k in d.files}
        out_dict["y_pred"] = new_y_pred
        out_dict["obs_seq"] = new_obs
        out_dict["y_pred_d5"] = d["y_pred"].astype(np.float32)   # keep for reference
        np.savez_compressed(out_npz, **out_dict)
        print(f"  saved -> {out_npz}", flush=True)


if __name__ == "__main__":
    main()
