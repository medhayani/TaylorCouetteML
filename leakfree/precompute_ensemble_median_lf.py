"""Leak-free ensemble median over the supervised teacher pool.

Mirrors code/precompute_ensemble_targets_34.py (same normalisations, same
calls), restricted to the pool the DIST student of the paper was distilled
from -- precision (4 architectures x 3 seeds), CSON (5), CNP zero-shot (5),
STAR (7) -- and run on the LEAK-FREE descriptors of every elasticity
(branch structure, descriptors and anchors interpolated from the training
neighbours).  Families whose run folder is absent are skipped, so the
script can be run on a partial pool.

Usage:
  python precompute_ensemble_median_lf.py --repo <TaylorCouetteML root>
      --runs <folder holding precision_lf/ cson_lf/ cnp_leakfree/ star_lf/>
      --desc <branch_functional_descriptors_leakfree_all.csv>
      --out  <ensemble_median_targets_lf.npz>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

PRECISION_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry",
    "slope_left_local", "slope_right_local", "global_slope", "curvature_at_min",
    "roughness_rmse", "normalized_arc_length", "has_switch_left", "has_switch_right",
    "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]


def load_pool(runs: Path, device):
    from models.siren.model import SIRENRegressor
    from models.deeponet.model import DeepONet
    from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
    from models.envelope_siren.model import MultiModeEnvelopeSIREN
    from models.cson.model import CSON
    from models.cnp_tc.model import CNP_TC
    from models.star.model import STAR
    models = []
    def ld(m, p):
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["state_dict"]); m.eval(); return m
    seeds = lambda d: sorted([x for x in d.iterdir() if x.is_dir() and x.name.startswith("seed_")]) if d.exists() else []
    for sd in seeds(runs / "precision_lf"):
        if (sd / "siren" / "best.pt").exists():
            models.append((f"precision/{sd.name}/siren", "plain", ld(SIRENRegressor(ctx_dim=23, hidden=384, depth=8).to(device), sd / "siren" / "best.pt")))
        if (sd / "deeponet" / "best.pt").exists():
            models.append((f"precision/{sd.name}/deeponet", "plain", ld(DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4, latent_dim=192, fourier_bands=24).to(device), sd / "deeponet" / "best.pt")))
        if (sd / "chebyshev" / "best.pt").exists():
            models.append((f"precision/{sd.name}/chebyshev", "plain", ld(ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32).to(device), sd / "chebyshev" / "best.pt")))
        if (sd / "envelope_siren" / "best.pt").exists():
            models.append((f"precision/{sd.name}/envelope_siren", "plain", ld(MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7).to(device), sd / "envelope_siren" / "best.pt")))
    for sd in seeds(runs / "cson_lf"):
        if (sd / "best.pt").exists():
            models.append((f"cson/{sd.name}", "plain", ld(CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5, num_heads=8, dropout=0.05, use_spectral_norm=True).to(device), sd / "best.pt")))
    for sd in seeds(runs / "cnp_leakfree"):
        if (sd / "best.pt").exists():
            models.append((f"cnp/{sd.name}", "cnp", ld(CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6, n_dec_layers=3, n_heads=8, dropout=0.05, n_E_freq=32, n_k_freq=16, max_ctx_points=32).to(device), sd / "best.pt")))
    for sd in seeds(runs / "star_lf"):
        if (sd / "best.pt").exists():
            models.append((f"star/{sd.name}", "star", ld(STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True).to(device), sd / "best.pt")))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--desc", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ctx_stats", default=None, help="test_branches.npz whose ctx_mean/ctx_std normalise the descriptors (default: precision_lf, else cnp_leakfree)")
    a = ap.parse_args()
    repo, runs = Path(a.repo), Path(a.runs)
    sys.path.insert(0, str(repo / "code")); sys.path.insert(0, str(repo))
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desc = pd.read_csv(a.desc); desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    stats_path = Path(a.ctx_stats) if a.ctx_stats else next(p for p in [runs / "precision_lf" / "test_branches.npz", runs / "cnp_leakfree" / "test_branches.npz"] if p.exists())
    st = np.load(stats_path, allow_pickle=False)
    ctx_mean = st["ctx_mean"].astype(np.float32); ctx_std = st["ctx_std"].astype(np.float32) + 1e-6
    print("ctx stats from", stats_path)
    anc_all = np.stack([desc["k_left"].values, desc["k_right"].values,
                        np.log10(desc["Ta_min"].clip(lower=1e-6).values), np.log10(desc["Ta_max"].clip(lower=1e-6).values)], axis=1).astype(np.float32)
    anc_mean = anc_all.mean(axis=0); anc_std = anc_all.std(axis=0) + 1e-6
    models = load_pool(runs, device)
    fam = {}
    for n, _, _ in models: fam[n.split("/")[0]] = fam.get(n.split("/")[0], 0) + 1
    print(f"pool: {len(models)} models {fam}")
    n_res = 101; s_grid = np.linspace(0.0, 1.0, n_res, dtype=np.float32); k_norm = 2.0 * s_grid - 1.0
    k_t = torch.from_numpy(k_norm).unsqueeze(0).to(device)
    zk = torch.zeros(1, 32, device=device); zm = torch.zeros(1, 32, device=device)
    med = np.zeros((len(desc), n_res), np.float32); E_arr = np.zeros(len(desc), np.float64); b_arr = np.zeros(len(desc), np.int32)
    for idx, (_, row) in enumerate(desc.iterrows()):
        E_val = float(row["E"]); b_id = int(row["branch_local_id"]); E_arr[idx] = E_val; b_arr[idx] = b_id
        ctx = np.array([float(row.get(c, 0.0)) if pd.notna(row.get(c, np.nan)) else 0.0 for c in PRECISION_FEATURES], np.float32)
        ctx = np.clip(np.nan_to_num((ctx - ctx_mean) / ctx_std), -5, 5).astype(np.float32)
        ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
        anc = np.array([row["k_left"], row["k_right"], np.log10(max(row["Ta_min"], 1e-6)), np.log10(max(row["Ta_max"], 1e-6))], np.float32)
        anc_t = torch.from_numpy(np.clip(np.nan_to_num((anc - anc_mean) / anc_std), -5, 5).astype(np.float32)).unsqueeze(0).to(device)
        logE_t = torch.tensor([np.log10(max(E_val, 1e-30))], dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for name, tag, m in models:
                if tag == "plain": preds.append(m(k_t, ctx_t).squeeze(0).cpu().numpy())
                elif tag == "cnp": preds.append(m(k_t, ctx_t, anc_t, logE_t, zk, zk, zm).squeeze(0).cpu().numpy())
                else: preds.append(m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy())
        med[idx] = np.clip(np.median(np.stack(preds, 0), axis=0), 0.0, 1.0)
        if (idx + 1) % 100 == 0: print(f"  {idx + 1}/{len(desc)}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, E=E_arr, branch_local_id=b_arr, ta_norm_median=med, k_norm=k_norm)
    out.with_suffix(".manifest.json").write_text(json.dumps({"n_models": len(models), "composition": fam, "members": [n for n, _, _ in models],
                                                             "descriptors": str(a.desc), "ctx_stats": str(stats_path)}, indent=2))
    print("saved", out, med.shape, f"range [{med.min():.3f}, {med.max():.3f}]")


if __name__ == "__main__":
    main()
