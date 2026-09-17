"""Leak-free evaluation of the supervised teacher families on the held-out
elasticities (zero-shot, inputs = interpolated descriptors and anchors).

For every family found under --runs (precision_lf, cson_lf, star_lf,
cnp_leakfree) the seed models are loaded with the family's own context
statistics (test_branches.npz of that run), evaluated on every predicted
branch of the test elasticities, and the seed-median curve is reduced to
Ta_c, k_c.  Outputs metrics.json and per_E_<family>.csv in --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

FEATS = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry", "slope_left_local",
         "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse", "normalized_arc_length",
         "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch",
         "is_last_branch", "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]
def mape(a, b): return float(100 * np.mean(np.abs(np.asarray(a) / np.asarray(b) - 1)))
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def load_family(fam: str, runs: Path, device):
    from models.siren.model import SIRENRegressor
    from models.deeponet.model import DeepONet
    from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
    from models.envelope_siren.model import MultiModeEnvelopeSIREN
    from models.cson.model import CSON
    from models.cnp_tc.model import CNP_TC
    from models.star.model import STAR
    # folder names of the repository -> architecture of the family
    arch = {"precision": "precision_lf", "cson": "cson_lf", "star": "star_lf",
            "cnp": "cnp_leakfree", "cnp_nbr": "cnp_leakfree", "cnp_block": "cnp_leakfree",
            "dist": "distil_lf"}.get(fam, fam)
    d = runs / fam
    if not d.exists(): return None, None, None
    ld = lambda m, p: (m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["state_dict"]), m.eval(), m)[2]
    seeds = sorted([x for x in d.iterdir() if x.is_dir() and x.name.startswith("seed_")])
    out = {}   # sub-family -> list of (tag, model)
    for sd in seeds:
        if arch == "precision_lf":
            for sub, ctor in [("siren", lambda: SIRENRegressor(ctx_dim=23, hidden=384, depth=8)),
                              ("deeponet", lambda: DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4, latent_dim=192, fourier_bands=24)),
                              ("chebyshev", lambda: ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32)),
                              ("envelope_siren", lambda: MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7))]:
                if (sd / sub / "best.pt").exists(): out.setdefault(sub, []).append(("plain", ld(ctor().to(device), sd / sub / "best.pt")))
        elif arch == "cson_lf" and (sd / "best.pt").exists():
            out.setdefault("cson", []).append(("plain", ld(CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5, num_heads=8, dropout=0.05, use_spectral_norm=True).to(device), sd / "best.pt")))
        elif arch == "cnp_leakfree" and (sd / "best.pt").exists():
            out.setdefault("cnp", []).append(("cnp", ld(CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6, n_dec_layers=3, n_heads=8, dropout=0.05, n_E_freq=32, n_k_freq=16, max_ctx_points=32).to(device), sd / "best.pt")))
        elif arch == "star_lf" and (sd / "best.pt").exists():
            out.setdefault("star", []).append(("star", ld(STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True).to(device), sd / "best.pt")))
        elif arch == "distil_lf" and (sd / "best.pt").exists():   # DIST student = STAR architecture distilled from the leak-free median
            out.setdefault("dist", []).append(("star", ld(STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True).to(device), sd / "best.pt")))
    st = np.load(d / "test_branches.npz", allow_pickle=False) if (d / "test_branches.npz").exists() else None
    return out, st, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--desc", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="precision,cson,star,cnp,dist", help="comma-separated folders of models_trained to evaluate")
    a = ap.parse_args(); repo, runs, out = Path(a.repo), Path(a.runs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "code")); sys.path.insert(0, str(repo))
    device = torch.device("cpu")
    desc = pd.read_csv(a.desc); desc["log10E"] = np.log10(desc.E.clip(lower=1e-30))
    split = pd.read_csv(a.split); test_E = set(np.round(split.loc[split.split == "test", "E"].to_numpy(float), 6))
    raw = pd.read_csv(repo / "data" / "Input" / "combined_data.csv"); raw.columns = ["Ta", "k", "E"]
    anc_all = np.stack([desc.k_left, desc.k_right, np.log10(desc.Ta_min.clip(lower=1e-6)), np.log10(desc.Ta_max.clip(lower=1e-6))], 1).astype(np.float32)
    anc_mean, anc_std = anc_all.mean(0), anc_all.std(0) + 1e-6
    s = np.linspace(0, 1, 101); k_t = torch.tensor(2 * s - 1, dtype=torch.float32).unsqueeze(0); zk = torch.zeros(1, 32); zm = torch.zeros(1, 32)
    metrics = {}
    for fam in a.families.split(","):
        subs, st, seeds = load_family(fam, runs, device)
        if not subs: continue
        cm = st["ctx_mean"].astype(np.float32) if st is not None else None; cs = (st["ctx_std"].astype(np.float32) + 1e-6) if st is not None else None
        am = st["anc_mean"] if (st is not None and "anc_mean" in st.files) else anc_mean; asd = st["anc_std"] if (st is not None and "anc_std" in st.files) else anc_std
        for sub, models in subs.items():
            rows = []
            for E, g in desc.groupby("E"):
                E = float(E)
                if round(E, 6) not in test_E: continue
                kp, Tp = [], []
                for _, r in g.sort_values("branch_local_id").iterrows():
                    ctx = np.array([float(r.get(f, 0.0)) if pd.notna(r.get(f, np.nan)) else 0.0 for f in FEATS], np.float32)
                    if cm is not None: ctx = np.clip(np.nan_to_num((ctx - cm) / cs), -5, 5).astype(np.float32)
                    anc = np.array([r.k_left, r.k_right, np.log10(max(r.Ta_min, 1e-6)), np.log10(max(r.Ta_max, 1e-6))], np.float32)
                    anc = np.clip(np.nan_to_num((anc - am) / asd), -5, 5).astype(np.float32)
                    ct, at, lt = torch.from_numpy(ctx).unsqueeze(0), torch.from_numpy(anc).unsqueeze(0), torch.tensor([np.log10(E)], dtype=torch.float32)
                    with torch.no_grad():
                        ys = [(m(k_t, ct) if tag == "plain" else m(k_t, ct, at, lt, zk, zk, zm) if tag == "cnp" else m(k_t, ct, at, lt)).squeeze(0).numpy() for tag, m in models]
                    y = np.median(np.stack(ys), 0)
                    kp.append(r.k_left + s * (r.k_right - r.k_left)); Tp.append(r.Ta_min + y * (r.Ta_max - r.Ta_min))
                kp, Tp = np.concatenate(kp), np.concatenate(Tp); o = np.argsort(kp); kp, Tp = kp[o], Tp[o]
                gt = raw[np.isclose(raw.E, E)].sort_values("k"); kt, Tt = gt.k.values, gt.Ta.values
                Ti = np.interp(kt, kp, Tp, left=np.nan, right=np.nan); v = ~np.isnan(Ti); ip, it = np.argmin(Tp), np.argmin(Tt)
                rows.append(dict(E=E, Ta_c_true=Tt[it], k_c_true=kt[it], Ta_c_pred=Tp[ip], k_c_pred=kp[ip], curve_MAPE=100 * np.mean(np.abs(Ti[v] / Tt[v] - 1)) if v.sum() > 5 else np.nan))
            df = pd.DataFrame(rows).sort_values("E"); df.to_csv(out / f"per_E_{sub}.csv", index=False)
            metrics[sub] = dict(family=fam, n_models=len(models), n_test_E=len(df), MAPE_Ta=mape(df.Ta_c_pred, df.Ta_c_true), MAE_k=mae(df.k_c_pred, df.k_c_true), curve_MAPE=float(np.nanmean(df.curve_MAPE)))
            print(f"{sub:16s} ({fam}, {len(models)} models): test MAPE Ta_c {metrics[sub]['MAPE_Ta']:.2f} %  MAE k_c {metrics[sub]['MAE_k']:.2f}  curve {metrics[sub]['curve_MAPE']:.2f} %", flush=True)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
