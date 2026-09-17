"""Evaluate the leak-free CNP on the held-out elasticities and draw the figures.

Inputs
  --run      folder downloaded from the Kaggle kernel (runs/cnp_leakfree):
             descriptors_leakfree.csv, split_by_E.csv, test_branches.npz,
             seed_*/best.pt
  --repo     TaylorCouetteML repository root (models/, data/)
Outputs (in --out)
  metrics.json, per_E.csv, fig_critical_curves.png, fig_marginal_curves.png,
  fig_error_by_regime.png

Zero-shot inference: no context point; inputs = interpolated descriptors and
anchors (never the target curve); five seeds averaged.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.interpolate import PchipInterpolator

STATIC_FEATURES = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry",
    "slope_left_local", "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse",
    "normalized_arc_length", "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch", "left_width", "right_width", "left_rise",
    "right_rise", "mean_abs_slope"]
HP = dict(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6, n_dec_layers=3,
          n_heads=8, dropout=0.05, n_E_freq=32, n_k_freq=16, max_ctx_points=32)


def mape(a, b): return float(100 * np.mean(np.abs(np.asarray(a) / np.asarray(b) - 1)))
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--critical_csv", required=True, help="data/critical_values_submitted_version.csv: truth and critical values of the models of the first submission, for comparison")
    ap.add_argument("--out", required=True); ap.add_argument("--n_pred", type=int, default=101)
    a = ap.parse_args()
    run, repo, out = Path(a.run), Path(a.repo), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "code")); sys.path.insert(0, str(repo))
    from models.cnp_tc.model import CNP_TC
    # hyperparameters of the run, as saved by train_cnp_leakfree.py
    try:
        targs = json.load(open(run / "leakfree_diag.json"))["args"]
        for k_src, k_dst in [("n_cheb", "n_cheb"), ("n_cos", "n_cos"), ("d_model", "d_model"), ("n_enc_layers", "n_enc_layers"),
                             ("n_dec_layers", "n_dec_layers"), ("n_heads", "n_heads"), ("dropout", "dropout"),
                             ("n_E_freq", "n_E_freq"), ("n_k_freq", "n_k_freq"), ("max_ctx_points", "max_ctx_points")]:
            if k_src in targs: HP[k_dst] = targs[k_src]
        print("HP from run:", HP)
    except Exception as e:
        print("leakfree_diag.json not read, default HP used:", e)

    desc = pd.read_csv(run / "descriptors_leakfree.csv"); desc["log10E"] = np.log10(desc.E.clip(lower=1e-30))
    split = pd.read_csv(run / "split_by_E.csv"); lab = {round(float(r.E), 8): r.split for r in split.itertuples()}
    stats = np.load(run / "test_branches.npz")
    ctx_mean, ctx_std, anc_mean, anc_std = stats["ctx_mean"], stats["ctx_std"], stats["anc_mean"], stats["anc_std"]
    raw = pd.read_csv(repo / "data" / "Input" / "combined_data.csv"); raw.columns = ["Ta", "k", "E"]
    ce = pd.read_csv(a.critical_csv).set_index("E")

    models = []
    for sd in sorted(run.glob("seed_*/best.pt")):
        m = CNP_TC(**HP); m.load_state_dict(torch.load(sd, map_location="cpu")["state_dict"]); m.eval(); models.append(m)
    print(f"{len(models)} seeds loaded")
    s_grid = np.linspace(0, 1, a.n_pred); k_norm = torch.tensor(2 * s_grid - 1, dtype=torch.float32).unsqueeze(0)
    zk = torch.zeros(1, HP["max_ctx_points"]); zm = torch.zeros(1, HP["max_ctx_points"])
    # neighbour-conditioned run: context = curve interpolated from the training neighbours
    ctxc = None
    if (run / "context_curves.npz").exists():
        z = np.load(run / "context_curves.npz")
        ctxc = {(round(float(e), 8), int(b)): t for e, b, t in zip(z["E"], z["branch_local_id"], z["ta"])}
        n_ctx = int(targs.get("n_ctx", 32)) if "targs" in dir() else 32
        idx = np.unique(np.round(np.linspace(0, ctxc[next(iter(ctxc))].shape[0] - 1, n_ctx)).astype(int))
        print(f"conditioned on the training neighbours: {len(ctxc)} context curves, {len(idx)} points each")

    rows, curves = [], {}
    for E, g in desc.groupby("E"):
        E = float(E); kp, Tp = [], []
        for _, r in g.sort_values("branch_local_id").iterrows():
            ctx = np.array([float(r.get(f, 0.0)) if pd.notna(r.get(f, np.nan)) else 0.0 for f in STATIC_FEATURES], np.float32)
            ctx = np.clip(np.nan_to_num((ctx - ctx_mean) / ctx_std), -5, 5).astype(np.float32)
            anc = np.array([r.k_left, r.k_right, np.log10(max(r.Ta_min, 1e-6)), np.log10(max(r.Ta_max, 1e-6))], np.float32)
            anc = np.clip(np.nan_to_num((anc - anc_mean) / anc_std), -5, 5).astype(np.float32)
            ck, cta, cm = zk, zk, zm
            if ctxc is not None:
                c = ctxc[(round(E, 8), int(r.branch_local_id))]
                ck = torch.zeros(1, HP["max_ctx_points"]); cta = torch.zeros(1, HP["max_ctx_points"]); cm = torch.zeros(1, HP["max_ctx_points"])
                ck[0, :len(idx)] = k_norm[0, idx]; cta[0, :len(idx)] = torch.from_numpy(c[idx]); cm[0, :len(idx)] = 1.0
            with torch.no_grad():
                ys = [m(k_norm, torch.from_numpy(ctx).unsqueeze(0), torch.from_numpy(anc).unsqueeze(0),
                        torch.tensor([np.log10(E)], dtype=torch.float32), ck, cta, cm).squeeze(0).numpy() for m in models]
            y = np.mean(ys, axis=0)
            kp.append(r.k_left + s_grid * (r.k_right - r.k_left)); Tp.append(r.Ta_min + y * (r.Ta_max - r.Ta_min))
        kp, Tp = np.concatenate(kp), np.concatenate(Tp); o = np.argsort(kp); kp, Tp = kp[o], Tp[o]
        gt = raw[np.isclose(raw.E, E)].sort_values("k"); kt, Tt = gt.k.values, gt.Ta.values
        Ti = np.interp(kt, kp, Tp, left=np.nan, right=np.nan); v = ~np.isnan(Ti)
        ip = np.argmin(Tp); it = np.argmin(Tt)
        Ek = round(E, 8); cE = ce.index[np.argmin(np.abs(ce.index.values - E))]
        rows.append(dict(E=E, split=lab.get(Ek, "?"), Ta_c_true=Tt[it], k_c_true=kt[it], Ta_c_pred=Tp[ip], k_c_pred=kp[ip],
                         curve_MAPE=100 * np.mean(np.abs(Ti[v] / Tt[v] - 1)) if v.sum() > 5 else np.nan,
                         Ta_c_old=ce.loc[cE, "Ta_c_cnp"], k_c_old=ce.loc[cE, "k_c_cnp"]))
        curves[E] = (kp, Tp, kt, Tt)
    df = pd.DataFrame(rows).sort_values("E"); df.to_csv(out / "per_E.csv", index=False)

    # PCHIP baseline on Ta_c(E), k_c(E), fitted on the training elasticities
    tr = df[df.split == "train"]; x_tr = np.log10(tr.E.values)
    pT = PchipInterpolator(x_tr, np.log10(tr.Ta_c_true.values)); pK = PchipInterpolator(x_tr, tr.k_c_true.values)
    df["Ta_c_pchip"] = 10 ** pT(np.log10(df.E.values)); df["k_c_pchip"] = pK(np.log10(df.E.values))
    kj = np.where(np.abs(np.diff(df.k_c_true.values)) > 1.0)[0]
    dist = np.array([np.min(np.abs(i - kj - 0.5)) if len(kj) else 99 for i in range(len(df))]); df["near_exchange"] = dist <= 2.5
    met = {}
    for name, msk in [("test", df.split == "test"), ("val", df.split == "val"), ("train", df.split == "train"),
                      ("test_near_exchange", (df.split == "test") & df.near_exchange), ("test_far", (df.split == "test") & ~df.near_exchange)]:
        d = df[msk]
        met[name] = dict(n=int(len(d)),
                         leakfree_MAPE_Ta=mape(d.Ta_c_pred, d.Ta_c_true), leakfree_MAE_k=mae(d.k_c_pred, d.k_c_true),
                         leakfree_curve_MAPE=float(np.nanmean(d.curve_MAPE)),
                         old_CNP_MAPE_Ta=mape(d.Ta_c_old, d.Ta_c_true), old_CNP_MAE_k=mae(d.k_c_old, d.k_c_true),
                         PCHIP_MAPE_Ta=mape(d.Ta_c_pchip, d.Ta_c_true), PCHIP_MAE_k=mae(d.k_c_pchip, d.k_c_true))
    json.dump(met, open(out / "metrics.json", "w"), indent=1); print(json.dumps(met, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    te = df[df.split == "test"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax[0].semilogx(df.E, df.Ta_c_true, "k-", lw=1.2, label="Floquet (base, 420 E)")
    ax[0].semilogx(te.E, te.Ta_c_old, "o", mfc="none", mec="tab:blue", ms=6, label="CNP submitted (target anchors)")
    ax[0].semilogx(te.E, te.Ta_c_pred, "s", color="tab:red", ms=5, label="CNP leak-free (interpolated inputs)")
    ax[0].semilogx(te.E, te.Ta_c_pchip, "x", color="tab:green", ms=6, label="PCHIP of training values")
    ax[0].set_ylabel(r"$Ta_c$"); ax[0].legend(fontsize=8); ax[0].set_title("Held-out elasticities (64): critical Taylor number and wavenumber")
    ax[1].semilogx(df.E, df.k_c_true, "k-", lw=1.2); ax[1].semilogx(te.E, te.k_c_old, "o", mfc="none", mec="tab:blue", ms=6)
    ax[1].semilogx(te.E, te.k_c_pred, "s", color="tab:red", ms=5); ax[1].semilogx(te.E, te.k_c_pchip, "x", color="tab:green", ms=6)
    ax[1].set_ylabel(r"$k_c$"); ax[1].set_xlabel("E"); fig.tight_layout(); fig.savefig(out / "fig_critical_curves.png", dpi=150)

    picks = []
    for target in [0.003, 0.03, 0.13, 0.5, 2.0, 8.0]:
        cand = te.iloc[np.argmin(np.abs(np.log10(te.E.values) - np.log10(target)))]; picks.append(float(cand.E))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7)); axs = axs.ravel()
    for axx, E in zip(axs, picks):
        kp, Tp, kt, Tt = curves[E]
        axx.plot(kt, Tt, "k.", ms=3, label="Floquet"); axx.plot(kp, Tp, "-", color="tab:red", lw=1.5, label="CNP leak-free")
        axx.set_title(f"E = {E:.4g}  (test)"); axx.set_xlabel("k"); axx.set_ylabel("Ta"); axx.set_ylim(0, np.nanmax(Tt) * 1.1)
    axs[0].legend(fontsize=8); fig.suptitle("Marginal curves at held-out elasticities: zero-shot leak-free CNP"); fig.tight_layout()
    fig.savefig(out / "fig_marginal_curves.png", dpi=150)

    bins = [(1e-4, 7e-3, "low"), (7e-3, 0.6, "resonance"), (0.6, 10.1, "high")]
    labels, L, O, P = [], [], [], []
    for lo, hi, nm in bins:
        d = te[(te.E >= lo) & (te.E < hi)]
        if len(d) == 0: continue
        labels.append(f"{nm}\n(n={len(d)})"); L.append(mape(d.Ta_c_pred, d.Ta_c_true)); O.append(mape(d.Ta_c_old, d.Ta_c_true)); P.append(mape(d.Ta_c_pchip, d.Ta_c_true))
    x = np.arange(len(labels)); w = 0.26
    fig, axx = plt.subplots(figsize=(7, 4))
    axx.bar(x - w, O, w, color="tab:blue", label="CNP submitted"); axx.bar(x, L, w, color="tab:red", label="CNP leak-free"); axx.bar(x + w, P, w, color="tab:green", label="PCHIP")
    axx.set_xticks(x); axx.set_xticklabels(labels); axx.set_ylabel(r"MAPE on $Ta_c$ (%)"); axx.set_title("Error on the held-out elasticities, by regime"); axx.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_error_by_regime.png", dpi=150)
    print("figures written to", out)


if __name__ == "__main__":
    main()
