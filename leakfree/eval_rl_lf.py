"""Evaluate the leak-free RL refiner (SARL, 5 seeds) on the held-out elasticities.

1. Window level: MAE of the base (leak-free ensemble median) and of the
   corrected prediction on the 49-point test windows (normalised units).
2. Curve level: the seed-averaged correction is blended into the full median
   curve of each predicted branch (Tukey taper at the window edges), the
   physical curve is rebuilt with the leak-free anchors, and Ta_c, k_c are
   read at the minimum; compared with the base median and with the truth.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--rl_run", required=True, help="folder with seed_*/best.pt of SARL")
    ap.add_argument("--windows", required=True, help="rl_switch_windows_lf_test.npz"); ap.add_argument("--median", required=True)
    ap.add_argument("--desc", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--taper", type=float, default=0.20)
    ap.add_argument("--agg", choices=["mean", "weighted"], default="mean",
                    help="weighted: seeds weighted by 1/best validation MAE (history.json), the aggregation used in the paper")
    a = ap.parse_args(); repo, out = Path(a.repo), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "code")); sys.path.insert(0, str(repo))
    from data_pipeline.dataset import HydraWindowsDataset
    from models.sac_pro.feature_extractor import SARLProFeatureExtractor
    from models.sac_pro.sac_pro import SACPro
    device = torch.device("cpu")
    ds = HydraWindowsDataset(Path(a.windows))                      # own statistics, as in inference_sarl_v2.py
    obs = torch.from_numpy(ds.obs_seq); sv = torch.from_numpy(ds.static_vec)
    corrs, vmae = [], []
    for ck in sorted(Path(a.rl_run).glob("seed_*/best.pt")):
        h = json.load(open(ck.parent / "history.json")); vmae.append(min(r["val_mae"] for r in h if r.get("val_mae") is not None))
        st = torch.load(ck, map_location=device, weights_only=False); cfg = st["cfg"]
        ext = SARLProFeatureExtractor(obs.shape[2], obs.shape[1], st["static_dim"], cfg["feature_extractor"]).to(device)
        sac = SACPro(feature_dim=ext.out_dim, action_dim=st["action_dim"], actor_layers=cfg["actor_layers"], critic_layers=cfg["critic_layers"]).to(device)
        ext.load_state_dict(st["extractor"]); sac.load_state_dict(st["sac"]); ext.eval(); sac.eval()
        with torch.no_grad():
            corrs.append(torch.tanh(sac.actor.mean(sac.actor.body(ext(obs, sv)))).numpy())
    print(f"{len(corrs)} SARL seeds loaded, {len(ds)} test windows")
    if a.agg == "weighted":
        w_seed = 1.0 / (np.array(vmae) + 1e-6); w_seed /= w_seed.sum()
        corr = np.einsum("i...,i->...", np.stack(corrs), w_seed); print("seed weights:", np.round(w_seed, 4))
    else:
        corr = np.mean(np.stack(corrs), 0)
    yp, yt = ds.y_pred, ds.y_true; yc = yp + corr
    win = dict(n_windows=int(len(ds)), MAE_base=float(np.mean(np.abs(yp - yt))), MAE_corrected=float(np.mean(np.abs(yc - yt))),
               RMSE_base=float(np.sqrt(np.mean((yp - yt) ** 2))), RMSE_corrected=float(np.sqrt(np.mean((yc - yt) ** 2))))
    print("window level:", json.dumps(win))

    # ---- curve level ----
    desc = pd.read_csv(a.desc); med = np.load(a.median)
    mkey = {(round(float(e), 6), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
    s_full = np.linspace(0, 1, med["ta_norm_median"].shape[1])
    raw = pd.read_csv(repo / "data" / "Input" / "combined_data.csv"); raw.columns = ["Ta", "k", "E"]
    split = pd.read_csv(a.split); test_E = set(np.round(split.loc[split.split == "test", "E"].to_numpy(float), 6))
    wkey = {(round(float(e), 6), int(b)): i for i, (e, b) in enumerate(zip(ds.E, ds.branch_local_id))}
    rows = []
    for E, g in desc.groupby("E"):
        E = float(E)
        if round(E, 6) not in test_E: continue
        kp, Tb, Tc = [], [], []
        for _, r in g.iterrows():
            key = (round(E, 6), int(r.branch_local_id)); base = med["ta_norm_median"][mkey[key]].copy(); refined = base.copy()
            if key in wkey:
                i = wkey[key]; c0 = float(ds.center_pred[i]); h = float(ds.window_half_width[i])
                s_w = np.clip(c0 + h * ds.local_grid[i], 0, 1)
                corr_full = np.interp(s_full, s_w, corr[i], left=0.0, right=0.0)
                d = np.abs(s_full - c0) / max(h, 1e-9)                       # Tukey taper: 1 inside, cosine to 0 at the edge band
                w = np.where(d <= 1 - a.taper, 1.0, np.where(d >= 1, 0.0, 0.5 * (1 + np.cos(np.pi * (d - (1 - a.taper)) / a.taper))))
                refined = np.clip(base + w * corr_full, 0.0, 1.0)
            kp.append(r.k_left + s_full * (r.k_right - r.k_left)); amp = r.Ta_max - r.Ta_min
            Tb.append(r.Ta_min + base * amp); Tc.append(r.Ta_min + refined * amp)
        kp, Tb, Tc = np.concatenate(kp), np.concatenate(Tb), np.concatenate(Tc); o = np.argsort(kp); kp, Tb, Tc = kp[o], Tb[o], Tc[o]
        gt = raw[np.isclose(raw.E, E)].sort_values("k"); kt, Tt = gt.k.values, gt.Ta.values; it = np.argmin(Tt)
        cm = lambda T: (lambda Ti: 100 * np.mean(np.abs(Ti[~np.isnan(Ti)] / Tt[~np.isnan(Ti)] - 1)) if (~np.isnan(Ti)).sum() > 5 else np.nan)(np.interp(kt, kp, T, left=np.nan, right=np.nan))
        rows.append(dict(E=E, Ta_c_true=Tt[it], k_c_true=kt[it], Ta_c_base=Tb[np.argmin(Tb)], k_c_base=kp[np.argmin(Tb)],
                         Ta_c_rl=Tc[np.argmin(Tc)], k_c_rl=kp[np.argmin(Tc)], curve_MAPE_base=cm(Tb), curve_MAPE_rl=cm(Tc)))
    d = pd.DataFrame(rows).sort_values("E"); d.to_csv(out / "per_E_rl.csv", index=False)
    mape = lambda x, t: float(100 * np.mean(np.abs(x / t - 1))); mae = lambda x, t: float(np.mean(np.abs(x - t)))
    kj = np.where(np.abs(np.diff(pd.read_csv(a.split).sort_values("E").E.values)) >= 0)[0]  # placeholder, not used
    res = dict(agg=a.agg, window=win, n_test_E=int(len(d)),
               base_MAPE_Ta=mape(d.Ta_c_base, d.Ta_c_true), base_MAE_k=mae(d.k_c_base, d.k_c_true),
               rl_MAPE_Ta=mape(d.Ta_c_rl, d.Ta_c_true), rl_MAE_k=mae(d.k_c_rl, d.k_c_true),
               base_curve_MAPE=float(np.nanmean(d.curve_MAPE_base)), rl_curve_MAPE=float(np.nanmean(d.curve_MAPE_rl)))
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(f"curve level, {len(d)} test E:  base median  MAPE Ta_c {res['base_MAPE_Ta']:.2f} %  MAE k_c {res['base_MAE_k']:.2f}   |   "
          f"RL-refined  MAPE Ta_c {res['rl_MAPE_Ta']:.2f} %  MAE k_c {res['rl_MAE_k']:.2f}"
          f"   |   curve: base {res['base_curve_MAPE']:.2f} %  RL {res['rl_curve_MAPE']:.2f} %")


if __name__ == "__main__":
    main()
