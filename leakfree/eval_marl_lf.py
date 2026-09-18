"""Evaluate the leak-free MARL refiner on the held-out elasticities.

Same protocol as eval_rl_lf.py for SARL: the seed corrections are averaged with
weights 1/(best validation MAE), blended into the base prediction (the median of
the 29 leak-free teachers) through a Tukey window, and the rebuilt curve is
compared with Floquet.  MARL differs only in the way the correction is produced:
three agents return eight scalars, from which a fixed analytic form generates the
49 values of the window.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]            # repository root
OUT = REPO / "results/marl"; OUT.mkdir(parents=True, exist_ok=True)
RUN = REPO / "models_trained/marl"
sys.path.insert(0, str(REPO / "code")); sys.path.insert(0, str(REPO))
from data_pipeline.dataset import HydraWindowsDataset                      # noqa: E402
from models.marl_3sac.marl_model import MARLProSystem                      # noqa: E402

S = np.linspace(0, 1, 101); TAPER = 0.20
R6 = lambda e: round(float(e), 6)                                          # noqa: E731
cfg = yaml.safe_load((REPO / "code/configs/sizes.yaml").read_text(encoding="utf-8"))["marl_pro"]
cfg["agent_features"].setdefault("seq_out", cfg["agent_features"]["seq_hidden"])   # as in the trainer


def reconstruct(actions, T):
    """The correction of the window, from the eight scalars of the three agents."""
    a_loc, a_sh, a_geo = actions
    s = torch.linspace(0, 1, T).unsqueeze(0).expand(a_loc.size(0), T)
    dc = a_loc[:, 0:1] * 0.5 + 0.5
    da = a_loc[:, 1:2] * 0.5
    bump = da * torch.exp(-((s - dc) ** 2) / (2 * 0.05 ** 2))
    sh = a_sh[:, 0:1] + a_sh[:, 1:2] * bump
    sl, sr, w, asy = a_geo[:, 0:1], a_geo[:, 1:2], a_geo[:, 2:3], a_geo[:, 3:4]
    geo = (sl * (s - dc).clamp(max=0) + sr * (s - dc).clamp(min=0)
           + w * (s - dc).abs() + asy * (s - 0.5))
    return sh + 0.1 * geo


ds = HydraWindowsDataset(REPO / "data/rl_windows/rl_switch_windows_lf_test.npz")
obs, sv = torch.from_numpy(ds.obs_seq), torch.from_numpy(ds.static_vec)
corrs, vmae = [], []
for ck in sorted(RUN.glob("seed_*/best.pt")):
    m = MARLProSystem(obs_seq_dim=obs.shape[2], obs_seq_T=obs.shape[1], static_dim=sv.shape[1], cfg=cfg)
    m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"]); m.eval()
    with torch.no_grad():
        h = m.encode(obs, sv)
        acts = [torch.tanh(ag.actor.mean(ag.actor.body(h[:, i, :]))) for i, ag in enumerate(m.agents)]
        corrs.append(reconstruct(acts, obs.shape[1]).numpy())
    hist = json.loads((ck.parent / "history.json").read_text())
    vmae.append(min(r["val_mae"] for r in hist if r.get("val_mae") is not None))
w_seed = 1.0 / (np.array(vmae) + 1e-6); w_seed /= w_seed.sum()
corr = np.einsum("i...,i->...", np.stack(corrs), w_seed)
print(f"{len(corrs)} seeds MARL, poids {np.round(w_seed, 3)}, meilleures val {np.round(vmae, 5)}")

yp, yt = ds.y_pred, ds.y_true
win = dict(n_windows=int(len(ds)), MAE_base=float(np.mean(np.abs(yp - yt))),
           MAE_corrected=float(np.mean(np.abs(yp + corr - yt))),
           RMSE_base=float(np.sqrt(np.mean((yp - yt) ** 2))),
           RMSE_corrected=float(np.sqrt(np.mean((yp + corr - yt) ** 2))))
print("fenetres:", json.dumps(win))

desc = pd.read_csv(REPO / "data/branch_functional_descriptors_leakfree_all.csv")
med = np.load(REPO / "data/ensemble_median_targets_lf.npz")
mkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
wkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(ds.E, ds.branch_local_id))}
raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
split = pd.read_csv(REPO / "data/split_by_E.csv"); test_E = set(np.round(split.loc[split.split == "test", "E"].to_numpy(float), 6))
rows = []
for E, g in desc.groupby("E"):
    E = float(E)
    if R6(E) not in test_E:
        continue
    kp, Tb, Tc = [], [], []
    for _, r in g.iterrows():
        key = (R6(E), int(r.branch_local_id)); base = med["ta_norm_median"][mkey[key]].copy(); ref = base.copy()
        if key in wkey:
            i = wkey[key]; c0 = float(ds.center_pred[i]); hw = float(ds.window_half_width[i])
            cf = np.interp(S, np.clip(c0 + hw * ds.local_grid[i], 0, 1), corr[i], left=0.0, right=0.0)
            d = np.abs(S - c0) / max(hw, 1e-9)
            wt = np.where(d <= 1 - TAPER, 1.0, np.where(d >= 1, 0.0, 0.5 * (1 + np.cos(np.pi * (d - (1 - TAPER)) / TAPER))))
            ref = np.clip(base + wt * cf, 0.0, 1.0)
        k_b = r.k_left + S * (r.k_right - r.k_left); amp = r.Ta_max - r.Ta_min
        kp.append(k_b); Tb.append(r.Ta_min + base * amp); Tc.append(r.Ta_min + ref * amp)
    kp, Tb, Tc = np.concatenate(kp), np.concatenate(Tb), np.concatenate(Tc)
    o = np.argsort(kp); kp, Tb, Tc = kp[o], Tb[o], Tc[o]
    gt = raw[np.isclose(raw.E, E)].sort_values("k"); kt, Tt = gt.k.values, gt.Ta.values; it = int(np.argmin(Tt))
    cm = lambda T: (lambda Ti: 100 * np.mean(np.abs(Ti[~np.isnan(Ti)] / Tt[~np.isnan(Ti)] - 1)))(
        np.interp(kt, kp, T, left=np.nan, right=np.nan))
    rows.append(dict(E=E, Ta_c_true=Tt[it], k_c_true=kt[it],
                     Ta_c_base=Tb[np.argmin(Tb)], k_c_base=kp[np.argmin(Tb)], curve_base=cm(Tb),
                     Ta_c_marl=Tc[np.argmin(Tc)], k_c_marl=kp[np.argmin(Tc)], curve_marl=cm(Tc)))
d = pd.DataFrame(rows).sort_values("E"); d.to_csv(OUT / "per_E_marl.csv", index=False)
mape = lambda x, t: float(100 * np.mean(np.abs(x / t - 1)))
res = dict(seeds=len(corrs), best_val_mae=list(map(float, vmae)), window=win, n_test_E=int(len(d)),
           base=dict(MAPE_Ta=mape(d.Ta_c_base, d.Ta_c_true), MAE_k=float((d.k_c_base - d.k_c_true).abs().mean()),
                     curve_MAPE=float(d.curve_base.mean())),
           marl=dict(MAPE_Ta=mape(d.Ta_c_marl, d.Ta_c_true), MAE_k=float((d.k_c_marl - d.k_c_true).abs().mean()),
                     curve_MAPE=float(d.curve_marl.mean())))
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
print(f"courbes, {len(d)} elasticites de test :")
print("  base mediane   Ta_c %.2f %%  k_c %.2f  courbe %.2f %%" % (res["base"]["MAPE_Ta"], res["base"]["MAE_k"], res["base"]["curve_MAPE"]))
print("  mediane + MARL Ta_c %.2f %%  k_c %.2f  courbe %.2f %%" % (res["marl"]["MAPE_Ta"], res["marl"]["MAE_k"], res["marl"]["curve_MAPE"]))
