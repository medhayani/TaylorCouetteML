"""Leak-free prediction of a marginal stability curve Ta(k) at a given elasticity.

Nothing of the marginal curve of the requested elasticity E is used.  The branch
structure, the 23 descriptors and the 4 anchors come from the two nearest
TRAINING elasticities (data/split_by_E.csv), interpolated in log10 E, exactly as
during training and evaluation (see leakfree_descriptors.py).  If E belongs to
the database, its own curve is excluded (leave-self-out) and is only used, on
request, to draw the reference.

Models (weights in models_trained/):
    cnp       conditional neural process, zero-shot (model 1 of the paper)
    cnp_nbr   the same, conditioned on the curve interpolated from the
              neighbours, given as 32 context points
    dist      distilled spectral transformer (model 2)
    sarl      median of the 29 teachers, refined by the RL agent at the cusps
              (model 3); available for the elasticities of the database, for
              which the median and the windows are shipped
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "leakfree"))

from leakfree_descriptors import interpolate_descriptors, interpolate_curves  # noqa: E402

FEATS = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry", "slope_left_local",
         "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse", "normalized_arc_length",
         "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch",
         "is_last_branch", "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]
MODELS = ("cnp", "cnp_nbr", "dist", "sarl")
S = np.linspace(0.0, 1.0, 101)
TAPER = 0.20
R6 = lambda e: round(float(e), 6)                                            # noqa: E731


# ----------------------------------------------------------------- database
class Database:
    """Floquet database and the branch structure of its elasticities."""

    def __init__(self, repo: Path = REPO):
        self.repo = repo
        raw = pd.read_csv(repo / "data" / "combined_data.csv"); raw.columns = ["Ta", "k", "E"]
        self.raw = raw
        self.curves = {R6(E): g.sort_values("k") for E, g in raw.groupby("E")}
        self.desc = pd.read_csv(repo / "data" / "branch_functional_descriptors.csv")
        split = pd.read_csv(repo / "data" / "split_by_E.csv")
        self.train_E = split.loc[split.split == "train", "E"].to_numpy(float)
        self.split = {R6(e): s for e, s in zip(split.E, split.split)}

    def branches(self, E: float, margin: float = 0.05) -> pd.DataFrame:
        """Leak-free branch structure, descriptors and anchors at E."""
        E = float(E)
        table = self.desc
        if not np.isclose(table.E.to_numpy(float), E, rtol=1e-9).any():       # elasticity outside the database
            nearest = float(table.E.to_numpy(float)[np.argmin(np.abs(np.log10(table.E.to_numpy(float)) - np.log10(E)))])
            placeholder = table[np.isclose(table.E, nearest)].copy(); placeholder["E"] = E
            table = pd.concat([table, placeholder], ignore_index=True)
        lf = interpolate_descriptors(table, self.train_E, margin=margin)
        rows = lf[np.isclose(lf.E.to_numpy(float), E, rtol=1e-9)].sort_values("branch_local_id").reset_index(drop=True)
        rows["log10E"] = np.log10(rows.E.clip(lower=1e-30))
        return rows

    def truth(self, E: float):
        """Floquet curve at E, or None when E is not in the database."""
        g = self.curves.get(R6(E))
        return (g.k.to_numpy(), g.Ta.to_numpy()) if g is not None else None


# ----------------------------------------------------------------- models
def _norm(x, mean, std):
    return np.clip(np.nan_to_num((x - mean) / (std + 1e-12)), -5, 5).astype(np.float32)


def _inputs(rows: pd.DataFrame, stats):
    ctx = np.array([[float(r[f]) if (f in rows.columns and pd.notna(r[f])) else 0.0 for f in FEATS]
                    for _, r in rows.iterrows()], np.float32)
    anc = np.stack([rows.k_left, rows.k_right, np.log10(rows.Ta_min.clip(lower=1e-6)),
                    np.log10(rows.Ta_max.clip(lower=1e-6))], 1).astype(np.float32)
    return _norm(ctx, stats["ctx_mean"], stats["ctx_std"]), _norm(anc, stats["anc_mean"], stats["anc_std"])


def _shapes_cnp(db: Database, rows: pd.DataFrame, model_dir: Path, conditioned: bool) -> np.ndarray:
    from models.cnp_tc.model import CNP_TC
    args = json.load(open(model_dir / "leakfree_diag.json"))["args"]
    hp = dict(ctx_dim=23, **{k: args[k] for k in ["n_cheb", "n_cos", "d_model", "n_enc_layers", "n_dec_layers",
                                                  "n_heads", "dropout", "n_E_freq", "n_k_freq", "max_ctx_points"]})
    stats = np.load(model_dir / "test_branches.npz")
    ctx, anc = _inputs(rows, stats)
    n, p = len(rows), hp["max_ctx_points"]
    ck = np.zeros((n, p), np.float32); cta = np.zeros((n, p), np.float32); cm = np.zeros((n, p), np.float32)
    if conditioned:
        curves = interpolate_curves(db.raw, rows, n=101)
        idx = np.unique(np.round(np.linspace(0, 100, int(args.get("n_ctx", 32)))).astype(int))
        for i, r in rows.iterrows():
            ck[i, :len(idx)] = (2 * S - 1)[idx]
            cta[i, :len(idx)] = curves[(round(float(r.E), 8), int(r.branch_local_id))][idx]
            cm[i, :len(idx)] = 1.0
    kq = torch.tensor(np.tile(2 * S - 1, (n, 1)), dtype=torch.float32)
    out = []
    for ck_pt in sorted(model_dir.glob("seed_*/best.pt")):
        m = CNP_TC(**hp); m.load_state_dict(torch.load(ck_pt, map_location="cpu")["state_dict"]); m.eval()
        with torch.no_grad():
            out.append(m(kq, torch.from_numpy(ctx), torch.from_numpy(anc),
                         torch.from_numpy(np.log10(rows.E.to_numpy(float)).astype(np.float32)),
                         torch.from_numpy(ck), torch.from_numpy(cta), torch.from_numpy(cm)).numpy())
    return np.mean(out, 0)                                                   # mean over the seeds


def _shapes_dist(rows: pd.DataFrame, model_dir: Path) -> np.ndarray:
    from models.star.model import STAR
    hp = json.load(open(model_dir / "hparams.json"))
    stats = np.load(model_dir / "test_branches.npz")
    ctx, anc = _inputs(rows, stats)
    lt = torch.tensor(np.log10(rows.E.to_numpy(float)), dtype=torch.float32)
    kq = torch.tensor(2 * S - 1, dtype=torch.float32).unsqueeze(0)
    out = []
    for ck_pt in sorted(model_dir.glob("seed_*/best.pt")):
        m = STAR(ctx_dim=23, **hp)
        m.load_state_dict(torch.load(ck_pt, map_location="cpu", weights_only=False)["state_dict"]); m.eval()
        with torch.no_grad():
            out.append(np.concatenate([m(kq, torch.from_numpy(ctx[i:i + 1]), torch.from_numpy(anc[i:i + 1]),
                                         lt[i:i + 1]).numpy() for i in range(len(rows))], 0))
    return np.median(np.stack(out), 0)                                       # median over the seeds


def _shapes_sarl(repo: Path, rows: pd.DataFrame, model_dir: Path) -> np.ndarray:
    """Median of the 29 teachers, refined by the RL agent inside its window."""
    from data_pipeline.dataset import HydraWindowsDataset
    from models.sac_pro.feature_extractor import SARLProFeatureExtractor
    from models.sac_pro.sac_pro import SACPro
    med = np.load(repo / "data" / "ensemble_median_targets_lf.npz")
    mkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
    missing = [int(r.branch_local_id) for _, r in rows.iterrows() if (R6(r.E), int(r.branch_local_id)) not in mkey]
    if missing:
        raise SystemExit("the RL refiner works on the elasticities of the database only "
                         "(the median of the teachers is shipped for those); use --model cnp_nbr or dist instead")
    base = np.stack([med["ta_norm_median"][mkey[(R6(r.E), int(r.branch_local_id))]] for _, r in rows.iterrows()])
    windows = {}
    for split in ("train", "val", "test"):
        f = repo / "data" / "rl_windows" / f"rl_switch_windows_lf_{split}.npz"
        if f.exists():
            z = HydraWindowsDataset(f)
            for i, (e, b) in enumerate(zip(z.E, z.branch_local_id)):
                windows[(R6(e), int(b))] = (z, i)
    agents = []
    for ck_pt in sorted(model_dir.glob("seed_*/best.pt")):
        st = torch.load(ck_pt, map_location="cpu", weights_only=False); cfg = st["cfg"]
        ext = SARLProFeatureExtractor(9, 49, st["static_dim"], cfg["feature_extractor"])
        sac = SACPro(feature_dim=ext.out_dim, action_dim=st["action_dim"],
                     actor_layers=cfg["actor_layers"], critic_layers=cfg["critic_layers"])
        ext.load_state_dict(st["extractor"]); sac.load_state_dict(st["sac"]); ext.eval(); sac.eval()
        v = json.load(open(ck_pt.parent / "history.json"))
        agents.append((ext, sac, min(r["val_mae"] for r in v if r.get("val_mae") is not None)))
    w_seed = np.array([1.0 / (v + 1e-6) for _, _, v in agents]); w_seed /= w_seed.sum()
    for i, (_, r) in enumerate(rows.iterrows()):
        key = (R6(r.E), int(r.branch_local_id))
        if key not in windows:
            continue
        ds, j = windows[key]
        obs = torch.from_numpy(ds.obs_seq[j:j + 1]); sv = torch.from_numpy(ds.static_vec[j:j + 1])
        with torch.no_grad():
            corr = sum(w * torch.tanh(sac.actor.mean(sac.actor.body(ext(obs, sv)))).numpy()[0]
                       for (ext, sac, _), w in zip(agents, w_seed))
        c0 = float(ds.center_pred[j]); h = float(ds.window_half_width[j])
        full = np.interp(S, np.clip(c0 + h * ds.local_grid[j], 0, 1), corr, left=0.0, right=0.0)
        d = np.abs(S - c0) / max(h, 1e-9)
        taper = np.where(d <= 1 - TAPER, 1.0, np.where(d >= 1, 0.0,
                         0.5 * (1 + np.cos(np.pi * (d - (1 - TAPER)) / TAPER))))
        base[i] = np.clip(base[i] + taper * full, 0.0, 1.0)
    return base


def shapes(db: Database, rows: pd.DataFrame, model: str) -> np.ndarray:
    """Normalised shape y(s) of every predicted branch, on 101 points."""
    d = db.repo / "models_trained" / model
    if not d.exists():
        raise SystemExit(f"weights not found: {d}")
    if model in ("cnp", "cnp_nbr", "cnp_block"):
        return _shapes_cnp(db, rows, d, conditioned=model == "cnp_nbr")
    if model == "dist":
        return _shapes_dist(rows, d)
    if model == "sarl":
        return _shapes_sarl(db.repo, rows, d)
    raise SystemExit(f"unknown model: {model} (choose from {', '.join(MODELS)})")


# ----------------------------------------------------------------- curve
def assemble(pieces, dk: float = 0.1):
    """Physical curve: branches put side by side, the lowest one kept where they overlap."""
    k_lo = min(k.min() for k, _ in pieces); k_hi = max(k.max() for k, _ in pieces)
    grid = np.round(np.arange(k_lo, k_hi + 1e-9, dk), 4)
    out = np.full(len(grid), np.nan)
    for k_arr, ta in pieces:
        m = (grid >= k_arr.min() - 1e-9) & (grid <= k_arr.max() + 1e-9)
        v = np.interp(grid[m], k_arr, ta)
        out[m] = np.where(np.isnan(out[m]), v, np.minimum(out[m], v))
    ok = ~np.isnan(out)
    if ok.sum() >= 2:
        out = np.interp(grid, grid[ok], out[ok])
    return grid, out


def predict(db: Database, E: float, model: str = "cnp_nbr", margin: float = 0.05):
    """Return (k, Ta) of the predicted marginal curve, and the branch table."""
    rows = db.branches(E, margin=margin)
    y = shapes(db, rows, model)
    pieces = [(r.k_left + S * (r.k_right - r.k_left), r.Ta_min + y[i] * (r.Ta_max - r.Ta_min))
              for i, (_, r) in enumerate(rows.iterrows())]
    k, ta = assemble(pieces)
    return k, ta, rows, pieces


def critical(k, ta):
    i = int(np.nanargmin(ta))
    return float(ta[i]), float(k[i])
