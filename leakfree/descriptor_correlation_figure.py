"""Spearman correlation between the informative descriptors, drawn for the paper.

Ordered by hierarchical clustering on 1 - |rho|, lower triangle only, the pairs
above 0.9 in absolute value boxed.  No title inside the figure: it belongs to
the caption.  Writes figures/descriptor_correlation.png and the list of the
redundant pairs as JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

# the seventeen that vary over the base, with the labels used in the paper
LABELS = {
    "log10E": r"$\log_{10}E$",
    "width_k": r"width $w$",
    "amplitude": r"amplitude $A$",
    "left_width": r"left width",
    "right_width": r"right width",
    "left_rise": r"left rise",
    "right_rise": r"right rise",
    "width_asymmetry": r"width asymmetry",
    "rise_asymmetry": r"rise asymmetry",
    "slope_left_local": r"local slope, left",
    "slope_right_local": r"local slope, right",
    "global_slope": r"global slope",
    "mean_abs_slope": r"mean $|\mathrm{d}\mathrm{Ta}/\mathrm{d}k|$",
    "mean_abs_curvature": r"mean $|\mathrm{d}^2\mathrm{Ta}/\mathrm{d}k^2|$",
    "curvature_at_min": r"curvature at minimum",
    "roughness_rmse": r"roughness",
    "normalized_arc_length": r"arc length $/\,w$",
}

R6 = lambda e: round(float(e), 6)                                            # noqa: E731


def main() -> None:
    desc = pd.read_csv(REPO / "data/branch_functional_descriptors.csv")
    split = pd.read_csv(REPO / "data/split_by_E.csv")
    lab = {R6(e): s for e, s in zip(split.E, split.split)}
    d = desc[desc.E.map(lambda e: lab.get(R6(e)) == "train")].copy()
    d["log10E"] = np.log10(d.E.clip(lower=1e-30))

    cols = [c for c in LABELS if c in d.columns]
    corr = d[cols].corr(method="spearman")

    # order: hierarchical clustering on 1 - |rho|, so the redundant blocks show
    dist = 1.0 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    cols = [cols[i] for i in order]
    corr = corr.loc[cols, cols]
    n = len(cols)

    pairs = [(a, b, float(corr.loc[a, b]))
             for i, a in enumerate(cols) for b in cols[i + 1:]
             if abs(corr.loc[a, b]) >= 0.9]
    pairs.sort(key=lambda t: -abs(t[2]))

    m = corr.to_numpy().copy()
    m[np.triu_indices(n, k=1)] = np.nan                   # lower triangle only

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[c] for c in cols], rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels([LABELS[c] for c in cols], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)

    # the redundant pairs, boxed
    idx = {c: i for i, c in enumerate(cols)}
    for a, b, r in pairs:
        i, j = idx[a], idx[b]
        lo, hi = min(i, j), max(i, j)
        ax.add_patch(Rectangle((lo - 0.5, hi - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.8))

    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    cb.set_label("Spearman correlation", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "descriptor_correlation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "descriptor_correlation.pdf", bbox_inches="tight")

    (OUT / "descriptor_correlation_pairs.json").write_text(
        json.dumps([{"a": a, "b": b, "rho": round(r, 3)} for a, b, r in pairs], indent=1),
        encoding="utf-8")

    print(f"{n} descriptors, {len(pairs)} pairs with |rho| >= 0.9")
    for a, b, r in pairs:
        print(f"  {r:+.3f}  {a} / {b}")


if __name__ == "__main__":
    sys.exit(main())
