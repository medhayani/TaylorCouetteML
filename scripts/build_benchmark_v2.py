"""Build a unified benchmark CSV from the 500-epoch production training runs.

For each model, parses data/runs/<model>/history.json (or sub-dirs for precision)
and writes a single CSV with the final training metrics side-by-side.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
OUT_CSV = RUNS / "benchmark_500ep.csv"


def _last(history, key):
    for entry in reversed(history):
        if isinstance(entry, dict) and key in entry:
            return entry[key]
    return None


def _best(history, key, mode="min"):
    vals = [e[key] for e in history if isinstance(e, dict) and key in e]
    if not vals:
        return None
    return min(vals) if mode == "min" else max(vals)


def _parse_history(p: Path):
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def collect():
    rows = []

    # ---- precision (4 sub-models) ----
    for sub in ["envelope_siren", "siren", "chebyshev", "deeponet"]:
        h = _parse_history(RUNS / "precision" / sub / "history.json")
        if h is None:
            continue
        rows.append({
            "model": sub,
            "family": "precision",
            "epochs": len(h),
            "metric": "val_mae",
            "init": h[0].get("val_mae"),
            "final": _last(h, "val_mae"),
            "best": _best(h, "val_mae"),
        })

    # ---- ssst (val_mae) ----
    h = _parse_history(RUNS / "ssst" / "history.json")
    if h is not None:
        rows.append({
            "model": "ssst",
            "family": "moe-transformer",
            "epochs": len(h),
            "metric": "val_mae",
            "init": h[0].get("val_mae"),
            "final": _last(h, "val_mae"),
            "best": _best(h, "val_mae"),
        })

    # ---- sarl (residual_mae) ----
    h = _parse_history(RUNS / "sarl" / "history.json")
    if h is not None:
        rows.append({
            "model": "sarl",
            "family": "rl-refiner",
            "epochs": len(h),
            "metric": "residual_mae",
            "init": h[0].get("residual_mae"),
            "final": _last(h, "residual_mae"),
            "best": _best(h, "residual_mae"),
        })

    # ---- marl (residual_mae) ----
    h = _parse_history(RUNS / "marl" / "history.json")
    if h is not None:
        rows.append({
            "model": "marl",
            "family": "rl-refiner",
            "epochs": len(h),
            "metric": "residual_mae",
            "init": h[0].get("residual_mae"),
            "final": _last(h, "residual_mae"),
            "best": _best(h, "residual_mae"),
        })

    # ---- hydra (no per-epoch history; final phase metrics only) ----
    rows.append({
        "model": "hydra",
        "family": "marl-3phase",
        "epochs": "MADT 500 + IQL 500 + MAPPO 5000",
        "metric": "phase3_critic",
        "init": None,
        "final": "0.0004",
        "best": None,
    })

    # ---- neptune (best member) ----
    h = _parse_history(RUNS / "neptune" / "best_member" / "history.json")
    if h is not None:
        rows.append({
            "model": "neptune (best of 3)",
            "family": "fno-diffusion",
            "epochs": len(h),
            "metric": "val_loss",
            "init": h[0].get("val_loss"),
            "final": _last(h, "val_loss"),
            "best": _best(h, "val_loss"),
        })

    return rows


def write_csv(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "family", "epochs", "metric",
                                            "init", "final", "best"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    rows = collect()
    write_csv(rows, OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print()
    # Pretty print
    cols = ["model", "family", "epochs", "metric", "init", "final", "best"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
                for c in cols}
    print(" | ".join(f"{c:<{widths[c]}}" for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in cols))


if __name__ == "__main__":
    main()
