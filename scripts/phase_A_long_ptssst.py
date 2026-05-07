"""Phase A — long Sparse-MoE-T (PyTorch) training.

Goal: improve val MAE from 0.033 -> ~0.020 by training 300 epochs
with cosine LR + warm restart, identical 21M architecture.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE7 = ROOT.parent / "code7"

t0 = time.time()
out_dir = ROOT.parent / "data" / "runs_strong" / "ptssst_long"
out_dir.mkdir(parents=True, exist_ok=True)
log_file = out_dir / "train.log"

cmd = [sys.executable, str(CODE7 / "scripts" / "train_ssst_pro.py"),
       "--profile_csv", str(ROOT.parent / "code4"
                              / "04_StepC__Build_Modeling_Datasets" / "02_outputs"
                              / "04_StepC__stepC_modeling_datasets"
                              / "model_profile_level_dataset.csv"),
       "--out_dir", str(out_dir),
       "--epochs", "300",            # was 100
       "--batch", "8",
       "--lr", "1.2e-4"]
print(f">>> Phase A — Sparse-MoE-T (PyTorch), 300 epochs", flush=True)
print(f"  out_dir = {out_dir}")
print(f"  log     = {log_file}")
with open(log_file, "w", encoding="utf-8") as f:
    rc = subprocess.call(cmd, cwd=str(CODE7), stdout=f, stderr=subprocess.STDOUT)
print(f"  exit = {rc}, elapsed = {(time.time()-t0)/60:.1f} min")
if rc == 0:
    (out_dir / ".done").write_text(f"OK at {time.ctime()}", encoding="utf-8")
sys.exit(rc)
