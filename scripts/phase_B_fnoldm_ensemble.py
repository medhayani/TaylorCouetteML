"""Phase B — small FNO + Latent Diffusion ensemble (M=5, 200 epochs).

Replaces the failed 49M PRO version with the converged 2.9M architecture
(from code6's NEPTUNE), trained 200 epochs per member instead of 80.
Expected test MAE ~0.035 with calibrated epistemic uncertainty.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE6 = ROOT.parent / "code6"

t0 = time.time()
out_dir = ROOT.parent / "data" / "runs_strong" / "fnoldm_ensemble"
out_dir.mkdir(parents=True, exist_ok=True)
log_file = out_dir / "train.log"

cmd = [sys.executable, str(CODE6 / "scripts" / "train_neptune.py"),
       "--profile_csv", str(ROOT.parent / "code4"
                              / "04_StepC__Build_Modeling_Datasets" / "02_outputs"
                              / "04_StepC__stepC_modeling_datasets"
                              / "model_profile_level_dataset.csv"),
       "--out_dir", str(out_dir),
       "--epochs", "200",            # was 80
       "--batch", "8",
       "--ensemble", "5"]            # 5 independent members
print(f">>> Phase B — FNO+LDM ensemble M=5, 200 epochs/member", flush=True)
print(f"  out_dir = {out_dir}")
print(f"  log     = {log_file}")
with open(log_file, "w", encoding="utf-8") as f:
    rc = subprocess.call(cmd, cwd=str(CODE6), stdout=f, stderr=subprocess.STDOUT)
print(f"  exit = {rc}, elapsed = {(time.time()-t0)/60:.1f} min")
if rc == 0:
    (out_dir / ".done").write_text(f"OK at {time.ctime()}", encoding="utf-8")
sys.exit(rc)
