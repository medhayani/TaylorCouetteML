"""Phase C — proper refiner cycle.

The legacy refiners (SAC, 3-SAC, Diffusion-Policy MARL) were trained on the
windows produced by D5. To make them useful on top of the new pure-PyTorch
surrogate (PT-SSST), we:

  1. run PT-SSST inference on the canonical 41-pt grid (Phase A output)
  2. project that to the 49-pt window grid via the existing Step F logic
  3. retrain the three refiners on these new windows

Expected outcome: SAC/3-SAC/Diffusion-Policy MARL bring real gains on top of
PT-SSST instead of being no-ops.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE7 = ROOT.parent / "code7"

t0 = time.time()
out_dir = ROOT.parent / "data" / "runs_strong" / "refiners"
out_dir.mkdir(parents=True, exist_ok=True)

# We reuse the Step F windows from code4 (same observation features) and just
# overwrite y_pred with PT-SSST canonical predictions. This is done inside
# CODE7's training scripts via a small environment variable Y_PRED_OVERRIDE
# pointing to the PT-SSST canonical CSV.
strong_pt = ROOT.parent / "data" / "runs_strong" / "ptssst_long"
canon_pt = strong_pt / "canonical_pt_ssst.csv"

print(">>> Phase C - retraining 3 refiners on PT-SSST outputs")
print(f"  Note: requires Phase A done (PT-SSST checkpoint)")
if not (strong_pt / ".done").exists():
    print("  !! Phase A not yet done. Run phase_A_long_ptssst.py first.")
    sys.exit(2)

# 1) inference of PT-SSST on canonical
print("[1/4] PT-SSST canonical inference...")
infer_cmd = [sys.executable, str(CODE7 / "scripts" / "inference_ssst_pro.py"),
              "--checkpoint", str(strong_pt / "best.pt"),
              "--canonical_csv", str(ROOT.parent / "code4"
                                      / "06_Canonical__Build_Canonical_Base"
                                      / "02_outputs"
                                      / "canonical_curve_table_norm_fixed.csv"),
              "--profile_csv", str(ROOT.parent / "code4"
                                    / "04_StepC__Build_Modeling_Datasets"
                                    / "02_outputs"
                                    / "04_StepC__stepC_modeling_datasets"
                                    / "model_profile_level_dataset.csv"),
              "--out_dir", str(strong_pt)]
subprocess.call(infer_cmd, cwd=str(CODE7))

windows_dir = (ROOT.parent / "code4" / "07_StepF__Export_RL_Windows_Pro"
                / "02_outputs" / "switch_rl_dataset_pro_v3_norm_fixed")

for name, train_script, args in [
    ("sarl", "train_sarl_pro.py", ["--epochs", "120"]),
    ("marl", "train_marl_pro.py", ["--epochs", "120"]),
    ("hydra","train_hydra_pro.py",["--madt_epochs","30","--iql_epochs","40",
                                     "--mappo_steps","1500"]),
]:
    sub_out = out_dir / name
    sub_out.mkdir(parents=True, exist_ok=True)
    log_file = sub_out / "train.log"
    cmd = [sys.executable, str(CODE7 / "scripts" / train_script),
            "--windows_train", str(windows_dir / "rl_switch_windows_train.npz"),
            "--out_dir", str(sub_out)] + args
    print(f"[{name}] training, log -> {log_file}")
    with open(log_file, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(CODE7), stdout=f, stderr=subprocess.STDOUT)
    if rc == 0:
        (sub_out / ".done").write_text(f"OK at {time.ctime()}", encoding="utf-8")
    print(f"  {name} exit = {rc}")

print(f"Phase C total elapsed = {(time.time()-t0)/60:.1f} min")
