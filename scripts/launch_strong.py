"""Master script that launches Phase A, then B, then C in sequence.

Each phase writes its own .done stamp; the master script skips any phase
already marked .done so you can interrupt and resume.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASES = [
    ("Phase A — long PT-SSST",        "phase_A_long_ptssst.py",   "ptssst_long"),
    ("Phase B — FNO+LDM ensemble",    "phase_B_fnoldm_ensemble.py","fnoldm_ensemble"),
    ("Phase C — refiner cycle",       "phase_C_refiner_cycle.py", "refiners"),
]
runs = HERE.parent.parent / "data" / "runs_strong"

t_total = time.time()
for label, script, key in PHASES:
    stamp = runs / key / ".done"
    if stamp.exists():
        print(f"  SKIP {label} (already .done)")
        continue
    print(f"\n>>> START {label}  ({time.strftime('%H:%M:%S')})", flush=True)
    t = time.time()
    rc = subprocess.call([sys.executable, str(HERE / script)])
    print(f"<<< END   {label}  exit={rc}  elapsed={(time.time()-t)/60:.1f} min")
    if rc != 0:
        print(f"  !! Phase failed, stopping here.")
        sys.exit(rc)

print(f"\nALL STRONG RETRAINING DONE in {(time.time()-t_total)/3600:.1f} h")
print("Now run:  python scripts/build_strong_figures.py")
