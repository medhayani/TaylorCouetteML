"""Master script — runs all 5 PRO trainings in sequence.

Order: SARL_PRO -> MARL_PRO -> HYDRA_PRO -> SSST_PRO -> NEPTUNE_PRO
       (cheapest first so quick wins are recorded first)

Each training writes its own log + checkpoints under ../data/runs_pro/<name>/
plus a per-run stamp file <name>/.done so you can resume / skip.
"""
from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE4 = ROOT.parent / "code4"
PROFILE_CSV = (CODE4 / "04_StepC__Build_Modeling_Datasets" / "02_outputs"
                / "04_StepC__stepC_modeling_datasets" / "model_profile_level_dataset.csv")
WIN_TRAIN = (CODE4 / "07_StepF__Export_RL_Windows_Pro" / "02_outputs"
              / "switch_rl_dataset_pro_v3_norm_fixed" / "rl_switch_windows_train.npz")


def run(name: str, args: list, out_dir: Path) -> bool:
    stamp = out_dir / ".done"
    if stamp.exists():
        print(f"\n>>> SKIP {name} (already done)")
        return True
    print(f"\n>>> START {name}  ({time.strftime('%H:%M:%S')})", flush=True)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{name}_train.log"
    with open(log_file, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / args[0])] + args[1:],
            cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode == 0:
        stamp.write_text(f"done at {time.ctime()}, elapsed={elapsed/60:.1f} min\n",
                          encoding="utf-8")
        print(f"<<< OK    {name}  in {elapsed/60:.1f} min", flush=True)
        return True
    else:
        print(f"<<< FAIL  {name}  exit={proc.returncode}  see {log_file}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=Path("../data/runs_pro"), type=Path)
    ap.add_argument("--small", action="store_true",
                    help="run with reduced epochs/steps for a smoke test (~30 min)")
    args = ap.parse_args()
    data_root = args.data_root.resolve()
    print(f"runs root = {data_root}")
    print(f"profile_csv = {PROFILE_CSV}  exists={PROFILE_CSV.exists()}")
    print(f"windows_train = {WIN_TRAIN}  exists={WIN_TRAIN.exists()}")

    if args.small:
        # smoke-test settings (~30 min total)
        sarl_args =  ["--epochs", "3"]
        marl_args =  ["--epochs", "3"]
        hydra_args = ["--madt_epochs", "2", "--iql_epochs", "2", "--mappo_steps", "30"]
        ssst_args =  ["--epochs", "3"]
        neptune_args = ["--epochs", "3", "--ensemble", "1"]
    else:
        # production-ish (long but doable on CPU in ~20 h)
        sarl_args =  ["--epochs", "60"]
        marl_args =  ["--epochs", "60"]
        hydra_args = ["--madt_epochs", "20", "--iql_epochs", "30", "--mappo_steps", "600"]
        ssst_args =  ["--epochs", "100"]
        neptune_args = ["--epochs", "100", "--ensemble", "3"]

    overall_t0 = time.time()
    results = {}

    # ---- 1) SARL_PRO ----
    out = data_root / "sarl_pro"
    results["sarl_pro"] = run("sarl_pro", [
        "train_sarl_pro.py", "--windows_train", str(WIN_TRAIN),
        "--out_dir", str(out), *sarl_args], out)

    # ---- 2) MARL_PRO ----
    out = data_root / "marl_pro"
    results["marl_pro"] = run("marl_pro", [
        "train_marl_pro.py", "--windows_train", str(WIN_TRAIN),
        "--out_dir", str(out), *marl_args], out)

    # ---- 3) HYDRA_PRO ----
    out = data_root / "hydra_pro"
    results["hydra_pro"] = run("hydra_pro", [
        "train_hydra_pro.py", "--windows_train", str(WIN_TRAIN),
        "--out_dir", str(out), *hydra_args], out)

    # ---- 4) SSST_PRO ----
    out = data_root / "ssst_pro"
    results["ssst_pro"] = run("ssst_pro", [
        "train_ssst_pro.py", "--profile_csv", str(PROFILE_CSV),
        "--out_dir", str(out), *ssst_args], out)

    # ---- 5) NEPTUNE_PRO ----
    out = data_root / "neptune_pro"
    results["neptune_pro"] = run("neptune_pro", [
        "train_neptune_pro.py", "--profile_csv", str(PROFILE_CSV),
        "--out_dir", str(out), *neptune_args], out)

    print("\n" + "=" * 60)
    print(f"ALL DONE in {(time.time()-overall_t0)/3600:.1f} h")
    for k, v in results.items():
        print(f"  {k:<14} {'OK' if v else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
