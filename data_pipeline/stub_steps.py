"""Stub modules for the data pipeline steps that still call into code4 scripts.

Steps A, B, AB, C, F are unchanged in spirit from code4 (they work, see audit).
We expose tiny wrapper functions that simply invoke the corresponding code4
script via subprocess. The actual numerics are in those scripts and don't need
to be rewritten — only NEPTUNE (replaces SSST) and HYDRA-MARL (replaces SARL/MARL)
are new.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run_python(script: Path, args: Optional[list] = None) -> int:
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)
    return subprocess.call(cmd)


def run_step_a(code4_root: Path) -> int:
    script = code4_root / "01_StepA__Segmented_Functional_Analysis" / "01_scripts" \
             / "01_StepA__stepA_segmented_functional_analysis__func_main.py"
    return _run_python(script)


def run_step_b(code4_root: Path) -> int:
    script = code4_root / "02_StepB__HSIC_Screening" / "01_scripts" \
             / "02_StepB__stepB_hsic_screening__func_main.py"
    return _run_python(script)


def run_step_ab(code4_root: Path) -> int:
    script = code4_root / "03_StepAB__Summary_A_B" / "01_scripts" \
             / "03_StepAB__summarize_stepA_stepB__func_main.py"
    return _run_python(script)


def run_step_c(code4_root: Path) -> int:
    script = code4_root / "04_StepC__Build_Modeling_Datasets" / "01_scripts" \
             / "04_StepC__stepC_build_modeling_datasets__func_main.py"
    return _run_python(script)
