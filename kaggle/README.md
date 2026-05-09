# Kaggle production training (500 epochs)

This folder contains the artifacts that ran the 500-epoch production training of all 6 model families on Kaggle's free P100 GPU. Total wall-clock time: ~5 hours, total GPU usage: ~10–12 hours, total cost: 0 €.

## Layout

```
kaggle/
├── README.md                    this file
├── datasets/
│   ├── tcml-aux/dataset-metadata.json     metadata for tcml-aux  (combined_data.csv + 3 rl_switch_windows .npz)
│   └── code/dataset-metadata.json    metadata for tcml-code (zipped repo source, auto-extracted by Kaggle)
└── kernels/
    ├── precision/                    SIREN + DeepONet + Chebyshev + Envelope-SIREN (4 models)
    ├── ssst/                         Sparse-MoE Transformer
    ├── sarl/                         Single-agent SAC refiner
    ├── marl/                         3-SAC + Cross-Attn refiner
    ├── neptune/                      FNO + Latent Diffusion ensemble (M=3)
    └── hydra/                        Diffusion-Policy MARL (3 phases)

each kernels/<model>/ contains run.ipynb (the notebook executed on Kaggle)
plus kernel-metadata.json (kernel ID, GPU/internet flags, dataset attachments).
```

## How to reproduce

### 1. Install the Kaggle CLI

```bash
pip install kaggle
```

Place your `kaggle.json` (from https://www.kaggle.com/settings → API → Create New Token) under `~/.kaggle/kaggle.json`.

### 2. Build the two datasets used by every kernel

```bash
# tcml-aux  — auxiliary input files that are gitignored or too large for the repo
mkdir -p _build_aux && cd _build_aux
cp ../path/to/combined_data.csv .
cp ../path/to/rl_switch_windows_*.npz .
cp ../kaggle/datasets/tcml-aux/dataset-metadata.json .
kaggle datasets create -p .
cd ..

# tcml-code — zipped repo source (so the kernel doesn't need internet to git clone)
mkdir -p _build_code && cd _build_code
zip -rq tcml-code.zip ../.. -x "*.git*" "*__pycache__*" "kaggle/*" "data/runs/*"
cp ../kaggle/datasets/code/dataset-metadata.json .
kaggle datasets create -p .
cd ..
```

After both are `ready`, the kernels below can be pushed.

### 3. Push and run each kernel

Each kernel is self-contained: it pip-installs torch 2.4.1+cu121 (compatible with P100 sm_60), copies the code from `/kaggle/input/tcml-code/`, copies the aux data from `/kaggle/input/tcml-aux/`, and runs the corresponding training script for 500 epochs with `--device cuda`.

```bash
# precision (4 models in one kernel, ~40 min)
cd kaggle/kernels/precision && kaggle kernels push -p . && cd -

# the heavy ones (push 2 at a time — Kaggle limits to 2 concurrent GPU sessions)
cd kaggle/kernels/ssst    && kaggle kernels push -p . && cd -
cd kaggle/kernels/sarl    && kaggle kernels push -p . && cd -
# wait for the 2 above to complete (kaggle kernels status <id>) then push:
cd kaggle/kernels/marl    && kaggle kernels push -p . && cd -
cd kaggle/kernels/neptune && kaggle kernels push -p . && cd -
# wait again then push:
cd kaggle/kernels/hydra   && kaggle kernels push -p . && cd -
```

Status / logs of any running kernel:

```bash
kaggle kernels status hayanichoujaamohamed/<kernel-slug>
kaggle kernels output hayanichoujaamohamed/<kernel-slug> -p ./out
```

### 4. Download the trained checkpoints

Once a kernel reports `COMPLETE`:

```bash
kaggle kernels output hayanichoujaamohamed/<kernel-slug> -p ./out
# best.pt + history.json land under ./out/runs/<model>/
```

### 5. Phone verification — required

Free Kaggle accounts must verify a phone number before GPU and Internet are allowed in kernels (https://www.kaggle.com/settings → Phone Verification). Without it, every kernel falls back to CPU and `git clone` / `pip install` are blocked.

## Why this layout (instead of one notebook per `MODEL`)

Kaggle imposes a 9 h cap per session and a 30 h GPU quota per week. Running 6 separate notebooks in parallel pairs (the cap is 2 concurrent GPU sessions) finishes the whole pipeline in ~5 h wall-clock. Splitting per model also makes it easy to re-run a single one without re-paying for the others.

## Footprint summary

| Kernel | epochs | wall-clock on P100 | output size |
|---|---|---|---|
| precision | 500 × 4 models | ~6 min | 25 MB |
| ssst | 500 | ~24 min | 80 MB |
| sarl | 500 | ~12 min | 21 MB |
| marl | 500 | ~14 min | 106 MB |
| neptune | 500 × 3 ensemble members | ~3.5 h | 333 MB (best member only after pruning) |
| hydra | MADT 500 + IQL 500 + MAPPO 5000 steps | ~14 min | 89 MB |

The `data/runs/` directory at the repo root contains the trained checkpoints (Git LFS).
