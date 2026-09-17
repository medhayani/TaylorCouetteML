# SIREN + FiLM

Sitzmann, Martel, Bergman, Lindell, Wetzstein. *Implicit Neural Representations
with Periodic Activation Functions*. NeurIPS 2020.

MLP with sinusoidal activations (omega_0 = 30) and FiLM conditioning on the
23 branch descriptors (log10 E included).

```
SIRENRegressor(ctx_dim=23, hidden=384, depth=8)      # configuration trained here
  ctx_enc:   Linear(23, 128) -> GELU -> Linear -> GELU -> Linear   (128-dim latent)
  first:     SineLayer(1, 384, omega_0 = 30)
  body:      8 x [FiLM(128, 384) -> SineLayer(384, 384, omega_0 = 30)]
  out:       Linear(384, 1)
```

**Trained by** `train/train_precision_ensemble.py` (the four precision
architectures together, 3 seeds each, kink-weighted MSE), through the
notebook `kaggle_notebooks/precision_lf/`.

**Measured** on the 64 held-out elasticities, seed median, leak-free inputs
(`results/teachers/metrics.json`):

| Ta_c | k_c | marginal curve |
|---|---|---|
| 2.26 % | 0.78 | 2.10 % |

**Load**: `from models.siren.model import SIRENRegressor`; the weights of the
29 teachers are not shipped, the notebook retrains them.
