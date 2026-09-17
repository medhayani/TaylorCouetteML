# Chebyshev Spectral Regressor

Decomposes each branch on the truncated Chebyshev basis of the first kind on
[-1, 1]; a Transformer encodes the 23 descriptors and predicts the
coefficients.

```
ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32,
                           d_model=256, num_layers=4, num_heads=8)
  elem_proj: Linear(1, 256)             # one token per descriptor
  encoder:   TransformerEncoder x 4     # CLS + 23 tokens
  head:      LayerNorm -> Linear -> GELU -> Linear -> 33 coefficients
```

Note: a truncated polynomial basis rings near a cusp (Gibbs); this is the
weakest of the four families on this problem, which the numbers below show.

**Trained by** `train/train_precision_ensemble.py` (the four precision
architectures together, 3 seeds each, kink-weighted MSE), through the
notebook `kaggle_notebooks/precision_lf/`.

**Measured** on the 64 held-out elasticities, seed median, leak-free inputs
(`results/teachers/metrics.json`):

| Ta_c | k_c | marginal curve |
|---|---|---|
| 3.03 % | 0.83 | 2.58 % |

**Load**: `from models.chebyshev_spectral.model import ChebyshevSpectralRegressor`; the weights of the
29 teachers are not shipped, the notebook retrains them.
