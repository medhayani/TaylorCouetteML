# Multi-Mode Envelope SIREN

Architecture built for this problem: the marginal curve is the lower envelope
of several mode-specific threshold curves,

    Ta(k, E) = min over m of T_m(k, E),

and the hard minimum produces a cusp wherever the winning mode changes, which
is exactly how a mode exchange appears in the Floquet problem.

```
MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7)
  ctx_enc:   Linear(23, 128) -> GELU -> Linear -> GELU -> Linear   (128 latent)
  modes:     8 x [SineLayer + 7 x (FiLM + SineLayer) + Linear]
  offsets:   Parameter(8)                                          (symmetry breaking)
  forward:   stack the modes, hard minimum
```

It is the best of the 29 teachers taken alone on the held-out elasticities.

**Trained by** `train/train_precision_ensemble.py` (the four precision
architectures together, 3 seeds each, kink-weighted MSE), through the
notebook `kaggle_notebooks/precision_lf/`.

**Measured** on the 64 held-out elasticities, seed median, leak-free inputs
(`results/teachers/metrics.json`):

| Ta_c | k_c | marginal curve |
|---|---|---|
| 1.08 % | 0.67 | 1.84 % |

**Load**: `from models.envelope_siren.model import MultiModeEnvelopeSIREN`; the weights of the
29 teachers are not shipped, the notebook retrains them.
