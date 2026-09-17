# DeepONet

Lu, Jin, Pang, Zhang, Karniadakis. *Learning nonlinear operators via DeepONet*.
Nature Machine Intelligence 3, 218 (2021).

The branch network encodes the 23 descriptors, the trunk network encodes the
coordinate k through Fourier features; the prediction is their inner product.

```
DeepONet(ctx_dim=23, branch_layers=(384, 384, 384, 384),
         trunk_layers=(384, 384, 384, 384), latent_dim=192, fourier_bands=24)
```

**Trained by** `train/train_precision_ensemble.py` (the four precision
architectures together, 3 seeds each, kink-weighted MSE), through the
notebook `kaggle_notebooks/precision_lf/`.

**Measured** on the 64 held-out elasticities, seed median, leak-free inputs
(`results/teachers/metrics.json`):

| Ta_c | k_c | marginal curve |
|---|---|---|
| 2.39 % | 0.95 | 1.90 % |

**Load**: `from models.deeponet.model import DeepONet`; the weights of the
29 teachers are not shipped, the notebook retrains them.
