# Multi-Mode Envelope SIREN ★

Architecture maison conçue pour le problème de stabilité Floquet, où la
courbe $T_a(k, E)$ est l'enveloppe inférieure de plusieurs modes propres :

$$T_a(k, E) = \min_{m=1\ldots M} \, T_m(k, E)$$

Cette opération min produit naturellement des cusps là où l'argmin change
— exactement comme un croisement de modes propres dans le problème Floquet.

```
MultiModeEnvelopeSIREN(ctx_dim=24, n_modes=4, hidden=192, depth=5)
  ctx_enc:   Linear(24,128)→GELU→Linear→GELU→Linear     (128 latent)
  modes:     4 × [SineLayer + 5 × (FiLM+SineLayer) + Linear]
  offsets:   Parameter(4)                                  (rupture de symétrie)
  forward:   stack 4 modes, hard-min, return env
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE | Gain vs legacy |
|---|---|---|---|---|
| 1.77 M | 0.011 | **0.549** | **0.785** | ×2.4 |

Meilleur compromis MAE global / kink-zone observé sur les 720 branches.

**Entraîner** : `python scripts/train_precision_curves.py` (loss kink-weighted ×5).
**Charger** :
```python
from models.envelope_siren.model import MultiModeEnvelopeSIREN
m = MultiModeEnvelopeSIREN(ctx_dim=24, n_modes=4)
ck = torch.load("checkpoints/envelope_siren/best.pt", weights_only=False)
m.load_state_dict(ck["state_dict"]); m.eval()
```
