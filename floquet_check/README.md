# floquet_check -- wall-condition test of the Floquet base

Purpose: verify, independently of the training data, that the base
`data/combined_data.csv` was computed for cylinders oscillating **in
phase** (co-oscillating cell), and not in opposition.

## Files

| File | Content |
|---|---|
| `verify_epp.m` | Recomputes the Floquet multipliers at tabulated critical points `(E, k_c, Ta_c)` for **both** wall conditions, `epp = +1` (in phase) and `epp = -1` (in opposition); refines `Ta` to marginality by `fzero`; writes `verify_epp_N<N>_M<M>.csv`. |
| `diag_epp.m` | Diagnostic of the discrete operator against resolution `N`: spectrum of `Q(t)`, and comparison of the exponential stepping with `ode45`. |
| `verify_epp_N12_M50.csv`, `verify_epp_N16_M100.csv` | Results at `N = 12` and `N = 16` Chebyshev points, `M` steps per period. |

The linear operator is the one of the mode-classification scripts of
the paper (base flow `V1, V2`, stresses `T1..T8`, matrices `A, B`,
boundary conditions `u = u' = v = 0` eliminated through `G`),
`gamma = 5`, `S = 1` (UCM), `eps = d/R1 = 0.14`. The monodromy matrix
is integrated by exponential (Magnus midpoint) stepping,
`Phi <- expm(h Q(t + h/2)) Phi`, which agrees with `ode45`
(`RelTol = 1e-3`) to three digits.

## Run

```
matlab -batch "cd('floquet_check'); verify_epp(16, 100)"
```

Output columns: `E, k, Ta_tab, raw_sign, epp, absmu_at_Ta_tab, Ta_star,
dev_pct, absmu_star, arg_over_pi, seconds`. A marginal point must give
`absmu_at_Ta_tab = 1`; `dev_pct` is the deviation of the refined
threshold from the tabulated one.

## Result

| E | k_c | Ta_c (base) | in phase, `epp = +1` | in opposition, `epp = -1` |
|---|---|---|---|---|
| 0.0008 | 5.0 | 179.12 | N=12: -0.00 % (\|mu\| = 1.0001) ; N=16: -0.12 % | +7.2 % (\|mu\| = 0.39) ; +7.1 % |
| 0.0033 | 4.9 | 163.08 | -0.10 % ; -0.33 % | -1.9 % ; -2.1 % |

The tabulated thresholds are those of the **in-phase** operator. The
first point comes from a raw file stored with a negative sign, the
second from a positive one: the sign of the raw files is a bookkeeping
artefact (the combined table takes absolute values), not a physical
one.

## Limits of this check

* The operator, as written, develops spurious growing modes of the
  Chebyshev fourth-derivative matrix for `N >= 24` (real parts of
  order +100, independent of `epp`); it is clean for `N <= 16`.
* At `N <= 16` only the low-elasticity points (`k_c` about 5) are
  resolved. Points with `k_c >= 10`, and all points with `E >= 0.5`,
  are not: the base velocity `V1 ~ cos(gamma*beta*x)` with
  `gamma*beta ~ 5*sqrt(2*De)` oscillates 7 to 25 times across the gap
  for `E = 0.7` to `10` and needs a far higher resolution.
* Extending the check to the resonance band and to the elastic branch
  therefore requires a discretisation whose basis satisfies the wall
  conditions (e.g. `cheb4c` of Weideman and Reddy) at `N` of order
  100; this is left to do.
