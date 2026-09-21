# Double Heston regular PINN improvement decision

- Source ref: `origin/DhruvaBamb/double-heston`
- Source HEAD: `9c44422cf6afc46e718cb21ad561756445e49b8a`
- Local branch: `viraj/double-heston-improvement`
- Baseline checkpoint: `double_sobolev_s17`
- Baseline checkpoint SHA256: `a9fec59d4c58b5eca66f59eeefcb78fd4ac36b724d47f8c0b416e4e193304f56`

## Data provenance limitation

The original ignored training arrays were absent from all local checkouts. A
deterministic regeneration used the recorded seed and current canonical teacher,
but did not reproduce the historical bytes:

| File | Historical SHA256 | Regenerated SHA256 |
|---|---|---|
| `train.npz` | `f18c296f1d0183ac266448830a05e80e8e7cdd3789cca5c6a5434dd09c7fdc28` | `ddfa96d8393150763d1d523bddfed868d2c03dd87d82b9a4f7c1928cc296c9d9` |
| `collocation.npz` | `a3820957488f9dacbbcebeb2c3425b766473c7b65ffb76c9a8a763f193c422af` | `95c5a25ac500f95b5e139ea1c280fe687555687c98d358a8c16d257994e0d91b` |

The current Windows/PyTorch regeneration retained 129,451 of 131,072 training
rows, while the historical Apple run retained all rows. Results below are a
controlled development experiment, not an exact replay of pretraining.

## Development comparison

All recovery results use the existing four fixed development truths, seed
`906311`, three deterministic starts, seed `906777`, and the unchanged gate:
all positive-parameter relative errors at most 5% and both rho absolute errors
at most 0.05.

| Model | Scaled recovery RMSE | All-10 passes | Mean IV RMSE at truth | Mean price RMSE at truth |
|---|---:|---:|---:|---:|
| Baseline Sobolev | 0.124257 | 0/4 | 8.2051e-05 | 1.8379e-05 |
| Phase 1 float64 L-BFGS | 0.150639 | 0/4 | 9.5450e-05 | 2.2237e-05 |
| Phase 2 Jacobian pilot | 0.132324 | 0/4 | 8.6553e-05 | 1.6051e-05 |

On a common 1,024-point collocation sample, dimensionless PDE RMSE
(all / smile / wide) was:

- Baseline: `0.060993 / 0.005598 / 0.121601`
- Phase 1: `0.072373 / 0.006496 / 0.144308`
- Phase 2: `0.060728 / 0.005689 / 0.121056`

All three diagnostic runs were finite with no convexity or calendar violations.

## Decision

Phase 1 failed: its optimization loss and fixed-batch PDE term fell, but recovery,
price/IV accuracy, and fresh PDE diagnostics degraded.

The Phase 2 pilot reduced its training-surface normalized Jacobian mismatch from
`0.004684` to `0.003630` and weak-direction mismatch from `22.4138` to `17.2047`.
On the fixed development cases, however, mean Jacobian relative error improved
only from `0.002934` to `0.002869`, recovery stayed `0/4`, and the worst
linearized shift increased from `24.47` to `29.86` tolerance units.

Stop here. The evidence supports ill-conditioned weak inverse directions as the
dominant bottleneck, but neither float64 optimizer refinement nor this small
Jacobian pilot establishes a useful recovery improvement. Before any major
architecture rewrite, the next experiment should use the original byte-identical
training arrays (or a formally declared replacement cohort), more independent
training surfaces/seeds, and a predeclared weak-direction validation criterion.
