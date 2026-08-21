# Parameter Contract Matrix — Canonical vs Archive-2

Node C, overnight 2026-08-22. Genesis `642702e`.

Canonical source: `src/constants.py:7-22`, `src/constraints.py`,
`configs/parameter_bounds_PROVISIONAL.yaml:11-28`, `models/pinn_model.py`.
Archive-2 source: `src/dheston/calibration/transforms.py:9-94`,
`src/dheston/models/losses.py:44-45`, `src/dheston/calibration/optimize.py:31-32`.

## 1. Order and identity mapping

Canonical order (v0 LAST, factor named by speed):

| idx | canonical | Archive-2 idx | Archive-2 name | physical meaning |
|-----|-----------|---------------|----------------|------------------|
| 0 | kappa_slow | 6 | kappa2 | slow mean-reversion speed |
| 1 | theta_slow | 7 | theta2 | slow long-run variance |
| 2 | sigma_slow | 8 | sigma2 | slow vol-of-vol |
| 3 | rho_slow | 9 | rho2 | slow spot-variance correlation |
| 4 | v0_slow | 5 | v02 | slow initial (current) variance |
| 5 | kappa_fast | 1 | kappa1 | fast mean-reversion speed |
| 6 | theta_fast | 2 | theta1 | fast long-run variance |
| 7 | sigma_fast | 3 | sigma1 | fast vol-of-vol |
| 8 | rho_fast | 4 | rho1 | fast spot-variance correlation |
| 9 | v0_fast | 0 | v01 | fast initial (current) variance |

Archive-2 uses v0-FIRST within each group and numbers factors with **factor 1
= FAST** (its sampler draws `kappa2 <= kappa1`, `transforms.py:57`, and its
ordering penalty forbids `kappa2 > kappa1`, `losses.py:44-45`). The canonical
stack uses factor index 0-4 = SLOW and enforces `kappa_slow < kappa_fast`.

**Any positional tensor exchange between the stacks is a double transposition
(v0 position + slow/fast role). A silent swap produces a valid-looking but
semantically wrong vector. ADAPTER REQUIRED.**

Adapter (verified in `probe_residuals.py: canonical_to_archive2`):

```
archive2 = [v0_fast, kappa_fast, theta_fast, sigma_fast, rho_fast,
            v0_slow, kappa_slow, theta_slow, sigma_slow, rho_slow]
```

## 2. Per-parameter compatibility

| contract item | canonical | Archive-2 | compatibility |
|---|---|---|---|
| kappa ordering | hard: `kappa_slow < kappa_fast` (structural map + raise) | hard: `kappa2 <= kappa1` (sigmoid sandwich) + redundant soft penalty | DIRECTLY COMPATIBLE after index remap (note canonical is strict, Archive-2 non-strict at the boundary) |
| v0 semantics | initial = current variance state at valuation | same (COS pricer input) | DIRECTLY COMPATIBLE |
| kappa bounds | 0.05-3.0 (slow), 0.10-12.0 (fast) (YAML, sampling only) | kappa1 0.30-8.00, kappa2 0.10-6.00 (hard sigmoid box) | ADAPTER REQUIRED (Archive-2 box is narrower in places, e.g. no kappa > 8, kappa2 >= 0.10) |
| theta bounds | 0.005-0.25 / 0.002-0.20 | 0.01-0.60 both factors | ADAPTER REQUIRED (overlapping but neither contains the other) |
| sigma bounds | 0.005-1.0 / 0.005-1.5, hard-capped below Feller ceiling 0.995*sqrt(2 kappa theta) | 0.05-1.50, NO Feller coupling | ADAPTER REQUIRED + SEMANTIC CONFLICT (see 3) |
| rho range | (-1, 1) each + joint disk rho_s^2+rho_f^2 < 1 (hard, polar map) | (-0.95, -0.05) each, INDEPENDENT, no disk | SEMANTIC CONFLICT (see 3) |
| v0 bounds | 0.005-0.30 / 0.002-0.25 | 0.01-0.60 | ADAPTER REQUIRED |
| Feller 2*kappa*theta > sigma^2 | HARD per factor (raise at pricing; sigma hard-capped in NN map) | ABSENT | SEMANTIC CONFLICT |
| correlation disk | HARD (raise + polar map) | ABSENT | SEMANTIC CONFLICT |

## 3. Demonstrated semantic conflicts (reproducible)

`probe_results.json: archive2_constraint_gap`: an input inside Archive-2's OWN
box (`constrain_parameter_tensor` at sigmoid extremes) emits
`rho1 = rho2 = -0.95`, `sigma1 = sigma2 = 1.5`, `kappa = 0.3/0.1`,
`theta = 0.01`. Mapped to canonical order, `validate_parameters` returns:

- `slow-factor Feller gap must be strictly positive` (gap = -2.248)
- `fast-factor Feller gap must be strictly positive` (gap = -2.244)
- `rho_slow^2 + rho_fast^2 must be strictly less than 1` (disk = 1.805)

I.e. **Archive-2's hard output space contains vectors that the canonical model
contract rejects.** Conversely Archive-2's rho box (-0.95, -0.05) also EXCLUDES
canonical-valid vectors (rho in (-0.05, 0) or exactly 0, and any (rho_s, rho_f)
with rho_f > -0.05), so neither space contains the other. Conclusion: an
adapter can map names/units (Section 1) but CANNOT map the admissible sets;
Archive-2 outputs feeding the canonical pipeline require a constraint
projection step, and Archive-2 training data cannot represent the full
canonical parameter space.

## 4. Units and transformations

- All variances (theta, v0) are plain (annualised) variance units in both
  stacks; both stacks price with the same interpretation. No unit conflict.
- Transform families differ: canonical NN map = softplus(+eps), Feller-capped
  sigmoid for sigma, polar disk map for (rho_s, rho_f), additive gap for
  kappa ordering (`models/pinn_model.py:22-53,126-154`); Archive-2 = plain
  per-parameter sigmoid box, kappa2 sandwiched under kappa1
  (`transforms.py:78-94`). Both are differentiable; only the canonical map
  guarantees canonical validity BY CONSTRUCTION.
- The canonical z-score `TargetStandardizer` (`models/parameter_transform.py`)
  applies to the CANONICAL order; reusing its statistics for Archive-2-ordered
  vectors would silently permute standardisation. UNKNOWN risk today only
  because no code currently does this — flagged for Node A.

## 5. Verdict

- Names/identity/order: ADAPTER REQUIRED (double transposition).
- Constraints/admissible set: SEMANTIC CONFLICT (Feller absent, disk absent,
  rho sign restriction) — cannot be fixed by an adapter alone.
- Bounds: ADAPTER REQUIRED (overlapping boxes).
- No UNKNOWN items remain after tonight's experiments.
