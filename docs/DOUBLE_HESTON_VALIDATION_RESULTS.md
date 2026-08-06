# Double Heston Validation Results

Status date: 06 August 2026

## Scope

These are controlled results for the independent canonical reimplementation. They do not reproduce the unavailable teammate engine and do not demonstrate real NIFTY performance.

## Automated tests

- Full suite: **54 passed**.
- Engine-focused file: **36 passed**.
- Covered invariants include `phi(0)=1`, finite outputs, call and put bounds, parity, strike monotonicity, non-negativity, shapes, determinism, invalid-input rejection, factor-swap symmetry before label ordering, declared-order rejection, 64/96-node convergence, a near-one-factor limit, scalar/surface agreement, state propagation, and fixture reproducibility.

## Pricing checks

For the 18-quote controlled fixture:

| Check | Result |
|---|---:|
| Fixture maximum absolute regression error | `1.4210854715202004e-14` |
| All discounted no-arbitrage bounds | Passed |
| Minimum lower-bound margin | `0.27132077106226404` |
| Minimum upper-bound margin | `71.2353390442214` |
| Maximum absolute put-call parity error | `1.4210854715202004e-14` |
| 64-vs-96-node RMSE | `3.668193360898033e-12` |
| 64-vs-96-node maximum absolute difference | `4.291678123991005e-12` |

No price was clipped or replaced to produce these results.

## Clean synthetic recovery

Three deterministic starts were run with an 80-function-evaluation limit. One start used a disclosed perturbation around the known synthetic parameter coordinate; it is not hidden as an uninformed start.

The best clean result came from the neutral transform midpoint and met SciPy’s gradient stopping condition:

| Metric | Best clean result |
|---|---:|
| Normalized loss | `1.1096408482162216e-27` |
| Price RMSE | `1.6125487393436679e-13` |
| Parameter RMSE | `2.2178553561322553e-11` |
| Maximum relative parameter error | `3.269520071347287e-11` |

The other two starts reached the evaluation limit. Their price RMSE values were `0.0007613018409270122` and `2.8888805455663838e-05`; one remained materially different in parameter space. Optimizer convergence is not treated as proof of unique recovery.

## One-percent price-noise recovery

The deterministic Gaussian realization had an RMS relative noise of `0.010574835822177494` (about 1.057%). All three starts reached the 80-evaluation limit without satisfying a SciPy stopping condition.

The candidate with the smallest normalized objective had:

| Metric | Selected noisy candidate |
|---|---:|
| Normalized loss | `7.935315640803738e-05` |
| Price RMSE versus noisy observations | `0.17631291031892615` |
| Parameter RMSE versus known parameters | `0.43113598206350984` |
| Maximum relative parameter error | `2.7263524509056496` |

Across starts, noisy-observation price RMSE ranged from `0.17380446031955443` to `0.17631291031892615`. The three candidates were parameter-unstable even though their repricing errors were similar.

## Failures, boundaries, and identifiability

- Six starts were recorded in full.
- One start met an optimizer stopping criterion; five stopped at the evaluation limit.
- No start crashed or produced a malformed/non-finite pricing result.
- Three noisy candidates were marked boundary-near.
- Two approached the slow-factor Feller boundary.
- The selected normalized-loss noisy candidate approached both the slow Feller boundary and the provisional lower hard bound for `rho_fast`.

These results illustrate practical non-identifiability: similar surface fit does not imply stable or unique recovery of all ten parameters. The clean exact recovery is a controlled self-consistency result, not proof that arbitrary starts or noisy surfaces identify the true vector.

## Pilot generation and ANN integration

- Generated **12** pilot surfaces and **1,296** quote rows.
- Every row is labeled `GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA`.
- The pilot used the real canonical adapter and provisional pilot-only ranges.
- The dummy generator remains available only through the explicit smoke-test path.
- Full ANN research training was not started.

## Saved evidence

`outputs/double_heston_validation/` contains:

- `validation_summary.json`
- `clean_recovery_starts.csv`
- `noise_1pct_recovery_starts.csv`
- `clean_surface.csv`
- `noisy_surface.csv`
- `parameter_comparison.csv`
- `pricing_convergence.csv`
- `failures.csv`
- `pilot_surfaces/dataset_metadata.json`
- `pilot_surfaces/surfaces.csv`

## What is and is not validated

Validated here: implementation invariants, deterministic controlled pricing, discounted bounds, parity, quadrature refinement, self-consistency recovery, one deterministic 1% noise experiment, repeated starts, adapter routing, smoke-path separation, and small genuine-engine pilot generation.

Not validated here: equivalence to unavailable source, externally confirmed original bounds, exhaustive extreme-parameter stability, uniqueness, broad multi-seed robustness, ANN research training, real-market calibration, chronological NIFTY generalization, or ANN/PINN/model superiority.
