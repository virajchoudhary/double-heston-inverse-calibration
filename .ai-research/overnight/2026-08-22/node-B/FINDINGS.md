# Node B Findings

## Phase D — Jacobian conditioning across geometries (representative 4 cases + 16 interior samples)

Conventions identical to committed G2 diagnostics (spot-normalized prices, range-scaled parameters,
central differences, practical-rank tolerance 1e-6 relative). Median over the 4 representative cases:

| Geometry | Quotes | Condition number | sigma_min | Practical rank |
| --- | ---: | ---: | ---: | ---: |
| full108 (provisional grid) | 108 | 2.55e4 | 2.0e-5 | 10/10 |
| full108 calls only (= puts only) | 54 | 2.55e4 | 1.4e-5 | 10/10 |
| wings4x6 (±0.20,±0.30; 6 maturities) | 48 | 2.74e4 | 8.7e-6 | 10/10 |
| central5x6 (5 central nodes; 6 maturities) | 60 | 6.42e4 | 6.0e-6 | 10/10 |
| long3 (60/90/180d) | 54 | 1.27e5 | 1.2e-5 | 10/10 |
| short3 (7/14/30d) | 54 | 1.70e6 | 3.2e-7 | 9.5 |
| central5 market 27/55 (G2 anchor) | 20 | 6.54e8 | 1.0e-8 | 7.5 |
| single maturity (7/30/90/180d) | 18 | 1.4e11–5.0e11 | ~5e-13 | ~5 |

Key conclusions:
1. The G2 committed ill-conditioning (central-5, two maturities) REPLICATES (6.5e8 vs committed 5.1e7 median on their reduced-grid panel; both catastrophically noise-dominated, rank 7.5).
2. The FULL provisional 108 grid is locally full practical rank with condition ~2.6e4 — the provisional grid is NOT locally rank-deficient. Maturity span drives conditioning (2 -> 6 maturities improves ~4.5 orders); moneyness width adds little beyond ~4 nodes.
3. Calls and puts are informationally redundant under the carry contract: calls-only, puts-only, and calls+puts share the same condition number to 5 digits (puts are exact parity transforms of calls).
4. Noise-to-sensitivity: sigma_min ~ 1.4e-5 in normalized units sits BELOW realistic noise (0.5% of a 0.05-normalized ATM price = 2.5e-4): naive linear error propagation displaces the estimate by O(10) full-range widths along the weakest direction even on the full grid. Local "full practical rank" therefore coexists with noise-driven parameter collapse.
5. Exact factor-swap symmetry verified: swapping the slow and fast parameter blocks reproduces the 108-quote surface bitwise (max diff 0.0). This exact permutation degeneracy is excluded from the declared space only by kappa_slow < kappa_fast.

Implication: on the provisional 108 grid the failure is NOT classical local rank deficiency; it is (a) noise scale versus weakest sensitivities and (b) global structure (multi-modality / near-equivalent distant solutions), tested next in Phases B/C/F.

## Phase A — canonical baseline reproduced (production entry point, 108 grid, 3 canonical starts x 4 cases, clean + 1%)

- Clean: best starts recover truth exactly (price RMSE 4.4e-14, parameter RMSE 4.3e-12); worst starts disperse to parameter RMSE 0.39 with price RMSE 1.3e-4. Start sensitivity CONFIRMED on full grid. 8% boundary-near.
- 1% noise: price RMSE 0.123-0.138 (at the noise floor), parameter RMSE 0.17-1.33, 92% boundary-near. Committed RESULTS_TO_DATE pattern REPRODUCED; no discrepancy found.
- Runtime 742 s for 24 starts (production pricer).

## Phases B + C — multi-start dispersion and noise robustness (12 starts x 4 cases x 4 noise levels, G2 conventions, fast validated pricer)

| Noise | median price RMSE (norm.) | median param RMSE (full-range) | boundary-hit fraction | optimizer success |
| --- | ---: | ---: | ---: | ---: |
| 0% | 1.0e-6 | 0.097 | 35% | 42% |
| 0.5% | 7.2e-4 | 0.309 | 100% | 52% |
| 1% | 1.4e-3 | 0.306 | 98% | 54% |
| 2% | 2.6e-3 | 0.287 | 81% | 60% |

- Parameter error is a STEP function of noise (0.097 -> ~0.31 at 0.5%, then flat): the collapse happens entirely between 0 and 0.5% noise; further noise adds little.
- Repricing error tracks the noise floor at every level (repricing stays "good" while parameters collapse) and noisy solutions still reprice the TRUE surface within 2-5e-4.
- Boundary saturation: 100% of 0.5%-noise solutions hit at least one declared boundary/Feller/ordering constraint.

## Linearized noise propagation (Jacobian bridge)

Expected parameter RMSE from trace(J^+ Sigma J^+T): 3.1-28.9 full-range widths at 0.5% noise (9.4-115 at 2%); observed ~0.3 (truncated by the constraint box). The local Jacobian ALONE predicts the collapse scale; boundaries merely cap it. Case ordering matches (best-conditioned case_4 has the smallest prediction and the best observed recovery).

## Phase E — objective landscape / compensated profiles (case_1, worst-displacement case)

- Fixing kappa_slow at up to 7x its true value (1.5 widths above truth), the remaining 9 parameters re-optimize to objective ~1e-12 (machine-level repricing) with free-parameter RMSE <= 0.19.
- Fixing theta_fast at 5-8x truth keeps repricing below the 0.5% noise floor (objective ~1e-6).
- The flat compensated directions EXIST IN THE LANDSCAPE; they are not optimizer artifacts. Truth remains a genuine minimizer (exact recovery observed for several starts/cases).

## Phase F — near-equivalence tolerance spectrum on the full 108 grid (clean)

| price RMSE tolerance (normalized) | solutions | max parameter RMSE (full-range) |
| ---: | ---: | ---: |
| 2.5e-7 | 18/48 | 0.011 |
| 1e-6 | 23/48 | 0.159 |
| 3e-6 | 33/48 | 0.235 |
| 1e-5 | 41/48 | 0.323 |
| 1e-4 | 45/48 | 0.380 |
| 1e-3 | 45/48 | 0.380 (saturation) |

- Interpretation: the full 108 grid DOES identify better than the committed central-5 geometry at strict precision (at 2.5e-7, distant near-equivalents are gone: max 0.011 vs committed central-5 median 0.1485), BUT the equivalence radius grows monotonically with tolerance and reaches ~38% of the parameter box at a price tolerance (1e-4 normalized = 0.01 currency units on spot 100) that is still ~2.5x BELOW the smallest realistic noise floor.
- Strongest preserved example (case_1, deterministic_broad_10, price RMSE 6.7e-7, param RMSE 0.159): kappa_slow +141%, theta_slow +64%, v0_slow +64%, theta_fast -88% (to its floor), v0_fast -69%, rho_fast sign flip. Slow-factor inflation vs fast-factor collapse — the committed v0_slow/v0_fast, theta_slow/theta_fast compensation pattern reappears on the full grid.
- Heterogeneity: case_4 remains identified to tol 1e-5 (its truth has high vol-of-vol sigma_slow=0.76, sigma_fast=0.91); cases 1-3 disperse earlier. Non-identifiability severity is truth-dependent.
