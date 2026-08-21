# NTPC Double Heston Stability Reparameterization

## Predeclared research question and decision rules

Does a structure-aware, one-to-one transformed coordinate system for the **same canonical ten-parameter Double Heston space** reduce NTPC multi-start calibration instability while preserving calibration and holdout pricing quality? This tests optimization geometry. It does not establish structural identification unless globally separated solutions disappear over equivalent attainable space.

Base commit: `dd539150898bf5ca4d168c5dba3f3a33c69628e2`. Starts: `12` paired canonical starts. Pricer: unchanged canonical Double Heston at `64` production nodes. Optimizer: unchanged SciPy `least_squares(method="trf")`, `max_nfev=160`, tolerances and `diff_step`. Primary loss: unchanged unweighted calibration price residual vector.

Near-equivalent and material displacement rules are unchanged from the reviewed pilot. Complete-linkage clusters use the predeclared full-range-scaled distance `0.05`. Strong dispersion reduction requires at least 25% reductions in both median and maximum pairwise separation plus fewer clusters; partial requires at least 10% reductions with no extra clusters. These rules were fixed before calibration.

## What changed and what remained identical

Only optimizer coordinates changed. The canonical scientific target/order, pricing model, NTPC observations and frozen row roles, price field, activity screen, valuation date, spot, maturity basis, RBI/futures carry contract, IV inversion, 64-node production pricer, primary loss, start population, optimizer settings, and thresholds remained identical. Protected baseline hashes were verified before and after the run: `{'selected_options.csv': '981F4B04B816338BDC9EB6729182EA2DACFADA21578C03338A605BC7C124123D', 'carry_contract.csv': '1C81C80EDD12D07C0E27478813879816B30376F0BA717BCB2BB5A0118CB03477', 'model_comparison.csv': '6BB98FE854D820EF4BE0DAD66B286510069DFDE449DF26EA21AF0EEE7785CAFF', 'double_heston_multistart.csv': '4E092F2BEC5F53033E61EFB1D2B2D761C9D3AB8F72F17F33D6E989946FC1EB70', 'parameter_stability.json': '064B15EE0BED6565109CD466FF3F64D1BC556BCD147DFB8526E191513C627B20', 'docs/evidence/NTPC_SINGLE_STOCK_PILOT_MANIFEST.json': 'A19800FBEDBF00F8226A3F413A17D861420E82064D94384C9BA521FD2F1B9ADC'}`.

The exact transformed order is:

1. `z_v0_total`: bounded logistic over `(v0_slow.lower + v0_fast.lower, v0_slow.upper + v0_fast.upper)`.
2. `z_alpha_v`: bounded logistic selecting `v0_slow` from the total-conditional interval; `v0_fast=v0_total-v0_slow`, and reported `alpha_v=v0_slow/v0_total`.
3. `z_theta_total`: analogous total long-run variance.
4. `z_alpha_theta`: analogous conditional allocation, reported as `theta_slow/theta_total`.
5. `z_kappa_slow`: bounded logistic over the unchanged slow-kappa range.
6. `z_delta_kappa`: bounded logistic for `kappa_fast` over `(max(kappa_fast.lower,kappa_slow+1e-5), kappa_fast.upper)`.
7-8. `z_sigma_slow`, `z_sigma_fast`: bounded logistics over the unchanged lower bound and `min(configured_upper, sqrt(2*kappa*theta)*(1-1e-7))`, preserving Feller validity without clipping.
9. `z_rho_slow`: bounded logistic over the unchanged configured interval `(-0.95,0.95)`.
10. `z_rho_fast`: bounded logistic over `max(-0.95,-sqrt(1-rho_slow^2))` to `min(0.95,+sqrt(1-rho_slow^2))`, a one-to-one representation of the full unchanged hard-bound/unit-disk intersection.

The final output is always the original canonical ten-vector. Alpha values are optimization coordinates/derived diagnostics, **not new Double Heston scientific parameters**.

## Search-space equivalence

`EXPERIMENT_VALIDITY = PASSED_EQUIVALENT_SEARCH_SPACE`.

Exact bijection on the same numerical interior: rectangle interiors for v0/theta are expressed by total plus conditional allocation; kappa uses the identical conditional ordering interval; sigma uses the identical Feller-safe conditional interval; correlations use a one-to-one conditional parameterization of the full intersection between the individual hard bounds and the unit disk. The audit covered `2029` vectors: the existing best fit, `2000` deterministic random interiors, `20` near-boundary cases, and `8` explicit valid correlation-annulus cases with radius at least 0.95. Failures: `0`; empirical lost fraction: `0`; maximum absolute round-trip error: `3.55e-15`. No silent clipping is used.

## Pricing and runtime

| metric | baseline | reparameterized |
|---|---:|---:|
| calibration_price_rmse | 0.234320742 | 0.233174148 |
| calibration_iv_rmse | 0.00778971248 | 0.00780908549 |
| holdout_price_rmse | 0.92682472 | 0.921582641 |
| holdout_iv_rmse | 0.072591562 | 0.0718592663 |
| runtime_seconds | 228.606755 | 197.079995 |

Pricing preservation: **True**. The reparameterized holdout price RMSE does not beat the Standard Heston reference `0.910569` by at least 5% (required `<= 0.86504055`).

## Multi-start stability

| metric | baseline | reparameterized |
|---|---:|---:|
| valid starts | 12 | 12 |
| near-equivalent starts | 12 | 11 |
| materially displaced | 11 | 7 |
| complete-linkage clusters | 7 | 6 |
| median pairwise distance | 0.357338794 | 0.388626346 |
| maximum pairwise distance | 0.564149107 | 0.593889521 |
| maximum distance from best | 0.491085492 | 0.567424778 |
| boundary-hit rate | 1 | 1 |
| optimizer-cap rate | 0.833 | 0.833 |

Median pairwise dispersion: **8.756% increase**. Maximum pairwise dispersion: **5.272% increase**. Parameters retaining at least 0.05 of their configured full-range width across near-equivalent solutions: `['kappa_slow', 'theta_slow', 'sigma_slow', 'rho_slow', 'v0_slow', 'kappa_fast', 'theta_fast', 'sigma_fast', 'rho_fast', 'v0_fast']`. Full parameter-wise ranges and coefficients of variation are in `stability_comparison.json`.

## Allocation and mean-reversion diagnostics

Best transformed solution: `v0_total=0.0406351084`, `alpha_v=0.326188176`, `theta_total=0.158604213`, `alpha_theta=0.845058571`, `kappa_slow=2.79839276`, `kappa_fast=11.6326149`, slow half-life `90.4085819` days, fast half-life `21.7490843` days. Cross-start total/allocation and half-life values are recorded in `reparameterized_near_equivalent.csv` and Figures 4-6.

| diagnostic | baseline range | reparameterized range | baseline CV | reparameterized CV |
|---|---:|---:|---:|---:|
| v0_total | 0.00084325563 | 0.00080434754 | 0.00565965143 | 0.00900289483 |
| alpha_v | 0.824720781 | 0.824571809 | 0.9964427 | 0.761476887 |
| theta_total | 0.0711120479 | 0.239085619 | 0.219016198 | 0.437996345 |
| alpha_theta | 0.936486481 | 0.936487847 | 1.21469564 | 0.73027149 |
| delta_kappa | 9.11796896 | 9.51810429 | 0.613714822 | 0.598856192 |
| slow_half_life_days | 4845.34497 | 178.046032 | 2.01897558 | 0.553935833 |
| fast_half_life_days | 63.2571334 | 240.72416 | 0.397155405 | 1.28616921 |

Neither the variance totals nor slow/fast allocations may be called stable unless their ranges and coefficients of variation materially contract. The table shows the direct baseline-versus-reparameterized comparison; alpha values remain coordinate diagnostics only.

## Interpretation and decision

Classification: **INSUFFICIENT**. Invalid reasons: `[]`.

Total-plus-allocation coordinates were tested because short-maturity prices may constrain aggregate variance more directly than the factor split. The experiment changes the coordinate chart, not the attainable scientific model. Any remaining separated canonical solutions are therefore still consistent with structural non-identification; optimizer-coordinate improvement alone is not identification evidence.

This result does **not** alter the canonical ten-parameter project target. It does not by itself justify proceeding to regularization: regularization requires a separate predeclared experiment and must not be inferred from local geometry alone.

## Exact recommended next experiment

Run a separately predeclared **optimizer-cap sensitivity diagnostic**, not regularization: replay these same 12 paired canonical starts under both baseline and structure-aware charts at `max_nfev=160` and `320`, with the same frozen NTPC rows, objective, 64-node pricer, tolerances, and all existing stability thresholds. Predeclare that persistence of separated near-equivalent clusters after the cap rate materially falls supports continuing structural ambiguity, while a collapse under both charts indicates the 160-evaluation cap was materially numerical. Do not select a budget after seeing results and do not add data, priors, regularization, ANN, or PINN.

## Figures

- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/01_baseline_vs_transformed_rmse.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/02_multistart_parameter_dispersion.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/03_pairwise_cluster_separation.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/04_v0_total_and_allocation.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/05_theta_total_and_allocation.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/06_kappa_and_half_life.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/07_canonical_near_equivalent_vectors.png`
- `market_data_audit/stage_a/derived/ntpc_dh_stability_reparameterization/figures/08_pricing_error_vs_displacement.png`
