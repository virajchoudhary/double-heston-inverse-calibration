# NTPC Double Heston optimizer-cap sensitivity

## Purpose and frozen design

This last optimizer-budget-only experiment tested whether the NTPC instability was materially caused by `max_nfev=160`. Exactly four paired cells used the same valuation date, 12 calibration rows, 7 holdout rows, row hashes, prices, activity screen, spot, maturities, carry/RBI inputs, IV inversion, canonical 64-node pricer, canonical bounds and constraints, unweighted price residual, `least_squares(method="trf")`, tolerances, `diff_step`, 12 canonical starts and IDs, near-equivalent/material-distance rules, and complete-linkage cutoff. The only treatment was `max_nfev: 160 -> 320`. No data, model, objective, weighting, clipping, prior, penalty, or regularization changed.

The 160 baseline reproduction gate was **FAILED**. The transformed 160 cell reproduced the reviewed baseline exactly. The canonical 160 cell did not: its median pairwise separation was 0.316942165 versus reviewed 0.357338794, and its maximum distance from best was 0.491411289 versus reviewed 0.491085492. Because required canonical stability metrics did not reproduce within the predeclared deterministic tolerance, the four-cell comparison is invalid and the 320 results below are descriptive only. Calibration and holdout row hashes are `F44371DD418304789DC4B97C1710DCE60CDC0232A75C172FDC90E220738A0B7F` and `E8B552A1B218F5405D5871C11F9AAA4F6460310BFEC8E300501A73C00F1FBA07`.

## Results

| cell | cap | best calibration RMSE | best holdout RMSE | holdout IV RMSE | displaced | clusters | median separation | max separation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_160 | 10/12 (0.833) | 0.234320731 | 0.926824647 | 0.0725915341 | 11 | 7 | 0.316942165 | 0.564166617 |
| canonical_320 | 10/12 (0.833) | 0.234290042 | 0.926571624 | 0.0725052581 | 11 | 6 | 0.316319898 | 0.478635312 |
| transformed_160 | 10/12 (0.833) | 0.233174148 | 0.921582641 | 0.0718592663 | 7 | 6 | 0.388626346 | 0.593889521 |
| transformed_320 | 10/12 (0.833) | 0.233165379 | 0.921452762 | 0.0718128912 | 7 | 6 | 0.388394876 | 0.593808112 |

Canonical percentage changes 160 -> 320: calibration RMSE `-0.013%`, holdout RMSE `-0.027%`, holdout IV RMSE `-0.119%`, runtime `92.462%`.

Transformed percentage changes 160 -> 320: calibration RMSE `-0.004%`, holdout RMSE `-0.014%`, holdout IV RMSE `-0.065%`, runtime `100.217%`.

Cap decisions: canonical **CAP_NOT_RESOLVED**; transformed **CAP_NOT_RESOLVED**. Dispersion decisions: canonical **DISPERSION_PERSISTS**; transformed **DISPERSION_PERSISTS**. The most variable canonical parameters by raw range at 320 were canonical: `kappa_fast, rho_slow, rho_fast, sigma_fast`; transformed: `kappa_fast, kappa_slow, rho_slow, rho_fast`. Total/split variance, total/split theta, slow/fast kappa and half-life dispersion are recorded under each 320 cell's `derived_coordinate_statistics` in the machine-readable evidence.

| 320 diagnostic | canonical range / CV | transformed range / CV |
|---|---:|---:|
| v0_total | 0.000833539576 / 0.00559644159 | 0.000796174711 / 0.00902682347 |
| alpha_v | 0.824713587 / 1.00000243 | 0.824575389 / 0.761487687 |
| theta_total | 0.065670881 / 0.210274119 | 0.238948435 / 0.439974487 |
| alpha_theta | 0.936457299 / 1.20745061 | 0.93648065 / 0.730272035 |
| kappa_slow | 0.247535593 / 0.0226816199 | 2.03491469 / 0.288027277 |
| kappa_fast | 8.36745236 / 0.379685697 | 11.0334743 / 0.433442289 |
| slow_half_life_days | 7.58425665 / 0.0243329425 | 177.818774 / 0.549007786 |
| fast_half_life_days | 48.5645153 / 0.331318903 | 240.678056 / 1.28696533 |

The allocation diagnostics `alpha_v` and `alpha_theta` remain widely dispersed in both charts; they are diagnostics, not new scientific parameters. Slow/fast allocation therefore remains ambiguous in the descriptive 320 evidence. This cannot repair the failed validity gate.

## Interpretation

Final classification: **INVALID**.

The cap rate did **not** fall: it remained `10/12 = 0.833` under both coordinate charts, so the numerical-cap confounder was not resolved. Separately, if a valid future comparison lowers the cap rate while separated solutions remain, the correct interpretation would be that the optimizer received substantially more opportunity but materially different parameter basins still fit nearly equivalently—evidence consistent with persistent/global parameter ambiguity, not mathematical proof of structural non-identification.

Best Double Heston holdout RMSE was `0.921452762` versus Standard Heston `0.910569`. The predeclared material-win threshold was `0.865041`: **NO**.

No pricing, cap, cluster, separation, allocation, or Heston-superiority claim from the 320 cells is accepted as a scientific conclusion because the experiment is invalid. Optimizer-only experiments should **not continue automatically**. Exact next recommendation: **Repair the matched frozen experiment contract before drawing a scientific conclusion.** No new scientific experiment should begin until that reproducibility defect is resolved under a separately authorized protocol.

## Evidence and figures

Per-start final canonical vectors, errors, termination, success, margins, runtime, paired movement and basin-collapse diagnostics are in ignored generated CSV evidence. The eight mentor-ready figures are in `market_data_audit/stage_a/derived/ntpc_dh_optimizer_cap_sensitivity/figures/`; their hashes are sealed by the tracked evidence manifest. Render-only replay regenerates the report, figures, summaries, and manifest from preserved completed optimizer CSVs without rerunning fits.
