# NTPC Double Heston optimizer-cap sensitivity

## Purpose and frozen design

This last optimizer-budget-only experiment tested whether the NTPC instability was materially caused by `max_nfev=160`. Within each coordinate chart, the corrected 160 control and 320 treatment used the same valuation date, 12 calibration rows, 7 holdout rows, stable-primitive pricing reconstruction, row hashes, prices, activity screen, spot, maturities, carry/RBI inputs, IV inversion, canonical 64-node pricer, canonical bounds and constraints, unweighted price residual, `least_squares(method="trf")`, tolerances, `diff_step`, 12 canonical starts and IDs, near-equivalent/material-distance rules, and complete-linkage cutoff. The only within-chart treatment was `max_nfev: 160 -> 320`. No data, model, objective, weighting, clipping, prior, penalty, or regularization changed.

The required historical-reference and corrected-control gates were **PASSED**. The canonical corrected-control 160 cell exactly reproduced its historical stable-primitive baseline: its median pairwise separation was 0.357338794 versus reviewed 0.357338794, and its maximum distance from best was 0.491085492 versus reviewed 0.491085492. The historical transformed-160 reference used CSV-serialized derived inputs and is therefore retained only as a non-identical-contract historical reference, not claimed as reproduced. The transformed-160 cell reported here is the hash-sealed corrected control built from stable primitives; it and transformed-320 use the identical pricing-input, row, start, coordinate, objective, tolerance, bounds, and optimizer contract except for `max_nfev`. The corrected calibration and holdout pricing-input hashes are `E9F22C823E59A7E4CC71AA8B93A1FC1D4483DAFE62F7984C0504983BC4058098` and `98B6B7B043F95B4F144DCF06625E63D27D1E524AB4C2DF60E088DA2EC7F1D4FA`.

The frozen dispersion classifier is unchanged in meaning: strong reduction requires at least 25% reductions in both median and maximum pairwise separation and fewer clusters; partial reduction requires at least 10% reductions in both separation metrics and no increase in clusters. Materially displaced count remains reported but is not a dispersion-classification gate. Cap-rate classification remains separate.

## Results

| cell | cap | best calibration RMSE | best holdout RMSE | holdout IV RMSE | displaced | clusters | median separation | max separation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_160 | 10/12 (0.833) | 0.234320742 | 0.92682472 | 0.072591562 | 11 | 7 | 0.357338794 | 0.564149107 |
| canonical_320 | 10/12 (0.833) | 0.234290045 | 0.926571662 | 0.0725052689 | 11 | 7 | 0.337210328 | 0.560694158 |
| transformed_160 | 10/12 (0.833) | 0.233174061 | 0.921436878 | 0.0718426463 | 7 | 6 | 0.388560477 | 0.593838944 |
| transformed_320 | 10/12 (0.833) | 0.233165399 | 0.921452862 | 0.0718129472 | 7 | 6 | 0.388481867 | 0.5938481 |

Canonical percentage changes 160 -> 320: calibration RMSE `-0.013%`, holdout RMSE `-0.027%`, holdout IV RMSE `-0.119%`, runtime `72.257%`.

Transformed percentage changes 160 -> 320: calibration RMSE `-0.004%`, holdout RMSE `0.002%`, holdout IV RMSE `-0.041%`, runtime `87.889%`.

Cap decisions: canonical **CAP_NOT_RESOLVED**; transformed **CAP_NOT_RESOLVED**. Dispersion decisions: canonical **DISPERSION_PERSISTS**; transformed **DISPERSION_PERSISTS**. The most variable canonical parameters by raw range at 320 were canonical: `kappa_fast, kappa_slow, rho_slow, rho_fast`; transformed: `kappa_fast, kappa_slow, rho_slow, rho_fast`. Total/split variance, total/split theta, slow/fast kappa and half-life dispersion are recorded under each 320 cell's `derived_coordinate_statistics` in the machine-readable evidence.

| 320 diagnostic | canonical range / CV | transformed range / CV |
|---|---:|---:|
| v0_total | 0.000834082154 / 0.00558790449 | 0.000793928166 / 0.00901327202 |
| alpha_v | 0.82470083 / 1.00000533 | 0.824575315 / 0.761482046 |
| theta_total | 0.0660922333 / 0.210242878 | 0.238828279 / 0.439845303 |
| alpha_theta | 0.936456415 / 1.20741334 | 0.936470485 / 0.73028068 |
| kappa_slow | 2.94869042 / 0.296870892 | 2.03498223 / 0.289352854 |
| kappa_fast | 8.36739599 / 0.380036361 | 11.0335664 / 0.43366241 |
| slow_half_life_days | 4846.49773 / 2.73888205 | 177.837136 / 0.553005961 |
| fast_half_life_days | 48.5634332 / 0.33142819 | 240.703179 / 1.28668881 |

The allocation diagnostics `alpha_v` and `alpha_theta` remain widely dispersed in both charts; they are diagnostics, not new scientific parameters. Slow/fast allocation therefore remains ambiguous after doubling the optimizer budget.

## Interpretation

Final classification: **OPTIMIZER_CAP_UNRESOLVED**.

The cap rate did **not** fall: it remained `10/12 = 0.833` under both coordinate charts, so the numerical-cap confounder was not resolved. Separately, if a valid future comparison lowers the cap rate while separated solutions remain, the correct interpretation would be that the optimizer received substantially more opportunity but materially different parameter basins still fit nearly equivalently—evidence consistent with persistent/global parameter ambiguity, not mathematical proof of structural non-identification.

Best Double Heston holdout RMSE was `0.921436878` versus Standard Heston `0.910569`. The predeclared material-win threshold was `0.865041`: **NO**.

The experiment is valid, but it does not establish that the 160 cap caused the parameter instability. Pricing changed only trivially, cap incidence did not fall, and separated near-equivalent basins persisted. Optimizer-only work is therefore **CLOSED** for this stage. Exact next recommendation: **STOP optimizer-only work and proceed to the predeclared real NTPC multi-date calibration using 2026-07-01, 2026-07-15, and 2026-07-22.** Do not try 640, retune tolerances, change optimizers, or invent another coordinate system.

## Evidence and figures

Per-start final canonical vectors, errors, termination, success, margins, runtime, paired movement and basin-collapse diagnostics are in ignored generated CSV evidence. The eight mentor-ready figures are in `market_data_audit/stage_a/derived/ntpc_dh_optimizer_cap_sensitivity/figures/`; their hashes are sealed by the tracked evidence manifest. Render-only replay regenerates the report, figures, summaries, and manifest from preserved completed optimizer CSVs without rerunning fits.
