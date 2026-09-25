# Heston and Double-Heston teammate context

Version date: 2026-08-06  
Audit verdict: **VALIDATED_WITH_DISCLOSED_LIMITATIONS** (60/60 independent checks passed; 0 critical failures).

## Decision

Keep the current locked **Single Heston** as the historical winner. Do not promote the current Double Heston: on 8,814 identical finite rows, Single RMSE is 0.044200 and Double RMSE is 0.045510; the stock-session bootstrap interval for `Double - Single RMSE` is [0.000607, 0.001993].

## Data and evaluation boundary

- Locked source: `outputs/019fc8a0/model_input_option_prices.csv`.
- SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`.
- 572,512 total clean-release rows; 215,636 model-ready rows; 11 model stocks.
- Chronological split: 70% train / 15% validation / 15% test independently per stock.
- Exact expiry dates are used; no Tuesday/Thursday weekday rewriting.
- Forecast scores are conditional on realized target spot and target strike/expiry coordinates. They are not pure pre-market forecasts.
- The historical test was previously viewed; future locked dates after 2026-08-03 are required for pristine confirmation.

## Implemented equations

Single Heston has four structural parameters `(kappa, theta, sigma, rho)` and one date-specific variance state `v0`. Double Heston has slow and fast copies of those four structural parameters plus two date-specific states, for ten operational parameters. The implementation constrains positive parameters, Single/Double Feller gaps, slow/fast ordering, and `rho_slow^2 + rho_fast^2 < 1`.

### Selected Single-Heston parameters

| Symbol | Candidate | kappa | theta | sigma | rho | v0 train median | Val IV RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADANIENSOL | half_a | 9.996663 | 0.402712 | 2.822393 | -0.173073 | 0.151467 | 0.049179 |
| ADANIGREEN | half_a | 3.981365 | 0.999666 | 2.806315 | -0.074662 | 0.167640 | 0.077256 |
| ADANIPOWER | main | 9.996659 | 0.560397 | 3.039316 | -0.055311 | 0.325296 | 0.023363 |
| CESC | half_a | 5.506302 | 0.879373 | 3.095346 | 0.134853 | 0.039588 | 0.073465 |
| JSWENERGY | main | 9.996663 | 0.426153 | 2.903374 | -0.122050 | 0.098113 | 0.044995 |
| NHPC | half_a | 9.996663 | 0.258595 | 2.261676 | 0.012605 | 0.045768 | 0.042614 |
| NTPC | half_a | 9.996663 | 0.120830 | 1.545998 | -0.094111 | 0.056204 | 0.048292 |
| POWERGRID | main | 9.996663 | 0.155460 | 1.753598 | -0.114274 | 0.029378 | 0.046688 |
| SJVN | main | 9.996443 | 0.434051 | 2.930124 | -0.001738 | 0.127590 | 0.044968 |
| TATAPOWER | main | 9.996663 | 0.175414 | 1.862741 | 0.030042 | 0.064200 | 0.052841 |
| TORNTPOWER | half_b | 9.996663 | 0.395235 | 2.796071 | -0.113463 | 0.056940 | 0.050559 |

`v0 train median` is descriptive only; operational forecasts use the saved origin state propagated to the target session.

### Selected Double-Heston parameters

| Symbol | Candidate | k slow | theta slow | sigma slow | rho slow | v0 slow med | k fast | theta fast | sigma fast | rho fast | v0 fast med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADANIENSOL | training_half_a | 4.755650 | 0.353849 | 1.821515 | 0.567525 | 0.045393 | 9.917287 | 0.206907 | 2.013533 | -0.760214 | 0.093458 |
| ADANIGREEN | training_half_a | 3.813725 | 0.171570 | 1.137861 | -0.949681 | 0.029927 | 3.927227 | 0.827656 | 2.536069 | -0.000001 | 0.095305 |
| ADANIPOWER | full_start_3 | 4.998340 | 0.845936 | 2.892500 | -0.531453 | 0.078482 | 9.997614 | 0.153734 | 1.208556 | 0.787016 | 0.209173 |
| CESC | training_half_a | 0.051704 | 0.150343 | 0.013593 | -0.001108 | 0.005843 | 5.762406 | 0.847978 | 3.109475 | -0.000000 | 0.024120 |
| JSWENERGY | full_start_1 | 4.830192 | 0.336053 | 1.790856 | 0.650200 | 0.026803 | 9.869913 | 0.235272 | 2.142738 | -0.691422 | 0.047385 |
| NHPC | training_half_a | 0.063574 | 0.135629 | 0.129923 | -0.944830 | 0.004114 | 3.360026 | 0.660608 | 2.095670 | -0.000007 | 0.030068 |
| NTPC | training_half_b | 4.997313 | 0.081178 | 0.895853 | 0.650635 | 0.022222 | 9.997454 | 0.066465 | 1.146657 | -0.691616 | 0.030299 |
| POWERGRID | full_start_3 | 4.995617 | 0.153755 | 1.232828 | -0.735950 | 0.011819 | 5.113148 | 0.109849 | 1.054234 | 0.600222 | 0.013167 |
| SJVN | full_start_1 | 1.063320 | 0.150794 | 0.563266 | -0.949680 | 0.016240 | 4.253820 | 0.842980 | 2.663736 | -0.000002 | 0.053298 |
| TATAPOWER | full_start_1 | 0.051665 | 0.030971 | 0.056253 | -0.948730 | 0.006149 | 9.996651 | 0.174263 | 1.856621 | -0.000001 | 0.053749 |
| TORNTPOWER | full_start_3 | 4.850334 | 0.254662 | 1.561242 | 0.641609 | 0.033945 | 9.781598 | 0.209969 | 2.015034 | -0.698837 | 0.039619 |

## Accuracy and overfitting evidence

- Original locked Single next-session test: RMSE 0.044469, R2 0.814899, versus prior-session median-IV baseline RMSE 0.068079.
- Identical common rows: Single RMSE 0.044200, Double RMSE 0.045510. Double wins only JSWENERGY and TORNTPOWER.
- Same-day three-fold strike crossfit: Heston RMSE 0.030028, within-surface R2 0.617874. A quadratic same-day smile baseline is better at RMSE 0.014712.
- Boundary warnings: 39 flags across 22 Single half-sample fits; 55/55 Double candidates boundary-near.
- Exact synthetic recovery: Single relative error 3.222e-10; Double structural error 1.189e-05. With 1% price noise, Double max structural error rises to 28.875%.

## Continuation protocol

1. Freeze the current code, parameters, checksums, and this report.
2. Do not use the previously viewed historical test for more tuning or model selection.
3. Acquire future authentic NSE sessions after 2026-08-03 and process them through the locked cleaning pipeline.
4. Generate predictions using only information available at each origin session.
5. Score Single, Double, and the declared baseline on identical finite row keys; retain every failure row.
6. Cluster-bootstrap by stock-session and report both pooled and per-stock metrics.
7. Promote Double only if a preregistered future-window criterion is met and parameter stability improves materially.

## File map

- Core pricing: `single_heston.py`, `double_heston.py`.
- Forecast/evaluation: `forecast_single_heston_next_day.py`, `compare_single_double_heston.py`.
- Independent audit: `audit_heston_handoff.py`.
- Machine audit: `outputs/heston_handoff/independent_heston_audit_checks.csv`.
- Sanitized parameters: `outputs/heston_handoff/single_heston_selected_parameter_catalog.csv`, `double_heston_selected_parameter_catalog.csv`.
- Full report: `outputs/heston_handoff/Heston_Double_Heston_Validated_Teammate_Handoff.docx`.

## Non-negotiable limitations

- No finite historical test can prove zero overfitting or a global optimum.
- The historical test period was previously viewed during Single-Heston development; it is not pristine final holdout evidence.
- Next-session scoring is conditional on realized target spot and the target session's listed strike/expiry coordinates.
- Double Heston underperforms Single Heston on the identical historical rows and has severe parameter-boundary/identifiability warnings.
- The retrospective Single-Heston recalibration must be confirmed on future locked NSE dates before being called forward validated.

## Audit check index

| Category | Check | Passed | Observed |
| --- | --- | --- | --- |
| source | input_sha256_matches_locked_NSE_model_input | True | a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3 |
| source | model_ready_row_count_is_locked | True | 215636 |
| source | model_input_row_keys_unique | True | 0 |
| source | model_input_contains_11_power_stocks | True | 11 |
| source | model_ready_filter_reproduces_locked_subset | True | all=572512, ready=215636 |
| source | source_paths_are_NSE_bhavcopies | True | 0 |
| source | referenced_raw_NSE_files_exist_locally | True | 0 |
| expiry | expiry_distance_matches_saved_days | True | 0 |
| expiry | maturity_is_exact_days_divided_by_365 | True | 4.931506591976387e-11 |
| expiry | multiple_historical_expiry_weekdays_preserved | True | Monday, Thursday, Tuesday |
| chronology | single_candidates_strictly_precede_validation | True | 0 |
| chronology | single_origin_strictly_precedes_target | True | 0 |
| chronology | single_saved_split_matches_independent_70_15_15 | True | 0 |
| chronology | single_validation_and_test_keys_disjoint | True | 0 |
| selection | single_validation_selection_recomputed | True | 11 |
| selection | single_selection_declares_validation_only | True | 0 |
| selection | single_has_three_train_only_candidates_per_stock | True | 3 |
| lineage | single_test_rows_resolve_to_source | True | 0 |
| lineage | single_test_values_match_source_rows | True | 0 |
| equation | single_variance_propagation_recomputed | True | 1.1102230246251565e-16 |
| equation | single_saved_prices_reprice_exactly | True | 1.1368683772161603e-12 |
| equation | single_saved_IV_inverts_saved_prices | True | 5.703770789011742e-13 |
| validity | single_prices_within_no_arbitrage_bounds | True | 0 |
| constraints | all_single_candidates_Feller_valid | True | 0.02495313693270962 |
| constraints | all_single_candidates_within_declared_bounds | True | 33 candidates |
| metrics | single_selected_heston_metrics_recomputed | True | 9.367506770274758e-17 |
| metrics | single_prior_session_expiry_median_iv_metrics_recomputed | True | 8.326672684688674e-17 |
| chronology | double_candidates_strictly_precede_validation | True | 0 |
| selection | double_has_five_train_only_candidates_per_stock | True | 5 |
| selection | each_double_validation_key_has_all_five_candidates | True | 5 |
| selection | double_validation_selection_recomputed | True | 11 |
| selection | double_selection_declares_validation_only | True | 0 |
| constraints | all_double_candidates_satisfy_both_Feller_conditions | True | 3.28336959995474e-05 |
| constraints | all_double_candidates_have_ordered_factors | True | 0 |
| constraints | all_double_candidates_satisfy_joint_correlation_disk | True | 0.9496814173760569 |
| overfit | double_boundary_near_candidates_recorded | True | 55 |
| equation | double_variance_propagation_recomputed | True | 1.1102230246251565e-16 |
| equation | double_saved_prices_reprice_exactly | True | 6.821210263296962e-13 |
| equation | double_saved_IV_inverts_saved_prices | True | 1.1843370728570335e-10 |
| validity | double_prices_within_no_arbitrage_bounds | True | 0 |
| coverage | single_test_universe_has_8818_rows | True | 8818 |
| coverage | double_common_and_failure_rows_partition_single_universe | True | common=8814, failures=4, universe=8818 |
| coverage | double_invalid_rows_are_documented_not_imputed | True | 4 |
| lineage | single_and_double_use_identical_common_rows | True | 8814 |
| metrics | single_heston_comparison_metrics_recomputed | True | 7.632783294297951e-17 |
| metrics | double_heston_comparison_metrics_recomputed | True | 8.326672684688674e-17 |
| metrics | double_minus_single_cluster_bootstrap_recomputed | True | 0.0 |
| overfit | historical_bootstrap_favors_single_Heston | True | 95% interval [0.00060695, 0.00199293] |
| overfit | single_three_fold_strike_crossfit_metrics_recomputed | True | 3.469446951953614e-18 |
| leakage | all_strike_crossfit_leakage_checks_pass | True | 11/11 |
| overfit | quadratic_same_day_smile_beats_single_Heston | True | quadratic=0.014712; Heston=0.030028 |
| overfit | single_half_sample_boundary_flags_recorded | True | 39 |
| honesty | generalized_single_result_labeled_retrospective | True | retrospective historical-test comparison; the original test graph was viewed before the improvement was designed, so future NSE sessions are required for pristine forward confirmation |
| honesty | conditional_evaluation_scope_explicit | True | ['CONDITIONAL_ON_REALIZED_TARGET_SPOT_AND_QUOTES'] |
| controlled | single_exact_synthetic_recovery | True | 3.221947400236205e-10 |
| controlled | single_one_percent_noise_sensitivity_recorded | True | 0.1241260944772804 |
| controlled | double_exact_synthetic_recovery | True | 1.1886345312058081e-05 |
| controlled | double_one_percent_noise_sensitivity_recorded | True | 0.28874634698904245 |
| privacy | no_credential_patterns_in_reviewed_code_and_reports | True | [] |
| privacy | raw_machine_artifacts_with_local_paths_identified | True | ['outputs/single_heston_next_day_forecast/parameter_candidates_train_only.csv', 'outputs/single_heston_next_day_forecast/next_day_forecast_summary.json'] |
