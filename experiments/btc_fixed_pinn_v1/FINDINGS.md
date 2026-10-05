# Saved C3 PINN with hardcoded parameters: Bitcoin findings

The actual saved two-seed C3 PINN was used, without retraining or calibration.
The experiment is **retrospective and exploratory**, not untouched confirmation.

## Coverage

The fixed eligibility rules admitted **201 quotes on 10 of 51 requested archived
dates**: 139 quotes on seven shock dates, 62 quotes on three calm dates. Actual
retained maturities were 7.00–67.04 days. Forty-one dates had no scored quotes.
The largest exclusion was missing sufficiently close earlier call/put anchor
pairs (6,464 later instrument observations). No rules were relaxed after scoring.

## What worked and what did not

| Fixed scenario | PINN vs market USD RMSE | SH numerical vs market | BS fixed vs market | PINN vs numerical DH USD RMSE |
|---|---:|---:|---:|---:|
| 28% initial total volatility | 857.65 | 760.18 | 892.89 | 0.762 |
| 45% initial total volatility | 408.87 | 420.11 | 411.25 | 0.868 |
| 60% initial total volatility | 845.94 | 786.20 | 1,028.19 | 1.449 |

Dollar RMSE is pooled across quotes, per one-BTC underlying option premium.
The primary normalized metrics and equal-date medians are in `artifacts/metrics.csv`;
they can produce a different ranking because they weight the data differently.
For example, at 45%, pooled forward-normalized RMSE favors BS (0.007068) over the
PINN (0.007465), even though pooled dollar RMSE narrowly favors the PINN.
There is **no robust, universal Double Heston superiority demonstrated here**.

The 45% scenario had the lowest aggregate market error among the three displayed
PINN scenarios; it is **not an independently validated optimum or a final model
selection**. All parameter vectors were fixed before this run. The 45%/60%
scenarios are proposed in-domain stress states based on the existing literature
shape, not published Bitcoin parameter estimates.

Neural price approximation is much smaller than total market error. The 45% and
60% scenarios passed all inherited price/IV fidelity gates. The 28% scenario
passed the price gates but failed the IV gate: 0.287 volatility points versus the
0.2 threshold; 194/201 PINN/reference IV comparisons were valid. Failed IV
inversions never removed the corresponding price errors.

## Validity checks completed

- Raw archive and checkpoint hashes verified before and after scoring.
- All models scored identical quotes; no target-price, IV or error-based
  cherry-picking. Malformed inputs, domain exclusions and reserved anchors logged.
- Separate 06:00–07:00 UTC anchor trades estimated forwards. Scored trades are
  from 07:00–08:00, with both option types of anchor strikes excluded.
- On every scored date, perturbing all second-hour prices and IV/mark fields
  left membership, forward inputs and actual PINN predictions unchanged.
- Self-test passed: parity/units, time/strike separation, target perturbation,
  duplicates and parameter-domain constraints.
- Existing `test_regular_pinn_torch_physics.py` and v4 `test_experiment.py`:
  **12 tests passed**. No model/PDE source or saved weights were changed.
- Report-rendering identifier mismatch repaired separately, without changing any
  frozen source, settings or score files (`artifacts/reporting_repair.json`).

These checks address specific failure modes, not a mathematical guarantee of
absence of all bias. Asynchronous trade prices, estimated basis, zero discounting,
limited coverage and prior exposure to these historical dates remain limitations.
Bitcoin model parameters were not fitted. The source data are authentic archived
trades, not synthetic generated market observations. The synthetic fixture in
`self_test()` is isolated test code and is never included in scores.

## Files

- `artifacts/REPORT.md`: full report and two figures.
- `artifacts/protocol.json`: exact ten-parameter vectors, all three scenarios,
  baseline settings, rules and input hashes, frozen before this evaluation.
- `artifacts/predictions.csv`: per-option actual prices and model predictions.
- `artifacts/metrics.csv`, `daily_metrics.csv`, `fidelity.json`: complete scores.
- `artifacts/cleaning_audit.json`, `forward_anchors.csv`, `leakage_probes.json`,
  `integrity.json`: coverage and integrity evidence.

Correct BTC-to-dollar conversion follows the [Deribit inverse-option
documentation](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options):
USD premiums use the index price; implied volatility uses the forward. Zero
discounting and carrying the earlier forward/index ratio into the scoring hour
are explicit common assumptions. This differs from the old BTC experiment, so
its calibrated dollar errors must not be compared directly with these results.
