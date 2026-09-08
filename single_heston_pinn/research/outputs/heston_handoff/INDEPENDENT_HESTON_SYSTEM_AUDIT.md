# Independent Heston System Audit

Audit date: 2026-08-06  
Scope: Single Heston and Double Heston only  
Verdict: **VALIDATED_WITH_DISCLOSED_LIMITATIONS**

## Evidence result

- 60 of 60 independently executed checks passed.
- Critical failures: 0.
- Warning/limitation records: 9.
- Locked authentic model-input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`.
- Single-Heston historical next-session rows: 8,818.
- Identical finite Single/Double comparison rows: 8,814; documented Double failures: 4.

## Honest interpretation

The saved lineage, chronology, parameter-selection, state propagation, pricing, implied-volatility inversion, no-arbitrage, coverage, and metric claims were recomputed from row-level artifacts. No critical inconsistency was found. This supports reproducibility of the saved experiment; it does **not** prove zero overfitting, globally optimal parameters, or deployable future performance.

The original locked Single Heston beats the prior-session median-IV baseline on the historical next-session test. On the identical finite comparison rows, Double Heston is worse than Single Heston, and the 95% stock-session bootstrap interval for `Double RMSE - Single RMSE` is [0.000607, 0.001993], entirely above zero. The current Double model should therefore be rejected in favor of Single Heston for this evidence set.

## Required caveats

- The historical test period had already been viewed during earlier Single-Heston development; future locked NSE dates are required for pristine forward confirmation.
- The next-session surface score is conditional on realized target spot and listed target strike/expiry coordinates. It is not a pure pre-market forecast of spot or of which contracts will list.
- The same-day Heston strike cross-fit is not unseen-date forecasting. A quadratic same-day smile baseline has lower IV RMSE than Heston (0.014712 vs 0.030028).
- All 55 Double-Heston candidates are boundary-near under the declared diagnostic; multiple parameter combinations can yield similar prices.
- Raw machine outputs containing absolute local paths should not be shared verbatim. The catalogs in this handoff are sanitized.

## Machine-readable evidence

See `independent_heston_audit_checks.csv`, the two parameter catalogs, `heston_metric_catalog.csv`, `privacy_security_scan.json`, and `independent_heston_audit_summary.json`.
