# Leakage-Controlled Next-Session Heston Forecast

## Result

The locked Heston specification was selected separately for each stock using **validation dates only**, then evaluated once on later test sessions. It produced test IV RMSE **0.044469** (MAE 0.028048, pooled R² 0.814899) across **8,818** authentic NSE option quotes and **286** stock-sessions. The prior-session expiry-median-IV baseline produced RMSE **0.068079**. By pooled test RMSE, the winner is **Heston**.

The daily-cluster bootstrap estimate of Heston RMSE minus baseline RMSE is **-0.021160**, with 95% interval [-0.023322, -0.018821]. Negative values favor Heston.

Heston beats the baseline by test RMSE for **10 of 11 stocks**. The baseline is better or tied for: **ADANIPOWER**. This stock-level exception is retained rather than hidden by the pooled result.

## What was forecast

For every origin session t, all known clean t option quotes were used to calibrate the one-day state `v0`. Its next-session conditional expectation was computed exactly as `theta + (v0 - theta) * exp(-kappa * calendar_days / 365)`. Rates and dividend yields were carried from t. The file `next_session_normalized_full_surface.csv` is the actual origin-time forecast artifact on a normalized spot of 100, 25 forward-moneyness points per surviving expiry, and contains no target prices or target spot.

This daily NSE bhavcopy source cannot support an intraday volatility path. “Full day” here means the complete next-session cross-sectional volatility surface across available expiries and moneyness, not minute-by-minute volatility.

## Parameter search without test leakage

Three economically coherent train-only calibrations were tried per stock: the main full-training fit and two independent training-half fits (`half_a`, `half_b`). The lowest validation IV RMSE selected the candidate for each stock. No test quote selected parameters, starting values, or stopping rules. Test failure was not hidden by continuing to tune.

## Evaluation boundary

Quote-level test scoring is conditional on the realized target-day spot and listed strike/expiry coordinates. Those fields locate the observed surface after the forecast; authentic target option prices are labels used only for scoring. Therefore this is an **unseen-date conditional volatility-surface forecast**, not a pre-open option-price forecast. A pre-open price forecast additionally requires a separately validated spot process and independently sourced rate/dividend curves.

Expiry handling uses exact expiry dates and exact calendar gaps. It makes no Tuesday/Thursday weekday assumption, so historical expiry-rule changes are naturally retained.

## Integrity

- Authentic input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`
- Authentic ready rows loaded: 215,636
- Parameter candidates: 33 (3 per stock)
- Test evaluation rows: 8,818
- Normalized forecast-grid rows: 7,625
- Integrity checks passed: 19/19

The forecast grid is model-generated and explicitly labelled as such. It must never be presented as original NSE observations. The evaluation rows retain their authentic NSE source-file paths and row keys.
