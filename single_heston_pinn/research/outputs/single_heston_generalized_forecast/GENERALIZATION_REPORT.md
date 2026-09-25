# Heston Generalisation Improvement

## Result

In the retrospective historical-test comparison, the guarded validation-calibrated forecast reduces IV RMSE from **0.044469** to **0.036590** (17.72% relative improvement). Pooled R² rises from **0.814899** to **0.874685**. The diagnostic forecast-on-actual slope moves from **0.806598** toward the ideal value 1, reaching **0.878963**.

The red dashed graph line is always the fixed identity reference `forecast = actual`; its 45-degree angle is deliberately identical for every stock and is not the model fit. Each stock panel now also contains a black empirical test-trend line whose slope is reported in the title. That black line is diagnostic only and is not used to generate predictions.

## Generalisation protocol

Structural candidate selection used only the early chronological validation dates. Recalibration complexity used only later validation dates. A correction was accepted only with at least 10 validation sessions and at least 5% improvement in late-validation daily RMSE. Accepted corrections were refit on all validation data. No historical test price fitted a coefficient or directly selected a candidate/specification.

However, the original test graph had already been viewed before this improvement was designed. The safeguard design was therefore made with awareness of historical test behavior. This means the comparison is **not a pristine final generalisation test**. The code must now remain locked and be scored on future authentic NSE sessions before making a forward-performance claim.

Corrections were accepted for: **NHPC, NTPC, POWERGRID, TATAPOWER**. Other stocks retain their raw Heston forecast because the validation evidence did not clear both guards.

## Scope

This remains a next-session conditional implied-volatility-surface forecast. Target prices are evaluation labels only. The normalized full-surface file is model-generated, contains no target market prices or target spot, and is explicitly labelled as generated rather than authentic NSE observations. Exact expiry dates are retained without a Tuesday/Thursday rule.

## Integrity

- Authentic input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`
- Test rows: 8,818
- Normalized forecast-grid rows: 7,625
- Integrity checks passed: 16/16
