# Independent single-Heston overfitting and leakage audit

## Verdict

The fitted model is **not perfectly matching the quotes and there is no evidence of row-level label memorization**. Three-fold cross-fitting assigns every strike, including both its call and put, wholly to calibration or evaluation. All 11 independent leakage checks passed.

However, the original result is an **inverse calibration on observed dates**, not a fully unseen-date forecast. Structural parameters are learned from early dates, but each later date uses same-day anchor option prices to fit one variance state (`v0`) and infer carry. The publication must call the test “held-out-strike evaluation within later dates.”

## Three-fold cross-fit result

| Test | IV RMSE | Pooled R² | Within-surface centered R² |
|---|---:|---:|---:|
| Heston, same-day anchor `v0` | 0.030028 | 0.903410 | 0.617874 |
| Flat IV from same-day anchors | 0.040577 | 0.823628 | -0.008919 |
| Quadratic smile from same-day anchors | 0.014712 | 0.976814 | 0.862330 |
| Heston, fixed training-median `v0` | 0.067473 | 0.512323 | 0.597245 |
| Heston, prior available `v0` | 0.042245 | 0.805614 | 0.599650 |
| Heston, shuffled `v0` control | 0.068991 | 0.489141 | 0.359333 |
| Heston, train-half A parameters | 0.030932 | 0.897505 | 0.625543 |
| Heston, train-half B parameters | 0.041316 | 0.817046 | 0.624193 |

The pooled R² is much larger than the within-surface centered R² because pooled variation includes easy differences in volatility level between symbols, dates and expiries. Only the centered measure tests how well the smile shape is captured after those levels are removed.

The cluster-bootstrap 95% interval for `Heston RMSE - flat-anchor RMSE` is [-0.012206, -0.008641]. A negative interval means Heston improves on the simple same-date flat baseline; an interval crossing zero means the improvement is not established.

The quadratic anchor-smile control is substantially more accurate than Heston on held-out strikes. It also uses more date/expiry-specific flexibility, so it is not a like-for-like structural model, but this result forbids any claim that Heston is the best empirical interpolator.

## Coverage and selection

The cross-fit contains 4,709 unique held-out quotes from 37,080 model-ready rows in the chronological test periods (12.70%). Results therefore apply to liquid, paired, parity-consistent quotes that pass the documented maturity, IV, moneyness and vega filters—not to every raw NSE option row.

## Parameter honesty

Two disjoint halves of the selected training dates were fitted independently for every stock. Across the 22 fits there were 39 parameter-boundary flags. The largest half-sample changes were: `|log kappa ratio|` 1.651, `|log theta ratio|` 1.755, `|log sigma ratio|` 0.204, and `|rho difference|` 0.109.

Consequently, pricing performance may be stable while individual Heston parameters remain weakly identified. Do not describe the parameters as uniquely recovered economic constants.

Wing performance is materially worse than near-the-money performance, and CESC has negative pooled R². These subgroup failures must accompany the aggregate result. A fully no-same-day-option-input forecast was not claimed because the dataset lacks independently aligned point-in-time interest-rate and dividend inputs; same-day anchor quotes currently provide effective carry.

## Publication-safe claim

“On a strictly separated subset of later-date NSE power-stock option strikes, a one-factor Heston calibration with train-only structural parameters and same-day anchor-based variance state achieved the reported cross-fit error. Both sides of every held-out strike were excluded from carry and state estimation. Performance is conditional on liquidity and parity filters; it is not a no-same-day-data forecast, and structural parameters exhibit boundary and sample-instability warnings.”
