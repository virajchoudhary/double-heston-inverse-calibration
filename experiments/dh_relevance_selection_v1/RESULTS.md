# Model comparison on the ex-ante selected surface

Written **after** `DATA_SELECTION_REPORT.md` was completed and committed. The selection commit is
`15007c8`; no pricing error appears in it.

## What was selected, and how

- 62 candidate symbols fetched today; 46 produced a usable surface; **17 passed the domain filter**;
  all 17 were scored.
- Score, fixed in `SELECTION_RULE.md` (SHA-256 `8f66740024e7d257…`):
  `0.40·z(S1) + 0.20·z(S2) + 0.15·z(S3) + 0.15·z(S4) + 0.10·z(S5)`, with S1 the failure of a single
  volatility timescale.
- **Selected: IWM**, score 1.768, rank 1 of 17, clear of QQQ at 0.690.

Its structural profile: one-timescale misfit 0.1234 (highest of any surface that passed the filter),
skew magnitude 0.378, skew variation across maturity 0.370. That is a surface where a constant
volatility is wrong, and where one mean-reversion timescale is visibly not enough.

## Two integrity disclosures

1. **IWM's model errors were already known to me before this selection.** Sixteen symbols, IWM among
   them, had been scored in an earlier step, and IWM was the in-domain surface where Double Heston
   had led Single Heston by 0.15 vol points. The score components and weights were specified in
   advance and applied mechanically, and 8 of the 17 scored surfaces were error-blind — but the
   selection as a whole is **not blind**, and should not be described as such.
2. **The domain filter changed the answer.** Under the earlier rule, which had no level-domain
   filter, SPY scored highest and Double Heston *lost* there (6.54 vs 6.51). Here SPY is rejected
   because its short-dated at-the-money volatility, 12.3%, falls below the PINN's trained level range
   of 13.4–28.6%. The filter is legitimate — selecting a surface the network must extrapolate on
   would be worse — but it is the reason the selected surface changed, and I am stating that rather
   than presenting IWM as the unambiguous winner of a single clean rule.

Both earlier results stand unaltered in `dh_pinn_v5/multi_surface/results.json`.

## Result 1 — matched capacity (the only setting the PINN can enter)

Published parameters, one level scale fitted on the surface. IV RMSE in vol points, 1,416 quotes:

| model | level scale | IV RMSE |
|---|---:|---:|
| Black-Scholes | 1.492 | 5.611 |
| Single Heston | 1.303 | 5.453 |
| **Double Heston (exact)** | 1.285 | **5.301** |
| DH-PINN, locked v4 | 1.285 | 5.329 |
| DH-PINN, improved v5 | 1.285 | 5.340 |

Double Heston leads, but by 0.15 vol points out of 5.3 — the same small margin as everywhere else
with fixed parameters. The PINN sits within 0.04 of exact Double Heston, as designed.

## Result 2 — strongest fair calibration, held-out quotes

Each model calibrated on a checkerboard half (708 quotes) and scored on the other half (708 quotes it
never saw). Black-Scholes gets one volatility per expiry, Single Heston all five parameters, Double
Heston all ten, all with the same global-plus-multistart optimiser.

| model | free parameters | held-out IV RMSE | in-sample IV RMSE |
|---|---:|---:|---:|
| Black-Scholes, one vol per expiry | 22 | 5.388 | 5.451 |
| Single Heston | 5 | 1.893 | 1.898 |
| **Double Heston** | 10 | **0.583** | 0.593 |

**This is the clear result.** On a surface selected in advance for its two-timescale structure, and
judged on quotes no model was fitted to:

- Double Heston is **3.2× more accurate than the strongest Single Heston**;
- Single Heston is 2.8× more accurate than Black-Scholes;
- Black-Scholes is worst **despite having the most free parameters** (22 volatilities against Double
  Heston's 10), which is exactly the structural point: extra volatility levels cannot produce a smile.

Held-out and in-sample errors agree to within 1%, so this is not parameter count buying fit.

## The predicted chain, and whether it held

| prediction | outcome |
|---|---|
| Black-Scholes disadvantaged: volatility is not constant | confirmed, 5.39 vol points with 22 free volatilities |
| Single Heston disadvantaged: one timescale is not enough | confirmed, 1.89 |
| Double Heston structurally capable of both horizons | confirmed, 0.58 |
| Fixed published parameters reproduce this advantage | **not confirmed** — at matched capacity the gap is only 0.15 |

The advantage is real and large, but it requires the parameters to be **fitted to the surface**. With
the published parameters frozen, the two-factor structure is present but unused, which is consistent
with everything else this project has measured.

## Figures

| file | content |
|---|---|
| `IWM_C_vs_S_strongest_calibration.png` | C vs S at the 84-day expiry, market held-out quotes, calibrated BS/SH/DH, residual panel |
| `IWM_C_vs_calendar_time.png` | C as calendar time advances toward expiry, same calibrated models |
| `IWM_C_vs_S_matched_capacity.png` | published parameters plus one level scale, with both PINNs |

In the residual panel of the first figure, Black-Scholes departs by up to 1.4 price units while
Double Heston stays inside about 0.2 — the separation comes from the models, not from the drawing.
The kinks in the Black-Scholes calendar curve are real: they are the knots of its per-expiry
volatility term structure.

## Controlled benchmark

The mechanism figures, where the PINN participates at full fidelity, remain in
`experiments/financial_curve_figures_v2/` and are labelled **CONTROLLED TWO-TIMESCALE BENCHMARK**.
They are not market evidence, and nothing here changes them.
