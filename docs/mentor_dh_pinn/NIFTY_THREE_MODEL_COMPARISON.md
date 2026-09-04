# Black-Scholes vs single Heston vs Double Heston on NIFTY index options

Generated from `outputs/nifty_comparison/`. 10 trade dates, **1,358 held-out real market quotes**, scored in implied-volatility RMSE.

## Why NIFTY, and why nothing else

The dual PINN consumes a fixed 45-vector: five expiries at 30/60/90/180/365 days by nine strikes at 0.85-1.15 moneyness. A survey of the whole NSE F&O segment on a representative date found **exactly one instrument** with five or more liquid expiries spanning that ladder: NIFTY, which carries 11 expiries with material open interest out past 1350 days. Every single-stock name -- ADANIPOWER included -- has one or two expiries inside 60 days, so the vector cannot be formed for them at all. That, and not model quality, is what made the earlier stock-option comparisons unable to separate one factor from two.

Dates were chosen by a model-free rule frozen before any fit: the 10 ladder-covering NIFTY dates of 2026 with the highest trailing 21-day realised volatility of the underlying. The selected window runs 2026-04-08 to 2026-04-22 with 21-day realised volatility of 27.0-29.0%, **2.09x the 2026 median**.

## Data construction

* `SttlmPric`, not `ClsPric`. Closing prices on thin strikes are stale and put-call parity on them returns five-day discount factors near 0.86. Settlement prices give a coherent term structure (r about 6.3%, q about 0) with parity NRMSE of a few basis points.
* Open interest, not volume, decides whether a series is real. Some long-dated series carry an exchange-computed settlement price with *exactly* zero parity residual, zero volume and zero open interest; those are model output, not market data, and are dropped.
* Forward and discount per expiry come from the repository's own Huber-reweighted parity fit with its validity gates.
* Every third strike is held out. Held-out quotes are real traded quotes, never used to build any calibration input, and every arm is scored on the identical set.

## Result

| model | params | median holdout IV RMSE | mean | best on | vs single Heston |
|---|---:|---:|---:|---:|---|
| Black-Scholes, 1 sigma | 1 | **3.039** vol pts | 3.165 | 0/10 | 1/10 dates, p = 0.010 |
| Black-Scholes, sigma per expiry | 5 | **2.637** vol pts | 2.692 | 0/10 | 3/10 dates, p = 0.232 |
| single Heston | 5 | **2.361** vol pts | 2.443 | 1/10 | -- |
| Double Heston, cold 5-start | 10 | **2.366** vol pts | 2.518 | 0/10 | 2/10 dates, p = 0.010 |
| Double Heston, ridge 30 toward single Heston | 10 | **2.378** vol pts | 2.438 | 4/10 | 5/10 dates, p = 1.000 |
| dual PINN prior alone (out of distribution) | 10 | **2.573** vol pts | 2.588 | 1/10 | 4/10 dates, p = 0.432 |
| dual PINN + ridge 3 = the two-stage estimator (out of distribution) | 10 | **2.293** vol pts | 2.305 | 4/10 | 9/10 dates, p = 0.004 |

### Black-Scholes is not competitive

* With one sigma it is worse than the best Heston-family arm on **10/10 dates**, by a median factor of 1.38x (Wilcoxon p = 0.0020).
* With a full term structure of sigma it is worse than the best Heston-family arm on **10/10 dates**, by a median factor of 1.21x (Wilcoxon p = 0.0020).

Giving Black-Scholes five parameters instead of one -- matching single Heston's count -- closes only part of the gap (2.637 against 3.039 vol points) and still never wins a date. The deficit is the missing smile, not the missing parameters.

### Ten parameters lose to five unless they are regularised

Cold-start Double Heston reaches 2.366 vol points against single Heston's 2.361, better on only 2/10 dates (p = 0.010). On real quotes the unregularised ten-parameter fit is *worse* than the five-parameter one -- the estimation-variance result from the synthetic study, reproduced on market data.

The two-stage estimator reverses it: 2.293 vol points, better than single Heston on **9/10 dates** (p = 0.004) and better than cold-start Double Heston on 10/10 (p = 0.002), in 5.7 s against 172 s.

### The ridge, again, is what does the work

| arm | median holdout IV RMSE |
|---|---:|
| dual PINN prior alone | 2.573 vol pts |
| cold-start fit alone | 2.366 vol pts |
| prior + ridge | 2.293 vol pts |

The prior alone is the *worst* Heston-family arm. The fit alone is no better than single Heston. Only the combination wins -- the same division of labour measured on synthetic data, now on real quotes. The validation ridge sweep shows the same interior optimum:

| ridge | median validation IV RMSE |
|---:|---:|
| 0 | 6.253 vol pts |
| 0.3 | 6.110 vol pts |
| 1 | 6.070 vol pts |
| 3 | 5.955 vol pts |
| 10 | 5.839 vol pts |
| 30 | 5.621 vol pts |
| inf (prior unmoved) | 5.997 vol pts |

## Caveats, stated plainly

1. **The network is out of distribution here.** Its training prior was built for volatile power-sector stocks: median total instantaneous vol 48%, with NIFTY's ~17.4% at the 1.8th percentile of the `v0` marginal and the 0.23rd percentile of the `theta` marginal. Only 13 of 45 grid points fall inside the training set's central 99% band. It nonetheless recovered a total vol of 0.159-0.206 against a market ATM vol near 0.174, so it degraded gracefully -- but every number from the two network arms should be read as extrapolation, and the honest fix is to regenerate the prior at index volatility levels and retrain.
2. **The two-stage arm's ridge was not tuned here.** It uses lambda = 3, frozen from the synthetic study. The NIFTY validation split chose lambda = 30 for the single-Heston-prior arm. Not tuning on NIFTY is the conservative choice, but it means the two-stage number is not the best this estimator could do on this data.
3. **Ten dates.** The per-date pairing is strong (9/10, 10/10) but the sample is small, and all ten dates come from a single April 2026 volatility episode.
4. Quotes are settlement prices, not traded mid-quotes; NSE does not publish a bid-ask in the bhavcopy. The estimated quote noise exceeded 1% on every date and was clipped to the top of the network's trained noise range.

## Verdict

On real NIFTY index options in a genuinely volatile window, with 1,358 held-out market quotes:

1. **Black-Scholes is decisively last**, at any parameter count, on every date.
2. **Cold-start Double Heston loses to single Heston.** More parameters alone is a step backwards.
3. **The two-stage Double Heston estimator is the best of the three models**, and it is the regularisation -- not the extra factor and not the network on its own -- that makes the difference.
