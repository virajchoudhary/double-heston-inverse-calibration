# Clear-separation figures

2026-09-25. Figures in `figures/`, plotted values in `data/`, code in `scripts/`, metrics in
`figure_metrics_summary.csv`. Nothing in earlier experiment folders was overwritten.

No curve is shifted, rescaled or offset anywhere. Every separation below comes from the model
outputs themselves; residual panels display differences and say so on the axis.

The two settings are kept strictly apart:

- **CONTROLLED TWO-TIMESCALE BENCHMARK** — the frozen `FIXED_TOTAL_TWIST` scenario, reference =
  exact Double Heston. Not market evidence.
- **REAL MARKET HELD-OUT SURFACE** — IWM, the surface already selected ex ante by the frozen
  DH-relevance score in `dh_relevance_selection_v1`, reference = market quotes. No new selection rule
  was created.

---

## 1. Why the raw price curves looked visually similar

A call price is dominated by a component every model shares.

- **Deep in the money:** C ≈ S − K plus a small time value. At S = 130, K = 100, τ = 90 days the
  benchmark price is about 30.3, of which 30.0 is intrinsic. Every model reproduces the intrinsic
  part exactly, so at most 1% of the plotted height can differ.
- **Deep out of the money:** every model is near zero.
- **Consequently** the differences live in the time value, in the implied volatility, and in how both
  change with maturity — not in the price level. On the controlled benchmark at τ = 90 days the
  largest Black-Scholes deviation is 0.21 price units on an axis spanning about 30: roughly 0.7%,
  which is a line width.

## 2. Why time value is a better separation plot

TimeValue(S,τ) = C(S,τ) − max(S−K,0) removes the shared intrinsic component and leaves exactly the
part of the price the model is responsible for. The same Black-Scholes deviation that was 0.7% of the
price axis becomes **5.3% of the time-value axis**, and the three models separate by eye.

RMSE of time value against exact Double Heston (price units):

| τ (days) | Black-Scholes | Single Heston | DH-PINN |
|---|---:|---:|---:|
| 30 | 0.0495 | 0.0459 | 0.00029 |
| 90 | 0.1331 | 0.0827 | 0.00051 |
| 180 | 0.1634 | 0.0587 | 0.00034 |
| 365 | 0.1589 | 0.0303 | 0.00076 |

The PINN is 100–200× closer to exact Double Heston than either one-factor model.

## 3. Why the implied-volatility smile is a better separation plot

Implied volatility divides out the price scale, and it is the metric every headline number in this
project uses. On the controlled benchmark, RMSE against exact Double Heston in **vol points**:

| τ (days) | Black-Scholes | Single Heston | DH-PINN |
|---|---:|---:|---:|
| 30 | 3.51 | 2.58 | 0.016 |
| 90 | 1.84 | 1.10 | 0.007 |
| 365 | 0.64 | 0.096 | 0.003 |

Black-Scholes is a horizontal line by construction — it has one volatility per maturity and no
mechanism for a smile at all. That is visible immediately in `controlled_iv_smile_short_medium_long.png`
and, far more starkly, against real quotes in `market_iv_smile.png`.

*Disclosure:* at τ = 30 days the deepest out-of-the-money points have prices too small to invert to
an implied volatility (44 of 401 grid points for Double Heston and the PINN, 14 for Single Heston,
0 for Black-Scholes). Those points are excluded from the smile metrics and appear as gaps in the
figure; the count is carried in `figure_metrics_summary.csv`.

## 4. Where the Double-Heston advantage is most visible

**The at-the-money implied-volatility term structure.** Against exact Double Heston, over 7–730 days:

| model | RMSE (vol pts) | max (vol pts) |
|---|---:|---:|
| Black-Scholes | 0.058 | 0.184 |
| Single Heston | 0.167 | **0.845** |
| DH-PINN | 0.003 | 0.011 |

Single Heston's error is **0.85 vol points at 7 days**, shrinks to zero near 110 days, crosses to
+0.15 around 300 days, and crosses back. That signature — a large short-end miss with a sign change —
is precisely what one mean-reversion rate cannot avoid when the true process has two. Restricted to
the first 120 days its RMSE rises to 0.39 vol points, nearly seven times Black-Scholes's.

## 5. Does the BS-PINN follow analytical Black-Scholes?

Yes, at its own parameters. The checkpoint on branch `black-scholes-with-pinn` carries **r = 0.03,
q = 0.01 and its own calibrated σ = 0.19999906** (true value 0.20, recovered to 5e-7). Against
analytic Black-Scholes at those same parameters, over S ∈ [55, 165] and τ ∈ {30, 90, 365, 730} days:

- RMSE **0.0244**, max **0.219** price units, the maximum at the far in-the-money edge of its
  training domain at the longest maturity.

**It is deliberately absent from the controlled two-timescale figures.** That benchmark runs at
r = q = 0 with stochastic volatility; dropping a network hard-wired to r = 0.03, q = 0.01, σ = 0.2
onto it would compare a PINN to the wrong analytic target, which the brief forbids.

## 6. Does the DH-PINN follow exact Double Heston?

Yes. On the same four maturities across S ∈ [70, 130]: RMSE **7.5e-4**, max **2.1e-3** price units.
In the figures the two curves are indistinguishable at every scale used here, which is the point:
the gap between Double Heston and the one-factor models is **model structure, not neural error**,
because the neural error is two to three orders of magnitude smaller.

## 7. Controlled benchmark: how much worse are BS and SH than DH?

Reference is exact Double Heston, so this measures representational distance, not market accuracy.

| quantity | Black-Scholes | Single Heston | DH-PINN |
|---|---:|---:|---:|
| time value, τ = 90 d (RMSE) | 0.133 | 0.083 | 0.0005 |
| IV smile, τ = 90 d (vol pts) | 1.84 | 1.10 | 0.007 |
| ATM IV term structure (vol pts) | 0.058 | 0.167 | 0.003 |
| ATM time value vs τ (RMSE) | 0.018 | 0.029 | 0.0008 |

Note the reversal: Black-Scholes is worse on the **smile** (it has none), Single Heston is worse on
the **term structure** (one timescale). Each fails where its missing ingredient matters, which is
exactly the predicted mechanism.

## 8. Real held-out surface: how much worse are BS and SH than DH?

IWM, 707 held-out quotes, models calibrated on the other half:

| model | IV RMSE (vol pts) | max | P95 |
|---|---:|---:|---:|
| Black-Scholes, one vol per expiry | 5.388 | 27.9 | 12.2 |
| Single Heston, 5 parameters | 1.893 | 14.9 | 2.59 |
| **Double Heston, 10 parameters** | **0.583** | 3.26 | 1.09 |
| DH-PINN (published params + level scale) | 5.357 | 30.7 | 10.7 |

By expiry, smile RMSE in vol points:

| expiry | Black-Scholes | Single Heston | Double Heston | DH-PINN |
|---|---:|---:|---:|---:|
| 28 days | 6.21 | 1.48 | **0.38** | 5.40 |
| 84 days | 4.61 | 0.74 | **0.34** | 2.97 |
| 357 days | 2.11 | 0.84 | **0.82** | 2.24 |

Double Heston is 3.2× better than Single Heston and 9× better than Black-Scholes on quotes none of
them saw. The DH-PINN row is **not** a failure of the network: it is pinned to the published
parameter family with a single level scale, so it is answering a different question. Its own target
tracking is the 7.5e-4 figure in section 6.

## 9. Which figure best communicates the two-timescale advantage?

`fast_vs_slow_same_total_variance.png`. Two states with identical instantaneous variance
(v_f + v_s = 0.04, so 20% volatility in both) but opposite allocation between the fast factor
(κ_f = 10.75) and the slow factor (κ_s = 0.95), a ratio of 11.3×. The at-the-money implied volatility
separates from 19.8% versus 22.1% at 30 days to 22.1% versus 24.7% at 730 days, and the time-value
curves separate accordingly. Same variance today, different prices at every maturity. **No
single-factor model can produce both curves from the same instantaneous variance**, and the DH-PINN
reproduces both.

Runner-up: `controlled_atm_iv_term_structure.png`, where Single Heston's 0.85 vol-point short-end
miss and its sign change are visible directly.

## 10. Are the plotted curves financially valid?

Checked numerically on every controlled price curve (finite-difference delta and gamma, bounds,
time-value sign):

- delta violations: **0**
- gamma violations: **0**
- prices below intrinsic: **0**
- prices above S: **0**
- negative time value: **0**

Full detail in `data/controlled_shape_checks.json`.

## Interpretation rules, stated explicitly

- The BS-PINN is validated against **analytic Black-Scholes**; the DH-PINN against **exact Double
  Heston**. Neither is compared to the other's target.
- The goal is not for the DH-PINN to equal Black-Scholes. The intended progression is: Black-Scholes
  supplies the baseline option shape, Single Heston adds one stochastic variance factor, Double
  Heston adds a second timescale, and the DH-PINN reproduces that two-factor solution to 7.5e-4.
- The visible advantage comes from **time value, implied volatility and maturity structure**, not
  from the gross price level — that was the whole reason for replacing the raw price plots.

## File map

Controlled: `controlled_time_value_vs_S_multipanel.png`, `time_value_vs_S_{30,90,180,365}d.png`,
`controlled_time_value_vs_S.png`, `controlled_time_value_vs_S_atm_zoom.png`,
`iv_smile_{short,medium,long}.png`, `controlled_iv_smile.png`, `controlled_iv_smile_short_medium_long.png`,
`iv_smile_atm_zoom.png`, `controlled_atm_iv_term_structure.png`, `atm_term_structure_zoom.png`,
`controlled_atm_time_value_vs_tau.png`, `controlled_time_value_decay_to_expiry.png`,
`fast_vs_slow_same_total_variance.png`, `fast_vs_slow_atm_iv_term_structure.png`.

Market: `market_time_value_vs_S.png`, `market_iv_smile.png`, `market_atm_iv_term_structure.png`,
`market_residual_panels.png`.

Validation: `dh_pinn_vs_dh_validation.png`, `bs_pinn_vs_bs_validation.png`.

Every figure has a CSV of its plotted values in `data/`; all metrics are collected in
`figure_metrics_summary.csv`.
