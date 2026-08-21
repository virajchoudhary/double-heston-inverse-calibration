# NTPC Single-Stock Calibration Pilot

## Decision boundary

This is a bounded real-market calibration pilot for **NTPC only**, valued on **2026-07-15**. It compares Black-Scholes, Standard Heston, and the canonical ten-parameter Double Heston engine. The Double Heston vector is a **BEST-FIT NTPC CALIBRATION UNDER THE DECLARED CONTRACT**, not true NTPC parameters.

The model-comparison classification is **NO_CLEAR_WINNER**. In the completed near-expiry arithmetic diagnostic, **HESTON** has the smallest absolute difference from realized volatility. The Heston quantities are risk-neutral (`Q`) expected-average variances inferred from option prices, whereas realized volatility is a physical-measure (`P`) outcome. No variance-risk-premium mapping was estimated, so this is not a validated physical-volatility forecast or a general forecasting-winner claim. The middle and far expiries were after the fixed 2026-08-12 as-of date and have no fabricated ex-post result.

## Official market and carry contract

- Official NSE F&O UDiFF source: `BhavCopy_NSE_FO_0_0_0_20260715_F_0000.csv` (SHA-256 `5FCD3437CFA4D6C489608687D790BF26F11F8B0D02D088CF907D5A005A360FD5`).
- Spot: official NTPC EQ close and matching F&O `UndrlygPric`, **INR 344.35**.
- Actual expiries: 2026-07-28, 2026-08-25, 2026-09-29; DTE `13/41/76`; `T=DTE/365`.
- Primary price: official `ClsPric`, admitted only when `TtlTradgVol>0`, `TtlNbOfTxsExctd>0`, and `OpnIntrst>0`.
- Primary domain: near and middle expiries and exact `abs(log(K/S)) <= 0.10`. Far option volume was zero in all 26 rows and the far expiry is diagnostic-only.
- Deterministic unique strike matching minimizes total absolute distance to targets `[-0.10,-0.05,0,+0.05,+0.10]` separately by expiry and option type, with a strict maximum target distance of 0.05. Inner three targets are calibration; outer two are holdout. The near-call negative-outer target had no qualifying active strike and was left unmatched, giving the smallest fail-closed alternative: 12 calibration and 7 holdout rows.
- Risk-free source: [official RBI Treasury-bill auction result](https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63150), Press Release 2026-2027/672 dated 2026-07-15, showing the 91-day T-bill cut-off price **98.6880** and YTM **5.3324%**. The declared contract holds that simple yield flat across the three expiries: this is a short-end extrapolation for 13 and 41 days and a 91-day proxy for 76 days, not an acquired daily zero curve. It gives `D(T)=1/(1+yT)` and the maturity-equivalent continuous rate `-log(D)/T`.
- Rate provenance: the successful dated official HTML response and normalized field extract `raw/rbi_91_day_tbill_observation_20260715.json` are both hash-sealed in the manifest.
- The separately preserved `raw/rbi_current_rates_archive_20260715.html` is an earlier RBI perimeter-challenge response. It is retained and hash-listed for audit disclosure only, is not numerical evidence, and is not used by the carry contract.
- Carry: matched active NTPC futures close by actual expiry. `q=r-log(F/S)/T`; this is labelled futures-implied carry, not an observed dividend yield.

| expiry_date | DTE | T | futures_close | discount_factor | continuous_rate | futures_implied_carry |
|---|---|---|---|---|---|---|
| 2026-07-28 | 13 | 0.0356164 | 345.55 | 0.998104 | 0.0532734 | -0.0443997 |
| 2026-08-25 | 41 | 0.112329 | 345.85 | 0.994046 | 0.0531649 | 0.0144698 |
| 2026-09-29 | 76 | 0.208219 | 345.75 | 0.989019 | 0.0530301 | 0.033544 |

## Data selection and market IV

Raw NTPC option rows: **146**. Retained rows: **19**. Every excluded row and reason is in the ignored detailed CSV. Market IV is the robust bracketed solution of the forward-Black equation; impossible prices are rejected, never clipped.

| sample_role | expiry_date | option_type | target_log_moneyness | strike | log_moneyness | observed_price | market_implied_volatility |
|---|---|---|---|---|---|---|---|
| CALIBRATION | 2026-07-28 | call | -0.05 | 330 | -0.0425659 | 16.5 | 0.222629 |
| CALIBRATION | 2026-07-28 | call | 0 | 345 | 0.00188584 | 5.7 | 0.208948 |
| CALIBRATION | 2026-07-28 | call | 0.05 | 362.5 | 0.0513659 | 1 | 0.226842 |
| CALIBRATION | 2026-07-28 | put | -0.05 | 327.5 | -0.0501705 | 0.55 | 0.211003 |
| CALIBRATION | 2026-07-28 | put | 0 | 345 | 0.00188584 | 5.15 | 0.208908 |
| CALIBRATION | 2026-07-28 | put | 0.05 | 360 | 0.0444454 | 15.35 | 0.199253 |
| CALIBRATION | 2026-08-25 | call | -0.05 | 330 | -0.0425659 | 20 | 0.228396 |
| CALIBRATION | 2026-08-25 | call | 0 | 345 | 0.00188584 | 11.15 | 0.233604 |
| CALIBRATION | 2026-08-25 | call | 0.05 | 360 | 0.0444454 | 5.4 | 0.235368 |
| CALIBRATION | 2026-08-25 | put | -0.05 | 330 | -0.0425659 | 4.3 | 0.22989 |
| CALIBRATION | 2026-08-25 | put | 0 | 345 | 0.00188584 | 9.3 | 0.211685 |
| CALIBRATION | 2026-08-25 | put | 0.05 | 350 | 0.0162746 | 12.6 | 0.225116 |
| HOLDOUT | 2026-07-28 | call | 0.1 | 380 | 0.0985127 | 0.3 | 0.284714 |
| HOLDOUT | 2026-07-28 | put | -0.1 | 315 | -0.0890859 | 0.95 | 0.366007 |
| HOLDOUT | 2026-07-28 | put | 0.1 | 380 | 0.0985127 | 35.1 | 0.340607 |
| HOLDOUT | 2026-08-25 | call | -0.1 | 320 | -0.0733376 | 30 | 0.303455 |
| HOLDOUT | 2026-08-25 | call | 0.1 | 380 | 0.0985127 | 1.95 | 0.249024 |
| HOLDOUT | 2026-08-25 | put | -0.1 | 315 | -0.0890859 | 1.95 | 0.255047 |
| HOLDOUT | 2026-08-25 | put | 0.1 | 370 | 0.0718444 | 27 | 0.233592 |

## Model comparison

The winner rule requires at least a 5% holdout-price-RMSE improvement over the runner-up and no worse holdout IV RMSE. Otherwise the result is `NO_CLEAR_WINNER`.

| model | parameter_count | calibration_price_rmse | holdout_price_rmse | calibration_price_mae | holdout_price_mae | calibration_relative_price_error_mean | holdout_relative_price_error_mean | calibration_iv_rmse | holdout_iv_rmse | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| BLACK_SCHOLES | 1 | 0.326902 | 1.05334 | 0.271631 | 0.81641 | 0.049873 | 0.268419 | 0.0118538 | 0.0804675 | 0.0118729 |
| HESTON | 5 | 0.238578 | 0.910569 | 0.163483 | 0.631472 | 0.0300421 | 0.204466 | 0.00790466 | 0.0729962 | 41.2127 |
| DOUBLE_HESTON | 10 | 0.234321 | 0.926825 | 0.16415 | 0.642102 | 0.039357 | 0.20087 | 0.00778971 | 0.0725916 | 228.607 |

## Best-fit parameters and mean reversion

| parameter | model | value |
|---|---|---|
| kappa | HESTON | 4.74856 |
| theta | HESTON | 0.110983 |
| sigma | HESTON | 1.02663 |
| rho | HESTON | 0.0506946 |
| v0 | HESTON | 0.0399547 |
| kappa_slow | DOUBLE_HESTON | 2.62553 |
| theta_slow | DOUBLE_HESTON | 0.139994 |
| sigma_slow | DOUBLE_HESTON | 0.856638 |
| rho_slow | DOUBLE_HESTON | 0.676276 |
| v0_slow | DOUBLE_HESTON | 0.0134335 |
| kappa_fast | DOUBLE_HESTON | 11.7639 |
| theta_fast | DOUBLE_HESTON | 0.0241024 |
| sigma_fast | DOUBLE_HESTON | 0.752136 |
| rho_fast | DOUBLE_HESTON | -0.66618 |
| v0_fast | DOUBLE_HESTON | 0.0272417 |

At calibration time, `v0_total = v0_slow + v0_fast = 0.04067514`. Expected total variance is `v_total(t)=v_slow(t)+v_fast(t)`. Prices are produced by one joint characteristic function; Heston prices are not added.

| component | kappa | half_life_years | half_life_calendar_days |
|---|---|---|---|
| Heston | 4.74856 | 0.14597 | 53.2791 |
| Double Heston slow | 2.62553 | 0.264002 | 96.3609 |
| Double Heston fast | 11.7639 | 0.0589217 | 21.5064 |

Double Heston multi-start stability: **UNSTABLE_MATERIAL_MULTI_START_DISPLACEMENT**; near-equivalent starts `12`, materially displaced `11`, maximum full-range-scaled distance `0.491085`. Optimizer convergence is not treated as parameter-identification evidence.

The selected Heston and Double Heston best iterates both reached the declared `max_nfev=160` cap without a SciPy convergence termination. They remain valid finite capped iterates and are reported as such; this further limits parameter interpretation.

## Predicted versus actual volatility

Here “predicted” follows the experiment's mechanical formula, not a claim of a physical-measure forecast: BS is the fitted option-implied constant and Heston/Double Heston are `Q`-measure expected-average volatilities from option-calibrated parameters. Comparing them with ex-post `P`-measure realized volatility is descriptive because no variance-risk-premium or `Q`-to-`P` mapping was fitted.

| expiry_date | DTE | T | market_atm_iv | bs_predicted_volatility | heston_predicted_average_volatility | double_heston_predicted_average_volatility | actual_ex_post_realized_volatility | actual_status | bs_absolute_realized_volatility_error | heston_absolute_realized_volatility_error | double_heston_absolute_realized_volatility_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-28 | 13 | 0.0356164 | 0.208928 | 0.224129 | 0.213627 | 0.214097 | 0.157747 | COMPLETE | 0.066382 | 0.0558797 | 0.0563505 |
| 2026-08-25 | 41 | 0.112329 | 0.222645 | 0.224129 | 0.236506 | 0.237136 | n/a | UNAVAILABLE_FUTURE_EXPIRY_AS_OF_2026-08-12 | n/a | n/a | n/a |
| 2026-09-29 | 76 | 0.208219 | n/a | 0.224129 | 0.256656 | 0.260331 | n/a | UNAVAILABLE_FUTURE_EXPIRY_AS_OF_2026-08-12 | n/a | n/a | n/a |

Official NSE CM closes from 2026-07-15 through the completed 2026-07-28 near expiry supplied **9** log returns. The official NSE corporate-actions API returned zero NTPC actions for that window, so the official EQ close series was used without adjustment. Middle/far realized volatility is unavailable because those expiries were future dates at the fixed as-of date.

## Figures

- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/01_strike_support.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/02_market_iv_smile.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/03_market_vs_predicted_price.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/04_residual_vs_moneyness.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/05_market_iv_vs_model_iv.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/06_variance_decomposition.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/07_kappa_half_life.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/08_predicted_vs_realized_volatility.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/09_double_heston_multistart_stability.png`
- `market_data_audit/stage_a/derived/ntpc_single_stock_pilot/figures/10_model_comparison.png`

## Remaining limitations

- Historical bid/ask and quote sizes are unavailable in the free NSE UDiFF files; close-price activity filtering does not recreate them.
- Calls and puts are not independent after carry is fixed; retaining both exposes observed parity/microstructure differences but does not double structural information.
- The far expiry is not calibrated because every far NTPC option row had zero volume and zero executed trades.
- Only the near realized-volatility horizon is complete as of 2026-08-12.
- Real-market parameter truth is unknown; price fit and optimizer success do not validate the ten NTPC parameters.
