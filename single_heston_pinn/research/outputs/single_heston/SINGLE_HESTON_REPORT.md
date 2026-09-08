# Leakage-safe single-Heston result

## Scope and authenticity

- Input: `model_input_option_prices.csv`
- Input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`
- Authentic model-ready rows examined: 215,636 across 11 power-sector stocks
- Honest model-ready period: 8 July 2024 through 3 August 2026
- The ten-year raw archive was **not** described as ten years of Heston-ready data; older rows lack an exchange-published spot value.
- Observed NSE prices were not smoothed, changed, filled, or synthesized. Invalid/noisy records were excluded and counted.

## Model and leakage barrier

This is the canonical one-factor, five-parameter Heston model. Dates were split chronologically 70/15/15 independently for each stock. Four structural parameters (`kappa`, `theta`, `sigma`, `rho`) were estimated only from the earliest 70%. A date-specific `v0` was then fitted on designated calibration strikes.

An entire strike—both its call and put—was assigned either to calibration or holdout. Holdout-strike prices were excluded from put-call-parity carry estimation, structural fitting, and `v0` fitting. They were used once for evaluation. A fresh one-to-one join independently confirmed that all 1,547 final test rows, raw prices, dates, symbols and source filenames match the original model-input CSV exactly.

## Genuine held-out results

| Measure | Test result |
|---|---:|
| Held-out quotes | 1,547 |
| IV RMSE | 0.028112 |
| IV MAE | 0.015564 |
| IV bias | -0.001175 |
| IV R-squared | 0.904226 |
| Raw-price RMSE (INR) | 1.109087 |
| Raw-price MAE (INR) | 0.473421 |

Aggregate test calibration-strike IV RMSE was 0.030469, while untouched holdout-strike IV RMSE was 0.028112. The absence of a worse holdout result is evidence against observed strike-level overfitting; it is not a guarantee about future regimes.

The primary comparison graph uses **market-implied volatility calculated from authentic NSE closing prices** versus Heston-implied volatility. Volatility itself is not directly published as an observed field, so calling market IV “actual volatility” without this qualification would be misleading.

The separate realized-volatility graph compares causal trailing 20-session realized volatility with `sqrt(v0)`. On test states their correlation is 0.451, RMSE is 0.118 and MAE is 0.092. This is materially weaker than option-IV capture and is reported rather than hidden.

## Noise isolation

The processed-date audit rejected 899 expiry groups with unreliable anchor-only put-call parity, 647 with fewer than six paired strikes, 299 outside the 7–180 day maturity window, 78 dates with too few clean quotes, and 16 individual quotes failing IV/moneyness/vega rules. A further 29,581 unpaired or duplicate-side rows were not used. Exclusion never altered an observed price.

## Parameter-identifiability warning

All parameters are finite, positive where required, have `|rho| < 1`, and satisfy the imposed Feller condition. However, nine stocks' `kappa` estimates are near the search ceiling, ten stocks' `sigma` estimates are near the Feller cap, and CESC's `theta` is near its ceiling. These boundary contacts mean individual structural parameters are not strongly identified by these short-dated surfaces. The held-out pricing/IV results remain valid; the parameters must not be presented as uniquely recovered economic truth.

CESC is the weakest individual test result (IV RMSE 0.041791 and R-squared -0.421). The overall result must not be used to conceal that stock-level failure.

## Controlled numerical tests

Synthetic prices are isolated in `single_heston_controlled_tests.json` and are never mixed with NSE results. With known parameters, all five were recovered to a maximum relative error of `3.22e-10`. With deterministic 1% price noise, the recovered price RMSE was 0.02233. Three materially different starting values converged to essentially the same optimum in both tests.

## Files

- `actual_vs_heston_iv.png`: requested genuine market-IV versus Heston-IV graph
- `realized_vs_heston_spot_vol.png`: causal historical-realized versus Heston-state graph
- `surfaces/*_heston_surface.png`: one surface for each of 11 stocks, with authentic held-out market points
- `single_heston_predictions.csv`: row-level calibration and holdout results with NSE row keys and source files
- `single_heston_surface_grid.csv`: explicitly labelled model-generated surface grid
- `single_heston_parameters.csv`, `single_heston_daily_state.csv`, `single_heston_metrics.csv`: fitted outputs and metrics
- `single_heston_noise_audit.csv`, `single_heston_checks.csv`: exclusions and strict validity checks
- `single_heston_parameter_diagnostics.csv`: non-fatal parameter-identifiability warnings
- `single_heston_controlled_tests.json`: exact/noisy/start-value controlled tests, separate from market data

