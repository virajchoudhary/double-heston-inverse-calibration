# NTPC Mentor Checkpoint

## What was done

NTPC was selected as the Power-sector primary at moderate confidence in Stage A. The pilot used only official NSE observations on **2026-07-15** from `BhavCopy_NSE_FO_0_0_0_20260715_F_0000.csv` (SHA-256 `5FCD3437CFA4D6C489608687D790BF26F11F8B0D02D088CF907D5A005A360FD5`): spot **INR 344.35**, actual expiries **2026-07-28 / 2026-08-25 / 2026-09-29**, DTE **13 / 41 / 76**, and `T=DTE/365`.

The primary price is active-row `ClsPric`. Near/middle rows were restricted to `abs(log(K/S))<=0.10`; unique nearest listed strikes to `[-0.10,-0.05,0,+0.05,+0.10]` were selected separately by expiry/type under a strict 0.05 target-distance gate. Inner targets formed 12 calibration rows. The unsupported near-call negative-outer target was left unmatched, so the untouched holdout has 7 rows. The far expiry had no traded option rows and was excluded.

Market IV solves the forward-Black equation from observed prices. Carry uses active matched NTPC futures and a declared flat proxy based on the official RBI 91-day T-bill auction YTM 5.3324% from Press Release 2026-2027/672 dated 2026-07-15. The 13- and 41-day rates are short-end extrapolations; no exact daily zero curve was acquired. The successful official HTML and dated field extract are hash-sealed in the manifest. Futures-implied `q` is not called an observed dividend yield.

## Results

| model | calibration_price_rmse | holdout_price_rmse | calibration_relative_price_error_mean | holdout_relative_price_error_mean | calibration_iv_rmse | holdout_iv_rmse | runtime_seconds |
|---|---|---|---|---|---|---|---|
| BLACK_SCHOLES | 0.326902 | 1.05334 | 0.049873 | 0.268419 | 0.0118538 | 0.0804675 | 0.0118729 |
| HESTON | 0.238578 | 0.910569 | 0.0300421 | 0.204466 | 0.00790466 | 0.0729962 | 41.2127 |
| DOUBLE_HESTON | 0.234321 | 0.926825 | 0.039357 | 0.20087 | 0.00778971 | 0.0725916 | 228.607 |

Classification: **NO_CLEAR_WINNER**.

Black-Scholes fitted one common volatility, `sigma_BS=0.22412887`. Double Heston has the smallest calibration RMSE; Heston has the smallest holdout price RMSE, but its advantage over Double Heston is below the predeclared 5% margin and its holdout IV RMSE is slightly worse. Therefore no clear repricing winner is declared.

Best-fit Heston: `{'kappa': 4.748556755795085, 'theta': 0.110982744138846, 'sigma': 1.02663092773475, 'rho': 0.0506945716172238, 'v0': 0.0399547445627715}`.

Best-fit canonical Double Heston: `{'kappa_slow': 2.625533262486748, 'theta_slow': 0.139993794100438, 'sigma_slow': 0.8566378479188727, 'rho_slow': 0.6762760209383789, 'v0_slow': 0.0134334621687306, 'kappa_fast': 11.763863089764897, 'theta_fast': 0.0241024135565524, 'sigma_fast': 0.7521362932264983, 'rho_fast': -0.666180486338069, 'v0_fast': 0.0272416814109967}`.

`v0_slow=0.01343346`, `v0_fast=0.02724168`, `v0_total=0.04067514`. At each horizon, expected `v_total(t)=v_slow(t)+v_fast(t)` inside one joint characteristic function; Heston option prices are not added. Multi-start stability: **UNSTABLE_MATERIAL_MULTI_START_DISPLACEMENT**.

The selected Heston and Double Heston iterates both reached the declared `max_nfev=160` cap without a SciPy convergence termination. They are finite capped best fits, not convergence or parameter-identification evidence.

| component | kappa | half_life_years | half_life_calendar_days |
|---|---|---|---|
| Heston | 4.74856 | 0.14597 | 53.2791 |
| Double Heston slow | 2.62553 | 0.264002 | 96.3609 |
| Double Heston fast | 11.7639 | 0.0589217 | 21.5064 |

## Volatility checkpoint

| expiry_date | DTE | T | market_atm_iv | bs_predicted_volatility | heston_predicted_average_volatility | double_heston_predicted_average_volatility | actual_ex_post_realized_volatility | actual_status | bs_absolute_realized_volatility_error | heston_absolute_realized_volatility_error | double_heston_absolute_realized_volatility_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-28 | 13 | 0.0356164 | 0.208928 | 0.224129 | 0.213627 | 0.214097 | 0.157747 | COMPLETE | 0.066382 | 0.0558797 | 0.0563505 |
| 2026-08-25 | 41 | 0.112329 | 0.222645 | 0.224129 | 0.236506 | 0.237136 | n/a | UNAVAILABLE_FUTURE_EXPIRY_AS_OF_2026-08-12 | n/a | n/a | n/a |
| 2026-09-29 | 76 | 0.208219 | n/a | 0.224129 | 0.256656 | 0.260331 | n/a | UNAVAILABLE_FUTURE_EXPIRY_AS_OF_2026-08-12 | n/a | n/a | n/a |

The near expiry has 9 official close-to-close returns and a completed annualized realized volatility of 0.157747. **HESTON** has the smallest near-horizon numerical absolute difference. However, Heston/Double Heston values are option-implied risk-neutral (`Q`) expected-average volatilities and realized volatility is a physical-measure (`P`) outcome. No variance-risk-premium mapping was estimated, so no physical forecasting winner is claimed. Middle/far actual volatility remains unavailable because the expiries had not occurred as of 2026-08-12.

## Unresolved

No bid/ask history, no far-expiry active option sample, only one completed realized horizon, and no known real NTPC Double Heston parameter truth. The best-fit vector is a calibration result, not truth.
