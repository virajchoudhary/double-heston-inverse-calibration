# Fully Disclosed Single- versus Double-Heston Comparison

## Honest result

On **8,814 common finite authentic NSE-derived test quotes**, Single Heston has IV RMSE **0.044200** and Double Heston has IV RMSE **0.045510**. Double Heston beats Single Heston for **2 of 11 stocks**: JSWENERGY, TORNTPOWER.

The source test universe contains **8,818** rows. Double Heston returned invalid implied values for **4** rows, all retained in `double_heston_test_prediction_failures.csv`; common finite coverage is **99.9546%**. No replacement values were fabricated. Candidate selection likewise used only validation quote keys with finite predictions from all five candidates, and the excluded validation failures are separately retained.

The stock-session cluster-bootstrap 95% interval for `Double RMSE - Single RMSE` is [0.000607, 0.001993]. A negative interval favors Double Heston. The interval is entirely positive. The historical model preference is therefore **Single Heston**.

**Recommendation:** retain Single Heston and reject the current Double-Heston calibration for production use. Double Heston may be reconsidered only after a locked future-date evaluation shows a stable, material improvement.

## Parameter selection and leakage controls

Each stock had five Double-Heston parameter candidates fitted only on chronological training dates: three independent optimizer starting points on all selected training surfaces plus two fits on disjoint alternating training-date halves. Validation mean session IV RMSE selected one candidate per stock. Parameters and the selection rule were frozen before test scoring. The two models were then compared on exactly the same row keys.

The Double-Heston factor order is fixed by `kappa_slow < kappa_fast`. Both factors satisfy the Feller inequality, and `rho_slow² + rho_fast² < 1` enforces a valid joint spot/factor correlation structure. These constraints reduce invalid solutions but do not prove parameter identifiability.

The separate controlled-test JSON records exact and 1%-noise synthetic recovery. Synthetic rows are never mixed with NSE results.

## What cannot honestly be guaranteed

No finite historical experiment can prove zero overfitting or a global optimum. There were **55** boundary-near candidate fits, and the parameter-stability table retains the variation across starts and training halves. Similar prices can still arise from materially different Double-Heston parameters.

The historical test dates had already been viewed during the earlier Single-Heston analysis. No test price entered this Double-Heston optimizer or validation selector, but the comparison is still retrospective researcher-level evidence rather than a pristine final holdout. The code and selected parameters must now be locked and evaluated on future authentic NSE sessions before any live generalisation claim.

## Data authenticity

- Input SHA-256: `a8a56dd7b17074d8fa32f88936c8b404456f22f296d9dc4665faf3b13621f1d3`
- Model-ready source rows: 215,636
- Test source-universe rows: 8,818
- Common finite comparison rows: 8,814
- Integrity checks passed: 20/20

The CSV predictions are model-generated outputs linked to authentic NSE row keys and source paths. They are never described as original exchange observations. Exact expiry dates are used directly; no Tuesday/Thursday expiry assumption is made.
