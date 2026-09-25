# Amendment 01: Black-Scholes term structure grouped by expiry

Written and hashed **before** any corrected Black-Scholes fit was run. The frozen v1 files (`engine.py`, `run.py`, `config.json`, …) are not modified. The original BS_TERM outputs (`artifacts/fits/*/*/BS_TERM.json`, `artifacts/test_scores.csv` and `artifacts/results.json`) are preserved as the flawed v1 record.

## Defect

`data.clean()` computes each quote's maturity from its own trade timestamp, so quotes in the same expiry have slightly different `tau` values. `engine.fit_bs_term()` uses `np.unique(tau)` as its knots. That gives **one volatility per quote**, not one per expiry: 139 knots for 8 expiries on 2026-02-11. The consequences:

- BS_TERM reproduced its calibration quotes exactly: in-sample IV RMSE had a median of 0.0.
- Its held-out predictions interpolated total variance between individual calibration quotes rather than between expiries.

That is not the predeclared model, which PROTOCOL.md calls "one volatility per expiry for Black-Scholes". The Heston fits do not use knots and are unaffected.

## Correction: the only change

`BS_EXPIRY` groups calibration quotes by the `expiry` column.

- For each calibration expiry: one volatility, minimising the same vega-weighted objective over the same bounds `bs_vol`. The knot is the median `tau` of that expiry's calibration quotes.
- **Prediction:**
  - A quote in a calibrated expiry uses that expiry's volatility at its own `tau`.
  - A quote in any other expiry, including a design-B held-out expiry, linearly interpolates total variance between the knots. Beyond the knots it uses flat volatility. This is the same rule as v1's `bs_predict`.

Everything else is unchanged and reused as is:

- dates, surfaces and designs
- Heston fits and the variant selection
- metrics and the endpoint
- bootstrap seed and test

Outputs go to `artifacts/amend01/`. The v1 `results.json` is not overwritten. Both results are reported.

## Other checks disclosed with this amendment (no change made)

- **Nesting.** Across all 132 fits, the DH in-sample objective is never worse than SH's.
- **Bound contacts at the best solution.** The vol-of-vol upper bound `sigma = 10` binds for SH in 63/132 fits and for DH in 80/132 (slow) and 76/132 (fast). DH `rho_s = -0.99` binds in 44/132.
  - Changing the bounds after the test fits were seen is **not permitted** under the frozen rules. It is recorded as the leading candidate improvement for a *new*, independently frozen experiment on data not yet seen.
