# Ex-ante DH-relevance selection rule

Fixed and hashed before the score was computed on any new surface, and before any model error was
computed on any surface outside the previously disclosed set. Components and weights are taken from
the specification supplied by the user; nothing here was tuned.

## Step 1 — domain filter (applied first, market data only)

A candidate surface is rejected unless all hold:

1. at least 6 usable expiries and at least 150 usable quotes after the standard filters
   (mid of bid/ask, bid > 0, ask > bid, ask <= 3*bid, out-of-the-money only, 7 <= days <= 730,
   |x| = |log(F/K)| <= 0.36, >= 6 quotes per expiry, parity forward with R^2 >= 0.999);
2. short-end at-the-money implied volatility inside **[13.4%, 28.6%]**, which is
   sqrt(0.0255 * s) for the PINN's trained level scale s in [0.7, 3.2], where 0.0255 is the
   published initial total variance. This is computed from market quotes only, not from a fit;
3. at least 3 expiries below 90 days and at least 2 expiries above 180 days, so the term structure
   is actually observed on both ends.

## Step 2 — structural components (market data only, no pricing model of any kind)

For each surface, per expiry, the at-the-money implied volatility is the market smile interpolated
to x = 0, and the total variance is w(tau) = sigma_atm(tau)^2 * tau.

- **S1 — one-timescale misfit.** Fit w(tau) = theta*tau + (v0 - theta)*(1 - exp(-kappa*tau))/kappa
  by least squares over (theta, v0, kappa) and take RMSE(residual) / mean(w).
- **S2 — term-structure curvature.** RMS of the second difference of sigma_atm against log(tau),
  divided by mean sigma_atm.
- **S3 — smile/skew magnitude.** Mean over expiries of |IV(x = +0.2) - IV(x = -0.2)| / IV_atm,
  using only expiries whose quotes span that range.
- **S4 — skew variation across maturity.** (max - min) of the same per-expiry skew, divided by mean
  IV_atm.
- **S5 — short-versus-long separation.** |median IV_atm below 60 days - median IV_atm above 180 days|
  divided by mean IV_atm.

Each component is standardised across the surviving candidates as a z-score.

## Step 3 — score

    DH_SCORE = 0.40*z(S1) + 0.20*z(S2) + 0.15*z(S3) + 0.15*z(S4) + 0.10*z(S5)

The highest score is selected, frozen, and only then are pricing errors computed.

## Step 4 — headline figure

The headline curves are produced for the selected real surface **and** for the predeclared controlled
`FIXED_TOTAL_TWIST` benchmark, each labelled for what it is. The controlled benchmark is never
described as market evidence.

## Integrity disclosure

Sixteen surfaces (`_SPX, _NDX, _RUT, _DJX, SPY, QQQ, IWM, DIA, GLD, TLT, XLE, JPM, AAPL, MSFT, NVDA,
TSLA`) already had their Black-Scholes, Single Heston and Double Heston errors computed in an earlier
step of this project, and those results are known. Selection over those surfaces is therefore **not
blind**. Every candidate is labelled `error_known` or `error_blind` in the selection report, and the
score is applied identically to both groups.
