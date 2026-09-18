# BTC multifactor experiment: protocol

Written and frozen before any option surface was fetched or any model fitted. `config.json` is the complete specification; this note explains the reasoning behind it.

## Question

In periods where a two-timescale variance structure should matter, does a fully calibrated Double Heston price **held-out** Bitcoin options better than a fully calibrated Single Heston and the strongest non-leaking Black-Scholes? Held-out means options that none of the models saw during calibration.

## Why this design, and what it fixes

The NIFTY test (`../nifty_multifactor_v4/FINAL_REPORT.md`) found that the Double Heston market error was almost entirely **parameter level**. Fixed DJIA-literature parameters implied roughly 15% vol against a roughly 10% market, and the neural solver and the one-vs-two-factor structure were not the cause. It also found that single-expiry surfaces identify only about one of Double Heston's ten parameter directions. This experiment changes exactly those conditions.

1. **Calibration:** every model is calibrated to each date's own quotes: 10 parameters for Double Heston, 5 for Single Heston, one volatility per expiry for Black-Scholes. The same global-plus-12-start optimiser is used for both Heston models.
2. **Data:** Deribit BTC options have 9–12 liquid expiries per date, from days out to a year. That is enough maturity range to identify two variance factors.
3. **Convention:** the separable four-shock convention allows opposite-signed factor correlations. The controlled study found this to be the mechanism behind short-dated smirk-slope flexibility.
4. **Feller condition:** strict or relaxed, chosen on validation dates, with the same rule for both Heston models.

## Periods: set in advance, model-free

- **Shock dates** are defined from the DVOL index alone. DVOL must be at least 1.2× its trailing 60-day median, onsets are de-duplicated over 20 days, and surfaces are taken 1, 4 and 8 days after each onset. A volatility shock excites a fast variance factor on top of a slow one, so these are the dates where a second timescale should matter.
- **Calm dates** are a control set, drawn by a fixed seed.
- Dates are split chronologically: the earliest 30% are validation (used only to choose the Feller variant) and the rest are test.

## Tests

- **Primary:** held-out expiries (design B). Calibrate on alternate expiries and price the omitted ones, including their full smiles.
- **Secondary:** held-out strikes (design A), calm dates, USD price errors, and mark-price targets.
- **Endpoint:** see `config.json`. Double Heston "beats" a competitor only under a one-sided Wilcoxon test over shock *episodes*, with p < 0.05, **and** a positive episode-cluster bootstrap lower bound.

## Integrity

- The results are reported whichever way they come out, including the calm control.
- Nothing is changed after the test fits are seen.
- Unusable or unavailable dates are disclosed, never replaced.
- Trade prices are not bid/ask quotes.
