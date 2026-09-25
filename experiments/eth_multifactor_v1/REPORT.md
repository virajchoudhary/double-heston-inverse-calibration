# ETH multifactor replication: report (2026-09-18)

**Setup.** This is the BTC design applied to Deribit ETH options:

- same engine, bounds, optimiser and metrics
- Black-Scholes as BS_EXPIRY
- `feller_free` for both Heston models, inherited from BTC
- shock dates from ETH DVOL

Every model is calibrated per date. The rules were frozen before any ETH option data was fetched.

**Data.** 60 dates were selected and 53 are usable: 27 shock dates across 10 episodes (2021–2026) and 26 calm dates. On average each date has 6.6 expiries and 68 quotes.

## Results (median held-out IV RMSE, vol points)

| cell | BS | SH | DH | DH vs SH | DH vs BS |
|---|---|---|---|---|---|
| **shock B (primary)** | 8.76 | 3.23 | **2.32** | 22/27 dates · 10/10 episodes · p = 0.00098 · [+0.61, +1.46]: **met** | 27/27 · [+5.6, +12.0]: **met** |
| shock A | 7.20 | 2.72 | **1.95** | 25/27 · 10/10 · p = 0.00098: met | 27/27: met |
| calm B | 8.77 | 2.68 | **2.34** | 18/26 · p = 0.0026 · [+0.22, +0.86]: met | 26/26: met |
| calm A | 6.12 | 2.14 | **1.56** | 25/26 · p = 3.7e-7: met | 26/26: met |

- **Mark-price target:** DH beats both models in all four cells.
- **Nesting check:** the DH in-sample objective is never worse than SH's (0 of 106 fits).

**Conclusion.** The BTC result replicates in full on ETH. The gain over SH (mean 0.96 vol points in the primary cell) is even larger than on BTC (0.57).
