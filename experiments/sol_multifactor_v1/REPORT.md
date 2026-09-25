# SOL multifactor replication: report (2026-09-18)

**Setup.** This is the XRP design applied to Deribit SOL_USDC options. It uses the same engine, bounds, optimiser, metrics and BS_EXPIRY, and every model is calibrated per date. The rules were frozen before any SOL option data was fetched.

**Data.** 58 dates were selected and 32 are usable: 16 shock dates across 7 episodes and 16 calm dates, 2024–2026. On average each date has 4.2 expiries and 84 quotes. The other 26 dates had fewer than 4 usable expiries.

## Results (median held-out IV RMSE, vol points)

| cell | BS | SH | DH | DH vs SH | DH vs BS |
|---|---|---|---|---|---|
| **shock B (primary)** | 10.15 | 3.82 | **3.67** | 8/16 dates · 4/7 episodes · p = 0.23 · [−0.10, +1.14]: **not met** | 15/16 · 7/7 · p = 0.008 · [+3.4, +8.9]: **met** |
| shock A | 10.75 | 4.63 | **4.31** | 13/16 · 6/7 · p = 0.023 · [+0.18, +1.28]: met | 16/16: met |
| calm B | 6.98 | 2.42 | **2.36** | 9/16 · p = 0.84; mean −0.48 (SH better on average): not met | 16/16: met |
| calm A | 5.88 | 2.51 | **2.41** | 11/16 · p = 0.15: not met | 16/16: met |

- **Mark-price target:** DH beats SH only in shock A. DH beats BS everywhere.
- **Nesting check:** the DH in-sample objective is never worse than SH's (0 of 64 fits).

**Conclusion.** DH beats Black-Scholes decisively. Against SH, DH has the lower median error in every cell, but the difference is significant only for held-out strikes on shock dates. This is the same thin-surface pattern as XRP and HYPE.
