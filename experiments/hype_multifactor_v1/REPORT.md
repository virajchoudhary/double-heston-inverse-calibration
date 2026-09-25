# HYPE multifactor replication: report (2026-09-18)

**Setup.** Same engine, bounds, optimiser, per-expiry Black-Scholes, metrics and statistics as BTC and XRP. Every model is calibrated per date, with no hard-coded parameters. The rules were frozen before any HYPE option data was fetched (see `PROTOCOL.md`).

**Data.** Deribit HYPE_USDC options were listed on 2026-06-22.

- Every day from 2026-06-23 to 2026-09-16 is a test date: 86 in all.
- 55 are usable (2026-06-26 to 2026-09-14). The other 31 had fewer than 4 usable expiries and are listed in `artifacts/data_audit.json`.
- The surfaces are thin: on average 4.2 expiries and 45 quotes per date.
- The primary unit is the ISO week, giving 13 weeks.

## Results, all 55 dates (median held-out IV RMSE, vol points)

| cell | BS | SH | DH | DH vs SH (dates · weeks · p · boot95) | DH vs BS |
|---|---|---|---|---|---|
| **held-out expiries (primary)** | 4.68 | 3.61 | **3.26** | 36/55 · 9/13 · p = 0.029 · [−0.06, +0.74]: **not met** | 44/55 · 12/13 · p = 0.0004 · [+1.4, +7.0]: **met** |
| held-out strikes | 5.06 | 2.88 | **2.75** | 38/55 · 9/13 · p = 0.040 · [+0.11, +0.71]: met | 51/55 · 13/13: met |
| held-out expiries, mark prices | 4.94 | 3.45 | **3.10** | 35/55: met | met |
| held-out strikes, mark prices | 5.20 | 2.87 | **2.71** | 38/55: not met | met |

- **Primary endpoint.** DH beats BS ✔. DH beats SH ✘, narrowly: the Wilcoxon test passes (p = 0.029), but the week-cluster bootstrap lower bound is −0.06.
- **Direction.** DH has the lowest median error in every cell.
- **Shock window.** Only one realised-volatility shock (onset 2026-08-19) falls inside the options history, and only 4 of its dates are usable. That is descriptive only:
  - Held-out strikes: SH 11.09, DH 6.84.
  - Held-out expiries: SH 7.28, DH 6.56, BS 6.30.
- **Nesting check.** The DH in-sample objective is never worse than SH's (0 of 110 fits).

## Reading, across the three assets

| asset | surface richness | DH vs BS | DH vs SH |
|---|---|---|---|
| BTC | ~7 expiries, ~90 quotes | wins everywhere | wins everywhere, significant |
| HYPE | ~4 expiries, ~45 quotes | wins everywhere | lower error everywhere; significant in 2 of 4 cells; primary narrowly missed |
| XRP | ~4 expiries, ~40 quotes | wins everywhere | mixed; SH better on unseen expiries |

Double Heston consistently beats Black-Scholes. Its edge over Single Heston is real, but it shrinks as the option surface thins out. It is decisive only where there are many liquid expiries to identify the two variance factors.
