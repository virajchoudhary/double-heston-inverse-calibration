# BTC multifactor experiment: final report (2026-09-18)

**Question.** In periods chosen, model-free, as the ones where a second variance timescale should matter, does a per-date calibrated Double Heston price **held-out** Deribit BTC options better than Single Heston and Black-Scholes?

**Answer.** Yes, on every predeclared cell. It also wins in the calm control, so the advantage is **not specific to volatility shocks**.

## 1. What changed relative to the NIFTY study, and why

The NIFTY final test found that Double Heston was held back by three things:

- **Parameter level.** Fixed DJIA-literature parameters implied about 15% vol against a roughly 10% market.
- **Identification.** A single expiry pins down only about one of Double Heston's ten parameter directions.
- **Convention.** Correlation restrictions removed the short-dated smirk flexibility.

This experiment addresses each of these (full detail in `PROTOCOL.md`):

| limitation | change |
|---|---|
| parameters not calibrated to the market | all 10 DH parameters (5 for SH) calibrated to each date's own quotes, with the same DE plus 12-start least-squares optimiser for both models |
| one expiry, weak identification | 5–12 Deribit expiries per date, from 3 to 400 days |
| restricted correlations | separable four-shock convention with independent ρ_slow and ρ_fast (opposite signs allowed) |
| Feller condition forced | strict or free, chosen on validation dates only. Free won for both models: validation median 3.81 vs 6.88 (DH) and 5.04 vs 8.18 (SH) |

## 2. Periods (fixed from the DVOL index before any option data was fetched)

- **Shock:** DVOL ≥ 1.2× its 60-day median. Onsets are de-duplicated over 20 days, and surfaces are taken 1, 4 and 8 days after each onset.
  - Validation: 4 episodes, 2021–2023.
  - **Test: 10 episodes, 29 usable dates, 2023-03 to 2026-07.**
- **Calm control:** 30 dates drawn by seed. **Test: 21 dates, 2022–2025.**
- **Unusable dates:** six, all with fewer than 5 expiries after filters (2 shock, 4 calm). They are disclosed and were not replaced.
- **Designs:**
  - **B (primary):** alternate expiries are held out, including their whole smiles.
  - **A:** alternate strikes are held out.

## 3. Results on test dates (median held-out IV RMSE, vol points)

Black-Scholes is the corrected per-expiry version (see §5).

| cell | BS | SH | DH | DH beats SH (dates · units · p · boot95) | DH beats BS |
|---|---|---|---|---|---|
| **shock B (primary)** | 8.13 | 2.50 | **1.86** | 23/29 · 10/10 episodes · p = 0.00098 · [+0.39, +0.71] ✔ | 29/29 · p = 0.00098 · [+5.2, +8.3] ✔ |
| shock A | 7.65 | 2.27 | **1.51** | 29/29 · 10/10 · p = 0.00098 · [+0.73, +1.15] ✔ | 29/29 ✔ |
| calm B | 6.40 | 2.53 | **1.37** | 19/21 · p = 6.5e-5 · [+0.64, +1.65] ✔ | 21/21 ✔ |
| calm A | 5.14 | 2.02 | **1.10** | 19/21 · p = 6.5e-5 · [+0.42, +0.89] ✔ | 21/21 ✔ |

**Primary endpoint:** DH beats SH ✔ and DH beats BS ✔. In both cases the one-sided episode Wilcoxon gives p < 0.05 and the episode-cluster bootstrap lower bound is above 0.

### Predeclared secondary analyses (all agree)

- **Held-out price RMSE (USD, median):**
  - shock B: BS 195, SH 85, **DH 62**
  - calm B: BS 213, SH 151, **DH 81**
- **Mark-price target.** The same fits are scored against Deribit mark prices instead of last trades. Shock B: BS 8.25, SH 2.40, **DH 1.74**. DH still beats both models in all four cells.
- **Maturity buckets** (pooled held-out IV RMSE, shock B):

  | maturity | SH | DH | BS |
  |---|---|---|---|
  | ≤ 7 days | 6.12 | **5.04** | 18.1 |
  | 7–30 days | 2.86 | **2.35** | — |
  | 30–90 days | 2.47 | **1.98** | — |
  | > 90 days | 2.55 | **2.19** | — |

  DH is ahead in every bucket except calm A ≤ 7d, where SH 3.98 and DH 4.05 are level.
- **Moneyness buckets.** DH's gain is largest in the OTM put wing (x > 0.15): calm A SH 3.56 vs DH 1.55, and shock A SH 6.44 vs DH 4.07. That is where independent fast and slow correlations add smirk flexibility.

## 4. Honest reading

1. **DH generalises better than SH on BTC, but not only in shocks.** The calm-control gap (mean around 1 vol point) is at least as large as the shock gap (around 0.55). The model-free "two-timescale" story for shocks is therefore not what drives the result. BTC surfaces seem to need two factors in general.
2. **Neither Heston model is near market precision at the very short end.** Shock ≤ 7-day errors are 5–6 vol points. The largest remaining weakness is the short-dated wing.
3. **Bound contacts are material.** At the best solution, the vol-of-vol upper bound σ = 10 binds for:
   - SH: 63 of 132 fits
   - DH slow factor: 80 of 132
   - DH fast factor: 76 of 132

   DH's ρ_slow = −0.99 binds in 44 of 132 fits. Under the frozen rules these bounds **cannot be changed now**, because the test fits have been seen. Widening σ, or adding jumps for the ≤ 7-day wing, is the leading candidate improvement for a *new* experiment frozen in advance on unseen data (for example, BTC dates after 2026-07 or ETH options).
4. **Optimiser.**
   - DH's in-sample objective is never worse than SH's (0 of 132 fits), so the nesting check passes.
   - DH reaches its best objective from fewer starts: median 5 of 12, versus 12 of 12 for SH. No global optimum is claimed.
5. **Data.** Last trades in a 06:00–08:00 UTC window are not synchronised bid/ask quotes, so no spread-based claims are made. The mark-price check guards against stale trades.

## 5. Bug found after evaluation, and how it was handled

`BS_TERM` in the frozen v1 code used `np.unique(tau)` as knots. Because `tau` is computed from each trade's own timestamp, that produced **one Black-Scholes vol per quote** (139 knots for 8 expiries on 2026-02-11). The result was a meaningless in-sample IV RMSE of 0 and held-out predictions interpolated between individual quotes.

The fix was handled as follows:

- It was documented in `AMENDMENT_01_BS_EXPIRY.md` and hashed in `artifacts/amend01/manifest.json` **before** any corrected fit was run.
- The fixed model groups by expiry.
- Only the BS arm was refitted. Nothing else was changed.
- The flawed v1 outputs (`artifacts/results.json`, `test_scores.csv` and `fits/*/*/BS_TERM.json`) are preserved.

| | v1 (flawed) | amended |
|---|---|---|
| BS held-out IV RMSE, shock B | 9.63 | 8.13 |
| BS in-sample IV RMSE | 0.00 | 8.14 |

The DH vs SH conclusion is untouched, and DH beats BS under both versions.

## 6. Files

| file | contents |
|---|---|
| `config.json`, `PROTOCOL.md`, `engine.py`, `data.py`, `run.py`, `test_btc.py` | the v1 experiment, frozen (manifest `artifacts/manifest.json`) |
| `AMENDMENT_01_BS_EXPIRY.md`, `amend01.py`, `test_amend01.py` | the BS correction (manifest `artifacts/amend01/manifest.json`) |
| `artifacts/results.json` | v1 evaluation (flawed BS) |
| `artifacts/amend01/results.json` | primary evaluation with corrected BS |
| `artifacts/secondary/` | mark-price target, maturity and moneyness buckets, price RMSE |
| `artifacts/data_audit.json`, `artifacts/raw/`, `artifacts/surfaces/` | data provenance with raw sha256 |
| `artifacts/fits/`, `artifacts/logs/` | every fit, with all starts retained |
