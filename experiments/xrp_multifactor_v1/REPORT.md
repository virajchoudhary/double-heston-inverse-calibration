# XRP multifactor replication: report (2026-09-18)

**Asked for:** ENJ. ENJ has no listed options on Deribit, Binance, OKX or Bybit, so XRP (Deribit XRP_USDC options) was used instead, at the user's choice.

**Setup.** Identical to BTC: engine, bounds, optimiser, per-expiry Black-Scholes, metrics and endpoint. Every model is calibrated per date, and every date is a test date. The rules were frozen before any XRP option data was fetched. Changes forced by the market are listed in `PROTOCOL.md`, including one pre-fetch amendment to the calm rule.

**Data.** 58 dates were selected, 26 of them usable: 18 shock dates across 9 episodes (2024-10 to 2026-08) and 8 calm dates. The other 32 had fewer than 4 usable expiries and are disclosed in `artifacts/data_audit.json`. The surfaces are thin: 4–5 expiries and 20–97 quotes per date, against about 7 expiries and 90 quotes for BTC.

## Results (median held-out IV RMSE, vol points)

| cell | BS | SH | DH | DH vs SH | DH vs BS |
|---|---|---|---|---|---|
| **shock B (primary)** | 13.09 | **5.24** | 5.72 | 12/18 dates, 7/9 episodes, p = 0.037, boot95 [−0.08, +1.82]: **not met** | 17/18, p = 0.002, [+5.0, +11.3]: **met** |
| shock A | 10.54 | 4.57 | **4.08** | 14/18, 9/9, p = 0.002, [+0.28, +2.19]: met | 18/18: met |
| calm B | 7.00 | **3.28** | 3.52 | 3/8, p = 0.73: not met | 8/8: met |
| calm A | 5.67 | **2.19** | 2.52 | 3/8, p = 0.47: not met | 8/8: met |

- **Primary endpoint:**
  - DH beats BS: ✔
  - DH beats SH: ✘. The mean difference favours DH (+0.59), but the bootstrap lower bound is below 0 and the median date favours SH.
- **Mark-price target:** DH beats SH in both shock cells (shock B: SH 4.27, DH 4.14) but not in the calm cells. DH beats BS everywhere.
- **Price RMSE (USD per XRP option, median):** DH and SH are about equal. Both are well below BS.
- **Nesting check:** the DH in-sample objective is never worse than SH's. See `artifacts/logs`.

## Reading

1. **Against Black-Scholes, the BTC result replicates fully.** DH wins in every cell, on 51 of 52 date-cells.
2. **Against Single Heston, it replicates only partly.**
   - DH wins where it has the most information to fit on: held-out strikes on shock dates.
   - It does not win when whole expiries are held out, or on calm dates.
   - With only 3 calibration expiries (design B), 10 parameters are weakly identified. DH fits better in-sample (2.84 vs 3.71), but that gain does not reliably carry over to the unseen expiry.
3. **The practical rule.** Double Heston's advantage over Single Heston needs a rich surface: many liquid expiries, as on BTC. On a thin surface like XRP's, Single Heston is as good or slightly better on unseen expiries.
