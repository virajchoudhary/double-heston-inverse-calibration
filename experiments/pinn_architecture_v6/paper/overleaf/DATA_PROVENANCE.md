# What data was used, everywhere

Every number in the paper patches traces to a committed artifact. This file lists every dataset
used across the whole body of work, what it is, and which claim it supports.

---

## 1. The paper's own data — SYNTHETIC, and that is correct

**Nothing in the paper uses a market quote.** `PROTOCOL.md` states the scope explicitly:
"no market quote reads, calibration, parameter-domain changes, teacher changes, scenario edits or
gate relaxation... No final market test is opened."

| item | value |
|---|---|
| source | generated from the **exact Double-Heston pricer** (the "teacher"), not observed |
| final fidelity panel | **4,096 price points, seed 206104**; **512 PDE points, seed 206105** |
| generated | only AFTER arm selection and after all checkpoints and code were hashed |
| evaluated | **once**, on the selected candidate and the frozen baseline, on common points |
| IV metric subset | 3,785 of 4,096 rows are invertible to an implied volatility |
| development panel | separate; committed as `artifacts/PINN_ARCHITECTURE_ABLATION.csv` |
| parameters | published Christoffersen-Heston-Jacobs (2009) set, hard-coded, never fitted |
| sampling law | Table I of the paper: moneyness U[-0.36, 0.36], tau log-U[7/365, 730/365], v1 log-U[0.00015, 0.18], v2 log-U[0.001, 0.22], level scale U[0.7, 3.2], r = q = 0 |
| seeds | 17 and 43, two genuine parent checkpoints. **No third seed was fabricated.** |

This is the right choice: measuring a network's approximation error requires a ground truth, and only
a synthetic panel provides one. Swapping in market data would destroy the claim, not strengthen it.

**The one gap:** Table III's values for A0, CONTROL, ARCH_B and ARCH_C appear in **no committed
artifact**. The development CSV gives different numbers for those same arms. Only the SHARED and
ARCH_A rows match `final_fidelity/`. See `PAPER_PATCHES.tex` blocks 2A and 2B.

---

## 2. Controlled two-timescale benchmark — SYNTHETIC (figures, not the paper)

Used for every price-curve figure in `primary_pricing_curves_v1` and `clear_separation_figures_v1`.

| item | value |
|---|---|
| r, q | 0, so forward = spot |
| strike | K = 100 |
| variance states | frozen `FIXED_TOTAL_TWIST`: fast-heavy (v_f 0.035, v_s 0.005), slow-heavy (v_f 0.005, v_s 0.035); both total 0.04, i.e. 20% instantaneous |
| mean reversion | kappa_f 10.7526, kappa_s 0.9491 (ratio 11.3x) |
| S grid | 601 points, 0.60K to 1.40K |
| tau grid | 601 points, 7 to 730 days |
| Single Heston | frozen strongest fit, global search + 12 restarts — deliberately not weakened |
| Black-Scholes | TWO baselines kept: constant sigma = 20%, and the calibrated non-leaking term sigma(tau) |

---

## 3. Black-Scholes PINN — its own convention (branch `black-scholes-with-pinn`)

| item | value |
|---|---|
| r, q | 0.03, 0.01 — **not** the r = q = 0 benchmark |
| sigma | calibrated 0.19999906 |
| domain | S/K in [0.550, 1.650], tau <= 2 years |
| architecture | 5 x 96 hidden, 37,634 parameters |

Never placed on the controlled benchmark, because its rate, dividend and volatility all differ.
Compared only against analytic Black-Scholes at its own parameters.

---

## 4. Crypto option markets — REAL, Deribit

Five assets. Dates were selected from **spot or volatility-index data only**, before any option
surface was fetched, and every manifest records `option_data_fetched_before_freeze: false` and
`models_fitted_before_freeze: false`.

| asset | venue | usable dates | quotes | median expiries/day | date range | DH beats SH? | DH beats BS? |
|---|---|---|---|---|---|---|---|
| BTC | Deribit inverse, BTC-settled | 66 | 6,171 | 7 | 2021-05-24 .. 2026-07-06 | **yes** | **yes** |
| ETH | Deribit inverse, ETH-settled | 53 | 3,619 | 6 | 2021-05-24 .. 2026-07-03 | **yes** | **yes** |
| SOL | Deribit linear USDC | 32 | 2,687 | 4 | 2024-11-09 .. 2026-08-27 | no | yes |
| XRP | Deribit linear USDC | 26 | 1,249 | 4 | 2024-10-03 .. 2026-08-27 | no | yes |
| HYPE | Deribit linear USDC | 55 | 2,460 | 4 | 2026-06-26 .. 2026-09-14 | no | yes |

**The pattern:** Double Heston beats Black-Scholes everywhere, but beats Single Heston only where
roughly seven expiries per day exist. At four expiries there is not enough maturity structure for a
second variance factor to identify itself.

Date-selection rules, all frozen in advance:
* BTC, ETH: DVOL shock onset, ratio >= 1.2 over a 60-day median, de-duplicated 20 days, surfaces at
  1, 4 and 8 days after onset; plus a calm sample.
* SOL, XRP, HYPE: no volatility index exists, so realised RV10 from the perpetual was used, ratio
  >= 1.5. XRP's calm cap was amended **before fetching** from a fixed 1.3 to a q25 cap, because 1.3
  admitted 1 of 899 days; the empty first attempt is preserved as `dates_v0_empty_calm.json`.
* HYPE: options history too short for an episode sample, so every calendar day is a test date.

Statistics: one-sided Wilcoxon over clusters (episodes, or ISO weeks for HYPE) **and** a
cluster-bootstrap 95% lower bound above zero, 5,000 replicates, seed 20260920. Both must pass.

---

## 5. SPX — REAL, the market-ceiling result

| item | value |
|---|---|
| quotes | 10,706, VIX 14.2 |
| split | frozen checkerboard on (expiry rank + strike rank): 4,271 calibration, 1,082 development, 5,353 held out |

Results in volatility points: exact Double Heston 5.818; DH-PINN v4 5.850; DH-PINN v5 5.847.

**This is the ceiling argument.** Improving the surrogate 1.6x moved SPX error by **0.003** of
**5.85**. Network error is 0.074 against a market error of 5.85, about **1.3%**. A perfect surrogate
would still sit at 5.818. Architecture work cannot repair a fixed-parameter model's market error.

A market-residual head cut held-out error from ~5.8 to ~2.0 — but **the gain came from the head, not
the prior**: under an identical head the Double-Heston prior did **not** beat the Single-Heston prior
(1.990 vs 1.970, and the residual left by each is statistically indistinguishable). Reported as a
negative result.

---

## 6. IWM — REAL, ex-ante selected

Selection rule frozen and hashed (`SELECTION_RULE.md`, SHA-256 `8f66740024e7d257...`) **before any
model error was computed**. Score = 0.40 z(S1) + 0.20 z(S2) + 0.15 z(S3) + 0.15 z(S4) + 0.10 z(S5),
using market/surface characteristics only. IWM selected, score 1.768, rank 1 of 17.

1,416 quotes, 22 expiries. Calibration 709 quotes, held out 707.

| model | (a) published prior, one level scale | (b) calibrated, held out |
|---|---|---|
| Black-Scholes per expiry | 5.611 | 5.388 |
| Single Heston | 5.453 | 1.893 |
| Double Heston | **5.301** | **0.583** |

**Both panels must be reported.** Panel (a) is the honest same-prior comparison, where Double Heston
wins by only 0.15 vol points out of 5.3. Quoting only panel (b) would overstate the case.

**Two integrity disclosures, carried forward:**
1. IWM's model errors were **already known** before this selection. The weights were fixed in advance
   and applied mechanically, and 8 of 17 scored surfaces were error-blind, but the selection as a
   whole is **not blind** and must not be described as such.
2. **The domain filter changed the winner.** Under the earlier rule SPY scored highest and Double
   Heston *lost* there (6.54 vs 6.51). SPY is rejected here because its short-dated ATM volatility,
   12.3%, falls below the PINN's trained range of 13.4-28.6%. The filter is legitimate, but it is the
   reason the selected surface changed.

---

## Summary: which data supports which claim

| claim | data |
|---|---|
| the 9.24% architecture result (the paper) | **synthetic**, exact-DH teacher, 4,096 points |
| price curves are financially valid | **synthetic** controlled benchmark |
| DH beats BS in real markets | **real**, 5 crypto assets, 16,186 quotes total |
| DH beats SH only with enough expiries | **real**, BTC/ETH yes at 6-7 expiries; SOL/XRP/HYPE no at 4 |
| architecture gains do not move market error | **real**, SPX, 10,706 quotes |
| DH's structural advantage when calibrated | **real**, IWM, 707 held-out quotes |
| the DH prior does not beat the SH prior under an equal-capacity head | **real**, SPX — negative result |
