# Hard-coded (published) parameters vs real markets, with the PINN

Frozen before any SPX data was fetched (`config.json`, `PROTOCOL.md`, manifest `artifacts/manifest.json`).

## What was tested

Four pricers, no parameter fitted:

- **DH_PUB** — the published Double Heston set (Christoffersen, Heston and Jacobs)
- **SH_PUB** — the published Single Heston set
- **BS_FIXED** — Black-Scholes at 15.97%, the initial total volatility of DH_PUB
- **DH_PINN** — the locked C3 network (two-seed mean) at the same DH_PUB parameters

Datasets, scored only where the PINN is trained (|log F/K| ≤ 0.36, 7 days to 2 years), so all four see identical quotes:

| dataset | why | quotes | median market IV |
|---|---|---|---|
| SPX, 2026-09-23 | the market the parameters were estimated on | 10,706 | 16.7% |
| BTC shock days (40) | the most volatile surfaces available | 2,721 | 59.2% |
| ETH shock days (27) | as above | 1,266 | 74.7% |

**Not tested:** the original 1990–2004 estimation window and its crises. Historical SPX chains are not freely available. Today's SPX is calm: VIX 14.21, the 26th percentile since 1990.

## 1. Nothing fitted (IV RMSE, vol points)

| dataset | BS_FIXED | SH_PUB | DH_PUB | DH_PINN |
|---|---|---|---|---|
| SPX today | 6.86 | 5.36 | **5.34** | 5.36 |
| BTC shock | 46.12 | **46.12** | 46.19 | 46.24 |
| ETH shock | 63.49 | 63.08 | **62.76** | 63.13 |

The crypto numbers are not a model comparison: the signed bias is −43 (BTC) and −59 (ETH) vol points, so all four are simply pricing a 16% world against a 60–75% one. On SPX the three stochastic-volatility pricers beat fixed-volatility Black-Scholes by 1.5 vol points, and Double Heston beats Single Heston by 0.02, which is nothing.

## 2. One number fitted per surface: the literature scale

| dataset | scale used | BS_FIXED | SH_PUB | DH_PUB | DH_PINN |
|---|---|---|---|---|---|
| SPX today | 0.70–0.80 | 7.27 | 6.00 | **5.82** | 5.85 |
| BTC shock | 3.2 (at the bound) | 37.55 | 37.17 | **36.06** | 36.14 |
| ETH shock | 3.2 (at the bound) | 54.96 | 53.20 | **52.95** | 53.00 |

Double Heston is now best everywhere, but the crypto bias is still −31 to −48 vol points because the scale stops at 3.2, the range the PINN was trained on. Those two rows remain level failures, not structural verdicts.

**SPX by maturity, with the level scaled:**

| maturity | BS_FIXED | SH_PUB | DH_PUB | DH_PINN | quotes |
|---|---|---|---|---|---|
| ≤ 30 days | **7.33** | 7.42 | 7.49 | 7.51 | 3,441 |
| 30–90 days | 7.75 | 6.08 | **5.51** | 5.55 | 2,919 |
| 90–365 days | 7.00 | 4.66 | **4.45** | 4.48 | 3,728 |
| > 365 days | 6.18 | 3.82 | **3.56** | 3.57 | 618 |

This is the only clean structural signal in the whole test: beyond 30 days the second variance factor helps, by 0.2–0.6 vol points, and the advantage grows with maturity. Inside 30 days the fixed shape fits nothing well and Black-Scholes is marginally best.

## 3. The PINN

On these real quote geometries the PINN reproduces exact Double Heston to a forward price RMSE of 1e-5 (SPX) and 5e-6 (crypto), which is 0.08–0.19 vol points RMSE, with a worst single quote of 2.1–2.5 vol points in the far wings (`artifacts/pinn_fidelity_on_real_quotes.json`).

So the PINN **does not beat Double Heston, and is not supposed to**: it is a fast surrogate for it. It does beat Single Heston and Black-Scholes wherever Double Heston does, by the same margins, and it inherits Double Heston's failures too.

## 4. Answer

- **Does the PINN beat Single Heston and Double Heston with hard-coded values?** It matches Double Heston (to 0.1–0.2 vol points) and beats Single Heston and Black-Scholes only where Double Heston does: on SPX beyond 30 days.
- **Does the hard-coded Double Heston beat the others?** On SPX, yes but barely, and only once the level is scaled; the gain is concentrated at 30 days and beyond. On volatile crypto surfaces, no: every fixed-parameter model fails by 30–60 vol points, which is the same level failure found in the NIFTY study.
- **The requested test — the original data in its most volatile periods — could not be run.** It needs paid historical SPX chains. What is shown instead is the original market today (calm) and the most volatile surfaces available (crypto), and both are labelled as such.
