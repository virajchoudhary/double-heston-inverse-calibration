# Hard-coded parameter comparison: protocol

Frozen before any SPX data was fetched.

## Question

With **nothing fitted**, how do these four price real options?

- Double Heston with the published parameters
- Single Heston with the published parameters
- Black-Scholes at a fixed volatility
- the locked DH-PINN, evaluated at the same published Double Heston parameters

## Two things this can and cannot answer

- **The PINN cannot beat exact Double Heston.** It is trained to reproduce it, so a match is a success and any gap is network error. The meaningful comparison is DH and PINN against Single Heston and Black-Scholes.
- **The original data cannot be fully reproduced.** The published parameters were estimated on S&P 500 index options in the 1990s and early 2000s. Historical SPX option chains are not freely available, so this uses today's SPX chain, which is calm (VIX 14.21). The requested "most volatile periods" are therefore tested on the most volatile surfaces this project does have: the frozen BTC and ETH shock dates.

## Why a level-free second analysis is reported

A fixed parameter set carries a fixed volatility level. The NIFTY study already showed that on a market whose level differs, that mismatch dominates every structural difference. So alongside the pure fixed-parameter result, one number is fitted per surface — the literature scale factor `s`, inside the range the PINN was trained on — and the comparison is repeated. That isolates the smile and term-structure **shape**, which is where a second variance factor should help.

## Integrity

See `config.json`. Everything is reported as it comes out, including a failure of the published parameters on every market.
