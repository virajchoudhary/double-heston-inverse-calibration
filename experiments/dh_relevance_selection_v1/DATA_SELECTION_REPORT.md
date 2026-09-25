# Data selection report (ex ante)

Rule: `SELECTION_RULE.md`, SHA-256 `8f66740024e7d25743269162ba0fa70231364a82e027fcdc2ccde8d061d20534`.

No pricing-model error of any kind appears in this report. Errors were computed only after this
file was written and hashed.

Candidates examined: **62**.  Usable surfaces: **46**.  Passed the domain filter: **17**.  Scored: **17**.

## Selected surface

**IWM**, DH-relevance score **1.768**.  Errors for this surface were ALREADY KNOWN before selection.

## Ranking of surfaces that passed the domain filter

| rank | symbol | error status | S1 one-timescale misfit | S2 curvature | S3 skew | S4 skew variation | S5 short-vs-long | DH score |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | IWM | error known | 0.1234 | 0.0973 | 0.378 | 0.370 | 0.063 | **1.768** |
| 2 | QQQ | error known | 0.0552 | 0.0669 | 0.409 | 0.327 | 0.207 | **0.690** |
| 3 | AAPL | error known | 0.0759 | 0.0765 | 0.157 | 0.249 | 0.101 | **0.473** |
| 4 | MA | error blind | 0.0819 | 0.1001 | 0.126 | 0.103 | 0.063 | **0.345** |
| 5 | V | error blind | 0.0687 | 0.0735 | 0.189 | 0.192 | 0.099 | **0.274** |
| 6 | KO | error blind | 0.0474 | 0.1312 | 0.203 | 0.151 | 0.016 | **0.097** |
| 7 | JPM | error known | 0.0409 | 0.1271 | 0.147 | 0.208 | 0.032 | **0.027** |
| 8 | XLK | error blind | 0.0545 | 0.1114 | 0.204 | 0.108 | 0.002 | **-0.001** |
| 9 | _RUT | error known | 0.0419 | 0.0327 | 0.407 | 0.255 | 0.078 | **-0.027** |
| 10 | MSFT | error known | 0.0384 | 0.0967 | 0.051 | 0.178 | 0.216 | **-0.098** |
| 11 | BAC | error blind | 0.0349 | 0.1219 | 0.183 | 0.130 | 0.054 | **-0.162** |
| 12 | PFE | error blind | 0.0402 | 0.1240 | 0.093 | 0.066 | 0.097 | **-0.232** |
| 13 | GLD | error known | 0.0415 | 0.0587 | 0.060 | 0.211 | 0.155 | **-0.280** |
| 14 | _NDX | error known | 0.0149 | 0.0252 | 0.384 | 0.238 | 0.209 | **-0.369** |
| 15 | TLT | error known | 0.0261 | 0.0445 | 0.138 | 0.092 | 0.200 | **-0.660** |
| 16 | HD | error blind | 0.0400 | 0.0514 | 0.054 | 0.125 | 0.037 | **-0.673** |
| 17 | CVX | error blind | 0.0184 | 0.0306 | 0.068 | 0.103 | 0.025 | **-1.171** |

## Rejected before scoring

| symbol | reason |
|---|---|
| AMD | short ATM IV 55.2% outside [13.4, 28.6]% |
| AMZN | short ATM IV 30.1% outside [13.4, 28.6]% |
| BA | short ATM IV 31.8% outside [13.4, 28.6]% |
| C | short ATM IV 30.1% outside [13.4, 28.6]% |
| CAT | short ATM IV 35.4% outside [13.4, 28.6]% |
| DIA | short ATM IV 13.2% outside [13.4, 28.6]% |
| EEM | no usable surface |
| EFA | no usable surface |
| EWJ | no usable surface |
| EWZ | no usable surface |
| F | no usable surface |
| FXI | no usable surface |
| GE | short ATM IV 31.6% outside [13.4, 28.6]% |
| GOOGL | short ATM IV 32.2% outside [13.4, 28.6]% |
| GS | short ATM IV 32.4% outside [13.4, 28.6]% |
| HYG | no usable surface |
| IEF | fewer than 150 quotes; short ATM IV 9.6% outside [13.4, 28.6]%; fewer than 2 expiries over 180d |
| INTC | short ATM IV 71.6% outside [13.4, 28.6]% |
| LQD | no usable surface |
| META | short ATM IV 49.7% outside [13.4, 28.6]% |
| MS | short ATM IV 32.1% outside [13.4, 28.6]% |
| NVDA | short ATM IV 31.5% outside [13.4, 28.6]% |
| SLV | short ATM IV 34.1% outside [13.4, 28.6]% |
| SMH | short ATM IV 34.0% outside [13.4, 28.6]% |
| SPY | short ATM IV 12.5% outside [13.4, 28.6]% |
| T | fewer than 150 quotes; fewer than 3 expiries under 90d |
| TSLA | short ATM IV 44.5% outside [13.4, 28.6]% |
| UNG | no usable surface |
| UNH | short ATM IV 33.5% outside [13.4, 28.6]% |
| USO | short ATM IV 56.9% outside [13.4, 28.6]% |
| VXX | fewer than 150 quotes; short ATM IV 45.0% outside [13.4, 28.6]%; fewer than 2 expiries over 180d |
| WFC | short ATM IV 28.7% outside [13.4, 28.6]% |
| WMT | fewer than 150 quotes |
| XBI | short ATM IV 31.2% outside [13.4, 28.6]% |
| XLE | fewer than 150 quotes |
| XLF | no usable surface |
| XLI | no usable surface |
| XLP | no usable surface |
| XLRE | no usable surface |
| XLU | no usable surface |
| XLV | no usable surface |
| XLY | no usable surface |
| XOM | short ATM IV 29.4% outside [13.4, 28.6]% |
| _DJX | short ATM IV 12.9% outside [13.4, 28.6]% |
| _SPX | short ATM IV 12.7% outside [13.4, 28.6]% |

## Notes

- The domain filter uses market quotes only: short-end at-the-money implied volatility must lie in
  [13.4%, 28.6%], the level range the PINN was trained on, so no surface is selected that would force
  the network to extrapolate.
- Components are standardised as z-scores across the surfaces that passed the filter, then combined
  with the fixed weights 0.40 / 0.20 / 0.15 / 0.15 / 0.10.
- Sixteen symbols had model errors computed earlier in this project and are labelled `error known`;
  the rest are labelled `error blind`. The identical score is applied to both groups.
