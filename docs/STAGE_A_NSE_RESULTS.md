# Stage A Official-NSE Market-Support Results

Status date: 09 August 2026

## Purpose and scope

This report records the auditable Stage A market-support milestone used to inform later stock-candidate and neural-representation decisions. It covers official NSE daily CM and F&O UDiFF observations only. It does not rank candidates, freeze a replacement representation, calculate implied volatility or final futures-implied carry, generate research data, or train an ANN/PINN.

## Dates and universe

The deterministic screen processed:

- 2026-07-01
- 2026-07-15
- 2026-07-22

The candidate universe is NTPC, POWERGRID, SUNPHARMA, CIPLA, INFY, TCS, ICICIBANK, and HDFCBANK. All eight were present on all three dates. NIFTY derivatives were processed separately as a reference-only block and never entered candidate ranking.

## Official source and provenance

The primary source was official NSE daily UDiFF bhavcopy:

```text
CM: BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
FO: BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
```

CM provided each stock's independent `EQ` close. F&O provided stock options (`STO`), stock futures (`STF`), NIFTY index options (`IDO`), and NIFTY index futures (`IDF`). Raw ZIPs and extracted CSVs remain ignored and unchanged. The local acquisition manifest records official URL, timestamp or explicit fallback, original filename, archive size, ZIP and CSV SHA-256, ZIP integrity, member name, encoding, delimiter, market, and trading date. Existing evidence is verified and never overwritten.

## Surface counts and DTE findings

One logical surface is one stock plus one valuation date containing every observed expiry slice. The result contains 24 candidate stock surfaces: three per stock. The screen selected 4,740 stock-option rows.

| Valuation date | Actual stock-option DTE pattern |
|---|---|
| 2026-07-01 | 27, 55, 90 |
| 2026-07-15 | 13, 41, 76 |
| 2026-07-22 | 6, 34, 69 |

No selected stock option or future had an `XpryDt` / `FininstrmActlXpryDt` mismatch.

## Moneyness coverage

Counts are observed support across 72 stock-expiry slices. `DIRECT` used a strict `1e-12` log-moneyness tolerance; no interpolation was performed.

| log(K/S) node | Observed support | Outside support |
|---:|---:|---:|
| -0.30 | 2/72 | 70/72 |
| -0.20 | 36/72 | 36/72 |
| -0.10 | 72/72 | 0/72 |
| -0.05 | 72/72 | 0/72 |
| 0.00 | 72/72 | 0/72 |
| +0.05 | 72/72 | 0/72 |
| +0.10 | 72/72 | 0/72 |
| +0.20 | 20/72 | 52/72 |
| +0.30 | 7/72 | 65/72 |

The central `-0.10` through `+0.10` nodes were consistently observed. Both extreme wings were sparse.

## Candidate maturity-grid support

Counts are over 24 stock surfaces. `NEAR_MATCH` used a ±2-calendar-day tolerance.

| Candidate node | Direct | Near | Bracketed | Outside |
|---:|---:|---:|---:|---:|
| 7 days | 0 | 8 | 0 | 16 |
| 14 days | 0 | 8 | 8 | 8 |
| 30 days | 0 | 0 | 24 | 0 |
| 60 days | 0 | 0 | 24 | 0 |
| 90 days | 8 | 0 | 0 | 16 |
| 180 days | 0 | 0 | 0 | 24 |

The 30- and 60-day nodes were bracketed consistently. Support for 7, 14, and 90 days was date-dependent. The 180-day node was unsupported on every candidate surface.

## Price-observation and activity summary

All selected stock-option rows reported close, last, settlement, NSE Total Traded Qty, open interest, and executed-trade count. Percentages below are positive-value coverage, not bid/ask usability.

| Candidate | Rows | Close positive | Settlement positive | Last positive | Traded qty positive | OI positive | Trades positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIPLA | 608 | 100.0% | 97.9% | 48.0% | 31.1% | 47.0% | 31.1% |
| HDFCBANK | 529 | 100.0% | 97.9% | 83.2% | 70.7% | 82.8% | 70.7% |
| ICICIBANK | 646 | 100.0% | 95.0% | 57.6% | 44.0% | 57.4% | 44.0% |
| INFY | 817 | 100.0% | 97.1% | 87.1% | 69.5% | 87.0% | 69.5% |
| NTPC | 492 | 100.0% | 94.9% | 56.3% | 39.6% | 55.7% | 39.6% |
| POWERGRID | 530 | 100.0% | 96.0% | 48.5% | 32.3% | 47.7% | 32.3% |
| SUNPHARMA | 436 | 100.0% | 96.1% | 52.8% | 38.3% | 51.6% | 38.3% |
| TCS | 682 | 100.0% | 97.9% | 76.5% | 60.1% | 76.1% | 60.1% |

`TtlTradgVol` magnitude and `TtlTrfVal` magnitude were not used to rank candidates.

## Futures availability and spot consistency

The stock block contained 72 futures rows: three expiries per stock and date. Every stock futures expiry matched an observed option expiry, and every stock option expiry had a corresponding future. Close, settlement, last, and activity flags were retained; no final futures-implied `q` was calculated.

All 24 candidate surfaces had an independent CM `EQ` close and a unique F&O `UndrlygPric`. The observed difference `CM close - F&O UndrlygPric` was exactly zero for all 24 surfaces. This is an empirical result for these files, not an assumed identity.

## NIFTY reference status

NIFTY derivatives were present on all three dates and remained reference-only. Each date had 18 option expiries and three futures expiries; 45 longer-dated option expiries across the three dates lacked a matching NIFTY future. No CM equity record was used as NIFTY spot. A later frozen evaluation should obtain the index close from a separate official NSE historical-index source.

## Known missing quote fields

Free NSE bhavcopy does not provide historical bid, ask, bid size, or ask size. These were marked unavailable and never inferred. Bloomberg may later provide supplementary quote-quality evidence, but it is not required to proceed to candidate ranking and does not replace official NSE as the primary Stage A source.

## Explicit conclusions

- The deterministic official-NSE Stage A screen is complete and reproducible.
- All eight candidate stocks have three usable logical surfaces for support comparison.
- The current 108-grid is unsuitable as the final unchanged representation, especially at 180 DTE and the extreme wings.
- Candidate ranking can proceed using official NSE presence, support, price observations, activity, futures availability, missingness, and spot consistency while keeping quote quality separate.

## Explicit non-conclusions

- No candidate winner or final four-stock set has been selected.
- No 54-, 57-, 30-, or other replacement feature count has been frozen.
- Stage A market availability does not prove parameter identifiability or model superiority.
- No final 10,000-surface dataset exists.
- No ANN or PINN research training or frozen real-market evaluation exists.

## Gate status and next decision

```text
STAGE_A_NSE_SCREEN = COMPLETE
CANDIDATE_SELECTION = PENDING
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
REPLACEMENT_REPRESENTATION = OPEN
G2 = NOT_PASSED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_STARTED
```

The next decision is Stage A candidate ranking to select one primary underlying per sector. Common-support analysis across those four stocks then precedes the representation decision and G2.
