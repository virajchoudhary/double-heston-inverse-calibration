# Volatile Stock Selection Audit for Phase 3B

Classification: `VOLATILE_STOCK_PHASE3B_READY_POWERGRID`

Audit date: 2026-08-29

## Mentor interpretation

Five years is the broad historical context across market regimes. Three months is the focused three-calendar-month realized-volatility lookback ending at the final valid close strictly before the option valuation date. Three months is not an option-expiry requirement; listed option expiry and tau are screened separately when the CALL surface is built.

## Frozen methodology

- History request: 2021-01-01 through 2026-07-21, official NSE EQ-series security-wise bhavdata only.
- Return: `ln(S_t / S_(t-1))` from linked closes. A previous-close linkage break invalidates affected windows; no corporate-action adjustment is invented.
- Three-month RV: sample standard deviation of all valid daily log returns in the calendar interval, annualized by `sqrt(252)`. A window requires at least 50 returns; exact close/return counts are reported.
- Five-year context: the same rolling three-calendar-month estimator at every valid endpoint, summarized by median, p75, p90, maximum, and valid-window count. Percentile rank is the empirical percentage of valid rolling observations less than or equal to the candidate RV.
- Selection precedes all pricing-model evaluation. Primary ranking is RV_3M among pairs passing provenance, history, CALL, surface, domain, and split gates; percentile rank is supporting context.

## Official acquisition and integrity

1,368 accepted official NSE daily reports were preserved under the external market-data audit tree. Each accepted response passed exact schema and embedded-date checks before publication. Weekends, holidays, and NSE holiday endpoints that returned a prior-day report were not published under the requested date. One official 2022-08-08 response was an OOXML workbook behind a `.csv` URL; its original bytes were preserved and parsed under the same 15-column schema.

Nine symbols have 1,368 EQ closes from 2021-01-01 through 2026-07-21. ADANIPOWER has 1,164 EQ closes because NSE reported it under non-EQ series during portions of the horizon; BE observations were not substituted. Valid candidate windows still have complete linked EQ closes.

## Candidate audit

`CALL count` means genuine activity-eligible STO/CE rows with positive close, volume, open interest, and trade count. PUT rows are counted only for information and are never converted. `Domain status` includes normalized spot, every active CALL K/S and tau, continuous r, futures-implied q, variance-state bounds, minimum surface support, and deterministic split feasibility.

| Symbol | Option date | 5Y start | 5Y end | Obs | 3M start | 3M end | Closes/returns | RV_3M | 5Y median | p75 | p90 | max | rolling n | percentile | CALLs | PUTs | strikes | expiries | K/S range | r range | q range | Domain status / reason |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ADANIENT | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.369371 | 0.377159 | 0.537268 | 0.667058 | 1.416474 | 958 | 46.56% | 33 | 86 | 32 | 2 | 0.763456-1.081563 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| ADANIENT | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.339332 | 0.377159 | 0.537268 | 0.667058 | 1.416474 | 958 | 34.55% | 40 | 79 | 31 | 2 | 0.825240-1.142639 | 0.053165-0.053273 | -0.061452--0.026443 | FAIL: q outside frozen domain |
| ADANIENT | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.338170 | 0.377159 | 0.537268 | 0.667058 | 1.416474 | 958 | 33.40% | 48 | 79 | 34 | 2 | 0.762461-1.143692 | 0.053192-0.053301 | 0.016761-0.138396 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| ADANIPOWER | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1164 | 2026-04-01 | 2026-06-30 | 60/59 | 0.386740 | 0.493735 | 0.659599 | 0.800208 | 2.989177 | 713 | 34.64% | 27 | 56 | 23 | 2 | 0.881601-1.234241 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| ADANIPOWER | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1164 | 2026-04-15 | 2026-07-14 | 62/61 | 0.384951 | 0.493735 | 0.659599 | 0.800208 | 2.989177 | 713 | 33.80% | 36 | 50 | 26 | 2 | 0.867699-1.278714 | 0.053165-0.053273 | -0.015280--0.009485 | FAIL: q outside frozen domain |
| ADANIPOWER | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1164 | 2026-04-22 | 2026-07-21 | 62/61 | 0.358320 | 0.493735 | 0.659599 | 0.800208 | 2.989177 | 713 | 23.00% | 39 | 52 | 25 | 3 | 0.875778-1.290620 | 0.053057-0.053301 | -0.011648-0.095376 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| CIPLA | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.269041 | 0.224944 | 0.256230 | 0.275752 | 0.300126 | 958 | 85.91% | 24 | 100 | 21 | 2 | 0.823780-1.139562 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| CIPLA | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.267246 | 0.224944 | 0.256230 | 0.275752 | 0.300126 | 958 | 84.13% | 29 | 104 | 25 | 2 | 0.904033-1.140473 | 0.053165-0.053273 | -0.015290--0.005241 | FAIL: q outside frozen domain |
| CIPLA | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.269359 | 0.224944 | 0.256230 | 0.275752 | 0.300126 | 958 | 86.22% | 44 | 105 | 29 | 2 | 0.918663-1.158929 | 0.053192-0.053301 | 0.011549-0.117818 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| HDFCBANK | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.260669 | 0.200554 | 0.272543 | 0.353189 | 1.454698 | 978 | 73.31% | 48 | 86 | 35 | 2 | 0.828990-1.155561 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| HDFCBANK | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.228954 | 0.200554 | 0.272543 | 0.353189 | 1.454698 | 978 | 68.51% | 55 | 88 | 35 | 3 | 0.797106-1.177264 | 0.053030-0.053273 | -0.004052-0.034343 | FAIL: q outside frozen domain |
| HDFCBANK | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.250668 | 0.200554 | 0.272543 | 0.353189 | 1.454698 | 978 | 71.78% | 72 | 95 | 41 | 3 | 0.902875-1.274646 | 0.053057-0.053301 | -0.035484--0.013944 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| ICICIBANK | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.228471 | 0.196604 | 0.236951 | 0.292348 | 0.443164 | 958 | 71.09% | 31 | 107 | 26 | 2 | 0.920423-1.130599 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| ICICIBANK | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.193073 | 0.196604 | 0.236951 | 0.292348 | 0.443164 | 958 | 46.03% | 44 | 108 | 28 | 2 | 0.917949-1.129784 | 0.053165-0.053273 | 0.035436-0.076454 | FAIL: q outside frozen domain |
| ICICIBANK | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.188884 | 0.196604 | 0.236951 | 0.292348 | 0.443164 | 958 | 43.01% | 61 | 116 | 34 | 3 | 0.867634-1.138336 | 0.053057-0.053301 | -0.018439-0.076316 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| INFY | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.376205 | 0.246193 | 0.274765 | 0.314779 | 0.393014 | 958 | 98.54% | 97 | 147 | 80 | 3 | 0.852532-1.476708 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| INFY | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.389488 | 0.246193 | 0.274765 | 0.314779 | 0.393014 | 958 | 99.16% | 101 | 133 | 77 | 3 | 0.817616-1.351854 | 0.053030-0.053273 | 0.079372-0.105427 | FAIL: active CALL K/S outside frozen domain; q outside frozen domain |
| INFY | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.386231 | 0.246193 | 0.274765 | 0.314779 | 0.393014 | 958 | 98.75% | 108 | 133 | 79 | 3 | 0.798403-1.363939 | 0.053057-0.053301 | -0.079542-0.074641 | FAIL: active CALL K/S outside frozen domain; active CALL tau outside frozen domain; q outside frozen domain |
| NTPC | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.207202 | 0.234483 | 0.270749 | 0.304308 | 0.460315 | 958 | 34.34% | 29 | 98 | 27 | 2 | 0.894104-1.285275 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| NTPC | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.197363 | 0.234483 | 0.270749 | 0.304308 | 0.460315 | 958 | 29.96% | 35 | 73 | 26 | 2 | 0.929287-1.219689 | 0.053165-0.053273 | -0.044400-0.014470 | FAIL: q outside frozen domain |
| NTPC | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.192330 | 0.234483 | 0.270749 | 0.304308 | 0.460315 | 958 | 28.39% | 37 | 75 | 24 | 2 | 0.941512-1.184023 | 0.053192-0.053301 | 0.036360-0.122765 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| POWERGRID | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.202122 | 0.247672 | 0.290159 | 0.656615 | 0.717669 | 958 | 26.83% | 23 | 88 | 21 | 2 | 0.930273-1.199791 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| POWERGRID | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.176854 | 0.247672 | 0.290159 | 0.656615 | 0.717669 | 958 | 8.98% | 30 | 87 | 23 | 3 | 0.917349-1.211258 | 0.053030-0.053273 | 0.012990-0.028278 | PASS |
| POWERGRID | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.181534 | 0.247672 | 0.290159 | 0.656615 | 0.717669 | 958 | 12.94% | 32 | 87 | 19 | 2 | 0.933287-1.106118 | 0.053192-0.053301 | 0.049482-0.105893 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| SUNPHARMA | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.238400 | 0.193075 | 0.253270 | 0.266565 | 0.293839 | 958 | 65.34% | 18 | 68 | 17 | 2 | 0.964527-1.157432 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| SUNPHARMA | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.219845 | 0.193075 | 0.253270 | 0.266565 | 0.293839 | 958 | 57.10% | 28 | 75 | 20 | 2 | 0.840379-1.127338 | 0.053165-0.053273 | 0.001765-0.040328 | FAIL: q outside frozen domain |
| SUNPHARMA | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.219775 | 0.193075 | 0.253270 | 0.266565 | 0.293839 | 958 | 56.99% | 36 | 78 | 22 | 3 | 0.782295-1.132270 | 0.053057-0.053301 | 0.003738-0.175528 | FAIL: active CALL tau outside frozen domain; q outside frozen domain |
| TCS | 2026-07-01 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-01 | 2026-06-30 | 60/59 | 0.347911 | 0.197280 | 0.240958 | 0.261727 | 0.366968 | 958 | 98.43% | 58 | 109 | 44 | 2 | 0.907899-1.412287 | unavailable | unavailable | FAIL: no preserved official risk-free-rate observation; r/q domain unverified |
| TCS | 2026-07-15 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-15 | 2026-07-14 | 62/61 | 0.361200 | 0.197280 | 0.240958 | 0.261727 | 0.366968 | 958 | 99.69% | 74 | 114 | 44 | 3 | 0.822218-1.315549 | 0.053030-0.053273 | 0.129318-0.214048 | FAIL: active CALL K/S outside frozen domain; q outside frozen domain |
| TCS | 2026-07-22 | 2021-01-01 | 2026-07-21 | 1368 | 2026-04-22 | 2026-07-21 | 62/61 | 0.362917 | 0.197280 | 0.240958 | 0.261727 | 0.366968 | 958 | 99.69% | 73 | 116 | 43 | 3 | 0.760766-1.304171 | 0.053057-0.053301 | -0.065039-0.083375 | FAIL: active CALL K/S outside frozen domain; active CALL tau outside frozen domain; q outside frozen domain |

## Selection

POWERGRID (Power Grid Corporation of India Limited) on 2026-07-15 is the only candidate/date pair that passes every frozen eligibility gate. Its three-calendar-month interval is 2026-04-15 through 2026-07-14 (62 closes, 61 returns), with annualized RV_3M `0.1768540817014945`. Its empirical five-year rolling-three-month percentile rank is `8.977035490605427%`.

The pair satisfies the protocol's mentor-aligned comparative selection rule because it is the highest-RV pair among the fully eligible set (the set contains only POWERGRID 2026-07-15). The secondary context is adverse: this window is not unusually volatile relative to POWERGRID's own five-year history. That limitation is disclosed rather than re-defining the window or ranking after inspection.

The surface contains 30 active genuine calls, 23 distinct strikes, and 3 expiries. Active K/S is `0.9173494834342715-1.2112575703598147`; tau is `0.03561643835616438-0.20821917808219179`; continuous r is `0.05303014284116871-0.05327342725949597`; q is `0.01299023301335582-0.028278390376017006`. Normalized spot is 1.0. All are inside the frozen domain, and both frozen variance-state calibration ranges have positive width.

## Proposed CALL-only calibration/holdout split

This proposal is not executed. For each of the two activity-richest expiries, sort distinct active CALL contracts by strike. Select deterministic strike-rank quantiles `[0, 0.25, 0.50, 0.75, 1]` using `round(q * (n-1))`. The two extremes are untouched holdout; the three inner quantiles are calibration. Ties are resolved by stable ascending strike order. The rule uses only contract coordinates/activity, preserves two maturities and the observed strike range, creates no duplicates, and never uses prices or model errors for membership.

| Role | Expiry | Strike quantile | Strike | K/S | Close | Volume | OI | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| holdout | 2026-07-28 | 0.00 | 257.50 | 0.917349 | 26.35 | 6 | 5700 | 2 |
| calibration | 2026-07-28 | 0.25 | 282.50 | 1.006413 | 4.00 | 922 | 769500 | 755 |
| calibration | 2026-07-28 | 0.50 | 295.00 | 1.050944 | 0.90 | 672 | 1238800 | 334 |
| calibration | 2026-07-28 | 0.75 | 310.00 | 1.104382 | 0.10 | 303 | 2449100 | 159 |
| holdout | 2026-07-28 | 1.00 | 340.00 | 1.211258 | 0.05 | 2 | 38000 | 1 |
| holdout | 2026-08-25 | 0.00 | 275.00 | 0.979694 | 12.00 | 2 | 9500 | 2 |
| calibration | 2026-08-25 | 0.25 | 280.00 | 0.997506 | 9.35 | 131 | 216600 | 46 |
| calibration | 2026-08-25 | 0.50 | 290.00 | 1.033131 | 5.15 | 106 | 399000 | 83 |
| calibration | 2026-08-25 | 0.75 | 310.00 | 1.104382 | 1.45 | 9 | 43700 | 8 |
| holdout | 2026-08-25 | 1.00 | 315.00 | 1.122195 | 1.05 | 28 | 266000 | 16 |

Result: 6 calibration observations for only `v_slow` and `v_fast`, plus 4 untouched holdout observations.

## Rejected candidates

Every 2026-07-01 pair is rejected because no exact stable official risk-free-rate artifact was acquired for that date, so r/q compatibility remains unverified. No proxy or inferred value is used. For 2026-07-22, the preserved 2026-07-15 RBI auction result is used as the last available official observation; all pairs fail at least the frozen minimum tau because active six-day calls are present, with additional K/S or q failures where reported. The 2026-07-15 exact pair-specific reasons are listed below.

- ADANIENT 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- ADANIENT 2026-07-15: q outside frozen domain.
- ADANIENT 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- ADANIPOWER 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- ADANIPOWER 2026-07-15: q outside frozen domain.
- ADANIPOWER 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- CIPLA 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- CIPLA 2026-07-15: q outside frozen domain.
- CIPLA 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- HDFCBANK 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- HDFCBANK 2026-07-15: q outside frozen domain.
- HDFCBANK 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- ICICIBANK 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- ICICIBANK 2026-07-15: q outside frozen domain.
- ICICIBANK 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- INFY 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- INFY 2026-07-15: active CALL K/S outside frozen domain; q outside frozen domain.
- INFY 2026-07-22: active CALL K/S outside frozen domain; active CALL tau outside frozen domain; q outside frozen domain.
- NTPC 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- NTPC 2026-07-15: q outside frozen domain.
- NTPC 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- POWERGRID 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- POWERGRID 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- SUNPHARMA 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- SUNPHARMA 2026-07-15: q outside frozen domain.
- SUNPHARMA 2026-07-22: active CALL tau outside frozen domain; q outside frozen domain.
- TCS 2026-07-01: no preserved official risk-free-rate observation; r/q domain unverified.
- TCS 2026-07-15: active CALL K/S outside frozen domain; q outside frozen domain.
- TCS 2026-07-22: active CALL K/S outside frozen domain; active CALL tau outside frozen domain; q outside frozen domain.

## Provenance

- Underlying raw reports: `C:\ann_inverse_calibration\market_data_audit\stage_a\raw\nse\underlying_history\`
- Underlying raw-report manifest: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\volatile_stock_selection\underlying_raw_report_manifest.csv`
- Per-symbol acquisition manifest and immutable extracts: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\volatile_stock_selection\`
- Candidate evidence: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\volatile_stock_selection\candidate_audit.csv`
- Proposed split: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\volatile_stock_selection\proposed_call_split.csv`
- Option/futures raw reports: `C:\ann_inverse_calibration\market_data_audit\stage_a\raw\nse\2026-07-01\`, `2026-07-15\`, and `2026-07-22\`
- Option acquisition manifest: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\acquisition_manifest.csv`
- Official RBI 91-day T-bill observation: `C:\ann_inverse_calibration\market_data_audit\stage_a\derived\ntpc_single_stock_pilot\raw\rbi_91_day_tbill_observation_20260715.json`

No bulk raw report, output, checkpoint, dataset, ZIP, cache, notebook, or scientific Phase 1/2/3A artifact is intended for Git.

## Prohibited work confirmation

No PINN training or retraining occurred. No Black-Scholes, Heston, Double Heston, PINN, calibration, holdout, or market-pricing evaluation occurred. No model error or holdout result influenced stock/date selection. No q value was clamped, replaced, or extrapolated. No put was converted and no missing option observation was fabricated.
