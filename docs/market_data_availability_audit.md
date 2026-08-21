# Stage A Market-Data Availability Audit

## Purpose and boundary

Stage A is a preparation and availability audit for real-market inputs to the Double Heston inverse-calibration capstone. The primary source is official NSE daily CM and F&O UDiFF bhavcopy. Bloomberg is not required to continue; it may later provide supplementary historical bid/ask and quote-size evidence.

This audit does not change the frozen Double Heston pricing engine, parameter bounds, sampler, synthetic generator, ANN/PINN implementation, or neural input contract. It does not rank candidates, calculate implied volatility or final futures-implied carry, interpolate or extrapolate observations, or freeze a replacement representation.

## Universe and dates

Candidate stocks are:

- Power: NTPC, POWERGRID
- Healthcare/Pharma: SUNPHARMA, CIPLA
- IT: INFY, TCS
- Financial/Banking: ICICIBANK, HDFCBANK

NIFTY is a separate, non-ranked reference. The completed screen covers 2026-07-01, 2026-07-15, and 2026-07-22.

## Primary official NSE files

The deterministic acquisition layer accepts only these official conventions:

```text
CM: BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
FO: BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
```

CM UDiFF supplies the independent `EQ` cash-market close for each stock. F&O UDiFF supplies stock options (`STO`), stock futures (`STF`), NIFTY index options (`IDO`), and NIFTY index futures (`IDF`). NIFTY spot/index close requires a separate official NSE historical-index source; an unrelated CM equity record must not be substituted.

Every raw archive is retained unchanged under the ignored Stage A tree with its official URL, retrieval timestamp or explicit filesystem-time fallback, filename, byte size, ZIP SHA-256, ZIP-integrity result, member name, extracted CSV SHA-256, encoding, delimiter, and trading date. Existing raw evidence is validated and never overwritten. Third-party mirrors and access-control bypasses are prohibited.

## Surface definition

One surface is exactly one `underlying + valuation_date`, containing all observed expiry slices together. An individual expiry is a slice, not a surface. Near, mid, far, and any additional labels follow actual ordered expiry dates rather than fixed DTE cutoffs.

## Preserve NSE semantics

Exact NSE raw field names are retained. In particular:

- `TtlTradgVol` is labelled NSE **Total Traded Qty**. Its value may establish reported/not-reported and zero/nonzero activity, but it is not called contracts, lots, or shares and its magnitude is not used for cross-stock ranking.
- `TtlTrfVal` is labelled NSE **Total Traded Value**. No denomination or scale is assumed and its magnitude is not used for ranking.
- `NewBrdLotQty` is retained as **Market Lot Size**; traded lots are not inferred from it.
- `XpryDt` and `FininstrmActlXpryDt` are both preserved. Derived DTE uses the actual expiry when present, falls back to the original expiry only when required, and records `expiry_fields_match`.

## Three separate evidence concepts

### Price observations

Close, last, and settlement are reported and tested independently for presence and positivity. A price observation does not require positive volume or open interest and is not described as bid/ask quote usability.

### Activity

`TtlTradgVol`, `OpnIntrst`, and `TtlNbOfTxsExctd` are each reported through separate present and positive flags. Unresolved raw quantities are not converted or compared by magnitude across stocks.

### Historical quote quality

Free NSE bhavcopy contains no historical bid, ask, bid-size, or ask-size fields. Those flags remain false rather than inferred. Bloomberg may later supplement this layer, but it does not replace official NSE as the primary Stage A acquisition source.

## Allowed diagnostics

The audit derives only observational diagnostics:

- valuation date, actual expiry, calendar DTE, and `T = DTE / 365`;
- independent CM close and F&O `UndrlygPric`, including their difference;
- `K/S`, `log(K/S)`, option type, and ordered expiry slot;
- normalized close and settlement where spot is valid;
- separate price-observation and activity flags;
- expiry, strike, ATM-bracketing, futures-alignment, and grid-support summaries.

No interpolation or extrapolation is performed. Candidate moneyness and maturity nodes are comparison points only, not a completed tensor design.

## Completed evidence and open gates

The official-NSE screen produced 24 candidate stock surfaces and 4,740 stock-option rows. All eight candidates were present on all three dates. All 24 CM closes matched the corresponding unique F&O `UndrlygPric`; all stock futures expiries aligned with stock-option expiries; and no selected stock expiry-field mismatch was observed.

Stage A rejects the current 108-grid as the final unchanged representation because 180 DTE is unsupported and extreme wings have sparse observed support. It does not freeze a 54-, 57-, 30-, or other replacement feature count. Candidate selection is pending and `G2 = NOT_PASSED`. See [STAGE_A_NSE_RESULTS.md](STAGE_A_NSE_RESULTS.md).

```text
STAGE_A_NSE_SCREEN = COMPLETE
CANDIDATE_SELECTION = PENDING
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
REPLACEMENT_REPRESENTATION = OPEN
G2 = NOT_PASSED
```
