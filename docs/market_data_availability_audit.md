# Stage A Market-Data Availability Audit

## Purpose and boundary

Stage A is a preparation and availability audit for real-market inputs to the Double Heston inverse-calibration capstone. It asks whether Bloomberg exports contain sufficiently complete and usable option, futures, and spot observations for later research. No Bloomberg observations are present yet, this audit makes no market conclusion, and it does not authorize model training.

This work does not change the frozen Double Heston pricing engine, parameter bounds, sampler, synthetic generator, ANN/PINN implementation, or the existing 108-input contract. The market representation decision remains **OPEN**. In particular, Stage A does not define a 54-feature, 57-feature, or other replacement neural representation.

## Stage A universe

Ranked candidates are grouped by sector:

- Power: NTPC, POWERGRID
- Healthcare/Pharma: SUNPHARMA, CIPLA
- IT: INFY, TCS
- Financial/Banking: ICICIBANK, HDFCBANK

NIFTY is collected separately as a non-ranked, reference-only surface. It can provide context about index-option availability, but it must not be included in candidate rankings or used as a substitute for a missing candidate observation.

The audit valuation dates are:

- 2026-07-01
- 2026-07-15
- 2026-07-22

## Surface definition

One surface is exactly one underlying plus one valuation date, containing **all near-, mid-, and far-expiry option slices together**. An individual expiry is a slice of a surface, not a surface. Near, mid, and far labels are assigned by the actual ordered expiry dates within an underlying-date surface, not by fixed DTE cutoffs.

## Expected raw files and semantic fields

Each underlying-date directory contains `options_raw.xlsx`, `futures_raw.xlsx`, `spot_raw.xlsx`, and `collection_manifest.yaml`. The canonical semantic column names are:

- Options required: `valuation_date`, `underlying`, `expiry_date`, `strike`, `option_type`, `bid`, `ask`, `volume`, `open_interest`.
- Options optional: `security_id`, `last_price`, `currency`, `observation_timestamp`.
- Futures required: `valuation_date`, `underlying`, `expiry_date`, `futures_price`.
- Futures optional: `security_id`, `bid`, `ask`, `settlement_price`, `currency`, `observation_timestamp`.
- Spot required: `valuation_date`, `underlying`, `spot`.
- Spot optional: `security_id`, `currency`, `observation_timestamp`.

Candidate Bloomberg field mnemonics in the Stage A configuration are provisional collection hints only. They must be checked against the available Bloomberg function, security type, entitlement, and export layout before collection. The canonical semantic names above are the loader contract; exports may be mapped to them without changing the raw observations.

No fake Bloomberg rows should be placed in the Stage A input tree. Tiny mock tables are confined to unit tests.

## Price usability and activity are separate

Price usability is quote-based. The provisional basic flag requires a positive spot and strike, a future expiry, finite non-negative bid, finite positive ask, a non-crossed quote (`ask >= bid`), and a positive bid/ask mid. No spread threshold is selected at Stage A; relative spread is reported for later audit decisions.

Activity is reported independently through whether volume and open interest are present and whether each is positive. A price-usable observation must **never** fail solely because daily volume or open interest is zero or missing. Conversely, positive activity does not repair an invalid price quote.

## Derived audit fields

The audit layer may derive only observational diagnostics:

- calendar DTE and `T = DTE / 365`;
- `K/S` and `log(K/S)`;
- bid/ask mid, normalized mid (`mid/S`), and relative bid-ask spread (`(ask-bid)/mid`);
- independent price-usability and volume/open-interest activity flags;
- near/mid/far labels from actual expiry order; and
- surface-level option and futures coverage summaries.

All calculations return new tables and leave loaded raw tables unchanged.

## Futures-implied-carry availability audit

Stage A separately checks whether a usable futures observation can be matched by underlying, valuation date, and maturity with spot and a separately supplied risk-free rate. Where `F > 0`, `S > 0`, and `T > 0`, the helper may calculate:

`q_impl = r - ln(F/S) / T`

This is an availability diagnostic only. Stage A does not choose futures-implied carry as the final carry convention, does not choose a risk-free-rate source, and does not silently fill unmatched maturities.

## No-extrapolation rule

Coverage is calculated only from actually observed strikes and expiries. The audit must not extrapolate quotes, carry, activity, or coverage beyond the observed strike/expiry domain. The current candidate grids are recorded only to compare observed availability with the existing research grid; they do not authorize interpolation, extrapolation, or a replacement input layout.

## Preparing the local drop folders

The ignored local input tree can be created without any Bloomberg dependency:

```powershell
python scripts/create_market_data_audit_stage_a_structure.py
```

The generator creates candidate and NIFTY date directories plus pending `collection_manifest.yaml` files. It deliberately does not create `.xlsx` workbooks or observations and does not overwrite an existing manifest. Raw Stage A content under `market_data_audit/stage_a/` is ignored by Git.

After exports arrive, place them under:

```text
market_data_audit/stage_a/candidates/<sector>/<underlying>/<date>/
market_data_audit/stage_a/reference/NIFTY/<date>/
```

Run the loader against one underlying-date directory. A successful load validates the configured required columns; it does not assert that Bloomberg coverage is sufficient or decide the eventual representation.
