# G8 Pre-Execution Audit

Status: **COMPLETE — PROTOCOL FROZEN; DATA ACQUISITION REMAINS BLOCKED** — 25 August 2026.
Repository audited: `C:\dh_g8_protocol`.
Branch and base verified immediately before this audit: `research/g8-final-eval-protocol` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; the tracked tree was clean before creating the five declared freeze artifacts.

## Verdict

```text
G8_PROTOCOL = FROZEN_BEFORE_ANY_REAL_MARKET_OUTCOME
G8_DATA = NOT_ACQUIRED
UNTOUCHED_DATA_AVAILABILITY = NOT_PROVEN
NTPC_2026-07-15_PILOT_CLASSIFICATION = DEVELOPMENT
FINAL_REAL_MARKET_EVALUATION = NOT_STARTED
```

The repository does **not** contain provenance for a suitable untouched G8 market panel. This is not solved by inventing dates or downloading archives now. The companion protocol defines the deterministic acquisition/selection contract to execute only after every participating inverse method is frozen.

## Evidence inventory

| Evidence block | Market data touched | Classification |
|---|---|---|
| Stage A official-NSE availability screen | Official CM/F&O UDiFF for 2026-07-01/15/22; eight stock candidates plus reference-only NIFTY | Development/support |
| Stage A candidate ranking and Power tie-break | Original three-date panel for all eight candidates; 2026-07-08/29 additions for NTPC and POWERGRID | Development/instrument selection |
| NTPC single-stock pilot | NTPC FO on 2026-07-15; NTPC EQ closes through completed 2026-07-28 near expiry; dated RBI 91-day T-bill; corporate-action query | **DEVELOPMENT PILOT** |
| NTPC multi-date calibration | NTPC CM/F&O on 2026-07-01/15/22; RBI observations of 2026-07-01/15 | Development/calibration |
| NTPC reparameterization/cap replay | Hash-frozen rows and outputs from the above pilots | Development diagnostics |
| G2 R2-vs-R3 market support | All five NTPC development dates (78/100 usable R2 slots) | Development/support |
| Final R2 synthetic dataset and comparison | Synthetic-only 10,000 surfaces; Models 1/2 and traditional calibration | Synthetic research results |

The NTPC 15-Jul-2026 experiment produced useful preprocessing and interface evidence, but it was calibrated and interpreted before G8. It must not be relabeled as final unseen-market evaluation.

## Development exclusion registry

- All eight Stage A candidates are excluded as observations on 1, 15, and 22 Jul 2026.
- NTPC and POWERGRID are excluded as observations on 8 and 29 Jul 2026.
- NTPC cash closes and corporate-action context through 28 Jul 2026 are excluded because they entered realized-volatility construction.
- NIFTY remains reference-only and is prohibited in G8.
- The July RBI observations do not make a future surface development data. For each G8 date, however, the latest official auction on or before that date must be acquired and hash-sealed; an older July artifact may not be silently substituted when a later eligible observation exists.
- Every G8 valuation date must be on or after **2026-09-30**, after all July development expiries used by the pilot have completed.
- Acquisition must compare proposed instrument identities against preserved development manifests and fail closed on an unresolved overlap.

The registry excludes used *observations* and derived summaries. It does not treat the public ticker labels NTPC, CIPLA, INFY, or HDFCBANK as permanently contaminated; later dates can remain unseen while preserving the already reviewed sector design.

## Interface audit

- `src/r2_representation` is the sole canonical R2 interface: exactly 20 nominal slots, two ranked expiries, central-five targets, calls and puts, actual maturity conditioning, and genuine boolean masks. Missing slots are `mask=False`, price exactly `0.0`; NaN, interpolation, extrapolation, and model fills are rejected.
- `src/r2_representation.real.build_real_surface` deliberately accepts only the five NTPC development dates. It cannot silently construct a G8 date; a separately reviewed G8 adapter is required.
- The production Double Heston pricer is frozen at canonical parameter order and 64 Gauss-Laguerre nodes. Cross-stack positional parameter passing remains prohibited.
- The completed synthetic comparison uses one shared 100-feature R2 builder, frozen Model 1/Model 2 configurations and seeds, three-start traditional calibration, and mask-aware repricing metrics.
- Real-market weight updates are quarantined by `src/dheston.real_market_policy`. G8 additionally requires inference/checkpoint loads only.
- Forward-Black IV inversion and activity cleaning were independently reviewed in the NTPC pilot. They may be reused only with the same fail-closed semantics; free UDiFF still supplies no bid/ask history.
- Existing G8 references consistently say `G8 = NOT_STARTED` and protect the five NTPC dates. No stale protocol claims that G8 has run.

## Current blocker

No tracked or locally present G8 archive exists in this worktree. July raw NSE/RBI bytes are ignored local-chain-of-custody artifacts and are also absent from this clean checkout; their provenance survives only in tracked reports/manifests. Therefore the repository cannot prove that a post-September panel is present, complete, official, unchanged, or disjoint from prior use. The identity validator does not replace the required later contract-key overlap check.

The six ignored neural checkpoint files are also absent from this clean checkout. Their paths and SHA-256 values are sealed in the machine contract from the committed R2 evidence manifests; restoration and byte-for-byte verification are mandatory before data acquisition.

Exact acquisition requirements are machine-readable in `configs/g8_final_real_market.yaml` and normative in `docs/G8_FINAL_REAL_MARKET_PROTOCOL.md`. In summary:

1. Freeze every participating inverse method first; absent Model 3 may then be reported unavailable rather than delaying G8.
2. Acquire official NSE UDiFF CM and F&O ZIP archives only for dates scanned under the deterministic rule starting 2026-09-30.
3. Preserve official URL, UTC retrieval time, filename, size, ZIP integrity, ZIP SHA-256, member name, extracted CSV SHA-256, encoding/delimiter, and date.
4. Preserve and field-validate the latest prior RBI 91-day auction response and normalized extract for each valuation date.
5. Record structural support, masks, and rejection reasons only. Do not compute or inspect a calibration/pricing outcome during selection.
6. Compare proposed option/futures contract keys against preserved development manifests and fail closed on unresolved overlap; symbol/date checks alone are insufficient.
7. Freeze the data manifest by hash before running any model. Any mismatch aborts.
