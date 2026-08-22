# R2 Representation Contract — Canonical Post-G2 Interface

Status: CANONICAL — 22 August 2026
Implementation: `src/r2_representation/` (tests: `tests/test_r2_representation.py`, 33 tests)
Tracking issue: #25

This document defines the single canonical software representation of the
frozen G2 research representation. It is the shared representation contract
for final synthetic-data generation, ANN/Model-2 dataset construction, and
frozen real-market evaluation. No pipeline may invent its own R2 ordering or
missing-data behavior.

```text
G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING
```

## 1. Scientific rationale

The predeclared self-governed R2-vs-R3 study
([G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md))
was executed once and sealed on 22 August 2026
([G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md](G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md)).
Rule 2 of the frozen stopping rule selected **R2**: R3's strong clean-data
improvement collapsed at every realistic noise level
(NO_MATERIAL_IMPROVEMENT at 0.5%, 1%, and 2%), and R3's added third expiry
rank contributed zero usable central-five slots on every NTPC development
date. Both candidates remain practically non-identifiable at market noise;
that finding is retained, not re-litigated. This interface formalizes the
frozen winner without reopening selection.

## 2. Canonical slot identity and exact order

Each nominal quote slot is identified by the key

```text
(expiry_rank, target_log_moneyness, option_type)
```

with exactly **20** canonical keys in the exact order below (the reviewed G2
study ordering, reused unchanged; identical to
`src/g2_r2r3/geometry.representation_slots(profile, "R2")`):

```text
slot  0: (1, -0.10, call)     slot 10: (1, -0.10, put)
slot  1: (1, -0.05, call)     slot 11: (1, -0.05, put)
slot  2: (1,  0.00, call)     slot 12: (1,  0.00, put)
slot  3: (1, +0.05, call)     slot 13: (1, +0.05, put)
slot  4: (1, +0.10, call)     slot 14: (1, +0.10, put)
slot  5: (2, -0.10, call)     slot 15: (2, -0.10, put)
slot  6: (2, -0.05, call)     slot 16: (2, -0.05, put)
slot  7: (2,  0.00, call)     slot 17: (2,  0.00, put)
slot  8: (2, +0.05, call)     slot 18: (2, +0.05, put)
slot  9: (2, +0.10, call)     slot 19: (2, +0.10, put)
```

The nesting is **option-type major (all calls, then all puts), then expiry
rank, then target log-moneyness ascending** — the same nesting convention as
the historical 108 grid, with the maturity axis being *eligible listed expiry
ranks with actual DTE* instead of fixed calendar DTEs. The exact sequence is
pinned literally in `tests/test_r2_representation.py`
(`test_adversarial_d_slot_order_locked_by_test`): any change to identity or
order fails the suite. `R2Surface` and every serialization path reject any
other ordering.

`expiry_rank` 1 and 2 are the first two **eligible** listed expiry ranks
under the official-NSE support/activity contract. Rank-2 maturity must
strictly exceed rank-1 maturity (listed expiries are chronological).

## 3. Mask semantics

- `mask[i] = True` — slot `i` carries a usable observation (a real quote
  selected under the official contract, or a synthetic slot).
- `mask[i] = False` — slot `i` is unsupported or unusable. Its stored price
  is exactly `0.0` (dense-serialization placeholder only), and no strike,
  raw price, or other provenance exists for it.
- Valid prices are strictly positive by contract (real activity-eligible
  closes and synthetic canonical prices are always > 0), so `0.0` can never
  collide with a real observation.
- **NaN/Inf are never allowed anywhere.** Missingness is never encoded as
  NaN; a missing slot is `mask=False` + `price=0.0`.
- Consumers must gate every numerical use of `prices` by `mask`
  (`valid_prices_array()` provides the gated view).
- Constructing or deserializing a surface with a non-zero value at a masked
  slot, a non-positive value at a valid slot, or any non-finite value raises
  `RepresentationContractError`.

### Prohibited imputation

A missing real quote must never be filled with: a Double Heston model price,
a Black-Scholes price, interpolation, extrapolation, or neighboring
observations. Real surfaces are constructed by
`build_real_surface(...)`, which masks unsupported slots; no code path in
this package writes any value into a masked slot. Any future non-primary
compatibility utility that needs dense arrays must read them through an
explicit, separately reviewed adapter and can never become canonical.

## 4. Maturity semantics

`maturities[i]` is the **actual** time-to-maturity in years of the expiry
rank owning slot `i`, computed as `DTE / 365` from the listed expiry date
and valuation date. Maturities are known for every slot of an eligible rank
(including masked quote slots) and are therefore always 20 finite positive
values, constant within a rank and strictly increasing across ranks. R2
never assumes a fixed-DTE grid.

## 5. Rate/carry semantics

`rates[i]` and `carries[i]` are the existing per-rank conditioning, reused
from the sealed G2 audit unchanged: continuous rate from the committed
hash-sealed RBI 91-day T-bill observation (with that contract's documented
carry-forward convention), and futures-implied carry
`rate - log(forward/spot)/maturity` from the matched future. They are
constant within a rank; the interface validates this and rejects
inconsistent vectors.

## 6. Normalization

`prices` are spot-normalized: `price / spot`. `spot` is carried per surface.
Synthetic surfaces use the G2 normalization spot `100.0`
(`SYNTHETIC_NORMALIZATION_SPOT`, equal to `src/g2_r2r3/frozen.SYNTHETIC_SPOT`
by test) — Black-Scholes homogeneity makes target-moneyness normalized
prices independent of the normalization spot. Real surfaces carry the NTPC
close as both normalization spot and pricing spot, and retain the exact raw
observed prices in metadata provenance. Denormalization
(`denormalized_prices_array()`) returns `price * spot` at valid slots and
`0.0` at masked slots; round-trips agree with raw values to float64 unit
round-off.

## 7. Synthetic behavior

`build_synthetic_surface(parameters, conditioning, surface_id=...)`:

- exactly 20 nominal slots, all `mask=True` (complete by construction);
- parameters are the canonical ten-vector in frozen order, validated against
  the canonical structural constraints (no silent clamping);
- prices come from the **unchanged frozen production pricer**
  (`src/double_heston.price_double_heston_surface`), priced per
  constant-conditioning rank piece at target-moneyness strikes
  `spot * exp(k)`;
- actual maturities and per-rank rate/carry conditioning preserved;
- the ten parameter values are stored in `metadata` under
  `parameters_canonical_order` for dataset/manifest use.

The final 10,000-surface dataset is **not** generated by this milestone;
only small test/smoke surfaces are.

## 8. Real-market behavior

`build_real_surface(date_id, audit_report=None)`:

- reuses the merged official-NSE quote-selection contract exactly
  (`src/g2_r2r3/market.audit_date`: raw UDiFF archives, Hungarian assignment
  with the 0.05 target gate, activity/moneyness/Black-IV eligibility on the
  matched-futures forward, sealed rate observations) — no new market logic;
- produces the same 20 canonical positions as synthetic R2;
- unavailable observations stay in their canonical positions with
  `mask=False`, price exactly `0.0`, and a recorded failure reason — never a
  model fill;
- actual listed expiry/DTE and market rate/carry conditioning retained;
- per-slot raw provenance retained (`actual_strikes`,
  `actual_log_moneyness`, `observed_raw_prices`, `failure_reasons`, with
  `null` at masked slots);
- only the five frozen NTPC development dates (2026-07-01, 07-08, 07-15,
  07-22, 07-29) are constructible; all remain DEVELOPMENT / DIAGNOSTIC and
  permanently excluded from final G8 (`development_date_excluded_from_g8:
  true` in metadata). Any other date raises
  `RealSurfaceNotConstructibleError`. G8-date construction is a later,
  separately controlled milestone and must not silently reuse this
  development mapping.
- Sealed usable counts reproduce exactly through this interface: 11/20
  (07-01), 18/20 (07-08), 19/20 (07-15), 18/20 (07-22), 12/20 (07-29);
  aggregate 78/100.

If a date's first two eligible ranks do not exist (rank absence is not slot
masking), the surface is not constructible and the constructor raises.

## 9. Serialization / schema

`src/r2_representation/serialization.py` defines the versioned payload
(round-trip tested bit-identically for all contract fields):

| field | content |
|---|---|
| `representation_name` | `FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE` |
| `representation_version` | `1.0` |
| `slot_keys` | 20 `[expiry_rank, target_log_moneyness, option_type]` triples in canonical order |
| `prices` | 20 spot-normalized prices (masked slots exactly `0.0`) |
| `mask` | 20 JSON booleans |
| `maturities` | 20 actual times-to-maturity (years) |
| `rates` | 20 rate-conditioning values |
| `carries` | 20 carry-conditioning values |
| `spot` | normalization spot |
| `surface_id` | unique surface identifier |
| `source` | provenance label (`synthetic_canonical_double_heston_production_pricer` / `real_nse_official_contract_development_date`) |
| `metadata` | JSON-safe provenance dict |

Rules: `allow_nan=False` everywhere; file output is deterministic
(`sort_keys`, indent 2). Payloads with the wrong name/version, wrong slot
identity/order, non-boolean masks, non-finite numbers, non-positive valid
prices, or non-zero masked prices are rejected at the schema boundary
(`validate_payload`). `dataset_manifest(...)` / `manifest_from_payload(...)`
wrap per-surface payloads for dataset manifests and train/validation/test
files, verifying one canonical slot order across all surfaces. Version
changes require an explicit migration; there is no silent cross-version
reading.

## 10. Compatibility policy (legacy-108 / R3 / Archive-2 audit)

A repository-wide audit (recorded in the Issue #25 milestone report)
classified every legacy coupling:

| item | classification | disposition |
|---|---|---|
| `src/constants.py` `LOG_MONEYNESS_GRID`/`MATURITY_DAYS_GRID`; `src/surface_grid.py` defaults; `src/dataset.py` `from_surface_frame` 108 enforcement | KEEP_AS_HISTORICAL (COMPATIBILITY_ONLY) | untouched this milestone; they serve the historical 108 pilot/smoke paths only. The canonical R2 contract does not import them (enforced by test). Future R2 dataset loaders must be built on `src/r2_representation`, not by re-pointing these. |
| `src/synthetic_dataset.py` `build_surface_grid(spot)` in `_generate_and_save` | ADAPT_TO_R2 (REQUIRES_LATER_WORK) | the generic 108-row generator; the final synthetic-generation milestone must generate through `build_synthetic_surface` instead. Not modified here (no broad refactors; final generation is a separate controlled milestone). |
| `configs/ann_dataset_FIRST_RESEARCH.yaml` (108-dimension plan, `PREPARED_NOT_EXECUTED`) | DEPRECATE / REQUIRES_LATER_WORK | superseded by this contract; must be rewritten (not edited in place) when the final dataset manifest is frozen. |
| `src/run_pinn_*.py`, `src/train*.py` input sizing | KEEP_AS_HISTORICAL | all derive input size from data (`features.shape[1]`); they will accept R2-shaped datasets once R2 loaders exist. |
| legacy 108-row `surfaces.csv` artifacts under `outputs/`, `market_data_audit/stage_a/derived/*108*` | KEEP_AS_HISTORICAL | evidence only; never reinterpretable as R2 (length-108 vectors and payloads are rejected with an explicit diagnostic). |
| rejected R3 (30-slot) study geometry | KEEP_AS_HISTORICAL | preserved inside the sealed `src/g2_r2r3` study; 30-length vectors rejected as non-canonical. |
| Archive-2 `src/dheston` variable-length surfaces | COMPATIBILITY_ONLY | experimental donor code; different ordering and semantics; interop only via explicit named adapters (none required here). |

There is deliberately **no 108→R2 adapter**: the legacy grid's fixed-DTE
axis cannot be losslessly converted to actual ranked expiries, and silent
reinterpretation is prohibited. Adapters, where later needed, are explicit
and one-way (legacy → analysis), never into the canonical R2 path.

## 11. Relationship to G2 evidence

- Slot order, central-five targets, rank count, and mask philosophy are the
  sealed study's, re-stated here as production constants and verified equal
  to `src/g2_r2r3/frozen.py`/`geometry.py` by test (no runtime coupling on
  the synthetic path).
- Real-market construction is literally the sealed audit contract, re-run
  through the same code; the interface reproduces the sealed usable counts.
- The production pricer, canonical parameter order/constraints/bounds, and
  all G2 evidence/protocol documents are untouched.

## 12. Known limitations

1. The real constructor is wired only to the five NTPC development dates
   (sealed raw-archive mapping); G8-date construction requires registering
   G8 raw archives in a later, separately controlled milestone.
2. Final synthetic dataset generation, split freezing, ANN/Model-2 loading,
   and evaluation consumers are future milestones; this contract is the
   foundation they must use, not yet wired into `src/dataset.py` /
   `src/synthetic_dataset.py` (classified above).
3. `representation_version` `1.0` has one migration path only: explicit
   code change plus schema tests. Payloads from unknown versions are
   rejected outright.
4. Real per-slot provenance arrays hold `null` at masked slots; consumers
   must not confuse `actual_log_moneyness` provenance (realized strike
   moneyness) with the canonical `target_log_moneyness` slot identity.
5. The interface expresses practical non-identifiability honestly: it makes
   missing information explicit; it does not (and cannot) add identifying
   information.

## 13. Adversarial review

Two independent adversarial passes ran before the PR (representation/mask/
ordering semantics; serialization/legacy-compatibility). Verdicts on the
five predeclared attack questions:

- **A (same slot meaning synthetic vs real):** no material issue. The
  listed-vs-representation rank mapping was verified empirically for
  rank-skip scenarios; target-vs-actual strike semantics are retained as
  per-slot provenance (limitation 4). One provenance defect found and
  fixed: a caller-supplied `audit_report` for a different date is now
  rejected instead of silently mislabeling the surface's date.
- **B (masked quote becoming an observation):** no material issue. All
  construction/serialization/accessor paths enforce the 0.0-placeholder and
  strict-positivity invariants; `valid_prices_array()` physically excludes
  masked positions.
- **C (legacy 108 passing as R2):** no material issue. Every entry point
  rejects 108-length and 30-length data with explicit diagnostics; no frame
  loader exists; slot identity is never inferred from numbers.
- **D (silent ordering drift):** no material issue. The exact 20-key
  sequence is pinned literally by test and cross-checked against the sealed
  G2 ordering; every path validates identity and order.
- **E (serialization losing conditioning):** round-trip verified
  bit-identical for all contract fields on synthetic and real surfaces. Two
  defects found and fixed: NaN/Infinity JSON literals can no longer enter
  through metadata on the read path (`parse_constant` guard + metadata
  finiteness validation), and non-JSON metadata types now raise
  `RepresentationContractError` instead of bare `TypeError`; the manifest
  reader now validates top-level slot keys and rejects duplicate
  `surface_id`s. Regression tests cover each fix.

## Reproducing the contract checks

```bash
python -m pytest tests/test_r2_representation.py     # 28 focused tests
python -m pytest tests/test_g2_r2r3_harness.py       # sealed G2 study regression
```
