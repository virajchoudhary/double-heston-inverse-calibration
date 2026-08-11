# G2 Common-Support Analysis

## Gate result

**G2 = NOT_PASSED.** The market-support evidence uniquely proposes two direct listed expiry positions (`near`, `middle`), five symmetric log-moneyness nodes `[-0.10, -0.05, 0.00, +0.05, +0.10]`, and both calls and puts. That surface geometry contains 20 normalized-price features plus the two actual maturity coordinates `T_near` and `T_middle`, a **22-feature surface subtotal**. The final inverse input dimension is not frozen because the conditioning treatment for risk-free rate and dividend/carry remains unresolved.

This is a representation decision only. No final 10,000-surface dataset, ANN/PINN training, or later gate was run.

## Balanced market panel and provenance

The analysis used exactly `4` stocks × `3` dates = `12` stock/date surfaces, `36` expiry slices, and `2,446` option rows:

- stocks: `NTPC | CIPLA | INFY | HDFCBANK`;
- dates: `2026-07-01 | 2026-07-15 | 2026-07-22`; and
- source: only the six canonical official-NSE CM/FO manifest identities.

The Power-only dates `2026-07-08` and `2026-07-29`, all backup stocks, NIFTY, and Bloomberg were excluded. Securities were never pooled or averaged. Before analysis, archive identity, official URL, archive/member bytes, archive SHA-256, extracted CSV SHA-256, schema, date, and market were checked fail-closed against the canonical manifest.

## Market-support decision rule declared before proposing geometry

A geometry candidate passes market support only when all of the following hold:

1. all four stocks and all three dates are present with no excluded date/security;
2. no maturity or strike extrapolation is required;
3. maturity nodes are direct listed-expiry positions unless a separately validated maturity-price interpolation contract exists;
4. at least `2` maturity positions remain, preserving term-structure information;
5. every retained moneyness node is bracketed for every stock/date/expiry-position/option-type slice;
6. active bracketing is at least `75%` overall and in the worst stock, date, expiry position, and option type;
7. the adjacent-strike interpolation bracket is at most `0.05` log-moneyness;
8. the moneyness grid is symmetric and zero-centered;
9. both option types remain unless a parity/carry contract is independently justified; and
10. among passing candidates in the same representation family, select the unique maximal symmetric contiguous moneyness set under strict set inclusion.

The implemented maximal-set rule derives `relative_near_mid_central5` and is invariant to candidate ordering; it does not optimize a feature count or use absolute NSE traded-quantity magnitude. Passing this geometry rule is necessary but not sufficient for G2: a carry-conditioned inverse contract must also be frozen and validated.

## Maturity support

All four stocks shared the same direct listed DTE schedules on each date: `27|55|90`, `13|41|76`, and `6|34|69`. Relative expiry positions are therefore direct on all 12 surfaces and remain implementable on future dates by sorting revised/actual expiries, while carrying the actual `T` coordinates into the model.

| Relative position | Direct support | Observed DTE values | DTE range |
|---|---:|---|---:|
| near | 12/12 (100.0%) | `6|13|27` | 6–27 |
| middle | 12/12 (100.0%) | `34|41|55` | 34–55 |
| far | 12/12 (100.0%) | `69|76|90` | 69–90 |

Fixed-DTE alternatives were classified independently on every surface:

| Target DTE | Direct | Bounded interpolation | Unsupported/extrapolated | Worst nearest distance | Largest bracket span |
|---:|---:|---:|---:|---:|---:|
| 7 | 0/12 | 4/12 | 8/12 | 20 d | 28 d |
| 14 | 0/12 | 8/12 | 4/12 | 13 d | 28 d |
| 27 | 4/12 | 8/12 | 0/12 | 14 d | 28 d |
| 30 | 0/12 | 12/12 | 0/12 | 11 d | 28 d |
| 34 | 4/12 | 8/12 | 0/12 | 7 d | 28 d |
| 41 | 4/12 | 8/12 | 0/12 | 14 d | 35 d |
| 45 | 0/12 | 12/12 | 0/12 | 11 d | 35 d |
| 55 | 4/12 | 8/12 | 0/12 | 14 d | 35 d |
| 60 | 0/12 | 12/12 | 0/12 | 16 d | 35 d |
| 69 | 4/12 | 8/12 | 0/12 | 14 d | 35 d |
| 76 | 4/12 | 4/12 | 4/12 | 14 d | 35 d |
| 90 | 4/12 | 0/12 | 8/12 | 21 d | 0 d |
| 180 | 0/12 | 0/12 | 12/12 | 111 d | — |

`180` DTE is unsupported on 12/12 surfaces. The old `7`, `14`, and `90` nodes require extrapolation on part of the panel. Although `30` and `60` DTE are bounded on all surfaces, they are direct on 0/12 and have worst nearest-expiry distances of 11 and 16 days; no maturity-price or total-variance interpolation contract (including rate/dividend inputs) is frozen. They are not promoted merely to keep a fixed grid.

The far listed expiry is direct but fails central activity support: across the five proposed nodes its expiry-position active support is only `16.7%–29.2%`. Near is 100% active at every proposed node; near+middle retains 87.5%–97.9% overall active support and a 75.0% worst balanced view. Therefore near+middle is the largest maturity geometry that passes the market-support rule.

## Moneyness support

The proposed market price field is official NSE `ClsPric`: it is positive on `100.0%` of the 2,446 selected rows, versus `70.4%` for `LastPric` and `97.0%` for `SttlmPric`. Last and settlement remain diagnostics, not silent substitutes; historical bid/ask is unavailable and is not inferred.

A fixed target `k = log(K/S)` almost never equals a listed strike exactly: direct exact-node support is 0% at every candidate node. The relevant distinction is therefore bounded adjacent-strike interpolation versus extrapolation. The policy is linear interpolation of normalized price in `K/S` between the two adjacent listed strikes for the same security, valuation date, revised/actual expiry, and option type. No cross-security, cross-expiry, or strike extrapolation is allowed.

The table below uses only the proposed near+middle positions (48 option-type slices per node):

| log(K/S) | Observed bracket | Exact direct | Active bracket | Worst stock | Worst date | Worst position | Worst call/put | Max bracket width |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | nan |
| -0.20 | 50.0% | 0.0% | 33.3% | 0.0% | 25.0% | 25.0% | 25.0% | 0.033902 |
| -0.15 | 95.8% | 0.0% | 64.6% | 33.3% | 56.2% | 41.7% | 54.2% | 0.046520 |
| -0.10 | 100.0% | 0.0% | 87.5% | 83.3% | 81.2% | 75.0% | 75.0% | 0.022473 |
| -0.05 | 100.0% | 0.0% | 97.9% | 91.7% | 93.8% | 95.8% | 95.8% | 0.019803 |
| +0.00 | 100.0% | 0.0% | 95.8% | 83.3% | 87.5% | 91.7% | 95.8% | 0.020203 |
| +0.05 | 100.0% | 0.0% | 93.8% | 83.3% | 81.2% | 87.5% | 91.7% | 0.017700 |
| +0.10 | 100.0% | 0.0% | 89.6% | 75.0% | 75.0% | 79.2% | 87.5% | 0.016807 |
| +0.15 | 75.0% | 0.0% | 56.2% | 16.7% | 37.5% | 50.0% | 54.2% | 0.031749 |
| +0.20 | 50.0% | 0.0% | 45.8% | 0.0% | 37.5% | 33.3% | 45.8% | 0.071594 |
| +0.30 | 16.7% | 0.0% | 16.7% | 0.0% | 6.2% | 16.7% | 12.5% | 0.056552 |

Only `[-0.10, -0.05, 0.00, +0.05, +0.10]` form the largest symmetric contiguous set that has 100% observed bracketing, meets every 75% active-support view, and keeps the largest adjacent-strike bracket below 0.05. The `±0.15` extension fails 100% observed support; `±0.20` and `±0.30` are materially weaker. The extreme `-0.30` node has no observed bracket anywhere in the panel.

## Call and put finding

Both calls and puts are retained. Calls and puts exist in every selected near/middle slice and each option type meets the support rule. The Double Heston engine can derive puts by put-call parity, so synthetic prices are theoretically redundant conditional on risk-free rate and dividend yield. The market preprocessing contract does not yet freeze those carry inputs, and observed calls/puts contain distinct activity and microstructure information. Removing either type would therefore impose an unsupported carry/parity assumption rather than a scientifically demonstrated simplification.

## Candidate comparison

| Candidate | Maturity direct/interpolated/unsupported | Observed/active moneyness | Worst active view | Price + coordinate inputs | Decision |
|---|---:|---:|---:|---:|---|
| `provisional_108` | 5.6% / 50.0% / 44.4% | 65.7% / 46.1% | 0.0% | 108 + 0 = 108 | REJECTED |
| `fixed_30_60_central5` | 0.0% / 100.0% / 0.0% | 100.0% / 69.4% | 16.7% | 20 + 0 = 20 | REJECTED |
| `fixed_34_central5` | 33.3% / 66.7% / 0.0% | 100.0% / 92.9% | 75.0% | 10 + 0 = 10 | REJECTED |
| `fixed_34_69_central5` | 33.3% / 66.7% / 0.0% | 100.0% / 69.4% | 16.7% | 20 + 0 = 20 | REJECTED |
| `relative_near_central5` | 100.0% / 0.0% / 0.0% | 100.0% / 100.0% | 100.0% | 10 + 1 = 11 | REJECTED |
| `relative_near_mid_central3` | 100.0% / 0.0% / 0.0% | 100.0% / 95.8% | 81.2% | 12 + 2 = 14 | PASS_BUT_DOMINATED |
| `relative_near_mid_central5` | 100.0% / 0.0% / 0.0% | 100.0% / 92.9% | 75.0% | 20 + 2 = 22 | PROPOSED_GEOMETRY |
| `relative_near_mid_central7` | 100.0% / 0.0% / 0.0% | 95.8% / 83.6% | 16.7% | 28 + 2 = 30 | REJECTED |
| `relative_near_mid_far_central5` | 100.0% / 0.0% / 0.0% | 100.0% / 69.4% | 16.7% | 30 + 3 = 33 | REJECTED |
| `relative_near_mid_call_only` | 100.0% / 0.0% / 0.0% | 100.0% / 90.8% | 50.0% | 10 + 2 = 12 | REJECTED |
| `relative_near_mid_put_only` | 100.0% / 0.0% / 0.0% | 100.0% / 95.0% | 66.7% | 10 + 2 = 12 | REJECTED |

## Proposed surface geometry (not a frozen inverse contract)

- **Representation type:** coordinate-aware relative listed-expiry surface.
- **Maturity representation:** first and second revised/actual listed expiries (`near`, `middle`), sorted by actual expiry; include `T_near` and `T_middle` using calendar DTE/365.
- **Moneyness nodes:** `[-0.10, -0.05, 0.00, +0.05, +0.10]`, where `k = log(K/S)` and `S` is the independent same-date CM close.
- **Option types:** calls and puts.
- **Price features:** 2 expiry positions × 5 nodes × 2 option types = **20**.
- **Coordinate features:** 2 actual maturity fractions.
- **Surface subtotal:** **22** features before any carry-conditioning coordinates or transformation.
- **Final inverse input dimension:** **UNRESOLVED / NOT FROZEN**.
- **Ordering:** `[T_near, T_middle]`, then option-major (`call`, `put`), expiry-major (`near`, `middle`), moneyness ascending.
- **Market price and normalization:** official NSE `ClsPric / S` (`normalized_close`), with `S` equal to the independent same-date CM close; synthetic theoretical prices later use the same `/ S` scale. `T=DTE/365` is already dimensionless and receives no cross-stock normalization at this gate. Do not normalize or rank by absolute NSE traded quantity.
- **Strike interpolation:** bounded adjacent listed strikes only, linear normalized price in `K/S`, separately within security/date/expiry/type; maximum accepted log-moneyness bracket width 0.05.
- **Maturity interpolation:** none.
- **Extrapolation:** prohibited for both maturity and strike.
- **Missing support:** mark the surface unavailable; do not impute, borrow another security, or silently reduce the mask.
- **Market-support applicability:** the geometry passes for all four primaries on all three dates. Future dates must rerun the same support/preprocessing checks; `near`/`middle` are reproducible positions, not assumed fixed DTEs.

Compared with the rejected provisional 108-price representation, the proposed geometry removes the unsupported 180-DTE node, all fixed-DTE semantics, the far listed expiry, and the weak `±0.20/±0.30` wings. Its supported subtotal is 20 price inputs plus 2 maturity coordinates, while retaining both option types; this is not yet the final ANN dimension.

## Blocking ambiguity and minimum additional evidence

The Double Heston engine prices conditionally on risk-free rate `r` and dividend yield/carry `q`. The current synthetic contract fixes `r=0.02` and `q=0.01`, but Stage A intentionally did not freeze a real-market rate source, dividend/carry source, tenor mapping, or a validated forward/discount normalization. Allowing market `r/q` to vary without conditioning can confound carry with the ten Heston targets, so the 22-feature surface subtotal cannot by itself close G2.

Minimum evidence to reopen and pass G2:

1. predeclare and provenance-validate either (a) rate/dividend-carry coordinates aligned to near/middle maturities, (b) discount-factor and forward coordinates, or (c) a forward/discount normalization that demonstrably removes carry;
2. show availability and deterministic preprocessing for all 12 stock/date surfaces without maturity extrapolation;
3. update the declared ordering and exact total dimension for the chosen conditioning contract; and
4. run a local Jacobian-rank/conditioning and noisy multi-start recovery check on the proposed two-maturity × five-strike geometry, treating carry according to that contract. If conditioning is inadequate, reopen the geometry rather than force G2.

## Downstream compatibility (identified, not implemented)

The later implementation must update `src/constants.py`, `src/surface_grid.py`, `src/dataset.py`, `src/synthetic_dataset.py`, `src/models.py`, `configs/ann_dataset_FIRST_RESEARCH.yaml`, `configs/ann_baseline.yaml`, dataset/shape/model tests, and the real-market preprocessing entry point. The Double Heston pricing interface already accepts quote-aligned maturity arrays, both option types, `r`, and `q`; its ten-parameter target order remains `kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow, kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast`. No downstream file was changed in this task.

## Reproducibility and preservation

Run `python scripts/run_g2_common_support_analysis.py`. The four CSVs and three figures are generated under the ignored Stage A derived tree. The script validates provenance before writing and verifies the eight canonical Stage A output hashes after writing.

| Canonical Stage A output | SHA-256 preserved in this run |
|---|---|
| `acquisition_manifest.csv` | `E062841DC09DD0821A8391A9A28167B18E2D7617C712DA9D550F5B70CF4C9D8E` |
| `surface_summary.csv` | `25AC3E6BADFBC9C85D71159B93BCA1385FCB83BB93523E3736387C4D0BF066EA` |
| `expiry_coverage.csv` | `61CFF054884CECCBF640CFA8051F2CD2E91629AFF0CF78554542BB0A82BF5636` |
| `moneyness_coverage.csv` | `A482F0398F55A74C40D4ED98AF812E6A9C2F473F45CA894D2D947AC03B03D68D` |
| `candidate_grid_support.csv` | `25C18BDE999EBD1A1EAAE6E7D316EFCB2CE6D8E9E3F475D237C79E86AAF3651F` |
| `futures_availability.csv` | `2BBE783896E98B78E61EB74C9A66FF7BBAFB976BEB31D06ADE310C475C4B996C` |
| `spot_consistency.csv` | `FBDD5A1C152FC82E601E04D7BEB2D4F7BF734671914F4CEB1B239FFB8F9A64E1` |
| `universe_presence.csv` | `E20E86985896FAD3B1A92282DD54A38A1E6DED39341D549F1A873A1ECA8F5987` |

```text
G2 = NOT_PASSED
PROPOSED_SURFACE_INPUT_SUBTOTAL = 22
FINAL_TOTAL_INPUT_DIMENSION = UNRESOLVED
CARRY_CONDITIONING = UNRESOLVED
FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```
