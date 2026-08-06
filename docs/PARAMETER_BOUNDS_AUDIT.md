# Provisional Parameter-Bounds Audit

Status date: 06 August 2026

## Audit design

`src/audit_parameter_bounds.py` generated exactly 5,000 raw vectors using a deterministic SciPy Latin hypercube with seed `20260806` over the provisional empirical sampling ranges. Every vector was evaluated against positivity, hard numerical bounds, slow-fast ordering, both strict Feller constraints, individual correlation bounds, and the joint correlation disk.

Boundary proximity uses normalized distances fixed before the results were observed. A deterministic ten-stratum sample of 250 accepted vectors was then priced with the production 64-node engine on six maturities, seven strikes, and both option types: 84 quotes per surface and 21,000 prices in total. This was an audit-only sample, not an ANN research dataset.

## Acceptance and proximity results

| Metric | Result |
|---|---:|
| Raw candidates | 5,000 |
| Accepted | 2,776 (`55.52%`) |
| Rejected | 2,224 (`44.48%`) |
| Accepted near any declared boundary | 907 / 2,776 (`32.6729%`) |
| Accepted near either Feller boundary | 196 / 2,776 (`7.0605%`) |
| Accepted near a hard bound | 748 / 2,776 (`26.9452%`) |
| Accepted near the correlation disk | 0 / 2,776 (`0.00%`) |
| Accepted with weak slow-fast separation | 258 / 2,776 (`9.2939%`) |

Rejection uses the full 5,000-vector raw population. Proximity rates use only the 2,776 accepted, constraint-valid vectors; all rejected vectors retain their negative margins and rejection reasons but have accepted-near flags set to false. These proportions are sampling-design diagnostics, not estimates of market parameter frequencies.

## Priced-surface results

All 21,000 prices were finite. Across the 250 surfaces there were zero no-arbitrage, call-strike-monotonicity, put-strike-monotonicity, or strike-convexity failures. Normalized prices ranged from `3.2116531656356525e-14` to `0.507713948497342`; normalized ATM call prices ranged from `0.017258319766710953` to `0.3609447008938868`. Every surface produced six distinct rounded ATM maturity features. Total surface-generation runtime was about `20.12` seconds on the final primary run.

Implied-volatility range analysis was skipped because the repository does not contain a validated IV inversion. Calendar monotonicity was not imposed as a universal rule because the correct comparison depends on discounting, carry, strike/forward alignment, and option definition.

## Sampling-design and identifiability exposure

Seventeen priced parameter pairs had normalized surface RMSE at or below `1e-3` while remaining at least `0.10` apart in normalized parameter space. Every qualifying pair identity and distance is retained in `priced_surface_summary.csv`. This is a warning about inverse-label stability and sampling design, not a claim of statistical identifiability or non-identifiability.

Uniform raw sampling over the current provisional ranges is not recommended for ANN generation: rejection is material, boundary-near concentration is high, and some separated parameter vectors generate similar controlled surfaces.

## Recommendations

- **KEEP:** Retain strict numerical validity constraints and the documented hard safety envelope.
- **REVIEW:** Do not approve the existing empirical ranges for broad ANN generation without domain review.
- **SPLIT_SAMPLING_RANGE:** Separate interior training samples from explicitly labelled boundary-near challenge samples.
- **REVIEW:** Revisit Feller-margin and slow-fast-separation sampling margins before generation.
- **REQUIRE_FINANCIAL_REVIEW:** Obtain domain approval for economically plausible training, noise-test, and out-of-distribution ranges.
- **REVIEW:** Add a validated implied-volatility inversion before using smile-range diagnostics.

The source provisional file was not modified. `configs/parameter_sampling_REVIEWED.yaml` records the current evidence, retained provisional values, challenge/noise/OOD sections, and review status for every value. Detailed audit tables are under `outputs/parameter_bounds_audit/`.
