# G2 Self-Governed Representation Protocol

Status: PREDECLARED — READY FOR IMPLEMENTATION

Date: 22 August 2026

This document replaces the former mentor-approval blocker for G2 with a bounded, self-governed research protocol. The project team owns the technical design and will preserve all negative findings. No result-dependent threshold changes are allowed after this protocol is merged.

## 1. Purpose

G2 will no longer require evidence that the canonical ten-parameter Double Heston inverse problem is uniquely identifiable under realistic market noise. The overnight identifiability audit showed that this is not a defensible universal requirement: a representation can be locally full rank and still be practically non-identifiable at market tolerance.

G2 now asks:

> Which representation is most defensible for the primary study, given both real-market support and practical information content, and what residual ambiguity remains after that representation is frozen?

A G2 pass therefore means **representation selection and freeze with ambiguity explicitly characterized**, not proof of universal unique parameter recovery.

## 2. Fixed scientific contracts

The following remain unchanged during this milestone:

- canonical ten-parameter order;
- production Double Heston pricer;
- canonical positivity, Feller, correlation, and slow/fast-ordering constraints;
- reviewed parameter bounds and sampling populations;
- calibration optimizer/objective unless an existing diagnostic already defines a comparison arm;
- synthetic truth as the only basis for parameter-recovery claims;
- no ANN or inverse-network research training;
- no real-market neural weight updating;
- all previously used July NTPC dates remain DEVELOPMENT / DIAGNOSTIC and excluded from final G8.

No priors, temporal smoothing, realized-volatility supervision, new bounds, new optimizer, or new sector may be introduced to make G2 pass.

## 3. Candidate representations

The final unchanged 108-grid is not a candidate.

### R2 — ranked two-expiry central-five baseline

For each surface:

- first two eligible listed expiry ranks;
- central log-moneyness targets `[-0.10, -0.05, 0.00, +0.05, +0.10]`;
- calls and puts;
- spot-normalized option price;
- actual time-to-maturity supplied explicitly;
- existing carry/rate conditioning retained;
- no interpolation or extrapolation beyond the existing quote-selection contract.

Nominal price slots: `2 x 5 x 2 = 20`.

### R3 — ranked three-expiry masked central-five

Same contract as R2, but with the first three listed expiry ranks.

Nominal price slots: `3 x 5 x 2 = 30`.

If an expiry/quote fails the existing support/activity/usability contract, that slot is **masked**, not interpolated, extrapolated, or filled using a model price. Actual maturity remains an explicit input/conditioning variable.

R3 is therefore a maximum-size masked representation, not a claim that every real surface contains 30 equally reliable observations.

## 4. Development market-support panel

Use all five already-designated NTPC development dates, subject only to existing official-NSE support/activity rules:

- 2026-07-01
- 2026-07-08
- 2026-07-15
- 2026-07-22
- 2026-07-29

Do not select or drop dates based on the new experiment's results. Report per-date and aggregate:

- eligible expiry ranks;
- usable central-five call/put slots;
- mask rate;
- actual DTE distribution;
- activity/support failures;
- quote-selection failures;
- resulting R2/R3 surface completeness.

All five dates remain ineligible for final G8 regardless of outcome.

## 5. Synthetic identifiability panel

Use the existing reviewed parameter contract and a deterministic predeclared truth panel:

- retain the four standing representative G2 truth cases;
- add 16 deterministic reviewed-interior truths using **truth-selection seed `20260822`**;
- select those 16 before running any R2/R3 outcome computation;
- do not replace difficult truths after results are seen.

The experiment manifest must record the exact 20 truth vectors and their source identifiers before the first representation-comparison result is written.

For each truth and each candidate representation, run the same diagnostic stack.

### Noise levels

- 0%
- 0.5%
- 1.0%
- 2.0%

### Frozen randomization

- truth-selection seed: `20260822`;
- multi-start seed: `20260823`;
- noise base seed: `20260824`.

Noise must be keyed deterministically by `(truth_id, expiry_rank, moneyness, option_type, noise_level)` so every quote slot common to R2 and R3 receives the identical perturbation. R3-only third-expiry slots receive deterministic additional draws from the same keyed scheme. Do not regenerate noise until a more favorable realization appears.

### Multi-start calibration

- 12 deterministic starts per truth per noise level, generated from seed `20260823`;
- identical starts across R2 and R3;
- identical optimizer, bounds, constraints, and stopping rules across candidates;
- retain all solutions, including failures and boundary hits.

## 6. Required diagnostics

For R2 and R3 report separately:

### Market support

- usable quote count / mask rate;
- DTE distribution;
- per-date completeness;
- existing support/activity-rule pass rate.

### Local information

- Jacobian singular values;
- smallest singular value;
- condition number;
- practical numerical rank under the existing tolerance convention;
- per-parameter normalized sensitivity;
- weakest singular directions.

### Global / practical recovery

- range-scaled parameter RMSE;
- per-parameter error;
- repricing RMSE;
- valid-solution rate;
- boundary-hit rate;
- number of materially separated near-equivalent clusters;
- median and maximum pairwise parameter separation;
- factor-swap/tie-breaking behavior;
- runtime and convergence status.

No candidate may be selected from repricing RMSE alone.

## 7. Decision rules

The purpose is to choose a representation, not to keep searching indefinitely for uniqueness.

### Hard requirements

A candidate is eligible to freeze only if:

1. it uses no unsupported interpolation/extrapolation;
2. its real-market construction is reproducible from the existing official-NSE quote-selection contract;
3. it does not change the canonical ten-parameter or pricing contracts;
4. it preserves synthetic/real separation and G8 protection; and
5. all missing observations are explicit through masking rather than silently imputed from the model under test.

### Comparative evidence

Use the existing project interpretation bands where applicable:

- **strong improvement:** at least 25% improvement in both median and maximum dispersion with fewer separated clusters;
- **partial improvement:** at least 10% improvement in both dispersion measures with no increase in clusters;
- existing 5% real-market holdout-deterioration ceiling remains a guardrail where a directly comparable holdout price metric is available.

Also report Jacobian conditioning, noise RMSE, and boundary pressure. These are supporting evidence, not single-metric pass/fail switches.

### Freeze rule

1. If R3 is market-supported under masking and shows strong or partial information improvement over R2 without violating the holdout guardrail, freeze R3.
2. If R3 does not improve practical information materially, freeze R2 as the simpler market-supported representation.
3. If both remain practically non-identifiable at realistic noise, **do not reopen an unlimited representation search**. Freeze the better-supported candidate under rules 1-2 and record `PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING`.
4. Escalate only if both representations fail the hard market-construction requirements, not merely because unique recovery is absent.

This is the stopping rule that prevents post-hoc representation hunting.

## 8. G2 completion labels

At the end of this bounded milestone record exactly one:

- `G2 = PASSED_REPRESENTATION_FROZEN_IDENTIFIABILITY_ACCEPTABLE`
- `G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY`
- `G2 = FAILED_MARKET_CONSTRUCTION_REQUIREMENTS`

The second label is a valid scientific outcome and allows the primary comparison to proceed, provided ambiguity is reported honestly throughout the study.

## 9. After G2

Only after the representation is frozen:

1. formalize the production representation interface;
2. regenerate/revalidate the final synthetic surface contract;
3. generate the final research dataset;
4. run integrity/validity checks;
5. freeze train/validation/test splits;
6. run traditional calibration vs Model 1 ANN vs Model 2 constraint+repricing-informed inverse network;
7. report truth recovery, repricing, validity, stability, tolerance/equivalence-class recovery, noise robustness, and runtime separately;
8. freeze neural models; then
9. evaluate on untouched G8 real-market dates without weight updating.

A genuine PDE-informed Model 3 remains optional and must not be used to delay the primary comparison.

## 10. Reproducibility requirements

The G2 implementation PR must include:

- a machine-readable experiment manifest;
- fixed seeds and truth identifiers;
- exact candidate representation definitions;
- exact Git SHA;
- commands used;
- all raw and summarized outputs;
- failed runs retained;
- no manual deletion of outliers or failed optimizations; and
- a final decision file applying the rules above without changing them after results are known.
