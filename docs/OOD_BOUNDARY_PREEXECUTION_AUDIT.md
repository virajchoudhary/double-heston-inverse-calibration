# OOD and Boundary Protocol — Pre-Execution Audit

Audit date: **25 August 2026**. Branch: `research/ood-boundary-protocol`.
Base: `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`. Worktree was clean before
editing. This is an additive protocol milestone: completed primary experiment
files are read-only provenance, not objects to repair or reinterpret.

## Scope and outcome firewall

The purpose is to freeze evaluation cohorts and decision rules **before Model 3
research results exist**. The audit deliberately did not open the values in
`synthetic_test_comparison.csv`, `parameter_metrics.json`,
`repricing_metrics.json`, `validity_metrics.json`, seed-result CSVs, or
traditional result CSVs. It recorded their paths and SHA-256 identities only.
Those artifacts may later serve as immutable ID-baseline denominators; they
cannot alter cohort difficulty, tolerances, materiality thresholds, or method
selection. No real-market observation or weight update participates in this
protocol.

## A. Frozen R2 train/validation/test ranges

`data/final_r2_clean_10000/surfaces.jsonl` has SHA-256
`148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`, exactly
10,000 whole surfaces, and stored splits 7,500/1,250/1,250. A target-panel-only
read found nearly identical parameter support across splits. Observed extrema
include slow kappa `[0.1985,2.3996]`, slow theta `[0.0173,0.2000]`, slow v0
`[0.0199,0.2400]`, fast kappa `[1.4999,9.9992]`, fast theta
`[0.0119,0.1599]`, and fast v0 `[0.0144,0.2000]`. These facts describe synthetic
truths only and do not reveal any model outcome.

## B. Parameter bounds and interior sampler

`configs/parameter_bounds_PROVISIONAL.yaml` fixes hard numerical safety bounds.
Canonical validation requires strict positivity, `kappa_slow < kappa_fast`,
strict positive per-factor Feller gaps, correlations strictly inside
`(-1,1)`, and joint disk radius `<1`; implementation is `src/constraints.py`.
The reviewed sampler uses latent-coordinate Latin hypercube draws with
conditional transforms and retains every candidate/rejection
(`src/audit_reviewed_sampling.py`). Its ordinary interior/wide populations use a
shared near-boundary union gate. The global historical status remains
`NEEDS_SAMPLER_CORRECTION`; this new protocol does not upgrade that status.

## C. Prior boundary/OOD ideas

`configs/parameter_sampling_REVIEWED.yaml` already declared but did not generate
a four-regime boundary challenge (near Feller, weak factor separation, hard
bound, correlation disk) and an evaluation-only disjoint high-tail
`kappa_fast` OOD population. R2's final generation contract explicitly excluded
both from the clean core. This protocol preserves those ideas as separate,
deterministic evaluation cohorts rather than silently mixing them into training.

## D. Representation constraints

R2 is frozen at 20 canonical slots: option-type major, expiry ranks one/two,
then five central log-moneyness values. Slot identity/order, spot-normalized
prices, actual maturity, rate/carry conditioning, version, and source checks are
enforced by `src/r2_representation/contract.py`, `surface.py`, and
`serialization.py`. Legacy-108 and rejected-R3 vectors cannot pass through the
canonical boundary. The OOD work does not reopen R2 versus R3.

## E. Masks and incomplete surfaces

A valid slot has finite positive price; a masked slot has genuine boolean
`mask=False`, price exactly `0.0`, no NaN/Inf, and no imputation. Maturity,
rate, and carry remain known for all nominal slots. `R2Surface` rejects truthy
coercion and nonzero masked placeholders. Missing-surface evaluation therefore
changes only the input mask/placeholder fields and retains the complete parent
separately for counterfactual clean repricing.

## F. Evaluation metrics

The primary comparison froze parameter recovery, constraint validity,
production-pricer repricing, identifiability-aware tolerance reporting,
stability, and runtime (`src/r2_primary/evaluation.py`). Range and standardizer
scales come from train truths only. Repricing success remains distinct from
parameter recovery because practical non-identifiability is retained. This
protocol reuses those definitions, adds paired missing/clean degradation, and
makes post-hoc OOD/ID ratios explicit.

## G. Primary interfaces

- Ordinary ANN / Model 1: `models.ann_model.ANNInverseCalibrator`, unconstrained
  physical-vector output after inverse standardization; validity measured.
- Constraint/repricing-informed Model 2:
  `models.pinn_model.PhysicsInformedInverseCalibrator`, constraint-map head;
  float64 vectorized repricing term.
- Traditional: `src/r2_primary/calibration.py` calls the unchanged canonical
  calibration module with three starts, retained failures, and deterministic
  representative selection by lowest objective then start index.
- Future Model 3: accepted only as an adapter returning an `(N,10)` finite
  matrix aligned to manifest surface IDs and canonical order. No architecture,
  training result, or claim exists.

## H. Distinctness and feasibility check

Boundary stress targets constraint geometry. Distribution shift broadens to two
admissible regimes away from central concentration: safe low-slow-kappa/high
slow variance/v0 and disjoint high fast-kappa tail. Conditioning shift holds
the accepted wide-valid parameter screen while using long maturities plus
rate/carry supports disjoint from training's lattice. Missingness changes mask
semantics without changing R2. At the audited production-pricer speed of about
16 ms per 20-quote surface, 360 clean pricing calls are lightweight; no neural
training/calibration is run here. Future traditional calibration has a frozen
minimum 60-surface feasibility subset so expensive comparison remains bounded.

## I. Freeze state at this document's creation

No OOD research surface, selected panel, cohort manifest, replay artifact, or
Model 3 prediction existed in this branch. Protocol settings, seeds, quotas,
thresholds, and interpretation rules were fixed before any such output. Cohort
generation is gated on commit/push plus explicit remote-checkpoint confirmation.
