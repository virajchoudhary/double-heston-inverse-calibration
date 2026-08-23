# R2 Primary Comparison Protocol (FROZEN)

Status: **FROZEN_BEFORE_ANY_PRIMARY_RESEARCH_TRAINING** — 23 August 2026.
Machine-readable twin: `configs/r2_primary_comparison_FINAL.yaml` (the two are
kept consistent by `tests/test_r2_primary_comparison_protocol.py`).
Branch: `research/primary-r2-model-comparison` (base: canonical main `fdbdc35`).
Pre-training audit: `docs/R2_PRIMARY_COMPARISON_PRE_TRAINING_AUDIT.md`.

This protocol commits the complete scientific design of the primary Double
Heston comparison **before any research training or calibration result
exists**. Nothing here may change after results are observed. Implementation
repair (section 11) adapts code to this design; it never alters it.

## 1. Scientific question

Given an inherently ill-conditioned Double Heston inverse problem, does a
constraint-informed + differentiable-repricing-informed inverse network
produce more stable, structurally valid, and useful parameter representatives
than (0) traditional numerical calibration and (1) an ordinary ANN inverse
model?

Retained framing: practical non-identifiability is a research finding, not a
bug to hide. Repricing quality is **never** evidence of unique parameter
recovery. Method 2 is **not** described as a PDE-informed PINN; it contains no
learned forward-PDE residual. Archive-2's broken PDE loss is quarantined
(Issue #20) and not imported. Optional genuine PDE-informed Model 3 is out of
scope.

## 2. Frozen data identity

- Dataset: `data/final_r2_clean_10000/surfaces.jsonl`
- SHA-256: `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`
- Status marker: `FINAL_R2_CLEAN_10000_RESEARCH_SYNTHETIC_TRUTH_DATASET_FROZEN_BEFORE_MODEL_TRAINING`
- Representation: `FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE` v1.0, 20 canonical slots
- Splits (stored per surface, unchangeable): train 7,500 / validation 1,250 / test 1,250
- Split integrity: no surface-id overlap, no parameter-vector-hash overlap;
  the test split is never used for training or hyperparameter choice; all
  normalization statistics are fit on train rows only.
- Noise: the clean dataset only. No noise/boundary/OOD cohorts exist in this
  milestone and none may be generated for it.

## 3. Input representation (both neural methods, one shared builder)

100 features in canonical slot order, float32, raw (no input normalization):

| Block | Size | Definition |
|---|---|---|
| `prices_masked` | 20 | spot-normalized price per canonical slot; masked slots exactly `0.0` |
| `mask` | 20 | 1.0/0.0 validity flag per canonical slot |
| `maturities_years` | 20 | actual time to maturity per slot (dte/365) |
| `rates` | 20 | per-slot risk-free rate (rank-constant in this dataset) |
| `carries` | 20 | per-slot carry, consumed in the pricer's dividend-yield slot (generation identity `carry = rate + carry_offset`) |

Canonical R2 only: the legacy 108 grid and the rejected R3 (30) geometry are
structurally rejected. Targets are the canonical ten parameters
(`kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow, kappa_fast,
theta_fast, sigma_fast, rho_fast, v0_fast`) from
`metadata.parameters_canonical_order`, standardized by train-split-fitted
per-parameter mean/std.

## 4. Method 0 — traditional numerical calibration (frozen)

The existing canonical module `src/calibrate_double_heston.py`, unchanged:

- production pricer, `node_count=64`;
- objective: `least_squares` residuals `(predicted − observed) / max(observed, 1.0)` on **dollar** prices (normalized × spot);
- optimizer: SciPy `least_squares(method="trf")`, tolerances 1e-10, `diff_step=2e-5`, `max_nfev=300`;
- constraint-satisfying reparameterization with the hard safety bounds of `configs/parameter_bounds_PROVISIONAL.yaml`;
- 3 deterministic starts (seed 42): neutral transform midpoint; broad `N(0, 1.25²)`; **disclosed** truth-informed perturbation `truth coordinate + N(0, 0.35²)` (favors the traditional baseline; retained from the canonical module; every start is retained and reported);
- representative per surface: lowest final objective among all starts (ties → lowest `start_index`); failed starts retained, never silently rerun;
- executed on the test split; surface-level parallel workers change wall time only, never the per-surface budget.

Pre-protocol timing (train-split surfaces only): 124–170 s per surface at
these settings; budget frozen regardless.

## 5. Method 1 — ordinary ANN inverse model (frozen)

`models.ann_model.ANNInverseCalibrator`, pre-existing implemented defaults:

- MLP `(512, 256, 128, 64)`, ReLU, dropout 0.10, no layer normalization;
- output: unconstrained 10-vector in standardized-target space
  (structural constraints **not** enforced; validity is measured);
- Adam, lr 1e-3, weight decay 1e-5, batch 256, ≤ 200 epochs;
- early stopping: patience 20 on validation standardized MSE; best-validation
  checkpointing; loss = MSE on standardized targets.

## 6. Method 2 — constraint + differentiable-repricing-informed (frozen)

`models.pinn_model.PhysicsInformedInverseCalibrator`, pre-existing defaults:

- backbone `(512, 512, 256, 256, 128)`, GELU, dropout 0.05, LayerNorm blocks
  (residual where widths match);
- `DoubleHestonConstraintMap` head: strict positivity (softplus + 1e-6),
  `kappa_fast = kappa_slow + softplus` (strict ordering), per-factor
  `sigma ≤ sqrt(2·kappa·theta)·0.995`, joint correlation disk radius ≤ 0.995;
- loss: `1.0 · MSE(standardized params) + 1.0 · masked MSE(normalized repriced
  prices, observed normalized prices)`;
- repricing term: the differentiable Torch mirror of the production pricer at
  `node_count=64`, computed in **float64** (see audit finding D: float32 is
  numerically invalid; the training path upcasts and gradients flow through
  the cast);
- Adam, lr 5e-4, weight decay 1e-5, batch 64, ≤ 200 epochs; early stopping
  patience 20 on validation total loss; best-validation checkpointing.

Information fairness: Method 2 sees exactly the same R2 observation as
Method 1 (the same 20 normalized prices and conditioning; no extra market
information).

Implementation note (feasibility, pre-declared): the differentiable repricing
term is evaluated through a batch-vectorized implementation of the identical
formulation; equivalence to the existing loop implementation and to the
production pricer is pinned by tests before any research run.

## 7. Seeds and hyperparameter policy

- Neural research seeds: **11, 22, 33** (first three of the pre-existing
  `evaluation.repeated_seeds` in `configs/ann_baseline.yaml`).
- Every seed is trained and reported; none is discarded. Cross-seed mean and
  standard deviation are reported for every headline metric.
- Traditional calibration uses its fixed module start seed 42; variability is
  measured by multi-start dispersion, not seed resampling.
- One frozen configuration per method — the pre-existing implemented
  defaults recorded above. No architecture or loss-weight search is performed
  on any split of this dataset. Validation data is used only for early
  stopping/checkpointing. Smoke runs (section 12) are pipeline validation
  only; no scientific choice is tuned from them.

## 8. Frozen metric families

**Parameter recovery** (synthetic truth): per-parameter MAE / median AE / RMSE;
per-parameter range-scaled error (train-range scaling); aggregate
standardized-unit RMSE; factorwise aggregates `v0_total`, `theta_total`,
per-factor half-life `ln(2)/kappa`; factor-swap confusion rate. Never add
kappa/sigma/rho across factors.

**Constraint validity**: fraction of predictions satisfying
`validate_parameters`; per-constraint violation rates (positivity, ordering,
per-factor Feller, correlation disk).

**Repricing** (production pricer, node 64, on predicted representatives):
normalized-price RMSE, MAE, max absolute error. IV error: not reported
(numerical validity not established).

**Identifiability-aware**: repricing-tolerance success rates at normalized
RMSE ≤ 1e-4 and ≤ 1e-3 cross-reported with parameter-tolerance recovery at
range-scaled parameter RMSE ≤ 0.10 and ≤ 0.25; explicit reporting of
near-equivalent repricing with materially different parameters. Repricing
success is never equated with truth recovery.

**Stability**: cross-seed standard deviation of every headline metric; mean
per-surface cross-seed prediction dispersion; traditional per-surface
multi-start parameter dispersion and disagreement rates; validity rates per
seed/start.

**Runtime**: neural per-surface amortized inference (full test split, one
batch, eval mode, CPU); traditional per-surface wall seconds including all
starts; training wall time reported separately; hardware documented.

## 9. Fairness rules

All three methods are evaluated on the same frozen 1,250-surface test split.
Models 1 and 2 share split, input representation, parameter targets, primary
metrics, and evaluation pricer. Method 2 receives no observational
information unavailable to Method 1. Traditional calibration receives the same
R2 surfaces. No method is ranked on repricing RMSE alone.

## 10. Pre-training checkpoint (gates training)

Focused contract tests (`tests/test_r2_primary_comparison_protocol.py`) must
pass and the protocol commit must be pushed and remote-verified before any
research training. They verify: exact dataset hash; 10,000 count; exact
7,500/1,250/1,250 split counts; zero cross-split overlap; R2-only
representation; canonical 20-slot identity; no 108; canonical parameter
order; no real data (synthetic-only records); quarantine active; protocol
config/doc consistency; seeds and metrics frozen in the config; checkpoint
output directories absent.

## 11. Implementation repair (after freeze, before training)

Add: an R2-native dataset/feature path (one shared builder); the float64
batch-vectorized differentiable repricing term with equivalence tests; R2
training entrypoints for both models (deterministic seeding, provenance-rich
checkpoints); the R2-native evaluation module (production pricer); focused
tests for R2 input dimensions, no-legacy-108 imports, parameter order,
constraint transforms, repricing-gradient existence, split isolation, and
checkpoint metadata. The frozen scientific design is never altered by repair.

## 12. Smoke runs

Tiny pipeline-validation runs marked
`DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT` (loss decreases; gradients finite;
checkpoint save/load works; evaluation executes; no leakage). Never used for
scientific tuning.

## 13. Execution order and evidence

1. Train Model 1 and Model 2 for seeds 11/22/33 (no test metrics until all
   runs complete and models are frozen at best-validation checkpoints).
2. Run frozen traditional calibration on the test split.
3. Evaluate all methods on the untouched test split; produce the unified
   comparison table and per-family reports.
4. Evidence lives in `evidence/r2_primary_comparison_20260823/` (protocol.json,
   dataset_identity.json, training_run_manifest.json, per-method result CSVs,
   metric JSONs, runtime metrics). Checkpoints stay untracked; commit
   histories, summaries, metrics, and hashes.

## 14. Claim discipline

Allowed (if supported): "Method 2 improves structural validity / stability /
repricing under the tested synthetic protocol"; "Method 2 improves parameter
recovery on some metrics." Forbidden: any claim of unique parameter
recovery; equating low repricing error with correct parameters; "PINN solves
identifiability." Practical non-identifiability remains part of the thesis.

## 15. Explicit boundaries

```text
FINAL_10K = UNCHANGED
REAL_MARKET_WEIGHT_UPDATES = NONE
G8 = NOT_STARTED
R2 = UNCHANGED
PRODUCTION_PRICER = UNCHANGED
CANONICAL_PARAMETER_ORDER = UNCHANGED
PRACTICAL_NON_IDENTIFIABILITY = RETAINED
OPTIONAL_PDE_MODEL_3 = NOT_STARTED
NOISE_COHORTS = NOT_GENERATED
```
