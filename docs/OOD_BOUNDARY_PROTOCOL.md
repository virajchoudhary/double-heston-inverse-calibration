# Double Heston OOD and Boundary Robustness Protocol (FROZEN)

Status: **FROZEN_BEFORE_MODEL3_RESEARCH_RESULTS**, 25 August 2026.
Config twin: [../configs/ood_boundary_protocol.yaml](../configs/ood_boundary_protocol.yaml).
Pre-execution audit: [OOD_BOUNDARY_PREEXECUTION_AUDIT.md](OOD_BOUNDARY_PREEXECUTION_AUDIT.md).
Branch/base: `research/ood-boundary-protocol` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`.

This protocol evaluates robustness of inverse-calibration representatives under
predeclared parameter-boundary, admissible distribution-shift, conditioning,
and incomplete-observation challenges. It does not change R2, the production
pricer, canonical parameters/constraints, completed primary evidence, or the
retained finding of practical non-identifiability.

## 1. Non-negotiable leakage controls

```text
MODEL_3_RESULTS_USED_IN_DESIGN = false
REAL_MARKET_WEIGHT_UPDATES = NONE
FROZEN_PRIMARY_TEST_METRICS_USED_TO_TUNE_DESIGN = false
COMPLETED_PRIMARY_EXPERIMENT_FILES = READ_ONLY
NOISE_OR_THRESHOLD_REDESIGN_AFTER_RESULTS = FORBIDDEN
```

Primary test artifacts are hashed in the config and used only after evaluation
to form aggregate OOD/ID degradation ratios. They cannot choose cohort support,
sample size, materiality, methods, or interpretation. Real-market data may never
update weights and does not enter synthetic generation.

## 2. Shared scientific contract

All records use canonical R2 v1.0: 20 slots, exact frozen slot order, masks,
actual maturities, rates/carries, spot 100, and 100-feature construction when
consumed by neural models. Parameters remain the canonical ten-vector. Every
truth passes strict positivity, slow/fast ordering, Feller, individual
correlation, and joint-disk checks. Prices come only from unchanged
`price_double_heston_surface` at 64 nodes. Failures retain identity/error and
fail closed; there is no clamping, refill, replacement, imputation, or silent
retry.

## 3. Development sanity cohort

A 12-row development panel contains the first four eligible selected parameter
vectors from each active parameter cohort. It is labeled
`DEVELOPMENT_SANITY_NOT_RESEARCH_RESULT`, has no priced surface, and tests
shape, canonical validity, hashes, serialization plumbing, and manifests only.
It is never pooled into research metrics.

## 4. Frozen research cohorts

| Cohort | Records | Purpose |
|---|---:|---|
| `boundary_challenge` | 120 | Valid but constraint-proximate truths |
| `distribution_shift` | 120 | Admissible regimes displaced from central support |
| `maturity_conditioning_shift` | 120 | Same model, shifted long-maturity/rate/carry design |
| `incomplete_observation` | 60 | Deterministic frozen-mask derivatives |

There are 420 serialized research records and 360 clean pricing calls. All are
evaluation-only and ineligible for train/validation.

### 4.1 Boundary challenge

Use the existing reviewed four-regime latent-LHS challenge algorithm with pool
160, quota 120, seed 20260825, and 30 accepted rows per regime:
near Feller, weak slow/fast separation, near hard bound, near correlation disk.
Canonical validity and regime labels are checked. Proximity is intentional but
every vector remains inside the hard bounds and strict canonical admissible set.
Boundary and distribution-shift cohorts use the typical R2 training-conditioning
lattice with a 21-day minimum rank-1 maturity. The floor excludes—not clamps or
replaces—short-expiry cases where a valid near-zero-v0 boundary truth can produce
a tiny negative deep-ITM pricer value. It isolates parameter effects from the
separate maturity-shift cohort while preserving strict positive-price validity.

### 4.2 Distribution shift

Generate a fixed latent-LHS pool of 300 with seed 20260826 under wide-valid
transforms and its exclude-any-boundary-near gate. Select the first 60 eligible
rows in each named regime:

- **slow low mean reversion/high variance:** `kappa_slow=[0.20,0.29]`,
  `theta_slow=[0.205,0.2325]`, `v0_slow=[0.255,0.2775]`, gap `[1.25,2.50]`;
- **fast high mean reversion:** `kappa_fast=[10.25,11.35]` via conditional
  transform.

These supports avoid the observed R2 maxima in the named dimensions while
remaining more than the reviewed 5% hard-distance margin from singular bounds.
Sigma follows Feller-safe conditional transforms; correlations follow polar
joint-disk transforms.

### 4.3 Maturity/conditioning shift

Sample 120 accepted wide-valid-margin parameter vectors from a fixed pool of
300 with seed 20260827. Use deterministic mixed-radix stride 179 over rank-1
DTE `[105,120,135,150,165,180]`, rank-2 gaps `[105,120,135,150,165,180]`,
rates `[0.065,0.075,0.085]`, and carry offsets
`[-0.055,-0.040,-0.025]`. Rank-2 DTE always exceeds rank-1 and carry equals
rate plus offset. This stays within the canonical pricing model while being
disjoint from the typical training lattice.

### 4.4 Incomplete observation

Take 60 clean parents by round robin across boundary, distribution-shift, and
maturity cohorts. Apply patterns cyclically: rank-1 only, rank-2 only, central
three moneyness, calls only, even-slot checkerboard. Each derivative has at
least 10 usable slots. Masked slots contain exactly zero and retain known
maturity/rate/carry. The parent is retained separately; it is not supplied as
input for the masked run, but is allowed later as a clean counterfactual
repricing target.

## 5. Determinism and provenance

Cohort generation creates the entire declared pools before selection, retains
candidates/rejections, removes duplicate vectors and any hash collision with
frozen R2, selects ascending eligible candidate IDs within regime, and assigns
conditioning by predeclared stride. Surface payloads store cohort/regime/pattern,
candidate/generation indices, sampler/conditioning seeds, parameter-vector hash,
config/source hashes, real-market flag, and evaluation-only status. Output
includes CSV panels, clean/incomplete/all JSONL, numerical sanity, integrity
report, and a manifest. A second independent generation must reproduce all
scientific artifact bytes bit-for-bit; timestamps/environment metadata are
provenance only and never RNG inputs.

Generation is forbidden until this protocol checkpoint is committed, pushed,
and remote-verified. The CLI additionally requires
`--remote-checkpoint-confirmed`.

## 6. Metrics and comparisons

Every aligned method prediction is evaluated for:

- parameter MAE/median AE/RMSE/bias, range-scaled error using frozen train
  ranges, standardized RMSE using frozen train statistics, total-v0/total-theta
  errors, half-life errors, and factor-swap confusion;
- finite/shape/execution success, strict canonical validity, and each violation
  class;
- normalized-price RMSE, MAE, and maximum absolute error against clean parent
  prices at 64 nodes. For masked inputs this is an explicitly labeled
  counterfactual: the parent supplies the output target only, never input data;
- cross-seed neural dispersion where multiple seeds exist; multi-start
  dispersion/failure for traditional calibration.

Degradation ratio is `max(OOD metric,floor)/max(ID baseline metric,floor)`,
with floor `1e-8`. Relative degradation is material at ratio `>1.25`; validity
failure increase is material at absolute `>0.05`. Missing-vs-clean comparison
is paired by parent/method. OOD-vs-ID comparison is aggregate/post hoc and
never alters the protocol. Uncertainty uses 2,000 deterministic bootstrap
resamples (seed 20260829), 95% intervals, and reports whether a decision
interval crosses materiality.

Model 1, Model 2, and any ready future Model 3 adapter are evaluated on all
420 records. Traditional calibration has a frozen minimum feasible subset of 15
surfaces per active cohort (60 total), evenly spaced from zero; more may be run
without changing thresholds, but no subset is chosen from outcomes. Future
Model 3 must return an aligned finite `(N,10)` matrix and otherwise receives
`INCONCLUSIVE_INTERFACE`, not a scientific PASS.

## 7. Interpretation

- **ROBUST_PASS:** generation/replay verified and no headline family has a
  material degradation in any active cohort.
- **OOD_SENSITIVE_NEGATIVE:** any material recovery, repricing, or validity
  failure degradation under the predeclared rules.
- **INCONCLUSIVE:** replay fails, more than 1% required predictions are
  missing/nonfinite/misaligned, a bootstrap interval spans materiality, Model 3
  lacks a valid aligned adapter, or traditional execution failures exceed 5%.

A method may be described as more robust only with aligned paired predictions,
uncertainty, both recovery and repricing context, and non-worse validity.
Repricing quality alone is never superiority or correct-parameter evidence.
Negative and inconclusive outcomes are reportable results; they do not authorize
threshold redesign, representation redesign, selective seed removal, or another
unregistered challenge until a separately frozen protocol is approved.

## 8. Execution gates

Allowed now: focused contract/sampler/mask tests and read-only review.
Forbidden now: cohort generation before push verification, expensive method
evaluation, neural research training/fine-tuning, real-market ingestion, and
long multiprocessing.
Next scientific step after push verification:

```bash
python -m src.ood_boundary_protocol validate-contract
python -m src.ood_boundary_protocol generate --output evidence/ood_boundary_protocol_v1 --remote-checkpoint-confirmed
python -m src.ood_boundary_protocol replay --output evidence/ood_boundary_protocol_v1 --replay-output evidence/ood_boundary_protocol_v1_replay
```

Only after identical replay may the separately authorized lightweight method-
prediction harness run. Expensive traditional execution remains a distinct
bounded follow-up.
