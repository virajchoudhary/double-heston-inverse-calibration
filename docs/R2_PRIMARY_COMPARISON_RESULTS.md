# R2 Primary Comparison — Results Reconciliation

Status: COMPLETE (final frozen synthetic-test evaluation executed once).
Protocol: `docs/R2_PRIMARY_COMPARISON_PROTOCOL.md` (frozen before training at commit
`5ea9fd0`, remote-verified; Issue #32).
Dataset: `data/final_r2_clean_10000/` — surfaces SHA-256 `148b579a…f1f6`,
splits 7,500 / 1,250 / 1,250 stored per surface.
Evidence bundle: `evidence/r2_primary_comparison_20260823/`.
Every number below comes only from the committed evidence bundle.

## 1. Execution chronology

| Step | Artifact |
|---|---|
| Protocol freeze (pre-training) | commit `5ea9fd0` (pushed, remote-verified); Issue #32 |
| Implementation repair | commits `ab11b09`, `9bf1d49` |
| Smoke runs (`DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT`) | `checkpoints/r2_primary_comparison/smoke/`, `smoke_calibration_starts.csv` |
| Research training Model 1 seeds 11/22/33 (local CPU) | `checkpoints/r2_primary_comparison/model1_seed*/` |
| P100 benchmark gate PASSED (before any research training) | `cloud_benchmark_1d9bdc14e51a.json` |
| CLI/device plumbing repair (execution-only) | commits `f0d2878`, `2b5d41c`; argparse-failing early cloud attempts produced no run |
| Research training Model 2 seeds 11/22/33 (uniform Tesla P100 cohort) | `model2_seed*/`, git_sha `2b5d41c` recorded in every summary/checkpoint |
| Local CPU Model-2 seed-11 replication retained (never primary) | `model2_seed11_local_cpu_replication/` (`MODEL2_LOCAL_SEED11_EXECUTION_ENVIRONMENT_REPLICATION`) |
| Traditional calibration on test split (frozen settings) | journal 465 → 553 → 1250; final SHA-256 `fe73e696…0ce51` |
| Pre-unsealing evaluator repair (YAML parsing; no metric existed) | commit `285a724` |
| Final untouched synthetic-test evaluation (run ONCE) | `synthetic_test_comparison.csv` + metric JSONs |

Execution-order note: the first 465 traditional-calibration surfaces were computed
while Model-2 seeds 22/33 were still untrained — classified
`EXECUTION_ORDER_DEVIATION_NO_METRIC_LEAKAGE` (settings pre-frozen; calibration
independent of neural training; no tuning or selection used any calibration or
test metric).

## 2. Training manifest (from training_run_manifest.json)

Model 1 (local CPU, git_sha per checkpoint manifest): seed 11 best epoch 124,
seed 22 best epoch 90, seed 33 best epoch 43; validation-only selection,
`test_set_used_for_selection=false` everywhere.
Model 2 (uniform P100 cohort, git_sha `2b5d41c` all three): seed 11 best epoch
46 (66 epochs), seed 22 best epoch 60 (80), seed 33 best epoch 62 (82);
validation-only selection, zero test-selection. Validation total losses
(0.8155 / 0.8155 / 0.8090) are training-selection evidence only, not research
performance. Full hashes: `P100_MODEL2_COHORT_MANIFEST.json`.

## 3. Unified comparison (`synthetic_test_comparison.csv`, seed-mean rows)

| Metric | Model 1 ordinary ANN | Model 2 constraint+repricing-informed | Traditional calibration |
|---|---|---|---|
| Range-scaled parameter RMSE ↓ | 0.1659 | 0.1664 | **0.1151** |
| Standardized parameter RMSE ↓ | 0.9013 | 0.9040 | **0.6002** |
| v0_total MAE ↓ | 0.01006 | 0.01037 | **0.00056** |
| theta_total MAE ↓ | 0.03302 | 0.03302 | **0.01456** |
| Half-life slow MAE (years) ↓ | 0.3462 | 0.3463 | **0.2750** |
| Half-life fast MAE (years) ↓ | 0.05202 | **0.05195** | 0.02893 |
| Factor-swap confusion rate ↓ | 0.0000 | 0.0000 | 0.0112 |
| Constraint validity rate ↑ | 1.000 | 1.000 | 1.000 |
| Repricing nRMSE mean ↓ | 7.27e-04 | 7.89e-04 | **5.70e-09** |
| Repricing nRMSE p95 ↓ | 1.97e-03 | 2.11e-03 | **2.37e-08** |
| Repricing ≤1e-4 success ↑ | 2.00% | 0.96% | **100%** |
| Repricing ≤1e-3 success ↑ | 77.84% | 78.08% | **100%** |
| Parameter recovery ≤0.25 ↑ | 91.20% | 91.04% | **94.00%** |

Per-seed headline metrics (`neural_seed_results.csv`): Model 1 range-scaled
RMSE 0.16599 / 0.16593 / 0.16605; Model 2 0.16719 / 0.16731 / 0.16624 — both
methods extremely seed-stable (below).

## 4. Findings by frozen metric family

### 4.1 Parameter recovery

Traditional numerical calibration achieves the best point recovery on every
aggregate (range-scaled RMSE 0.115 vs ≈0.166 for both networks) and on every
per-parameter range-scaled MAE except none — largest gaps on rho_slow/rho_fast
(0.034–0.037 vs ≈0.16–0.17) and sigmas (≈0.017–0.019 vs ≈0.09–0.10). Both
neural methods are statistically indistinguishable from each other on aggregate
parameter recovery (Δ ≈ 0.0005, cross-seed std of the metric ≈ 0.0006 for
Model 2). v0_total and theta_total (the physically additive combinations) are
recovered far better than individual parameters by all methods, consistent
with compensated-direction ambiguity. No method recovers kappa/half-life well
(half-life slow MAE ≈ 0.28–0.35 years even for traditional).

### 4.2 Constraint validity

All three methods emit structurally valid Double Heston parameters on 100% of
the 1,250 test surfaces. Model 2's constraint map guarantees this by
construction; Model 1 happens to satisfy the box/ordering/Feller/disk checks
on every test surface despite having no enforced constraints (its errors stay
inside valid regions); traditional calibration enforces bounds through its
constrained transform.

### 4.3 Repricing

Traditional calibration reprices essentially perfectly (mean normalized-price
RMSE 5.7e-09; p95 2.4e-08) — expected, since its objective IS repricing under
a 300-evaluation budget with truth-informed disclosed starts. The two neural
models reach mean nRMSE ≈ 7.3–7.9e-04 with 78% of surfaces ≤1e-3 but only
~1–2% ≤1e-4. Model 1 is marginally better than Model 2 here (7.27e-04 vs
7.89e-04 mean), a difference without practical significance given cross-seed
dispersion (~1.1e-04 std).

### 4.4 Identifiability-aware interpretation

THE CENTRAL RESULT. Conditioned on essentially perfect repricing, parameter
recovery still fails materially:
- Traditional: 100% of surfaces reprice ≤1e-4, yet only 71.9% achieve
  range-scaled parameter RMSE ≤0.10 (94.0% ≤0.25). Thus ≥6% of surfaces admit
  parameters that reprice to ~9 significant digits while being >25% wrong in
  range-scaled terms.
- Neural: 78% of surfaces reprice ≤1e-3, but only 7.6% of those also achieve
  parameter RMSE ≤0.10. Good repricing almost never certifies parameter
  accuracy for the networks' operating regime.
This is exactly the retained practical-non-identifiability finding: compensated
parameter directions and call/put parity redundancy allow near-equivalent
repricing from materially different parameter vectors.

### 4.5 Stability

Neural cross-seed stability is excellent for both models (headline-metric
cross-seed std ≤3e-3 standardized units). Per-surface prediction dispersion
across seeds: Model 1 mean 0.0069 (standardized units), Model 2 0.0174 —
adding the differentiable repricing term did NOT reduce seed-to-seed
prediction dispersion. Traditional multi-start dispersion is large: mean
range-scaled cross-start std per parameter up to 0.32 (kappa_fast), with an
89.2% per-surface start-disagreement rate (>0.5 range-scaled units on some
parameter between some pair of the 3 starts) — direct multi-start evidence of
an ill-conditioned inverse problem. Of 3,750 starts, 558 flagged optimizer
success and 3,192 hit the frozen max_nfev=300 budget; the frozen
representative rule (lowest final objective; ties by lowest start_index)
retains every start's record regardless of flag.

### 4.6 Runtime

Common evaluation environment (local CPU inference, batched): Model 1 ≈
0.0034–0.0041 ms/surface amortized; Model 2 ≈ 0.0080–0.0083 ms/surface
(both full-split single forward passes; see `neural_seed_results.csv`).
Traditional calibration consumed a mean 376 s per surface (p95 614 s; 3
starts × max_nfev 300) — i.e., neural inference is ~5–7 orders of magnitude
cheaper per surface than the numerical baseline that defines the accuracy
frontier. Training wall-times are environment-specific provenance only
(Model 2 P100 sessions ≈ 463–591 s/seed; local CPU Model 1 ≈ 264–529 s/seed;
local CPU Model-2 replication 7,314 s) and must not be compared as quality.

## 5. Claim-discipline reconciliation

Allowed and supported: on this synthetic protocol, traditional calibration
dominates both networks on parameter recovery AND repricing at a
computationally prohibitive per-surface cost; both networks produce 100%
structurally valid parameters at millisecond inference with moderate
parameter errors (range-scaled RMSE ≈ 0.17) and good-but-not-exact repricing;
the constraint+repricing-informed model shows no measurable advantage over
the ordinary ANN on this dataset's aggregates, and does not reduce cross-seed
dispersion. Forbidden claims NOT made: no unique parameter recovery is claimed
for any method; low repricing error is not interpreted as correct parameters
(the data explicitly refute such an interpretation); no claim that the PINN-
style model "solves" identifiability. Practical non-identifiability is
RETAINED as a research finding and is directly visible in the conditioned
metrics (§4.4). No result-based tuning, seed cherry-picking, or protocol
change occurred after the freeze; the first synthetic-test read was also the
final one.

## 6. Limitations

Single synthetic generator (frozen R2 representation, complete surfaces,
zero observation noise); results do not establish real-market performance
(G8 out of scope). Traditional calibration received a disclosed truth-informed
start favoring it, per the frozen design. Model 2's repricing weight was not
tuned (single frozen configuration). Neural training-environment differences
(CPU vs P100) are provenance facts, not scientific variables. Inference
runtimes are hardware-specific. The 6% of traditional surfaces with poor
recovery despite near-perfect repricing lower-bounds (does not exhaust) the
practical non-identifiability problem.
