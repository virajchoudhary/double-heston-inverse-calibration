# Execution Continuation Decision — R2 Primary Comparison

Status: DECISION_FROZEN_BEFORE_ANY_FURTHER_RESEARCH_COMPUTE
Written: 2026-08-23
Recovery HEAD: `d7863ed33df4f9f2b58aa07f39093a0b7778f6e7`
Frozen protocol commit: `5ea9fd00d9703ffe922de861c4434bbfe9f1416e` (protocol config +
`docs/R2_PRIMARY_COMPARISON_PROTOCOL.md`; Issue #32; remote-verified)
This note records execution-continuation policy ONLY. No frozen scientific
setting is changed by this document.

## 1. Recovered run state (verified during recovery)

| Item | State |
|---|---|
| Model 1 seeds 11/22/33 | COMPLETE (best-validation checkpoints + summaries under `checkpoints/r2_primary_comparison/model1_seed*/`) |
| Model 2 seed 11 (local CPU) | COMPLETE (`model2_seed11/`, best_epoch 38, git_sha `3486ca2`, torch_threads=2) |
| Model 2 seeds 22/33 | NOT STARTED (`model2_seed22/` directory created empty before interruption) |
| Traditional calibration journal | **465 / 1250** test surfaces persisted (crash-safe JSONL, 3 starts each) |
| Synthetic test metrics computed | NONE (no test metrics read at any point; `final_evaluation` not run) |
| Superseded attempts | Archived as SUPERSEDED_IMPLEMENTATION_RUN_NOT_RESEARCH_RESULT in `evidence/attic_superseded_pilot_runs/` |

## 2. Calibration chronology deviation — CLASSIFICATION

**Classification: EXECUTION_ORDER_DEVIATION_NO_METRIC_LEAKAGE**

The frozen protocol's intended evaluation order places traditional test-split
calibration after all neural research models are frozen. The 465 journaled
surfaces were computed while Model-2 seeds 22/33 had not yet been trained.
This is an execution-order deviation, not a protocol change, and it does NOT
invalidate the 465 rows because:

1. All calibration settings were already frozen at `5ea9fd0` BEFORE any
   calibration execution (max_nfev=300, ftol/xtol/gtol=1e-10, diff_step
   2e-05, node_count 64, 3 deterministic starts, start_seed 42, frozen bounds).
2. Traditional calibration is mathematically independent of neural training:
   it reads only the dataset surfaces and its own fixed settings.
3. No calibration metric and no synthetic test metric was used for neural
   hyperparameter tuning, architecture changes, seed selection, loss-weight
   changes, early-stopping decisions, or protocol changes.
4. The 465 rows were written incrementally by the same deterministic fixed
   settings; per-surface results are independent of when they were computed.
5. No rows were selected, discarded, or rerun based on outcome.

Disposition of existing rows: ALL 465 journaled surfaces are PRESERVED.
No row may be deleted, recomputed, or filtered by outcome.

## 3. Test-metric leakage statement

No synthetic-test metrics have been computed, inspected, or recorded for any
method as of this decision. Calibration journal rows contain per-start fit
statistics only; they have not been aggregated, summarized, or compared
against neural models. Nobody may inspect calibration performance metrics
until AFTER the canonical Model-2 research cohort is fully frozen.

## 4. Cloud execution cohort rule (FROZEN)

Model-2 primary seeds must NOT be mixed across unrelated hardware
environments. If cloud GPU migration is selected, the canonical cloud Model-2
cohort is exactly:

    seed 11, seed 22, seed 33

all run on the SAME cloud accelerator class and the same software environment,
using the identical frozen architecture/loss/optimizer/batch sizes/seeds.

### Treatment of the completed local CPU Model-2 seed 11

The already-completed local CPU seed-11 run is retained permanently and
reported as:

    MODEL2_LOCAL_SEED11_EXECUTION_ENVIRONMENT_REPLICATION

It must NOT be deleted or hidden. It must NOT be chosen over the cloud
seed-11 result based on validation/test performance. It is retained purely as
execution-environment replication evidence of determinism/provenance.

If the predeclared uniform cloud cohort 11/22/33 passes the benchmark gate,
the primary 3-seed aggregate uses that uniform cloud cohort. If cloud
migration does NOT pass the benchmark / numerical-equivalence gate, seeds
22/33 continue locally instead and the existing local seed 11 remains part of
the primary CPU cohort (uniform local cohort 11/22/33).

The gate decision is based on EXECUTION FEASIBILITY AND NUMERICAL EQUIVALENCE
ONLY — never on model quality, validation loss, or any test metric.

## 5. Cloud benchmark gate (uses committed `scripts/cloud/`)

Preferred free target: Kaggle P100 IF actually available in the session.
Do not assume T4/L4 is faster: this workload is float64/complex128 bound, so
FP64 throughput dominates (P100 ~4.7 TFLOPS FP64 vs T4 ~0.1).

Gate requirements (all must hold, benchmark uses TRAIN/DEVELOPMENT data only):

1. Exact branch SHA / protocol-config hash / dataset hash verified by
   `scripts/cloud/verify_environment.py` (fails closed on drift).
2. float64 + complex128 supported on the accelerator.
3. Vectorized Torch pricer numerically matches the local/production
   implementation under the existing frozen tolerances.
4. Gradients finite through the full repricing path.
5. Same model architecture / loss / optimizer / batch size / seeds.
6. Measured meaningful speed advantage (README gate: >= 2x vs measured local
   clean baseline 0.26 s/step at 2 CPU threads).

Benchmark runs are DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT and must not read
synthetic test metrics.

## 6. Runtime fairness rule

Training runtime is environment-specific and must always be reported with its
hardware provenance. Local CPU training wall-time must never be compared
against cloud GPU training wall-time as if it were a model-quality result.
The final inference-runtime comparison between methods follows the frozen
protocol's common evaluation environment (CPU inference on the test split,
per protocol section `metrics.runtime`).

## 7. Traditional-calibration resume rule

- The 465-row journal stays UNTOUCHED until Model-2 seeds 11/22/33 are all
  complete and frozen.
- Resume FROM 465, never from zero: the runner skips already-complete surface
  ids (`src/r2_primary/calibration.py` journal-resume logic). No already-
  journaled successful surface/start rows are rerun; failed starts are
  retained and reported, never silently dropped or rerun with new settings.
- Resume only AFTER the canonical Model-2 cohort is frozen; do not inspect
  calibration performance metrics before then.
- Resume settings are exactly the frozen ones: same 1250 test surface IDs,
  same 3 deterministic starts (module seed 42), max_nfev = 300,
  ftol/xtol/gtol = 1e-10, diff_step 2e-05, same objective/bounds/node_count.
- If migrating off the laptop: prefer a >= 16-vCPU cloud VM. Kaggle/Colab GPU
  is not automatically useful for scipy least_squares (Kaggle offers 4 cores).

## 8. Out-of-scope reminders (unchanged)

G8 / real-market / noise-cohort / boundary / OOD evaluations remain out of
scope. The frozen protocol remains unchanged. Nothing in this note authorizes
any protocol modification.

## 9. Addendum — cloud benchmark outcome and GPU plumbing repair (factual log)

Date: 2026-08-23. Facts only; no scientific change.

- Kaggle P100 benchmark PASSED before any cloud research training:
  CUDA Model-2 step 0.0479512682 s vs local frozen baseline 0.26 s
  (~5.4x speedup); float64 gradients finite; vectorized-vs-loop max diff
  5.68e-14 and vectorized-vs-production max diff 7.11e-14, both inside the
  1e-9 frozen tolerance; environment verification and 14/14 implementation
  tests passed on the P100 (Torch 2.10.0+cu126, Tesla P100-PCIE-16GB).
- A CLI/device-plumbing defect prevented any cloud research seed from
  starting: the training module accepted ``device=`` only as a Python
  keyword, argparse exposed no ``--device`` flag, and the differentiable
  repricing term built its input tensors on CPU regardless of model device.
- The attempted Model-2 seed 11/22 cloud commands exited at argparse
  argument parsing. NO research checkpoint, training history, summary, or any
  other research output was created by those attempts.
- Repair commit wires an explicit ``--device {cpu,cuda}`` flag (default cpu,
  never auto-selecting CUDA) through both research and smoke paths, makes
  ``_repricing_loss`` construct every pricer input tensor on the prediction
  device at the frozen float64 dtype, and updates ``scripts/cloud/
  README_CLOUD_RUN.md`` to document the real CLI. Placement-only change:
  architecture, optimizer, losses/weights, batch sizes, seeds, node_count,
  early stopping, dataset/splits, metrics, and calibration settings are
  untouched; CPU numerics are proven bitwise unchanged by test.
- Traditional calibration journal remains untouched at 465/1250; no local or
  cloud training has been started by this repair.

## 10. Addendum — uniform P100 Model-2 cohort imported (factual log)

Date: 2026-08-24. Facts only; no scientific change.

- Intake: four ZIPs + SHA manifest in `incoming_p100_artifacts/` (local,
  untracked). All four computed SHA-256 values match `P100_ARTIFACT_SHA256.txt`
  exactly. Source ZIPs remain untouched.
- Local CPU seed 11 was verified (`device_used=cpu`, `git_sha 3486ca2`,
  seed 11, run_kind RESEARCH) and moved BEFORE cloud import to
  `checkpoints/r2_primary_comparison/model2_seed11_local_cpu_replication/`.
  Status: MODEL2_LOCAL_SEED11 = RETAINED_EXECUTION_ENVIRONMENT_REPLICATION.
- The uniform P100 cohort (seeds 11/22/33; same accelerator class, same
  software environment, same git_sha `2b5d41c`, RESEARCH, cuda, validation-
  only selection, zero test-selection) was staged, fully provenance-checked,
  checkpoint-load-tested, and imported to
  `checkpoints/r2_primary_comparison/model2_seed{11,22,33}/`.
  MODEL2_P100_SEED11 = COMPLETE; MODEL2_P100_SEED22 = COMPLETE;
  MODEL2_P100_SEED33 = COMPLETE; MODEL2_PRIMARY_COHORT =
  COMPLETE_UNIFORM_P100. Per-file SHA-256 hashes are recorded in
  `P100_MODEL2_COHORT_MANIFEST.json` (checkpoints stay untracked per repo
  `.gitignore` policy; no force-add).
- Cloud evidence import: the archived journal copy is byte-different from the
  local canonical journal ONLY by line endings (local CRLF vs cloud LF);
  normalized SHA-256 identical (`6f95bf44…a993`), 465 lines / 465 distinct
  surface_ids both, all row values identical. The LOCAL journal remains
  canonical and was NOT overwritten. Imported new files only:
  `cloud_benchmark_1d9bdc14e51a.json` (P100 benchmark PASSED before research
  training) and four `cloud_provenance_1d9bdc14e51a*.json` snapshots
  (environment history incl. pre-repair f0d2878 / cu128 attempt /
  post-repair 2b5d41c states). All other archive entries were content-
  identical modulo EOL to existing files and were not overwritten.
- Chronology of record: P100 benchmark passed BEFORE any research training;
  CLI/device plumbing repair (f0d2878, 2b5d41c) occurred BEFORE research
  training; the earliest cloud seed 11/22 attempts exited at argparse and
  produced NO research output; the uniform cohort then completed on P100;
  traditional calibration remained paused at 465/1250 during all cloud
  training; NO synthetic test metrics were inspected at any point.
- Validation losses in the cohort are validation-only training/selection
  evidence with hardware recorded; they are NOT final research performance,
  and no method winner is declared here.
- Frozen protocol config SHA-256 and dataset SHA-256 re-verified unchanged
  after import (values as recorded in section header above).
