# Cloud Run Package — R2 Primary Comparison (execution-environment migration ONLY)

This package moves REMAINING compute for the frozen R2 primary comparison to
cloud hardware (Kaggle/Colab) WITHOUT changing any frozen scientific setting.
The frozen protocol is `configs/r2_primary_comparison_FINAL.yaml` +
`docs/R2_PRIMARY_COMPARISON_PROTOCOL.md` (commit `5ea9fd0`, Issue #32).

```text
SCIENTIFIC_SETTINGS = UNCHANGED (data, splits, features, architectures,
losses, weights, optimizer settings, seeds 11/22/33, calibration budget,
metrics, representative rule)
EXECUTION_ENVIRONMENT = MAY_CHANGE (CPU -> GPU/other CPU, threads, workers)
```

## CRITICAL float64 warning — read before choosing an accelerator

Model 2's differentiable repricing term runs in **float64/complex128**
(`src/torch_double_heston.py::price_double_heston_surface_batch_vectorized`;
float32 is numerically invalid for this formulation — see
`docs/R2_PRIMARY_COMPARISON_PRE_TRAINING_AUDIT.md` D).

GPU FP64 throughput by accelerator (published specs):

| Accelerator | FP64 | FP64:FP32 | Suitability for this workload |
|---|---|---|---|
| Kaggle NVIDIA P100 16GB | ~4.7 TFLOPS | 1:2 | GOOD on paper — benchmark first |
| Colab T4 16GB | ~0.1 TFLOPS | 1:32 | POOR — likely slower than local CPU |
| Colab L4 | ~0.25 TFLOPS | 1:32 | POOR |
| Colab A100 (Pro) | ~9.7 TFLOPS | 1:2 | GOOD on paper — benchmark first |
| Local GTX 1650 | ~0.14 TFLOPS | 1:32 | POOR (and local torch is CPU-only) |
| Local Ryzen 5 4600H CPU | measured baseline | — | current baseline |

Do NOT assume CUDA ⇒ faster. Run `python scripts/cloud/benchmark_cloud.py`
FIRST (development benchmark, `DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT`); it
measures actual Model-2 seconds/step and float64 pricer equivalence on the
available device, and prints a JSON report. Migrate Model 2 only if the
measured speedup is meaningful (recommend >= 2x vs the local clean baseline
0.26 s/step at 2 CPU threads; local contended runs are slower).

Traditional calibration (scipy/numpy, per-surface parallel) needs MANY CPU
CORES, not GPU: Kaggle notebooks give 4 CPU cores and Colab ~2 — both WORSE
than the local 12-thread Ryzen. Keep calibration local unless migrating to a
real multi-core cloud VM (>= 16 vCPU recommended).

## What to upload

1. This repository at the exact migration commit (git archive, below).
2. `data/final_r2_clean_10000/surfaces.jsonl` (37.7 MB, SHA-256
   `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`).
3. For calibration resume (if migrating calibration): copy
   `evidence/r2_primary_comparison_20260823/traditional_calibration_starts_journal.jsonl`
   to the same relative path on the target machine.

Create the archive at the exact SHA:

```bash
git archive --format=tar.gz -o r2_primary_cloud.tar.gz <MIGRATION_SHA>
tar -xzf r2_primary_cloud.tar.gz -C r2_cloud_repo
cp data/final_r2_clean_10000/surfaces.jsonl r2_cloud_repo/data/final_r2_clean_10000/
```

## Setup on the cloud machine / notebook

```bash
cd r2_cloud_repo
pip install -r scripts/cloud/requirements-cloud.txt
python scripts/cloud/verify_environment.py   # hashes + provenance; must PASS
python scripts/cloud/benchmark_cloud.py      # dev benchmark; read JSON output
python -m pytest tests/test_r2_primary_implementation.py -q   # must pass
```

`verify_environment.py` FAILS if the dataset hash, protocol-config hash, or
protocol status marker differ from the frozen values. It writes
`evidence/r2_primary_comparison_20260823/cloud_provenance_<host>.json`
recording Python/torch/numpy/scipy versions, CUDA availability, device name,
FP64 capability, core count, and the git SHA.

## Model 2 on an accelerator (after a POSITIVE benchmark)

The training CLI exposes an explicit execution-placement flag (default
`cpu`; CUDA is never auto-selected; placement only — frozen numerics,
architecture, loss, optimizer, batch sizes, and seeds are unchanged;
provenance records the device):

```bash
python -m src.r2_primary.training --model model2 --seed 11 --device cuda
python -m src.r2_primary.training --model model2 --seed 22 --device cuda
python -m src.r2_primary.training --model model2 --seed 33 --device cuda
```

(Leave off `--device` or pass `--device cpu` for CPU execution. There is no
environment-variable device override by design: placement must be explicit.)

Uniformity rule (predeclared, not result-based): if ANY Model-2 seed runs on
an accelerator, run ALL THREE seeds there; a local CPU run of a seed already
completed locally is retained as extra provenance evidence, never swapped in
or out based on metrics.

Model-2 training has NO mid-run resume: stopping a run loses it (checkpoints
contain no optimizer/RNG/epoch-cursor state). Stopping is safe ONLY between
seeds. A from-scratch rerun with the same seed is scientifically exact
(determinism pinned by
`tests/test_r2_primary_implementation.py::test_model1_training_is_deterministic_and_writes_provenance`
and identical seeding logic in Model 2).

## Traditional calibration (resume-safe anywhere)

```bash
python -m src.r2_primary.calibration --workers <CORES-1> --split test \
  --output evidence/r2_primary_comparison_20260823/traditional_calibration_starts.csv
```

The runner journal-appends every completed surface (3 start rows each) to
`..._journal.jsonl` and resumes by skipping already-complete surface ids.
The journal is plain JSONL: copy it between machines freely. There is no
recomputation and no surface cherry-picking — every one of the 1,250 test
surfaces must be present exactly once before `final_evaluation` runs. Each
surface's starts are deterministic (module seed 42) and independent of the
machine; cross-machine rows agree to floating-point library differences
(~1e-12), which the frozen analysis does not depend on. The final CSV is
rebuilt from the journal only when every requested surface is complete.

## Result synchronization (cloud -> local)

Download after each completed stage:

- `checkpoints/r2_primary_comparison/model2_seed*/` (training_summary.json,
  training_history.csv, both checkpoints)
- `evidence/r2_primary_comparison_20260823/cloud_provenance_*.json`
- benchmark JSON if produced
- for calibration migration: the updated `..._journal.jsonl`

Then run locally (or on any verified machine):

```bash
python -m src.r2_primary.final_evaluation
```

## Session limits (plan around them)

- Kaggle P100 session: up to ~9-12 h GPU; Model-2 seed ~2 h local CPU
  (58 epochs early-stopped) — one seed per session start if speedup is <3x,
  all three if >=3x.
- Colab free T4: unsuitable for this FP64 workload (expected).
- Colab Pro A100: availability varies; same benchmark gate applies.
