# Reproducibility

## Verified environment

- Python 3.13.9
- PyTorch 2.11.0+cpu
- CPU execution; CUDA unavailable
- Default deterministic seed: 42

Install dependencies with the active interpreter:

```powershell
python -m pip install -r requirements.txt
```

## Validation commands

Run from the repository root:

```powershell
python -m compileall .
python -m pytest tests -q
python -m src.run_independent_pricing_benchmark
python -m src.audit_parameter_bounds
python -m src.run_double_heston_validation
python -m src.run_smoke_test
python -m src.evaluate_repricing
```

After every command passes, refresh the decision metadata without re-running the audit:

```powershell
python -m src.audit_parameter_bounds --freeze-only --commands-passed
```

Benchmark, audit, and freeze evidence is written under `outputs/double_heston_benchmark/`, `outputs/parameter_bounds_audit/`, and `outputs/engine_freeze/`. Smoke evidence remains beneath `outputs/metrics/smoke_test/`. Generated checkpoints, dummy surfaces, predictions, and row-level smoke errors are reproducible and intentionally excluded from Git.

## Determinism and normalization

Python, NumPy, and PyTorch receive the same non-negative seed. PyTorch deterministic algorithms are requested. Complete surfaces are assigned to splits by surface ID. Target means and scales are fitted on training targets only, then reused for validation, testing, and inverse transformation. The test split never selects a checkpoint.

## Research-mode isolation

Research pricing routes to the independent canonical engine and never falls back to `dummy_surface_generator_for_smoke_test`. Full generation should use reviewed bounds. Provisional bounds require an explicit opt-in and are currently limited to controlled pilot generation of at most 100 surfaces.

## Reviewing the canonical pricing engine

1. Review the documented equations and repository-specific correlation convention.
2. Re-run the frozen 36-case adaptive-quadrature benchmark and inspect every failure/outlier row.
3. Re-run compilation, all tests, controlled recovery, pilot generation, smoke flow, and repricing.
4. Preserve every start, failure, boundary diagnostic, seed, and configuration.
5. Do not treat the implementation's own fixture as independent proof of correctness.

## Frozen benchmark and audit determinism

- Benchmark tolerances are fixed in source and fixture metadata before execution.
- The reference uses `epsabs=1e-10`, `epsrel=1e-10`, and integration limit `500`.
- The bounds audit uses `scipy.stats.qmc.LatinHypercube`, seed `20260806`, exactly 5,000 raw candidates, and a deterministic ten-stratum priced subset capped at 250 surfaces.
- Runtime columns naturally vary; numerical and status outputs are tested for determinism within tight numerical tolerance.
- `configs/parameter_bounds_PROVISIONAL.yaml` is read-only input to the audit. Reviewed evidence is written separately to `configs/parameter_sampling_REVIEWED.yaml`.

## Hash preservation

Record SHA-256 hashes for the validated pricing source, imported helpers, parameter-bound configuration, immutable split manifests, generated datasets, training configuration, and accepted checkpoints. Store large or confidential datasets outside Git and commit only non-sensitive manifests where authorized.
