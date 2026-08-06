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
python -m src.run_double_heston_validation
python -m src.run_smoke_test
python -m src.evaluate_repricing
```

Expected lightweight evidence is written beneath `outputs/metrics/smoke_test/`. The generated checkpoint, dummy surfaces, predictions, and row-level errors are reproducible and intentionally excluded from Git.

## Determinism and normalization

Python, NumPy, and PyTorch receive the same non-negative seed. PyTorch deterministic algorithms are requested. Complete surfaces are assigned to splits by surface ID. Target means and scales are fitted on training targets only, then reused for validation, testing, and inverse transformation. The test split never selects a checkpoint.

## Research-mode isolation

Research pricing routes to the independent canonical engine and never falls back to `dummy_surface_generator_for_smoke_test`. Full generation should use reviewed bounds. Provisional bounds require an explicit opt-in and are currently limited to controlled pilot generation of at most 100 surfaces.

## Reviewing the canonical pricing engine

1. Review the documented equations and repository-specific correlation convention.
2. Benchmark representative prices using an independent implementation or adaptive quadrature.
3. Re-run compilation, all tests, controlled recovery, pilot generation, smoke flow, and repricing.
4. Preserve every start, failure, boundary diagnostic, seed, and configuration.
5. Do not treat the implementation's own fixture as independent proof of correctness.

## Hash preservation

Record SHA-256 hashes for the validated pricing source, imported helpers, parameter-bound configuration, immutable split manifests, generated datasets, training configuration, and accepted checkpoints. Store large or confidential datasets outside Git and commit only non-sensitive manifests where authorized.
