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
python -m src.audit_reviewed_sampling
python -m src.run_double_heston_validation
python -m src.run_smoke_test
python -m src.evaluate_repricing
```

The reviewed audit deliberately does not self-attest the complete external
validation chain, so the canonical freeze remains `PENDING_PRIMARY_RERUN`. There
is no trusted metadata-only finalizer in this workflow. The legacy
`src.audit_parameter_bounds` command and its `--freeze-only` mode reproduce the
historical bounds-audit workflow and must not be used to refresh the canonical
reviewed freeze because they overwrite reviewed config or freeze evidence.

Benchmark, reviewed-audit, and freeze evidence is written under
`outputs/double_heston_benchmark/`, `outputs/reviewed_sampling_audit/`, and
`outputs/engine_freeze/`. Historical bounds-audit evidence remains under
`outputs/parameter_bounds_audit/`. Smoke evidence remains beneath
`outputs/metrics/smoke_test/`. Generated checkpoints, dummy surfaces,
predictions, and row-level smoke errors are reproducible and intentionally
excluded from Git.

## Determinism and normalization

Python, NumPy, and PyTorch receive the same non-negative seed. PyTorch deterministic algorithms are requested. Complete surfaces are assigned to splits by surface ID. Target means and scales are fitted on training targets only, then reused for validation, testing, and inverse transformation. The test split never selects a checkpoint.

## Research-mode isolation

Research pricing routes to the independent canonical engine and never falls back to `dummy_surface_generator_for_smoke_test`. Full generation should use reviewed bounds. Provisional bounds require an explicit opt-in and are currently limited to controlled pilot generation of at most 100 surfaces.

## Prepared reviewed-core ANN pilot

`configs/ann_dataset_FIRST_RESEARCH.yaml` and
`outputs/core_dataset_readiness/core_dataset_readiness.json` record a prepared,
unexecuted 10,000-surface normal-clean plan. Its price-only estimate for 108
quotes per surface is about 16.04 minutes mean and 16.34 minutes p95; it excludes
selection, validation, serialization, hashing, retries, and contention. The
reviewed-core generator has not been implemented, so no generation command is
currently available. Implementing and testing that generator is the next
milestone; dataset generation still requires separate execution authorization.

## Reviewing the canonical pricing engine

1. Review the documented equations and repository-specific correlation convention.
2. Re-run the frozen 36-case adaptive-quadrature benchmark and inspect every failure/outlier row.
3. Re-run compilation, all tests, controlled recovery, pilot generation, smoke flow, and repricing.
4. Preserve every start, failure, boundary diagnostic, seed, and configuration.
5. Do not treat the implementation's own fixture as independent proof of correctness.

## Frozen benchmark and audit determinism

- Benchmark tolerances are fixed in source and fixture metadata before execution.
- The reference uses `epsabs=1e-10`, `epsrel=1e-10`, and integration limit `500`.
- The historical bounds audit uses `scipy.stats.qmc.LatinHypercube`, seed `20260806`, exactly 5,000 raw candidates, and a deterministic ten-stratum priced subset capped at 250 surfaces.
- The reviewed sampling audit uses latent-coordinate `scipy.stats.qmc.LatinHypercube`, seed `20260807`, fixed populations of 10,000 interior, 5,000 wide-valid, 2,000 challenge, and 2,000 OOD candidates, with clean pricing caps of 500/250/250/250 and retained raw-noise diagnostics at levels `0`, `0.005`, `0.01`, and `0.02`.
- The reviewed audit does not self-attest the external validation chain. Freeze evidence remains `PENDING_PRIMARY_RERUN` until a separately trusted finalization records the primary-session results.
- Runtime columns naturally vary; numerical and status outputs are tested for determinism within tight numerical tolerance.
- `configs/parameter_bounds_PROVISIONAL.yaml` is read-only input to the audit. Reviewed evidence is written separately to `configs/parameter_sampling_REVIEWED.yaml`.

## Hash preservation

Record SHA-256 hashes for the validated pricing source, imported helpers, parameter-bound configuration, immutable split manifests, generated datasets, training configuration, and accepted checkpoints. Store large or confidential datasets outside Git and commit only non-sensitive manifests where authorized.
