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

The current package layout is run from the parent directory of `ann_inverse_calibration`:

```powershell
python -m compileall ann_inverse_calibration
python -m pytest ann_inverse_calibration/tests -q
python -m ann_inverse_calibration.src.run_smoke_test
```

Expected lightweight evidence is written beneath `outputs/metrics/smoke_test/`. The generated checkpoint, dummy surfaces, predictions, and row-level errors are reproducible and intentionally excluded from Git.

## Determinism and normalization

Python, NumPy, and PyTorch receive the same non-negative seed. PyTorch deterministic algorithms are requested. Complete surfaces are assigned to splits by surface ID. Target means and scales are fitted on training targets only, then reused for validation, testing, and inverse transformation. The test split never selects a checkpoint.

## Research-mode isolation

Research generation requires both a validated Double Heston pricing adapter and fully confirmed bounds marked `TEAMMATE_CONFIRMED`. It must fail clearly when either is absent. It must never fall back to `dummy_surface_generator_for_smoke_test`.

## Adding the real pricing engine

1. Verify the received source and helper checksums.
2. Run the teammate's original pricing and controlled-recovery tests unchanged.
3. Document its callable input/output shapes, numerical settings, tolerances, and failure behavior.
4. Add only a thin adapter in `src/pricing_interface.py`; do not alter the validated mathematics silently.
5. Re-run compilation, all local tests, controlled recovery, genuine generation, and repricing checks.

## Hash preservation

Record SHA-256 hashes for the validated pricing source, imported helpers, parameter-bound configuration, immutable split manifests, generated datasets, training configuration, and accepted checkpoints. Store large or confidential datasets outside Git and commit only non-sensitive manifests where authorized.
