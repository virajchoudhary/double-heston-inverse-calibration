# Double Heston Engine Freeze

Status date: 06 August 2026

## Frozen evidence

`outputs/engine_freeze/` contains lightweight text evidence only:

- `engine_manifest.json`
- `source_checksums.json`
- `fixture_checksums.json`
- `benchmark_summary.json`
- `parameter_sampling_summary.json`
- `validation_commands.txt`
- `decision.json`

The manifest records the pre-commit revision `dea6a19238e13fccf5243935ffb6df8199135595`, Python/NumPy/SciPy versions, production node counts, reference tolerances, parameter order, constraint definitions, and generation timestamp. SHA-256 files bind the reviewed production/reference/runner/audit sources, configs, and fixtures to the freeze evidence.

The freeze does not claim equivalence with unavailable teammate code, correctness merely from agreement, or validation on real NIFTY data. No large synthetic dataset was generated and no ANN or PINN training was started.

## Decision gate

The recorded decision is:

```text
NEEDS_BOUNDS_REVIEW
```

The independent pricing benchmark passed, all 69 tests passed, and every required validation command completed. The gate is not `READY_FOR_SYNTHETIC_GENERATION` because the bounds audit found material sampling-design exposure: `44.48%` rejection and, among accepted vectors, `32.6729%` near any declared boundary, `7.0605%` Feller-near, `26.9452%` hard-bound-near, `9.2939%` with weak slow-fast separation, plus 17 similar-surface/separated-parameter pairs. Rejected invalid vectors are reported separately rather than classified as near-boundary.

The production pricing source is frozen as the benchmarked engine candidate. The ANN sampling design is reviewed evidence but remains provisional pending financial/domain approval and a revised interior-versus-challenge sampling policy.
