# Double Heston Engine Freeze

Status date: 07 August 2026

## Frozen evidence

`outputs/engine_freeze/` contains lightweight text evidence only:

- `engine_manifest.json`
- `source_checksums.json`
- `fixture_checksums.json`
- `benchmark_summary.json`
- `parameter_sampling_summary.json`
- `reviewed_sampling_summary.json`
- `reviewed_sampling_decision.json`
- `validation_commands.txt`
- `decision.json`

The manifest records the pre-commit revision `dea6a19238e13fccf5243935ffb6df8199135595`, Python/NumPy/SciPy versions, production node counts, reference tolerances, parameter order, constraint definitions, and generation timestamp. SHA-256 files bind the reviewed production/reference/runner/audit sources, configs, and fixtures to the freeze evidence.

The freeze does not claim equivalence with unavailable teammate code, correctness merely from agreement, or validation on real NIFTY data. No large synthetic dataset was generated and no ANN or PINN training was started.

## Decision gate

The recorded decision is:

```text
NEEDS_SAMPLER_CORRECTION
```

The independent pricing benchmark passed and the earlier full primary validation
chain was observed passing when the suite contained 82 tests. After the final
standards and config-metadata corrections, the current post-correction compile,
84-test suite passed and the reviewed-sampling audit command completed
successfully with decision `NEEDS_SAMPLER_CORRECTION`; the entire benchmark,
controlled-validation, smoke, and repricing chain was not rerun again. The
reviewed sampling gate is not
`READY_FOR_SYNTHETIC_GENERATION`: 19,000 candidates were audited, normal
acceptance was 81.16% interior and 67.42% wide-valid, and four clean
boundary-challenge surfaces retained pricing validity failures. The audit does
not self-attest external validation, so `decision.json` truthfully preserves
`PENDING_PRIMARY_RERUN` and records the earlier observed command evidence
separately.
Rejected candidates and priced failures remain in the audit artifacts.

The production pricing source is frozen as the benchmarked engine candidate.
The ANN sampling design is reviewed evidence but remains provisional pending
financial/domain approval and sampler correction. No ANN training, final large
dataset, real-market/NIFTY validation, or teammate-equivalence claim is made.
