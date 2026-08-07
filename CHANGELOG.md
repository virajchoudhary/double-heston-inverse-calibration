# Changelog

All notable project changes are recorded here. This private academic repository does not grant an open-source licence or imply permission for external reuse.

## 0.4.0 - 2026-08-07

### Added

- Reviewed four-population Double Heston sampling design with deterministic
  interior, wide-valid, boundary-challenge, and OOD cohorts.
- 19,000-candidate reviewed audit evidence, per-distribution diversity metrics,
  retained raw-noise diagnostics, and expanded sampler tests.
- Reviewed sampling documentation and updated freeze/reproducibility records.

### Changed

- Replaced the prior single-box sampling recommendation with explicit challenge
  and OOD isolation and a `NEEDS_SAMPLER_CORRECTION` gate.

### Limitations

- Four clean challenge pricing-validity failures remain retained in the audit.
- Ranges remain provisional pending financial/domain review; no ANN training,
  final large dataset, teammate-equivalence claim, or NIFTY validation exists.

## 0.3.0 - 2026-08-06

### Added

- Independent adaptive-quadrature Double Heston reference pricer with retained integration diagnostics
- Frozen 36-case production-versus-reference benchmark and grouped error/runtime evidence
- Deterministic 5,000-candidate provisional-bounds audit with a 250-surface priced subset
- Reviewed sampling configuration separating hard limits, training ranges, margins, challenges, noise tests, and OOD tests
- Lightweight engine manifest, checksums, summaries, commands, and decision-gate evidence
- Reference/benchmark/audit-integrity tests, bringing the full suite to 69 passing tests

### Changed

- Expanded benchmark acceptance so 64-node, 96-node, reference reliability, no-arbitrage, and parity all participate in the case-level gate
- Classified the freeze as `NEEDS_BOUNDS_REVIEW` after material rejection, boundary concentration, and similar-surface parameter-pair findings
- Updated project status and reproducibility guidance for the pre-ANN decision gate

### Limitations

- Pricing agreement does not prove universal correctness or equivalence with unavailable teammate code
- Sampling ranges remain provisional pending financial/domain review
- No large synthetic research dataset, ANN/PINN training, or real NIFTY validation was performed

## 0.2.0 - 2026-08-06

### Added

- Independent canonical Double Heston European-option pricing engine
- Little-Heston-Trap characteristic function and 64-node Gauss-Laguerre default
- Engine-focused tests and canonical reimplementation regression fixture
- Deterministic constrained SciPy calibration with full repeated-start records
- Clean and 1% noise validation outputs
- Twelve-surface genuine canonical-engine pilot
- Provisional bounds separating hard safety and pilot sampling ranges
- Engine and validation documentation

### Changed

- Routed the ANN research pricing adapter to the canonical engine
- Kept the dummy generator exclusive to explicit smoke-test mode
- Updated status and handoff documentation for single-person continuation
- Made repository-root module commands the supported invocation form

### Limitations

- No equivalence claim to unavailable teammate source
- Provisional ranges are not externally confirmed original bounds
- No full ANN/PINN research training or NIFTY validation
- No unique-recovery claim from optimizer success or synthetic fit

## 0.1.0 - 2026-08-06

### Added

- Ordinary ANN inverse-calibration starter infrastructure
- Ten-parameter contract
- Constraint validation
- Surface grid
- PyTorch ANN
- Dataset and training system
- Evaluation modules
- Unit tests
- Smoke-test pipeline
- Heston PDF handoff and Markdown companion
- Research-status documentation

### Fixed

- Applied the deterministic seed before ANN construction in smoke and CLI training flows
- Removed the absolute local checkpoint path from tracked smoke metadata

### Limitations

- No real Double Heston engine
- No confirmed bounds
- No genuine synthetic surfaces
- No NIFTY validation
- No ANN research result
