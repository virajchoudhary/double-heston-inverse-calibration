# Reviewed Double Heston Parameter Sampling

Status date: 07 August 2026

## Decision

The reviewed sampler is deterministic and auditable, but the gate remains
`NEEDS_SAMPLER_CORRECTION`. The clean priced subset retained four challenge
failures: three tiny lower-bound/no-arbitrage deviations (minimum normalized
price about `-7.12e-8`) and one put-strike monotonicity failure. No failed row was
dropped and no range was tuned after observing the result.

## Design

The sampler uses SciPy Latin-hypercube coordinates in latent space. Conditional
transforms are documented as latent-coordinate LHS, not physical-space LHS.
`kappa_slow` is sampled first; `kappa_fast` is then sampled with the declared
slow/fast gap. For each factor, `sigma = f * sqrt(2*kappa*theta)` gives a
strict Feller margin. Correlations use polar coordinates with component-wise
envelopes and the joint correlation disk enforced. Fixed candidate populations
retain every rejection; there is no accepted-row refill.

The four populations are:

- `interior_train`: 10,000 candidates, 8,116 accepted (`81.16%`). Accepted
  hard-bound, Feller, weak-separation, disk, and union-boundary rates were all
  zero.
- `wide_valid_train`: 5,000 candidates, 3,371 accepted (`67.42%`).
- `boundary_challenge`: 2,000 valid candidates, balanced across
  `near_feller`, `weak_separation`, `near_hard_bound`, and
  `near_correlation_disk` (500 each). Rows are challenge-labelled and excluded
  from ordinary train/validation unless explicitly opted in.
- `ood_test`: 2,000 valid evaluation-only candidates. Every row is outside the
  normal `kappa_fast` support: observed minimum `10.250775...` versus normal
  upper support `10.0`; no row receives a train or validation assignment.

The requested noise levels are exactly `0`, `0.005`, `0.01`, and `0.02`.
Deterministic raw-noise diagnostics are retained in the priced metrics and the
summary. They do not clip, project, or drop failures, and they are not claimed
to be arbitrage-preserving or READY-gating evidence.

## Evidence and limitations

The audit prices 500 interior, 250 wide, 250 challenge, and 250 OOD surfaces on
the fixed diagnostic grid. It records finite-price, no-arbitrage,
strike-monotonicity, convexity, maturity, skew, smile-curvature, term-structure,
factor-contribution, and per-distribution parameter-correlation diagnostics.
All 13 requested audit artifacts are under
`outputs/reviewed_sampling_audit/`. The production pricing source checksum is
unchanged. An earlier full primary validation chain passed when the suite
contained 82 tests. After final standards and config-metadata corrections, the
post-correction compile and 84-test suite passed, and the reviewed audit command
completed successfully with decision `NEEDS_SAMPLER_CORRECTION`; the entire
benchmark, controlled-validation, smoke, and repricing chain was not rerun.
The freeze therefore remains `PENDING_PRIMARY_RERUN` and does not self-attest
external validation.

This is synthetic engineering evidence only. It is not financial approval,
real-market or NIFTY validation, ANN training evidence, a claim of equivalence
to unavailable teammate code, or a statistical identifiability result.
