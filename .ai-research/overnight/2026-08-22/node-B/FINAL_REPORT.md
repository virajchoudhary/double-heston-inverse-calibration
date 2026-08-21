# Node B Final Report — Double Heston Identifiability & Calibration Investigation

Overnight run 2026-08-22, ~00:50–~03:50 IST. Branch `overnight/20260822-b-identifiability`
from genesis `642702e6706a3d17b3031619f35bda39bc144483` (== origin/main at start).
Compute: CPU-only (12 cores), Python 3.13.5, numpy 2.2.5, scipy 1.15.3, torch 2.13.0.
No environment mutation. All experiments are synthetic; no real-market data touched any
neural or calibration weight update; no final dataset generated; G2 untouched.

---

## Executive conclusion

The ten-parameter recovery failure is a **MIXED mechanism dominated by practical
non-identifiability at realistic observation tolerance, with a secondary optimizer-basin
component on clean data**:

1. **The provisional 108-quote grid is NOT locally rank-deficient** (median condition
   number ~2.6e4, practical rank 10/10) — four orders of magnitude better than the
   committed G2 central-5 geometry (6.5e8, rank 7.5). Local conditioning is not the
   explanation for failures on the full grid.
2. **The weakest sensitivities sit below the noise floor.** The smallest singular value
   (~1.4e-5 in spot-normalized units) is smaller than 0.5–2% observational noise
   (~2.5e-4–3e-3). Linearized propagation predicts 3–29 full-range widths of parameter
   displacement at 0.5% noise; the constraint box truncates this to ~0.3 — matching the
   observed 100% boundary-saturation and the parameter-RMSE step from 0.097 (clean) to
   ~0.31 (0.5% noise).
3. **The ambiguity is tolerance-dependent, not binary.** On the full grid, machine-precision
   fits (≤2.5e-7 normalized price RMSE) pin parameters near truth (max RMSE 0.011 — the
   committed central-5 distant near-equivalents are gone), but the equivalence radius grows
   monotonically to 0.38 full-range RMSE at tolerance 1e-4 normalized (0.01 currency units
   on spot 100) — still ~2.5× below the smallest realistic noise floor. Realistic noise
   places any market-precision calibration deep inside the ambiguous regime.
4. **Optimizer capacity and initialization are ruled out as primary mechanisms.** Eight-fold
   budget from fresh starts leaves dispersion unchanged; Sobol starts change nothing;
   relative-price (vega-like) weighting makes parameters worse while converging more
   confidently. Warm-started polish fixes one of four cases — basins, not budget.

Classification: **ILL-CONDITIONED-AT-NOISE-SCALE + STRUCTURALLY/PRACTICALLY NON-IDENTIFIABLE
at market tolerance, MIXED with case-dependent optimizer-basin sensitivity on clean data.**
REPRESENTATION-LIMITED applies to the committed central-5 market geometry (rank 7.5,
κ 6.5e8) but NOT to the provisional full grid.

## Baseline reproduced

Using the production entry point `src.calibrate_double_heston.calibrate_double_heston`
unchanged (3 canonical starts, TRF, max_nfev=300, production 64-node pricer) on the full
108 grid, 4 representative truth vectors (committed global-ambiguity selection):

- Clean: best starts recover truth exactly (price RMSE 4.4e-14, parameter RMSE 4.3e-12);
  worst starts disperse to parameter RMSE 0.39. Matches committed "one exact best result,
  start sensitivity retained".
- 1% noise: price RMSE 0.123–0.138 (at the noise floor), parameter RMSE 0.17–1.33,
  92% boundary-near. Matches committed "comparatively reasonable price fit coexists with
  severe deterioration in ten-parameter identification".
- No discrepancy with committed results found. Runtime 742 s / 24 starts.

## Experiments completed

All logged in `EXPERIMENTS.jsonl` with seeds, configs, runtimes; artifacts under
`artifacts/`, `tables/`, `figures/`.

| ID | Experiment | Size | Seed base |
| --- | --- | --- | --- |
| fast_pricer_validation | Vectorized diagnostic pricer vs production engine | 4 cases × (108 + 20 quotes) | deterministic |
| phase_a_canonical_baseline | Production-entry recovery, clean + 1% | 4 cases × 2 noise × 3 starts | 42 / 20260822-derived |
| phase_d_jacobian_conditioning | Scaled-Jacobian SVD across 16 geometries | 4 representative + 16 interior samples | 20260822 |
| phase_b_multistart_full108 | Multi-start recovery, 0/0.5/1/2% noise | 4 cases × 4 noise × 12 starts = 192 | 27182818 |
| linearized_noise_propagation | trace(J⁺ΣJ⁺ᵀ) vs observed collapse | 4 cases × 3 noise levels | — |
| phase_e_landscape_profiles | Valley scan + compensated profiles | 3 scans × 41 pts; 3 params × 11 offsets × 2 starts | 20260822 |
| phase_f_tolerance_spectrum | Equivalence radius vs price tolerance | 48 clean solutions × 6 tolerances | — |
| phase_h_improvement_arms | baseline/sobol/relweight/polish/prior | 4 cases × 5 arms × 12 starts | 20260823 |
| phase_h_fresh800_capacity_control | Fresh starts, 8× budget | 4 cases × 12 starts | 20260824 |

## Strongest quantitative findings

1. **Conditioning spectrum** (median over 4 representative truths):
   full108 κ=2.55e4 (rank 10/10); central5x6 κ=6.42e4; wings4x6 κ=2.74e4; long3 1.27e5;
   short3 1.70e6 (rank 9.5); central-5 market 27/55 κ=6.54e8 (rank 7.5) — committed G2
   ill-conditioning independently replicated; single maturities κ≥1.4e11 (rank ~5).
   **Maturity span, not moneyness width, buys conditioning.** Calls and puts are
   informationally redundant (identical conditioning to 5 digits).
2. **Noise collapse step** (12 starts × 4 cases, full grid): median parameter RMSE
   0.097 → 0.309 → 0.306 → 0.287 at 0/0.5/1/2% noise; boundary-hit fraction 35% → 100% →
   98% → 81%; median price RMSE tracks the noise floor at every level; noisy solutions
   still reprice the TRUE surface within 2–5e-4.
3. **Linearized noise propagation**: predicted displacement 3.1–28.9 widths at 0.5% noise
   (9.4–115 at 2%); observed ~0.3 truncated by the declared box; case ordering matches
   (best-conditioned case_4 smallest prediction and best observed recovery).
4. **Tolerance spectrum** (clean, full grid): max parameter RMSE 0.011 / 0.159 / 0.235 /
   0.323 / 0.380 at price-RMSE tolerance 2.5e-7 / 1e-6 / 3e-6 / 1e-5 / 1e-4 (normalized).
5. **Compensated profiles** (case_1): kappa_slow fixed at up to 7× truth — other nine
   re-optimize to objective ~1e-12; theta_fast at 5–8× truth — repricing stays below the
   0.5% noise floor.

## Near-equivalent solutions

Strongest preserved example (case_1, start `deterministic_broad_10`, price RMSE 6.7e-7
normalized = 6.7e-5 currency units, parameter RMSE 0.159; full vectors in
`artifacts/phase_f_tolerance_spectrum.json`):

| parameter | truth | recovered | relative error |
| --- | ---: | ---: | ---: |
| kappa_slow | 0.7201 | 1.7320 | +141% |
| theta_slow | 0.0627 | 0.1027 | +64% |
| sigma_slow | 0.2120 | 0.2376 | +12% |
| rho_slow | -0.1803 | -0.1603 | +11% |
| v0_slow | 0.0731 | 0.1200 | +64% |
| kappa_fast | 3.7035 | 4.4269 | +20% |
| theta_fast | 0.0433 | 0.0054 | -88% (floor) |
| sigma_fast | 0.2628 | 0.1858 | -29% |
| rho_fast | -0.1036 | +0.0058 | sign flip |
| v0_fast | 0.0679 | 0.0210 | -69% |

The committed compensation pattern (slow-factor inflation vs fast-factor collapse;
v0_slow/v0_fast, theta_slow/theta_fast) reappears on the full grid. Latent interpretation
differs materially: recovered half-lives, slow/fast variance shares, and expected variance
paths diverge while the observable surface agrees to 6.7e-7 (see `phase_f_exemplars.json`
latent tables). 18/48 clean runs are strict-threshold near-equivalent; displaced solutions
form one coherent alternative cluster per case.

## Jacobian / conditioning findings

See Finding 1 above and `tables/phase_d_jacobian_conditioning.csv`. Weakest scaled
directions are broad combinations dominated by slow-factor (kappa, theta, v0_slow) and
fast-factor (theta_fast, v0_fast) terms — consistent with the observed compensation pairs.
The exact factor-swap symmetry was verified bitwise (max price difference 0.0): the model
is exactly invariant under exchanging the two factor blocks; the declared
`kappa_slow < kappa_fast` ordering is the only thing breaking this degeneracy. Local
Jacobian evidence is labeled local sensitivity/conditioning only.

## Noise robustness

Parameter recovery collapses discontinuously between 0 and 0.5% noise (0.097 → 0.31) and
does not degrade much further to 2%; repricing error tracks the noise floor throughout;
the noisy solutions remain good repricers of the true surface (clean-price RMSE 2–5e-4).
Degradation is broad-based across 8–10 parameters (rho_slow and sigma_fast most sensitive;
per-parameter table in `tables/phase_c_per_parameter_degradation.csv`). This is the
signature of a near-degenerate weakest subspace excited by noise, not of one fragile
parameter.

## Optimizer findings

- TRF `success` flags are uninformative: relweight arm converges 92% of the time to the
  WORST parameters; clean baseline converges only 42% yet often lands exactly on truth.
- Capacity: 8× budget from fresh starts changes nothing (case medians 0.16/0.06/0.32/0.00);
  warm-started polish fixes case_3 only (0.259 → 0.0002). Committed `OPTIMIZER_CAP_UNRESOLVED`
  is answered for the clean full-grid problem: capacity alone does not resolve dispersion;
  basin selection (informed/warm starts) helps case-dependently.
- Initialization coverage (Sobol) does not help. Relative-price (vega-like) weighting hurts
  parameter recovery while improving convergence statistics.
- Prior-ranges arm: stabilizes only the case whose truth lies inside the prior; actively
  biases the others (truths outside prior support). Prior-driven stabilization is distinct
  from data-driven identification and must be reported separately.

## Representation information findings (provisional only)

- The full 108 grid's calls+puts duplication adds no parameter information over one option
  type given the carry contract (identical Jacobian conditioning) — a genuine
  representation-economy observation for the later decision.
- Maturity span dominates information content: 6 maturities ≫ 2 maturities ≫ 1 maturity;
  long maturities (60–180d) carry more than short (7–30d). This provisional evidence may
  inform — not freeze — the G2 representation decision. No new grid is proposed as final.

## Failed hypotheses

- **Local rank deficiency on the full grid** — refuted (rank 10/10, κ~2.6e4).
- **Optimizer budget as the primary bottleneck** — refuted by the fresh-800 control.
- **Initialization coverage (quasi-random starts)** — refuted (sobol ≈ baseline).
- **Objective weighting (relative-price/vega-like) as a fix** — refuted (worse parameters).
- **Numerical precision of the pricer** — refuted (diagnostic pricer bitwise-equal to the
  frozen engine; clean exact recoveries occur routinely).
- **Prior ranges as a free lunch** — refuted when prior is mis-specified (bias, not stability).

## Scientific interpretation

- **Established (multi-run, cross-validated, reproducible tonight):** production baseline
  behavior; G2 central-5 conditioning; full-grid local full rank; noise-collapse step with
  boundary saturation; linearized noise propagation scale; tolerance-dependent equivalence
  radius; exact factor-swap symmetry; optimizer/weighting negative results.
- **Strong inference:** the market-precision inverse problem on any single-date surface
  (even the full provisional grid) is practically non-identifiable at the ten-parameter
  level — the equivalence set at realistic tolerance spans a large fraction of the declared
  box; the constraint box, not the data, terminates noise-driven drift.
- **Unresolved speculation:** whether the ambiguity radius varies systematically with
  vol-of-vol magnitude (case_4's high sigmas and its unique stability suggest so, n=1);
  whether multi-date or complementary-observable conditioning can push the weakest
  singular values above the noise floor globally (committed evidence so far: insufficient);
  the exact geometry of the alternative basins (tonight: one coherent cluster per case).

## Implications for ANN/PINN work

- A network trained on (parameter → surface) pairs learns the forward map fine — the
  forward problem is well-posed. The INVERSE map it is asked to approximate is set-valued
  at realistic tolerance: many materially different ten-vectors map to
  observationally indistinguishable surfaces. Expecting unique parameter recovery from a
  single 108-quote surface is not supported by the information content measured tonight.
- Consequences: (a) evaluation metrics must separate repricing error from parameter error
  (tonight and committed evidence show they decouple completely); (b) noisy-surface
  training will teach the network the equivalence SET (posterior-like behavior), and its
  parameter outputs will be prior-regularized by the training distribution — that must be
  measured and labelled, not mistaken for identification; (c) any ANN/PINN comparison
  should report recovery against the tolerance spectrum (as in `phase_f_tolerance_spectrum.csv`),
  not a single RMSE; (d) PINN physics residuals (Node C: canonical PDE certified) cannot add
  information the observables lack — physics regularizes, it does not identify.
- Node A's finding that the two PINN stacks disagree on parameter ORDER and constraint
  semantics compounds this: cross-stack comparisons without a permutation/contract adapter
  would confound stack differences with the identifiability limits documented here.

## Changes made

Added only (no `src/`, no canonical documents touched):
`scripts/node_b_toolkit.py`, `scripts/run_node_b_validate_fast_pricer.py`,
`scripts/run_node_b_phase_a_baseline.py`, `scripts/run_node_b_phase_d_jacobian.py`,
`scripts/run_node_b_phase_b_multistart.py`, `scripts/run_node_b_phase_e_profiles.py`,
`scripts/run_node_b_phase_h_improvements.py`, `scripts/run_node_b_noise_propagation.py`,
plus `.ai-research/overnight/2026-08-22/node-B/` evidence (STATUS, FINDINGS,
EXPERIMENTS.jsonl, artifacts/tables/figures). Commits: 77b8f2e, 61905d0, and this report.

## Tests

`python3 -m pytest tests/test_parameter_order.py tests/test_constraints.py tests/test_double_heston_engine.py -q`
→ **43 passed, 1 failed**. The failure
(`test_canonical_reimplementation_fixture_is_reproducible`) is PRE-EXISTING: it reproduces
identically at the genesis commit in a clean worktree; the deviation is 3.3e-12 against a
demanded atol of 1e-12 (platform floating-point rounding, numpy 2.2.5 / macOS arm64).
Documented, not silently rewritten.

## Recommended next experiments (ordered by scientific value)

1. **Tolerance-spectrum at scale**: repeat the phase-F spectrum on ~100–300 maximin
   interior truths (cheap with the validated fast pricer) to map the ambiguity radius
   distribution and its covariates (vol-of-vol, kappa separation, v0/theta ratios).
2. **Noise-floor-vs-sigma_min intervention**: synthesize noise levels bracketing
   sigma_min (1e-5 … 1e-3) to measure the exact collapse threshold per case and validate
   the linearized prediction quantitatively.
3. **Identifiability-weighted representation search**: rank candidate grids by
   sigma_min/noise-floor ratio (not condition number alone) under a market-realistic noise
   model, then run recovery only on the top candidates — feeds the G2 representation
   decision with the metric that actually failed.
4. **Basin cartography for case_1**: map the alternative basin (its extent, boundary
   contact, latent-factor interpretation) to explain the case heterogeneity.
5. **Equivalence-set learning target**: prototype an ANN whose target is a distribution
   over the equivalence set (or identifiable functionals: total-variance path, half-lives)
   instead of point parameters — directly aligned with the measured information content.

## Branch and final commit

Branch: `overnight/20260822-b-identifiability` (pushed). Final commit: see `git log -1`
recorded in STATUS.md at close. Genesis: `642702e6706a3d17b3031619f35bda39bc144483`.
