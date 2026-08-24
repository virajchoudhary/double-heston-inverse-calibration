# R2 Observation-Noise Robustness Protocol

Status: **FROZEN_BEFORE_ANY_NOISY_RESEARCH_RESULT**
Date: 2026-08-24 · Branch: `research/r2-noise-robustness` · Issue: #34
Canonical base (merged primary milestone): main @ `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
Frozen config: `configs/r2_noise_robustness_FINAL.yaml`
Frozen config SHA-256: `2fa49b3eb885d3427c01ab0cfe447fc6ddd7f19957db73c4b4ed782476c57c5a`
Frozen clean dataset: `data/final_r2_clean_10000/surfaces.jsonl` SHA-256 `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`
Frozen primary protocol SHA-256 (unchanged, referenced only): `33ca0f763ec10bb2424eefb02448c9c8e50021854b96a948e420f44bdba70781`

Nothing in this protocol may change after any positive-noise research result
exists. The completed primary comparison and its evidence are immutable
canonical baselines; this study never modifies them.

## 1. Scientific question

How do the already-frozen primary inverse methods — ordinary ANN,
constraint + differentiable-repricing-informed model (NOT a PDE-informed
PINN), and traditional numerical calibration — degrade when R2 option-price
observations are perturbed by controlled market-scale noise while the true
Double Heston parameter vectors remain known?

**Central hypothesis under test (not presumed):** parameter recovery
deteriorates substantially faster than repricing quality as observation noise
increases. The design must be able to refute this.

## 2. Frozen models and population

- Model 1 checkpoints: seeds 11/22/33 (`checkpoints/r2_primary_comparison/model1_seed*/`)
- Model 2 primary cohort: uniform P100 seeds 11/22/33 (`model2_seed*/`, provenance cuda + git_sha `2b5d41c…`)
- Excluded everywhere: `model2_seed11_local_cpu_replication/`
  (`MODEL2_LOCAL_SEED11_EXECUTION_ENVIRONMENT_REPLICATION`)
- No retraining, fine-tuning, new seeds, checkpoint reselection, or hyperparameter search.
- Evaluation population: **test split only**, exactly the 1,250 canonical test
  surfaces, identical ordering to the primary evaluation. Train/validation
  surfaces are never read by this study.

## 3. Observation-noise semantics (frozen)

Inherited functional form from the project's documented donor contract
(`src/g2_r2r3/noise.py`, G2 R2-vs-R3 study), re-derived in a new module with a
new base seed so realizations are independent of all historical studies:

- **Type:** multiplicative, `observed = clean_price * (1 + level * z)`,
  `z ~ standard_normal`, one draw per quote slot via
  `numpy.random.default_rng(sha256_slot_seed)`.
- **Key:** `"{base}|{surface_id}|rank{expiry_rank}|k{moneyness:+.2f}|{option_type}|level{level:.4f}"`
  with `base = 20260825` (new, dedicated; donor used 20260824).
  Seed = first 8 bytes of SHA-256, big-endian, masked to 63 bits.
- **Calls/puts independent:** option_type is part of every key.
- **Realization sharing:** keys contain no method/model/seed component — all
  three methods and all neural seeds observe byte-identical noisy cohorts.
- **Unchanged fields:** spot, rates, carries, maturities, mask, slot_keys,
  canonical truth parameters, all source metadata. Truth is known and fixed.
- **Levels (exact):** `0.0, 0.001, 0.0025, 0.005, 0.01` (0%, 0.10%, 0.25%,
  0.50%, 1.00%). Low levels expose degradation onset; 0.50%/1.00% anchor to
  historically validated market-scale levels from the G2 studies.
- **Floor policy:** positivity asserted; a negative draw (requires z < −1/level;
  effectively impossible) triggers deterministic counter-suffixed redraws
  (`#r{counter}`, cap 64) then raises. No silent clamping of prices or factors.
- **Static-arbitrage policy:** raw independent noise RETAINED — never clipped,
  smoothed, or projected onto a no-arbitrage manifold. Put-call-parity breaks
  and vertical-spread violations are counted per surface as separate
  diagnostics only.
- **Namespace:** derived records carry noise fields under `observation_noise`;
  the generator's pre-existing `user_metadata.noise_level` (0.0 on every clean
  surface) is never modified or reused.

## 4. Traditional-calibration compute design (frozen before results)

Strategy B-hybrid:

- Neural methods: **full 1,250-surface test split at every level**, both models,
  all seeds → full-population neural robustness curves.
- Traditional calibration: **predeclared stratified subset of N=250** test
  surfaces, selected ONCE by a pure function of clean truths and ids:
  - strata: `v0_total = v0_slow + v0_fast` terciles × `kappa_slow` terciles
    (3×3 cells, edges computed on test-split truths);
  - proportional largest-remainder allocation to exactly 250;
  - within-cell ordering by ascending SHA-256 hex of surface_id;
  - no RNG, no outcome participation (`selection_uses_outcomes=false`);
  - committed selection artifact:
    `evidence/r2_noise_robustness/traditional_subset_ids.json`.
- Traditional runs ALL levels including 0% (subset reproduction gate) with
  EXACTLY the frozen primary settings (3 starts, start_seed 42, max_nfev 300,
  ftol/xtol/gtol 1e-10, diff_step 2e-05, node_count 64, production pricer,
  same bounds, same representative rule). Cost ≈ 130 CPU-hours (~13 h at 10 workers).
- **Population discipline:** three-way method comparisons are reported on the
  subset population ONLY and labeled as such; neural-only robustness curves use
  the full test set; paired degradation deltas always compare identical surface
  sets. Full-set-neural vs subset-traditional comparisons as if identical
  populations are FORBIDDEN.
- Subset sufficiency: N=250 gives ±~6.2% two-sigma precision around 90%-scale
  success rates and resolves degradation deltas an order of magnitude larger
  than sampling error.

## 5. Metric contract

Reuse `src/r2_primary/evaluation.py` families unchanged wherever possible.

- Parameter recovery: range-scaled RMSE, standardized RMSE, per-parameter
  errors, v0_total/theta_total MAE, half-life MAEs, factor-swap confusion.
- **Two distinct repricing quantities (never conflated):**
  1. *Fit-to-noisy-observation*: errors vs noisy observed prices (what the
     calibration objective sees);
  2. *Clean-latent repricing*: errors vs original clean prices (parameter-
     fidelity-facing distance to the true surface).
  A model may fit noise extremely well while moving farther from truth.
- Constraint validity: existing full DH structural contract.
- Stability: neural cross-seed dispersion per level; traditional multi-start
  dispersion/disagreement per level.
- Identifiability-aware: tolerance-conditioned rates for BOTH repricing
  quantities; parameter recovery conditioned on each; fraction of surfaces
  where repricing stays acceptable while parameter recovery degrades
  materially vs the 0% level.
- Robustness degradation: paired deltas `metric(level) − metric(0%)` on
  identical surfaces, ratios where defined, failure-rate curves at frozen
  thresholds (repricing nRMSE ≤1e-4/≤1e-3; param RMSE ≤0.10/≤0.25).

## 6. 0%-noise reproduction gate (mandatory)

Before ANY positive-noise result is interpreted:

- Neural 0% through THIS pipeline must reproduce the merged canonical primary
  headline metrics exactly (same code path/checkpoints/seeds; bitwise-equal CSV
  values required).
- Traditional subset 0% rows must match canonical journal rows for the same
  ids to documented floating-point equality (expected bitwise).
- On any mismatch: STOP, investigate, never overwrite canonical evidence.

## 7. Cohort identity and storage

- Storage: `data/r2_noise_robustness/levels/{level_label}/noisy_surfaces.jsonl`
  + `MANIFEST.json` (per-file SHA-256, counts, base seed, provenance); tracked
  in git; append-only immutable derived artifacts; byte-for-byte replayable
  from the clean dataset plus the frozen module.
- Every record preserves: source surface_id, true parameters, clean prices,
  noisy prices, observation-noise level/base-seed/per-slot realization ids,
  unmodified R2 metadata, derivation module+config SHAs.

## 8. Execution order (post-freeze)

freeze (this commit) → 0% reproduction gate → deterministic cohort generation →
neural evaluations (all levels) → traditional subset runs (all levels) →
aggregation + degradation curves → results reconciliation.

## 9. Claim discipline

The hypothesis may be supported, refuted, or bounded. Repricing quality is
never equated with parameter recovery; no unique-recovery claim is permitted;
practical non-identifiability remains the retained prior finding unless the
new evidence genuinely contradicts it. Clean-primary conclusions are not
re-litigated here. Model 2 is never described as PDE-informed.
