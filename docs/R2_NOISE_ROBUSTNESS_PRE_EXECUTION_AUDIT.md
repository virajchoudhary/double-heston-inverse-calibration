# R2 Noise-Robustness Pre-Execution Audit

Status: COMPLETE_BEFORE_FREEZE · Date: 2026-08-24 · Issue: #34
Branch: `research/r2-noise-robustness` (from canonical main `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`)
Purpose: document every existing noise-related code path, prior evidence item,
and reuse decision BEFORE the protocol freeze and before any noisy result.

## 1. Repository noise/perturbation inventory

| Location | Nature | Disposition for this study |
|---|---|---|
| `src/g2_r2r3/noise.py` | Donor keyed multiplicative noise: `clean*(1+level*z)`, per-slot SHA-256 seeds (`truth_id, expiry_rank, moneyness, option_type, level`), base seed 20260824, negative price raises; bit-replayable | **FUNCTIONAL-FORM LINEAGE.** Re-derived in new module `src/r2_noise/perturbation.py` with NEW base seed 20260825 and R2 surface-id keys; donor module itself NOT imported at runtime |
| `src/g2_r2r3/frozen.py` | Donor frozen constants: `NOISE_BASE_SEED=20260824`, `NOISE_LEVELS=(0.0, 0.005, 0.01, 0.02)`, `NON_IDENTIFIABILITY_NOISE_LEVEL=0.005` | Historical reference only; this study freezes its own levels `(0.0, 0.001, 0.0025, 0.005, 0.01)` and seed 20260825 |
| `src/g2_r2r3/{calibration,decision,geometry,starts}.py`, G2 scripts/tests | G2 R2-vs-R3 harness (representation comparison, ambiguity/basin analyses) | DONOR-ONLY context; no code imported; their conclusions are prior evidence, not assumptions of this protocol |
| `src/synthetic_dataset.py`, `src/dheston/data/synthetic.py`, `src/dataset.py` | Older generation-era synthetic/noise utilities predating the R2 contract | HISTORICAL/donor-only; not used; legacy-grid era assumptions (108-dim etc.) explicitly excluded by the R2 contract |
| `src/r2_synthetic_generation.py`, `src/r2_final_generation.py` | Clean final-dataset generator; carries a *generation-conditioning* field `user_metadata.noise_level` (=0.0 on every clean record — verified on a 500-record sample) | Not reused here; derived records use a separate `observation_noise` namespace to avoid semantic collision |
| `tests/test_g2_*`, `tests/test_reviewed_sampling*`, `tests/test_core_dataset_readiness.py` | Tests exercising donor noise/harness paths | Not modified; not executed as part of this study's gates |

## 2. Prior noise evidence (context, not presumption)

- `docs/G2_INFORMATION_REMEDIATION.md`: at 0.5% observation noise both 2-expiry
  and 3-expiry representations showed optimizer success collapse (6-7/12) with
  recovery pass **0/12** while median price RMSE stayed at noise scale
  (~3.5e-04) — repricing fit survived, parameter recovery did not.
- `docs/G2_GLOBAL_AMBIGUITY_ANALYSIS.md`: median parameter RMSE moved from
  ~0.203 clean → ~0.332 @0.5% → ~0.371 @1.0% while every noisy solution hit a
  declared boundary; noise treated strictly as a stability probe.
- `docs/R2_REPRESENTATION_CONTRACT.md`: NO_MATERIAL_IMPROVEMENT verdicts at
  0.5%/1%/2%; practical non-identifiability retained at market noise.

These are PRIOR EVIDENCE on different (smaller, older-generation) populations.
This study must be capable of refuting the same-direction hypothesis on the
primary methods; nothing above is baked into metric definitions or thresholds.

## 3. Primary-comparison artifacts audited for immutability

- Canonical main `72ad8e1`; primary protocol config SHA-256
  `33ca0f76…70781`; clean dataset SHA-256 `148b579a…f1f6`; calibration journal
  SHA-256 `fe73e696…0ce51` — all re-verified on this branch before freezing.
- Checkpoints under `checkpoints/r2_primary_comparison/` are read-only inputs;
  the CPU seed-11 replication directory is excluded from every metric.
- The clean 0%-noise primary results are the immutable baseline; the 0%
  reproduction gate (protocol §6) reconciles THIS pipeline against them
  before any positive-noise interpretation.

## 4. Design decisions recorded

1. Multiplicative keyed per-slot noise (donor functional form), independent
   calls/puts, shared realization across all methods/seeds (keys exclude
   method/model/seed).
2. Levels {0, 0.10%, 0.25%, 0.50%, 1.00%}: fine onset resolution + historical
   anchors at 0.50%/1.00%.
3. Retain-and-flag arbitrage policy: no clipping/projection; parity/spread
   violation counts as diagnostics only.
4. Traditional compute: predeclared stratified N=250 subset (v0_total ×
   kappa_slow terciles, hash-ordered within cells, largest-remainder
   allocation), selected purely from clean truths/ids and committed WITH this
   freeze (`evidence/r2_noise_robustness/traditional_subset_ids.json`);
   neural full-test at every level; subset-only three-way comparisons labeled.
5. Two separated repricing quantities (fit-to-noisy vs clean-latent) plus
   paired degradation-from-clean deltas and failure-rate curves.
6. 0%-reproduction gate mandatory before any positive-noise interpretation.

## 5. Leakage statement

Selection uses only known synthetic truths and surface ids from the clean
dataset BEFORE any noisy outcome exists; no model/calibration output
participates. Noisy observations are evaluation inputs only. Nothing in this
study may alter primary models, architecture, losses, seeds, the R2
representation, or clean-test conclusions; the clean primary comparison
remains independently reproducible from main.
