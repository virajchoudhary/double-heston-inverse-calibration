# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

The 22 August 2026 three-node diagnostic swarm is complete. Its durable conclusions are recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md).

The previous mentor-approval blocker has been retired. The predeclared [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md) **has been executed and sealed** (22 August 2026): the frozen stopping rule froze **R2**, with practical non-identifiability retained as a research finding. Full results: [G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md](G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md).

## Immediate sequence

### 1. Quarantine non-canonical Archive-2 training behavior — COMPLETE

Issue #20 is closed. PR #29 (merged, canonical main `75ad4d0`) added the
fail-closed real-market weight-update quarantine
(`src/dheston/real_market_policy.py`): `real_finetune` / `--continuous` can
no longer silently update weights on real data in normal repository usage,
historical smoke evidence is preserved, canonical synthetic-only paths are
unchanged, and Archive-2's current PDE loss is not imported.

### 2. Formalize the frozen R2 representation interface — COMPLETE (merged PR #28)

G2 selected **R2** (ranked two-expiry central-five calls/puts, **20 NOMINAL
slots** with explicit mask/missingness for unsupported or unusable real-market
observations — never impute a missing real quote with a model price;
spot-normalized; actual maturity conditioning; existing rate/carry
conditioning; the synthetic G2 panel is complete by construction). The
canonical interface is implemented as `src/r2_representation/` with the
contract document
[R2_REPRESENTATION_CONTRACT.md](R2_REPRESENTATION_CONTRACT.md): one
deterministic tested slot order, explicit mask semantics, a synthetic
constructor over the unchanged production pricer, a real-market constructor
reusing the sealed official-NSE quote-selection audit, versioned JSON
serialization, structural rejection of legacy-108 and rejected-R3 data, and
a full focused test suite. The canonical parameter order, pricer,
constraints, and target semantics are unchanged.

### 3. Generate final synthetic truth data — COMPLETE (generated, validated, full-replay verified)

Only after representation freeze (done) and interface formalization (done,
`src/r2_representation/`):

1. final sampling/generation contract frozen (`configs/r2_synthetic_generation_FINAL.yaml`, commit `53aed7b`);
2. contracted 240-surface development pilot executed and validated with a `VERIFIED_IDENTICAL` deterministic replay (0 pricing failures);
3. fixed final 15k/5k candidate pools verified sufficient (12,217 ≥ 8,334 interior; 3,371 ≥ 1,666 wide) without pricing the final core — merged PR #28 / Issue #27 closed;
4. the frozen 10,000-row parameter panel was sealed as
   `evidence/final_r2_candidate_pool_readiness_20260822/final_parameter_panel.csv`
   (FINAL_PARAMETER_PANEL_ONLY / SURFACES_NOT_GENERATED / NOT_YET_TRAINING_DATA);
5. the separate explicit final-generation gate (`src/r2_final_generation.py`) authorized
   final generation via committed, remote-verified authorization markers
   (`evidence/R2_FINAL_10K_GENERATION_AUTHORIZED.txt`; authorization SHAs recorded on
   Issue #30: v1 `2cbf0fc`, v2 `bfbed9d`, v3 `b13caf0`); and
6. the final dataset was generated from the exact frozen panel (10,000 surfaces,
   0 pricing failures, 0 retained failures, noise 0, no replacement/refill, no
   real-market inputs, no training), validated against the full frozen contract,
   and verified byte-identical by a full deterministic re-pricing replay of all
   10,000 surfaces. Canonical output: `data/final_r2_clean_10000/`
   (dataset status `FINAL_R2_CLEAN_10000_RESEARCH_SYNTHETIC_TRUTH_DATASET_FROZEN_BEFORE_MODEL_TRAINING`;
   surfaces.jsonl SHA-256 `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`).

### 4. Run the primary comparison

Primary methods:

- traditional numerical calibration;
- Model 1 — ordinary ANN inverse model;
- Model 2 — canonical constraint + differentiable-repricing informed inverse model.

Hold constant:

- frozen representation;
- canonical ten-parameter order;
- synthetic truth splits;
- pricing/evaluation contract;
- real-market freeze policy; and
- metric families.

Report parameter recovery separately from repricing and include structural validity, multi-seed stability, tolerance/equivalence-class recovery, noise robustness, and runtime. No method wins on repricing RMSE alone.

### 5. Frozen real-market evaluation

Only after the primary models are frozen, evaluate them chronologically on untouched real-market dates reserved under G8. Do not update primary network weights with those observations.

### 6. Optional Model 3

A genuine PDE-informed model is not a blocker for the primary study.

If added later, its scientific question should be whether network-side physics regularization improves structural validity/stability, not whether a PDE term creates unique identification from an observationally ambiguous surface.

## Current state

```text
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_PROTOCOL = SELF_GOVERNED_R2_VS_R3_EXECUTED_AND_SEALED
G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING
R2_GENERATION_CONTRACT = FROZEN_AND_PILOT_VALIDATED_MERGED_PR28
FINAL_CANDIDATE_POOL_READINESS = VERIFIED_NO_PRICING_MERGED_PR28
ARCHIVE2_REAL_MARKET_WEIGHT_UPDATES = QUARANTINED_MERGED_PR29
FINAL_PARAMETER_PANEL = SEALED_10000_ROWS_CONSUMED_BY_FINAL_GENERATION
FINAL_10K = GENERATED_AND_VALIDATED_FULL_REPLAY_VERIFIED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
NOISE_COHORT = NOT_GENERATED
BOUNDARY_CHALLENGE = NOT_GENERATED
OOD = NOT_GENERATED
G8 = NOT_STARTED
```

The exact next scientific action is:

`PRIMARY MODEL COMPARISON ON THE FROZEN FINAL 10K DATASET (TRADITIONAL CALIBRATION, MODEL 1 ANN, MODEL 2 CONSTRAINT+REPRICING-INFORMED) — SEPARATELY AUTHORIZED, NOT STARTED`
