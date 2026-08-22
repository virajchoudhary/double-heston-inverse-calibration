# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

The 22 August 2026 three-node diagnostic swarm is complete. Its durable conclusions are recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md).

The previous mentor-approval blocker has been retired. The predeclared [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md) **has been executed and sealed** (22 August 2026): the frozen stopping rule froze **R2**, with practical non-identifiability retained as a research finding. Full results: [G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md](G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md).

## Immediate sequence

### 1. Quarantine non-canonical Archive-2 training behavior

Implement Issue #20 as a focused code/test PR:

- prevent `real_finetune` / `--continuous` from silently updating weights on real data in normal repository usage;
- preserve historical smoke evidence;
- leave canonical synthetic-only paths unchanged;
- do not import Archive-2's current PDE loss.

This is policy hardening and may proceed in parallel with the next research milestone.

### 2. Formalize the frozen R2 representation interface — EXACT NEXT ACTION

G2 selected **R2** (ranked two-expiry central-five calls/puts, **20 NOMINAL
slots** with explicit mask/missingness for unsupported or unusable real-market
observations — never impute a missing real quote with a model price;
spot-normalized; actual maturity conditioning; existing rate/carry
conditioning; the synthetic G2 panel is complete by construction). Make the
production dataset/model interface explicitly represent that frozen contract,
including the mask channel for real surfaces.

Keep the canonical parameter order, pricer, constraints, and target semantics unchanged. Because R2 carries actual maturities and per-rank rate/carry conditioning, the interface must supply those explicitly rather than assuming a fixed grid.

### 3. Generate final synthetic truth data

Only after representation freeze:

1. freeze the final sampling + representation manifest;
2. generate the final synthetic dataset;
3. run validity/integrity checks;
4. freeze train/validation/test splits; and
5. preserve normal, noise, challenge, and OOD cohorts separately.

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
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

The exact next scientific action is:

`FORMALIZE THE FROZEN R2 REPRESENTATION INTERFACE`
