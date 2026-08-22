# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

The 22 August 2026 three-node diagnostic swarm is complete. Its durable conclusions are recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md).

The previous mentor-approval blocker has been retired. The project now proceeds under the predeclared [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md).

## Immediate sequence

### 1. Quarantine non-canonical Archive-2 training behavior

Implement Issue #20 as a focused code/test PR:

- prevent `real_finetune` / `--continuous` from silently updating weights on real data in normal repository usage;
- preserve historical smoke evidence;
- leave canonical synthetic-only paths unchanged;
- do not import Archive-2's current PDE loss.

This is policy hardening and may proceed in parallel with G2 implementation preparation.

### 2. Execute the self-governed G2 representation study

Run exactly the frozen protocol in [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md).

The bounded comparison is:

- **R2:** ranked two-expiry central-five calls/puts with actual maturity conditioning;
- **R3:** ranked three-expiry masked central-five calls/puts with actual maturity conditioning and explicit masking.

The unchanged 108-grid is not a candidate.

Use all five existing NTPC development dates for market-support evidence and the fixed synthetic truth panel for identifiability. Run identical multi-start diagnostics at 0%, 0.5%, 1%, and 2% noise.

Do not alter thresholds, candidate definitions, truth selection, optimizer, constraints, or stopping rules after results are seen.

### 3. Freeze the representation exactly once

Apply the protocol's stopping rule:

- freeze R3 if it is market-supported under masking and shows the predeclared strong/partial information improvement without violating the existing holdout guardrail;
- otherwise freeze R2 as the simpler supported representation;
- if realistic-noise practical non-identifiability remains, retain it as a research finding rather than reopening unlimited representation search;
- reopen G2 design only if both candidates fail the hard market-construction requirements.

Valid outcomes:

```text
G2 = PASSED_REPRESENTATION_FROZEN_IDENTIFIABILITY_ACCEPTABLE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
G2 = FAILED_MARKET_CONSTRUCTION_REQUIREMENTS
```

### 4. Formalize the selected representation interface

After G2 chooses R2 or R3, make the production dataset/model interface explicitly represent that frozen contract.

Keep the canonical parameter order, pricer, constraints, and target semantics unchanged.

### 5. Generate final synthetic truth data

Only after representation freeze:

1. freeze the final sampling + representation manifest;
2. generate the final synthetic dataset;
3. run validity/integrity checks;
4. freeze train/validation/test splits; and
5. preserve normal, noise, challenge, and OOD cohorts separately.

### 6. Run the primary comparison

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

### 7. Frozen real-market evaluation

Only after the primary models are frozen, evaluate them chronologically on untouched real-market dates reserved under G8. Do not update primary network weights with those observations.

### 8. Optional Model 3

A genuine PDE-informed model is not a blocker for the primary study.

If added later, its scientific question should be whether network-side physics regularization improves structural validity/stability, not whether a PDE term creates unique identification from an observationally ambiguous surface.

## Current state

```text
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
G2_PROTOCOL = SELF_GOVERNED_R2_VS_R3_PREDECLARED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

The exact next scientific action is:

`EXECUTE SELF-GOVERNED G2 REPRESENTATION SELECTION`
