# Overnight Swarm Decision Record — 22 August 2026

Status: CONSOLIDATED FOR HUMAN REVIEW

This document records the durable project conclusions from the three-node overnight diagnostic run on 22 August 2026. It does not pass G2, freeze a final representation, authorize the final 10k dataset, authorize ANN/PINN research training, or alter the mentor-controlled research contract.

## Provenance

Common starting point:

`642702e6706a3d17b3031619f35bda39bc144483`

Completed node states:

- Node A — coordinator / architecture: `401bcd7dbe4ae71cf1caba0f29ddfa16f9774c83`
- Node B — identifiability / calibration: `1dff8a33c82093db03d4845c5011a1e58913fbf7`
- Node C — PDE / physics: `c30dcef1ab6a8dd03f14fc2edb9cb430cdd4016d`

Stable archive refs:

- `archive/overnight-20260822-node-a`
- `archive/overnight-20260822-node-b`
- `archive/overnight-20260822-node-c`

Coordination history: GitHub Issue #18, `[OVERNIGHT 2026-08-22] Double Heston research swarm`.

The raw evidence remains on the node branches under `.ai-research/overnight/2026-08-22/`.

## 1. Canonical architecture decision

The existing canonical stack remains the project source of truth:

1. canonical parameter contract in `src/constants.py`;
2. frozen production Double Heston Gauss-Laguerre pricer;
3. independently checked differentiable Torch mirror;
4. canonical hard-by-construction constraint map;
5. synthetic-only primary ANN/inverse-model training;
6. validation-gated checkpointing; and
7. frozen, chronological, zero-leakage real-market evaluation after model freeze.

The imported `src/dheston` / Archive-2 implementation is not a second canonical stack. It is an experimental donor of selected patterns only.

Potentially reusable Archive-2 ideas, after review and adaptation:

- variable-length / masked surface representation pattern;
- chronological zero-leakage real-market splitting/evaluation pattern;
- independent COS pricer as a numerical cross-check.

Do not use positional tensors to interchange parameters between the two stacks. The parameter conventions differ by both factor placement and within-factor order. Cross-stack interoperability must use an explicitly verified adapter.

## 2. Physics / PINN terminology

The current canonical neural inverse path is accurately described as:

`constraint-informed + differentiable-repricing-informed inverse network`

It is not presently a genuine PDE-informed PINN.

Use two independent status axes:

```text
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

Recommended experimental taxonomy:

- Model 1: ordinary ANN inverse model;
- Model 2: canonical constraint + differentiable-repricing informed inverse model;
- Model 3: genuine network-side PDE-informed model, only if separately approved and correctly constructed.

Model 3 is not authorized by this record.

## 3. Archive-2 PDE-loss defect

Node A and Node C independently reproduced a critical implementation defect in Archive-2's PDE loss.

The implementation differentiates prices with respect to post-hoc parameter slice views that are not ancestors of the executed autograd graph. `autograd.grad` therefore returns `None`, and the `_safe_grad` fallback silently replaces those derivatives with zero.

Result: all variance-factor derivative terms are dropped. The implemented residual is not the intended Double Heston PDE residual.

Disposition:

`DO NOT ADOPT ARCHIVE-2 PDE LOSS INTO THE CANONICAL PATH.`

Even with correct derivative wiring, a PDE residual evaluated on an already accurate model pricer is approximately machine-zero and does not create independent information for inverse parameter identification. Physics may regularize or enforce structural validity; it must not be claimed to manufacture unique identification where the observed surface is ambiguous.

## 4. Identifiability conclusion

Node B established the key calibration result on the full provisional 108-quote grid:

- the local Jacobian is full practical rank in representative full-grid cases;
- strict-precision clean identification can be good;
- realistic small price noise causes severe parameter instability while repricing remains near the noise floor;
- multi-start solutions can have materially different parameter vectors with nearly indistinguishable surfaces;
- increased optimizer budget, alternative starts, and weighting changes do not remove the governing failure;
- the objective landscape contains compensated flat directions; and
- factor-swap symmetry is an exact/near-machine-precision degeneracy, broken by the declared slow/fast ordering convention.

The defensible classification is:

`ILL_CONDITIONED_AT_NOISE_SCALE + PRACTICALLY_NON_IDENTIFIABLE_AT_MARKET_TOLERANCE`,

with case-dependent optimizer-basin sensitivity on clean data.

Therefore future evaluation must report parameter recovery separately from repricing, and parameter-recovery claims must be tolerance/equivalence-class conditioned rather than framed as universal unique recovery.

## 5. Real-market training policy

Archive-2 exposes `real_finetune` / `--continuous` behavior that updates neural-network weights on real market observations. That conflicts with the canonical primary research protocol, which reserves real observations for frozen-model evaluation rather than weight updates.

Canonical disposition:

`REMOVE OR HARD-QUARANTINE FROM CANONICAL ENTRY POINTS.`

Historical smoke evidence remains evidence and must not be deleted.

A non-primary real-data adaptation ablation may be considered only after explicit human/mentor approval and must never be confused with the primary experiment.

## 6. Constraint policy

The canonical structural constraints remain authoritative:

- positivity;
- `kappa_slow < kappa_fast`;
- factorwise strict Feller conditions;
- individual correlations in `(-1, 1)`; and
- the declared joint correlation-disk constraint.

Do not silently clamp neural predictions to reviewed synthetic sampling boxes merely to improve recovery metrics. Structural validity and membership in a reviewed training/sampling region are different concepts and must be reported separately.

## 7. G2 and representation status

The overnight run did not alter any gate.

```text
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_RESEARCH_TRAINING = NOT_STARTED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

The current fixed-size model code already derives input size from data, which reduces architectural coupling to 108 features. Before final synthetic generation, formalize a representation interface so a mentor-approved final surface contract can be substituted without redesigning the model stack.

Archive-2's variable-length representation may inform that interface, but it is not adopted wholesale.

## 8. Immediate project sequence

Do the following in order:

1. preserve the overnight evidence and archive refs;
2. reconcile canonical documentation and terminology;
3. quarantine Archive-2 real-market weight-update paths from canonical entry points;
4. obtain the existing mentor decision on the G2 safeguard and bounded richer NTPC information study;
5. execute only the approved bounded information study if approved;
6. freeze the final representation only after the G2 decision criteria are satisfied;
7. formalize the representation interface;
8. generate and validate the final synthetic dataset only after representation freeze;
9. run the fair traditional-calibration vs Model-1 ANN vs Model-2 comparison under identical splits and metrics;
10. perform frozen, untouched real-market evaluation; and
11. consider Model 3 only as a separately justified physics-regularization extension.

## 9. Required evaluation dimensions

Future comparisons must report at least:

- parameter recovery;
- repricing error;
- structural validity;
- multi-start / multi-seed stability;
- tolerance-conditioned or equivalence-class recovery;
- noise robustness; and
- runtime.

No model may be declared superior from repricing RMSE alone.

## 10. Decisions still requiring mentor or explicit research approval

The following remain approval-sensitive:

1. the formal G2 safeguard and bounded richer NTPC information study;
2. the final representation / feature contract;
3. any change to reviewed parameter sampling policy;
4. whether a genuine Model-3 PDE-informed experiment enters the research plan; and
5. any real-market weight-update ablation.

## 11. Preservation rule

Do not delete the overnight branches, archive refs, Issue #18 history, failed experiments, negative findings, or defect reproductions merely because later code supersedes them. They form the provenance trail for the research decisions above.
