# Current Project Status

Status date: 22 August 2026

> Canonical research control is maintained in [RESEARCH_CONTROL_AND_CURRENT_STATUS.md](RESEARCH_CONTROL_AND_CURRENT_STATUS.md). The durable 22 August three-node audit conclusions are recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md). The active bounded representation-selection experiment is defined in [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md). Earlier detailed status remains preserved in Git history and the experiment-specific documents under `docs/`.

## Approved research objective

Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

## Current high-level state

The canonical Double Heston pricing/calibration foundation remains validated. The main project bottleneck is representation / information / practical identifiability, not pricing-engine correctness.

The 22 August three-node diagnostic swarm completed architecture, identifiability/calibration, and PDE/physics audits without changing the scientific gates, generating the final dataset, or running research-scale ANN/PINN training.

The project is no longer waiting for external mentor approval to continue. G2 is now governed by a frozen, self-declared R2-vs-R3 protocol. The project team will run that experiment once, apply its predeclared stopping rule, freeze a representation, and carry any remaining practical non-identifiability forward as an explicit research finding.

### Current status table

| Component | Status |
|---|---|
| Fixed ten-parameter contract | Complete |
| Canonical production pricer | Validated and frozen |
| Differentiable Torch mirror | Implemented and independently cross-checked |
| ANN infrastructure | Implemented |
| Constraint + repricing inverse infrastructure | Implemented, not research-trained |
| Genuine PDE-informed inverse model | Not implemented/validated as a research milestone |
| Stage A NSE market-support screen | Complete |
| Selected primary sector set | NTPC, CIPLA, INFY, HDFCBANK |
| G2 market-supported geometry | Established |
| Current 108-input grid | Rejected as final unchanged representation |
| G2 candidate protocol | Self-governed R2 vs R3, predeclared — executed and sealed 22 Aug 2026 |
| Final G2 representation | **Frozen: R2** (ranked two-expiry central-five, 20 nominal slots + explicit real-market masking) |
| Canonical R2 representation interface | Formalized: `src/r2_representation/` ([R2_REPRESENTATION_CONTRACT.md](R2_REPRESENTATION_CONTRACT.md)) |
| G2 | **PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY** |
| Global ten-parameter ambiguity | Established |
| Full-108 overnight identifiability diagnostic | Completed; locally full-rank can still be practically non-identifiable at noise scale |
| Final 10,000-surface research dataset | Not generated |
| ANN research training | Not started |
| Model-2 research training | Not started |
| Frozen real-market evaluation | Not started |
| Archive-2 PDE loss | Do not adopt; implementation defect independently reproduced |
| Archive-2 real-market fine-tuning | Non-canonical; quarantine/remove from canonical entry points |

## Canonical architecture

Keep the existing canonical stack as the source of truth:

- `src/constants.py` parameter contract;
- frozen production Double Heston pricer;
- differentiable Torch mirror;
- canonical hard-by-construction constraints;
- synthetic-only primary training;
- validation-gated model selection; and
- frozen chronological real-market evaluation.

Archive-2 / `src/dheston` is experimental donor code only. Selected patterns may be adapted behind explicit interfaces, but its parameter order, constraint semantics, PDE loss, and real-market weight-update path are not canonical.

Cross-stack positional parameter passing is forbidden. Use only a verified named adapter if interoperability is required.

## Neural / physics terminology

Use:

```text
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

The current canonical neural inverse path is **constraint-informed + differentiable-repricing-informed**, not a genuine PDE-informed PINN.

Recommended comparison taxonomy:

- Model 1 — ordinary ANN inverse model;
- Model 2 — canonical constraint + differentiable-repricing informed inverse model;
- Model 3 — optional future genuine network-side PDE-informed model.

Model 3 is not a blocker for the primary study.

## Identifiability status

The full provisional 108-quote grid can be locally full practical rank while still becoming practically non-identifiable at realistic market-noise tolerance.

Key overnight conclusion:

```text
STRICT_PRECISION_IDENTIFICATION = CAN_BE_GOOD
MARKET_TOLERANCE_IDENTIFICATION = PRACTICALLY_NON_IDENTIFIABLE / ILL_CONDITIONED
```

Therefore:

- price fit does not imply parameter recovery;
- parameter recovery must be reported separately from repricing;
- multi-start/multi-seed dispersion matters;
- noise robustness matters;
- equivalence-class / tolerance-conditioned recovery should be reported; and
- physics constraints may regularize but must not be claimed to create unique identification.

Factor-swap symmetry is an exact/near-machine-precision degeneracy; the declared `kappa_slow < kappa_fast` ordering is the tie-breaking convention that excludes the swapped twin.

## PDE/physics status

Node A and Node C independently reproduced the same Archive-2 PDE-loss implementation bug: variance-factor derivatives are silently zeroed by autograd slice-view handling, so the implemented residual is not the intended Double Heston PDE.

Do not import the current Archive-2 PDE loss.

A correctly wired residual on an already accurate model pricer is approximately machine-zero and is not an independent source of parameter-identification information. A future Model 3 would require a different network-side formulation and a research question centered on regularization/structural validity rather than assumed uniqueness.

## G2 outcome (22 August 2026)

The predeclared self-governed R2-vs-R3 protocol was executed once and sealed
([G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md](G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md)).
The frozen stopping rule selected **R2** (rule 2: R3 showed no strong/partial
practical-information improvement at any realistic noise level, despite better
Jacobian conditioning and a strong clean-data improvement that noise
destroyed). R2 is 20 NOMINAL slots with explicit mask/missingness for
unsupported or unusable real-market observations (never impute a missing real
quote); the synthetic G2 panel is complete by construction. Both candidates remain practically non-identifiable at realistic
noise, retained as a central research finding. R3's third expiry rank
contributed zero usable central-five slots on all five development dates
(far-month NTPC chains inactive under the existing contract).

## Current gate summary

```text
PRODUCTION_DH_PRICER = VALIDATED_AND_FROZEN
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_PROTOCOL = SELF_GOVERNED_R2_VS_R3_EXECUTED_AND_SEALED
G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING
GLOBAL_AMBIGUITY = ESTABLISHED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

## Exact next action

`REGENERATE / REVALIDATE THE FINAL SYNTHETIC SURFACE CONTRACT ON THE FROZEN R2 INTERFACE`

The frozen R2 representation interface is formalized as the canonical
post-G2 representation contract (`src/r2_representation/`,
[R2_REPRESENTATION_CONTRACT.md](R2_REPRESENTATION_CONTRACT.md)): 20 nominal
slots in one deterministic tested order, explicit mask semantics (never
impute a missing real quote), synthetic construction through the unchanged
production pricer, real-market construction through the sealed official-NSE
audit contract, and a versioned serialization schema; legacy 108-feature and
rejected-R3 data are structurally rejected. The final 10k generation,
ANN/Model-2 research training, final G8 date selection, and frozen
real-market evaluation remain separately controlled milestones that have
not started.
