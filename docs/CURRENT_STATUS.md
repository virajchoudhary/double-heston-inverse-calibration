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
| G2 candidate protocol | Self-governed R2 vs R3, predeclared |
| Final G2 representation | Not frozen |
| G2 | NOT_PASSED |
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

## Active G2 protocol

G2 is now a bounded representation-selection milestone, not an open-ended uniqueness search.

Candidates:

- **R2:** two eligible listed expiry ranks x central five log-moneyness targets x calls/puts, with actual maturity conditioning;
- **R3:** three eligible listed expiry ranks x central five x calls/puts, with explicit masking for unsupported/unusable slots and actual maturity conditioning.

The unchanged 108-grid is excluded.

The experiment uses the five existing NTPC development dates for market support and a deterministic synthetic truth panel for identifiability. It runs identical 12-start calibration at 0%, 0.5%, 1%, and 2% noise and reports market support, local sensitivity, global dispersion/clusters, parameter recovery, repricing, boundary pressure, stability, and runtime.

If practical non-identifiability remains after representation selection, it is retained as a research result rather than used as a reason to keep changing the representation indefinitely.

## Current gate summary

```text
PRODUCTION_DH_PRICER = VALIDATED_AND_FROZEN
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_PROTOCOL = SELF_GOVERNED_R2_VS_R3_PREDECLARED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
GLOBAL_AMBIGUITY = ESTABLISHED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

## Exact next action

`EXECUTE SELF-GOVERNED G2 REPRESENTATION SELECTION`

Do not generate the final 10k or start research ANN/Model-2 training until the R2/R3 experiment has selected and frozen the final representation.
