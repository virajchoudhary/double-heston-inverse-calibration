# Research Control and Current Status

Status date: 22 August 2026

This is the canonical repository control/status document for the next scientific milestone. It records fixed contracts, validated evidence, and the self-governed G2 protocol that now controls representation selection.

The detailed 13 August 2026 state remains preserved in Git history and in the experiment-specific documents linked throughout `docs/`. The 22 August three-node diagnostic consolidation is recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md). The bounded representation-selection protocol is [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md).

## 1. Source-of-truth hierarchy

1. **Scientific objective and fixed contracts.** The project objective, canonical ten-parameter model, pricing engine, structural constraints, synthetic-truth recovery policy, and frozen real-market evaluation policy remain fixed unless a documented research decision changes them.
2. **Predeclared experiment protocols.** New experiments must be specified before their outcome is observed; thresholds and stopping rules may not be changed post hoc to manufacture a preferred result.
3. **Implementation truth — this repository.** Code, configurations, manifests, persisted evidence, tests, Git history, hashes, and merge history determine what was actually implemented and validated.

The project team owns technical research design. External mentor approval is not a prerequisite for routine continuation; mentor/advisor input may still be sought when useful, but the repository must not block progress solely waiting for it.

## 2. Fixed research contract

**Project objective:** Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

**Operational research question:** How do traditional numerical calibration, an ordinary ANN, and a constraint/physics-informed inverse method compare on the canonical ten-parameter Double Heston inverse problem under one frozen representation and evaluation protocol, especially when the inverse problem is practically non-identifiable at realistic market tolerance?

Canonical target order:

1. `kappa_slow`
2. `theta_slow`
3. `sigma_slow`
4. `rho_slow`
5. `v0_slow`
6. `kappa_fast`
7. `theta_fast`
8. `sigma_fast`
9. `rho_fast`
10. `v0_fast`

The contract preserves positivity, strict factorwise Feller conditions, the declared joint correlation disk, slow/fast ordering, and the frozen production Double Heston pricing engine. It preserves the Black-Scholes -> Standard Heston -> Double Heston comparison and the traditional calibration -> ANN -> informed-inverse comparison.

Synthetic truth remains the basis for parameter-recovery claims. Real-market fitted parameters are not ground truth. Primary ANN/inverse-model learning remains synthetic; primary real-market neural-weight updating is prohibited. Sector separation and frozen, unseen real-market final evaluation remain mandatory.

## 3. Pricing foundation

The canonical production Double Heston pricer remains validated and frozen. The overnight audit additionally confirmed machine-precision agreement with the differentiable Torch mirror in the tested region and close agreement with the independent Archive-2 COS implementation in the liquid region.

No overnight evidence warrants changing the production pricing engine.

## 4. G2 / identifiability evidence

The central bottleneck remains **representation / information / practical identifiability**, not pricing-engine correctness.

Standing evidence before the overnight run already established global ambiguity on the market-supported central geometry and rejected the provisional 108-grid as the final unchanged representation.

The 22 August Node B diagnostic extended the picture on the full provisional 108-quote grid:

- representative full-grid Jacobians can be locally full practical rank;
- strict-precision clean recovery can be good;
- realistic small price noise produces sharp parameter instability while repricing remains near the noise floor;
- materially different parameter vectors can occupy near-equivalent price-surface regions;
- increased optimizer budget, alternative starts, and weighting changes do not remove the governing failure; and
- factor-swap symmetry is an exact/near-machine-precision degeneracy broken by the declared slow/fast ordering convention.

The defensible interpretation is **ill-conditioned at realistic noise scale and practically non-identifiable at market tolerance**, with case-dependent optimizer-basin sensitivity on clean data.

Consequently:

- repricing quality is not parameter-recovery evidence;
- parameter-recovery claims must be tolerance/equivalence-class conditioned; and
- a physics term may regularize/structure the problem but must not be claimed to create missing identifying information.

## 5. Current neural / PINN status

The repository contains implemented inverse-model infrastructure, but no validated research-scale PINN result.

Use these two axes:

```text
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

The current canonical inverse model is accurately described as **constraint-informed + differentiable-repricing-informed**. It is not presently a genuine PDE-informed PINN.

Archive-2's current PDE loss must not be adopted into the canonical path. Node A and Node C independently reproduced an autograd slice-view defect that silently removes the variance-factor derivative terms from the implemented residual. Even with correct derivative wiring, a pricer-side residual on an already accurate model pricer is approximately machine-zero and is not an independent parameter-identification signal.

A genuine network-side PDE-informed Model 3 is optional future work, not a blocker for the primary study.

## 6. Architecture control

The canonical stack remains the source of truth for:

- parameter order and semantics;
- structural constraints;
- production pricing;
- differentiable repricing;
- synthetic training policy; and
- frozen real-market evaluation policy.

Archive-2 / `src/dheston` is experimental/donor code only. Selected patterns may be adapted behind explicit interfaces, including variable-length surfaces, chronological zero-leakage evaluation, and the COS pricer as a cross-check.

Cross-stack positional parameter passing is prohibited. Any interoperability requires an explicitly verified named adapter because the stacks differ in both factor placement and within-factor parameter order.

Archive-2 `real_finetune` / `--continuous` real-market weight updating is outside the canonical primary protocol and must be removed or hard-quarantined from canonical entry points before any future training milestone.

## 7. Current pipeline state

```text
PRODUCTION_DH_PRICER = VALIDATED_AND_FROZEN
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_PROTOCOL = SELF_GOVERNED_R2_VS_R3_EXECUTED_AND_SEALED
G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING
OPTIMIZER_ONLY_WORK = CLOSED
NTPC_THREE_DATE_INFORMATION = MULTI_DATE_INSUFFICIENT
GLOBAL_AMBIGUITY = ESTABLISHED
FINAL_10K = ABSENT
ANN_RESEARCH_TRAINING = NOT_STARTED
MODEL2_RESEARCH_TRAINING = NOT_STARTED
R2_SYNTHETIC_GENERATION_CONTRACT = FROZEN
R2_DEVELOPMENT_PILOT = VALIDATED_WITH_VERIFIED_IDENTICAL_REPLAY
FINAL_PARAMETER_POOL_READINESS = VERIFIED_WITHOUT_PRICING
CHALLENGE_OOD_FINAL_GENERATION = NOT_STARTED
G8 = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

The overnight run did not itself change any gate. The predeclared G2
representation study, executed and sealed on 22 August 2026
([G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md](G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md)),
changed the G2 gates exactly once: representation frozen to R2 and G2 passed
with practical non-identifiability retained.

## 8. Development-data registry and G8 protection

All previously used NTPC dates remain DEVELOPMENT / DIAGNOSTIC and ineligible for final frozen G8 ANN/inverse-model evaluation:

- 2026-07-01
- 2026-07-08
- 2026-07-15
- 2026-07-22
- 2026-07-29

Final G8 dates must be later and untouched, reserved before ANN/inverse-model evaluation, not selected using neural performance, not used during representation design, and not used to update primary neural weights.

No final G8 dates are selected here.

## 9. Self-governed G2 protocol — executed and sealed

The former mentor-approval blocker is retired. G2 was controlled by the predeclared [G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md](G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md), which was executed exactly once on 22 August 2026 with all thresholds frozen before outcomes.

The bounded experiment compared:

- **R2:** ranked two-expiry central-five calls/puts with actual maturity conditioning; and
- **R3:** ranked three-expiry masked central-five calls/puts with actual maturity conditioning and explicit masking of unsupported/unusable slots.

The unchanged 108-grid was not a candidate.

Execution used the five already-designated NTPC development dates for market support (R2 usable 78/100 slots; R3 78/150 — the third expiry rank contributed zero usable central-five slots on every date) and the frozen 20-truth synthetic panel for identifiability (1,920 calibration attempts: 20 truths x 2 representations x 4 noise levels x 12 starts; identical truths, common-slot perturbations, starts, optimizer, bounds, constraints, objective, and stopping rules across candidates).

## 10. G2 stopping rule — applied once

At the end of the bounded R2/R3 experiment:

- freeze R3 if it is market-supported under masking and shows the predeclared strong/partial information improvement over R2 without violating the existing holdout guardrail;
- otherwise freeze R2 as the simpler supported representation;
- if practical non-identifiability remains at realistic noise, preserve it as a central research finding rather than reopening an unlimited search; and
- reopen representation design only if both candidates fail the hard market-construction requirements.

**Applied outcome (22 August 2026):** rule 2 fired — R3 showed strong clean-data improvement
(median/max dispersion −35.0%/−30.8%, fewer clusters) that collapsed at every realistic
noise level (classification NO_MATERIAL_IMPROVEMENT at 0.5%, 1%, and 2%), so **R2 is
frozen**; both candidates remain practically non-identifiable at 0.5% noise (median
best parameter RMSE 0.383/0.356 range-scaled vs the 0.05 material-displacement
convention while repricing stays at noise scale), so the finding is retained. The
holdout guardrail was NOT_APPLICABLE by predeclared determination.

Valid completion labels are:

```text
G2 = PASSED_REPRESENTATION_FROZEN_IDENTIFIABILITY_ACCEPTABLE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
G2 = FAILED_MARKET_CONSTRUCTION_REQUIREMENTS
```

The second label is a legitimate pass state for moving into the primary method comparison.

## 11. Allowed now

Allowed immediately:

- implement and run the frozen G2 R2/R3 protocol;
- maintain and review documentation/evidence;
- quarantine non-canonical real-market training paths;
- build non-binding representation plumbing required to execute R2/R3 without freezing the winner early;
- run focused deterministic tests/diagnostics required by the G2 protocol; and
- preserve all failures and negative findings.

Still prohibited before G2 representation freeze:

- final 10k generation;
- research ANN/Model-2 training;
- final G8 date selection;
- real-market neural weight updating; and
- post-hoc methodology changes designed to improve the G2 outcome.

## 12. After G2

After the bounded G2 experiment selects and freezes R2 or R3:

1. formalize the selected representation interface;
2. update and revalidate the synthetic surface contract;
3. generate the final synthetic dataset;
4. run integrity/validity checks;
5. freeze train/validation/test splits;
6. run traditional calibration vs Model 1 ANN vs Model 2 constraint+repricing-informed inverse network under identical evaluation contracts;
7. report parameter recovery, repricing, structural validity, multi-seed stability, equivalence/tolerance-conditioned recovery, noise robustness, and runtime separately;
8. freeze models; and
9. run untouched chronological G8 real-market evaluation without weight updating.

A genuine PDE-informed Model 3 may be added later as a separately justified regularization/structural-validity extension, but it must not delay the primary comparison.

## 13. Exact next scientific action

`REGENERATE / REVALIDATE THE FINAL SYNTHETIC SURFACE CONTRACT ON THE FROZEN R2 INTERFACE`

The predeclared G2 R2/R3 study was executed and sealed on 22 August 2026. The
stopping rule froze **R2** (20 nominal slots with explicit real-market mask/missingness; synthetic panel complete by construction; never impute a missing real quote) after no strong/partial practical-information improvement
for R3 at any realistic noise level, despite better local conditioning), with
`PRACTICAL_NON_IDENTIFIABILITY = RETAINED_RESEARCH_FINDING`. The frozen R2
interface has been formalized as the canonical post-G2 representation
contract (`src/r2_representation/`,
[R2_REPRESENTATION_CONTRACT.md](R2_REPRESENTATION_CONTRACT.md)): explicit
actual maturities and per-rank rate/carry conditioning, 20 central-five
call/put slots in one deterministic tested order, explicit mask semantics,
versioned serialization, and structural rejection of legacy-108/rejected-R3
data. The next action is to regenerate and revalidate the final synthetic
surface contract on that interface. Final 10k generation,
ANN/Model-2 research training, final G8 date selection, and frozen real-market
evaluation remain separately controlled milestones.
