# Research Control and Current Status

Status date: 22 August 2026

This is the canonical repository control/status document for the next scientific decision. It records approved constraints and validated evidence; it does not itself approve or execute a new treatment.

The detailed 13 August 2026 state remains preserved in Git history and in the experiment-specific documents linked throughout `docs/`. The 22 August three-node diagnostic consolidation is recorded in [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md).

## 1. Source-of-truth hierarchy

1. **Canonical design — v2.0.** The mentor-updated research design remains the canonical scientific design.
2. **Controlled update — v2.1.** A status/update may record results or proposals, but a proposal is not approved merely because it appears in repository documentation.
3. **Implementation truth — this repository.** Code, configurations, manifests, persisted evidence, tests, Git history, hashes, and approved merge history determine what was actually implemented and validated.

The external v2.0/v2.1 documents are not tracked here. This hierarchy is a project-control rule, not a claim that those files exist in the repository.

## 2. Fixed approved research contract

**Project title / repository-approved objective:** Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

**Operational research question:** Can the canonical ten-parameter Double Heston inverse problem recover stable parameters from a market-supported option-surface representation, and how do traditional calibration, an ordinary ANN, and a physics-informed inverse method compare under one frozen protocol?

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

The approved contract preserves positivity, strict factorwise Feller conditions, the declared joint correlation disk, slow/fast ordering, and the frozen production Double Heston pricing engine. It preserves the Black-Scholes -> Standard Heston -> Double Heston comparison and the traditional calibration -> ANN -> physics-informed inverse comparison.

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

The repository now contains implemented inverse-model infrastructure, but no validated research-scale PINN result.

Use these two axes:

```text
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
```

The current canonical inverse model is accurately described as **constraint-informed + differentiable-repricing-informed**. It is not presently a genuine PDE-informed PINN.

Archive-2's current PDE loss must not be adopted into the canonical path. Node A and Node C independently reproduced an autograd slice-view defect that silently removes the variance-factor derivative terms from the implemented residual. Even with correct derivative wiring, a pricer-side residual on an already accurate model pricer is approximately machine-zero and is not an independent parameter-identification signal.

A genuine network-side PDE-informed Model 3 is a separate future research decision, not an approved current milestone.

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
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
OPTIMIZER_ONLY_WORK = CLOSED
NTPC_THREE_DATE_INFORMATION = MULTI_DATE_INSUFFICIENT
GLOBAL_AMBIGUITY = ESTABLISHED
FINAL_10K = ABSENT
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

The overnight run did not change any gate.

## 8. Development-data registry and G8 protection

All previously used NTPC dates remain DEVELOPMENT / DIAGNOSTIC and ineligible for final frozen G8 ANN/inverse-model evaluation:

- 2026-07-01
- 2026-07-08
- 2026-07-15
- 2026-07-22
- 2026-07-29

Final G8 dates must be later and untouched, reserved before ANN/inverse-model evaluation, not selected using neural performance, not used during representation design, and not used to update primary neural weights.

No final G8 dates are selected here.

## 9. Approval-sensitive G2 proposal

The previously prepared mentor decision remains the immediate scientific blocker.

Before final representation freeze, require both:

- a market-supported representation; and
- sufficient ten-parameter informativeness/stability under the approved tolerance/noise interpretation;

or an explicitly mentor-approved revised formulation.

The proposed bounded richer NTPC information study remains mentor-controlled. It must not silently introduce priors, regularization, temporal smoothing, realized-volatility supervision, CIR penalties, wider bounds, a new optimizer/objective, a new sector, ANN training, or inverse-model training.

Pre-existing dispersion/holdout rules remain unchanged. Any new minimum-date, minimum-row, boundary-pressure, or `RICHER_INFORMATION_*` decision thresholds must be predeclared and mentor-confirmed before results are seen.

## 10. Allowed now

Allowed before mentor approval:

- maintain and review documentation;
- preserve and review overnight evidence;
- quarantine non-canonical execution paths without changing the approved scientific method;
- prepare the mentor handoff;
- design non-binding software interfaces that do not freeze the representation; and
- perform ordinary code/test hygiene that does not create new research results.

## 11. Blocked work

Blocked pending the existing mentor/G2 decision:

- activating the richer NTPC information treatment;
- freezing the final representation;
- generating the final research dataset;
- research ANN training;
- research inverse/PINN training;
- final G8 date selection; and
- any methodology change not already approved.

## 12. After mentor approval

If the bounded G2 study is approved:

1. freeze the approved protocol before seeing results;
2. audit existing development evidence and acquire only approved missing evidence;
3. retain only dates satisfying the existing support rules;
4. run the approved information-only treatment;
5. analyze pricing, stability, identifiability, and boundary pressure;
6. apply the predeclared rules;
7. stop information-design work after the bounded study;
8. either recommend representation freeze or return to the mentor; and
9. do not automatically start ANN/inverse-model research training in the same experiment.

After G2 passes, formalize the representation interface, generate/validate the final synthetic dataset, freeze splits, and only then execute the fair traditional-calibration vs Model-1 ANN vs Model-2 comparison.

## 13. Exact next scientific action

`OBTAIN MENTOR DECISION`

- If approved: launch the bounded richer NTPC information milestone under the frozen protocol.
- If rejected: return to a mentor-approved formulation/G2 decision.

Repository cleanup/documentation may proceed in parallel, but it must not be confused with scientific gate passage.
