# Research Control and Current Status

Status date: 13 August 2026

This is the canonical repository control/status document for the next scientific decision. It records approved constraints and validated evidence; it does not approve or execute a new treatment.

## 1. Source-of-truth hierarchy

1. **Canonical design — v2.0.** The mentor-updated research design is the canonical scientific design.
2. **Controlled update — v2.1.** A v2.1 status/update may record results or proposals, but a proposal in v2.1 is not approved merely because it appears there.
3. **Implementation truth — this repository.** Code, configurations, manifests, persisted evidence, tests, Git history, and hashes determine what was actually implemented and validated.

The full v2.0 and v2.1 documents are not tracked in this repository. This hierarchy is therefore an external project-control rule, not a claim that those files exist here. Tracked documents corresponding to parts of the hierarchy include [ARCHITECTURE.md](ARCHITECTURE.md), [CURRENT_STATUS.md](CURRENT_STATUS.md), [NEXT_STEPS.md](NEXT_STEPS.md), the detailed experiment reports linked below, and their tracked evidence manifests.

## 2. Fixed approved research contract

**Project title / repository-approved objective:** Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery.

**Central research question (operational statement of the tracked contract):** Can the canonical ten-parameter Double Heston inverse problem recover stable parameters from a market-supported option-surface representation, and how do traditional calibration, an ordinary ANN, and a physics-informed inverse method compare under one frozen protocol? The full external v2.0 wording is not tracked here; this sentence summarizes the repository contract and does not replace v2.0.

The canonical target order remains:

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

The approved contract preserves positivity, strict factorwise Feller conditions, the joint correlation disk, slow/fast ordering, and the frozen production Double Heston pricing engine. It preserves the Black-Scholes -> Standard Heston -> Double Heston comparison and the traditional calibration -> ANN -> physics-informed inverse comparison.

Synthetic truth remains the basis for parameter-recovery claims. Real-market fitted parameters are not ground truth. Primary ANN/PINN training remains synthetic; primary real-market PINN weight updating is prohibited. Sector separation and a frozen, unseen real-market final evaluation remain mandatory. This milestone does not change equations, parameter definitions or order, bounds, optimizer, objective, pricer, or quote rules.

## 3. Verified merged checkpoint

The readiness baseline verified on 13 August 2026 is `main = 775b5cb2a204e5a55a024e6fe9e364172bc38109`, the normal merge of PR #16. Repair head `b9a64c8e761f09e7ff20926297ec4fa1976adbeb` is in that merge's ancestry. This SHA is the scientific checkpoint on which this status document was prepared; the later documentation-only merge containing this file may become the repository HEAD without changing the scientific checkpoint.

## 4. Current scientific evidence

### Pricing and NTPC classical comparison

The production Double Heston pricer is validated against the independent benchmark; see [DOUBLE_HESTON_VALIDATION_RESULTS.md](DOUBLE_HESTON_VALIDATION_RESULTS.md) and [ENGINE_FREEZE.md](ENGINE_FREEZE.md).

The 15-Jul NTPC holdout price RMSE values are:

| Model | Holdout price RMSE |
|---|---:|
| Black-Scholes | `1.053335898` |
| Standard Heston | `0.910569272` |
| single-date Double Heston | `0.926824720` |

Heston and Double Heston materially outperform Black-Scholes on this holdout. Double Heston has not clearly outperformed Standard Heston out of sample. See [NTPC_SINGLE_STOCK_CALIBRATION.md](NTPC_SINGLE_STOCK_CALIBRATION.md) and its [manifest](evidence/NTPC_SINGLE_STOCK_PILOT_MANIFEST.json).

### Single-date stability and optimizer-cap closure

The reviewed single-date shared-eight comparator has `11` materially displaced starts and `7` clusters. Optimizer convergence and price RMSE are not parameter-identification evidence.

The optimizer-budget-only study finished with:

```text
OPTIMIZER_CAP = OPTIMIZER_CAP_UNRESOLVED
OPTIMIZER_ONLY_WORK = CLOSED
```

Doubling the cap did not resolve cap incidence or separated near-equivalent basins. See [NTPC_DH_OPTIMIZER_CAP_SENSITIVITY.md](NTPC_DH_OPTIMIZER_CAP_SENSITIVITY.md) and its [manifest](evidence/NTPC_DH_OPTIMIZER_CAP_SENSITIVITY_MANIFEST.json).

### Three-date NTPC information study

| Metric | Single-date shared-eight | Three-date shared-eight |
|---|---:|---:|
| materially displaced | `11` | `7` |
| clusters | `7` | `3` |
| median separation | `0.399516908` | `0.324066116` |
| maximum separation | `0.627751647` | `0.481226608` |
| boundary-hit rate | `1.0` | `1.0` |
| 15-Jul holdout RMSE | `0.926824720` | `0.976300061` |

Temporal information materially improved dispersion, but the 15-Jul holdout deteriorated by `5.338%`, exceeding the inherited 5% ceiling. The final classification is `MULTI_DATE_INSUFFICIENT`. See [NTPC_DH_MULTI_DATE_CALIBRATION.md](NTPC_DH_MULTI_DATE_CALIBRATION.md) and its [manifest](evidence/NTPC_DH_MULTI_DATE_CALIBRATION_MANIFEST.json).

## 5. Current pipeline location

```text
PRODUCTION_DH_PRICER = VALIDATED
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED
OPTIMIZER_CAP = OPTIMIZER_CAP_UNRESOLVED
OPTIMIZER_ONLY_WORK = CLOSED
NTPC_THREE_DATE_INFORMATION = MULTI_DATE_INSUFFICIENT
FINAL_10K = ABSENT
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_IMPLEMENTED_OR_TRAINED
```

The bottleneck is **representation / information / identifiability**, not pricing-engine correctness.

## 6. Development-data registry and G8 protection

Any NTPC observation already used for market-support audit, representation design, traditional calibration, stability diagnosis, optimizer analysis, or multi-date information analysis is `DEVELOPMENT / DIAGNOSTIC` and is ineligible for final frozen G8 ANN/PINN evaluation.

| NTPC valuation date | Evidence already using the date | Registry status |
|---|---|---|
| `2026-07-01` | Original Stage A/G2 support work and three-date calibration | `DEVELOPMENT / DIAGNOSTIC` |
| `2026-07-08` | Five-Wednesday Stage A Power support extension only; not yet used in richer calibration | `DEVELOPMENT / DIAGNOSTIC`; `PROPOSED DEVELOPMENT CANDIDATE — NOT YET APPROVED/USED IN RICHER CALIBRATION` |
| `2026-07-15` | Original Stage A/G2 support, traditional calibration, stability, optimizer, and three-date calibration | `DEVELOPMENT / DIAGNOSTIC` |
| `2026-07-22` | Original Stage A/G2 support and three-date calibration | `DEVELOPMENT / DIAGNOSTIC` |
| `2026-07-29` | Five-Wednesday Stage A Power support extension only; not yet used in richer calibration | `DEVELOPMENT / DIAGNOSTIC`; `PROPOSED DEVELOPMENT CANDIDATE — NOT YET APPROVED/USED IN RICHER CALIBRATION` |

08-Jul and 29-Jul are not untouched: official NSE observations were already analyzed for market support. Their proposed status applies only to the not-yet-approved richer calibration treatment. All five dates above are excluded from final G8.

Final G8 dates must be later and untouched, reserved before ANN/PINN evaluation, not selected using ANN/PINN performance, not used during representation design, and not used to update neural weights in the primary design. No final G8 dates are selected here.

## 7. Approval-sensitive proposals

### Formal G2 safeguard — mentor approval required

Before final representation freeze, require both:

- a market-supported representation; and
- sufficient ten-parameter informativeness/stability;

or an explicitly mentor-approved revised formulation. This safeguard does not alter the canonical ten-parameter model.

### One bounded richer NTPC information study — mentor approval required

Candidate development dates are 01-Jul, 08-Jul, 15-Jul, 22-Jul, and 29-Jul. Only dates passing the existing official-NSE activity/support contract may enter the treatment. The ten parameters, pricer, bounds, constraints, optimizer, objective, quote rules, and shared-structure/date-specific-variance concept remain fixed; only temporal information density may change.

The study excludes priors, regularization, temporal smoothing, realized-volatility supervision, CIR penalties, wider bounds, a new optimizer, a new sector, ANN, and PINN.

### Decision rules

**Inherited/pre-existing rules:**

- material-distance and complete-linkage cluster diagnostics remain unchanged;
- strong dispersion reduction requires at least 25% reductions in both median and maximum pairwise separation and fewer clusters;
- partial dispersion reduction requires at least 10% reductions in both separation metrics and no increase in clusters;
- the existing 5% holdout-deterioration ceiling remains unchanged; and
- positivity, Feller, correlation-disk, slow/fast-ordering, bounds, and all other hard validity constraints remain unchanged.

**PROPOSED — REQUIRES MENTOR CONFIRMATION:**

- a minimum eligible-date count;
- a minimum selected-row count per date;
- an acceptable boundary-hit threshold; and
- the exact mapping from inherited dispersion/holdout evidence into any `RICHER_INFORMATION_*` label.

The repository does not currently provide approved numeric provenance for those four new rules. Their values must be predeclared and mentor-confirmed before results are seen; this document does not invent them or activate a new methodology.

## 8. Allowed now and blocked work

Allowed immediately under the existing contract: inspect repository evidence, maintain status/control documentation, prepare the mentor handoff, and perform normal documentation validation/review.

Blocked pending mentor approval: the formal G2 safeguard as an active methodology change, the richer NTPC study, and any representation freeze based on it.

Also blocked: final 10k generation, ANN, PINN, a new sector, changed bounds/methodology, priors, regularization, smoothing, realized-volatility supervision, CIR penalties, optimizer/objective changes, and final G8 date selection.

## 9. AFTER MENTOR APPROVAL

The next Codex milestone will:

1. freeze the approved protocol before seeing results;
2. audit existing 08-Jul/29-Jul evidence and acquire only approved missing evidence if needed;
3. retain only dates passing the existing support rules;
4. run the same information-only treatment;
5. analyze pricing, stability, and boundary pressure;
6. apply the predeclared rules;
7. stop information-design work after this bounded study;
8. either recommend representation freeze or return to the mentor; and
9. not automatically start ANN or PINN in the same experiment.

## 10. Exact next action

`OBTAIN MENTOR DECISION`

- If approved: launch the bounded richer NTPC milestone immediately.
- If rejected: return to a mentor-approved formulation/G2 decision.
