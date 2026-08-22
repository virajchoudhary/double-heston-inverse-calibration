# Architecture

The repository contains one canonical research stack and one non-canonical imported/experimental stack. The canonical stack remains the source of truth for parameter semantics, structural validity, pricing, training policy, and research evaluation.

See [OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md](OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md) for the evidence-backed architecture decision produced by the three-node audit.

## Canonical research stack

The ordinary ANN baseline is a parameter-supervised, non-physics neural network. It intentionally contains no PDE residual.

```mermaid
flowchart LR
    A["Option surface"] --> B["Approved surface representation"]
    B --> C["Data-derived model input size"]
    C --> D["Ordinary PyTorch MLP"]
    D --> E["Ten canonical Double Heston outputs"]
    E --> F["Independent canonical Double Heston repricer"]
```

The canonical inverse-model infrastructure additionally supports a differentiable Torch Double Heston repricer and hard-by-construction structural constraints. Its present scientific classification is **constraint-informed + differentiable-repricing-informed inverse network**. It is not presently a genuine PDE-informed PINN.

## Surface contract and G2 boundary

The historical/provisional candidate grid combines nine log-moneyness coordinates, six maturity coordinates, and separate call and put blocks: `9 x 6 x 2 = 108` normalized inputs.

That 108-feature representation is **not** the frozen final research representation. Current status is:

```text
CURRENT_108_GRID = REJECTED_AS_FINAL_UNCHANGED_GRID
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY
```

The model layer already derives its input size from the dataset rather than treating 108 as an immutable neural-network constant. The frozen G2 representation (selected 22 August 2026 by the predeclared
self-governed R2-vs-R3 protocol) is **R2**: first two eligible listed expiry
ranks x central-five log-moneyness x calls/puts = 20 NOMINAL spot-normalized
price slots with explicit mask/missingness for unsupported or unusable
real-market observations (never imputed with model prices; the synthetic G2
panel is complete by construction), with actual time-to-maturity supplied
explicitly and existing per-rank rate/carry conditioning. The canonical
software interface for this frozen contract is
`src/r2_representation/` ([R2_REPRESENTATION_CONTRACT.md](R2_REPRESENTATION_CONTRACT.md)):
one deterministic 20-slot key order, explicit mask semantics, a synthetic
constructor over the unchanged production pricer, a real-market constructor
over the sealed official-NSE audit contract, and a versioned JSON
serialization schema. Final synthetic generation and dataset loaders must
build on it rather than re-pointing the legacy 108-grid utilities.

Complete surfaces must remain together in exactly one train, validation, or test split.

## Exact canonical parameter order

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

These outputs represent eight structural parameters and two surface-specific initial variance states.

## Declared canonical constraints

- Positive `kappa`, `theta`, `sigma`, and `v0` for both factors
- `kappa_slow < kappa_fast`
- `2 * kappa_slow * theta_slow - sigma_slow^2 > 0`
- `2 * kappa_fast * theta_fast - sigma_fast^2 > 0`
- Each correlation strictly inside `(-1, 1)`
- `rho_slow^2 + rho_fast^2 < 1`

Structural validity and reviewed sampling-box membership are deliberately separate concepts. Do not silently clamp network outputs to the synthetic training box merely to improve recovery metrics.

## Pricing ownership

The pricing hierarchy is:

1. frozen production canonical Double Heston engine in `src/double_heston.py` — scientific source of truth;
2. independently checked differentiable Torch mirror — canonical differentiable workhorse;
3. Archive-2 COS pricer — independent numerical cross-check only after parameter adaptation.

The overnight audit found the production and Torch implementations in machine-precision agreement in the tested liquid region, while the independent COS implementation agreed closely enough to support use as a cross-check. No production-pricer change is warranted from the overnight evidence.

## Archive-2 / `src/dheston` disposition

Archive-2 is a donor of selected patterns, not a second canonical implementation.

Potentially reusable after explicit adaptation:

- variable-length masked surface representation;
- chronological zero-leakage real-market evaluation pattern;
- COS pricer as an independent pricing cross-check.

Do not adopt directly:

- positional parameter interchange;
- Archive-2 parameter constraints as canonical semantics;
- negative-only rho box semantics;
- its current PDE loss;
- its real-market fine-tuning / continuous-training path.

The two stacks use different parameter layouts. Any interoperability must use an explicitly verified named adapter; positional tensor passing is forbidden.

## PDE / physics boundary

The overnight audit independently reproduced a defect in Archive-2's PDE loss: its autograd wiring silently zeros the variance-factor derivative terms, so the implemented residual is not the intended Double Heston PDE residual.

Do not import that loss into the canonical path.

More fundamentally, a correctly wired PDE residual evaluated on an already accurate model pricer is approximately machine-zero and does not create independent parameter-identification information. If a genuine PDE-informed Model 3 is later approved, it requires a separate network-side construction and must be evaluated as a regularization/structural-validity mechanism rather than assumed to resolve inverse non-identifiability.

## Research training and evaluation policy

Primary ANN and inverse-model learning is synthetic. Real-market observations are reserved for frozen-model evaluation and must not update primary neural-network weights.

The canonical experimental comparison should hold the surface contract, target order, synthetic splits, evaluation sets, and metric families fixed across methods.

Future reports must separate:

- parameter recovery;
- repricing;
- structural validity;
- seed/start stability;
- tolerance-conditioned or equivalence-class recovery;
- noise robustness; and
- runtime.

No method may be declared superior from repricing RMSE alone.

## Current implementation status

```text
PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED
PINN_RESEARCH_MILESTONE = NOT_VALIDATED_OR_TRAINED
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
FROZEN_REAL_MARKET_EVALUATION = NOT_STARTED
```

`src/pricing_interface.py` continues to route research generation and repricing to the canonical pricing engine. Development smoke-test paths remain explicitly labelled and must never become implicit fallbacks.
