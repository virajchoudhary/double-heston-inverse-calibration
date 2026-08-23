# R2 Primary Comparison — Pre-Training Audit

Audit date: 23 August 2026.
Branch: `research/primary-r2-model-comparison` (base: canonical main `fdbdc35`).
Auditor: single-agent Ox Alpha session (per milestone runbook model rule).

This document answers the ten mandated pre-training audit questions **before**
the primary comparison protocol is frozen and before any research training
occurs. Every answer cites repository evidence. Measured feasibility numbers
were obtained on **train-split surfaces only** (the test split was not read
beyond structural dataset verification: hash, counts, split disjointness,
representation identity).

Environment facts recorded during the audit:

- CPU-only PyTorch `2.11.0+cpu` (`torch.cuda.is_available() == False`).
- 12 logical cores.
- Full test suite at audit time: **484 passed, 7 failed**; all 7 failures are
  the pre-existing persisted-NTPC-artifact failures documented in
  `docs/REPOSITORY_AUDIT.md` and prior milestones (re-verified on canonical
  main before this branch; unchanged here).

---

## A. Can the existing ANN consume canonical R2 rather than historical 108 inputs?

**No — an R2-native loading path must be added (implementation repair only).**

`SurfaceParameterDataset.from_surface_frame` (`src/dataset.py:70-143`) consumes
a long-format CSV frame with the legacy grid columns (`option_type`,
`maturity_days`, `log_moneyness`, ...) and enforces exactly
`expected_input_size()` rows per surface — the rejected 108-slot legacy grid
(`src/surface_grid.py`). The canonical final dataset
(`data/final_r2_clean_10000/surfaces.jsonl`) is R2 JSONL: 20 canonical slots
per record with per-slot `prices` (spot-normalized), `mask`, `maturities`
(years), `rates`, `carries`, plus `slot_keys` in the frozen R2 order.

The repair adds a new constructor that reads the frozen JSONL directly and
builds the frozen R2 feature vector (see the protocol document). The legacy
CSV path remains untouched for its historical tests; the new path structurally
cannot receive 108-length vectors (the R2 contract module carries an explicit
`LEGACY_108_INPUT_SIZE` rejection guard, `src/r2_representation/contract.py`).

## B. Does it consume 20 normalized prices, maturity information, rate/carry conditioning, mask semantics?

**After the repair: yes, all four, for both neural methods.**

The frozen dataset stores per-slot: `prices` (spot-normalized, spot = 100
convention), `mask` (genuine booleans; synthetic surfaces complete), actual
`maturities` in years (per rank: dte/365), and per-rank `rates`/`carries`
replicated per slot (verified: within every final-dataset record the rate and
carry are rank-constant; generation identity `carry = rate + carry_offset`,
`src/r2_synthetic_generation.py:326-340`). The frozen feature construction is
defined in `docs/R2_PRIMARY_COMPARISON_PROTOCOL.md` §INPUT and is identical
for Model 1 and Model 2: `[prices_masked(20), mask(20), maturities(20),
rates(20), carries(20)]` = 100 inputs in canonical slot order, with masked
slots carrying `0.0` (the representation contract's
`MASKED_PRICE_PLACEHOLDER`).

## C. Does Method 2 consume exactly the same observational information as Model 1?

**Yes after the repair — and this is enforced by construction.**

Both models receive the identical 100-feature vector built by one shared
feature builder. Method 2's differentiable-repricing term reuses the *same*
observed spot-normalized prices (the first 20 features, gated by the same
mask) as its regression targets, plus the surface geometry (strikes
`spot*exp(k)` per canonical slot key, per-slot maturities/rates/carries) which
is deterministic given the frozen representation contract. No market
information beyond the R2 observation is available to either model.

## D. Does Method 2 use the differentiable Torch Double Heston pricer correctly?

**Not as currently wired — two defects were found and will be repaired; the
pricer mathematics itself is validated.**

1. **float32 defect (correctness):** `src/train_pinn.py` builds float32 batch
   tensors. The Torch pricer inherits dtype from its parameters
   (`_real_dtype`, `src/torch_double_heston.py`); in float32 the Gauss–Laguerre
   compensation / characteristic-function evaluation produces non-finite
   prices on real final-dataset batches — reproduced directly during this
   audit (`FloatingPointError: non-finite Double Heston call price`). The
   validated regime is float64 (machine-precision agreement with the frozen
   production pricer; `tests/test_torch_double_heston.py`). Repair: the R2
   training path upcasts pricer inputs to float64 (gradients flow through the
   cast); an equivalence test pins Torch-vs-production agreement on
   final-dataset surfaces.
2. **throughput defect (feasibility):** the existing per-quote Python loop
   costs ≈13.9 s per 64-surface batch forward+backward (measured, float64,
   real train-split geometry) ≈ 27 min/epoch — infeasible for multi-seed
   CPU training. Repair: a batch-vectorized evaluation of the *same*
   formulation (identical Little Heston Trap exponent, identical
   Gauss–Laguerre rule and node count, identical guards), with an equivalence
   test against the existing loop implementation and the production pricer.
   This is a numerics-preserving implementation change, not a scientific-design
   change; it is recorded in the protocol §METHOD-2 before training.

## E. Are parameter outputs transformed/constrained consistently?

**Yes, and the consistency will be pinned by tests.**

- Canonical order everywhere: `src/constants.py PARAMETER_NAMES` ==
  dataset `metadata.parameters_canonical_order` key order ==
  constraint module order (verified by `tests/test_parameter_order.py` and
  the protocol checkpoint test).
- Model 1 outputs are unconstrained standardized-target predictions,
  inverse-transformed to physical units by the train-split
  `TargetStandardizer` (`models/parameter_transform.py`); structural
  validity is then *measured*, not enforced.
- Model 2 outputs pass through `DoubleHestonConstraintMap`
  (`models/pinn_model.py`): strict positivity (+1e-6), strict slow/fast
  kappa ordering (`kappa_fast = kappa_slow + softplus`), Feller-safe sigma
  ceiling `sqrt(2*kappa*theta)*0.995`, joint correlation disk radius
  `<= 0.995`. Both methods' parameter supervision uses the same
  train-split-fitted standardized targets.
- Traditional calibration uses the existing constraint reparameterization
  (`src/calibrate_double_heston.py:unconstrained_to_parameters`) with
  `configs/parameter_bounds_PROVISIONAL.yaml` hard safety bounds and the
  same declared constraint set.

## F. Are train/validation/test splits read from the frozen dataset without leakage?

**Yes — the split is stored per surface in the frozen dataset itself.**

Verified by counting `metadata.user_metadata.split` over all 10,000 records:
train = 7,500, validation = 1,250, test = 1,250; zero surface-id overlap and
zero parameter-vector-hash overlap across splits (asserted in the pre-training
checkpoint test). `surfaces.jsonl` SHA-256 verified =
`148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`
(the canonical hash). The training path uses stored split labels only; target
standardization is fit on train rows only; early stopping/checkpointing uses
validation only; the test split is not loaded by any training entrypoint and
is touched only by the final frozen evaluation step.

## G. Are any old 108 assumptions still present?

**Retained but unreachable from the new R2 path.**

Legacy-108 lives in `src/surface_grid.py`, `SurfaceParameterDataset.
from_surface_frame`, `configs/ann_baseline.yaml`/`pinn_baseline.yaml`
(infrastructure-test configs, explicitly marked), and historical tests. The
new R2 loader and training paths import none of it; the R2 contract module
structurally rejects 108- and 30-length vectors; the frozen protocol pins
input size 100 = 5×20 canonical R2 blocks. A dedicated test asserts no
legacy-grid import in the new training path.

## H. Are any real-market observations reachable during training?

**No.**

The frozen dataset is synthetic-only (manifest:
`real_market_inputs_used: false`; every record `metadata.synthetic: true`,
`source: synthetic_canonical_double_heston_production_pricer`). Training
entrypoints read only this JSONL. The fail-closed real-market weight-update
quarantine (`src/dheston/real_market_policy.py`, Issue #20 / PR #29) is active
on this branch (quarantine test suite: 20/20 passed during this audit).

## I. Is Issue #20 quarantine still active?

**Yes** — `python -m pytest tests/test_real_market_weight_update_quarantine.py`:
20 passed (run on this branch during this audit, 2026-08-23).

## J. Are evaluation utilities currently compatible with R2?

**Partially.**

- `src/evaluate_parameters.py::evaluate_parameter_recovery` is
  representation-agnostic (matrix in, metrics out) — reusable directly.
- `src/evaluate_repricing.py` expects the legacy long CSV frame — not
  R2-compatible. The repair adds an R2-native evaluation module that reprices
  predicted parameters through the **production pricer** (node_count = 64,
  the dataset's generation convention) and computes the frozen metric
  families.

---

## Measured feasibility notes (train-split timing only)

- Production pricer reproduces stored dataset prices to machine precision
  (max abs diff 8.9e-16 on a train record; node_count 64, carry passed in the
  dividend-yield slot — the generation convention).
- Production pricer: ≈16.3 ms per 20-quote surface.
- Traditional calibration, existing module settings (3 starts × max_nfev
  300): 124–170 s per surface (3 train surfaces measured; optimizers routinely
  exhaust the 300-evaluation budget while polishing near the machine-precision
  repricing manifold — the documented practical non-identifiability behavior).
  Full test split (1,250 surfaces) ≈ 52 core-hours → executed with parallel
  surface-level worker processes; the frozen per-surface budget is unchanged.
- Torch differentiable pricer (existing loop, float64): ≈13.9 s per
  64-surface batch forward+backward → ≈27 min/epoch for Model 2 on 7,500
  surfaces → the vectorization repair (question D) is required for the
  frozen multi-seed plan.

## Audit conclusion

The scientific design (R2 representation, canonical parameter order,
constraints, production pricer, frozen splits) is intact and sealed. The
required work is strictly implementation repair: an R2-native dataset/
feature path, a float64 batch-vectorized differentiable repricing term, and
an R2-native evaluation module. No repair changes the frozen scientific
design. Training is NOT authorized until the protocol commit is pushed and
remote-verified.
