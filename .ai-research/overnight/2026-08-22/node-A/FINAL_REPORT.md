# Node A Final Report — Overnight 2026-08-22

**STATUS: DRAFT (auto-refreshed as evidence lands; peer sections pending until ~07:00 IST)**

## Executive conclusion

The canonical inverse-calibration architecture should remain the **canonical stack**
(`src/constants.py` parameter contract + frozen production Gauss-Laguerre pricer + its
machine-precision torch mirror + `DoubleHestonConstraintMap` + fixed-size surface features +
synthetic-only training with validation-gated checkpointing). Archive-2 (`src/dheston`)
should be treated as a **donor of patterns, not a second implementation**: adopt (behind
adapters) its variable-length surface pattern, chronological/zero-leakage real-market
evaluation pattern, and COS pricer as an independent cross-check; quarantine its parameter
order, box constraints, and real-finetuning trainer. The "physics-informed" claim must be
re-labeled honestly: the current canonical model is constraint+repricing-informed (Model 2);
a genuine PDE-informed tier (Model 3) requires a research decision and a network-side PDE
construction that neither stack currently contains.

## Repository state audited

Genesis SHA `642702e6706a3d17b3031619f35bda39bc144483` (= origin/main at start; verified
unchanged during run). Branch `overnight/20260822-a-architecture`. Working tree clean at each
checkpoint. No merges of peer branches.

## Canonical stack findings

**Strengths:** correct canonical parameter contract; structural constraints hard by
construction (positivity, kappa ordering, Feller margin, correlation disk); torch pricer
matches production at ~1e-15 relative (diagnostic A-004); enforced disjoint splits; train-only
standardizer; validation-gated checkpoints; explicit `real_market_data_used: False` logging;
29 focused test files; `input_size` fully data-derived (G2-resilient model layer).

**Weaknesses:** "physics" loss is repricing consistency only (no PDE term) — accurate
labeling required; constraint map guarantees structural validity but not reviewed-box
membership (OOD predictions possible — research decision); 108-representation provisional
(G2 open); no variable-length surface interface yet.

**Status:** KEEP as canonical seam. Infrastructure implemented; not research-trained.

## Archive-2 findings

**Useful pieces:** variable-length masked point-cloud surfaces + padded batching;
chronological split with `verify_zero_leakage`; well-formed Double Heston PDE residual
(correct term-by-term); independent COS pricer (agrees with production 1e-12..1e-9 in the
liquid region).

**Conflicts:** incompatible parameter order (double permutation — factor swap + reorder);
box constraints allowing Feller-violating and correlation-disk-violating regions;
negative-only rho; `ordering_penalty` dead by construction; single test file.

**Critical defect (F19):** the PDE residual loss is incorrectly implemented — an autograd
slice-view bug (`grad(prices, chosen_params[:,0])` -> None -> `_safe_grad` zeros) silently
drops all six variance-factor derivative terms, so the implemented residual is
`d_tau - (diffusion + drift)`, a wrong PDE. Identity-proven (residual == dropped terms
exactly); correctly wired it is machine-zero (~1e-14), hence non-discriminating for
parameters. DEPRECATE; do not import. Repro: `artifacts/diag_pde_bug_repro.py`.

**Risks:** `real_finetune` mode and `--continuous` flag update neural weights on real NSE
market data — direct violation of the canonical research control. REMOVE FROM CANONICAL PATH.

## Parameter-contract conclusion

Not "simply reordered" — a **double permutation** (slow/fast factor identity swap between
stack positions AND within-factor reordering [k,θ,σ,ρ,v0] vs [v0,κ,θ,σ,ρ]). An adapter is
sufficient and has been **verified numerically** via exact gradient permutation (A-004). No
rescaling needed; positional tensor interop between stacks is forbidden.

## Constraint-contract conclusion

Stack A: all canonical constraints hard by construction; no sampling-box restriction.
Stack B: box bounds hard by construction; Feller absent; joint correlation disk absent;
negative-only rho. The spaces are semantically different (neither contains the other).
Constraint-map OOD concern is real but must not be "fixed" by silent restriction — research
decision required (fairness vs OOD interpretability).

## Pricer ownership conclusion

1. Frozen production engine (source of truth) — KEEP, unchanged.
2. Torch Gauss-Laguerre mirror — validated at ~1e-15; KEEP as the differentiable workhorse.
3. Archive-2 COS — equivalent mathematics, independent numerics; ADAPT as cross-check only;
   far-OTM tail caveat for near-zero prices.
No pricer bug found; no production change warranted.

## Physics/PINN classification

Current canonical model = **constraint-informed + repricing-informed inverse network**.
Stack B = same family plus a nominal PDE-residual term that constrains the pricer, not the
network. Proposed paper taxonomy: Model 1 (ordinary ANN), Model 2 (constraint+repricing,
current), Model 3 (PDE-informed — gated on research decision + Node C-verified network-side
construction).

## Surface/G2 coupling

G2 NOT PASSED; 108-feature representation provisional. Model layer already decoupled
(data-derived `input_size`); grid change costs sit in constants + generators + datasets.
Recommend a representation interface before the 10k generation, informed by B's
variable-length pattern. No grid chosen tonight.

## Training-policy findings

Canonical: clean (synthetic-only, no leakage path found, see F10). Archive-2:
`real_finetune`/`--continuous` = REMOVE FROM CANONICAL PATH; synthetic-on-real-grids =
ISOLATE + disclose; box sampling = ISOLATE.

## Fair ANN/PINN experiment contract

See F7: identical splits/representation/repricer/metrics families, multi-seed, frozen
real-market stage, no real fine-tuning asymmetry, no RMSE-only winner, constraint-ablation
reporting for Model 2.

## Documentation contradictions

README accurate; `RESEARCH_CONTROL...md:97` (`NOT_IMPLEMENTED_OR_TRAINED`) and
`CURRENT_STATUS.md:48` ("development ... Not started") stale on the infrastructure axis.
Proposed two-axis vocabulary (`PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED` /
`PINN_RESEARCH_MILESTONE = NOT_DERIVED_OR_TRAINED`) — for human-approved docs PR, not applied
tonight.

## Changes made

Node A evidence only: `STATUS.md`, `FINDINGS.md` (F1–F10), `EXPERIMENTS.jsonl` (A-000–A-006),
`architecture/SEAM_MATRIX.md`, `artifacts/diag_pricer_agreement.py` (+ its committed output
summary in FINDINGS). No source, config, dataset, or docs changes.

## Tests

`tests/test_parameter_order.py`, `tests/test_constraints.py`, `tests/test_pinn_forward.py`,
`tests/test_archive2_pricing_smoke.py` — **15 passed** (CPU, 3.11s). Diagnostic A-004 seeded
and reproducible.

## Remaining uncertainty

- Whether the reviewed sampling box should constrain predictions (RDD).
- Model 3 PDE construction design (RDD; Node C input pending).
- Node B identifiability evidence and Node C PDE audit — PENDING at draft time.
- Whether archive-2's variable-grid pattern should enter the canonical interface before or
  after G2 resolution (RDD, low urgency).

## Human decisions required

1. Approve removal/quarantine plan for `real_finetune` + `--continuous` (canonical path).
2. Approve two-axis PINN status vocabulary + docs PR.
3. Decide reviewed-box vs structural-only constraint policy (fairness/OOD).
4. Decide whether Model 3 (PDE-informed) enters the research plan, given the identifiability
   picture Node B reports.
5. Approve the parameter-adapter seam as the ONLY sanctioned cross-stack interop mechanism.

## Recommended next actions

1. Human review of this seam matrix; adopt/adjust classifications.
2. Docs reconciliation PR (two-axis PINN status).
3. Policy patch isolating real-finetuning from canonical entry points.
4. Representation-interface design (informed by B's pattern) BEFORE 10k generation.
5. Integrate Node B/C evidence into Model 3 go/no-go.

## Branch and final SHA

Branch: `overnight/20260822-a-architecture`. Final SHA: (filled at consolidation).
