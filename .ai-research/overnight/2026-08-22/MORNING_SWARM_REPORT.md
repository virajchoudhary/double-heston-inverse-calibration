# DOUBLE HESTON OVERNIGHT SWARM REPORT

**STATUS: DRAFT — ALL THREE NODES INTEGRATED (Node B landed 03:11 IST, mid-run; Node C
complete with verification rounds). Consolidation target 07:00–07:30 IST; Node B may push
more evidence before then (re-polled). No peer results invented.**

## 1. Executive summary

Node A completed the full architecture and research-contract audit. The canonical stack is
confirmed as the correct canonical inverse-calibration architecture; archive-2 is a donor of
patterns (variable-length surfaces, chronological zero-leakage evaluation, independent COS
pricer) behind adapters, never positional interop. The "physics-informed" label must be
downgraded honestly to constraint+repricing-informed until a genuine network-side PDE
construction is approved. One research-control violation was found and evidenced in the
archive-2 trainer (real-market weight updating), already exercised once at smoke scale.
Node C independently reproduced Node A's critical findings (PDE-loss bug, physics
classification, constraint incompatibility, policy violation); Node B contributed no pushed
evidence at draft time (see §8/§15).

## 2. Starting state

Genesis SHA `642702e6706a3d17b3031619f35bda39bc144483` (= origin/main at start, verified).
Scientific state per control doc: G2 NOT PASSED; representation not frozen (108-grid
rejected as final unchanged representation); final 10k not generated; no research ANN/PINN
training; real-market weight updating prohibited; mentor gate outstanding.

## 3. Architecture conclusion

Joint Node A + Node C position (independently derived, mutually supportive):
canonical parameter contract + frozen production pricer + machine-precision torch mirror +
hard-by-construction `DoubleHestonConstraintMap` + data-derived-input-size surface features +
synthetic-only training with validation-gated checkpoints + post-freeze chronological
zero-leakage real-market evaluation (pattern adapted from archive-2). Cross-stack parameter
interop only via the verified permutation adapter (node-A artifact). Archive-2's real
fine-tuning paths are removed from any canonical path. If a genuine PDE-informed tier
(Model 3) is approved, both nodes converge on: learned forward price network with
canonical-PDE collocation + leaf state inputs; inverse network coupled through the LEARNED
pricer's residual; validation/objective parity; claims limited to regularization/structural
validity (not identification).

## 4. Identifiability/calibration conclusion

Node B INTEGRATED (77b8f2e; mid-run — re-polled before consolidation). Core result:
**global ambiguity replicates on the FULL 108-quote provisional grid.** Phase B multistart
(case_1, 12 starts): clean median parameter RMSE 0.148 at price RMSE 1.1e-6; with 0.5–2%
noise, 12/12 boundary hits, parameter RMSE 0.31–0.34 with price fit at the noise floor.
Phase A production entry: the clean best start recovers exactly (2.9e-11) but other starts
disperse (0.39) — optimizer-path dependence is real. Jacobian conditioning: full108
condition 2.55e4, practical rank 10/10 (NOT locally rank-deficient); the G2-anchor central-5
geometry's catastrophically worse conditioning (6.5e8, rank 7.5) REPLICATES. Failure mode is
therefore noise-scale vs weakest sensitivities (sigma_min 1.4e-5 < realistic noise 2.5e-4)
plus global multimodality — not classical rank deficiency. Calls/puts are parity-redundant
(conditioning identical to 5 digits). Standing repo context: G2 global
ambiguity analysis established 40 near-equivalent clean solutions in 39 separated clusters —
median normalized price RMSE 4.708e-8 vs range-scaled parameter RMSE 1.485e-1 — i.e.
repricing fit does not imply parameter recovery. Node C added two identifiability-relevant
datapoints: the correctly-wired PDE residual floor (4-5e-9) sits at the same noise scale as
the near-equivalence price differences (a PDE term cannot distinguish near-equivalent
vectors), and deterministic variance-mean propagation is price-inconsistent (-1.1% at 7d to
-14.5% at 90d), vindicating per-date v0 fitting. Node A's architecture stance: evaluation
claims must separate repricing from recovery and report equivalence-class structure; a
neural model must not be described as uniquely recovering truth from an ambiguous surface.

## 5. PDE/physics conclusion

Node C INTEGRATED (F1-F10, 25/25 tests, derivation certified). Node A's independent
mathematical and numerical audit (F19, reproducible) — REPRODUCED by Node C with an
independent instrumentation: archive-2's PDE residual states the correct Double
Heston pricing PDE on paper, but is INCORRECTLY IMPLEMENTED — an autograd slice-view bug
makes `grad(prices, v01/v02)` return None and `_safe_grad` silently converts it to zeros,
so all six variance-factor derivative terms are exactly zero and the implemented residual
is `d_tau - (diffusion + drift)`, a wrong PDE. Proof: implemented residual equals the
dropped factor terms exactly (difference = 0 at machine precision); correct wiring yields
-1.5e-10. Neither stack is presently a PDE-informed inverse network: the canonical stack has
no PDE term, and archive-2's is both wrong-as-shipped and — even fixed — machine-zero for an
accurate pricer, hence non-discriminating for parameters. A genuine Model 3 requires a
research decision on a fresh network-side construction (Node C verification pending).

## 6. Findings reproduced or independently supported

- Node B global ambiguity on full 108 grid: consistent with standing G2_GLOBAL_AMBIGUITY
  evidence (40 solutions / 39 clusters, different grid) — SUPPORTED/EXTENDED; fast pricer
  validated 0.0-diff vs production.
- Factor-swap degeneracy: triangulated by ALL THREE nodes (Node B bitwise 0.0 fast pricer;
  Node C 4.26e-14 production; Node A 2.842e-14 production) — REPRODUCED; and the production
  gate REJECTS the swapped vector under enforce_ordering=True, proving the ordering
  constraint is what excludes the twin.

- Torch mirror = production pricer at ~1e-15 relative, and COS = production at 1e-12..1e-9:
  three independent implementations agree on the same mathematics (Node A diagnostics A-004;
  consistent with the repo's existing independent-pricing-benchmark history).
- Parameter-order conflict: verified by direct source read AND by exact gradient permutation
  (two independent methods, same mapping).
- Archive-2 PDE-loss autograd bug: REPRODUCED by two nodes with different instrumentations
  (Node A identity proof; Node C bit-exact broken-operator equality + 9-config invariance).
- Canonical PDE/physics contract certification: Node A machine-precision mirror agreement +
  Node C derivation and multi-limit certification (REPRODUCED/SUPPORTED).
- Archive-2 constraint-space incompatibility (disk 1.805 reachable; rho box excludes
  canonical-valid vectors): computed independently by both nodes (REPRODUCED).
- Canonical training-policy cleanliness: direct trainer audit + test suite (22 tests passed)
  + explicit `real_market_data_used: False` logging in the canonical benchmark runner.

## 7. Important negative findings

- `ordering_penalty` in archive-2 is dead-by-construction (kappa2<kappa1 already guaranteed);
  lambda_order=0 in all configs regardless.
- The constraint map provides ~zero protection w.r.t. reviewed sampling ranges (0.43%/0.33%
  of Feller-accepted pilot samples lie in the map's unrepresentable boundary shell; interior
  impact negligible, boundary-challenge populations over-represent it).
- Archive-2's PDE loss is incorrectly implemented (F19): autograd slice-view bug zeroes all
  variance-factor terms; the trained "physics" signal is a wrong PDE. The smoke-run 8.91 is
  the dropped-terms magnitude, not noise.

## 8. Remaining disagreements

None between Node A and Node C. One disposition nuance (recorded, not conflicting): Node A
classifies real_finetune as REMOVE FROM CANONICAL PATH; Node C as ISOLATE AS NON-PRIMARY
ABLATION + DISABLE BY DEFAULT — both forbid a canonical named mode; ablation status is
Human Decision #1. Documented intra-repo contradiction: stale PINN status tokens in two
status docs vs accurate README (F8). Node B's own status note: "Complementary; no conflicts" with Nodes A/C — confirmed on
Node A's side after full read.

## 9. Strongest quantitative evidence

| Quantity | Value | Source |
|---|---|---|
| Torch GLQ mirror vs production, max rel (liquid region) | ~1e-15 | A-004 |
| Archive-2 COS vs production, rel range (liquid region) | 1e-12..1e-9 | A-004 |
| Constraint-map outputs inside pilot sampling box | ~0% (any raw surrogate) | A-007 |
| Median pilot-edge violation multiples | 8x (N(0,1)) to 502x (U[-50,50]) | A-007 |
| Structural guarantee hold rate at N(0,3) raws (n=200k) | 1.0000 (Feller/disk/order/positivity) | A-007 |
| Unrepresentable Feller-shell occupancy (pilot, Feller-accepted) | 0.436% slow / 0.332% fast / 0% disk | A-011 |
| Correctly-wired PDE residual, GLQ autograd / COS autograd | ~1e-14 / ~1e-12 (machine zero) | A-013 |
| Implemented (buggy) PDE residual identity | residual == dropped f1+f2 exactly (diff 0) | A-014 |
| G2 ambiguity backdrop | price RMSE 4.708e-8 vs param RMSE 1.485e-1; 39 clusters | repo doc |
| Correctly-wired PDE residual floor (Node C) | 4-5e-9 relative — same noise scale as near-equivalence price differences | Node C F12 (supports: physics = regularization, not identification) |

## 10. Code worth considering for merge

Nothing tonight (protocol: no merges). After review, consider FROM Node A's branch:
- `artifacts/canonical_archive2_adapter.py` (adapter promotion, with tests) — commit will be
  finalized in §15.
- The seam matrix + fairness contract as docs.
From archive-2 (already on genesis; requires quarantine decisions first):
- `verify_zero_leakage` + chronological split pattern (re-homed on canonical ids);
- COS pricer as an independent cross-check harness.

## 11. Changes that should NOT be merged

- Any path exposing `real_finetune` / `--continuous` real-market weight updating as a
  canonical training mode (research-control violation; exercised once at smoke scale).
- Archive-2's `pde_residual_loss` in its current form (F19 bug — silently wrong PDE; fix
  would still be non-discriminating for an accurate pricer).
- Archive-2 box constraints / negative-only rho as canonical constraint semantics.
- Positional (adapter-free) cross-stack parameter passing.

## 12. Research gate status

- G2: UNCHANGED (NOT PASSED; no representation frozen; no grid chosen tonight).
- G8/real-market freeze: UNCHANGED (policy violation found and quarantined, not exercised).
- PINN milestone: NEW EVIDENCE ONLY (infrastructure classification clarified; no training).
- Mentor gate (G2 safeguard + NTPC temporal study): UNCHANGED, still the immediate blocker.

## 13. Recommended next five actions

1. Mentor decision on the G2 safeguard + bounded NTPC temporal study (existing gate).
2. Human review + adoption of the seam matrix; extend `docs/ARCHITECTURE.md` accordingly.
3. Quarantine/remove `real_finetune` + `--continuous` from canonical entry points (Decision #1).
4. Docs reconciliation PR: two-axis PINN status vocabulary (Decision #2).
5. Representation-interface design (informed by archive-2's variable-length pattern) BEFORE
   any 10k generation, respecting the F16 shell disclosure for challenge populations.

## 14. Human/mentor decisions required

1. Approve quarantine plan for real-market fine-tuning paths (policy violation, evidenced).
2. Approve two-axis PINN status vocabulary + docs PR.
3. Decide reviewed-box vs structural-only constraint policy (quantified OOD reach, A-007).
4. Decide whether Model 3 (PDE-informed) enters the research plan (Node C input pending).
5. Approve the permutation adapter as the only sanctioned cross-stack interop mechanism.

## 15. Git references

- Node A: `overnight/20260822-a-architecture` @ (final SHA at consolidation) — evidence in
  `.ai-research/overnight/2026-08-22/node-A/` (STATUS, FINDINGS F1–F17, EXPERIMENTS A-000–
  A-012, SEAM_MATRIX, FINAL_REPORT, 2 diagnostics + adapter artifact).
- Node B: `origin/overnight/20260822-b-identifiability` @ 77b8f2e (landed 03:11 IST,
  mid-run: STATUS/FINDINGS(Phase D)/EXPERIMENTS + artifacts/tables; Phases A/B results in
  STATUS checkpoint; FINAL_REPORT pending — re-polled before consolidation).
- Node C: `origin/overnight/20260822-c-pde` @ 751551a (4 commits; STATUS/FINDINGS/
EXPERIMENTS/FINAL_REPORT + derivations, tables, probe evidence, 25/25 focused tests).
Integrated at 01:30 IST without merging.

If Node B/C evidence lands before consolidation, sections 4/5/8/15 will be updated with
REPRODUCED/SUPPORTED/PRELIMINARY/DISPUTED/UNRESOLVED labels per protocol.
