# Node A Findings — Overnight 2026-08-22

Role: COORDINATOR_ARCHITECTURE. Scope: diagnostic + architecture + mathematical audit.
All results below are PROVISIONAL EVIDENCE unless they satisfy an existing documented gate.

---

## F1. Architecture map (verified read-only)

The repo contains two partially incompatible PINN stacks, confirmed at file level:

- **Stack A (canonical-aligned)**: `models/pinn_model.py` (`PhysicsInformedInverseCalibrator`,
  `DoubleHestonConstraintMap`), entry `src/train_pinn.py` + `src/run_pinn_*.py`,
  differentiable pricer `src/torch_double_heston.py`. Fixed 108-feature surface input via
  `src/dataset.py`. "Physics" loss = differentiable repricing consistency (no PDE residual).
- **Stack B (archive-2 import)**: `src/dheston/` package + root entry `train_double_heston.py`.
  COS-method pricer (`src/dheston/pricing/heston.py`), genuine PDE-residual loss
  (`src/dheston/models/losses.py`), variable-length point-cloud surfaces
  (`src/dheston/data/surfaces.py`), own parameter transform
  (`src/dheston/calibration/transforms.py`).

Candidate incompatibilities (from exploration, to be verified directly in Phase 1):

1. Parameter order: canonical `[kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow, ...]`
   (src/constants.py PARAMETER_NAMES) vs dheston `[v0, kappa, theta, sigma, rho] x 2` with
   factor 1 = fast (kappa2 <= kappa1). Positional cross-stack tensor passing silently swaps
   slow/fast roles.
2. Constraint semantics: Stack A softplus/Feller-margin/correlation-disk vs Stack B sigmoid
   box bounds with rho in (-0.95, -0.05) (negative-only, no Feller).
3. Physics-loss semantics: repricing consistency (A) vs PDE residual (B).
4. Pricing formulation: Gauss-Laguerre Little-Heston-Trap (A, mirrors frozen engine) vs
   COS with damping alpha=1.5 (B).
5. Surface representation: fixed 108-vector (A) vs variable point cloud from NSE rows (B).
6. Test coverage asymmetry: Stack A has 3 test files; Stack B only a pricing smoke test.
7. Stack B `ordering_penalty` targets theta ordering, not kappa ordering; default weight 0.

Verification status: **COMPLETE (00:57 IST)** — each candidate verified directly against source.

---

## F2. Phase 1 verification results (all seven candidates)

| # | Candidate | Verdict | Key evidence |
|---|---|---|---|
| 1 | Parameter order conflict | **CONFIRMED (double permutation)** | `src/constants.py:7-18` canonical `[kappa,theta,sigma,rho,v0]x2` slow-first; `src/dheston/calibration/transforms.py:9-20` `[v0,kappa,theta,sigma,rho]x2` with factor1=fast (`kappa2<=kappa1`, lines 57, 90). Cross-stack positional passing swaps slow/fast AND reorders within factors. A pure permutation adapter is sufficient — no rescaling needed. |
| 2 | Constraint semantics | **CONFIRMED (different semantic spaces)** | Stack A `models/pinn_model.py`: hard-by-construction softplus positivity, `kappa_fast = kappa_slow + softplus` (line 33), Feller ceiling `sigma = 0.995*sqrt(2*kappa*theta)*sigmoid` (lines 130-136), polar correlation disk `rho_s^2+rho_f^2 < 0.995^2` (lines 139-154). Stack B `transforms.py:22-33`: sigmoid box bounds; rho in (-0.95,-0.05) — **negative-only**; **no Feller** (sigma up to 1.5 with 2*kappa*theta as low as 0.006 possible); **no joint disk** (rho1=rho2=-0.95 gives disk sum 1.805 > 1 → invalid correlation matrix reachable). Neither space contains the other. |
| 3 | Physics-loss semantics | **CONFIRMED + deepened** | Stack A: no PDE term; physics = differentiable repricing consistency. Stack B `losses.py:78-134`: well-formed Double Heston PDE residual (verified term-by-term vs standard form: diffusion, drift, per-factor vol terms all correct). **BUT** residual is evaluated on prices from the spectral pricer itself, not on a network solution surface — so it measures pricer discretization/autograd error, and its gradient wrt network weights is near-vacuous when the pricer is accurate. Stack B's "PDE-informed" loss is architecturally a pricer-consistency check, not a classic PINN constraint. |
| 4 | Pricing formulation | **REFINED (same math, different inversion)** | Both use Heston characteristic-function ("Little Heston Trap") factor terms. Stack A `src/torch_double_heston.py`: Gauss-Laguerre quadrature mirroring frozen production engine. Stack B `src/dheston/pricing/heston.py`: COS method, Fang-Oosterlee truncation via analytic cumulants (`_cos_truncation_range_*`, lines 251-396), damping alpha=1.5. Same model mathematics; different discretization error profiles. |
| 5 | Surface representation | **CONFIRMED** | Stack A `src/dataset.py`: fixed 108-vector (= 9 log-moneyness x 6 maturities x 2 option types, `src/constants.py:40-51`) with masks, surface_ids, metadata. Stack B `src/dheston/data/surfaces.py`: variable-length point clouds from real option rows, padded batches, per-point masks. |
| 6 | Test coverage asymmetry | **CONFIRMED** | Canonical: 29 test files incl. `test_pinn_forward.py`, `test_pinn_training.py`, `test_torch_double_heston.py`, `test_parameter_order.py`, `test_constraints.py`. Stack B: exactly one (`test_archive2_pricing_smoke.py`). |
| 7 | ordering_penalty targets theta | **CORRECTED** | `losses.py:44-45` penalizes `params[:,6]-params[:,1]` = kappa2-kappa1 — it DOES target kappa ordering. However `networks.py:36` returns constrained params and `transforms.py:90` already guarantees kappa2<kappa1 by construction, so the penalty is **identically zero always** (dead term); `lambda_order=0.0` in all configs regardless. |

## F3. Training-policy finding (Phase G, early but decisive)

`train_double_heston.py` exposes three loss modes: `ordinary`, `physics`, **`real_finetune`**
(`configs/default_experiment.json:46-68`). The `real_finetune` mode sets `lambda_param=0.0`,
`lambda_price=1.0`, `lambda_pde=0.05` — i.e. **updates network weights directly on real market
prices** (line 356-374 builds real NSE surface records, chronological splits). A `--continuous`
flag (line 46) trains "continuously on real data until manually stopped" with checkpoint
auto-resume. This **directly violates** the canonical research control that real-market
observations must not update primary ANN/PINN weights.

Classification: `REMOVE FROM CANONICAL PATH` (the mode must not exist as a named top-level
training mode in any canonical entry point; historical code stays quarantined in archive import).

Secondary: Stack B synthetic records (`src/dheston/data/synthetic.py:10-40`) reuse real-market
strike/tau/spot/rate grids as templates but labels are fully synthetic (sampled params +
Stack B pricer + multiplicative noise) — a realism choice to disclose, not weight leakage;
however it couples synthetic geometry to real data availability.

## F4. Cross-stack comparability constraint

Any parameter vector used for cross-stack comparison must satisfy BOTH constraint sets
(Stack A structural validity AND Stack B box bounds incl. negative rho) — feasible but a
strict subset of each individual space. Diagnostic pricer-agreement runs must sample only
in this intersection.

## F5. Pricer agreement + adapter verification (diagnostic A-004 (run 00:59 IST), seeded, CPU)

Script: `artifacts/diag_pricer_agreement.py` (spot 100, r=0.05, q=0, taus 7d-365d,
log-moneyness -0.30..+0.30, canonical params in both stacks' constraint intersection).

- **Stack A torch GLQ mirror vs frozen production engine: max |rel| ~1e-15** across the
  liquid region (only exception a deep-OTM 7d call with price ~0, absolute diff negligible).
  The torch mirror is a faithful differentiable replica of the production pricer.
- **Stack B COS vs frozen production: |rel| 1e-12..1e-9 in the liquid region**, degrading
  only for near-zero deep-OTM prices (relative error meaningless there). COS with default
  `FourierConfig` is an equivalent independent implementation of the SAME model mathematics.
- **Parameter adapter verified by gradient permutation**: the Stack B price gradient wrt
  B-order params equals EXACTLY the Stack A gradient wrt canonical params under the
  factor-swap + within-factor-reorder mapping (slow<->factor2, fast<->factor1).
  The permutation adapter (no rescaling) is therefore numerically verified, not just inferred.

Phase E conclusion: pricing-engine differences are numerical, not mathematical. The frozen
production Gauss-Laguerre engine remains the scientific source of truth; Stack A torch mirror
is its validated differentiable image; Stack B COS is an acceptable independent cross-check
with a far-OTM tail caveat. No pricer bug suspected; no production change warranted.

---

## F6. Phase F — physics-informed classification (pending Node C confirmation)

Classification of the current canonical model: **constraint-informed + repricing-informed
inverse network** (NOT a PDE-informed PINN). Evidence: `models/pinn_model.py` supplies
structural constraint satisfaction; canonical training losses use differentiable repricing
through the torch mirror; no PDE residual exists anywhere in the canonical path.

Classification of Stack B's model: **constraint(box)-informed + repricing-informed + nominal
PDE-residual network**, where the PDE term is mathematically correct in form but operates on
the spectral pricer's own output surface (F2 #3) — it penalizes pricer/autograd discretization
error, not network physics violation. Calling it "PDE-informed" in a paper without disclosing
this would overstate the mechanism.

Recommended experimental taxonomy for the eventual paper:
- **Model 1** — ordinary ANN (existing `models/ann_model.py` baseline).
- **Model 2** — constraint + repricing-informed inverse network (current canonical PINN
  infrastructure; honest label: "physics-informed" only in the weak sense of model-consistent
  repricing).
- **Model 3** — PDE-informed inverse network. Recommend ONLY after a network-side PDE
  construction is designed and Node C-verified; importing Stack B's residual as-is does not
  achieve this. REQUIRES RESEARCH DECISION.

## F7. Phase I — fair ANN-vs-PINN fairness contract (proposal)

Held constant by construction (already satisfied by canonical seam): canonical parameter
order/targets; same `SurfaceParameterDataset` splits (index-disjoint, enforced); same surface
representation behind the interface; same production/torch repricer; train-only target
standardization; validation-gated checkpointing; identical final synthetic test set;
multi-seed (>=3) with reported spread; identical frozen-real-market evaluation stage;
parameter-recovery + repricing + validity + stability + runtime metric families reported
together. Explicit prohibitions: no model-specific real fine-tuning; no winner declaration on
repricing RMSE alone; no constraint-map advantage silently re-labeled as "physics" (Model 2
must be reported with its constraint ablation). Capacity/compute parity: report parameter
counts and wall-clock; justify differences.

## F8. Phase J — documentation contradictions

- `README.md:26,80`: "PINN infrastructure | Implemented; not research-trained" + explicit
  non-claim disclaimers — ACCURATE.
- `docs/RESEARCH_CONTROL_AND_CURRENT_STATUS.md:97`: `PINN = NOT_IMPLEMENTED_OR_TRAINED` —
  the NOT_IMPLEMENTED half is stale (infrastructure exists: `models/pinn_model.py`,
  `src/train_pinn.py`, `src/run_pinn_*.py`, `tests/test_pinn_*.py`).
- `docs/CURRENT_STATUS.md:48,165`: "PINN development/comparison | Not started" /
  `PINN = NOT_DERIVED_OR_TRAINED` — "development not started" is stale; "not derived/trained"
  remains accurate in the PDE-derivation sense.

Proposed reconciliation (for human approval; NOT applied tonight): adopt a two-axis
vocabulary — `PINN_INFRASTRUCTURE = IMPLEMENTED_NOT_RESEARCH_TRAINED` (true today) and
`PINN_RESEARCH_MILESTONE = NOT_DERIVED_OR_TRAINED` (true today) — and update the two status
docs' tokens via a reviewed docs PR. This matches the existing README terminology and avoids
implying milestone progress. Related: the project title's "Physics-Informed" should be
read aspirationally; docs should adopt the F6 taxonomy when describing Model 2.

## F9. Phase H — G2/representation coupling (verified)

No hardcoded 108 in model/trainer code: `input_size` derives from data
(`src/train.py:204`, `src/train_pinn.py:326`, `src/run_pinn_synthetic_baseline.py:61`) and
`src/surface_grid.py:86 expected_input_size()` computes from grid constants. G2 grid change
costs: constants + synthetic generator + regenerated datasets + retrained models; zero model
class changes. Stack B's variable-length masked-pooling pattern is the reference for a
future representation interface if G2 selects a non-rectangular or market-driven quote set.

## F10. Phase G — training-policy audit (complete)

Canonical path verdict: **ACCEPTABLE — synthetic-only, no leakage path found.**
- `src/train.py` (ANN): zero real-market references; synthetic only.
- `src/train_pinn.py`: enforced disjoint splits; `TargetStandardizer` fit on training rows only
  (`train_pinn.py:58-59`); best-validation checkpointing (line 81).
- `src/run_pinn_two_stage_baseline.py`: "supervised warm start then PINN fine-tune" — both
  stages synthetic (docstring line 1, structure lines 35-83).
- `src/run_pinn_improved_benchmark.py:116` logs `"real_market_data_used": False` explicitly.
- Remaining market-touching files are non-neural: `nse_stage_a.py` (candidate selection),
  `market_data_audit.py` (data audit), `calibrate_double_heston.py` (traditional optimizer
  calibration — permitted comparison arm).

Stack B verdicts (carried from F3): `real_finetune` + `--continuous` =
**REMOVE FROM CANONICAL PATH**; synthetic-on-real-grids = **ISOLATE + disclose**; box-bound
sampling = ISOLATE (archive sampling contract).

## F11. Phase D quantification — constraint-map OOD reach + fp32 edge notes (A-007)

Script: `artifacts/diag_constraint_ood.py` (n=200k per raw distribution, seeded).

**OOD reach (the reviewed-box concern, now quantified):**
- Fraction of constraint-mapped outputs inside PILOT empirical sampling ranges: **~0%**
  for every raw surrogate (N(0,1), N(0,3), N(0,10), uniform[-50,50]).
- Median violation multiples: 8-13x past the nearest pilot edge for N(0,1)/N(0,3) raws
  (theta_fast 99%@x8.7, v0_fast 99%@x7.0); up to 275-502x for uniform[-50,50].
- rho_slow spans the full (-0.995, 0.995) disk regardless of raw scale (pilot: [-0.75, 0.20]).
- Conclusion: the constraint map provides **zero protection** w.r.t. reviewed sampling-box
  membership. This is the quantified basis for the RDD: either report OOD predictions as OOD
  (recommended), or add an explicit box-compliance *reporting* layer — never silent clipping.

**fp32 edge phenomena (verified mechanisms, extreme-only):**
- raw_sigma < ~-88 → `sigmoid` underflows to exact 0.0 → sigma = 0.0 exactly (strict
  positivity marginally broken; singular vol-of-vol endpoint). Reproduced at raw = -90.
- kappa_slow ≳ 100 → `+ _POSITIVE_EPS` rounds away in fp32 → kappa_fast == kappa_slow
  exactly (strict ordering degenerates to a tie; never an inversion).
- At realistic scales (N(0,3) raws, n=200k): Feller/disk/strict-order/strict-positivity all
  hold at **exactly 1.0000**. The by-construction guarantees are robust across any plausible
  trained-network output distribution; no fix warranted tonight. Precision note for the
  future: if extreme-raw robustness ever matters, use a float64 head or a scaled margin.

## F12. Cross-stack adapter artifact (A-008)

`artifacts/canonical_archive2_adapter.py` — the documented, self-tested permutation
(canonical <-> archive-2). PROCESS NOTE (honest evidence): the first version of this artifact
had the two permutation directions INVERTED; the round-trip test alone would have passed
 silently, but the named-field spot check caught it. This is exactly why positional interop
is dangerous and why the artifact carries named-field verification. A-004 (gradient
permutation) was unaffected — that diagnostic built the archive vector field-by-field.

Final state: 10/10 named fields verified against archive-2 field names, round trip exact.
Deliberately NOT installed into src/ tonight: promotion requires human review (Decision #5).

## F13. NEXT_STEPS alignment check

`docs/NEXT_STEPS.md` confirms: the current 108-grid is already REJECTED as the final
unchanged representation; G2 NOT PASSED; the immediate gate is `OBTAIN MENTOR DECISION` on
the G2 safeguard + bounded NTPC temporal study; no ANN/PINN research work before that
decision. Node A's recommendations are therefore explicitly SUBORDINATED to that mentor
gate — the seam matrix and fairness contract define what to build once gates open. This
also strengthens Phase H: since the grid WILL change, the data-derived `input_size` seam is
the load-bearing defense against a costly rewrite.

## F14. Import-smoke forensics (experiments/import_smoke_20260821, A-009)

- Provenance: teammate machine paths (`/Users/dhruvasbamb/Desktop/archive-2/...` dataset,
  checkpoint under `double-heston-inverse-calibration-main`), updated 2026-08-21T23:33:14 —
  the archive-2 import was validated by a 1-epoch smoke run hours before this overnight.
- **The `real_finetune` stage has ALREADY EXECUTED once in this repository's imported
  artifacts** (stage_logs.real_finetune, 80 real train surfaces, 1 epoch, parameter loss
  pinned 0.0). Scale is smoke-only (no research training occurred), but the policy-violating
  path is exercised, not hypothetical. Reinforces REMOVE FROM CANONICAL PATH + the need for
  an explicit quarantine decision (Human Decision #1).
- PDE-term behavior at init: `train_pde = 8.91` vs `train_parameter = 0.104`; validation
  computes `valid_pde = 0.0` because validation runs with `pde_points=0`
  (`train_double_heston.py:257`). A residual of ~9 (scale-normalized, squared) at exact
  pricer outputs is direct numerical evidence that autograd second derivatives through the
  256-step COS integrator are noise-dominated — supporting F2 #3 (the PDE term does not
  carry usable physics signal for the inverse network in its current form).
- `configs/default_experiment.json` and `configs/archive2_default_experiment.json` are
  identical (byte-equal after normalization); `archive2_smoke_experiment.json` only shrinks
  caps/epochs. No hidden divergence between the two configs.

## F15. Independent verification round (A-010, 01:10-01:15 IST)

- `verify_zero_leakage` (`src/dheston/data/surfaces.py:138-157`): assert-based, dual-keyed
  (trade dates AND surface keys) — ADAPT recommendation for the frozen real-market evaluation
  stage confirmed sound; must be re-homed on canonical surface_ids.
- Archive-2 traditional calibration (`src/dheston/calibration/optimize.py`): scipy multistart
  on COS prices, box bounds, kappa-ordering via 1e6 barrier. Redundant with canonical
  `src/calibrate_double_heston.py` — classification ISOLATE (canonical owns the comparison arm).
- `src/pricing_interface.py`: research callers ALWAYS reach the frozen engine; the dummy
  pricer is a separately named smoke-test function that cannot be selected implicitly —
  no dummy-pricer leakage risk in the canonical path. KEEP confirmed.
- `src/double_heston_reference.py`: independent SciPy adaptive-quadrature benchmark that
  deliberately does not import the production pricer — true independence confirmed.
  Phase E hierarchy level 2 verified.
- Full canonical PINN/ANN test sweep: `test_pinn_training.py`, `test_torch_double_heston.py`,
  `test_ann_forward.py` — 7 passed (on top of the earlier 15).
- Determinism: diagnostic A-004 re-run twice — bit-identical outputs, same max-rel values.

## F16. Constraint-map representable set is strictly smaller than the declared set (A-011)

The map's safety margins carve a thin shell out of the declared constraint set:
sigma_i < 0.995*sqrt(2*kappa_i*theta_i) (vs declared >0 Feller) and disk radius < 0.995
(vs declared <1). Targets inside these shells are unreachable exactly (approached only
asymptotically by sigmoid/softplus heads), imposing a recovery-error floor that is a
property of the MAP, not the network.

Quantified on 501,212 Feller-accepted pilot-box samples (seed 42):
- slow-factor Feller shell: 0.436% · fast-factor shell: 0.332% · disk shell: 0.000%.
Interior training populations: negligible. **Boundary-challenge populations (which
deliberately hug boundaries) can over-represent the shell** — recovery metrics on those
populations must disclose this floor. Suggests either (a) report shell-membership of
challenge targets, or (b) a research decision on margin width. No code change tonight.

## F17. Documentation/architecture alignment + ambiguity-case representability (A-012)

- `docs/ARCHITECTURE.md` is CONSISTENT with the seam matrix: it documents the 108 contract,
  exact parameter order, declared constraints, pricing-interface research boundary, and the
  honest PINN-infrastructure framing. It does not cover the archive-2 stack — the seam matrix
  extends it. Recommendation: after human review, extend ARCHITECTURE.md with the archive-2
  seam section (adapter rule, quarantine classifications).
- G2 global-ambiguity backdrop (for Node B integration): 40 clean near-equivalent solutions /
  39 materially displaced clusters; median normalized price RMSE 4.708e-8 vs range-scaled
  parameter RMSE 1.485e-1; median nearest separation 0.2769. This is the identifiability
  context any Node B finding will extend.
- Cross-check: all four predeclared G2 ambiguity cases lie INSIDE the constraint map's
  representable set (e.g. case_4 sigma_fast 0.9116 < ceiling 0.9769). The F16 shell issue
  does NOT taint the existing G2 ambiguity evidence.

## F18. Archive-2 evaluation metrics note (completeness)

`src/dheston/evaluation/metrics.py`: raw-scale per-parameter MAE/RMSE only — NO range
normalization. Canonical recovery metrics (e.g., range-scaled RMSE used throughout the G2
analyses) normalize per parameter range; archive-2's smoke metrics (kappa1_rmse 2.41 etc.)
are therefore not directly comparable to canonical numbers without rescaling. Any
cross-stack metric comparison must state the normalization convention explicitly.

## F19. ARCHIVE-2 PDE RESIDUAL IS INCORRECTLY IMPLEMENTED — autograd slice-view bug (A-013/A-014)

**This finding CORRECTS the mechanism claims in F2 #3 and F14.** The earlier
"noise-dominated through the COS integrizer" reading was wrong; the provisional conclusion
was re-tested adversarially (as the role requires) and the truth is sharper:

**Bug.** `pde_residual_loss` (`src/dheston/models/losses.py:121-126`) differentiates wrt
`v01 = chosen_params[:, 0]` / `v02 = chosen_params[:, 5]` — slice views that are NOT on the
executed autograd graph (the pricer consumes parameters via `_broadcast_parameter_grid_*`
reshape). `torch.autograd.grad(prices, v01)` returns None (allow_unused=True), and
`_safe_grad` converts None into `torch.zeros_like` **silently**. All six variance-factor
derivative terms (d_v0, cross S-v0, second v0) are therefore EXACTLY ZERO in every
evaluation. The implemented residual is actually `d_tau - (diffusion + drift)` — a WRONG
PDE missing both factor terms.

**Proof chain** (`artifacts/diag_pde_bug_repro.py`, seeded, deterministic):
1. Repo loss on a minimal synthetic put batch: 0.5319 in float32 AND float64 (dtype/real-data
   independent; earlier hypotheses — fp32 precision, real-market geometry, batch mixing,
   parameter-shape — were each tested and ELIMINATED).
2. Mechanism: `v01.requires_grad=True` (guard passes) but `grad(prices, v01) = None` while
   `grad(prices, parameters)[:,0] = 41.36` (healthy).
3. Identity: implemented residual per point [9.581, 5.003, -3.453] equals the true (dropped)
   f1+f2 per point EXACTLY; residual − (f1+f2) = 0 to machine precision.
4. Correct wiring (differentiate the leaf, then slice): residual = -1.5e-10 — machine zero.

**Interpretation.** (a) The smoke-run's `train_pde = 8.91` is the magnitude of the DROPPED
terms, not autograd noise. (b) Even correctly implemented, the residual is machine-zero for
an accurate pricer (GLQ autograd ~1e-14; COS autograd ~1e-12) — so it carries NO
parameter-discriminating physics signal for the inverse network: the F2#3 "architecturally
vacuous" conclusion STANDS and is strengthened. (c) As shipped, the term is worse than
vacuous — it is a silently-wrong PDE whose parameter gradients flow only through
d_tau/delta/gamma.

**Classification change (seam matrix row 9):** archive-2 PDE-loss code moves from
"ADAPT-concept pending Node C" to **DEPRECATE (bug; do not import)**. A genuine Model 3
must be designed fresh with correct differentiation seams and Node C verification.

Pending: Node C independent mathematical audit (this finding is reproducible on demand).

## F20. Node C integration (branch landed 01:24 IST; read without merging)

Node C completed a full PDE/physics audit (10 findings, 25/25 focused tests, derivation
doc, probe evidence, final report). Convergence with Node A:

| Claim | Node A | Node C | Label |
|---|---|---|---|
| Archive-2 PDE loss broken: all variance-state derivatives exactly zero (slice views not graph ancestors; `_safe_grad` None->zeros) | F19 (A-014; identity residual == dropped terms at machine precision) | F2 [PROVEN] (bit-exact equality with manually-assembled broken operator; 9-config invariance; market-price invariance through full loss path) | **REPRODUCED (two independent instrumentations)** |
| Correctly-wired residual is quadrature/machine-noise level -> non-discriminating | F19 Part 3 (~1e-14 GLQ / ~1e-12 COS) | F3 (4.8e-9/3.8e-9 relative vs 7.3e-2/2.2e-1 broken; truncation-range autograd contamination ~1e-12 negligible) | **REPRODUCED** |
| Canonical stack = constraint + repricing-informed, NOT a PINN | F6 | F4 [PROVEN] incl. honest checkpoint metadata | **REPRODUCED** |
| Canonical physics/PDE contract sound; torch mirror satisfies PDE to machine precision | A-004 (~1e-15 vs production) | F1 [PROVEN] (<=1.3e-15; put-call parity 7e-15; CF additivity; two-half-factors==1-factor vs COS 3.2e-11; BS limit) | **REPRODUCED/SUPPORTED** |
| Archive-2 constraint space semantically incompatible; neither admissible set contains the other | F2#2/F4 (disk 1.805 reachable; negative-only rho) | F5 [PROVEN] (same 1.805; rho box excludes canonical-valid rho_f > -0.05) | **REPRODUCED** |
| real_finetune + --continuous violate research control | F3/F10/F14 (executed once at smoke scale) | F7 [PROVEN] | **REPRODUCED** (disposition nuance below) |
| default_experiment.json == archive2_default_experiment.json | F14 | F9 + NEW: `dheston/config.py` loads the former as ITS default — repo-level "default" IS the archive-2 config (naming hazard) | **REPRODUCED + EXTENDED** |

New Node-C-only evidence incorporated: F6 (validation objective mismatch — validation
disables PDE, hiding the F2 bug from model selection); F8 (a same-model PDE residual
cannot add identifying information beyond repricing — regularisation/validity, not
identification); F10 (canonical pricer ultra-short-maturity boundary: negative deep-OTM
prices below tau ~ 5e-3; benign within the research grid, worst -6.3e-12 at >= 7 days).

One disposition nuance (not a disagreement): Node A classifies real_finetune as
REMOVE FROM CANONICAL PATH; Node C recommends ISOLATE AS NON-PRIMARY ABLATION +
DISABLE BY DEFAULT. Compatible: both forbid a canonical named mode; the ablation
question is Human Decision #1.

No contradictions found between Node C evidence and Node A findings. Node C final
SHA at integration: 751551a.

Cross-verification (A-015): Node C's 25-test suite extracted read-only and run under
Node A's local pytest — 25/25 passed (initial 4 failures were /tmp path artifacts).
Node C evidence is independently reproduced end-to-end: derivation, bug, constraints,
limits. Their F10 boundary also spot-verified locally (-0.013/-0.089 at tau 1e-3/1e-4).

## F21. Node A mathematical review of Node C's derivation (requested by Node C)

Reviewed `node-C/derivations/CANONICAL_DOUBLE_HESTON_PDE.md` line-by-line. VERDICT: correct
and complete. The SDE reverse-engineering from the Little-Heston-Trap CF (4-BM construction,
independent variance drivers, additive spot variance) is properly evidenced from source; the
Itô table and backward/forward PDE match the operator I verified numerically to machine
precision (A-013: GLQ autograd residual ~1e-14). The factor-additivity identity
E(u,T;θ,v0)=2E(u,T;θ/2,v0/2) at 1.8e-15 is an elegant independent signature.

Adopted refinement to Node A Phase D: Node C §1.3 shows the canonical correlation DISK
(rho_s²+rho_f²<1) is a SUFFICIENT, conservative, state-independent condition for the
IMPLEMENTED 4-BM model — pointwise admissibility only requires |rho_i|<1. The disk is the
PD condition of the single-spot-driver 3-BM construction. Implication: the constraint map's
polar disk enforces the CONTRACT (a deliberate conservative choice), not bare mathematical
necessity — supports KEEP classification, and any future relaxation is a research decision,
not a bug fix.

## F22. Canonical-path adversarial clearance (completing the failed subagent's checklist)

The adversarial-review subagent failed twice on network errors; its checklist was completed
by direct Node A verification (each item independently checked):
- Validation/objective parity in `train_pinn.py`: `_evaluate_validation` receives the SAME
  `parameter_loss_weight`/`physics_loss_weight` as training (lines 118-125); selection on
  validation total; best-state checkpoint; test untouched. Node C F6 canonical claim
  INDEPENDENTLY VERIFIED.
- `run_pinn_synthetic_baseline.py`: split assignment via `assign_surface_splits`, explicit
  `_assert_no_leakage(dataset, train, validation, test)` guard (line 59); test indices used
  only for final evaluation artifacts. No leakage path.
- Node C parameter-contract matrix cross-checked: their adapter mapping is IDENTICAL to
  Node A's (F12), independently derived and verified on both sides (REPRODUCED).
Canonical path: no leakage, no objective mismatch, no dummy-pricer fallback, no real-market
training, enforced disjoint splits. KEEP classifications stand.

## F23. Node C refresh integration (0254c92, 01:41 IST) — mutual verification round

Node C ran their own adversarial-review pass (independent reviewer re-derived math and
re-executed decisive experiments; all load-bearing claims CONFIRMED). Changes material to
Node A records:

1. **CORRECTION TO NODE A F2#4 / seam matrix row 8**: `FourierConfig.alpha` (the "damping
   alpha=1.5" I cited), `u_max`, and `integration_eps` are DEAD config fields — never
   consumed by the COS pricer (only integration_steps, truncation_scaler,
   min_truncation_width are used). The series saturates via exponential CF decay below
   float64 resolution, not a frequency cap. Latent operational trap: editing dead fields
   changes nothing.
2. Certification domain stated (F1): 8-point machine precision (<=1.3e-15) + 120-point
   sweep bound <=1.6e-8 relative (worst short-maturity OTM corners, improving with
   maturity — same quadrature-degradation pattern as F10).
3. F2 mechanism sharpened: pricer creates its own SelectBackward nodes internally
   (heston.py:156-157); losses.py views are CHILDREN of the output node — structural,
   no workaround in the current design.
4. **Node C verified Node A claims**: F16 constraint shells CONFIRMED (saturated heads
   reach exactly the 0.995 caps asymptotically); F18 metrics normalization CONFIRMED.
   Mutual verification now bidirectional.
5. Connective insight (morning-report worthy): Node C's correctly-wired residual floor
   (4-5e-9) sits at the same numerical-noise scale as the G2 near-equivalence price
   differences — empirical support that a PDE-residual term cannot distinguish
   near-equivalent parameter vectors (physics = regularization/validity, not identification).
6. Repo-wide autograd scan: the `_safe_grad`/allow_unused zero-silencing defect class is
   UNIQUE to archive-2; canonical stack has no such pattern (all .detach() uses are
   validation guards). Latent minor: torch factor exponent early-returns a graph-free zero
   at maturity == 0.0 (non-differentiable point at the terminal boundary; LOW severity).

## F24. Node C F15 adopted into identifiability backdrop (87f238c)

Deterministic variance-mean propagation is materially price-inconsistent: substituting
propagated conditional means as deterministic states understates prices by -0.15% (1d) to
-14.5% (90d) at tau=1.0 (Node C probe). Implications adopted by Node A: (i) per-date v0
fitting (the NTPC multi-date design) is correct — propagate-then-price shortcuts would
inject double-digit percent errors at quarterly horizons; (ii) `propagate_variance_state`
must stay an initialisation/visualisation utility (currently unused by calibration —
verified by Node C). Cross-interpreter datapoint: repo canonical-engine tests 34/36 on
py3.9 with 2 explained environment artifacts (vs 25/25 of the focused suites on py3.13
here).
