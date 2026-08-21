# Node C Findings — PDE & Physics Contract Audit

Overnight 2026-08-22. Branch `overnight/20260822-c-pde` from genesis
`642702e6706a3d17b3031619f35bda39bc144483`. All numerical evidence:
`tests_evidence/probe_residuals.py` -> `probe_results.json`; pytest suite
`tests/test_node_c_pde_physics_audit.py`. Derivation:
`derivations/CANONICAL_DOUBLE_HESTON_PDE.md`.

Confidence labels: [PROVEN]=code+numerics, [DERIVED]=math from repo SDE with
numerical confirmation, [INFERRED]=structure argument, [UNVERIFIED].

---

## F1. Canonical physics contract is sound and numerically certified across a
stated domain [PROVEN]

The canonical production model is the affine two-spot-driver Double Heston
(independent variance factors, additive total spot variance, per-factor
leverage correlations). The derived canonical pricing PDE (see derivation) is
satisfied by the differentiable Gauss–Laguerre torch pricer to machine
precision at moderate collocation points: |relative residual| <= 1.3e-15 over
8 parameter/point combinations (tau in {0.25, 0.5, 1.0}, K in 90-110, S=100).
A broadened sweep (120 points: S in {80,100,120}, K/S in {0.85..1.15},
tau in {0.1, 0.5, 1.0, 2.0}, both vectors) bounds the relative residual by
1.6e-8, worst at short-maturity OTM corners and monotonically improving with
maturity (1.6e-8 @ tau=0.1 -> 2.6e-9 @ tau=2.0) — the same quadrature-degradation
pattern as F10, still 5-6 orders of magnitude below any perturbation signal
(`probe_extension_results.json: certification_sweep`). Sensitivity controls:
zeroing the two rho cross terms raises the relative residual to 6e-3..1.7e-1
(>= 7 orders of magnitude above); halving the cross coefficient produces
EXACTLY half the perturbed residual (internal consistency of the assembly).
Limiting cases all pass: put-call parity 7e-15; exact CF factor-additivity
identity 1.8e-15; two-identical-half-factors == single Heston cross-validated
against the independent COS pricer to 3.2e-11; deterministic-variance
Black–Scholes limit within 4e-4 relative, converging at O(sigma^2) (F11).

**The canonical pricer, torch mirror, and derived PDE are mutually consistent.
Nothing tonight needs changing in the production physics.**

## F2. Archive-2's PDE-residual derivative machinery is structurally broken:
all variance-state derivatives are exactly ZERO [PROVEN — highest impact]

`src/dheston/models/losses.py:96-126` computes
`chosen_params = parameters[surface_index]` (line 96), prices through the COS
pricer (line 104), and only THEN creates column views
`v01 = chosen_params[:, 0]`, `v02 = chosen_params[:, 5]` (lines 110, 115) and
differentiates w.r.t. them (lines 121-126). torch.autograd matches GRAPH
NODES, not tensor semantics: these fresh views are not ancestors of the pricer
output, so `torch.autograd.grad(prices, v01)` reports the input as UNUSED.
`_safe_grad` (lines 62-75) has `allow_unused=True` and converts that None into
`torch.zeros_like` — silently.

Instrumented reproduction (`archive2_derivative_instrumentation`):
- losses.py's `d_v01, d_v02, cross_sv01/2, d2_v01/2` are ALL exactly 0.0;
- the TRUE gradient `dV/dv01` (w.r.t. the actual ancestor `parameters`) is
  28.49 (VEC_A) / 13.15 (VEC_B) — two orders of magnitude of missing signal;
- consequence (`archive2_effective_residual`): residual == residual-without-
  v-terms EXACTLY (0.604568934256152 vs 0.604568934256152; 2.245350358320797
  vs 2.245350358320797).

**The PDE residual actually penalised is
`V_tau - [ 0.5(v1+v2) S^2 V_SS + (r-q) S V_S - r V ]`
— a Black–Scholes-type operator with ALL variance dynamics missing — evaluated
on the COS pricer's own output.** The published smoke-run value `train_pde:
8.9` (experiments/import_smoke_20260821/metrics.json) is qualitatively
consistent with a wrong-operator O(1)-scale signal (exact reproduction not
possible: the smoke dataset lives on another machine; single-point broken
losses here measure 0.005-0.05). Classification:
INCORRECT (implementation), independent of the residual's conceptual problem
(F3).

Mechanism sharpened by adversarial review (independently re-executed): the
pricer consumes the WHOLE 2-D `chosen_params` and creates its own internal
SelectBackward nodes (`heston.py:156-157` slices `parameters[..., 0]`,
`[..., 5]`); the losses.py views are CHILDREN (not ancestors) of the output
node. No mechanism — storage aliasing, grad_fn accumulation, or object reuse —
can ever route gradient into a post-hoc view; a view created even BEFORE the
forward pass also receives zero if never consumed. Without `allow_unused`,
`torch.autograd.grad` raises; with it, None -> zeros (`losses.py:73-74`). The
defect is structural and has no workaround inside the current design.

Closure strengthened (extension probe, `probe_extensions.py` ->
`probe_extension_results.json`):
- the PRODUCTION `pde_residual_loss` equals the manually-assembled
  broken-operator loss BIT-EXACTLY on identical single-point batches
  (0.005275259112477837 == 0.005275259112477837, VEC_A;
  0.05021953002489268 == 0.05021953002489268, VEC_B);
- the zero-derivative defect is invariant across 9 (spot, tau) configurations
  spanning S in {70,100,130}, tau in {0.1,0.75,2.0}, for both vectors;
- through the full `build_loss_components` path: the PDE component is
  independent of `market_price` (bit-identical when market prices are scaled
  7x, while the price component moves) — the residual is a pure self-referential
  term.

Note for Node B/A: the same broken pattern makes the PDE term a source of
bias/noise gradients toward whatever reduces a wrong operator's residual,
e.g. pushing toward BS-like regions of parameter space.

## F3. Even if F2 were fixed, the Archive-2 "PINN" differentiates the wrong
object — now QUANTIFIED [PROVEN]

The differentiated function is the analytic COS pricer
(`price_double_heston_torch`, losses.py:104), not a learned solution
`V_theta(S, v1, v2, tau)`. Because the analytic pricer satisfies the model PDE
up to quadrature error for ANY parameter vector, a correct residual would be
numerically ~0 regardless of parameters — carrying NO identifying information
about `surface -> params`, and market prices never enter the residual (proven
invariant under market-price rescaling, see F2 closure). Quantification
(`cos_correct_wiring_vs_broken`): with the SAME COS pricer and the SAME points
but correctly-wired leaf v0 inputs, the full canonical PDE residual is
4.8e-9 / 3.8e-9 (relative) — 7-8 orders of magnitude below the broken
operator's 7.3e-2 / 2.2e-1. So a fixed implementation would measure pure
quadrature noise; today's large values are defect signal, and after a fix the
term would be information-free. A genuine PDE-informed term requires a learned
price network with leaf state inputs. (Secondary caveat now measured as
NEGLIGIBLE: differentiating through the cumulant-dependent COS truncation
range shifts autograd delta by only ~1e-12 vs the fixed-node Gauss-Laguerre
pricer at the tested points — `delta_contamination`.)

## F4. Canonical current stack classification: constraint-informed +
differentiable-repricing-informed inverse network — NOT a PINN [PROVEN]

`models/pinn_model.py` maps a 108-feature spot-normalised surface to 10
canonical parameters through a hard structural constraint map (softplus
positivity, Feller-capped sigma, polar correlation disk, additive kappa
ordering — `models/pinn_model.py:22-53,126-154`).
`src/train_pinn.py:91-111` trains with parameter MSE (z-scored) + repricing
MSE through the differentiable Gauss–Laguerre pricer. There is NO PDE
residual anywhere in the canonical path; checkpoint metadata honestly records
`"physics_loss": "differentiable_double_heston_repricing"`. Scientific label:
**constraint + differentiable-repricing informed inverse model**. The "PINN"
name should not be used for this stack in the paper without qualification.

## F5. Archive-2 constraint contract is semantically incompatible [PROVEN]

Archive-2's hard output space (sigmoid boxes, `transforms.py:22-94`) has no
Feller condition and no correlation disk; its own valid extreme emits a vector
that the canonical contract rejects with THREE violations (Feller gaps
-2.248 slow / -2.244 fast; disk 1.805 > 1) — see `archive2_constraint_gap` and
`tables/PARAMETER_CONTRACT_MATRIX.md`. Also, its rho box (-0.95, -0.05)
excludes canonical-valid vectors (rho_f > -0.05 incl. 0), so neither
admissible set contains the other. Order/index mapping alone (double
transposition: v0-position + fast/slow role swap) is an ADAPTER issue; the
admissible-set conflict is SEMANTIC.

## F6. Validation/model selection excludes the PDE loss in Archive-2
[PROVEN] — objective mismatch, currently moot

`train_double_heston.py:253-257` validates with `pde_points=0` inside
`torch.no_grad()`; `_safe_grad` additionally zeroes under no_grad. Smoke
artifact shows `train_pde: 8.9, valid_pde: 0.0`. Selection on
`valid_mean["total"]` therefore optimizes a different objective than training
(param+price+boundary only). If a genuine PDE term is ever adopted, this
mismatch must be resolved or justified; today it merely hides F2 from
validation. Canonical `train_pinn.py` has NO such mismatch (same param+physics
weights on validation; selection metadata records validation-only selection,
test untouched).

## F7. Real-market fine-tuning in Archive-2 violates the canonical research
control [PROVEN]

`train_double_heston.py:517-580` fine-tunes ALL `DeepSurfaceInverseModel`
weights on real surfaces (`real_finetune` weights: lambda_param 0, lambda_price
1.0, lambda_pde 0.05, lambda_boundary 0.1; fresh Adam at half LR;
`--continuous` trains indefinitely). Canonical control (research-control
docs): real-market observations must NOT update primary ANN/PINN weights.
Chronology/leakage hygiene is otherwise good (chronological splits by trade
date, `verify_zero_leakage`, synthetic templates per split). Disposition
recommendation: **ISOLATE AS NON-PRIMARY ABLATION + DISABLE BY DEFAULT** for
the canonical path (it is a separate stack anyway); do not present as the
primary model. The canonical NSE/NTPC scripts are scipy least-squares
calibration of the production pricer — NOT neural fine-tuning — and do not
violate the control.

## F8. Interaction with identifiability (for Node B) [DERIVED]

A PDE residual derived from the SAME model that generates the training
surfaces cannot add identifying information about parameters beyond what the
repricing loss already encodes: both constrain the same map. For an inverse
network, PDE collocation can (a) regularize the solution manifold, (b) enforce
structural validity of a LEARNED pricing function, (c) break ties among
observationally equivalent parameter vectors only by injecting an artificial
preference — it cannot resolve structural non-identifiability. If Node B
confirms near-equivalent parameter vectors, the paper's claim about "physics"
should be regularisation/validity, not identification. Tonight's evidence
(F2/F3) additionally shows the current implementation provides neither.

## F9. Minor observations [PROVEN, low severity]

- `configs/default_experiment.json` is byte-identical to
  `configs/archive2_default_experiment.json` and `dheston/config.py` loads the
  former as its default — the repo-level "default" config IS the Archive-2
  config; naming hazard for Node A.
- COS `integration_steps`: 64 vs 256 changes the price in the 4th decimal
  (8.323761 vs 8.323844); 256 vs 1024 is bit-identical. CORRECTED (adversarial
  review): this saturation comes from exponential decay of the CF series terms
  below float64 resolution — NOT from a frequency cap; `FourierConfig.u_max`
  is a DEAD field (see F13).
- Archive-2 stack provenance: single commit `642702e` "Add PINN implementation
  from Desktop"; dataset path in config points to another machine's Desktop
  (does not exist locally); no docs mention the stack; not re-runnable as
  configured. De facto experimental/archive.
- `src/double_heston.py` uses `enforce_ordering=False` escape hatch "only for
  factor-symmetry diagnostics" — the two-identical-half-factors reduction test
  legitimately uses it.

## F10. Numerical-accuracy boundary of the canonical pricer at ultra-short
maturities [PROVEN, LOW severity, no action required for the research grid]

Terminal-condition sweep (`probe_extensions.py: terminal_condition`):
- ITM calls converge to the discounted payoff as tau -> 0 (error -0.11 at
  tau=1e-4, shrinking with tau) — correct.
- ATM calls approach the payoff at rate O(sqrt(tau)), as diffusion theory
  requires (value ~ 0.4 S sqrt(v tau)) — the slow approach is mathematical,
  not a defect.
- Deep-OTM calls at tau <= 1e-3 can return small NEGATIVE prices (down to
  -0.091 at K=105, tau=1e-3): the 64-node Gauss-Laguerre rule under-resolves
  the near-delta integrand at ultra-short maturity. Within the RESEARCH GRID
  (maturities >= 7 days, |log-moneyness| <= 0.30 -- and even to 0.40) the
  worst observed value is -6.3e-12, i.e. float-level rounding on an
  effectively-zero price. Classification: benign; document that the pricer's
  validated domain excludes tau <~ 5e-3 for far-OTM strikes.

## F11. Black-Scholes limit converges at O(sigma^2), not O(sigma) [PROVEN,
strengthens F1]

Sigma-halving error ratios 4.20 / 4.46 / 5.21 (`bs_limit_sigma_convergence`)
show the deterministic-variance limit error is QUADRATIC in sigma. This is
consistent with the correction structure: with rho = -0.02 = O(sigma), the
leading stochastic-vol price corrections are O(sigma^2) (vol-of-vol variance
and rho-cross terms both quadratic). Recorded because the probe's original
annotation guessed O(sigma); the measured order is stronger evidence of the
correct limiting behaviour.

## F12. Peer-result verification round (01:50 IST) [PROVEN]

Independently verified Node A claims from code/behaviour:
- F16 constraint-map shells CONFIRMED: saturated heads reach exactly the
  0.995·sqrt(2κθ) sigma cap and disk radius 0.995 (asymptotically); the
  shells (cap, Feller] and radius (0.995, 1) are unreachable — a map-imposed
  recovery-error floor on boundary-hugging targets.
- F18 metrics normalization CONFIRMED: dheston parameter_summary uses raw
  per-parameter RMSE (no range scaling); cross-stack metric comparisons must
  state the convention.
- COS damping alpha=1.5: the field EXISTS (`heston.py:11`) but is a DEAD
  config field (see F13) — Node A's "COS with damping alpha=1.5" describes an
  unused knob, not an active numerical treatment (correction to my earlier
  F12 phrasing and to Node A's seam-matrix wording).
- Cross-check with Node A's G2 ambiguity numbers (F17: 40 near-equivalent
  solutions; median normalized price RMSE 4.7e-8 vs parameter RMSE 0.149):
  my correctly-wired PDE residual floor (4-5e-9 relative) sits at the same
  numerical-noise scale as the near-equivalence price differences — direct
  empirical support for F8: a PDE-residual term cannot distinguish
  near-equivalent parameter vectors; claims should be regularisation/validity
  only.

## F13. Adversarial review outcome (01:55 IST) — all load-bearing claims
confirmed; corrections applied [PROVEN]

An independent adversarial reviewer re-derived the mathematics and re-executed
the decisive experiments from a clean checkout. Verdicts: (a) two-driver
construction, (b) PDE + tau convention, (d) post-hoc-view zero-gradient
mechanism ("airtight"), (e) disk sufficient-not-necessary, (f) additivity,
(g) half-factor reduction, (i) parameter mapping — all CONFIRMED; canonical
classifications (F2-F7) verified including exact artifact values
(train_pde 8.9055 / valid_pde 0.0; research-control prohibition verbatim at
docs/RESEARCH_CONTROL_AND_CURRENT_STATUS.md:36). No claim required retraction.
Corrections applied from the review: F1 certification domain stated (8-point
machine precision + 120-point sweep <= 1.6e-8); F5 gap values fixed; F9/F12
dead-field attributions corrected; boundary band phrased "satisfied up to
quadrature error" (smoke artifact train_boundary 1.1e-11, COS prices clamped
at 1e-8); derivation §1.2 equality-in-law wording, §1.3 sharpening, §3
O(sigma^2) + terminal phrasing.

New minor findings from the review (all LOW severity, none affect conclusions):
1. **Dead FourierConfig fields**: `alpha`, `u_max`, `integration_eps` are
   defined but NEVER consumed (only `integration_steps`,
   `truncation_scaler`, `min_truncation_width` are used) — latent operational
   trap: editing them changes nothing.
2. **Canonical-stack autograd scan CLEAN**: repo-wide, `torch.autograd.grad`
   / `allow_unused` appear only in Archive-2's `_safe_grad` (and Node C
   tests); `train_pinn.py` / `pinn_model.py` / `torch_double_heston.py` have
   no such pattern; all `.detach()` uses are validation guards. The F2 defect
   class is unique to Archive-2.
3. **Latent non-differentiable point**: the torch factor exponent early-returns
   a graph-free zero at `maturity == 0.0`
   (`torch_double_heston.py:295-296`) — unreachable via validated entry
   points today; flag for any future tau=0 collocation.
4. **Dead compute**: `boundary_penalty` builds and discards
   `_expand_batch_to_points` output (`losses.py:49-50`).
5. **Naming misnomer**: `intrinsic_call/put` in `boundary_penalty` actually
   hold DISCOUNTED European no-arbitrage lower bounds (the bounds themselves
   are correct and tight; only the names mislead).
6. **Probe hygiene**: my earlier FD delta used default-float32 spot tensors,
   corrupting the finite-difference estimate (3.4e-4 apparent error); fixed to
   float64 — FD now agrees with autograd to ~1e-9. The autograd-vs-GL
   comparison F3 relies on (9.7e-13) was unaffected.

## F14. Cross-node convergence (01:41 IST): Node A independently reproduced
F2 and certified the Node C derivation [PROVEN]

Node A's F19 (their own seeded repro: loss 0.5319 dtype-invariant on their
batch) independently discovered the same zero-derivative defect and reclassified
the Archive-2 PDE loss as DEPRECATE (bug; do not import) — compatible with
Node C's "KEEP ISOLATED + do not adapt its PDE loss" (deprecate the loss
component, keep non-PHE utilities isolated). Their F20 records full claim-level
convergence (F2/F3/F4/F1/F5/F7 all REPRODUCED across the two instrumentations)
and they extracted Node C's 25-test suite read-only and ran it under their
local pytest: 25/25 PASS — Node C evidence independently reproduced
end-to-end. Their F21 reviewed Node C's derivation line-by-line: VERDICT
correct and complete. Disposition nuance resolved: both forbid a canonical
real-finetune mode; the ablation question is Human Decision #1. The state of
evidence for the defect: THREE independent reproductions (Node C instrumented
+ bit-exact closure, Node C adversarial reviewer re-execution, Node A seeded
repro) and THREE independent PDE derivations in agreement (Node C, Node C's
independent cross-reviewer, Node A's numerical operator verification).

## F15. Deterministic variance-mean propagation is materially price-inconsistent
[PROVEN, diagnostic]

The true bridge is `U(S, v0, tau) = E_delta[U(S_delta, v_delta, tau-delta)]`
over the JOINT law of `(S_delta, v_delta)`. Substituting propagated
conditional means (`theta + (v0-theta)e^{-kappa delta}`, the formula in
`propagate_variance_state`) as deterministic states understates prices
progressively: measured gaps `U(S, E[v_delta], tau-delta) - U(S, v0, tau)` of
-0.15% (1 day), -1.1% (7 days), -4.6% (30 days), -14.5% (90 days) at
tau=1.0, VEC_A (`tests_evidence/probe_variance_propagation_gap.py` ->
`variance_propagation_gap.json`). Two implications: (i) any future
propagate-then-price shortcut would inject double-digit percent errors at
quarterly horizons — per-date v0 fitting (as the NTPC multi-date calibration
does) is the correct design; (ii) `propagate_variance_state` must remain an
initialisation/visualisation utility only (it is currently unused by
calibration scripts — verified).

## Required final classifications (Section 23 of the brief)

- Canonical current stack: **constraint + repricing-informed inverse network**
  (F4).
- Archive-2 PDE implementation: **INCONSISTENT WITH CANONICAL MODEL** as
  implemented (F2 — the evaluated operator is not the canonical PDE; F5 —
  admissible parameter set conflicts). Its residual FORMULA is correct;
  formula vs implementation distinction is documented.
- Archive-2 integration recommendation: **KEEP ISOLATED AS EXPERIMENTAL**;
  selective reuse (chronological splitting utilities, no-leakage assertions,
  COS single-factor pricer as an independent cross-check) through adapters;
  do not adapt its PDE loss without rewriting derivative construction
  (leaf-state inputs or learned price network) and adding canonical
  constraints.
