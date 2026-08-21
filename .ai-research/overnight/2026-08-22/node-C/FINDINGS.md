# Node C Findings — PDE & Physics Contract Audit

Overnight 2026-08-22. Branch `overnight/20260822-c-pde` from genesis
`642702e6706a3d17b3031619f35bda39bc144483`. All numerical evidence:
`tests_evidence/probe_residuals.py` -> `probe_results.json`; pytest suite
`tests/test_node_c_pde_physics_audit.py`. Derivation:
`derivations/CANONICAL_DOUBLE_HESTON_PDE.md`.

Confidence labels: [PROVEN]=code+numerics, [DERIVED]=math from repo SDE with
numerical confirmation, [INFERRED]=structure argument, [UNVERIFIED].

---

## F1. Canonical physics contract is sound and now numerically certified [PROVEN]

The canonical production model is the affine two-spot-driver Double Heston
(independent variance factors, additive total spot variance, per-factor
leverage correlations). The derived canonical pricing PDE (see derivation) is
satisfied by the differentiable Gauss–Laguerre torch pricer to machine
precision: |relative residual| <= 1.3e-15 over 8 parameter/point combinations
(two parameter vectors x four (S,K,tau) points). Sensitivity controls: zeroing
the two rho cross terms raises the relative residual to 6e-3..1.7e-1 (>= 7
orders of magnitude); halving the cross coefficient produces EXACTLY half the
perturbed residual (internal consistency). Limiting cases all pass: put-call
parity 7e-15; exact CF factor-additivity identity 1.8e-15; two-identical-half-
factors == single Heston cross-validated against the independent COS pricer to
3.2e-11; deterministic-variance Black–Scholes limit within 4e-4 relative.

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
8.9` (experiments/import_smoke_20260821/metrics.json) is consistent with this
large wrong-operator residual, not with quadrature noise. Classification:
INCORRECT (implementation), independent of the residual's conceptual problem
(F3).

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
that the canonical contract rejects with THREE violations (both Feller gaps
-2.245/-2.248; disk 1.805 > 1) — see `archive2_constraint_gap` and
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
  (8.323761 vs 8.323844); 256 vs 1024 is bit-identical (series saturated at
  u_max=120). The 256-default is adequate; residual identical at 256/384.
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
- COS damping alpha=1.5 CONFIRMED (`src/dheston/pricing/heston.py:11`).
- Cross-check with Node A's G2 ambiguity numbers (F17: 40 near-equivalent
  solutions; median normalized price RMSE 4.7e-8 vs parameter RMSE 0.149):
  my correctly-wired PDE residual floor (4-5e-9 relative) sits at the same
  numerical-noise scale as the near-equivalence price differences — direct
  empirical support for F8: a PDE-residual term cannot distinguish
  near-equivalent parameter vectors; claims should be regularisation/validity
  only.

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
