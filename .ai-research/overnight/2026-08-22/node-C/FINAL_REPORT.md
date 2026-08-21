# Node C Final Report — Double Heston PDE & Physics Audit

Overnight 2026-08-22. Branch `overnight/20260822-c-pde`, genesis
`642702e6706a3d17b3031619f35bda39bc144483`. All claims carry file/line
citations and/or executable evidence
(`tests/test_node_c_pde_physics_audit.py` 25/25 PASS; numerical probe
`tests_evidence/probe_residuals.py` -> `probe_results.json`).

---

## Executive conclusion

Is the repository's current PINN physics mathematically defensible?

- **Canonical stack: YES as physics, NO as "PINN".** The canonical production
  model and differentiable pricer are mathematically sound and now
  *numerically certified*: the derived canonical pricing PDE is satisfied to
  machine precision (relative residual <= 1.3e-15), with all limiting cases
  passing. However, the canonical model is **not PDE-informed**: its
  "physics" loss is differentiable repricing plus hard structural
  constraints. Label: **constraint + differentiable-repricing informed
  inverse network**.
- **Archive-2 stack: NO.** Its PDE residual formula is the correct canonical
  PDE on paper, but the implementation is broken: every variance-state
  derivative evaluates to exactly ZERO (autograd views created after the
  forward pass are not graph nodes; `_safe_grad`'s `allow_unused` silently
  returns zeros). The actually-penalized operator is a Black–Scholes-type
  operator missing ALL variance dynamics. Additionally, even a fixed version
  would differentiate the analytic pricer itself, so the residual carries no
  information about the inverse problem. Archive-2's parameter contract is
  also semantically incompatible with the canonical contract (no Feller, no
  correlation disk, rho restricted negative; demonstrated canonical-invalid
  emissions), and its real-market fine-tuning path violates the research
  control that real observations must not update primary NN weights.

No changes to the production pricer were needed or made.

## Canonical stochastic specification

Affine two-spot-driver Double Heston (proven from `src/double_heston.py`,
citations in `derivations/CANONICAL_DOUBLE_HESTON_PDE.md` §1):

```
dS/S = (r - q) dt + sqrt(v_s) dW_s + sqrt(v_f) dW_f
dv_i  = kappa_i (theta_i - v_i) dt + sigma_i sqrt(v_i) dB_i ,  i = s, f
d<W_i, B_j> = rho_i dt if i=j else 0 ;  d<B_s, B_f> = 0
```

Instantaneous total spot variance is `v_s + v_f`; variance factors are
independent and coupled to spot only through the per-factor leverage
correlations. The repo constraint `rho_s^2 + rho_f^2 < 1` is a sufficient,
state-independent PD condition (exactly PD for the alternative single-driver
correlation matrix), NOT a necessary condition of the implemented model —
document as a modelling choice. Canonical parameter order is
`[kappa, theta, sigma, rho, v0] x (slow, fast)` (`src/constants.py:7-22`),
preserved everywhere in the canonical path.

## Canonical PDE (forward tau form)

```
U_tau = 1/2 (v_s+v_f) S^2 U_SS + 1/2 sigma_s^2 v_s U_vs vs + 1/2 sigma_f^2 v_f U_vf vf
      + rho_s sigma_s v_s S U_Svs + rho_f sigma_f v_f S U_Svf
      + (r-q) S U_S + kappa_s (theta_s - v_s) U_vs + kappa_f (theta_f - v_f) U_vf - r U
```

No `U_{vs vf}` term (independent variance BMs); cross coefficients
`rho_i sigma_i v_i S`. Verified three ways: independent re-derivation
(term-by-term match + affine Riccati decoupling argument), single-factor
reduction to Heston-1993 (all seven terms), and machine-precision numerical
satisfaction by the torch pricer with coefficient-perturbation controls.

## Terminal and boundary conditions

Call/put payoffs at `tau = 0`; `S = 0` absorbing (call 0, put `K e^{-r tau}`);
large-S asymptote `S e^{-q tau} - K e^{-r tau}`; `v_i = 0` degenerate, no
condition needed under strict Feller (enforced). Archive-2's `boundary_penalty`
is a static no-arbitrage band satisfied identically by its analytic pricer —
not boundary physics. No terminal-condition loss exists in either stack.

## Canonical current model classification

**Constraint-informed + repricing-informed inverse network** (hybrid of those
two, specifically). Evidence: `models/pinn_model.py:22-53,126-154` (hard
structural map), `src/train_pinn.py:91-111` (parameter MSE + differentiable
repricing MSE; no PDE residual; checkpoint metadata says
"differentiable_double_heston_repricing"). The label "PINN" is not accurate
for this stack today.

## Archive-2 PDE audit (term-by-term: `tables/ARCHIVE2_PDE_TERM_MAP.md`)

Formula: CORRECT (matches the canonical PDE exactly, including tau sign and
absence of `V_v1v2`). Implementation: **INCONSISTENT WITH CANONICAL MODEL** —

1. all five variance-state derivatives are exactly zero (instrumented:
   losses.py construction returns 0.0 where the true dV/dv0 is 28.49/13.15);
   the effective residual equals the residual with all v-terms deleted,
   EXACTLY (0.604568934256152 == 0.604568934256152);
2. the differentiated object is the analytic COS pricer, so a corrected
   residual would be ~0 for any parameters (no identifying information;
   market prices never enter);
3. differentiating through the cumulant-dependent COS truncation range adds
   spurious derivative terms even after fixing (1).

## Parameter-contract differences

`tables/PARAMETER_CONTRACT_MATRIX.md`: order requires a double transposition
(v0 position + fast/slow role; Archive-2 factor 1 = FAST); admissible sets
are semantically incompatible (Archive-2 lacks Feller and the correlation
disk, restricts rho to (-0.95, -0.05); its own box emits canonical-invalid
vectors — both Feller gaps ~ -2.24, disk 1.805). ADAPTER REQUIRED for
order/bounds; SEMANTIC CONFLICT for constraints; no adapter can reconcile the
admissible sets.

## Constraint audit

Canonical: positivity, strict per-factor Feller, strict kappa ordering,
correlation disk — all HARD (raise-at-pricing; structural sigmoid/softplus/
polar maps in the NN; scipy-side hard reparameterization in calibration
scripts). Archive-2: sigmoid box bounds + structural kappa2<=kappa1 only;
Feller absent, disk absent, redundant soft ordering penalty. Demonstrated
invalid emission (reproducible test).

## Limiting-case tests (commands/results)

`/usr/bin/python3 .ai-research/overnight/2026-08-22/node-C/tests_evidence/run_node_c_tests.py`
(= pytest suite, 25/25): factor-additivity identity 1.8e-15; two-half-factor
reduction to single Heston vs independent COS pricer 3.2e-11; BS
deterministic-variance limit <= 4.0e-4 relative; put-call parity <= 7.1e-15;
Archive-2 zero-derivative proof; constraint-gap demonstration.

## Autograd/PDE implementation findings

Correct pattern (demonstrated): make the state entries (spot, tau, v0 slow,
v0 fast) genuine `requires_grad` LEAVES and assemble the parameter vector
from them; then `torch.autograd.grad` reaches everything and the canonical
pricer satisfies the PDE to 1e-15. Broken pattern (Archive-2): views of the
parameter tensor created after the forward pass are not graph nodes;
`allow_unused=True` + zeros fallback converts an autograd wiring bug into a
silent mathematical error. Rule for the canonical seam: never call
`autograd.grad` against freshly-created views; assert
`outputs.requires_grad` and that the returned gradient is non-None/non-zero
where mathematically expected.

## Training/validation loss audit

Archive-2: training includes `lambda_pde * pde`, validation evaluates with
`pde_points=0` inside `torch.no_grad()` (double silencing), model/checkpoint
selection on the PDE-free total (`train_double_heston.py:253-272`; artifact:
`train_pde: 8.9, valid_pde: 0.0`). Objective mismatch — currently moot given
the residual is broken, but must be resolved before any genuine PDE term is
adopted. Canonical `train_pinn.py`: validation uses the same parameter +
repricing objective; selection validation-only, test untouched; no mismatch.
Loss-weight audit (meaning/units): canonical parameter loss is z-scored MSE;
repricing losses are spot-normalized MSE/smooth-L1 (scale-free, comparable);
Archive-2 relative-residual scaling by max(|V|,1) is defensible. No weight
tuning performed tonight.

## Real-market training-policy conflict

Archive-2 `real_finetune` updates ALL network weights on real surfaces
(price + broken-PDE + bounds objectives; fresh Adam at half LR; `--continuous`
trains indefinitely). This directly conflicts with the research control
"real-market observations must NOT update primary ANN/PINN weights".
Chronology/leakage hygiene in that path is otherwise sound (chronological
trade-date splits, zero-leakage assertions, per-split synthetic templates).
Disposition: **ISOLATE AS NON-PRIMARY ABLATION, DISABLE BY DEFAULT** for the
canonical path. The canonical NSE/NTPC scripts are scipy least-squares
calibrations of the production pricer (not neural) and do not violate the
control.

## Interaction with identifiability (for Node B)

A PDE residual derived from the same model that generates the surfaces cannot
add identifying information beyond the repricing loss: both constrain the
same parameter-to-price map. It can regularize, enforce structural validity
of a learned pricing function, or impose an (artificial) preference among
observationally equivalent parameter vectors — it cannot resolve structural
non-identifiability. If Node B confirms near-equivalent parameter vectors,
the paper should claim regularization/validity, not identification. Current
Archive-2 implementation provides neither (broken residual). A low
price/repricing loss does NOT establish ten-parameter identification —
consistent with the repo's own control language.

## Strongest findings (ranked by confidence)

1. [PROVEN] Archive-2 PDE residual: all variance-state derivatives exactly
   zero; effective operator misses the entire variance dynamics; residual is
   a wrong-operator signal (consistent with train_pde=8.9).
2. [PROVEN] Canonical pricer satisfies the derived canonical PDE to machine
   precision (with sensitivity controls); all limiting cases pass. The
   canonical physics contract is sound.
3. [PROVEN] Canonical model is constraint + repricing informed, not
   PDE-informed ("PINN" mislabel).
4. [PROVEN] Parameter contract conflict: order double-transposition +
   semantic constraint conflicts (demonstrated invalid emissions).
5. [PROVEN] Archive-2 validation/selection excludes its PDE loss (objective
   mismatch); real fine-tuning violates the no-NN-update-on-real control.
6. [DERIVED] Even a fixed Archive-2 residual (analytic pricer object) carries
   no inverse-problem information; genuine physics seam requires a learned
   pricing function with leaf-state inputs.
7. [DERIVED] PDE physics cannot resolve structural non-identifiability; it
   can regularize.

## Failed hypotheses (seemed problematic, was actually correct)

- Hypothesis: "the canonical pricer may not satisfy the canonical PDE because
  Gauss-Laguerre quadrature noise dominates derivatives" — FALSE: residual is
  1e-15-level; perturbation controls confirm sensitivity.
- Hypothesis: "Archive-2's residual must be near-zero because it
  differentiates the analytic pricer (vacuous)" — the residual IS meaningless,
  but for a different, worse reason: broken derivative wiring makes it
  LARGE-and-wrong, actively injecting bias toward BS-like solutions.
- Hypothesis: "the disk constraint rho_s^2+rho_f^2<1 suggests the model uses a
  single spot driver (non-affine)" — FALSE: the CF structure proves the
  two-driver affine construction; the disk is a conservative sufficient
  condition.
- Hypothesis: "put-call parity/one-factor reduction might expose canonical
  pricer inconsistencies" — FALSE: parity exact to 7e-15; reduction to
  single-factor Heston exact to 3.2e-11 against an independent pricer.
- Initially suspected my own VEC_B was valid — it failed the slow-factor
  Feller gate (canonical validation caught it; gate works as designed).

## Changes made

- `tests/test_node_c_pde_physics_audit.py` (new, 25 evidence tests)
- `.ai-research/overnight/2026-08-22/node-C/` (STATUS, FINDINGS, EXPERIMENTS,
  FINAL_REPORT, derivations/, tables/, tests_evidence/ incl. probe + results)
- No production files modified. Commits: `a44c8bb` (+ final report commit).

## Tests

Command: `/usr/bin/python3
.ai-research/overnight/2026-08-22/node-C/tests_evidence/run_node_c_tests.py`
— 25/25 PASS (pytest-equivalent runner because the torch-capable interpreter
lacks pytest; compute policy forbids environment changes; the runner
implements pytest.parametrize cross-products by name). Tolerances and their
justification are in the test file docstrings/comments (observed vs bound).

## Recommended canonical physics seam (what to eventually implement)

If a genuine PDE-informed term is wanted for the paper:
1. a forward price network `V_theta(S, v_s, v_f, tau)` trained with
   collocation on the CANONICAL PDE (derived above) with leaf state inputs,
   terminal payoff + parity-consistent boundaries, parameter-conditioned
   (canonical order/contract, hard constraints exactly as the existing
   canonical map);
2. an inverse network mapping the observed surface to canonical parameters,
   coupled through the learned pricer AND the PDE residual of the LEARNED
   V (not an analytic pricer);
3. validation/objective parity: if PDE loss is trained on, it must appear in
   validation/selection or the mismatch must be explicitly justified;
4. expectations set per the identifiability analysis: claim regularization /
   structural validity, not new identifying information;
5. Archive-2 reusable through adapters: chronological splitting +
   zero-leakage utilities, COS single-factor pricer as independent
   cross-check. Do not adopt its PDE loss or parameter transform as-is.

## Remaining uncertainty (mentor/mathematical review)

- Intent of the correlation-disk constraint (uniform-observable-PD modelling
  choice vs single-driver carry-over) — paper should state which.
- Whether strict `kappa_slow < kappa_fast` identifiability ordering should
  also be imposed in any future learned-pricer coupling (it is a tie-breaking
  convention, not physics).
- Boundary truncation strategy for a future collocation implementation
  (log-spot vs spot coordinates; variance-domain truncation) — not settled
  tonight; no collocation code was written (out of scope).
- Node B's identifiability evidence was not yet pushed at last fetch; my
  identifiability conclusions are mathematical, not yet cross-checked
  against their numerical Jacobian analysis.

## Branch and final commit

Branch `overnight/20260822-c-pde` (from genesis `642702e`, pushed to origin).
Final commit: see `git log` head of this branch. main untouched; no force
push; no 10k generation; no real-market neural training; G2 and canonical
status documents unchanged; no environment mutation; production pricer
unmodified.

Peer corroboration (final fetch): Node A's updated FINDINGS independently
confirm the two-stack map, the parameter-order divergence, and classify the
Archive-2 residual as a pricer-consistency check ("near-vacuous") — consistent
with F3 — but did NOT detect the zero-derivative defect; F2 deepens their
result and was reported to the swarm issue (#18). Node B still had no pushed
evidence at final fetch.

## Final safety checklist

- [x] all Node C evidence committed and pushed to the Node C branch only
- [x] no secrets or large binaries added
- [x] focused tests run (25/25 pass)
- [x] Node A/B fetched; Node A corroborates; Node B had no evidence yet
- [x] main unchanged; no force push; no history rewrite; no destructive reset
- [x] no final 10k generation; no research-scale training; no real-market
      fine-tuning executed (audit was read-only for that path)
- [x] G2 not marked passed; representation not frozen
- [x] production pricer and canonical scientific-status documents untouched
- [x] every mathematical claim linked to derivation, code line, or test
