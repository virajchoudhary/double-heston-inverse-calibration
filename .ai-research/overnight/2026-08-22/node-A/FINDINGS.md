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

Verification status: **COMPLETE (01:10 IST)** — each candidate verified directly against source.

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

## F5. Pricer agreement + adapter verification (diagnostic A-004, seeded, CPU)

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
