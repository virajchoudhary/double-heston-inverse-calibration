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

Verification status: PENDING (Phase 1).
