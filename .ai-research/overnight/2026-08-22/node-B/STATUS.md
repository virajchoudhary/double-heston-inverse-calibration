# Node B Status — Identifiability & Calibration Investigator

Run window: 2026-08-22 ~00:50–07:30 IST
Branch: overnight/20260822-b-identifiability (from genesis 642702e6706a3d17b3031619f35bda39bc144483)
Compute: CPU-only, 12 cores, Python 3.13.5, torch 2.13.0, numpy 2.2.5, scipy 1.15.3

## Timeline
- [00:51] Git bootstrap verified (HEAD == origin/main == genesis). Branch created + pushed.
- [00:55] Codebase mapped: canonical pricer src/double_heston.py (Gauss-Laguerre 64-node Fourier, strict Feller, kappa_slow<kappa_fast ordering); calibration src/calibrate_double_heston.py (3 deterministic starts, TRF least_squares, sigmoid constraint transform, residual=(pred-obs)/max(obs,1)); 108-grid = 9 log-moneyness x 6 maturities x calls+puts (src/surface_grid.py, src/constants.py).
- [01:00] Prior evidence mapped: G2 central-5 conditioning ~5.1e7, 0/12 noisy recovery; GLOBAL_AMBIGUITY=ESTABLISHED on central-5 geometry (4 cases, 20 starts, near-equivalence 2.5e-7 normalized RMSE, median scaled param RMSE 0.1485); dominant compensation v0_slow/v0_fast.
- [02:58] Resumed after session break; evidence skeleton created.

## Current phase
Building fast vectorized diagnostic pricer + Phase A baseline reproduction.

## Open questions
- Does global ambiguity persist on the FULL 108-quote provisional grid (prior evidence: central-5 20 quotes)?
- Jacobian conditioning on full 108 grid vs central-5?
- Noise: separation of repricing vs parameter degradation on full grid.

## Checkpoint 03:07 IST
- Fast pricer validated (0.0 diff vs production on 108 grid; 11.7x speedup). PASS.
- Phase D done: full108 kappa~2.6e4 rank 10/10; central-5 6.5e8 rank 7.5; calls=puts redundancy; exact factor-swap symmetry.
- Phase A (production entry, case_1): clean best start exact (param RMSE 2.9e-11, price RMSE 1.1e-13) BUT other starts disperse to param RMSE 0.39; 1% noise: param RMSE up to 0.57 at price RMSE 0.13 (~noise floor).
- Phase B (case_1, 12 starts): clean median param RMSE 0.148 at price RMSE 1.1e-6; 0.5%/1%/2% noise: 12/12 boundary hits, param RMSE ~0.31-0.34, price RMSE at noise floor. GLOBAL AMBIGUITY REPLICATES ON FULL 108 GRID.
- Peers fetched: Node A (PINN stack incompatibilities, real_finetune control risk), Node C (canonical PDE certified; archive-2 derivatives broken). Complementary; no conflicts.
