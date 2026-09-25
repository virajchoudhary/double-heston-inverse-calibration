# Residual-connection forgetting test (exploratory)

**Question.** Does adding residual connections stop the regular Double Heston PINN from
"forgetting" parameters it had recovered while "discovering" others during training?

**Scope.** Exploratory, and separate from `docs/DOUBLE_HESTON_V2_EXECUTION.md`. It does not
modify, re-select or replace B0, B1 or B2. No trainer, data or assessment code was changed:
every arm is `scripts/mentor_dh_pinn/train_regular_pinn.py` with existing flags.

## Arms (all other settings identical to B0)

| arm | architecture | parameters | seeds |
|---|---|---:|---|
| A | plain, 5 hidden layers × 160 (the B0 architecture) | 107,201 | 17 (= B0), 29, 43 |
| B | plain, 11 hidden layers × 160 | 261,761 | 17, 29, 43 |
| C | 5 × 160 + 3 residual blocks (inner width 160) | 261,761 | 17, 29, 43 |

B and C have **identical depth and parameter count**; the only difference is the skip
connection `h + 0.1·down(tanh(up(h)))` with `down` zero-initialised. So C vs B isolates
residual connections from added capacity, and C vs A measures adding residual depth.
Verified before training: with the same seed, C at initialisation computes exactly A's
function (max output difference 0.0) and its backbone weights are bit-identical.

Shared settings: `--factors 2 --steps 12000 --width 160 --batch 1024 --pde-batch 256
--sensitivity 0.2 --weight-pde 0.2 --weight-decay 1e-6 --lr 0.001 --resample-every 1000
--save-every 2000`, data `double_data_v2_reproducible`. Commands: `scripts/jobs.txt`, `scripts/run_one.sh`.

## Measurement

The trainer's built-in recovery check uses 4 fixed truths. That is too few to separate
forgetting from sampling noise, so every scheduled checkpoint is re-scored on **24 fresh
Double Heston truths** (seed 913311, never used for training or checkpoint selection) using
the trainer's own recovery procedure: same 21-strike × 6-maturity grid, every third strike
held out, `fit_network(starts=3, seed=906777, max_nfev=200)`, same scaled error (rho scaled
by 0.5). `scripts/forgetting_eval.py` → `eval/<run>.npz` holding errors [checkpoint, case, parameter].

Metrics (`scripts/forgetting_analysis.py`), per run and per arm over seeds:
- recovery RMSE at every checkpoint; best, final and **regret** = final − best
- **churn**: mean |change| in scaled error between consecutive checkpoints
- **forget / discover events**: a (case, parameter) pair counts as recovered when |error| < 0.10;
  forget = recovered at one checkpoint and not the next; **forget rate** = forgets / recovered

## Context runs, scored the same way
- `double_sobolev_s17`, the plain parent of the earlier deeper runs
- `double_continue_pde02_s17`, a plain continuation from that parent (LR 2e-5 constant)
- `outputs/deeper_pinn/deep11_s17` (3 residual blocks, width 384) and `deep17_full_s29`
  (6 blocks): earlier residual continuations from the same parent, single seed each
