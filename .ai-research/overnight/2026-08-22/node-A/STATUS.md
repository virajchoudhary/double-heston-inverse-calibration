# Node A Status — COORDINATOR_ARCHITECTURE

Branch: `overnight/20260822-a-architecture`
Base SHA: `642702e6706a3d17b3031619f35bda39bc144483` (= origin/main at start, verified)

## Run log

| Time (IST) | Phase | Status |
|---|---|---|
| 00:50 | Git bootstrap complete: clean tree, HEAD = genesis, origin/main = genesis, no prior overnight branches. Branch created. | DONE |
| 00:55 | Repo architecture map completed (read-only exploration). Two PINN stacks located: Stack A (`src/train_pinn.py` + `models/pinn_model.py` + `src/torch_double_heston.py`) and Stack B (`src/dheston/*` + root `train_double_heston.py`). 7 candidate incompatibilities identified for verification. | DONE |
| 01:00 | Evidence scaffolding committed; beginning Phase 1 incompatibility verification. | IN PROGRESS |

## Current activity

Phase 1: verifying the candidate PINN-stack incompatibilities against actual code
(parameter order, factor roles, constraint maps, physics-loss semantics, pricing
formulations, surface representations).

## Scope confirmation

Authorized tonight: diagnostic + architecture + mathematical audit only.
No training-scale runs, no 10k generation, no G2 claims, no main pushes.
CPU-only torch confirmed by prior readiness audit; all workloads small and seeded.
