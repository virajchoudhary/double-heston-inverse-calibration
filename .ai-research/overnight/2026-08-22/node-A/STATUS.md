# Node A Status — COORDINATOR_ARCHITECTURE

Branch: `overnight/20260822-a-architecture`
Base SHA: `642702e6706a3d17b3031619f35bda39bc144483` (= origin/main at start, verified)

## Run log

| Time (IST) | Phase | Status |
|---|---|---|
| 00:50 | Git bootstrap complete: clean tree, HEAD = genesis, origin/main = genesis, no prior overnight branches. Branch created. | DONE |
| 00:55 | Repo architecture map completed (read-only exploration). Two PINN stacks located: Stack A (`src/train_pinn.py` + `models/pinn_model.py` + `src/torch_double_heston.py`) and Stack B (`src/dheston/*` + root `train_double_heston.py`). 7 candidate incompatibilities identified for verification. | DONE |
| 01:00 | Evidence scaffolding committed; beginning Phase 1 incompatibility verification. | DONE |
| 01:00 | **Session resumed after interruption.** Phase 1 COMPLETE: all 7 candidates verified (6 confirmed/refined, 1 corrected: ordering_penalty is kappa-not-theta but dead-by-construction). Pricer agreement diagnostic run: A-torch = production at ~1e-15; B-COS agrees 1e-12..1e-9; parameter adapter verified via exact gradient permutation. Training-policy red flag: Stack B `real_finetune` mode + `--continuous` violate canonical no-real-weight-update control. | DONE |
| 01:01 | Committing Phase 1 checkpoint; next: Phase B seam matrix + Phases C-J synthesis. | IN PROGRESS |

## Current activity

Phase B: canonical seam matrix construction from verified Phase 1 evidence, then
training-policy audit completion (Phase G), G2 coupling (H), fairness contract (I),
documentation contradictions (J).

## Scope confirmation

Authorized tonight: diagnostic + architecture + mathematical audit only.
No training-scale runs, no 10k generation, no G2 claims, no main pushes.
CPU-only torch confirmed by prior readiness audit; all workloads small and seeded.
