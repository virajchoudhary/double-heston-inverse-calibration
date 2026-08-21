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
| 01:01 | Committing Phase 1 checkpoint; next: Phase B seam matrix + Phases C-J synthesis. | DONE |
| 01:09 | All solo phases COMPLETE: seam matrix (20 seams), phases C-J (F6-F8), Phase G policy audit incl. run_pinn_* runners (F10), OOD quantification (F11), adapter artifact w/ named-field self-test that caught an inverted permutation (F12), NEXT_STEPS alignment (F13), import-smoke forensics: real_finetune executed at smoke scale + PDE term noise-dominated (F14), verification round incl. determinism re-run (F15), representable-set shell quantification (F16), ARCHITECTURE.md alignment + ambiguity-case representability (F17), metrics normalization note (F18). FINAL_REPORT + MORNING_SWARM_REPORT drafted. 22 canonical tests passing. | DONE |
| 01:10 | Adversarial reviewer relaunched after transient network failure (first attempt died at ~9.5 min). Peer heartbeat posted on Issue #18 (B at genesis, C absent). 30-min peer-poll automation active. | IN PROGRESS |

## Current activity

Waiting on: adversarial reviewer (background), Node B/C evidence (none pushed).
On receipt: integrate reviewer verdict into seam matrix; integrate peer evidence with
REPRODUCED/SUPPORTED/PRELIMINARY/DISPUTED/UNRESOLVED labels. Consolidation ~07:00 IST.

## Scope confirmation

Authorized tonight: diagnostic + architecture + mathematical audit only.
No training-scale runs, no 10k generation, no G2 claims, no main pushes.
CPU-only torch confirmed by prior readiness audit; all workloads small and seeded.
