# Model 3 PDE pretraining audit

Status: `MODEL3_PDE_PILOT_READY_AFTER_AUDIT_FIXES`. This is a concise readiness
audit, not a training result or engineering diary. Commit `f34a4d3` had been
marked pilot-ready although no Stage-A training had occurred. An adversarial
pre-pilot review found blocking execution defects, so readiness was revoked
before scientific execution; the repairs are pre-result corrections, not tuning
after outcomes. A fresh adversarial review marked every blocker resolved.

## Provenance

- Canonical main/base: `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- Working branch: `research/model3-pde-protocol`
- Issue #34 lineage remains separate at
  `codex/r2-noise-recovery`; it was neither merged nor copied into this branch.
- Final R2 dataset SHA-256:
  `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`.

## Artifact disposition

| Artifact | Classification | Disposition |
|---|---|---|
| `src/double_heston.py` | A - scientifically reusable | Production dynamics source; unchanged |
| `src/constants.py` | A | Canonical ten-parameter order reused |
| `src/constraints.py` | A | Structural constraints and joint disk reused |
| `configs/parameter_bounds_PROVISIONAL.yaml` | A for ceilings | Hard safety ceilings retained; validity remains separate |
| `src/r2_primary/dataset.py` | A | Frozen 100-feature R2 loader reused unchanged |
| `src/r2_primary/evaluation.py` | A | Metric families to be reused at evaluation |
| `models/pinn_model.py` | B engineering / D misleading name | Constraint map and encoder reused; it is repricing-informed, not PDE-informed |
| `src/torch_double_heston.py` | B engineering | Differentiable Fourier mirror useful for consistency checks, not the PDE operator itself |
| `src/train_pinn.py` | D misleading / C historical | Its “physics” loss is masked Fourier repricing |
| `configs/pinn_baseline.yaml` | C historical | Model 2 infrastructure configuration, not Model 3 |
| `src/dheston/models/losses.py` | C incompatible | Archive-2 residual drops variance derivatives through disconnected slice views |
| broader `src/dheston/**` | C donor / E direct reuse | Noncanonical layout, constraints, real-market update path, and PDE loss |
| historical `outputs/metrics/pinn_*` | C historical evidence | Never treated as Model 3 results |
| overnight decision record | A provenance | Preserves terminology and Archive-2 defect finding |
| Issue #34 recovery branch | A preserved lineage | Kept separate; numeric outcomes excluded from Model 3 tuning |

No preexisting implementation supplied a correct canonical state-differentiable
Double Heston residual. The narrow operator was therefore written separately
rather than importing Archive-2 or renaming Model 2.

## Implementation scope

- `src/model3_pde/operator.py`: float64 canonical PDE operator and parameter
  validation; rejects non-leaf/detached states and disconnected derivatives.
- `src/model3_pde/collocation.py`: generic and surface-conditioned deterministic
  samplers producing leaf tensors and source indices.
- `src/model3_pde/losses.py`: scaled residual, terminal, boundary, and masked
  reconstruction primitives.
- `src/model3_pde/model.py`: float64 conditional bounded forward network plus
  inverse R2 encoder with explicit float32-to-float64 input upcast; excludes
  duplicate v0 conditioning.
- `configs/model3_pde_protocol.yaml`: pre-result architecture/data/loss freeze.
- `scripts/run_model3_pde_smoke.py`: development-only affine-solution smoke.
- `tests/test_model3_pde_foundation.py`: focused mathematical and schema tests.
- `tests/test_real_market_weight_update_quarantine.py`: explicit exception for
  the new genuine-PDE namespace while preserving the Archive-2 quarantine.
- `scripts/run_model3_pde_pilot.py`: thin deterministic Stage-A driver with
  train/validation isolation, atomic primary checkpoints, optimizer export,
  RNG restoration, history gates, and clean-tracked-tree enforcement for real
  execution.

## Verification performed

- Production-pricer source, canonical constants/constraints, primary R2 loader,
  primary protocol, and final dataset had no tracked diff against canonical main
  before the documentation-only completion.
- Development smoke reports affine PDE residual `0.0` and finite values.
- Focused Model 3 tests cover exact affine residual, rejection of silent
  non-differentiable state, deterministic generic and surface-conditioned
  sampling, terminal payoff exactness, supported-range representability, finite
  system gradients, frozen configuration identity, and Torch-mirror consistency.
- Fresh independent documentation/code review found four issues: missing pilot
  driver for launch readiness, nonexecutable YAML tenor expressions, implicit
  float64 model conversion, and overstated leaf enforcement. Launch-readiness is
  correctly reported as blocked because adding the training driver was outside
  this approval; the other three issues were fixed and reverified locally.
- A later adversarial pre-pilot audit invalidated commit `f34a4d3`'s
  `MODEL3_PDE_PILOT_READY` claim after finding CUDA leaf construction,
  surface/slot contract alignment, RNG restoration, multi-batch history,
  checkpoint pairing, and dirty-tree identity defects. No Stage-A training had
  been run when readiness was revoked.
- Pre-result repair coverage includes device-aware leaf creation, seeded
  `(surface, eligible canonical slot)` contract selection, complete CPU/CUDA/
  NumPy/Python RNG checkpoint restoration, multi-batch history validation,
  authoritative in-checkpoint optimizer state with export-pair rejection, and
  clean-tracked-tree gates for real Stage A.

## Readiness assessment

Scientifically ready:

1. canonical dynamics are unambiguous;
2. the PDE and terminal convention are explicit;
3. Model 3 contains an autograd-evaluated PDE residual, unlike Model 2;
4. R2 inputs, constraint map, splits, metrics baselines, and anti-leakage rules
   are preserved;
5. loss weights and experiment budgets are frozen before outcomes;
6. lightweight mathematical and integration tests pass.

Pre-launch operational condition:

1. all identified execution defects were fixed and independently reviewed;
2. focused tests and tiny CPU smoke passed;
3. cloud Git/config/data identities and focused tests must still be verified in
   the execution session before GPU allocation.

No training, large calibration, long multiprocessing, or GPU workload was run.
