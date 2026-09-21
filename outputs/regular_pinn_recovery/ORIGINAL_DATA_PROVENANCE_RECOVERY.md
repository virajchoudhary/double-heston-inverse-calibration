# Original regular-PINN data provenance and recovery

Date: 2026-09-11

## Decision

The original `train.npz` and `collocation.npz` byte streams were not found in the accessible project scope. Exact recreation is not justified from the surviving provenance. No baseline evaluation, L-BFGS continuation, Jacobian pilot, or architecture change was run.

## Required artifacts

| File | Required SHA-256 | Found |
|---|---|---|
| `outputs/regular_pinn_recovery/double_data/train.npz` | `f18c296f1d0183ac266448830a05e80e8e7cdd3789cca5c6a5434dd09c7fdc28` | No |
| `outputs/regular_pinn_recovery/double_data/collocation.npz` | `a3820957488f9dacbbcebeb2c3425b766473c7b65ffb76c9a8a763f193c422af` | No |

The tracked directory contains only `manifest.json`. The NPZ files are excluded by `*.npz` in `.gitignore`, are absent from every local Git object/ref, and have no Git-LFS entries.

## Search boundary and candidates

The search covered the current worktree, the source checkout, every registered worktree, sibling project checkouts under `C:\Users\viraj\Documents\Codex`, all relevant ignored/untracked output trees, prior run manifests and source snapshots, local Git refs/objects/tags, and Git-LFS metadata. The recorded source path was on a different machine: `/Users/dhruvaambhaikar/Documents/Options pricing/double-heston-calibration/outputs/regular_pinn_recovery/double_data`.

Only these candidates were found:

| Candidate | Shape | SHA-256 |
|---|---:|---|
| `double_data_phase1_smoke_917017/train.npz` | `q=(64,12)` | `9bf378f4f720a96d4c8afb7f3d5ffdc4d61f46fa3bd6e25d643793766722b4ad` |
| `double_data_phase1_smoke_917017/collocation.npz` | `q=(18000,12)` | `c90dae591494c42beb1cac8642cfe6803d846fe075a70f3781d09833ea815d9a` |
| `double_data_regenerated_906210/train.npz` | `q=(131072,12)` | `ddfa96d8393150763d1d523bddfed868d2c03dd87d82b9a4f7c1928cc296c9d9` |
| `double_data_regenerated_906210/collocation.npz` | `q=(18000,12)` | `95c5a25ac500f95b5e139ea1c280fe687555687c98d358a8c16d257994e0d91b` |

The public repository page exposes no release containing these files, and the local history contains no workflow definition that could have uploaded them. Authenticated `gh` artifact enumeration was unavailable because the configured GitHub credential is invalid; therefore private or expired remote artifacts cannot be ruled out from this machine.

## Historical generator

Commit `3927ad7ff907504852423864afdd4d18ebf34298` exactly matches all three source hashes recorded by the original data manifest:

- `scripts/mentor_dh_pinn/prepare_regular_pinn.py`: `a8f2a10c2393f76d8074d3d3f99ff58ebe5d113e3d7f20967bfb60a2c4fa9845`
- `src/mentor_dh_pinn/regular_pinn_data.py`: `bb47652e896425d38be8e967269f1e81683cb879c37b9867c1cbde7cf404fb71`
- `src/mentor_dh_pinn/torch_pricer.py`: `109ae648fadfd7c89715aca385805ec1dde1a2cc55844f5635245292688c13d7`

The manifest records seed `906210`, 131,072 training candidates, 16,384 validation candidates, 18,000 collocation points, and 131,072/131,072 usable training rows. Sampling uses SciPy Latin hypercube; train, validation, and collocation seeds are `906210`, `906211`, and `906212`.

## Why regeneration differs

The later `9c44422` change adds optional full-coordinate gradients but does not change sampling bounds, seed derivation, ordering, rejection rules, or the default parameter-gradient result. A 512-row in-memory probe under one environment produced identical hashes for `q`, `price`, `w`, `g`, `usable`, `quadrature_difference`, and `dg_du` from the historical and current source. Code drift is therefore not sufficient to explain the mismatch.

The Windows regeneration produced 129,451 usable rows. All 1,621 rejected rows have non-finite Torch prices; the finite rows satisfy the IV, quadrature, and gradient predicates. The current runtime is Windows 11, Python 3.13.9, NumPy 2.4.3, SciPy 1.17.1, and PyTorch 2.10.0 CPU. Nearby runs from the original Mac record macOS 15.7.7 arm64, Python 3.14.5, NumPy 2.4.6, SciPy 1.17.1, and PyTorch 2.13.0. The data manifest itself does not record package versions or platform, so the surviving evidence cannot distinguish execution-environment numerical drift from an incorrect historical `usable` claim. It cannot support a byte-identical recreation.

NPZ SHA-256 is also not a canonical array identity: `np.savez_compressed` embeds ZIP metadata such as entry timestamps. Recreating equal arrays at another time does not by itself recreate the required file hash.

## Scientific status and controlled reset

The historical checkpoint remains usable as a fixed surrogate for inference and diagnostics tied to its recorded input hashes. It is not usable as evidence for exact retraining, exact continuation, or a reproducible generator-to-checkpoint chain while the source arrays are absent.

The recovery investigation originally recommended this reset:

1. Freeze one source revision and a platform-specific environment image/lock, including Python, NumPy, SciPy, PyTorch, CPU architecture, and thread settings.
2. Resolve the non-finite Torch pricing rows before cohort generation, then predeclare whether invalid candidates remain masked or are deterministically replaced.
3. Generate a new canonical cohort once; record per-array dtype/shape/content hashes as well as archive hashes, and store the raw artifacts in immutable durable storage rather than ignored local output only.
4. Retrain a new baseline from scratch on that sealed cohort, preserving the four development cases and ten-parameter gates, then run the already-fixed Phase 1 and Phase 2 commands once without tuning.

Architecture redesign is not justified by this provenance result. Weak-singular-subspace analysis was not reached because exact-data Phase 2 was not run.

## Controlled V2 reset completed

The subsequently authorized reset preserved the historical directory and created
`DOUBLE_DATA_V2_REPRODUCIBLE` separately. The 1,621 Windows failures were valid
samples: at the largest 128-node Gauss-Laguerre abscissa, a finite complex
subnormal from `exp(-d*tau)` caused Torch complex `log1p` to return `NaN`. A
small-argument complex series fixed that implementation failure without changing
the Double Heston model. All 1,621 rows then priced finitely; representative rows
agreed with the canonical NumPy pricer to machine precision and with independent
adaptive quadrature.

The V2 cohort contains 131,072 accepted training rows, 16,384 validation rows,
18,000 collocation rows, and zero rejected proposals. The frozen artifact hashes
are:

| File | SHA-256 |
|---|---|
| `train.npz` | `92b5c70d8a66fc2996ed945b54ba9a80feb6768874151ab666aea54a3e2c6250` |
| `validation.npz` | `479673a31342f2dd27eb05ecf6e72913f233e0b4ea41153fe74f6dfb6dca2bd1` |
| `collocation.npz` | `730d7352df74075464c4443240f5de753371ef2bf6c2593a3f861727b3171c6f` |
| `rejections.npz` | `49ce6c347f1982d33565299eee73aff71b212a78f51e5e68092bf96ef0ed3039` |
| `manifest.json` | `c673e1bd190e3608ed0b5e544c48e73c36ea33a8276d074a69d57080be45bd3a` |

Two independent small-cohort generations were byte-identical. A deterministic
192-row beginning/middle/end replay exactly reproduced every stored array. B0,
B1, and B2 remain unrun on V2 at this checkpoint.
