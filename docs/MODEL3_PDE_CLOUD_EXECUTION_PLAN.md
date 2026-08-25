# Model 3 PDE cloud execution plan

Status: `MODEL3_PDE_PILOT_READY`; prepared but not
executed. This plan defines environment, identity gates, pilot design,
checkpoint handling, resumability, and artifact return. It does not authorize
the pilot.

## Execution decision

Use a GPU cloud session, not the local laptop. Model 3 physics uses float64
second derivatives, so accelerator memory and determinism matter more than peak
FP16 throughput.

Recommendation:

1. **Kaggle T4/P100** for the short development pilot: simple snapshotting and
   sufficient memory; expect below 8 GiB accelerator memory at the frozen pilot
   shape.
2. **Isolated single-GPU cloud VM** for each research seed when persistence and
   long-session control are required.
3. Colab is acceptable only if disconnect/resume behavior is tested first.

Do not split one seed across inconsistent dependency environments. Do not use
multi-GPU training; it adds nondeterministic coupling without a predeclared
scientific need.

## Environment

Pin and record:

```text
Python 3.13.4
NumPy 2.2.6
pandas 2.3.2
SciPy 1.16.2
PyYAML 6.0.2
PyTorch 2.11.0 CUDA build matching the cloud driver
```

Record `python --version`, `nvidia-smi`, `pip freeze`, Git SHA, working-tree
status, dataset SHA-256, and config SHA-256 before allocation. Do not upgrade a
dependency merely because a newer release exists.

## Identity gate

Run before GPU allocation:

```bash
test "$(git rev-parse HEAD)" = "<REVIEWED_MODEL3_COMMIT_SHA>"
test -z "$(git status --porcelain -- configs/model3_pde_protocol.yaml data/final_r2_clean_10000/surfaces.jsonl src/double_heston.py)"
sha256sum configs/model3_pde_protocol.yaml
sha256sum data/final_r2_clean_10000/surfaces.jsonl
sha256sum configs/r2_primary_comparison_FINAL.yaml
python -m pytest -q tests/test_model3_pde_foundation.py
python scripts/run_model3_pde_smoke.py
```

Expected dataset hash:

```text
148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6
```

Replace `<REVIEWED_MODEL3_COMMIT_SHA>` with the pushed commit recorded in the
final Model 3 report. Stop if any check differs.

## Transfer

Clone the repository and check out the reviewed Model 3 commit. The final 10k
JSONL and configs are tracked repository inputs. Existing Model 1/Model 2
checkpoints are unnecessary for Model 3 training. Never copy the excluded
`model2_seed11_local_cpu_replication`, superseded positive-noise artifacts, or
real-market weight-update paths into the pilot workspace.

## Pilot execution contract

The reviewed milestone deliberately does not start training. The thin
implementation-only pilot driver is now implemented and independently reviewed.
It loads through `src.r2_primary.dataset.R2PrimaryDataset`, fits
target scaling on train only, instantiates `Model3PDESystem`, uses the configured
AdamW settings and deterministic samplers, and emits diagnostics. It contains
no test-split access and no real-market update path.

The driver command must have this exact public shape:

```bash
PYTHONPATH=. python scripts/run_model3_pde_pilot.py \
  --dataset data/final_r2_clean_10000/surfaces.jsonl \
  --output-root outputs/model3_pde_development_pilot \
  --train-limit 240 \
  --validation-limit 40 \
  --seed 4207 \
  --epochs 3 \
  --batch-size 16 \
  --interior-points 16 \
  --terminal-points 8 \
  --learning-rate 0.0002 \
  --weight-decay 0.00001
```

If that driver is absent, the status is not launch-ready. Do not improvise a
different training loop on a cloud console.

## Expected pilot cost and memory

The pilot processes at most 240 train and 40 validation surfaces for three
epochs. With batch 16, 16 interior points, and eight terminal points, one T4 or
P100 with 16 GiB is expected to remain below 8 GiB. Report measured peak memory;
do not infer it solely from this estimate. If memory exceeds 12 GiB, stop rather
than reducing scientific precision or physics points.

## Checkpoint and resume policy

Each epoch writes atomically to a temporary path and renames only after success:

```text
checkpoint.pt
optimizer.pt
epoch_metadata.json
train_history.csv
validation_history.csv
physics_diagnostics.csv
gradient_diagnostics.csv
environment_provenance.json
```

On restart:

1. verify Git, config, and dataset identities again;
2. load the latest complete epoch only;
3. reject a checkpoint whose recorded hashes differ;
4. resume optimizer and sampler state;
5. append history rather than rewriting prior rows.

Upload or snapshot the entire epoch directory after every epoch. A partially
transferred directory is never resumed.

## Research-run outline

After the pilot passes numerical-only diagnostics, run one process per seed
11,22,33. Use batch 32, 32 interior points, eight terminal points, maximum 120
epochs, patience 15, AdamW learning rate `0.0002`, and weight decay `0.00001`.
Select by minimum validation total loss only. Keep test closed until all three
final checkpoint manifests are verified.

## Artifact return and acceptance

Return the complete output directory, raw logs, checksums, and console
transcript. Before accepting results:

1. independently recompute every file hash;
2. confirm recorded Git/config/data identities;
3. reload each checkpoint and rerun validation diagnostics;
4. require finite losses/gradients and complete histories;
5. keep `.pt` files out of Git according to repository policy;
6. commit only summaries, manifests, small CSV/JSON diagnostics, and logs.

Never overwrite frozen primary evidence or Issue #34 evidence. A failed or
interrupted pilot remains development evidence and is not deleted.
