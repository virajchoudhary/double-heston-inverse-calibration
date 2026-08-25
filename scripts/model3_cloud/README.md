# Model 3 cloud execution package

These tools are execution infrastructure only. They do not authorize Stage B.

## Stage A external GPU launch

Run only on an already-authorized single-GPU session after cloning this branch
at the exact final pushed commit recorded in the Stage-A report. The launcher
fails closed before allocation or training if Git identity, tracked-tree
cleanliness, protocol hash, dataset hash, Python dependencies, CUDA
availability, device identity, compute capability, or memory preflight does not
pass.

```bash
PYTHONPATH=. python scripts/model3_cloud/preflight.py --require-cuda \
  --expected-git-sha <FINAL_PUSHED_STAGE_A_SHA>
PYTHONPATH=. python scripts/model3_cloud/launch_stage_a.py \
  --expected-git-sha <FINAL_PUSHED_STAGE_A_SHA> \
  --output-root outputs/model3_pde_development_pilot
```

The launcher records preflight, console, and command transcripts beside—not
inside—the run directory, then invokes the exact frozen Stage-A driver with
CUDA. Existing run directories are never overwritten; rerunning the driver
itself resumes only a complete checkpoint/optimizer pair. Supply new sidecar
paths/prefixes on every launch attempt.

After completion or interruption, preserve the entire directory and hash it:

```bash
PYTHONPATH=. python scripts/model3_cloud/package_outputs.py \
  outputs/model3_pde_development_pilot --failed
```

Omit `--failed` only for a completed run. Return `artifact_manifest.json`, the
directory, and all three sidecar evidence files. The receiver must run
`--verify-only` before scientific intake.

## Frozen Stage-B preparation

This emits three isolated seed packages without executing anything:

```bash
PYTHONPATH=. python scripts/model3_cloud/prepare_stage_b.py \
  --output handoff/model3_stage_b_seed_packages.json
```

Seeds 11, 22, and 33 have separate output roots. Shared settings are fixed at
7,500 train / 1,250 validation surfaces, batch 32, 32 interior points, eight
terminal points, AdamW learning rate 0.0002, weight decay 0.00001, maximum 120
epochs, patience 15 evaluations, minimum-validation-total checkpoint selection,
and explicit CUDA placement. The untouched test split remains closed.
