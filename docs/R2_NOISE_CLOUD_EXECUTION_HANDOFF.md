# R2 noise cloud execution handoff

Status: prepared but not executed. This is the handoff for the remaining
CPU-bound traditional calibration only. Do not change any scientific setting,
checkpoint, cohort, threshold, optimizer option, subset, or frozen protocol.

## Execution identity

- Repository: `C:\ann_inverse_calibration`
- Branch: `codex/r2-noise-recovery`
- Exact commit to run from: `b94447d71d8418a218e0e5f2a91ba807f2dcb687`
- Origin branch at preparation time: `b94447d71d8418a218e0e5f2a91ba807f2dcb687`
- Frozen protocol commit: `91b66b63af7fbda6ad425fe7beeddf045e6b99c0`
- Frozen config:
  `configs/r2_noise_robustness_FINAL.yaml`
- Frozen config SHA-256:
  `2fa49b3eb885d3427c01ab0cfe447fc6ddd7f19957db73c4b4ed782476c57c5a`
- Frozen clean dataset:
  `data/final_r2_clean_10000/surfaces.jsonl`
- Clean dataset SHA-256:
  `148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`
- Canonical primary merge commit:
  `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`

## Positive-noise cohort identity

The cohorts were validated against `data/r2_noise_robustness/MANIFEST.json`.
All four are test-split-only files with exactly 1,250 records and no positive
price-resample events.

| Level | SHA-256 |
|---|---|
| 0.10% | `bdee2dd02fc3333b8bd7900358933b845154db378f568f7be77321d255c68197` |
| 0.25% | `67cbb4c2c303e1a4f2cecabdf1625e52f9c39ba35e09fc6665317cba20f49a0c` |
| 0.50% | `186400092d016a2e2a305616a035d9b8b72ab624dad653d2560f1fe68149217f` |
| 1.00% | `7e0f6f948f1d2b13505b9e147214d5cfc8bc82fe4c82a1942323fa0ac4690368` |

The neural evaluation over all five levels, both models, and seeds
11/22/33 is already committed at `4ffddc2b9170ba494f75cedb2f5740cb806558c9`.
Its manifest is `evidence/r2_noise_robustness/neural/MANIFEST.json`; all listed
artifact hashes passed verification before this handoff was prepared.

## Frozen traditional subset

- Artifact:
  `evidence/r2_noise_robustness/traditional_subset_ids.json`
- SHA-256:
  `f856ef5ffcc33782a115180ca7cb7b1f4cfa4ebeb8fd1af45c7cde242c85aba7`
- Population: exactly 250 unique surfaces from the 1,250-surface untouched
  R2 test split.
- Selection: deterministic proportional allocation across nine strata formed
  by terciles of clean-truth `v0_total = v0_slow + v0_fast` and clean-truth
  `kappa_slow`, with ascending SHA-256 surface-ID ordering inside each cell.
- No RNG and no model outcome participates in selection.

## Exact traditional settings

These values come from `FROZEN_SETTINGS` in
`src/r2_primary/calibration.py` and are repeated in each journal provenance.

| Setting | Value |
|---|---|
| Module | `src/calibrate_double_heston.py` |
| Pricer | production pricer `src/double_heston.py`, unchanged |
| Optimizer | `trf` |
| Starts per surface | 3 |
| Start seed | 42 |
| Maximum function evaluations | 300 |
| `ftol` / `xtol` / `gtol` | each `1.0e-10` |
| `diff_step` | `2.0e-05` |
| Pricer node count | 64 |
| Residual scale | `max(observed_dollar_price, 1.0)` |
| Bounds | `configs/parameter_bounds_PROVISIONAL.yaml` |
| Representative rule | lowest final objective; ties by lowest start index |

Every start, including failures and boundary hits, must be retained.
Optimizer success is not interpreted as unique parameter recovery.

## Current traditional state

The 0% subset gate is complete and passed with all 750 starts retained. The
interrupted 0.10% run has a crash-safe journal containing 164 unique surfaces
and 492 starts. It was preserved at execution commit `b94447d`. There is no
final CSV for 0.10%, so it remains incomplete rather than complete.

- Partial provenance:
  `evidence/r2_noise_robustness/traditional/level_0_10pct/JOURNAL_PROVENANCE.json`
- Partial journal:
  `evidence/r2_noise_robustness/traditional/level_0_10pct/traditional_calibration_starts_journal.jsonl`

Remaining work:

1. Resume 0.10%: 86 of 250 surfaces (258 additional optimizer starts).
2. Run 0.25%, 0.50%, and 1.00%: 250 fresh surfaces and 750 starts each.

In total, the remaining work is up to 836 surface-level completions and 2,508
optimizer starts.

## Environment

Use Linux or Windows x86-64 with Python 3.13. The local strict-gate recovery
environment used Python 3.13.4, NumPy 2.2.6, pandas 2.3.2, SciPy 1.16.2,
PyYAML 6.0.2, and CPU PyTorch 2.11.0. Preserve these versions where practical;
do not upgrade NumPy, SciPy, or PyTorch merely to modernize.

Recommended cloud VM:

- At least 16 dedicated vCPUs for one run; 32 vCPUs if memory permits.
- 8 GB RAM or more.
- 20 GB persistent disk after repository/checkpoint transfer.
- A normal general-purpose CPU VM, not a GPU-only runner.
- Persistent disk or an attached object-store mount so the JSONL journal can
  survive interruption.

Kaggle and Colab can execute the same command, but scheduled termination,
nonpersistent local disks, variable core counts, and upload/download friction
make full completion less reliable. Kaggle commonly exposes more CPU than a
free Colab session, but neither is preferable to a multicore persistent VM for
this workload. If a notebook platform is unavoidable, prefer Kaggle with
frequent output download/snapshotting and keep `--workers` within its actual
available cores.

The workload is CPU-bound. A GPU does not accelerate the frozen SciPy
calibration.

## Checkpoint transfer

The six intended checkpoint directories are not Git-tracked. Transfer them
separately and never include the excluded local replication or smoke directory.

From the repository root on the source machine:

```bash
tar --exclude='checkpoints/r2_primary_comparison/model2_seed11_local_cpu_replication' \
    --exclude='checkpoints/r2_primary_comparison/smoke' \
    -czf r2_primary_checkpoints.tar.gz checkpoints/r2_primary_comparison
sha256sum r2_primary_checkpoints.tar.gz > r2_primary_checkpoints.tar.gz.sha256
```

Before execution, record and compare both archive hashes. After extraction,
the execution root must contain exactly these six directories under
`checkpoints/r2_primary_comparison`:

- `model1_seed11`
- `model1_seed22`
- `model1_seed33`
- `model2_seed11`
- `model2_seed22`
- `model2_seed33`

Do not add, rename, remove, or regenerate checkpoint contents.

## Cloud setup

```bash
git clone git@github.com:virajchoudhary/double-heston-inverse-calibration.git
cd double-heston-inverse-calibration
git checkout b94447d71d8418a218e0e5f2a91ba807f2dcb687

python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import numpy, pandas, scipy, yaml, torch
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
print('scipy', scipy.__version__)
print('PyYAML', yaml.__version__)
print('torch', torch.__version__)
PY
```

Extract the six-checkpoint archive at the repository root. Then verify the
execution commit and immutable inputs:

```bash
test "$(git rev-parse HEAD)" = b94447d71d8418a218e0e5f2a91ba807f2dcb687
sha256sum configs/r2_noise_robustness_FINAL.yaml
sha256sum data/final_r2_clean_10000/surfaces.jsonl
sha256sum evidence/r2_noise_robustness/traditional_subset_ids.json
python scripts/run_r2_noise_robustness.py validate
```

Expected hashes are recorded above. Stop without running calibration if any
identity check differs.

## Execution commands

Run levels serially. Choose a worker count no greater than physically available
dedicated cores. Ten workers reproduce the local process layout; sixteen are
suitable on a dedicated 16-vCPU VM. Do not oversubscribe to reduce wall time.

```bash
python scripts/run_r2_noise_robustness.py run-traditional --level '0.10%' --workers 16
python scripts/run_r2_noise_robustness.py run-traditional --level '0.25%' --workers 16
python scripts/run_r2_noise_robustness.py run-traditional --level '0.50%' --workers 16
python scripts/run_r2_noise_robustness.py run-traditional --level '1.00%' --workers 16
```

Alternatively, `--levels positive` resumes 0.10% and then runs the remaining
positive levels serially:

```bash
python scripts/run_r2_noise_robustness.py run-traditional --levels positive --workers 16
```

Do not use `--levels all`: the 0% final bundle already exists and must remain
immutable.

## Crash safety and resume

For each level, `run_traditional_calibration` appends one JSONL line after a
surface completes all three starts and flushes the file. On restart it reads
those surface IDs and submits only missing surfaces. It writes the final CSV
only after every requested surface is present.

If interrupted:

1. Do not edit or truncate the journal.
2. Re-run the exact same command and worker policy after verifying HEAD and
   input hashes.
3. Confirm the printed `resuming:` count equals completed unique surfaces.
4. Never manually append rows or recover a partially completed surface.

A partial journal has scientific value as execution evidence, but only the
runner may turn it into a final CSV after all 250 IDs are complete.

## Verification after each level

Immediately after a level completes, run:

```bash
LEVEL_DIR="evidence/r2_noise_robustness/traditional/level_0_10pct" # replace per level
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

level_dir = Path('evidence/r2_noise_robustness/traditional/level_0_10pct')
journal = level_dir / 'traditional_calibration_starts_journal.jsonl'
starts_csv = level_dir / 'traditional_calibration_starts.csv'
results_csv = level_dir / 'traditional_calibration_results.csv'
summary_path = level_dir / 'run_summary.json'

with journal.open(encoding='utf-8') as handle:
    lines = [json.loads(line) for line in handle if line.strip()]
frame = pd.read_csv(starts_csv)
summary = json.loads(summary_path.read_text(encoding='utf-8'))
assert len(lines) == 250
assert len({row['surface_id'] for row in lines}) == 250
assert len(frame) == 750
assert frame['surface_id'].nunique() == 250
assert results_csv.exists()
assert summary['surfaces_calibrated'] == 250
assert summary['starts_recorded'] == 750
assert summary['status'] == 'COMPLETE_FAILED_STARTS_RETAINED'
print('LEVEL_VERIFIED', summary['noise_level_label'])
PY
sha256sum "$LEVEL_DIR/traditional_calibration_starts.csv" \
  "$LEVEL_DIR/traditional_calibration_results.csv" \
  "$LEVEL_DIR/traditional_calibration_starts_journal.jsonl"
```

For later levels replace `level_0_10pct` with `level_0_25pct`,
`level_0_50pct`, or `level_1_00pct`.

Also hash the unchanged frozen config and clean dataset after each level:

```bash
sha256sum configs/r2_noise_robustness_FINAL.yaml \
  data/final_r2_clean_10000/surfaces.jsonl \
  evidence/r2_noise_robustness/traditional_subset_ids.json
```

Stop if any hash changes.

## Commit procedure

Commit one level at a time only after its verification block passes. Use the
same branch and do not rebase or amend completed evidence commits.

For example, after 0.10% completes:

```bash
git checkout codex/r2-noise-recovery
git pull --ff-only origin codex/r2-noise-recovery
git add evidence/r2_noise_robustness/traditional/level_0_10pct
git diff --cached --check
git commit -m "research(models): complete R2 traditional 0.10pct noise level"
git push origin codex/r2-noise-recovery
```

Repeat with:

- `level_0_25pct` / message `complete R2 traditional 0.25pct noise level`
- `level_0_50pct` / message `complete R2 traditional 0.50pct noise level`
- `level_1_00pct` / message `complete R2 traditional 1.00pct noise level`

After the last level, run aggregation locally or on the same VM:

```bash
python scripts/run_r2_noise_robustness.py aggregate
git status --short
```

Stage only newly generated aggregate artifacts explicitly named by that
command, inspect the diff, verify no frozen protocol/config/subset/cohort file
changed, and commit/push them separately.

## Return-to-repository checks

Before treating any returned evidence as canonical:

1. Verify execution commit `b94447d...` appears in its history.
2. Verify all frozen hashes still match this document.
3. For each completed level, require 250 unique journal records, 750 starts,
   250 representative rows, and a valid `run_summary.json`.
4. Reject any bundle with edited journals, rewritten historical rows, changed
   runtime settings, or missing failed starts.
5. Recompute hashes after transfer and preserve the original cloud logs and
   checksums outside the evidence tree if needed.
6. Do not interpret superseded contaminated-branch positive evidence.

## Workload estimate

The interrupted 0.10% sample averaged about 349.14 core-seconds per surface.
At that observed rate, the remaining 836 surface-level completions require
roughly 81 core-hours. This is not a wall-clock promise. The completed 0%
gate averaged about 880.85 core-seconds per surface; if other noisy levels
approach that cost, remaining work could approach 184 core-hours. The frozen
protocol's conservative planning estimate was approximately 130 CPU-hours
across all levels. Report actual per-start runtime rather than tuning budgets
to fit elapsed time.

## Research-status snapshot

Completed:

- BS vs Heston vs Double Heston comparison
- identifiability diagnostics
- final R2 representation
- clean 10k dataset
- Traditional vs ANN vs Model 2 primary comparison
- strict 0% noise reproduction
- deterministic positive-noise cohorts
- full-population neural positive-noise inference

Current:

- positive-noise robustness, now only traditional subset calibration

Future:

- genuine PDE-informed Model 3
- OOD/boundary robustness
- final unseen real-market evaluation

No factual contradiction was found in the requested frontier list. The current
work remains bounded to the frozen observation-noise protocol; Model 2 remains
constraint plus differentiable-repricing-informed, not a PDE-informed PINN.
