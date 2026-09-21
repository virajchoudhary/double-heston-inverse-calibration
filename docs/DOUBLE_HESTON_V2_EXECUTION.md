# Double Heston V2 controlled execution

This runbook freezes the single-development-seed comparison. Do not change the
commands or decision thresholds after observing B0, B1, or B2.

## Transfer and precheck

Extract the transfer archive over a checkout of the committed branch. From the
repository root, run:

```bash
python scripts/mentor_dh_pinn/precheck_double_v2_mac.py \
  --root . \
  --report outputs/regular_pinn_recovery/double_v2_mac_precheck.json
```

The precheck records Git HEAD, platform, architecture, available memory, and the
Python, MLX, NumPy, SciPy, and PyTorch versions. It verifies every frozen input
hash and row count, then performs a finite MLX forward/backward update and an
exact checkpoint save/load round trip. Any discrepancy is a hard stop.

## B0

```bash
python scripts/mentor_dh_pinn/train_regular_pinn.py \
  --data outputs/regular_pinn_recovery/double_data_v2_reproducible \
  --out outputs/regular_pinn_recovery/double_sobolev_v2_b0_s17 \
  --factors 2 --steps 12000 --seed 17 --width 160 --depth 5 \
  --batch 1024 --pde-batch 256 --sensitivity 0.2 --weight-pde 0.2 \
  --weight-decay 1e-6 --lr 0.001 --resample-every 1000 --save-every 2000
```

## B1

```bash
python scripts/mentor_dh_pinn/finetune_regular_pinn_lbfgs.py \
  --checkpoint outputs/regular_pinn_recovery/double_sobolev_v2_b0_s17 \
  --data outputs/regular_pinn_recovery/double_data_v2_reproducible \
  --out outputs/regular_pinn_recovery/double_sobolev_v2_b1_s917017 \
  --seed 917017 --anchor-batch 4096 --pde-batch 256 \
  --sensitivity-weight 0.2 --pde-weight 0.2 --lr 0.3 --max-iter 20 \
  --history-size 20 --tolerance-grad 1e-9 --tolerance-change 1e-12
```

## B2

Start independently from B0, not B1.

```bash
python scripts/mentor_dh_pinn/finetune_regular_pinn_lbfgs.py \
  --checkpoint outputs/regular_pinn_recovery/double_sobolev_v2_b0_s17 \
  --data outputs/regular_pinn_recovery/double_data_v2_reproducible \
  --out outputs/regular_pinn_recovery/double_sobolev_v2_b2_s927721 \
  --seed 927721 --anchor-batch 512 --pde-batch 32 \
  --sensitivity-weight 0.2 --pde-weight 0.2 --lr 0.2 --max-iter 10 \
  --history-size 10 --tolerance-grad 1e-9 --tolerance-change 1e-12 \
  --jacobian-data outputs/regular_pinn_recovery/double_jacobian_surfaces_v2_927721/surfaces.npz \
  --jacobian-batch 2 --jacobian-weight 0.05 --weak-weight 2e-5 \
  --weak-directions 2 --jacobian-normalization-floor 1e-4
```

## Common evaluation

Run each command for `b0`, `b1`, and `b2`, substituting its checkpoint and a new
output path:

```bash
python scripts/mentor_dh_pinn/evaluate_regular_pinn_development.py \
  --checkpoint CHECKPOINT --out DEVELOPMENT_JSON --starts 3 --max-nfev 200
python scripts/mentor_dh_pinn/diagnose_regular_pinn_validation.py \
  --checkpoint CHECKPOINT --out VALIDATION_JSON
python scripts/mentor_dh_pinn/check_regular_pinn_physics.py \
  --checkpoint CHECKPOINT --out PHYSICS_JSON --seed 907611 --points 4096 --chunk 128
```

The primary engineering gate is either at least 15% lower recovery RMSE than B0
or at least one additional all-ten-parameter pass, without material deterioration
in IV RMSE, price RMSE, PDE validity, or structural validity. If neither candidate
passes, stop without trying more settings or seeds and report
`ARCHITECTURE_REDESIGN_JUSTIFIED = YES`. Otherwise preserve the winning candidate
and report `ARCHITECTURE_REDESIGN_JUSTIFIED = NO`.
