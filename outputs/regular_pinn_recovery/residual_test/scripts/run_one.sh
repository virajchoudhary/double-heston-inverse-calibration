#!/bin/bash
cd "/Users/dhruvaambhaikar/Documents/Options pricing/double-heston-v2-controlled" || exit 1
name=$1; shift
"/Users/dhruvaambhaikar/Documents/Options pricing/.venv/bin/python" scripts/mentor_dh_pinn/train_regular_pinn.py \
  --data outputs/regular_pinn_recovery/double_data_v2_reproducible \
  --out outputs/regular_pinn_recovery/residual_test/$name \
  --factors 2 --steps 12000 --width 160 --batch 1024 --pde-batch 256 --sensitivity 0.2 --weight-pde 0.2 \
  --weight-decay 1e-6 --lr 0.001 --resample-every 1000 --save-every 2000 "$@" \
  > outputs/regular_pinn_recovery/residual_test/logs/$name.log 2>&1
echo "$name exit=$?"
