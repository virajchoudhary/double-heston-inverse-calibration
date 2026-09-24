#!/bin/zsh
PY="/Users/dhruvaambhaikar/Documents/Options pricing/.venv/bin/python"
cd "$(dirname "$0")"
for seed in 17 43; do "$PY" train.py FINAL --full --seed $seed > artifacts/logs/train_FINAL_s$seed.log 2>&1; done
echo FINAL_TRAINING_DONE
