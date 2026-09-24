#!/bin/zsh
PY="/Users/dhruvaambhaikar/Documents/Options pricing/.venv/bin/python"
cd "$(dirname "$0")"
for wave in "A0 A1 A2" "A3 A4 A5"; do
  for s in ${=wave}; do
    "$PY" train.py $s > artifacts/logs/train_$s.log 2>&1 &
  done
  wait
done
echo ALL_ABLATION_DONE
