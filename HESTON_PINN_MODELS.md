# Trained Heston PINNs — single factor and unified two factor

Everything here is **ready to load**. No retraining is required for any result in this branch.

## What is shipped

| model | checkpoint | what it does |
|---|---|---|
| **Unified Double Heston calibrator** | `outputs/unified_v6/unified.pt` | arbitrary quote set -> 10 parameters + full covariance + OOD status, with the exact Fourier engine refined inside `forward()` |
| **Projection fine-tuned** | `outputs/unified_v6/unified_ft.pt` | **best on multi-expiry real data** — beats classical Double Heston on NIFTY at 21.7x the speed. Identical architecture; load it exactly like `unified.pt` |
| Unified, second run | `outputs/unified_v6/unified_v2.pt` | retained for comparison only — **it is the worse model**, see below |
| Single Heston PINN | `single_heston_pinn/outputs/pinn_model_round1.safetensors` | forward pricing PINN; `physics_only` and `physics_and_anchor` ablations alongside |
| Dual PINN (previous architecture) | `outputs/dual_pinn_legacy/dual_{aware,independent}.pt` | the two-specialist design the unified model replaces |

### Which checkpoint to use

* **Multi-expiry surfaces (index options, several expiries):** `unified_ft.pt`. On NIFTY it
  reaches 0.02478 holdout IV RMSE against classical Double Heston's 0.02523, better than the
  base model on 9 of 10 dates (Wilcoxon p = 0.006), in 1.16 s against 25.26 s.
* **Single-expiry surfaces:** either checkpoint; the fine-tune has no measurable effect there
  (74/140 surfaces, p = 0.55) and both remain about 2.6x behind classical calibration. Use
  classical multi-start if accuracy matters more than latency.
* **Synthetic / parameter-recovery work:** `unified.pt`, which the reported metrics use.

**Use `unified.pt`, not `unified_v2.pt`.** v2 was trained later and through a corrected
physics layer, and I expected it to be better. It is not. On the held-out test split:

| model | refine steps | parameter RMSE | reprice |
|---|---:|---:|---:|
| **unified.pt** | **3** | **0.14699** | **6.4113e-04** |
| unified_v2.pt | 3 | 0.16370 | 7.7621e-04 |

Every reported number comes from `unified.pt`. The caveat is that this is one run against
one run, confounded with random initialisation; see `docs/mentor_dh_pinn/DELIVERABLE_F_FINAL_REPORT.md` section F.6.

## Load the unified Double Heston calibrator

```python
import torch
from src.mentor_dh_pinn.unified import UnifiedCalibrator

ck = torch.load("outputs/unified_v6/unified.pt", weights_only=False)
c = ck["config"]
model = UnifiedCalibrator(d_model=c["d_model"], rounds=c["rounds"], node_count=c["nodes"])
model.load_state_dict(ck["state_dict"]); model.eval()

# batch: spot, strike, tau, rate, carry, price, mask  (B, N) + noise_level (B,)
# N is whatever the market has -- one expiry or ninety irregular quotes are both valid
with torch.no_grad():
    out = model(batch, refine_steps=3)

out["params"]      # (B, 10) after exact-physics refinement  <- use this
out["params_pre"]  # (B, 10) network only, before refinement
out["L"]           # (B, 10, 10) Cholesky factor; Sigma = L @ L.T
```

`torch.set_default_dtype(torch.float64)` first — the characteristic function cancels
catastrophically at long maturity and float32 loses the leading digits.

## Data

| file | size | purpose |
|---|---:|---|
| `outputs/unified_v6/v6_test.npz` | 11 MB | 25,000 held-out surfaces — reproduces every reported metric |
| `outputs/unified_v6/v6_validation.npz` | 11 MB | validation split |
| `single_heston_pinn/outputs/pinn_collocation_*.npz` | 1.8 MB each | single Heston collocation points |
| `single_heston_pinn/outputs/pinn_quote_panel.parquet` | 2.8 MB | leakage-safe NSE quote panel |

**`v6_train.npz` (64 MB) is deliberately not shipped.** It is needed only to *train*, which
nobody has to do. It regenerates exactly — the seeds are fixed — in about 157 seconds:

```bash
python scripts/mentor_dh_pinn/build_v6_data.py   # train seed 101, validation 202, test 303
```

## Reproduce the metrics (no training)

```bash
python scripts/mentor_dh_pinn/evaluate_unified.py --n-full 4000 --n-baseline 100
python scripts/mentor_dh_pinn/evaluate_real_markets.py
python scripts/mentor_dh_pinn/run_ablations.py --n 1200
python -m unittest discover -s tests -p 'test_unified*.py'     # 22 tests
```

Results already stored in `outputs/unified_v6/*.json` and `outputs/real_markets/`.

## Read first

`docs/mentor_dh_pinn/DELIVERABLE_F_FINAL_REPORT.md` — the full report, including the three
defects found during development and the two headline findings, both of which are negative:
the model loses to classical calibration on real markets, and the retrain I predicted would
help made things worse.
