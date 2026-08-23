"""DEVELOPMENT benchmark for cloud migration decisions (NOT research results).

Measures, on the CURRENT machine (and GPU if visible):

- Model-2 training-step time (batch 64, float64 differentiable repricing,
  forward+backward+Adam step) — the dominant frozen-workload kernel;
- numerical equivalence of the vectorized pricer on each device vs the
  existing loop implementation and vs the production numpy pricer
  (frozen tolerance: max abs diff <= 1e-9 on dollar prices);
- float64 gradient stability on each device;
- optionally, traditional calibration seconds/surface on a few TRAIN surfaces.

Uses TRAIN-split surfaces only. Never reads test-split predictions or
metrics. Output: JSON report marked DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT.

Usage:
    python scripts/cloud/benchmark_cloud.py
    python scripts/cloud/benchmark_cloud.py --with-calibration
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import socket
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT))

from src.constants import PARAMETER_NAMES  # noqa: E402
from src.double_heston import price_double_heston_surface as production_price  # noqa: E402
from src.r2_primary.dataset import iter_r2_jsonl, _record_to_item  # noqa: E402
from src.torch_double_heston import (  # noqa: E402
    price_double_heston_surface_batch,
    price_double_heston_surface_batch_vectorized,
)

BATCH = 64
STEPS = 5
EQUIV_TOLERANCE = 1.0e-9


def _train_items(count: int):
    items = []
    for record in iter_r2_jsonl(REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"):
        if record["metadata"]["user_metadata"]["split"] == "train":
            items.append(_record_to_item(record))
        if len(items) >= count:
            break
    return items


def _batch_tensors(items, device):
    parameters = torch.tensor(
        np.stack([item.targets for item in items]), dtype=torch.float64, device=device
    )
    spots = torch.tensor([item.spot for item in items], dtype=torch.float64, device=device)
    strikes = torch.tensor(
        np.stack([item.strikes for item in items]), dtype=torch.float64, device=device
    )
    maturities = torch.tensor(
        np.stack([item.maturities for item in items]), dtype=torch.float64, device=device
    )
    rates = torch.tensor([item.rate for item in items], dtype=torch.float64, device=device)
    carries = torch.tensor([item.carry for item in items], dtype=torch.float64, device=device)
    option_types = [list(item.option_types) for item in items]
    observed = torch.tensor(
        np.stack([item.normalized_prices for item in items]),
        dtype=torch.float64,
        device=device,
    )
    mask = torch.tensor(
        np.stack([item.mask for item in items]).astype(np.float64),
        dtype=torch.float64,
        device=device,
    )
    return parameters, spots, strikes, maturities, rates, carries, option_types, observed, mask


def _model2_step_time(items, device) -> dict:
    from models.pinn_model import PhysicsInformedInverseCalibrator
    from src.r2_primary.training import build_model2, _repricing_loss

    device_obj = torch.device(device)
    model = build_model2().to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(), 5e-4, weight_decay=1e-5
    )
    features = torch.as_tensor(
        np.stack([item.features for item in items]), device=device_obj
    )
    args = _batch_tensors(items, device_obj)
    parameters, spots, strikes, maturities, rates, carries, option_types, observed, mask = args

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        repricing = price_double_heston_surface_batch_vectorized(
            predictions.to(torch.float64), spots, strikes, maturities, rates, carries,
            option_types, node_count=64,
        )
        normalized = repricing / spots.unsqueeze(1)
        active = mask
        loss = torch.sum(((normalized - observed) ** 2) * active) / active.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    losses = [step() for _ in range(2)]  # warmup
    started = time.perf_counter()
    for _ in range(STEPS):
        losses.append(step())
        torch.cuda.synchronize() if device_obj.type == "cuda" else None
    seconds = (time.perf_counter() - started) / STEPS
    finite = all(np.isfinite(value) for value in losses)
    return {
        "device": str(device_obj),
        "model2_step_seconds_mean": seconds,
        "model2_estimated_epoch_seconds_at_117_steps": seconds * 117,
        "losses_finite": finite,
        "first_last_loss": [losses[0], losses[-1]],
    }


def _pricer_equivalence(items, device) -> dict:
    device_obj = torch.device(device)
    parameters, spots, strikes, maturities, rates, carries, option_types, _, _ = (
        _batch_tensors(items, torch.device("cpu"))
    )
    loop = price_double_heston_surface_batch(
        parameters, spots, strikes, maturities, rates, carries, option_types, node_count=64
    ).detach().numpy()
    if device_obj.type == "cuda":
        vec = price_double_heston_surface_batch_vectorized(
            parameters.to(device_obj), spots.to(device_obj), strikes.to(device_obj),
            maturities.to(device_obj), rates.to(device_obj), carries.to(device_obj),
            option_types, node_count=64,
        ).detach().cpu().numpy()
    else:
        vec = price_double_heston_surface_batch_vectorized(
            parameters, spots, strikes, maturities, rates, carries, option_types, node_count=64
        ).detach().numpy()
    max_diff_loop = float(np.abs(vec - loop).max())
    production_max = 0.0
    for index in range(len(items)):
        item = items[index]
        produced = production_price(
            item.spot, item.strikes, item.maturities, item.rate, item.carry,
            item.option_types, item.targets, node_count=64,
        )
        production_max = max(
            production_max, float(np.abs(np.asarray(produced) - vec[index]).max())
        )
    trainable = parameters.clone().to(device_obj).requires_grad_(True)
    out = price_double_heston_surface_batch_vectorized(
        trainable,
        spots.to(device_obj), strikes.to(device_obj), maturities.to(device_obj),
        rates.to(device_obj), carries.to(device_obj), option_types, node_count=64,
    )
    (out.sum()).backward()
    grads_finite = bool(torch.isfinite(trainable.grad).all())
    return {
        "device": str(device_obj),
        "max_abs_diff_vs_loop_cpu": max_diff_loop,
        "max_abs_diff_vs_production_numpy": production_max,
        "float64_gradients_finite": grads_finite,
        "tolerance": EQUIV_TOLERANCE,
        "within_tolerance": max(max_diff_loop, production_max) <= EQUIV_TOLERANCE,
    }


def _calibration_timing(items) -> dict:
    from src.calibrate_double_heston import calibrate_double_heston

    timings = []
    for item in items[:2]:
        started = time.perf_counter()
        calibrate_double_heston(
            item.spot, item.strikes, item.maturities, item.rate, item.carry,
            item.option_types, item.dollar_prices, item.targets,
            REPO_ROOT / "configs" / "parameter_bounds_PROVISIONAL.yaml",
            node_count=64, max_nfev=300, seed=42,
        )
        timings.append(time.perf_counter() - started)
    return {
        "calibration_seconds_per_surface_mean_3starts": float(np.mean(timings)),
        "surfaces_measured": len(timings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-calibration", action="store_true")
    args = parser.parse_args()

    print("DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT")
    items = _train_items(BATCH)
    report: dict = {
        "marker": "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT",
        "purpose": "execution_environment_benchmark_only",
        "host": socket.gethostname(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_built": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "batch_size": BATCH,
        "steps": STEPS,
    }
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["cuda_compute_capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(0))
        )
    report["model2_step"] = [_model2_step_time(items, device) for device in devices]
    report["pricer_equivalence"] = [_pricer_equivalence(items, device) for device in devices]
    if args.with_calibration:
        report["calibration"] = _calibration_timing(items)

    output = (
        REPO_ROOT / "evidence" / "r2_primary_comparison_20260823" /
        f"cloud_benchmark_{socket.gethostname()}.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
