"""Run the bounded NTPC canonical-160 causal provenance probes.

This script runs only canonical start 6 at the frozen ``max_nfev=160`` budget.
It never runs the optimizer-cap experiment and never writes reviewed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_ntpc_dh_stability_reparameterization as geometry
from scripts import run_ntpc_single_stock_pilot as pilot
from src.calibrate_double_heston import load_hard_safety_bounds, unconstrained_to_parameters
from src.constants import PARAMETER_NAMES


MAX_NFEV = 160
START_ID = 6
REVIEWED_SHA256 = "4E092F2BEC5F53033E61EFB1D2B2D761C9D3AB8F72F17F33D6E989946FC1EB70"
REPLAY_SHA256 = "CF148D54639EA194E620BABB5E6CF741A91AB77C269A2C1E3BA3CFA25B33926E"
SINGLE_FIELD_PROBES = (
    "T",
    "log_moneyness",
    "discount_factor",
    "continuous_rate",
    "futures_implied_carry",
    "market_implied_volatility",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fit(frame: pd.DataFrame, start: np.ndarray, bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    calibration = frame.loc[frame["sample_role"] == "CALIBRATION"].copy()
    observed = calibration["observed_price"].to_numpy(float)

    def residual(z: np.ndarray) -> np.ndarray:
        parameters = unconstrained_to_parameters(z, bounds)
        return geometry.price_rows(calibration, parameters) - observed

    result = least_squares(
        residual,
        start,
        method="trf",
        max_nfev=MAX_NFEV,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        diff_step=2e-5,
    )
    parameters = unconstrained_to_parameters(result.x, bounds)
    rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return {
        "parameters": {name: float(value) for name, value in zip(PARAMETER_NAMES, parameters, strict=True)},
        "parameter_sha256": hashlib.sha256(parameters.tobytes()).hexdigest().upper(),
        "calibration_price_rmse": rmse,
        "nfev": int(result.nfev),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
    }


def _vector(row: pd.Series) -> np.ndarray:
    return row[list(PARAMETER_NAMES)].to_numpy(dtype=np.float64)


def _decorate(result: dict[str, Any], reviewed: np.ndarray, replay: np.ndarray) -> dict[str, Any]:
    vector = np.asarray([result["parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    result["maximum_abs_difference_from_reviewed"] = float(np.max(np.abs(vector - reviewed)))
    result["maximum_abs_difference_from_replay"] = float(np.max(np.abs(vector - replay)))
    return result


def run(reviewed_path: Path, replay_path: Path, output_path: Path) -> dict[str, Any]:
    if sha256(reviewed_path) != REVIEWED_SHA256 or sha256(replay_path) != REPLAY_SHA256:
        raise RuntimeError("causal-probe endpoint artifact provenance mismatch")
    geometry.verify_baseline_contract()
    _, rebuilt, _ = pilot.build_option_dataset()
    csv_loaded = pd.read_csv(pilot.OUTPUT_ROOT / "selected_options.csv")
    if rebuilt["source_sha256"].tolist() != csv_loaded["source_sha256"].tolist():
        raise RuntimeError("rebuilt and serialized rows do not share source provenance")
    if rebuilt["sample_role"].tolist() != csv_loaded["sample_role"].tolist():
        raise RuntimeError("rebuilt and serialized row ordering differs")

    reviewed_row = pd.read_csv(reviewed_path).set_index("start_id").loc[START_ID]
    replay_row = pd.read_csv(replay_path).set_index("start_id").loc[START_ID]
    reviewed_vector, replay_vector = _vector(reviewed_row), _vector(replay_row)
    bounds = load_hard_safety_bounds(pilot.BOUNDS_PATH)
    _, _, starts = geometry.paired_start_population(bounds)
    start = starts.loc[starts["start_id"] == START_ID, [f"baseline_z_{i}" for i in range(10)]].iloc[0].to_numpy(float)

    cases: dict[str, Any] = {
        "rebuilt_in_memory": _decorate(_fit(rebuilt, start, bounds), reviewed_vector, replay_vector),
        "csv_loaded": _decorate(_fit(csv_loaded, start, bounds), reviewed_vector, replay_vector),
    }
    for field in SINGLE_FIELD_PROBES:
        probe = csv_loaded.copy()
        probe[field] = rebuilt[field].to_numpy()
        cases[f"csv_with_rebuilt_{field}"] = _decorate(_fit(probe, start, bounds), reviewed_vector, replay_vector)
    repeats = [_decorate(_fit(csv_loaded, start, bounds), reviewed_vector, replay_vector) for _ in range(3)]

    result = {
        "analysis_id": "NTPC_CANONICAL160_CAUSAL_PROBE",
        "max_nfev": MAX_NFEV,
        "start_id": START_ID,
        "reviewed_artifact": {"path": str(reviewed_path), "sha256": sha256(reviewed_path)},
        "replay_artifact": {"path": str(replay_path), "sha256": sha256(replay_path)},
        "selected_options_sha256": sha256(pilot.OUTPUT_ROOT / "selected_options.csv"),
        "source_row_sha256_values": sorted(set(rebuilt["source_sha256"].astype(str))),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "optimizer_contract": {
            "method": "trf", "max_nfev": MAX_NFEV, "ftol": 1e-9,
            "xtol": 1e-9, "gtol": 1e-9, "diff_step": 2e-5,
        },
        "maximum_input_field_differences": {
            field: float(np.max(np.abs(rebuilt[field].to_numpy(float) - csv_loaded[field].to_numpy(float))))
            for field in SINGLE_FIELD_PROBES
        },
        "cases": cases,
        "csv_repeat_fits": repeats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reviewed, args.replay, args.output), indent=2))
