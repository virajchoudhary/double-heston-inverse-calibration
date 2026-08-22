"""Synthetic calibration cells for the G2 R2-vs-R3 study.

One cell = (truth, representation, noise level) and runs the frozen twelve
deterministic starts under the existing G2-ambiguity optimizer convention:
TRF least-squares, unweighted absolute residuals on spot-normalized prices,
latent parameterization via ``src.calibrate_double_heston.unconstrained_to_parameters``
(hard-by-construction constraints), identical optimizer settings for R2 and R3.

Every start is retained — failures, boundary hits, and all.
Results are appended incrementally to JSONL so interruption never loses work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..calibrate_double_heston import (
    boundary_diagnostics,
    unconstrained_to_parameters,
)
from ..constants import PARAMETER_NAMES
from . import frozen, noise as noise_module
from .geometry import DateProfile, build_geometry, representation_slots
from .pricer import normalized_observables
from .truths import load_bounds

BOUNDS = load_bounds()
WIDTHS = np.asarray([BOUNDS[name][1] - BOUNDS[name][0] for name in PARAMETER_NAMES])
LOWERS = np.asarray([BOUNDS[name][0] for name in PARAMETER_NAMES])


def scaled_coordinates(parameters: np.ndarray) -> np.ndarray:
    return (np.asarray(parameters, dtype=np.float64) - LOWERS) / WIDTHS


def clean_observables(
    truth_vector: np.ndarray, slots, *, spot: float = frozen.SYNTHETIC_SPOT
) -> np.ndarray:
    geometry = build_geometry(slots, spot=spot)
    return normalized_observables(
        truth_vector,
        geometry["strikes"],
        geometry["maturities"],
        geometry["option_types"],
        geometry["rates"],
        geometry["dividends"],
        spot=spot,
        node_count=frozen.NODE_COUNT,
    )


def observed_observables(
    truth_vector: np.ndarray,
    truth_id: str,
    slots,
    noise_level: float,
    *,
    spot: float = frozen.SYNTHETIC_SPOT,
) -> np.ndarray:
    """Keyed-noise-perturbed spot-normalized prices aligned to ``slots``."""
    clean = clean_observables(truth_vector, slots, spot=spot)
    observed = np.asarray(
        [
            noise_module.perturb_slot(
                float(value), truth_id, slot.rank, slot.moneyness, slot.option_type, noise_level
            )
            for value, slot in zip(clean, slots)
        ],
        dtype=np.float64,
    )
    if np.any(observed < 0.0):
        raise RuntimeError("keyed noise produced a negative normalized price")
    return observed


def fit_cell(
    truth_id: str,
    truth_index: int,
    truth_vector: np.ndarray,
    representation: str,
    noise_index: int,
    noise_level: float,
    profile: DateProfile,
    start_schedule: Sequence[tuple[str, np.ndarray]],
    *,
    spot: float = frozen.SYNTHETIC_SPOT,
) -> list[dict[str, Any]]:
    slots = representation_slots(profile, representation)
    geometry = build_geometry(slots, spot=spot)
    observed = observed_observables(
        truth_vector, truth_id, slots, noise_level, spot=spot
    )
    strikes = geometry["strikes"]
    maturities = geometry["maturities"]
    option_types = geometry["option_types"]
    rates = geometry["rates"]
    dividends = geometry["dividends"]
    truth_scaled = scaled_coordinates(truth_vector)

    def residuals(latent: np.ndarray) -> np.ndarray:
        candidate = unconstrained_to_parameters(latent, BOUNDS)
        predicted = normalized_observables(
            candidate,
            strikes,
            maturities,
            option_types,
            rates,
            dividends,
            spot=spot,
            node_count=frozen.NODE_COUNT,
        )
        return predicted - observed

    rows: list[dict[str, Any]] = []
    for start_index, (strategy, initial_latent) in enumerate(start_schedule):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "truth_id": truth_id,
            "truth_index": truth_index,
            "representation": representation,
            "noise_index": noise_index,
            "noise_level": noise_level,
            "profile_date": profile.date_id,
            "start_index": start_index,
            "start_strategy": strategy,
        }
        try:
            result = least_squares(
                residuals,
                initial_latent,
                method=frozen.OPTIMIZER_METHOD,
                max_nfev=frozen.OPTIMIZER_MAX_NFEV,
                ftol=frozen.OPTIMIZER_FTOL,
                xtol=frozen.OPTIMIZER_XTOL,
                gtol=frozen.OPTIMIZER_GTOL,
                diff_step=frozen.OPTIMIZER_DIFF_STEP,
            )
            recovered = unconstrained_to_parameters(result.x, BOUNDS)
            predicted = normalized_observables(
                recovered,
                strikes,
                maturities,
                option_types,
                rates,
                dividends,
                spot=spot,
                node_count=frozen.NODE_COUNT,
            )
            price_errors = predicted - observed
            parameter_errors = recovered - np.asarray(truth_vector, dtype=np.float64)
            scaled_displacement = scaled_coordinates(recovered) - truth_scaled
            scale = float(np.mean(np.abs(observed)))
            row.update(
                {
                    "success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(result.nfev),
                    "reached_cap": bool(result.nfev >= frozen.OPTIMIZER_MAX_NFEV),
                    "loss": float(np.mean(residuals(result.x) ** 2)),
                    "repricing_rmse": float(np.sqrt(np.mean(price_errors**2))),
                    "repricing_rmse_relative": float(
                        np.sqrt(np.mean((price_errors / np.maximum(np.abs(observed), 1e-12)) ** 2))
                    ),
                    "parameter_rmse_scaled": float(
                        np.sqrt(np.mean(scaled_displacement**2))
                    ),
                    "max_abs_price_error": float(np.max(np.abs(price_errors))),
                    "boundary_reasons": ";".join(
                        boundary_diagnostics(recovered, BOUNDS)
                    ),
                }
            )
            row.update(
                {f"recovered_{name}": float(value) for name, value in zip(PARAMETER_NAMES, recovered)}
            )
            row.update(
                {
                    f"scaled_error_{name}": float(value)
                    for name, value in zip(PARAMETER_NAMES, scaled_displacement)
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve every failed start
            row.update(
                {
                    "success": False,
                    "optimizer_status": -1,
                    "optimizer_message": f"{type(error).__name__}: {error}",
                    "nfev": 0,
                    "reached_cap": False,
                    "loss": float("nan"),
                    "repricing_rmse": float("nan"),
                    "repricing_rmse_relative": float("nan"),
                    "parameter_rmse_scaled": float("nan"),
                    "max_abs_price_error": float("nan"),
                    "boundary_reasons": "",
                }
            )
            row.update({f"recovered_{name}": float("nan") for name in PARAMETER_NAMES})
            row.update(
                {f"scaled_error_{name}": float("nan") for name in PARAMETER_NAMES}
            )
        row["runtime_seconds"] = float(time.perf_counter() - started)
        rows.append(row)
    return rows


class ResultLog:
    """Incremental JSONL persistence with resume support."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.completed: set[tuple[str, str, int, int]] = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") == "run":
                    self.completed.add(
                        (
                            record["truth_id"],
                            record["representation"],
                            record["noise_index"],
                            record["start_index"],
                        )
                    )

    def cell_complete(
        self, truth_id: str, representation: str, noise_index: int
    ) -> bool:
        expected = frozen.START_COUNT
        count = sum(
            1
            for key in self.completed
            if key[0] == truth_id
            and key[1] == representation
            and key[2] == noise_index
        )
        return count >= expected

    def append(self, rows: Sequence[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                payload = {"record_type": "run", **row}
                handle.write(json.dumps(payload, default=float) + "\n")
                self.completed.add(
                    (
                        row["truth_id"],
                        row["representation"],
                        row["noise_index"],
                        row["start_index"],
                    )
                )

    def append_event(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps({"record_type": "event", **event}, default=float) + "\n"
            )
