"""Run the controlled canonical Double Heston validation milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate_double_heston import calibrate_double_heston
from .constants import PARAMETER_NAMES
from .double_heston import (
    price_double_heston_call,
    price_double_heston_put,
    price_double_heston_surface,
)
from .synthetic_dataset import generate_canonical_pilot_dataset
from .utils import write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/double_heston_clean_fixture.json"
BOUNDS_PATH = REPOSITORY_ROOT / "configs/parameter_bounds_PROVISIONAL.yaml"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs/double_heston_validation"


def _load_fixture() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture.get("fixture_type") != "CANONICAL_REIMPLEMENTATION_FIXTURE":
        raise ValueError("Unexpected validation fixture provenance label")
    return fixture


def _best_row(frame: pd.DataFrame) -> pd.Series:
    finite = frame[np.isfinite(frame["loss"].to_numpy(dtype=np.float64))]
    if finite.empty:
        raise RuntimeError("Every calibration start failed")
    return finite.loc[finite["loss"].idxmin()]


def _parameter_vector(row: pd.Series) -> np.ndarray:
    return np.asarray([row[f"predicted_{name}"] for name in PARAMETER_NAMES])


def _surface_frame(fixture: dict[str, Any], observed: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "surface_id": "canonical_controlled_surface",
            "quote_index": np.arange(len(observed)),
            "spot": fixture["spot"],
            "risk_free_rate": fixture["rate"],
            "dividend_yield": fixture["dividend_yield"],
            "strike": fixture["strikes"],
            "maturity_years": fixture["maturities"],
            "option_type": fixture["option_types"],
            "observed_price": observed,
            "data_status": "CANONICAL_REIMPLEMENTATION_FIXTURE",
        }
    )


def main() -> None:
    """Execute pricing, convergence, recovery, noise, and pilot checks."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fixture = _load_fixture()
    parameters = np.asarray(fixture["parameters"], dtype=np.float64)
    strikes = np.asarray(fixture["strikes"], dtype=np.float64)
    maturities = np.asarray(fixture["maturities"], dtype=np.float64)
    option_types = np.asarray(fixture["option_types"], dtype=str)
    clean_prices = price_double_heston_surface(
        fixture["spot"],
        strikes,
        maturities,
        fixture["rate"],
        fixture["dividend_yield"],
        option_types,
        parameters,
        node_count=fixture["pricing_node_count"],
    )
    fixture_error = clean_prices - np.asarray(fixture["expected_prices"])
    _surface_frame(fixture, clean_prices).to_csv(
        OUTPUT_DIRECTORY / "clean_surface.csv", index=False
    )

    rng = np.random.default_rng(20260806)
    noise_draws = rng.normal(0.0, 0.01, size=clean_prices.shape)
    noisy_prices = clean_prices * (1.0 + noise_draws)
    if np.any(noisy_prices < 0.0):
        raise RuntimeError("the controlled 1% noise realization produced a negative price")
    noisy_frame = _surface_frame(fixture, noisy_prices)
    noisy_frame["clean_price"] = clean_prices
    noisy_frame["relative_noise"] = noise_draws
    noisy_frame.to_csv(OUTPUT_DIRECTORY / "noisy_surface.csv", index=False)

    calibration_arguments = (
        fixture["spot"],
        strikes,
        maturities,
        fixture["rate"],
        fixture["dividend_yield"],
        option_types,
    )
    clean_starts = calibrate_double_heston(
        *calibration_arguments,
        clean_prices,
        parameters,
        BOUNDS_PATH,
        output_csv=OUTPUT_DIRECTORY / "clean_recovery_starts.csv",
        node_count=fixture["pricing_node_count"],
        max_nfev=80,
        seed=42,
    )
    noise_starts = calibrate_double_heston(
        *calibration_arguments,
        noisy_prices,
        parameters,
        BOUNDS_PATH,
        output_csv=OUTPUT_DIRECTORY / "noise_1pct_recovery_starts.csv",
        node_count=fixture["pricing_node_count"],
        max_nfev=80,
        seed=42,
    )
    clean_best = _best_row(clean_starts)
    noise_best = _best_row(noise_starts)

    comparison_rows: list[dict[str, Any]] = []
    for index, name in enumerate(PARAMETER_NAMES):
        true_value = parameters[index]
        clean_value = float(clean_best[f"predicted_{name}"])
        noise_value = float(noise_best[f"predicted_{name}"])
        comparison_rows.append(
            {
                "parameter": name,
                "true_value": true_value,
                "best_clean_value": clean_value,
                "best_clean_absolute_error": abs(clean_value - true_value),
                "best_clean_relative_error": abs(clean_value - true_value)
                / max(abs(true_value), 1e-4),
                "best_noise_1pct_value": noise_value,
                "best_noise_1pct_absolute_error": abs(noise_value - true_value),
                "best_noise_1pct_relative_error": abs(noise_value - true_value)
                / max(abs(true_value), 1e-4),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIRECTORY / "parameter_comparison.csv", index=False
    )

    convergence_rows: list[dict[str, Any]] = []
    node_prices: dict[int, np.ndarray] = {}
    for node_count in (32, 48, 64, 96):
        node_prices[node_count] = price_double_heston_surface(
            fixture["spot"],
            strikes,
            maturities,
            fixture["rate"],
            fixture["dividend_yield"],
            option_types,
            parameters,
            node_count=node_count,
        )
    reference = node_prices[96]
    for node_count, prices in node_prices.items():
        differences = prices - reference
        convergence_rows.append(
            {
                "node_count": node_count,
                "reference_node_count": 96,
                "price_rmse_vs_reference": float(np.sqrt(np.mean(differences**2))),
                "max_abs_price_error_vs_reference": float(
                    np.max(np.abs(differences))
                ),
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        OUTPUT_DIRECTORY / "pricing_convergence.csv", index=False
    )

    discounted_spot = fixture["spot"] * np.exp(
        -fixture["dividend_yield"] * maturities
    )
    discounted_strike = strikes * np.exp(-fixture["rate"] * maturities)
    lower = np.where(
        option_types == "call",
        np.maximum(discounted_spot - discounted_strike, 0.0),
        np.maximum(discounted_strike - discounted_spot, 0.0),
    )
    upper = np.where(option_types == "call", discounted_spot, discounted_strike)
    parity_errors = []
    for strike, maturity in zip(strikes, maturities, strict=True):
        call = price_double_heston_call(
            fixture["spot"],
            float(strike),
            float(maturity),
            fixture["rate"],
            fixture["dividend_yield"],
            parameters,
        )
        put = price_double_heston_put(
            fixture["spot"],
            float(strike),
            float(maturity),
            fixture["rate"],
            fixture["dividend_yield"],
            parameters,
        )
        parity_errors.append(
            call
            - put
            - fixture["spot"]
            * np.exp(-fixture["dividend_yield"] * maturity)
            + strike * np.exp(-fixture["rate"] * maturity)
        )

    failures = pd.concat(
        [
            clean_starts.assign(experiment="clean").loc[lambda value: ~value["success"]],
            noise_starts.assign(experiment="noise_1pct").loc[
                lambda value: ~value["success"]
            ],
        ],
        ignore_index=True,
    )
    failure_columns = [
        "experiment",
        "start_index",
        "start_strategy",
        "success",
        "optimizer_status",
        "optimizer_message",
        "nfev",
        "loss",
        "runtime_seconds",
        "boundary_near",
        "boundary_reasons",
    ]
    failures.reindex(columns=failure_columns).to_csv(
        OUTPUT_DIRECTORY / "failures.csv", index=False
    )

    pilot = generate_canonical_pilot_dataset(
        OUTPUT_DIRECTORY / "pilot_surfaces",
        BOUNDS_PATH,
        n_surfaces=12,
        seed=42,
    )
    all_starts = pd.concat(
        [
            clean_starts.assign(experiment="clean"),
            noise_starts.assign(experiment="noise_1pct"),
        ],
        ignore_index=True,
    )
    summary = {
        "validation_status": "CANONICAL_REIMPLEMENTATION_VALIDATION",
        "equivalent_to_unavailable_teammate_source": False,
        "proves_real_nifty_performance": False,
        "fixture_max_abs_regression_error": float(np.max(np.abs(fixture_error))),
        "no_arbitrage": {
            "all_bounds_pass": bool(
                np.all(clean_prices >= lower - 1e-9)
                and np.all(clean_prices <= upper + 1e-9)
            ),
            "minimum_lower_bound_margin": float(np.min(clean_prices - lower)),
            "minimum_upper_bound_margin": float(np.min(upper - clean_prices)),
        },
        "put_call_parity_max_abs_error": float(np.max(np.abs(parity_errors))),
        "quadrature_64_vs_96": {
            "rmse": float(
                np.sqrt(np.mean((node_prices[64] - node_prices[96]) ** 2))
            ),
            "max_abs_error": float(np.max(np.abs(node_prices[64] - node_prices[96]))),
        },
        "clean_recovery": {
            "best_start_strategy": str(clean_best["start_strategy"]),
            "optimizer_success": bool(clean_best["success"]),
            "loss": float(clean_best["loss"]),
            "price_rmse": float(clean_best["price_rmse"]),
            "parameter_rmse": float(clean_best["parameter_rmse"]),
            "max_relative_parameter_error": float(
                clean_best["max_relative_parameter_error"]
            ),
        },
        "noise_1pct_recovery": {
            "realized_noise_rms_fraction": float(np.sqrt(np.mean(noise_draws**2))),
            "best_start_strategy": str(noise_best["start_strategy"]),
            "optimizer_success": bool(noise_best["success"]),
            "loss": float(noise_best["loss"]),
            "price_rmse_vs_noisy_observations": float(noise_best["price_rmse"]),
            "parameter_rmse_vs_known_parameters": float(noise_best["parameter_rmse"]),
            "max_relative_parameter_error": float(
                noise_best["max_relative_parameter_error"]
            ),
        },
        "repeated_starts": {
            "starts_per_experiment": 3,
            "total_optimizer_success_count": int(all_starts["success"].sum()),
            "failed_or_stopped_count": int((~all_starts["success"]).sum()),
            "boundary_near_count": int(all_starts["boundary_near"].sum()),
            "clean_price_rmse_range": [
                float(clean_starts["price_rmse"].min()),
                float(clean_starts["price_rmse"].max()),
            ],
            "noise_price_rmse_range": [
                float(noise_starts["price_rmse"].min()),
                float(noise_starts["price_rmse"].max()),
            ],
            "interpretation": (
                "Different locally adequate parameter vectors are evidence of "
                "practical non-identifiability; optimizer success is not proof "
                "of unique recovery."
            ),
        },
        "pilot_surfaces": {
            "data_status": "GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA",
            "surface_count": int(pilot["surface_id"].nunique()),
            "row_count": int(len(pilot)),
            "full_ann_training_started": False,
        },
    }
    write_json(OUTPUT_DIRECTORY / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
