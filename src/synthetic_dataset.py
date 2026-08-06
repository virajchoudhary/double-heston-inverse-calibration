"""Synthetic surface generation with strict research/smoke separation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .constants import (
    DEFAULT_SEED,
    GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA,
    NOT_RESEARCH_DATA,
    PARAMETER_NAMES,
)
from .constraints import validate_parameters, vector_to_dictionary
from .pricing_interface import (
    dummy_surface_generator_for_smoke_test,
    price_double_heston_surface,
)
from .surface_grid import build_surface_grid, normalize_prices_by_spot
from .utils import write_json

PricingFunction = Callable[
    [float, Sequence[float], Sequence[float], float, float, Sequence[str], Sequence[float]],
    np.ndarray,
]


def assign_surface_splits(
    surface_ids: Sequence[str],
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, str]:
    """Assign complete surfaces to deterministic train/validation/test splits."""
    ids = list(surface_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("surface_ids must be unique")
    if len(ids) < 3:
        raise ValueError("At least three surfaces are required for three splits")
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions <= 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to one")
    shuffled = np.asarray(ids, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_count = max(1, int(np.floor(len(ids) * train_fraction)))
    validation_count = max(1, int(np.floor(len(ids) * validation_fraction)))
    if train_count + validation_count >= len(ids):
        validation_count = 1
        train_count = len(ids) - 2
    split_map: dict[str, str] = {}
    for surface_id in shuffled[:train_count]:
        split_map[str(surface_id)] = "train"
    for surface_id in shuffled[train_count : train_count + validation_count]:
        split_map[str(surface_id)] = "validation"
    for surface_id in shuffled[train_count + validation_count :]:
        split_map[str(surface_id)] = "test"
    return split_map


def generate_research_dataset(
    output_directory: str | Path,
    bounds_path: str | Path,
    n_surfaces: int,
    seed: int = DEFAULT_SEED,
    noise_level: float = 0.0,
    allow_provisional_bounds: bool = False,
) -> pd.DataFrame:
    """Generate canonical engine surfaces; never fall back to a dummy mapping.

    Provisional ranges require an explicit opt-in because they are engineering
    selections rather than externally confirmed historical bounds.
    """
    bounds, bounds_status = load_parameter_bounds(
        bounds_path, allow_provisional=allow_provisional_bounds
    )
    if n_surfaces < 3:
        raise ValueError("n_surfaces must be at least three")
    price_double_heston_surface(
        100.0,
        [100.0],
        [30.0 / 365.0],
        0.05,
        0.0,
        ["call"],
        _sample_valid_parameter_vector(np.random.default_rng(seed), bounds),
    )
    rng = np.random.default_rng(seed)
    parameters = [
        _sample_valid_parameter_vector(rng, bounds) for _ in range(n_surfaces)
    ]
    return _generate_and_save(
        output_directory=output_directory,
        parameters=parameters,
        pricer=price_double_heston_surface,
        seed=seed,
        noise_level=noise_level,
        data_status=GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA,
        allow_dummy=False,
        bounds_status=bounds_status,
    )


def generate_canonical_pilot_dataset(
    output_directory: str | Path,
    bounds_path: str | Path,
    n_surfaces: int = 24,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate at most 100 genuine canonical-engine surfaces for validation."""
    if not 3 <= n_surfaces <= 100:
        raise ValueError("pilot n_surfaces must be between 3 and 100")
    output_path = Path(output_directory)
    if "pilot_surfaces" not in {part.lower() for part in output_path.parts}:
        raise ValueError("pilot data must be saved beneath a pilot_surfaces path")
    return generate_research_dataset(
        output_path,
        bounds_path,
        n_surfaces,
        seed=seed,
        noise_level=0.0,
        allow_provisional_bounds=True,
    )


def generate_smoke_test_dataset(
    output_directory: str | Path,
    n_surfaces: int = 48,
    seed: int = DEFAULT_SEED,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Generate development-only data under a smoke-test-labelled directory."""
    if n_surfaces < 3:
        raise ValueError("n_surfaces must be at least three")
    output_path = Path(output_directory)
    if "smoke_test" not in {part.lower() for part in output_path.parts}:
        raise ValueError("Smoke data must be saved beneath a smoke_test-labelled path")
    rng = np.random.default_rng(seed)
    parameters = [_sample_smoke_parameter_vector(rng) for _ in range(n_surfaces)]
    return _generate_and_save(
        output_directory=output_path,
        parameters=parameters,
        pricer=dummy_surface_generator_for_smoke_test,
        seed=seed,
        noise_level=noise_level,
        data_status=NOT_RESEARCH_DATA,
        allow_dummy=True,
        bounds_status="SMOKE_TEST_INTERNAL_RANGES",
    )


def load_confirmed_parameter_bounds(
    bounds_path: str | Path,
) -> dict[str, tuple[float, float]]:
    """Load only fully specified teammate-confirmed bounds in canonical order."""
    bounds, _ = load_parameter_bounds(bounds_path, allow_provisional=False)
    return bounds


def load_parameter_bounds(
    bounds_path: str | Path,
    *,
    allow_provisional: bool,
) -> tuple[dict[str, tuple[float, float]], str]:
    """Load confirmed bounds or explicitly opted-in provisional pilot ranges."""
    path = Path(bounds_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    if status == "TEAMMATE_CONFIRMED":
        raw_bounds = payload.get("parameter_bounds", {})
    elif status == "PROVISIONAL_CANONICAL_REIMPLEMENTATION" and allow_provisional:
        raw_bounds = payload.get("empirical_sampling_ranges", {}).get(
            "parameter_bounds", {}
        )
    else:
        raise ValueError(
            "Parameter bounds must be TEAMMATE_CONFIRMED, or provisional use must "
            "be explicitly enabled for controlled pilot generation"
        )
    confirmed: dict[str, tuple[float, float]] = {}
    for name in PARAMETER_NAMES:
        entry = raw_bounds.get(name, {})
        lower = entry.get("lower")
        upper = entry.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"Missing lower/upper bound for {name}")
        lower_value = float(lower)
        upper_value = float(upper)
        if not np.isfinite([lower_value, upper_value]).all() or lower_value >= upper_value:
            raise ValueError(f"Invalid bounds for {name}")
        confirmed[name] = (lower_value, upper_value)
    return confirmed, status


def _generate_and_save(
    output_directory: str | Path,
    parameters: Sequence[np.ndarray],
    pricer: PricingFunction,
    seed: int,
    noise_level: float,
    data_status: str,
    allow_dummy: bool,
    bounds_status: str,
) -> pd.DataFrame:
    if noise_level < 0.0 or not np.isfinite(noise_level):
        raise ValueError("noise_level must be finite and non-negative")
    if data_status == NOT_RESEARCH_DATA and not allow_dummy:
        raise ValueError("Dummy data requires an explicit smoke-test code path")
    surface_ids = [f"surface_{index:06d}" for index in range(len(parameters))]
    split_map = assign_surface_splits(surface_ids, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    for surface_index, (surface_id, vector) in enumerate(
        zip(surface_ids, parameters, strict=True)
    ):
        diagnostics = validate_parameters(vector)
        if not diagnostics["is_valid"]:
            raise ValueError(f"Sampled invalid parameters: {diagnostics['violations']}")
        spot = float(95.0 + 10.0 * rng.random())
        risk_free_rate = float(0.04 + 0.02 * rng.random())
        dividend_yield = float(0.005 + 0.015 * rng.random())
        grid = build_surface_grid(spot)
        clean_prices = pricer(
            spot,
            grid["strike"].to_numpy(dtype=np.float64),
            grid["maturity_years"].to_numpy(dtype=np.float64),
            risk_free_rate,
            dividend_yield,
            grid["option_type"].tolist(),
            vector,
        )
        if clean_prices.shape != (len(grid),) or not np.isfinite(clean_prices).all():
            raise ValueError(f"Pricer failed shape/finite checks for {surface_id}")
        noisy_prices = clean_prices.copy()
        if noise_level > 0.0:
            noisy_prices *= 1.0 + rng.normal(0.0, noise_level, size=noisy_prices.shape)
            if np.any(noisy_prices < 0.0):
                raise ValueError(
                    f"Noise realization produced negative prices for {surface_id}; "
                    "prices are not clipped or silently replaced"
                )
        normalized = normalize_prices_by_spot(noisy_prices, spot)
        parameter_values = vector_to_dictionary(vector)
        for row_index, grid_row in grid.iterrows():
            row = {
                "surface_id": surface_id,
                "spot": spot,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "generation_seed": seed + surface_index,
                "noise_level": noise_level,
                "split": split_map[surface_id],
                "log_moneyness": float(grid_row["log_moneyness"]),
                "strike": float(grid_row["strike"]),
                "maturity_days": int(grid_row["maturity_days"]),
                "maturity_years": float(grid_row["maturity_years"]),
                "option_type": str(grid_row["option_type"]),
                "generated_price": float(noisy_prices[row_index]),
                "normalized_price": float(normalized[row_index]),
                "mask": True,
                "data_status": data_status,
            }
            row.update(parameter_values)
            rows.append(row)
    frame = pd.DataFrame(rows)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path / "surfaces.csv", index=False)
    write_json(
        output_path / "dataset_metadata.json",
        {
            "data_status": data_status,
            "generation_seed": seed,
            "noise_level": noise_level,
            "surface_count": len(parameters),
            "row_count": len(frame),
            "dummy_mapping_used": allow_dummy,
            "parameter_bounds_status": bounds_status,
            "equivalent_to_unavailable_teammate_source": False,
            "warning": (
                "Development-only tensor-flow data; not Double Heston prices and "
                "not research evidence."
                if allow_dummy
                else (
                    "Genuine prices from the independent canonical Double Heston "
                    "engine; not evidence of equivalence to unavailable source or "
                    "of performance on real NIFTY data."
                )
            ),
        },
    )
    return frame


def _sample_valid_parameter_vector(
    rng: np.random.Generator,
    bounds: Mapping[str, tuple[float, float]],
    maximum_attempts: int = 100_000,
) -> np.ndarray:
    for _ in range(maximum_attempts):
        vector = np.asarray(
            [rng.uniform(*bounds[name]) for name in PARAMETER_NAMES], dtype=np.float64
        )
        if validate_parameters(vector)["is_valid"]:
            return vector
    raise RuntimeError(
        "Unable to sample a valid vector; the selected ranges may be incompatible "
        "with the strict constraints"
    )


def _sample_smoke_parameter_vector(rng: np.random.Generator) -> np.ndarray:
    kappa_slow = rng.uniform(0.6, 1.4)
    theta_slow = rng.uniform(0.04, 0.12)
    kappa_fast = rng.uniform(2.0, 4.0)
    theta_fast = rng.uniform(0.03, 0.10)
    sigma_slow = rng.uniform(0.25, 0.65) * np.sqrt(2.0 * kappa_slow * theta_slow)
    sigma_fast = rng.uniform(0.25, 0.65) * np.sqrt(2.0 * kappa_fast * theta_fast)
    angle = rng.uniform(-np.pi, np.pi)
    radius = rng.uniform(0.05, 0.75)
    rho_slow = radius * np.cos(angle)
    rho_fast = radius * np.sin(angle)
    return np.asarray(
        [
            kappa_slow,
            theta_slow,
            sigma_slow,
            rho_slow,
            rng.uniform(0.03, 0.12),
            kappa_fast,
            theta_fast,
            sigma_fast,
            rho_fast,
            rng.uniform(0.03, 0.12),
        ],
        dtype=np.float64,
    )


def main() -> None:
    """Run the deliberately small canonical pilot generation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="generate at most 100 real-engine surfaces")
    pilot.add_argument("--count", type=int, default=24)
    pilot.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pilot.add_argument(
        "--bounds",
        type=Path,
        default=Path("configs/parameter_bounds_PROVISIONAL.yaml"),
    )
    pilot.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/double_heston_validation/pilot_surfaces"),
    )
    arguments = parser.parse_args()
    if arguments.command == "pilot":
        frame = generate_canonical_pilot_dataset(
            arguments.output,
            arguments.bounds,
            n_surfaces=arguments.count,
            seed=arguments.seed,
        )
        print(
            f"Generated {frame['surface_id'].nunique()} genuine canonical Double "
            f"Heston pilot surfaces ({len(frame)} quote rows)."
        )


if __name__ == "__main__":
    main()
