"""Run a synthetic-only PINN baseline with held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.pinn_model import PhysicsInformedInverseCalibrator

from .constants import DEFAULT_SEED, PARAMETER_NAMES
from .constraints import validate_parameters, vector_to_dictionary
from .dataset import SurfaceParameterDataset
from .evaluate_parameters import evaluate_parameter_recovery
from .evaluate_repricing import evaluate_repricing
from .pricing_interface import price_double_heston_surface
from .surface_grid import build_surface_grid, normalize_prices_by_spot
from .synthetic_dataset import assign_surface_splits, load_parameter_bounds
from .train_pinn import predict_parameters, train_pinn
from .utils import write_json


def run_synthetic_pinn_baseline(
    output_directory: str | Path,
    *,
    bounds_path: str | Path = "configs/parameter_bounds_PROVISIONAL.yaml",
    surface_count: int = 120,
    seed: int = DEFAULT_SEED,
    noise_level: float = 0.0,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    parameter_loss_weight: float = 1.0,
    physics_loss_weight: float = 1.0,
    pricing_node_count: int = 16,
) -> dict[str, object]:
    """Generate, train, and evaluate a synthetic PINN without split leakage."""
    output_path = Path(output_directory)
    dataset_directory = output_path / "dataset"
    training_directory = output_path / "training"
    evaluation_directory = output_path / "evaluation"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    training_directory.mkdir(parents=True, exist_ok=True)
    evaluation_directory.mkdir(parents=True, exist_ok=True)

    frame = _generate_resampled_research_dataset(
        dataset_directory,
        bounds_path=bounds_path,
        surface_count=surface_count,
        seed=seed,
        noise_level=noise_level,
    )
    dataset = SurfaceParameterDataset.from_surface_frame(frame)
    train_indices = dataset.indices_for_split("train")
    validation_indices = dataset.indices_for_split("validation")
    test_indices = dataset.indices_for_split("test")
    _assert_no_leakage(dataset, train_indices, validation_indices, test_indices)

    model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    training_summary = train_pinn(
        model,
        dataset,
        train_indices,
        validation_indices,
        training_directory,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        parameter_loss_weight=parameter_loss_weight,
        physics_loss_weight=physics_loss_weight,
        node_count=pricing_node_count,
    )

    predicted = predict_parameters(
        training_summary["model"],
        dataset,
        test_indices,
    ).numpy()
    truth = dataset.targets[test_indices].numpy()
    test_surface_ids = [dataset.surface_ids[index] for index in test_indices]
    parameter_summary = evaluate_parameter_recovery(
        truth,
        predicted,
        test_surface_ids,
        evaluation_directory / "parameter_recovery",
    )
    prediction_frame = pd.DataFrame(
        {
            "surface_id": test_surface_ids,
            **{
                f"predicted_{name}": predicted[:, index]
                for index, name in enumerate(PARAMETER_NAMES)
            },
        }
    )
    repricing_summary = evaluate_repricing(
        frame.loc[frame["surface_id"].isin(test_surface_ids)].copy(),
        {
            surface_id: predicted[index]
            for index, surface_id in enumerate(test_surface_ids)
        },
        evaluation_directory / "repricing",
    )
    prediction_frame.to_csv(evaluation_directory / "test_predictions.csv", index=False)

    split_counts = {
        "train_surfaces": len(train_indices),
        "validation_surfaces": len(validation_indices),
        "test_surfaces": len(test_indices),
    }
    no_cheat_summary = {
        "selection_uses_test_set": False,
        "selection_uses_validation_only": True,
        "whole_surface_split_integrity": True,
        "train_validation_test_surface_ids_disjoint": True,
        "target_parameters_known_before_pricing": True,
        "real_market_data_used": False,
        "normalization_fit_on_train_only": True,
    }
    summary = {
        "dataset_directory": str(dataset_directory),
        "training_directory": str(training_directory),
        "evaluation_directory": str(evaluation_directory),
        "surface_count": surface_count,
        "noise_level": noise_level,
        "split_counts": split_counts,
        "training": {
            "best_epoch": training_summary["best_epoch"],
            "best_validation_total_loss": training_summary["best_validation_total_loss"],
            "physics_loss_weight": physics_loss_weight,
            "parameter_loss_weight": parameter_loss_weight,
            "pricing_node_count": pricing_node_count,
        },
        "test_parameter_recovery": parameter_summary,
        "test_repricing": repricing_summary,
        "no_cheat_summary": no_cheat_summary,
    }
    write_json(output_path / "run_summary.json", summary)
    return summary


def _generate_resampled_research_dataset(
    output_directory: str | Path,
    *,
    bounds_path: str | Path,
    surface_count: int,
    seed: int,
    noise_level: float,
) -> pd.DataFrame:
    if surface_count < 3:
        raise ValueError("surface_count must be at least three")
    bounds, bounds_status = load_parameter_bounds(bounds_path, allow_provisional=True)
    rng = np.random.default_rng(seed)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    surface_ids = [f"surface_{index:06d}" for index in range(surface_count)]
    split_map = assign_surface_splits(surface_ids, seed=seed)
    rows: list[dict[str, object]] = []
    accepted = 0
    attempts = 0
    max_attempts = surface_count * 200
    while accepted < surface_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Unable to generate {surface_count} stable synthetic surfaces after {max_attempts} attempts"
            )
        vector = np.asarray(
            [rng.uniform(*bounds[name]) for name in PARAMETER_NAMES],
            dtype=np.float64,
        )
        if not validate_parameters(vector)["is_valid"]:
            continue
        spot = float(95.0 + 10.0 * rng.random())
        risk_free_rate = float(0.04 + 0.02 * rng.random())
        dividend_yield = float(0.005 + 0.015 * rng.random())
        grid = build_surface_grid(spot)
        try:
            clean_prices = price_double_heston_surface(
                spot,
                grid["strike"].to_numpy(dtype=np.float64),
                grid["maturity_years"].to_numpy(dtype=np.float64),
                risk_free_rate,
                dividend_yield,
                grid["option_type"].tolist(),
                vector,
            )
        except Exception:
            continue
        noisy_prices = clean_prices.copy()
        if noise_level > 0.0:
            noisy_prices *= 1.0 + rng.normal(0.0, noise_level, size=noisy_prices.shape)
            if np.any(noisy_prices < 0.0) or not np.isfinite(noisy_prices).all():
                continue
        normalized = normalize_prices_by_spot(noisy_prices, spot)
        parameter_values = vector_to_dictionary(vector)
        surface_id = surface_ids[accepted]
        for row_index, grid_row in grid.iterrows():
            row = {
                "surface_id": surface_id,
                "spot": spot,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "generation_seed": seed + accepted,
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
                "data_status": "GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA",
            }
            row.update(parameter_values)
            rows.append(row)
        accepted += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "surfaces.csv", index=False)
    write_json(
        output_path / "dataset_metadata.json",
        {
            "data_status": "GENUINE_CANONICAL_DOUBLE_HESTON_SYNTHETIC_DATA",
            "generation_seed": seed,
            "noise_level": noise_level,
            "surface_count": surface_count,
            "row_count": len(frame),
            "parameter_bounds_status": bounds_status,
            "resampled_until_priceable": True,
            "generation_attempts": attempts,
            "warning": (
                "Genuine synthetic surfaces from the independent canonical Double Heston engine. "
                "Only parameter vectors that passed strict validity and priceability checks were retained."
            ),
        },
    )
    return frame


def _assert_no_leakage(
    dataset: SurfaceParameterDataset,
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
) -> None:
    train_ids = {dataset.surface_ids[index] for index in train_indices}
    validation_ids = {dataset.surface_ids[index] for index in validation_indices}
    test_ids = {dataset.surface_ids[index] for index in test_indices}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("surface leakage detected across train/validation/test splits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/pinn_synthetic_baseline"),
    )
    parser.add_argument("--bounds", type=Path, default=Path("configs/parameter_bounds_PROVISIONAL.yaml"))
    parser.add_argument("--surface-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--parameter-loss-weight", type=float, default=1.0)
    parser.add_argument("--physics-loss-weight", type=float, default=1.0)
    parser.add_argument("--pricing-node-count", type=int, default=16)
    args = parser.parse_args()
    summary = run_synthetic_pinn_baseline(
        args.output,
        bounds_path=args.bounds,
        surface_count=args.surface_count,
        seed=args.seed,
        noise_level=args.noise_level,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        parameter_loss_weight=args.parameter_loss_weight,
        physics_loss_weight=args.physics_loss_weight,
        pricing_node_count=args.pricing_node_count,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
