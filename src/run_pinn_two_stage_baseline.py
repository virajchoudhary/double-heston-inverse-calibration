"""Run a two-stage synthetic PINN experiment: supervised warm start then PINN fine-tune."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from models.pinn_model import PhysicsInformedInverseCalibrator

from .constants import DEFAULT_SEED, PARAMETER_NAMES
from .dataset import SurfaceParameterDataset
from .evaluate_parameters import evaluate_parameter_recovery
from .evaluate_repricing import evaluate_repricing
from .run_pinn_synthetic_baseline import (
    _assert_no_leakage,
    _generate_resampled_research_dataset,
)
from .train_pinn import predict_parameters, train_pinn
from .utils import write_json


def run_two_stage_pinn_experiment(
    output_directory: str | Path,
    *,
    bounds_path: str | Path = "configs/parameter_bounds_PROVISIONAL.yaml",
    surface_count: int = 120,
    seed: int = DEFAULT_SEED,
    noise_level: float = 0.0,
    stage1_epochs: int = 25,
    stage1_learning_rate: float = 5e-4,
    stage2_epochs: int = 12,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run an honest two-stage synthetic experiment with held-out test evaluation."""
    output_path = Path(output_directory)
    dataset_directory = output_path / "dataset"
    stage1_directory = output_path / "stage1_supervised"
    stage2_root = output_path / "stage2_candidates"
    final_eval_directory = output_path / "final_evaluation"
    for directory in (
        dataset_directory,
        stage1_directory,
        stage2_root,
        final_eval_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

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

    stage1_model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    stage1_summary = train_pinn(
        stage1_model,
        dataset,
        train_indices,
        validation_indices,
        stage1_directory,
        seed=seed,
        batch_size=batch_size,
        epochs=stage1_epochs,
        learning_rate=stage1_learning_rate,
        parameter_loss_weight=1.0,
        physics_loss_weight=0.0,
        node_count=8,
        patience=max(5, min(stage1_epochs, 10)),
    )
    warm_start_state = copy.deepcopy(stage1_summary["model"].state_dict())

    stage2_candidates = [
        {
            "name": "ft_phys_0p10_param_1_lr_2e4",
            "parameter_loss_weight": 1.0,
            "physics_loss_weight": 0.10,
            "learning_rate": 2e-4,
            "node_count": 8,
        },
        {
            "name": "ft_phys_0p25_param_1_lr_2e4",
            "parameter_loss_weight": 1.0,
            "physics_loss_weight": 0.25,
            "learning_rate": 2e-4,
            "node_count": 8,
        },
        {
            "name": "ft_phys_0p10_param_2_lr_1e4",
            "parameter_loss_weight": 2.0,
            "physics_loss_weight": 0.10,
            "learning_rate": 1e-4,
            "node_count": 8,
        },
    ]
    candidate_results: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    for candidate in stage2_candidates:
        model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
        model.load_state_dict(warm_start_state)
        summary = train_pinn(
            model,
            dataset,
            train_indices,
            validation_indices,
            stage2_root / candidate["name"],
            seed=seed,
            batch_size=batch_size,
            epochs=stage2_epochs,
            learning_rate=float(candidate["learning_rate"]),
            parameter_loss_weight=float(candidate["parameter_loss_weight"]),
            physics_loss_weight=float(candidate["physics_loss_weight"]),
            node_count=int(candidate["node_count"]),
            patience=max(4, min(stage2_epochs, 8)),
        )
        result = {
            **candidate,
            "best_epoch": int(summary["best_epoch"]),
            "best_validation_total_loss": float(summary["best_validation_total_loss"]),
        }
        candidate_results.append(result)
        if (
            best_candidate is None
            or result["best_validation_total_loss"]
            < best_candidate["best_validation_total_loss"]
        ):
            best_candidate = result
            best_model_state = copy.deepcopy(summary["model"].state_dict())

    if best_candidate is None or best_model_state is None:
        raise RuntimeError("No fine-tuning candidate completed successfully")

    stage1_eval = _evaluate_model(
        stage1_summary["model"],
        frame,
        dataset,
        test_indices,
        final_eval_directory / "stage1_supervised_test",
    )
    best_model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    best_model.load_state_dict(best_model_state)
    stage2_eval = _evaluate_model(
        best_model,
        frame,
        dataset,
        test_indices,
        final_eval_directory / "stage2_finetuned_test",
    )

    summary = {
        "surface_count": surface_count,
        "noise_level": noise_level,
        "split_counts": {
            "train_surfaces": len(train_indices),
            "validation_surfaces": len(validation_indices),
            "test_surfaces": len(test_indices),
        },
        "selection_policy": {
            "selected_by_validation_only": True,
            "test_set_used_for_model_selection": False,
            "whole_surface_split_integrity": True,
            "real_market_data_used": False,
        },
        "stage1_supervised": {
            "best_epoch": int(stage1_summary["best_epoch"]),
            "best_validation_total_loss": float(stage1_summary["best_validation_total_loss"]),
            "test": stage1_eval,
        },
        "stage2_candidates": sorted(
            candidate_results,
            key=lambda item: item["best_validation_total_loss"],
        ),
        "stage2_selected": {
            **best_candidate,
            "test": stage2_eval,
        },
        "improvement_over_stage1": {
            "repricing_rmse_delta": (
                stage2_eval["repricing"]["rmse"] - stage1_eval["repricing"]["rmse"]
            ),
            "repricing_mae_delta": (
                stage2_eval["repricing"]["mae"] - stage1_eval["repricing"]["mae"]
            ),
            "validity_rate_delta": (
                stage2_eval["parameter_recovery"]["full_vector_validity_rate"]
                - stage1_eval["parameter_recovery"]["full_vector_validity_rate"]
            ),
        },
    }
    write_json(output_path / "summary.json", summary)
    return summary


def _evaluate_model(
    model: PhysicsInformedInverseCalibrator,
    frame: pd.DataFrame,
    dataset: SurfaceParameterDataset,
    indices: list[int],
    output_directory: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    predicted = predict_parameters(model, dataset, indices).numpy()
    truth = dataset.targets[indices].numpy()
    surface_ids = [dataset.surface_ids[index] for index in indices]
    parameter_recovery = evaluate_parameter_recovery(
        truth,
        predicted,
        surface_ids,
        output_path / "parameter_recovery",
    )
    repricing = evaluate_repricing(
        frame.loc[frame["surface_id"].isin(surface_ids)].copy(),
        {
            surface_id: predicted[index]
            for index, surface_id in enumerate(surface_ids)
        },
        output_path / "repricing",
    )
    prediction_frame = pd.DataFrame(
        {
            "surface_id": surface_ids,
            **{
                f"predicted_{name}": predicted[:, column]
                for column, name in enumerate(PARAMETER_NAMES)
            },
        }
    )
    prediction_frame.to_csv(output_path / "test_predictions.csv", index=False)
    return {
        "parameter_recovery": parameter_recovery,
        "repricing": repricing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/pinn_two_stage_baseline"),
    )
    parser.add_argument("--bounds", type=Path, default=Path("configs/parameter_bounds_PROVISIONAL.yaml"))
    parser.add_argument("--surface-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--stage1-epochs", type=int, default=25)
    parser.add_argument("--stage1-learning-rate", type=float, default=5e-4)
    parser.add_argument("--stage2-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    summary = run_two_stage_pinn_experiment(
        args.output,
        bounds_path=args.bounds,
        surface_count=args.surface_count,
        seed=args.seed,
        noise_level=args.noise_level,
        stage1_epochs=args.stage1_epochs,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
