"""Run a larger synthetic benchmark for the improved two-stage PINN."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
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


def run_improved_pinn_benchmark(
    output_directory: str | Path,
    *,
    bounds_path: str | Path = "configs/parameter_bounds_PROVISIONAL.yaml",
    surface_count: int = 90,
    dataset_seed: int = DEFAULT_SEED,
    train_seeds: tuple[int, ...] = (11, 22),
    noise_level: float = 0.0,
    baseline_epochs: int = 8,
    curriculum_easy_epochs: int = 6,
    curriculum_full_epochs: int = 6,
    finetune_epochs: int = 4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Benchmark supervised-only against a curriculum two-stage PINN."""
    output_path = Path(output_directory)
    dataset_directory = output_path / "dataset"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    frame = _generate_resampled_research_dataset(
        dataset_directory,
        bounds_path=bounds_path,
        surface_count=surface_count,
        seed=dataset_seed,
        noise_level=noise_level,
    )
    dataset = SurfaceParameterDataset.from_surface_frame(frame)
    train_indices = dataset.indices_for_split("train")
    validation_indices = dataset.indices_for_split("validation")
    test_indices = dataset.indices_for_split("test")
    _assert_no_leakage(dataset, train_indices, validation_indices, test_indices)
    easy_indices = _easy_curriculum_indices(dataset, train_indices, keep_fraction=0.55)

    run_summaries: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        seed_directory = output_path / f"seed_{train_seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        baseline = _run_supervised_baseline(
            seed_directory / "baseline_supervised",
            dataset,
            frame,
            train_indices,
            validation_indices,
            test_indices,
            train_seed=train_seed,
            epochs=baseline_epochs,
            batch_size=batch_size,
        )
        improved = _run_curriculum_two_stage(
            seed_directory / "improved_two_stage",
            dataset,
            frame,
            easy_indices,
            train_indices,
            validation_indices,
            test_indices,
            train_seed=train_seed,
            curriculum_easy_epochs=curriculum_easy_epochs,
            curriculum_full_epochs=curriculum_full_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
        )
        run_summaries.append(
            {
                "train_seed": train_seed,
                "baseline": baseline,
                "improved": improved,
                "delta_improved_minus_baseline": _delta_summary(improved, baseline),
            }
        )

    aggregate = _aggregate_run_summaries(run_summaries)
    summary = {
        "date": "Friday, August 14, 2026",
        "surface_count": surface_count,
        "noise_level": noise_level,
        "dataset_seed": dataset_seed,
        "train_seeds": list(train_seeds),
        "split_counts": {
            "train_surfaces": len(train_indices),
            "validation_surfaces": len(validation_indices),
            "test_surfaces": len(test_indices),
            "easy_curriculum_train_surfaces": len(easy_indices),
        },
        "selection_policy": {
            "selected_by_validation_only": True,
            "test_set_used_for_model_selection": False,
            "whole_surface_split_integrity": True,
            "real_market_data_used": False,
            "curriculum_uses_train_truth_only": True,
        },
        "per_seed_results": run_summaries,
        "aggregate": aggregate,
        "focus_last_two_parameters_assumption": [
            "rho_fast",
            "v0_fast",
        ],
    }
    write_json(output_path / "summary.json", summary)
    return summary


def _run_supervised_baseline(
    output_directory: Path,
    dataset: SurfaceParameterDataset,
    frame: pd.DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    *,
    train_seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    training = train_pinn(
        model,
        dataset,
        train_indices,
        validation_indices,
        output_directory / "training",
        seed=train_seed,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=5e-4,
        parameter_loss_weight=1.0,
        physics_loss_weight=0.0,
        node_count=8,
        patience=max(4, epochs),
    )
    evaluation = _evaluate_model(
        training["model"],
        frame,
        dataset,
        test_indices,
        output_directory / "evaluation",
    )
    return {
        "best_epoch": int(training["best_epoch"]),
        "best_validation_total_loss": float(training["best_validation_total_loss"]),
        "test": evaluation,
    }


def _run_curriculum_two_stage(
    output_directory: Path,
    dataset: SurfaceParameterDataset,
    frame: pd.DataFrame,
    easy_indices: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    *,
    train_seed: int,
    curriculum_easy_epochs: int,
    curriculum_full_epochs: int,
    finetune_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    stage1_easy_model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    stage1_easy = train_pinn(
        stage1_easy_model,
        dataset,
        easy_indices,
        validation_indices,
        output_directory / "stage1_easy",
        seed=train_seed,
        batch_size=batch_size,
        epochs=curriculum_easy_epochs,
        learning_rate=5e-4,
        parameter_loss_weight=1.0,
        physics_loss_weight=0.0,
        node_count=8,
        patience=max(4, curriculum_easy_epochs),
    )
    stage1_full_model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    stage1_full_model.load_state_dict(copy.deepcopy(stage1_easy["model"].state_dict()))
    stage1_full = train_pinn(
        stage1_full_model,
        dataset,
        train_indices,
        validation_indices,
        output_directory / "stage1_full",
        seed=train_seed,
        batch_size=batch_size,
        epochs=curriculum_full_epochs,
        learning_rate=3e-4,
        parameter_loss_weight=1.5,
        physics_loss_weight=0.0,
        node_count=8,
        patience=max(4, curriculum_full_epochs),
    )
    warm_state = copy.deepcopy(stage1_full["model"].state_dict())
    finetune_candidates = [
        {
            "name": "ft_param_1p5_phys_0p10",
            "parameter_loss_weight": 1.5,
            "physics_loss_weight": 0.10,
            "learning_rate": 2e-4,
        },
        {
            "name": "ft_param_2p0_phys_0p10",
            "parameter_loss_weight": 2.0,
            "physics_loss_weight": 0.10,
            "learning_rate": 1e-4,
        },
    ]
    selected: dict[str, Any] | None = None
    selected_model_state: dict[str, torch.Tensor] | None = None
    candidate_results: list[dict[str, Any]] = []
    for candidate in finetune_candidates:
        model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
        model.load_state_dict(copy.deepcopy(warm_state))
        training = train_pinn(
            model,
            dataset,
            train_indices,
            validation_indices,
            output_directory / "stage2" / candidate["name"],
            seed=train_seed,
            batch_size=batch_size,
            epochs=finetune_epochs,
            learning_rate=float(candidate["learning_rate"]),
            parameter_loss_weight=float(candidate["parameter_loss_weight"]),
            physics_loss_weight=float(candidate["physics_loss_weight"]),
            node_count=8,
            patience=max(3, finetune_epochs),
        )
        result = {
            **candidate,
            "best_epoch": int(training["best_epoch"]),
            "best_validation_total_loss": float(training["best_validation_total_loss"]),
        }
        candidate_results.append(result)
        if selected is None or result["best_validation_total_loss"] < selected["best_validation_total_loss"]:
            selected = result
            selected_model_state = copy.deepcopy(training["model"].state_dict())
    if selected is None or selected_model_state is None:
        raise RuntimeError("No fine-tuning candidate was selected")
    selected_model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    selected_model.load_state_dict(selected_model_state)
    evaluation = _evaluate_model(
        selected_model,
        frame,
        dataset,
        test_indices,
        output_directory / "final_evaluation",
    )
    return {
        "stage1_easy_best_validation_total_loss": float(stage1_easy["best_validation_total_loss"]),
        "stage1_full_best_validation_total_loss": float(stage1_full["best_validation_total_loss"]),
        "stage2_candidates": sorted(
            candidate_results,
            key=lambda item: item["best_validation_total_loss"],
        ),
        "selected_stage2": selected,
        "test": evaluation,
    }


def _easy_curriculum_indices(
    dataset: SurfaceParameterDataset,
    train_indices: list[int],
    *,
    keep_fraction: float,
) -> list[int]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1]")
    scored = []
    for index in train_indices:
        target = dataset.targets[index].numpy()
        score = _difficulty_score(target)
        scored.append((score, index))
    scored.sort(key=lambda item: item[0])
    keep_count = max(3, int(np.ceil(len(scored) * keep_fraction)))
    return [index for _, index in scored[:keep_count]]


def _difficulty_score(target: np.ndarray) -> float:
    kappa_slow, theta_slow, sigma_slow, rho_slow, _v0_slow, kappa_fast, theta_fast, sigma_fast, rho_fast, _v0_fast = target
    fast_gap_pressure = sigma_fast / max(np.sqrt(2.0 * kappa_fast * theta_fast), 1e-8)
    slow_gap_pressure = sigma_slow / max(np.sqrt(2.0 * kappa_slow * theta_slow), 1e-8)
    correlation_pressure = abs(rho_slow) + abs(rho_fast)
    scale_separation = 1.0 / max(kappa_fast - kappa_slow, 1e-6)
    return float(
        1.5 * fast_gap_pressure
        + 0.75 * slow_gap_pressure
        + 1.25 * correlation_pressure
        + 0.50 * scale_separation
    )


def _evaluate_model(
    model: PhysicsInformedInverseCalibrator,
    frame: pd.DataFrame,
    dataset: SurfaceParameterDataset,
    indices: list[int],
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    predicted = predict_parameters(model, dataset, indices).numpy()
    truth = dataset.targets[indices].numpy()
    surface_ids = [dataset.surface_ids[index] for index in indices]
    parameter_recovery = evaluate_parameter_recovery(
        truth,
        predicted,
        surface_ids,
        output_directory / "parameter_recovery",
    )
    repricing = evaluate_repricing(
        frame.loc[frame["surface_id"].isin(surface_ids)].copy(),
        {
            surface_id: predicted[index]
            for index, surface_id in enumerate(surface_ids)
        },
        output_directory / "repricing",
    )
    focus = {
        name: {
            "mae": parameter_recovery["parameter_metrics"][name]["mae"],
            "rmse": parameter_recovery["parameter_metrics"][name]["rmse"],
        }
        for name in ("kappa_fast", "rho_fast", "v0_fast")
    }
    pd.DataFrame(
        {
            "surface_id": surface_ids,
            **{
                f"predicted_{name}": predicted[:, column]
                for column, name in enumerate(PARAMETER_NAMES)
            },
        }
    ).to_csv(output_directory / "test_predictions.csv", index=False)
    return {
        "parameter_recovery": parameter_recovery,
        "repricing": repricing,
        "focus_parameters": focus,
    }


def _delta_summary(improved: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "repricing_rmse": (
            improved["test"]["repricing"]["rmse"] - baseline["test"]["repricing"]["rmse"]
        ),
        "repricing_mae": (
            improved["test"]["repricing"]["mae"] - baseline["test"]["repricing"]["mae"]
        ),
        "rho_fast_mae": (
            improved["test"]["focus_parameters"]["rho_fast"]["mae"]
            - baseline["test"]["focus_parameters"]["rho_fast"]["mae"]
        ),
        "v0_fast_mae": (
            improved["test"]["focus_parameters"]["v0_fast"]["mae"]
            - baseline["test"]["focus_parameters"]["v0_fast"]["mae"]
        ),
        "kappa_fast_mae": (
            improved["test"]["focus_parameters"]["kappa_fast"]["mae"]
            - baseline["test"]["focus_parameters"]["kappa_fast"]["mae"]
        ),
    }


def _aggregate_run_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_rmse = np.asarray(
        [item["baseline"]["test"]["repricing"]["rmse"] for item in run_summaries],
        dtype=float,
    )
    improved_rmse = np.asarray(
        [item["improved"]["test"]["repricing"]["rmse"] for item in run_summaries],
        dtype=float,
    )
    baseline_rho_fast = np.asarray(
        [item["baseline"]["test"]["focus_parameters"]["rho_fast"]["mae"] for item in run_summaries],
        dtype=float,
    )
    improved_rho_fast = np.asarray(
        [item["improved"]["test"]["focus_parameters"]["rho_fast"]["mae"] for item in run_summaries],
        dtype=float,
    )
    baseline_v0_fast = np.asarray(
        [item["baseline"]["test"]["focus_parameters"]["v0_fast"]["mae"] for item in run_summaries],
        dtype=float,
    )
    improved_v0_fast = np.asarray(
        [item["improved"]["test"]["focus_parameters"]["v0_fast"]["mae"] for item in run_summaries],
        dtype=float,
    )
    return {
        "mean_baseline_repricing_rmse": float(baseline_rmse.mean()),
        "mean_improved_repricing_rmse": float(improved_rmse.mean()),
        "mean_repricing_rmse_delta": float((improved_rmse - baseline_rmse).mean()),
        "mean_baseline_rho_fast_mae": float(baseline_rho_fast.mean()),
        "mean_improved_rho_fast_mae": float(improved_rho_fast.mean()),
        "mean_rho_fast_mae_delta": float((improved_rho_fast - baseline_rho_fast).mean()),
        "mean_baseline_v0_fast_mae": float(baseline_v0_fast.mean()),
        "mean_improved_v0_fast_mae": float(improved_v0_fast.mean()),
        "mean_v0_fast_mae_delta": float((improved_v0_fast - baseline_v0_fast).mean()),
        "improved_beats_baseline_on_all_seed_repricing_rmse": bool(np.all(improved_rmse < baseline_rmse)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/pinn_improved_benchmark"),
    )
    parser.add_argument("--bounds", type=Path, default=Path("configs/parameter_bounds_PROVISIONAL.yaml"))
    parser.add_argument("--surface-count", type=int, default=90)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 22])
    parser.add_argument("--baseline-epochs", type=int, default=8)
    parser.add_argument("--curriculum-easy-epochs", type=int, default=6)
    parser.add_argument("--curriculum-full-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    summary = run_improved_pinn_benchmark(
        args.output,
        bounds_path=args.bounds,
        surface_count=args.surface_count,
        dataset_seed=args.dataset_seed,
        train_seeds=tuple(args.train_seeds),
        noise_level=args.noise_level,
        baseline_epochs=args.baseline_epochs,
        curriculum_easy_epochs=args.curriculum_easy_epochs,
        curriculum_full_epochs=args.curriculum_full_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
