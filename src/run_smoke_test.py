"""Quick end-to-end ordinary ANN infrastructure check using dummy data only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ann_inverse_calibration.models.ann_model import ANNInverseCalibrator

from .constants import DEFAULT_SEED, NOT_RESEARCH_DATA
from .dataset import SurfaceParameterDataset
from .evaluate_parameters import evaluate_parameter_recovery
from .surface_grid import expected_input_size
from .synthetic_dataset import generate_smoke_test_dataset
from .train import predict_parameters, train_ann
from .utils import set_deterministic_seed, write_json


def run_smoke_test(
    output_directory: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate dummy data, train briefly, and verify saved outputs and shapes."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = (
        Path(output_directory)
        if output_directory is not None
        else project_root / "outputs" / "metrics" / "smoke_test"
    )
    if "smoke_test" not in {part.lower() for part in output_path.parts}:
        raise ValueError("Smoke outputs must be written beneath a smoke_test path")
    set_deterministic_seed(seed)
    data_path = output_path / "data"
    frame = generate_smoke_test_dataset(data_path, n_surfaces=48, seed=seed)
    dataset = SurfaceParameterDataset.from_surface_frame(
        frame, allow_not_research_data=True
    )
    train_indices = dataset.indices_for_split("train")
    validation_indices = dataset.indices_for_split("validation")
    test_indices = dataset.indices_for_split("test")
    model = ANNInverseCalibrator(input_size=expected_input_size())
    training = train_ann(
        model,
        dataset,
        train_indices,
        validation_indices,
        output_path,
        seed=seed,
        batch_size=16,
        epochs=3,
        patience=3,
        device="cpu",
    )
    predictions = predict_parameters(
        training["model"],
        dataset,
        test_indices,
        training["standardizer"],
        device="cpu",
    )
    expected_shape = (len(test_indices), 10)
    if tuple(predictions.shape) != expected_shape:
        raise RuntimeError(
            f"Prediction shape {tuple(predictions.shape)} != expected {expected_shape}"
        )
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Smoke predictions contain non-finite values")
    test_surface_ids = [dataset.surface_ids[index] for index in test_indices]
    evaluate_parameter_recovery(
        dataset.targets[test_indices].numpy(),
        predictions.numpy(),
        test_surface_ids,
        output_path / "parameter_evaluation",
    )
    pd.DataFrame(
        predictions.numpy(),
        index=test_surface_ids,
    ).to_csv(output_path / "smoke_predictions.csv", index_label="surface_id")
    marker_path = output_path / NOT_RESEARCH_DATA
    marker_path.write_text(
        "Development-only dummy mapping. Not Double Heston pricing. "
        "Do not use as research evidence.\n",
        encoding="utf-8",
    )
    summary = {
        "status": "passed",
        "data_status": NOT_RESEARCH_DATA,
        "warning": "Dummy mapping only; no ANN research result is claimed.",
        "surface_count": len(dataset),
        "train_surfaces": len(train_indices),
        "validation_surfaces": len(validation_indices),
        "test_surfaces": len(test_indices),
        "input_size": dataset.features.shape[1],
        "prediction_shape": list(predictions.shape),
        "checkpoint_created": Path(training["checkpoint_path"]).is_file(),
        "loss_computed": len(training["history"]) > 0,
        "pde_loss": False,
    }
    write_json(output_path / "smoke_test_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = run_smoke_test(args.output_directory, args.seed)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
