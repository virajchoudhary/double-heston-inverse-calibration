"""Deterministic supervised training for the ordinary ANN baseline."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from ann_inverse_calibration.models.ann_model import ANNInverseCalibrator
from ann_inverse_calibration.models.parameter_transform import TargetStandardizer

from .constants import DEFAULT_SEED
from .dataset import SurfaceParameterDataset
from .utils import set_deterministic_seed, write_json


def train_ann(
    model: ANNInverseCalibrator,
    dataset: SurfaceParameterDataset,
    train_indices: list[int],
    validation_indices: list[int],
    output_directory: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    batch_size: int = 256,
    epochs: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 0.00001,
    patience: int = 20,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train with target MSE and validation-only early stopping/checkpointing."""
    if not train_indices or not validation_indices:
        raise ValueError("Both train_indices and validation_indices must be non-empty")
    if set(train_indices) & set(validation_indices):
        raise ValueError("Training and validation indices must be disjoint")
    if batch_size <= 0 or epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, epochs, and patience must be positive")
    set_deterministic_seed(seed)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device)
    training_targets = dataset.targets[train_indices]
    standardizer = TargetStandardizer().fit(training_targets)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=min(batch_size, len(validation_indices)),
        shuffle=False,
    )
    optimizer = Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_validation_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_examples = 0
        for features, targets, _, _ in train_loader:
            features = features.to(resolved_device)
            targets = targets.to(resolved_device)
            standardized_targets = standardizer.transform(targets)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, standardized_targets)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(features)
            train_examples += len(features)

        validation_loss = _evaluate_loss(
            model, validation_loader, standardizer, loss_function, resolved_device
        )
        train_loss = train_total / train_examples
        history.append(
            {
                "epoch": epoch,
                "train_mse_standardized": train_loss,
                "validation_mse_standardized": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "target_standardizer": standardizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_mse_standardized": best_validation_loss,
                    "seed": seed,
                    "input_size": model.input_size,
                    "output_size": model.output_size,
                    "device_used": str(resolved_device),
                    "selection_data": "validation_only",
                    "loss": "parameter_supervised_mse",
                    "pde_loss": False,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_path / "training_history.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_mse_standardized": best_validation_loss,
        "epochs_completed": len(history),
        "device_used": str(resolved_device),
        "checkpoint_path": checkpoint_path,
        "test_set_used_for_selection": False,
        "pde_loss": False,
    }
    serialized_summary = {**summary, "checkpoint_path": checkpoint_path.name}
    write_json(output_path / "training_summary.json", serialized_summary)
    return {
        **summary,
        "model": model,
        "standardizer": standardizer,
        "history": history,
    }


def predict_parameters(
    model: ANNInverseCalibrator,
    dataset: SurfaceParameterDataset,
    indices: list[int],
    standardizer: TargetStandardizer,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Predict parameters in original target units for selected complete surfaces."""
    if not indices:
        raise ValueError("indices must be non-empty")
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        standardized = model(dataset.features[indices].to(resolved_device))
        return standardizer.inverse_transform(standardized).cpu()


def _evaluate_loss(
    model: ANNInverseCalibrator,
    loader: DataLoader[Any],
    standardizer: TargetStandardizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    with torch.no_grad():
        for features, targets, _, _ in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = loss_function(predictions, standardizer.transform(targets))
            total += float(loss) * len(features)
            examples += len(features)
    return total / examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--allow-not-research-data", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset_csv)
    dataset = SurfaceParameterDataset.from_surface_frame(
        frame, allow_not_research_data=args.allow_not_research_data
    )
    set_deterministic_seed(DEFAULT_SEED)
    model = ANNInverseCalibrator(input_size=dataset.features.shape[1])
    train_ann(
        model,
        dataset,
        dataset.indices_for_split("train"),
        dataset.indices_for_split("validation"),
        args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
