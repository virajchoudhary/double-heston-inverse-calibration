"""Physics-informed inverse training for canonical Double Heston surfaces."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.parameter_transform import TargetStandardizer
from models.pinn_model import PhysicsInformedInverseCalibrator

from .constants import DEFAULT_SEED
from .dataset import SurfaceParameterDataset
from .torch_double_heston import price_double_heston_surface_batch
from .utils import set_deterministic_seed, write_json


def train_pinn(
    model: PhysicsInformedInverseCalibrator,
    dataset: SurfaceParameterDataset,
    train_indices: list[int],
    validation_indices: list[int],
    output_directory: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    batch_size: int = 64,
    epochs: int = 200,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.00001,
    patience: int = 20,
    parameter_loss_weight: float = 1.0,
    physics_loss_weight: float = 1.0,
    node_count: int = 64,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a PINN with parameter supervision plus repricing consistency."""
    if not train_indices or not validation_indices:
        raise ValueError("Both train_indices and validation_indices must be non-empty")
    if set(train_indices) & set(validation_indices):
        raise ValueError("Training and validation indices must be disjoint")
    if batch_size <= 0 or epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, epochs, and patience must be positive")
    if parameter_loss_weight < 0.0 or physics_loss_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    set_deterministic_seed(seed)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device)
    training_targets = dataset.targets[train_indices]
    standardizer = TargetStandardizer().fit(training_targets)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_indices,
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_indices,
        batch_size=min(batch_size, len(validation_indices)),
        shuffle=False,
    )
    optimizer = Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    mse = nn.MSELoss()
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
        for batch_indices in train_loader:
            batch = _prepare_batch(dataset, batch_indices, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch["features"])
            parameter_loss = mse(
                standardizer.transform(predictions),
                standardizer.transform(batch["targets"]),
            )
            repriced = price_double_heston_surface_batch(
                predictions,
                batch["spots"],
                batch["strikes"],
                batch["maturities"],
                batch["risk_free_rates"],
                batch["dividend_yields"],
                batch["option_types"],
                node_count=node_count,
            )
            normalized_prices = repriced / batch["spots"].unsqueeze(1)
            physics_loss = _masked_mse(
                normalized_prices,
                batch["features"],
                batch["masks"],
            )
            loss = parameter_loss_weight * parameter_loss + physics_loss_weight * physics_loss
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(batch_indices)
            train_examples += len(batch_indices)

        validation_metrics = _evaluate_validation(
            model,
            dataset,
            validation_loader,
            standardizer,
            resolved_device,
            parameter_loss_weight=parameter_loss_weight,
            physics_loss_weight=physics_loss_weight,
            node_count=node_count,
        )
        train_loss = train_total / train_examples
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_loss,
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_parameter_loss": validation_metrics["parameter_loss"],
                "validation_physics_loss": validation_metrics["physics_loss"],
            }
        )
        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "target_standardizer": standardizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_total_loss": best_validation_loss,
                    "seed": seed,
                    "input_size": model.input_size,
                    "output_size": model.output_size,
                    "device_used": str(resolved_device),
                    "selection_data": "validation_only",
                    "loss": "parameter_supervised_mse_plus_repricing_mse",
                    "physics_loss": "differentiable_double_heston_repricing",
                    "physics_loss_weight": physics_loss_weight,
                    "parameter_loss_weight": parameter_loss_weight,
                    "pricing_node_count": node_count,
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
        "best_validation_total_loss": best_validation_loss,
        "epochs_completed": len(history),
        "device_used": str(resolved_device),
        "checkpoint_path": checkpoint_path,
        "test_set_used_for_selection": False,
        "physics_loss": "differentiable_double_heston_repricing",
        "parameter_loss_weight": parameter_loss_weight,
        "physics_loss_weight": physics_loss_weight,
        "pricing_node_count": node_count,
    }
    write_json(
        output_path / "training_summary.json",
        {**summary, "checkpoint_path": checkpoint_path.name},
    )
    return {
        **summary,
        "model": model,
        "standardizer": standardizer,
        "history": history,
    }


def predict_parameters(
    model: PhysicsInformedInverseCalibrator,
    dataset: SurfaceParameterDataset,
    indices: list[int],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Predict constrained parameters for complete surfaces."""
    if not indices:
        raise ValueError("indices must be non-empty")
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        return model(dataset.features[indices].to(resolved_device)).cpu()


def _evaluate_validation(
    model: PhysicsInformedInverseCalibrator,
    dataset: SurfaceParameterDataset,
    loader: DataLoader[list[int]],
    standardizer: TargetStandardizer,
    device: torch.device,
    *,
    parameter_loss_weight: float,
    physics_loss_weight: float,
    node_count: int,
) -> dict[str, float]:
    mse = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_parameter = 0.0
    total_physics = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_indices in loader:
            batch = _prepare_batch(dataset, batch_indices, device)
            predictions = model(batch["features"])
            parameter_loss = mse(
                standardizer.transform(predictions),
                standardizer.transform(batch["targets"]),
            )
            repriced = price_double_heston_surface_batch(
                predictions,
                batch["spots"],
                batch["strikes"],
                batch["maturities"],
                batch["risk_free_rates"],
                batch["dividend_yields"],
                batch["option_types"],
                node_count=node_count,
            )
            normalized_prices = repriced / batch["spots"].unsqueeze(1)
            physics_loss = _masked_mse(
                normalized_prices,
                batch["features"],
                batch["masks"],
            )
            batch_total = parameter_loss_weight * parameter_loss + physics_loss_weight * physics_loss
            batch_size = len(batch_indices)
            total_loss += float(batch_total) * batch_size
            total_parameter += float(parameter_loss) * batch_size
            total_physics += float(physics_loss) * batch_size
            total_examples += batch_size
    return {
        "total_loss": total_loss / total_examples,
        "parameter_loss": total_parameter / total_examples,
        "physics_loss": total_physics / total_examples,
    }


def _prepare_batch(
    dataset: SurfaceParameterDataset,
    batch_indices: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    indices = [int(value) for value in batch_indices]
    metadata = [dataset.metadata[index] for index in indices]
    return {
        "features": dataset.features[indices].to(device),
        "targets": dataset.targets[indices].to(device),
        "masks": dataset.masks[indices].to(device),
        "spots": torch.tensor(
            [float(item["spot"]) for item in metadata],
            dtype=torch.float32,
            device=device,
        ),
        "risk_free_rates": torch.tensor(
            [float(item["risk_free_rate"]) for item in metadata],
            dtype=torch.float32,
            device=device,
        ),
        "dividend_yields": torch.tensor(
            [float(item["dividend_yield"]) for item in metadata],
            dtype=torch.float32,
            device=device,
        ),
        "strikes": torch.tensor(
            [item["strikes"] for item in metadata],
            dtype=torch.float32,
            device=device,
        ),
        "maturities": torch.tensor(
            [item["maturities_years"] for item in metadata],
            dtype=torch.float32,
            device=device,
        ),
        "option_types": [list(item["option_types"]) for item in metadata],
    }


def _masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask.to(dtype=predictions.dtype)
    denominator = active.sum().clamp_min(1.0)
    return torch.sum(((predictions - targets) ** 2) * active) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--allow-not-research-data", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset_csv)
    dataset = SurfaceParameterDataset.from_surface_frame(
        frame,
        allow_not_research_data=args.allow_not_research_data,
    )
    set_deterministic_seed(DEFAULT_SEED)
    model = PhysicsInformedInverseCalibrator(input_size=dataset.features.shape[1])
    train_pinn(
        model,
        dataset,
        dataset.indices_for_split("train"),
        dataset.indices_for_split("validation"),
        args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
