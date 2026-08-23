"""Frozen-protocol training entrypoints for the R2 primary comparison.

Implements exactly the frozen configuration in
``configs/r2_primary_comparison_FINAL.yaml`` (protocol sections MODEL 1 /
MODEL 2): single shared R2 feature builder, train-split-only target
standardization, validation-only early stopping and checkpointing, and
provenance-rich run artifacts.  No test-split data is ever loaded by these
entrypoints.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from models.ann_model import ANNInverseCalibrator
from models.parameter_transform import TargetStandardizer
from models.pinn_model import PhysicsInformedInverseCalibrator

from ..constants import PARAMETER_NAMES
from ..torch_double_heston import price_double_heston_surface_batch_vectorized
from ..utils import set_deterministic_seed, write_json
from .dataset import R2PrimaryDataset

PROTOCOL_CONFIG_PATH = Path("configs/r2_primary_comparison_FINAL.yaml")
CHECKPOINT_ROOT = Path("checkpoints/r2_primary_comparison")

# Frozen protocol hyperparameters (mirror of r2_primary_comparison_FINAL.yaml;
# consistency asserted by tests; do not change without a new frozen protocol).
MODEL1_SPEC: dict[str, Any] = {
    "hidden_sizes": [512, 256, 128, 64],
    "activation": "relu",
    "dropout": 0.10,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
}
MODEL2_SPEC: dict[str, Any] = {
    "hidden_sizes": [512, 512, 256, 256, 128],
    "activation": "gelu",
    "dropout": 0.05,
    "learning_rate": 0.0005,
    "weight_decay": 0.00001,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 20,
    "parameter_loss_weight": 1.0,
    "repricing_loss_weight": 1.0,
    "pricing_node_count": 64,
    "repricing_compute_dtype": "float64",
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def _environment_summary() -> dict[str, Any]:
    return {
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


def build_model1() -> ANNInverseCalibrator:
    return ANNInverseCalibrator(
        input_size=100,
        hidden_sizes=MODEL1_SPEC["hidden_sizes"],
        activation=MODEL1_SPEC["activation"],
        dropout=MODEL1_SPEC["dropout"],
    )


def build_model2() -> PhysicsInformedInverseCalibrator:
    return PhysicsInformedInverseCalibrator(
        input_size=100,
        hidden_sizes=MODEL2_SPEC["hidden_sizes"],
        activation=MODEL2_SPEC["activation"],
        dropout=MODEL2_SPEC["dropout"],
    )


def _repricing_loss(
    predicted_parameters: torch.Tensor,
    batch_items: list,
    node_count: int,
) -> torch.Tensor:
    """Masked MSE of repriced normalized prices vs observed (float64)."""
    batch_size = len(batch_items)
    spots = torch.tensor(
        [item.spot for item in batch_items], dtype=torch.float64
    )
    strikes = torch.tensor(
        np.stack([item.strikes for item in batch_items]), dtype=torch.float64
    )
    maturities = torch.tensor(
        np.stack([item.maturities for item in batch_items]), dtype=torch.float64
    )
    rates = torch.tensor(
        [item.rate for item in batch_items], dtype=torch.float64
    )
    carries = torch.tensor(
        [item.carry for item in batch_items], dtype=torch.float64
    )
    option_types = [list(item.option_types) for item in batch_items]
    observed = torch.tensor(
        np.stack([item.normalized_prices for item in batch_items]),
        dtype=torch.float64,
    )
    mask = torch.tensor(
        np.stack([item.mask for item in batch_items]), dtype=torch.float64
    )
    repriced = price_double_heston_surface_batch_vectorized(
        predicted_parameters.to(torch.float64),
        spots,
        strikes,
        maturities,
        rates,
        carries,
        option_types,
        node_count=node_count,
    )
    normalized = repriced / spots.unsqueeze(1)
    active = mask
    denominator = active.sum().clamp_min(1.0)
    return torch.sum(((normalized - observed) ** 2) * active) / denominator


def train_model1(
    dataset: R2PrimaryDataset,
    seed: int,
    output_directory: str | Path,
    *,
    max_epochs: int | None = None,
    max_train_surfaces: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train the ordinary ANN under the frozen protocol (validation only)."""
    spec = dict(MODEL1_SPEC)
    if max_epochs is not None:  # smoke runs only
        spec["max_epochs"] = max_epochs
    train_indices = dataset.indices_for_split("train")
    validation_indices = dataset.indices_for_split("validation")
    if not train_indices or not validation_indices:
        raise ValueError("train and validation splits must both be present")
    if max_train_surfaces is not None:
        train_indices = train_indices[:max_train_surfaces]
        validation_indices = validation_indices[:max_train_surfaces]

    set_deterministic_seed(seed)
    resolved_device = torch.device(device or "cpu")
    model = build_model1().to(resolved_device)
    standardizer = TargetStandardizer().fit(dataset.targets[train_indices])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_indices,
        batch_size=min(spec["batch_size"], len(train_indices)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_indices,
        batch_size=min(spec["batch_size"], len(validation_indices)),
        shuffle=False,
    )
    optimizer = Adam(
        model.parameters(), spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    loss_function = nn.MSELoss()

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, spec["max_epochs"] + 1):
        model.train()
        train_total = 0.0
        train_examples = 0
        for batch_indices in train_loader:
            features = torch.as_tensor(
                np.stack([dataset.items[int(index)].features for index in batch_indices])
            ).to(resolved_device)
            targets = torch.stack(
                [dataset.targets[int(index)] for index in batch_indices]
            ).to(resolved_device)
            standardized_targets = standardizer.transform(targets)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(
                predictions.to(torch.float64), standardized_targets.to(torch.float64)
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(features)
            train_examples += len(features)
        validation_loss = _model1_validation_loss(
            model, validation_loader, dataset, standardizer, loss_function, resolved_device
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_standardized": train_total / train_examples,
                "validation_mse_standardized": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= spec["patience"]:
                break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - started
    provenance = {
        "run_kind": "RESEARCH" if max_train_surfaces is None else "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT",
        "model": "model1_ordinary_ann",
        "seed": seed,
        "spec": spec,
        "git_sha": _git_sha(),
        "protocol_config": str(PROTOCOL_CONFIG_PATH),
        "parameter_order": list(PARAMETER_NAMES),
        "device_used": str(resolved_device),
        "environment": _environment_summary(),
        "test_set_used_for_selection": False,
        "selection_data": "validation_only",
        "loss": "parameter_supervised_mse",
        "repricing_loss": False,
        "train_surfaces": len(train_indices),
        "validation_surfaces": len(validation_indices),
        "runtime_seconds": runtime_seconds,
    }
    checkpoint = {
        "model_state_dict": best_state,
        "target_standardizer": standardizer.state_dict(),
        "best_epoch": best_epoch,
        "best_validation_mse_standardized": best_validation_loss,
        **provenance,
    }
    torch.save(checkpoint, output_path / "best_validation_checkpoint.pt")
    torch.save(
        {**checkpoint, "model_state_dict": model.state_dict(), "checkpoint_kind": "final_state"},
        output_path / "final_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output_path / "training_history.csv", index=False)
    write_json(
        output_path / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_validation_mse_standardized": best_validation_loss,
            "epochs_completed": len(history),
            **{key: value for key, value in provenance.items() if key != "spec"},
        },
    )
    return {
        "best_epoch": best_epoch,
        "best_validation_mse_standardized": best_validation_loss,
        "epochs_completed": len(history),
        "model": model,
        "standardizer": standardizer,
        "history": history,
        "runtime_seconds": runtime_seconds,
    }


def _model1_validation_loss(
    model: ANNInverseCalibrator,
    loader: DataLoader[Any],
    dataset: R2PrimaryDataset,
    standardizer: TargetStandardizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    with torch.no_grad():
        for batch_indices in loader:
            features = torch.as_tensor(
                np.stack([dataset.items[int(index)].features for index in batch_indices])
            ).to(device)
            targets = torch.stack(
                [dataset.targets[int(index)] for index in batch_indices]
            ).to(device)
            predictions = model(features)
            loss = loss_function(
                predictions.to(torch.float64),
                standardizer.transform(targets).to(torch.float64),
            )
            total += float(loss) * len(features)
            examples += len(features)
    return total / examples


def train_model2(
    dataset: R2PrimaryDataset,
    seed: int,
    output_directory: str | Path,
    *,
    max_epochs: int | None = None,
    max_train_surfaces: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train the constraint + repricing-informed model under the frozen protocol."""
    spec = dict(MODEL2_SPEC)
    if max_epochs is not None:  # smoke runs only
        spec["max_epochs"] = max_epochs
    train_indices = dataset.indices_for_split("train")
    validation_indices = dataset.indices_for_split("validation")
    if not train_indices or not validation_indices:
        raise ValueError("train and validation splits must both be present")
    if max_train_surfaces is not None:
        train_indices = train_indices[:max_train_surfaces]
        validation_indices = validation_indices[:max_train_surfaces]

    set_deterministic_seed(seed)
    resolved_device = torch.device(device or "cpu")
    model = build_model2().to(resolved_device)
    standardizer = TargetStandardizer().fit(dataset.targets[train_indices])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_indices,
        batch_size=min(spec["batch_size"], len(train_indices)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_indices,
        batch_size=min(spec["batch_size"], len(validation_indices)),
        shuffle=False,
    )
    optimizer = Adam(
        model.parameters(), spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    mse = nn.MSELoss()
    node_count = spec["pricing_node_count"]

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, spec["max_epochs"] + 1):
        model.train()
        train_total = 0.0
        train_examples = 0
        for batch_indices in train_loader:
            batch_items = [dataset.items[int(index)] for index in batch_indices]
            features = torch.as_tensor(
                np.stack([item.features for item in batch_items])
            ).to(resolved_device)
            targets = torch.stack(
                [dataset.targets[int(index)] for index in batch_indices]
            ).to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            parameter_loss = mse(
                standardizer.transform(predictions).to(torch.float64),
                standardizer.transform(targets).to(torch.float64),
            )
            repricing_loss = _repricing_loss(predictions, batch_items, node_count)
            loss = (
                spec["parameter_loss_weight"] * parameter_loss
                + spec["repricing_loss_weight"] * repricing_loss
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(batch_items)
            train_examples += len(batch_items)

        validation_metrics = _model2_validation_metrics(
            dataset, validation_loader, model, standardizer, resolved_device, node_count, spec
        )
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total / train_examples,
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_parameter_loss": validation_metrics["parameter_loss"],
                "validation_repricing_loss": validation_metrics["repricing_loss"],
            }
        )
        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= spec["patience"]:
                break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - started
    provenance = {
        "run_kind": "RESEARCH" if max_train_surfaces is None else "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT",
        "model": "model2_constraint_repricing_informed",
        "seed": seed,
        "spec": spec,
        "git_sha": _git_sha(),
        "protocol_config": str(PROTOCOL_CONFIG_PATH),
        "parameter_order": list(PARAMETER_NAMES),
        "device_used": str(resolved_device),
        "environment": _environment_summary(),
        "test_set_used_for_selection": False,
        "selection_data": "validation_only",
        "loss": "parameter_supervised_mse_plus_repricing_mse",
        "repricing_loss": "differentiable_double_heston_repricing_float64_vectorized",
        "pricing_node_count": node_count,
        "loss_weights": {
            "parameter_loss_weight": spec["parameter_loss_weight"],
            "repricing_loss_weight": spec["repricing_loss_weight"],
        },
        "train_surfaces": len(train_indices),
        "validation_surfaces": len(validation_indices),
        "runtime_seconds": runtime_seconds,
    }
    checkpoint = {
        "model_state_dict": best_state,
        "target_standardizer": standardizer.state_dict(),
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_validation_loss,
        **provenance,
    }
    torch.save(checkpoint, output_path / "best_validation_checkpoint.pt")
    torch.save(
        {**checkpoint, "model_state_dict": model.state_dict(), "checkpoint_kind": "final_state"},
        output_path / "final_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output_path / "training_history.csv", index=False)
    write_json(
        output_path / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_validation_total_loss": best_validation_loss,
            "epochs_completed": len(history),
            **{key: value for key, value in provenance.items() if key != "spec"},
        },
    )
    return {
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_validation_loss,
        "epochs_completed": len(history),
        "model": model,
        "standardizer": standardizer,
        "history": history,
        "runtime_seconds": runtime_seconds,
    }


def _model2_validation_metrics(
    dataset: R2PrimaryDataset,
    loader: DataLoader[Any],
    model: PhysicsInformedInverseCalibrator,
    standardizer: TargetStandardizer,
    device: torch.device,
    node_count: int,
    spec: dict[str, Any],
) -> dict[str, float]:
    mse = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_parameter = 0.0
    total_repricing = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_indices in loader:
            batch_items = [dataset.items[int(index)] for index in batch_indices]
            features = torch.as_tensor(
                np.stack([item.features for item in batch_items])
            ).to(device)
            targets = torch.stack(
                [dataset.targets[int(index)] for index in batch_indices]
            ).to(device)
            predictions = model(features)
            parameter_loss = mse(
                standardizer.transform(predictions).to(torch.float64),
                standardizer.transform(targets).to(torch.float64),
            )
            repricing_loss = _repricing_loss(predictions, batch_items, node_count)
            batch_total = (
                spec["parameter_loss_weight"] * parameter_loss
                + spec["repricing_loss_weight"] * repricing_loss
            )
            total_loss += float(batch_total) * len(batch_items)
            total_parameter += float(parameter_loss) * len(batch_items)
            total_repricing += float(repricing_loss) * len(batch_items)
            total_examples += len(batch_items)
    return {
        "total_loss": total_loss / total_examples,
        "parameter_loss": total_parameter / total_examples,
        "repricing_loss": total_repricing / total_examples,
    }


def predict_parameters(
    model: nn.Module,
    dataset: R2PrimaryDataset,
    indices: list[int],
    *,
    standardizer: TargetStandardizer | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Predict physical-unit parameters for the given dataset indices."""
    if not indices:
        raise ValueError("indices must be non-empty")
    resolved_device = torch.device(device or "cpu")
    model = model.to(resolved_device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            chunk = indices[start : start + 256]
            features = torch.as_tensor(
                np.stack([dataset.items[index].features for index in chunk])
            ).to(resolved_device)
            output = model(features)
            if standardizer is not None:
                output = standardizer.inverse_transform(output)
            predictions.append(output.to(torch.float64).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def load_run(checkpoint_directory: str | Path, model_kind: str) -> dict[str, Any]:
    """Load a best-validation checkpoint with its provenance intact."""
    path = Path(checkpoint_directory) / "best_validation_checkpoint.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model_kind == "model1":
        model = build_model1()
    elif model_kind == "model2":
        model = build_model2()
    else:
        raise ValueError("model_kind must be 'model1' or 'model2'")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    standardizer = TargetStandardizer()
    standardizer.mean = payload["target_standardizer"]["mean"]
    standardizer.scale = payload["target_standardizer"]["scale"]
    return {"model": model, "standardizer": standardizer, "payload": payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["model1", "model2"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/final_r2_clean_10000/surfaces.jsonl")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT pipeline run",
    )
    args = parser.parse_args()

    # Wall-clock scheduling only (never numerics): allow capping torch's
    # intra-op threads when sharing the machine with calibration workers.
    thread_cap = os.environ.get("R2_TORCH_THREADS")
    if thread_cap:
        torch.set_num_threads(max(1, int(thread_cap)))

    dataset = R2PrimaryDataset.from_jsonl(args.dataset)
    output = args.output or (
        CHECKPOINT_ROOT / f"{args.model}_seed{args.seed}"
        if not args.smoke
        else CHECKPOINT_ROOT / "smoke" / f"{args.model}_seed{args.seed}"
    )
    if args.smoke:
        print("DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT")
        if args.model == "model1":
            result = train_model1(
                dataset, args.seed, output, max_epochs=2, max_train_surfaces=64
            )
        else:
            result = train_model2(
                dataset, args.seed, output, max_epochs=2, max_train_surfaces=64
            )
    else:
        if output.exists() and any(output.glob("best_validation_checkpoint.pt")):
            raise SystemExit(f"refusing to overwrite existing research run at {output}")
        if args.model == "model1":
            result = train_model1(dataset, args.seed, output)
        else:
            result = train_model2(dataset, args.seed, output)
    print(
        f"{args.model} seed {args.seed}: best_epoch={result['best_epoch']} "
        f"epochs={result['epochs_completed']} runtime={result['runtime_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
