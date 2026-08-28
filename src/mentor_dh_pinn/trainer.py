"""Deterministic AdamW trainer for the mentor Double Heston forward PINN."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .collocation import (
    sample_high_s_boundary_points,
    sample_low_s_boundary_points,
    sample_pde_points,
    sample_terminal_points,
)
from .config import BaselineConfig, baseline_config_from_mapping
from .losses import (
    LossComponents,
    data_loss,
    high_s_boundary_loss,
    low_s_boundary_loss,
    pde_loss,
    terminal_loss,
    weighted_total_loss,
)
from .model import DoubleHestonForwardPINN
from .synthetic_data import SyntheticDataset, validate_dataset_identity


@dataclass(frozen=True)
class TrainingResult:
    """Paths and checkpoint selection information returned by ``train_baseline``."""

    output_dir: Path
    checkpoint_path: Path
    best_epoch: int
    epochs_completed: int
    best_validation_nrmse: float
    train_history_path: Path
    validation_history_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "best_validation_nrmse": self.best_validation_nrmse,
            "train_history_path": str(self.train_history_path),
            "validation_history_path": str(self.validation_history_path),
        }


def seed_everything(seed: int) -> None:
    """Set all local RNGs used by this trainer."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # The CPU path used by the baseline supports deterministic kernels;
        # retaining this fallback keeps diagnostic smoke runs portable.
        pass


def _git_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def runtime_identity(device: torch.device | str) -> dict[str, Any]:
    resolved = torch.device(device)
    identity: dict[str, Any] = {
        "device": str(resolved),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if resolved.type == "cuda":
        identity.update(
            cuda_device_name=torch.cuda.get_device_name(resolved),
            cuda_device_capability=list(torch.cuda.get_device_capability(resolved)),
        )
    return identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_indices(indices: np.ndarray, batch_size: int, *, seed: int) -> list[np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).numpy()
    shuffled = indices[order]
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def _as_float64_tensor(
    values: np.ndarray | torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float64)
    return torch.from_numpy(np.asarray(values, dtype=np.float64)).to(device=device)


def validation_metrics(
    model: torch.nn.Module,
    dataset: SyntheticDataset,
    indices: np.ndarray,
) -> dict[str, float]:
    """Evaluate only the supplied validation rows."""
    if len(indices) == 0:
        raise ValueError("validation indices must be non-empty")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        features = _as_float64_tensor(dataset.features[indices], device)
        references = _as_float64_tensor(dataset.reference_prices[indices], device)
        predictions = model(features).reshape(-1)
    errors = predictions - references
    normalized = errors / features[:, 0]
    return {
        "validation_rmse": float(torch.sqrt(torch.mean(errors.square())).item()),
        "validation_mae": float(torch.mean(errors.abs()).item()),
        "validation_nrmse": float(torch.sqrt(torch.mean(normalized.square())).item()),
    }


def _gradient_summary(model: torch.nn.Module) -> tuple[float, bool]:
    squared_sum = 0.0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            finite = False
            break
        squared_sum += float(torch.sum(parameter.grad.detach().square()).item())
    return math.sqrt(squared_sum), finite


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty history")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_identity(dataset: SyntheticDataset) -> dict[str, Any]:
    manifest = dict(dataset.manifest)
    # The storage filename is a locator, not scientific identity, and is added
    # only when the manifest is persisted. Excluding it keeps in-memory and
    # freshly reloaded representations identical while hashes remain binding.
    manifest.pop("dataset_filename", None)
    manifest["split_id_hashes"] = dataset.split_id_hashes()
    manifest["point_count"] = dataset.size
    return manifest


def validate_checkpoint_identities(
    checkpoint: dict[str, Any],
    dataset: SyntheticDataset,
    config: BaselineConfig,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Cross-check checkpoint, config, dataset, parameter, and git identities."""
    validate_dataset_identity(dataset, config)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    checks = (
        (checkpoint.get("config") == config.to_dict(), "checkpoint/config"),
        (checkpoint.get("dataset_identity") == _dataset_identity(dataset), "checkpoint/dataset"),
        (checkpoint.get("provenance") == dataset.parameter_source.provenance(), "checkpoint/parameter"),
        (checkpoint.get("git_sha") not in (None, "", "unknown"), "checkpoint git identity"),
        (checkpoint.get("git_sha") == _git_sha(root), "checkpoint/current git identity"),
    )
    for passed, name in checks:
        if not passed:
            raise ValueError(f"identity mismatch: {name}")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    config: BaselineConfig,
    dataset: SyntheticDataset,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_nrmse": best_metric,
        "config": config.to_dict(),
        "provenance": dataset.parameter_source.provenance(),
        "dataset_identity": _dataset_identity(dataset),
        "git_sha": _git_sha(repo_root),
        "seed": config.seed,
        "weights": config.losses.weights,
        "network_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime_identity": runtime_identity(next(model.parameters()).device),
    }


def train_baseline(
    model: DoubleHestonForwardPINN,
    dataset: SyntheticDataset,
    output_dir: str | Path,
    *,
    config: BaselineConfig,
    cohort_config: BaselineConfig | None = None,
    repo_root: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train the forward PINN and persist histories plus the best checkpoint."""
    config.validate()
    validate_dataset_identity(dataset, cohort_config or config)
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")
    train_indices = dataset.indices("train")
    validation_indices = dataset.indices("validation")
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("train and validation splits must be non-empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    seed_everything(config.seed)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
    model = model.double().to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    checkpoint_path = output / "checkpoint.pt"
    train_history_path = output / "train_history.csv"
    validation_history_path = output / "validation_history.csv"
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.training.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        batch_rows: list[dict[str, float]] = []
        finite_gradients = True
        batch_list = _batch_indices(
            train_indices,
            config.training.batch_size,
            seed=config.seed + epoch,
        )
        for batch_number, batch in enumerate(batch_list):
            features = _as_float64_tensor(dataset.features[batch], resolved_device)
            references = _as_float64_tensor(dataset.reference_prices[batch], resolved_device)
            optimizer.zero_grad(set_to_none=True)
            data_component = data_loss(model(features), references, features[:, 0])
            pde_points = sample_pde_points(
                config.training.pde_batch_size,
                config=config,
                parameter_source=dataset.parameter_source,
                seed=config.seed
                + config.training.collocation_seed_offset
                + epoch * config.training.epoch_seed_stride
                + batch_number,
                device=resolved_device,
            )
            pde_component, pde_residual_values = pde_loss(
                model,
                pde_points,
                scale_floor=config.losses.pde_scale_floor,
            )
            terminal_points = sample_terminal_points(
                config.training.terminal_batch_size,
                config=config,
                parameter_source=dataset.parameter_source,
                seed=config.seed
                + config.training.terminal_seed_offset
                + epoch * config.training.epoch_seed_stride
                + batch_number,
                device=resolved_device,
            )
            terminal_component, terminal_predictions, terminal_targets = terminal_loss(
                model, terminal_points
            )
            low_points = sample_low_s_boundary_points(
                config.training.boundary_batch_size,
                config=config,
                parameter_source=dataset.parameter_source,
                seed=config.seed
                + config.training.low_boundary_seed_offset
                + epoch * config.training.epoch_seed_stride
                + batch_number,
                device=resolved_device,
            )
            low_component, low_predictions, low_targets = low_s_boundary_loss(model, low_points)
            high_points = sample_high_s_boundary_points(
                config.training.boundary_batch_size,
                config=config,
                parameter_source=dataset.parameter_source,
                seed=config.seed
                + config.training.high_boundary_seed_offset
                + epoch * config.training.epoch_seed_stride
                + batch_number,
                device=resolved_device,
            )
            high_component, high_predictions, high_targets = high_s_boundary_loss(
                model, high_points
            )
            components = LossComponents(
                data=data_component,
                pde=pde_component,
                terminal=terminal_component,
                low_boundary=low_component,
                high_boundary=high_component,
            )
            total = weighted_total_loss(components, config.losses.weights)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite total loss at epoch {epoch}")
            total.backward()
            gradient_norm, gradients_are_finite = _gradient_summary(model)
            finite_gradients = finite_gradients and gradients_are_finite
            if not gradients_are_finite:
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            if config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            optimizer.step()
            with torch.no_grad():
                data_predictions = model(features).reshape(-1)
                data_rmse = torch.sqrt(torch.mean((data_predictions - references).square()))
            batch_rows.append(
                {
                    "total_loss": float(total.detach().item()),
                    "data_loss": float(data_component.detach().item()),
                    "pde_loss": float(pde_component.detach().item()),
                    "terminal_loss": float(terminal_component.detach().item()),
                    "low_boundary_loss": float(low_component.detach().item()),
                    "high_boundary_loss": float(high_component.detach().item()),
                    "raw_rmse": float(data_rmse.item()),
                    "pde_rms": float(torch.sqrt(torch.mean(pde_residual_values.detach().square())).item()),
                    "terminal_rmse": float(
                        torch.sqrt(torch.mean((terminal_predictions.detach() - terminal_targets).square())).item()
                    ),
                    "low_boundary_rmse": float(
                        torch.sqrt(torch.mean((low_predictions.detach() - low_targets).square())).item()
                    ),
                    "high_boundary_rmse": float(
                        torch.sqrt(torch.mean((high_predictions.detach() - high_targets).square())).item()
                    ),
                    "gradient_norm": gradient_norm,
                }
            )
        if not finite_gradients:
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        means = {
            key: float(np.mean([row[key] for row in batch_rows]))
            for key in batch_rows[0]
        }
        validation = validation_metrics(model, dataset, validation_indices)
        duration = time.perf_counter() - epoch_started
        train_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": means["total_loss"],
                "train_pde_loss": means["pde_loss"],
                "train_boundary_loss": means["low_boundary_loss"] + means["high_boundary_loss"],
                "train_boundary_low_loss": means["low_boundary_loss"],
                "train_boundary_high_loss": means["high_boundary_loss"],
                "train_terminal_loss": means["terminal_loss"],
                "train_data_loss": means["data_loss"],
                "validation_price_rmse": validation["validation_rmse"],
                "validation_price_mae": validation["validation_mae"],
                "validation_nrmse": validation["validation_nrmse"],
                "pde_residual_rms": means["pde_rms"],
                "terminal_rmse": means["terminal_rmse"],
                "boundary_low_rmse": means["low_boundary_rmse"],
                "boundary_high_rmse": means["high_boundary_rmse"],
                "gradient_norm": means["gradient_norm"],
                "finite_gradients": bool(finite_gradients),
                "duration_seconds": duration,
            }
        )
        validation_rows.append(
            {
                "epoch": epoch,
                "validation_price_rmse": validation["validation_rmse"],
                "validation_price_mae": validation["validation_mae"],
                "validation_nrmse": validation["validation_nrmse"],
            }
        )
        improved = validation["validation_nrmse"] < best_metric
        if improved:
            best_metric = validation["validation_nrmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    config=config,
                    dataset=dataset,
                    repo_root=root,
                ),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.patience:
                break
        _write_csv(train_history_path, train_rows)
        _write_csv(validation_history_path, validation_rows)

    if best_epoch == 0:
        raise RuntimeError("training completed without a finite validation checkpoint")
    _write_csv(train_history_path, train_rows)
    _write_csv(validation_history_path, validation_rows)
    selected_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_checkpoint["epochs_completed"] = len(train_rows)
    torch.save(selected_checkpoint, checkpoint_path)
    return TrainingResult(
        output_dir=output,
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        epochs_completed=len(train_rows),
        best_validation_nrmse=best_metric,
        train_history_path=train_history_path,
        validation_history_path=validation_history_path,
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[DoubleHestonForwardPINN, dict[str, Any]]:
    """Restore a V1 network and checkpoint metadata without touching test data."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = baseline_config_from_mapping(payload.get("config", {}))
    model = DoubleHestonForwardPINN(
        feature_min=config.domain.feature_min,
        feature_max=config.domain.feature_max,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def load_checkpoint_config(checkpoint_path: str | Path) -> BaselineConfig:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return baseline_config_from_mapping(payload.get("config", {}))


__all__ = [
    "TrainingResult",
    "load_checkpoint_model",
    "load_checkpoint_config",
    "runtime_identity",
    "validate_checkpoint_identities",
    "seed_everything",
    "train_baseline",
    "validation_metrics",
]
