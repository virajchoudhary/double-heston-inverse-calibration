"""Thin Stage-A driver for the frozen genuine-PDE Model 3 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from models.parameter_transform import TargetStandardizer
from src.model3_pde.collocation import sample_conditioned_collocation_states
from src.model3_pde.losses import masked_normalized_price_loss, pde_residual_loss
from src.model3_pde.model import Model3PDESystem
from src.r2_primary.dataset import R2PrimaryDataset
from src.utils import set_deterministic_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_CONFIG_PATH = REPO_ROOT / "configs" / "model3_pde_protocol.yaml"
FROZEN_DATASET_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
ALLOWED_SPLITS = frozenset({"train", "validation"})
FORBIDDEN_SPLIT = "test"
LOSS_WEIGHTS = {
    "parameter": 1.0,
    "reconstruction": 1.0,
    "pde_residual": 0.10,
    "terminal_diagnostic": 0.0,
    "boundary_penalty": 0.0,
}
HISTORY_FIELDS = (
    "epoch",
    "parameter_loss",
    "reconstruction_loss",
    "pde_residual_loss",
    "total_loss",
    "finite_gradients",
    "gradient_norm",
    "pde_residual_rms",
    "pde_residual_max_scaled_rms",
    "terminal_payoff_max_abs",
    "duration_seconds",
    "accelerator_memory_allocated_bytes",
    "accelerator_memory_reserved_bytes",
)
VALIDATION_FIELDS = (
    "epoch",
    "validation_parameter_loss",
    "validation_reconstruction_loss",
    "validation_pde_residual_loss",
    "validation_total_loss",
)
PHYSICS_FIELDS = (
    "epoch",
    "split",
    "batch_index",
    "surface_count",
    "collocation_point_count",
    "residual_mean",
    "residual_max_abs",
    "terminal_payoff_max_abs",
)
GRADIENT_FIELDS = (
    "epoch",
    "batch_index",
    "finite_gradients",
    "gradient_norm",
)
REQUIRED_ARTIFACTS = (
    "checkpoint.pt",
    "optimizer.pt",
    "epoch_metadata.json",
    "train_history.csv",
    "validation_history.csv",
    "physics_diagnostics.csv",
    "gradient_diagnostics.csv",
    "environment_provenance.json",
)


@dataclass(frozen=True)
class PilotSettings:
    """CLI-facing execution settings; scientific values remain protocol-frozen."""

    dataset: Path
    output_root: Path
    train_limit: int = 240
    validation_limit: int = 40
    seed: int = 4207
    epochs: int = 3
    batch_size: int = 16
    interior_points: int = 16
    terminal_points: int = 8
    learning_rate: float = 0.0002
    weight_decay: float = 0.00001
    device: str = "cpu"
    smoke_mode: bool = False

    def __post_init__(self) -> None:
        positive_integers = {
            "train_limit": self.train_limit,
            "validation_limit": self.validation_limit,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "interior_points": self.interior_points,
            "terminal_points": self.terminal_points,
        }
        if any(value <= 0 for value in positive_integers.values()):
            raise ValueError("counts and limits must be strictly positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and strictly positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")


@dataclass(frozen=True)
class RunIdentity:
    git_sha: str
    config_sha256: str
    dataset_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "protocol_name": "MODEL3_GENUINE_PDE_DOUBLE_HESTON",
            "protocol_version": "1.0",
            "allowed_splits": sorted(ALLOWED_SPLITS),
            "forbidden_split": FORBIDDEN_SPLIT,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def current_git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git identity is required for a Model 3 checkpoint") from error


def current_git_dirty_state() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git dirty-state declaration is required") from error
    lines = completed.stdout.splitlines()
    return {"git_dirty": bool(lines), "git_status_tracked": lines}


def build_run_identity(dataset_path: Path) -> RunIdentity:
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_sha256 != FROZEN_DATASET_SHA256:
        raise RuntimeError(f"frozen R2 dataset identity mismatch: {dataset_sha256}")
    return RunIdentity(
        git_sha=current_git_sha(),
        config_sha256=sha256_file(PROTOCOL_CONFIG_PATH),
        dataset_sha256=dataset_sha256,
    )


def load_pilot_dataset(
    dataset_path: Path, *, train_limit: int, validation_limit: int
) -> tuple[R2PrimaryDataset, list[int], list[int]]:
    """Load only the synthetic train/validation rows in stored file order."""
    dataset = R2PrimaryDataset.from_jsonl(dataset_path, splits=ALLOWED_SPLITS)
    if dataset.indices_for_split(FORBIDDEN_SPLIT):
        raise RuntimeError("test-split records reached the Stage-A loader")
    invalid_splits = {
        item.surface_id for item in dataset.items if item.split not in ALLOWED_SPLITS
    }
    if invalid_splits:
        raise RuntimeError(
            "non-pilot split reached the Stage-A loader: "
            + ", ".join(sorted(invalid_splits)[:5])
        )
    train_indices = dataset.indices_for_split("train")[:train_limit]
    validation_indices = dataset.indices_for_split("validation")[:validation_limit]
    if len(train_indices) != train_limit or len(validation_indices) != validation_limit:
        raise ValueError(
            "the frozen Stage-A subset requires all requested train and validation surfaces"
        )
    return dataset, train_indices, validation_indices


def fit_train_only_standardizer(
    dataset: R2PrimaryDataset, train_indices: list[int]
) -> TargetStandardizer:
    return TargetStandardizer().fit(dataset.targets[train_indices])


def build_system() -> Model3PDESystem:
    system = Model3PDESystem()
    system.train()
    return system


def build_optimizer(
    system: Model3PDESystem, *, learning_rate: float, weight_decay: float
) -> AdamW:
    return AdamW(system.parameters(), lr=learning_rate, weight_decay=weight_decay)


def make_batches(
    indices: list[int], batch_size: int, generator: torch.Generator
) -> list[list[int]]:
    order = torch.randperm(len(indices), generator=generator).tolist()
    return [
        [indices[position] for position in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def subset_signature(
    dataset: R2PrimaryDataset, train_indices: list[int], validation_indices: list[int]
) -> dict[str, Any]:
    def signature(indices: list[int]) -> dict[str, Any]:
        return {
            "count": len(indices),
            "surface_ids": [dataset.items[index].surface_id for index in indices],
            "parameter_vector_hashes": [
                dataset.items[index].parameter_vector_hash for index in indices
            ],
        }

    return {"train": signature(train_indices), "validation": signature(validation_indices)}


def checkpoint_metadata(
    settings: PilotSettings,
    identity: RunIdentity,
    subset_signature_value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_kind": (
            "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT"
            if settings.smoke_mode
            else "MODEL3_STAGE_A_DEVELOPMENT_PILOT_NOT_RESEARCH_RESULT"
        ),
        "settings": {**asdict(settings), "dataset": settings.dataset.as_posix()},
        "loss_weights": LOSS_WEIGHTS,
        "subset_signature": subset_signature_value,
        **identity.payload(),
    }


def validate_resume_identity(
    stored: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key in ("run_kind", "settings", "loss_weights", "subset_signature"):
        if key not in stored or stored[key] != expected[key]:
            raise RuntimeError(f"resume identity mismatch for {key}")
    identity_keys = (
        "git_sha",
        "config_sha256",
        "dataset_sha256",
        "protocol_name",
        "protocol_version",
        "allowed_splits",
        "forbidden_split",
    )
    mismatches = [key for key in identity_keys if stored.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(f"resume identity mismatch for {', '.join(mismatches)}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(frame.to_dict(orient="records"))
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_history(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    missing = set(fields) - set(frame.columns)
    if missing:
        raise RuntimeError(f"invalid history schema in {path}: missing {sorted(missing)}")
    return frame.loc[:, list(fields)].to_dict(orient="records")


def validate_history_consistency(
    histories: dict[str, list[dict[str, Any]]], *, next_epoch: int
) -> None:
    for name, rows in histories.items():
        epochs = [int(row["epoch"]) for row in rows]
        if any(epoch >= next_epoch for epoch in epochs):
            raise RuntimeError(f"{name} contains epochs at or beyond the resume point")
        if epochs and epochs != list(range(1, next_epoch)):
            raise RuntimeError(f"{name} is not a contiguous completed-epoch history")


def environment_provenance(settings: PilotSettings, identity: RunIdentity) -> dict[str, Any]:
    payload = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device_selected": settings.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "float64_physics_boundary": True,
        "real_market_inputs_used": False,
        "issue34_numeric_outcomes_used": False,
        **identity.payload(),
    }
    payload.update(current_git_dirty_state())
    return payload


def _terminal_state(
    *,
    spots: torch.Tensor,
    variance_slow: torch.Tensor,
    variance_fast: torch.Tensor,
    repeats: int,
    seed: int,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draw_slow = torch.rand((spots.numel(), repeats), generator=generator, dtype=torch.float64)
    draw_fast = torch.rand((spots.numel(), repeats), generator=generator, dtype=torch.float64)
    slow = (variance_slow.unsqueeze(1) * draw_slow).reshape(-1).to(device)
    fast = (variance_fast.unsqueeze(1) * draw_fast).reshape(-1).to(device)
    source_indices = torch.arange(spots.numel(), dtype=torch.int64).repeat_interleave(repeats)
    from src.model3_pde.operator import PDEState

    state = PDEState(
        spot=spots.repeat_interleave(repeats).to(device=device, dtype=torch.float64).requires_grad_(True),
        variance_slow=slow.requires_grad_(True),
        variance_fast=fast.requires_grad_(True),
        maturity=torch.zeros_like(slow).requires_grad_(True),
    )
    return state, source_indices, torch.zeros_like(slow)


def evaluate_batch(
    system: Model3PDESystem,
    dataset: Any,
    indices: list[int],
    standardizer: TargetStandardizer,
    settings: PilotSettings,
    *,
    batch_index: int = 0,
    epoch: int,
    optimizer: AdamW | None = None,
) -> dict[str, Any]:
    """Evaluate one batch with the exact frozen three-term objective."""
    resolved_device = torch.device(settings.device)
    items = [dataset.items[index] for index in indices]
    features = torch.stack([dataset.features[index] for index in indices]).to(resolved_device)
    targets = torch.stack([dataset.targets[index] for index in indices]).to(resolved_device)
    parameters = system.predict_parameters(features)
    standardized_targets = standardizer.transform(targets).to(dtype=torch.float64)
    parameter_loss = torch.mean(
        (standardizer.transform(parameters).to(dtype=torch.float64) - standardized_targets).square()
    )

    spots = torch.tensor([item.spot for item in items], dtype=torch.float64, device=resolved_device)
    observed_strikes = torch.as_tensor(
        np.stack([np.asarray(item.strikes, dtype=np.float64) for item in items]),
        dtype=torch.float64,
        device=resolved_device,
    )
    observed_maturities = torch.as_tensor(
        np.stack([np.asarray(item.maturities, dtype=np.float64) for item in items]),
        dtype=torch.float64,
        device=resolved_device,
    )
    observed_rates = torch.tensor([item.rate for item in items], dtype=torch.float64, device=resolved_device)
    observed_carries = torch.tensor([item.carry for item in items], dtype=torch.float64, device=resolved_device)
    observed_is_call = torch.as_tensor(
        [[slot == "call" for slot in item.option_types] for item in items],
        dtype=torch.bool,
        device=resolved_device,
    )
    from src.model3_pde.operator import PDEState

    observed_state = PDEState(
        spot=spots.repeat_interleave(observed_maturities.shape[1])
        .detach()
        .clone()
        .requires_grad_(True),
        variance_slow=parameters[:, 4]
        .detach()
        .clone()
        .repeat_interleave(observed_maturities.shape[1])
        .requires_grad_(True),
        variance_fast=parameters[:, 9]
        .detach()
        .clone()
        .repeat_interleave(observed_maturities.shape[1])
        .requires_grad_(True),
        maturity=observed_maturities.reshape(-1).detach().clone().requires_grad_(True),
    )
    observed_prices = system.predict_prices(
        observed_state,
        strike=observed_strikes.reshape(-1),
        risk_free_rate=observed_rates.repeat_interleave(observed_maturities.shape[1]),
        dividend_yield=observed_carries.repeat_interleave(observed_maturities.shape[1]),
        is_call=observed_is_call.reshape(-1),
        parameters=parameters.repeat_interleave(observed_maturities.shape[1], dim=0),
    ).reshape(len(items), -1)
    observed_normalized = torch.as_tensor(
        np.stack([np.asarray(item.normalized_prices, dtype=np.float64) for item in items]),
        dtype=torch.float64,
        device=resolved_device,
    )
    observed_mask = torch.stack([dataset.masks[index] for index in indices]).to(resolved_device)
    reconstruction_loss = masked_normalized_price_loss(
        observed_prices / spots.unsqueeze(1), observed_normalized, observed_mask
    )

    theta_slow = parameters[:, 1].detach()
    theta_fast = parameters[:, 6].detach()
    state, source_indices = sample_conditioned_collocation_states(
        observed_spots=spots,
        theta_slow=theta_slow,
        theta_fast=theta_fast,
        variance_slow_ceiling=0.30,
        variance_fast_ceiling=0.25,
        points_per_surface=settings.interior_points,
        seed=settings.seed + 100003 * epoch + batch_index,
    )
    state = PDEState(
        spot=state.spot.to(device=resolved_device),
        variance_slow=state.variance_slow.to(device=resolved_device),
        variance_fast=state.variance_fast.to(device=resolved_device),
        maturity=state.maturity.to(device=resolved_device),
    )
    physics_rates = observed_rates[source_indices]
    physics_carries = observed_carries[source_indices]
    physics_strikes = observed_strikes.reshape(-1)[source_indices]
    physics_is_call = observed_is_call.reshape(-1)[source_indices]
    physics_prices = system.predict_prices(
        state,
        strike=physics_strikes,
        risk_free_rate=physics_rates,
        dividend_yield=physics_carries,
        is_call=physics_is_call,
        parameters=parameters[source_indices],
    )
    physics_loss = pde_residual_loss(
        physics_prices,
        state,
        parameters[source_indices],
        risk_free_rate=physics_rates,
        dividend_yield=physics_carries,
    )
    residual_values = physics_loss.detach().sqrt().cpu().numpy()

    terminal_state, terminal_sources, _ = _terminal_state(
        spots=spots,
        variance_slow=parameters[:, 4].detach(),
        variance_fast=parameters[:, 9].detach(),
        repeats=settings.terminal_points,
        seed=settings.seed + 257 * epoch + 7919 * batch_index,
        device=resolved_device,
    )
    terminal_prices = system.predict_prices(
        terminal_state,
        strike=observed_strikes.reshape(-1)[terminal_sources],
        risk_free_rate=observed_rates[terminal_sources],
        dividend_yield=observed_carries[terminal_sources],
        is_call=observed_is_call.reshape(-1)[terminal_sources],
        parameters=parameters[terminal_sources],
    )
    terminal_payoff = torch.where(
        observed_is_call.reshape(-1)[terminal_sources],
        torch.clamp(terminal_state.spot - observed_strikes.reshape(-1)[terminal_sources], min=0.0),
        torch.clamp(observed_strikes.reshape(-1)[terminal_sources] - terminal_state.spot, min=0.0),
    )
    terminal_error = float(
        (terminal_prices.detach() - terminal_payoff.detach()).abs().max()
    )

    total_loss = (
        LOSS_WEIGHTS["parameter"] * parameter_loss
        + LOSS_WEIGHTS["reconstruction"] * reconstruction_loss
        + LOSS_WEIGHTS["pde_residual"] * physics_loss
    )
    finite_gradients = True
    gradient_norm = float("nan")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradients = [
            parameter.grad for parameter in system.parameters() if parameter.grad is not None
        ]
        finite_gradients = bool(
            gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        )
        if not finite_gradients or not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("non-finite Model 3 pilot loss or gradient")
        gradient_norm = float(
            torch.linalg.vector_norm(torch.cat([gradient.reshape(-1) for gradient in gradients]))
        )
        optimizer.step()

    accelerator_allocated = 0
    accelerator_reserved = 0
    if resolved_device.type == "cuda":
        accelerator_allocated = int(torch.cuda.memory_allocated(resolved_device))
        accelerator_reserved = int(torch.cuda.memory_reserved(resolved_device))
    return {
        "parameter_loss": float(parameter_loss.detach()),
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "pde_residual_loss": float(physics_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "finite_gradients": finite_gradients,
        "gradient_norm": gradient_norm,
        "pde_residual_rms": float(np.mean(residual_values)),
        "pde_residual_max_scaled_rms": float(np.max(residual_values)),
        "terminal_payoff_max_abs": terminal_error,
        "surface_count": len(items),
        "collocation_point_count": int(state.spot.numel()),
        "accelerator_memory_allocated_bytes": accelerator_allocated,
        "accelerator_memory_reserved_bytes": accelerator_reserved,
    }


def run_validation_epoch(
    system: Model3PDESystem,
    dataset: Any,
    indices: list[int],
    standardizer: TargetStandardizer,
    settings: PilotSettings,
    epoch: int,
) -> dict[str, float]:
    was_training = system.training
    system.eval()
    totals = {"parameter": 0.0, "reconstruction": 0.0, "physics": 0.0, "total": 0.0}
    generator = torch.Generator().manual_seed(settings.seed + 9176 + epoch)
    try:
        for batch_index, batch_indices in enumerate(
            make_batches(indices, settings.batch_size, generator)
        ):
            with torch.enable_grad():
                metrics = evaluate_batch(
                    system,
                    dataset,
                    batch_indices,
                    standardizer,
                    settings,
                    batch_index=batch_index,
                    epoch=epoch,
                    optimizer=None,
                )
            surfaces = len(batch_indices)
            totals["parameter"] += metrics["parameter_loss"] * surfaces
            totals["reconstruction"] += metrics["reconstruction_loss"] * surfaces
            totals["physics"] += metrics["pde_residual_loss"] * surfaces
            totals["total"] += metrics["total_loss"] * surfaces
    finally:
        system.train(was_training)
    count = len(indices)
    return {
        "validation_parameter_loss": totals["parameter"] / count,
        "validation_reconstruction_loss": totals["reconstruction"] / count,
        "validation_pde_residual_loss": totals["physics"] / count,
        "validation_total_loss": totals["total"] / count,
    }


def run_pilot(settings: PilotSettings) -> dict[str, Any]:
    """Run or resume Stage A; never access the untouched test split."""
    set_deterministic_seed(settings.seed)
    identity = build_run_identity(settings.dataset)
    dataset, train_indices, validation_indices = load_pilot_dataset(
        settings.dataset,
        train_limit=settings.train_limit,
        validation_limit=settings.validation_limit,
    )
    subsets = subset_signature(dataset, train_indices, validation_indices)
    expected_metadata = checkpoint_metadata(settings, identity, subsets)
    output_root = settings.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "environment_provenance.json", environment_provenance(settings, identity))

    system = build_system().to(torch.device(settings.device))
    standardizer = fit_train_only_standardizer(dataset, train_indices)
    optimizer = build_optimizer(
        system, learning_rate=settings.learning_rate, weight_decay=settings.weight_decay
    )
    checkpoint_path = output_root / "checkpoint.pt"
    optimizer_path = output_root / "optimizer.pt"
    metadata_path = output_root / "epoch_metadata.json"
    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch = 0
    if checkpoint_path.exists():
        stored_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_resume_identity(stored_checkpoint["metadata"], expected_metadata)
        stored_optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        system.load_state_dict(stored_checkpoint["model_state_dict"])
        standardizer.mean = stored_checkpoint["target_standardizer"]["mean"]
        standardizer.scale = stored_checkpoint["target_standardizer"]["scale"]
        optimizer.load_state_dict(stored_optimizer["optimizer_state_dict"])
        start_epoch = int(stored_checkpoint["completed_epoch"]) + 1
        best_validation_loss = float(stored_checkpoint["best_validation_loss"])
        best_epoch = int(stored_checkpoint["best_epoch"])
    elif any(path.name != "environment_provenance.json" and path.exists() for path in output_root.iterdir()):
        raise RuntimeError("refusing to resume incomplete output without checkpoint.pt")

    train_history: list[dict[str, Any]] = read_history(
        output_root / "train_history.csv", HISTORY_FIELDS
    )
    validation_history: list[dict[str, Any]] = read_history(
        output_root / "validation_history.csv", VALIDATION_FIELDS
    )
    physics_history: list[dict[str, Any]] = read_history(
        output_root / "physics_diagnostics.csv", PHYSICS_FIELDS
    )
    gradient_history: list[dict[str, Any]] = read_history(
        output_root / "gradient_diagnostics.csv", GRADIENT_FIELDS
    )
    validate_history_consistency(
        {
            "train_history": train_history,
            "validation_history": validation_history,
            "physics_diagnostics": physics_history,
            "gradient_diagnostics": gradient_history,
        },
        next_epoch=start_epoch,
    )

    for epoch in range(start_epoch, settings.epochs + 1):
        started = time.perf_counter()
        train_generator = torch.Generator().manual_seed(settings.seed + epoch)
        batch_metrics: list[dict[str, Any]] = []
        for batch_index, batch_indices in enumerate(
            make_batches(train_indices, settings.batch_size, train_generator)
        ):
            metrics = evaluate_batch(
                system,
                dataset,
                batch_indices,
                standardizer,
                settings,
                batch_index=batch_index,
                epoch=epoch,
                optimizer=optimizer,
            )
            metrics.update({"epoch": epoch, "batch_index": batch_index})
            batch_metrics.append(metrics)
        validation_metrics = run_validation_epoch(
            system, dataset, validation_indices, standardizer, settings, epoch
        )
        duration = time.perf_counter() - started
        aggregate = {
            key: float(
                np.average(
                    [metrics[key] for metrics in batch_metrics],
                    weights=[metrics["surface_count"] for metrics in batch_metrics],
                )
            )
            for key in (
                "parameter_loss",
                "reconstruction_loss",
                "pde_residual_loss",
                "total_loss",
                "gradient_norm",
                "pde_residual_rms",
                "pde_residual_max_scaled_rms",
                "terminal_payoff_max_abs",
                "accelerator_memory_allocated_bytes",
                "accelerator_memory_reserved_bytes",
            )
        }
        train_row = {
            "epoch": epoch,
            **aggregate,
            "finite_gradients": all(
                metrics["finite_gradients"] for metrics in batch_metrics
            ),
            "duration_seconds": duration,
        }
        validation_row = {"epoch": epoch, **validation_metrics}
        train_history.append(train_row)
        validation_history.append(validation_row)
        physics_history.extend(
            {
                "epoch": epoch,
                "split": "train",
                "batch_index": metrics["batch_index"],
                "surface_count": metrics["surface_count"],
                "collocation_point_count": metrics["collocation_point_count"],
                "residual_mean": metrics["pde_residual_rms"],
                "residual_max_abs": metrics["pde_residual_max_scaled_rms"],
                "terminal_payoff_max_abs": metrics["terminal_payoff_max_abs"],
            }
            for metrics in batch_metrics
        )
        gradient_history.extend(
            {
                "epoch": epoch,
                "batch_index": metrics["batch_index"],
                "finite_gradients": metrics["finite_gradients"],
                "gradient_norm": metrics["gradient_norm"],
            }
            for metrics in batch_metrics
        )

        rng_states: dict[str, list[int]] = {"cpu": torch.get_rng_state().tolist()}
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state_all().tolist()
        completed_model = {key: value.detach().cpu() for key, value in system.state_dict().items()}
        completed_standardizer = standardizer.state_dict()
        completed_optimizer = optimizer.state_dict()
        checkpoint_payload = {
            "completed_epoch": epoch,
            "model_state_dict": completed_model,
            "target_standardizer": completed_standardizer,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "rng_states": rng_states,
            "metadata": expected_metadata,
        }
        optimizer_payload = {
            "completed_epoch": epoch,
            "optimizer_state_dict": completed_optimizer,
            "metadata": expected_metadata,
        }
        _atomic_torch_save(checkpoint_payload, checkpoint_path)
        _atomic_torch_save(optimizer_payload, optimizer_path)
        _atomic_json(metadata_path, {**expected_metadata, "completed_epoch": epoch})
        _atomic_csv(output_root / "train_history.csv", pd.DataFrame(train_history), HISTORY_FIELDS)
        _atomic_csv(
            output_root / "validation_history.csv",
            pd.DataFrame(validation_history),
            VALIDATION_FIELDS,
        )
        _atomic_csv(
            output_root / "physics_diagnostics.csv", pd.DataFrame(physics_history), PHYSICS_FIELDS
        )
        _atomic_csv(
            output_root / "gradient_diagnostics.csv",
            pd.DataFrame(gradient_history),
            GRADIENT_FIELDS,
        )
        if validation_metrics["validation_total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["validation_total_loss"]
            best_epoch = epoch
            checkpoint_payload["best_validation_loss"] = best_validation_loss
            checkpoint_payload["best_epoch"] = best_epoch
            _atomic_torch_save(checkpoint_payload, checkpoint_path)

    return {
        "status": "PASSED" if settings.smoke_mode else "STAGE_A_EPOCHS_COMPLETE",
        "run_kind": expected_metadata["run_kind"],
        "completed_epoch": settings.epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "output_root": output_root,
        "artifacts": list(REQUIRED_ARTIFACTS),
    }


def smoke_settings(dataset: Path, output_root: Path, device: str = "cpu") -> PilotSettings:
    return PilotSettings(
        dataset=dataset,
        output_root=output_root,
        train_limit=2,
        validation_limit=2,
        seed=4207,
        epochs=1,
        batch_size=2,
        interior_points=1,
        terminal_points=1,
        learning_rate=0.0002,
        weight_decay=0.00001,
        device=device,
        smoke_mode=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=240)
    parser.add_argument("--validation-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4207)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--interior-points", type=int, default=16)
    parser.add_argument("--terminal-points", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny wiring-only CPU smoke; never a Stage-A or scientific result",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    settings = PilotSettings(
        dataset=arguments.dataset.resolve(),
        output_root=arguments.output_root.resolve(),
        train_limit=arguments.train_limit,
        validation_limit=arguments.validation_limit,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        interior_points=arguments.interior_points,
        terminal_points=arguments.terminal_points,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        device=arguments.device,
        smoke_mode=arguments.smoke,
    )
    if settings.smoke_mode:
        settings = replace(
            settings,
            train_limit=2,
            validation_limit=2,
            epochs=1,
            batch_size=2,
            interior_points=1,
            terminal_points=1,
            device="cpu",
        )
    result = run_pilot(settings)
    print(json.dumps(result, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
