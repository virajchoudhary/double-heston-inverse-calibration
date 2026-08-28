"""One-shot, test-isolated evaluation for a selected V1 checkpoint."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .collocation import (
    sample_high_s_boundary_points,
    sample_low_s_boundary_points,
    sample_pde_points,
    sample_terminal_points,
)
from .config import BaselineConfig
from .losses import high_s_boundary_loss, low_s_boundary_loss, pde_loss, terminal_loss
from .synthetic_data import SyntheticDataset
from .trainer import load_checkpoint_model, validate_checkpoint_identities


@dataclass(frozen=True)
class EvaluationResult:
    """Persisted one-shot evaluation result."""

    metrics: dict[str, Any]
    metrics_path: Path
    summary_path: Path


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.square())).item())


def _mae(values: torch.Tensor) -> float:
    return float(torch.mean(values.abs()).item())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_mentor_summary(
    output_dir: str | Path,
    *,
    metrics: dict[str, Any],
    filename: str = "MENTOR_SUMMARY.md",
    next_validation_lambda_study: str = (
        "Run a validation-only lambda study after mentor review; keep the test split sealed."
    ),
) -> Path:
    """Write a result summary from supplied run metrics only."""
    root = Path(output_dir)
    path = root / filename
    provenance = metrics.get("provenance", {})
    config = metrics.get("config", {})
    data_config = config.get("data", {})
    network_config = config.get("network", {})
    training_config = config.get("training", {})
    parameter_vector = provenance.get("parameter_vector", [])
    parameter_order = provenance.get("canonical_parameter_order", [])
    parameter_meanings = {
        "kappa_slow": "slow variance mean-reversion speed",
        "theta_slow": "slow variance long-run level",
        "sigma_slow": "slow variance volatility",
        "rho_slow": "spot/slow-variance correlation",
        "v0_slow": "selected source record's slow variance state",
        "kappa_fast": "fast variance mean-reversion speed",
        "theta_fast": "fast variance long-run level",
        "sigma_fast": "fast variance volatility",
        "rho_fast": "spot/fast-variance correlation",
        "v0_fast": "selected source record's fast variance state",
    }
    lines = [
        "# Mentor Double Heston PINN Baseline V1",
        "",
        "This summary is generated from the selected checkpoint and this run's persisted metrics.",
        "",
        "## Run identity",
        "",
        f"- Checkpoint: `{metrics.get('checkpoint_path', 'unknown')}`",
        f"- Dataset identity: `{metrics.get('dataset_sha256', 'unknown')}`",
        f"- Test split ID hash: `{metrics.get('test_split_id_hash', 'unknown')}`",
        f"- Parameter source surface: `{provenance.get('surface_id', 'unknown')}`",
        f"- Parameter identity: `{provenance.get('parameter_hash', 'unknown')}`",
        f"- Test rows evaluated once: `{metrics.get('test_count', 'unknown')}`",
        "",
        "## Inputs and output",
        "",
        "- Inputs: `S, v_slow, v_fast, tau, K, r, q`.",
        "- Output: raw European CALL price `C`.",
        "- For canonical reference pricing, sampled `v_slow` and `v_fast` replace `v0_slow` and `v0_fast`; the eight structural parameters remain fixed.",
        "",
        "## Ten Double Heston parameters",
        "",
        f"- Source: first eligible stored TRAIN record in the frozen dataset, surface `{provenance.get('surface_id', 'unknown')}`.",
        f"- Canonical vector: `{parameter_vector}`.",
    ]
    lines.extend(
        f"- `{name}` = `{value}`: {parameter_meanings.get(name, 'model parameter')}."
        for name, value in zip(parameter_order, parameter_vector, strict=False)
    )
    lines.extend(
        [
        "",
        "## Four explicit losses",
        "",
        "- `L_PDE`: penalizes violation of the Double Heston pricing PDE at interior collocation points.",
        "- `L_B`: the sum of low-stock and high-stock CALL boundary-condition MSEs.",
        "- `L_T`: enforces the explicit terminal payoff `max(S-K, 0)` at `tau=0`.",
        "- `L_data`: matches canonical synthetic prices through `mean(((C_hat-C_ref)/S)^2)`.",
        "- Lambdas: `lambda_PDE=lambda_B=lambda_T=lambda_data=1`. These neutral values are frozen and untuned.",
        "",
        "## Data, architecture, and training",
        "",
        f"- Train / validation / test counts: `{data_config.get('train_count', 'unknown')} / {data_config.get('validation_count', 'unknown')} / {data_config.get('test_count', 'unknown')}`.",
        f"- Architecture: 7 inputs, `{network_config.get('hidden_layers', 'unknown')}` hidden layers of width `{network_config.get('hidden_width', 'unknown')}`, tanh activation, one raw-price output; `{metrics.get('network_parameter_count', 'unknown')}` parameters.",
        f"- Optimizer / learning rate: `{training_config.get('optimizer', 'unknown')} / {training_config.get('learning_rate', 'unknown')}`.",
        f"- Epoch selected / epochs completed: `{metrics.get('checkpoint_epoch', 'unknown')} / {metrics.get('epochs_completed', 'unknown')}`.",
        "",
        "## Test metrics",
        "",
        f"- Price RMSE: `{metrics.get('price_rmse', 'unknown')}`",
        f"- Price MAE: `{metrics.get('price_mae', 'unknown')}`",
        f"- Normalized price RMSE: `{metrics.get('price_nrmse', 'unknown')}`",
        f"- Relative error mean / p95: `{metrics.get('relative_error_mean', 'unknown')}` / `{metrics.get('relative_error_p95', 'unknown')}`",
        f"- PDE RMS / max: `{metrics.get('pde_rms', 'unknown')}` / `{metrics.get('pde_max_abs', 'unknown')}`",
        f"- Terminal RMSE / max: `{metrics.get('terminal_rmse', 'unknown')}` / `{metrics.get('terminal_max_abs', 'unknown')}`",
        f"- Boundary low-S / high-S RMSE: `{metrics.get('boundary_low_s_rmse', 'unknown')}` / `{metrics.get('boundary_high_s_rmse', 'unknown')}`",
        f"- Inference runtime total / per contract (seconds): `{metrics.get('inference_seconds_total', 'unknown')}` / `{metrics.get('inference_seconds_per_contract', 'unknown')}`",
        "",
        "## WHAT PROVES",
        "",
        "- This run proves that the selected raw-price network, deterministic synthetic split, canonical CALL pricing source, and explicitly evaluated loss diagnostics produced the metrics above.",
        "- The checkpoint provenance records the frozen surfaces hash, selected TRAIN parameter identity, split hashes, seed, architecture, optimizer, and loss weights.",
        "",
        "## WHAT DOES NOT PROVE",
        "",
        "- This baseline does not prove inverse parameter identification, global Double Heston recovery, uniqueness, market performance, or readiness for a representation freeze.",
        "- A single sealed test evaluation does not authorize tuning on test metrics or establish a comparison against another model.",
        "",
        "## Next validation-only step",
        "",
        f"- {next_validation_lambda_study}",
        "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate_test_once(
    checkpoint_path: str | Path,
    dataset: SyntheticDataset,
    output_dir: str | Path,
    *,
    config: BaselineConfig,
) -> EvaluationResult:
    """Explicitly evaluate the sealed test split exactly once."""
    config.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / config.evaluation.metrics_filename
    claim_path = root / "test_evaluation_claim.json"
    if metrics_path.exists() or claim_path.exists():
        raise FileExistsError(
            f"test evaluation was already claimed in {root}; repeated test evaluation is disabled"
        )
    model, checkpoint = load_checkpoint_model(checkpoint_path)
    validate_checkpoint_identities(checkpoint, dataset, config)
    claim = {
        "schema_version": "mentor_dh_pinn_test_evaluation_claim_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_git_sha": checkpoint["git_sha"],
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "parameter_hash": dataset.parameter_source.parameter_hash,
        "test_split_id_hash": dataset.split_id_hashes()["test"],
    }
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(claim, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"test evaluation was already claimed at {claim_path}; repeated test evaluation is disabled"
        ) from error
    test_indices = dataset.indices("test")
    if len(test_indices) == 0:
        raise ValueError("test split must be non-empty")
    features = torch.from_numpy(np.asarray(dataset.features[test_indices], dtype=np.float64))
    references = torch.from_numpy(np.asarray(dataset.reference_prices[test_indices], dtype=np.float64))
    started = time.perf_counter()
    with torch.no_grad():
        predictions = model(features).reshape(-1)
    inference_seconds = time.perf_counter() - started
    errors = predictions - references
    normalized_errors = errors / features[:, 0]
    epsilon = float(config.losses.relative_error_epsilon)
    relative_errors = errors.abs() / references.abs().clamp_min(epsilon)

    # The following diagnostics intentionally occur only in this explicit
    # evaluator, never in trainer checkpoint selection.
    pde_points = sample_pde_points(
        config.training.pde_batch_size,
        config=config,
        parameter_source=dataset.parameter_source,
        seed=config.seed
        + config.training.collocation_seed_offset
        + config.training.evaluation_seed_offset,
    )
    _, pde_residual_values = pde_loss(
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
        + config.training.evaluation_seed_offset,
    )
    _, terminal_predictions, terminal_targets = terminal_loss(model, terminal_points)
    low_points = sample_low_s_boundary_points(
        config.training.boundary_batch_size,
        config=config,
        parameter_source=dataset.parameter_source,
        seed=config.seed
        + config.training.low_boundary_seed_offset
        + config.training.evaluation_seed_offset,
    )
    _, low_predictions, low_targets = low_s_boundary_loss(model, low_points)
    high_points = sample_high_s_boundary_points(
        config.training.boundary_batch_size,
        config=config,
        parameter_source=dataset.parameter_source,
        seed=config.seed
        + config.training.high_boundary_seed_offset
        + config.training.evaluation_seed_offset,
    )
    _, high_predictions, high_targets = high_s_boundary_loss(model, high_points)

    metrics: dict[str, Any] = {
        "schema_version": "mentor_dh_pinn_evaluation_v1",
        "test_evaluated_once": True,
        "test_evaluation_mode": config.evaluation.test_evaluation_mode,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_sha256": dataset.manifest.get("dataset_sha256"),
        "test_split_id_hash": dataset.split_id_hashes()["test"],
        "test_count": int(len(test_indices)),
        "relative_error_epsilon": epsilon,
        "relative_error_policy": "abs(pred-reference)/max(abs(reference), epsilon)",
        "price_rmse": _rmse(errors),
        "price_mae": _mae(errors),
        "price_nrmse": _rmse(normalized_errors),
        "relative_error_mean": float(torch.mean(relative_errors).item()),
        "relative_error_p95": float(torch.quantile(relative_errors, 0.95).item()),
        "pde_rms": _rmse(pde_residual_values.detach()),
        "pde_max_abs": float(pde_residual_values.detach().abs().max().item()),
        "terminal_rmse": _rmse((terminal_predictions - terminal_targets).detach()),
        "terminal_max_abs": float((terminal_predictions - terminal_targets).detach().abs().max().item()),
        "boundary_low_s_rmse": _rmse((low_predictions - low_targets).detach()),
        "boundary_high_s_rmse": _rmse((high_predictions - high_targets).detach()),
        "inference_seconds_total": float(inference_seconds),
        "inference_seconds_per_contract": float(inference_seconds / len(test_indices)),
        "provenance": dataset.parameter_source.provenance(),
        "network_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "config": checkpoint.get("config", config.to_dict()),
        "epochs_completed": checkpoint.get("epochs_completed", checkpoint.get("epoch")),
        "best_epoch": checkpoint.get("best_epoch"),
        "seed": config.seed,
        "loss_weights": config.losses.weights,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path = write_mentor_summary(
        root,
        metrics=metrics,
        filename=config.evaluation.summary_filename,
    )
    return EvaluationResult(metrics=metrics, metrics_path=metrics_path, summary_path=summary_path)


__all__ = ["EvaluationResult", "evaluate_test_once", "write_mentor_summary"]
