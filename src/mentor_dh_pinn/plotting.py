"""Required grayscale figures for the mentor Double Heston PINN baseline."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.double_heston import price_double_heston_call

from .collocation import sample_pde_points
from .config import BaselineConfig
from .losses import pde_loss
from .parameter_source import ParameterSource
from .synthetic_data import SyntheticDataset
from .trainer import load_checkpoint_model, validate_checkpoint_identities

FIGURE_NAMES = (
    "01_option_price_vs_strike.png",
    "02_option_price_vs_maturity.png",
    "03_absolute_pricing_error_vs_strike.png",
    "04_training_losses.png",
    "05_validation_price_error.png",
    "06_pde_residual_diagnostics.png",
)


def _read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                key: float(value) if key != "finite_gradients" else float(value == "True")
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _features(
    spot: float,
    variance_slow: float,
    variance_fast: float,
    tau: np.ndarray,
    strike: np.ndarray,
    rate: float,
    carry: float,
) -> torch.Tensor:
    count = len(tau)
    return torch.tensor(
        np.column_stack(
            (
                np.full(count, spot),
                np.full(count, variance_slow),
                np.full(count, variance_fast),
                tau,
                strike,
                np.full(count, rate),
                np.full(count, carry),
            )
        ),
        dtype=torch.float64,
    )


def _classical_prices(
    source: ParameterSource,
    *,
    spot: float,
    variance_slow: float,
    variance_fast: float,
    tau: np.ndarray,
    strike: np.ndarray,
    rate: float,
    carry: float,
    node_count: int,
) -> np.ndarray:
    parameters = source.parameters_for_state(variance_slow, variance_fast)
    return np.asarray(
        [
            price_double_heston_call(
                spot,
                float(k),
                float(t),
                rate,
                carry,
                parameters,
                node_count=node_count,
            )
            for k, t in zip(strike, tau, strict=True)
        ],
        dtype=np.float64,
    )


def _predict(model: torch.nn.Module, features: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(features).reshape(-1).detach().cpu().numpy()


def make_figures(
    checkpoint_path: str | Path,
    dataset: SyntheticDataset,
    output_dir: str | Path,
    *,
    config: BaselineConfig,
) -> list[Path]:
    """Render the six predeclared 300-dpi grayscale figures.

    The sparse black markers are held-out deterministic figure-grid reference
    samples.  The dashed curves are dense evaluations from the same canonical
    Double Heston pricer; neither grid is used for training or checkpoint
    selection.
    """
    root = Path(output_dir)
    figure_dir = root / config.evaluation.figures_subdirectory
    figure_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint_model(checkpoint_path)
    validate_checkpoint_identities(checkpoint, dataset, config)
    source = dataset.parameter_source
    train_rows = _read_csv(root / "train_history.csv")
    validation_rows = _read_csv(root / "validation_history.csv")
    spot = config.evaluation.slice_spot
    variance_slow = float(source.vector[1])
    variance_fast = float(source.vector[6])
    rate = config.evaluation.slice_rate
    carry = config.evaluation.slice_carry
    maturities = np.asarray(config.evaluation.slice_maturity_days, dtype=np.float64) / 365.0
    strike_min = spot * config.domain.moneyness_min
    strike_max = spot * config.domain.moneyness_max
    dense_strikes = np.linspace(strike_min, strike_max, config.evaluation.dense_grid_count)
    marker_strikes = np.linspace(strike_min, strike_max, config.evaluation.marker_strike_count)

    plt.style.use("grayscale")
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axis, tau_value in zip(axes, maturities, strict=True):
        dense_tau = np.full_like(dense_strikes, tau_value)
        marker_tau = np.full_like(marker_strikes, tau_value)
        classical = _classical_prices(
            source,
            spot=spot,
            variance_slow=variance_slow,
            variance_fast=variance_fast,
            tau=dense_tau,
            strike=dense_strikes,
            rate=rate,
            carry=carry,
            node_count=config.pricing_node_count,
        )
        reference = _classical_prices(
            source,
            spot=spot,
            variance_slow=variance_slow,
            variance_fast=variance_fast,
            tau=marker_tau,
            strike=marker_strikes,
            rate=rate,
            carry=carry,
            node_count=config.pricing_node_count,
        )
        pinn = _predict(
            model,
            _features(
                spot,
                variance_slow,
                variance_fast,
                dense_tau,
                dense_strikes,
                rate,
                carry,
            ),
        )
        axis.scatter(marker_strikes, reference, color="black", marker="o", s=18, label="held-out reference")
        axis.plot(dense_strikes, classical, color="0.35", linestyle="--", label="classical DH")
        axis.plot(dense_strikes, pinn, color="black", linestyle="-", label="PINN")
        axis.set_title(f"{round(tau_value * 365):d} days")
        axis.set_xlabel("strike K")
    axes[0].set_ylabel("call price")
    axes[-1].legend(frameon=False)
    fig.suptitle("Option Price vs Strike")
    path = figure_dir / FIGURE_NAMES[0]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    dense_tau = np.linspace(
        config.domain.tau_min, config.domain.tau_max, config.evaluation.dense_grid_count
    )
    marker_tau = np.asarray(config.evaluation.marker_maturity_days, dtype=np.float64) / 365.0
    dense_atm = np.full_like(dense_tau, spot)
    marker_atm = np.full_like(marker_tau, spot)
    classical_maturity = _classical_prices(
        source,
        spot=spot,
        variance_slow=variance_slow,
        variance_fast=variance_fast,
        tau=dense_tau,
        strike=dense_atm,
        rate=rate,
        carry=carry,
        node_count=config.pricing_node_count,
    )
    reference_maturity = _classical_prices(
        source,
        spot=spot,
        variance_slow=variance_slow,
        variance_fast=variance_fast,
        tau=marker_tau,
        strike=marker_atm,
        rate=rate,
        carry=carry,
        node_count=config.pricing_node_count,
    )
    pinn_maturity = _predict(
        model,
        _features(
            spot,
            variance_slow,
            variance_fast,
            dense_tau,
            dense_atm,
            rate,
            carry,
        ),
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(marker_tau * 365, reference_maturity, color="black", s=22, label="held-out reference")
    axis.plot(dense_tau * 365, classical_maturity, color="0.35", linestyle="--", label="classical DH")
    axis.plot(dense_tau * 365, pinn_maturity, color="black", linestyle="-", label="PINN")
    axis.set_xlabel("maturity (days)")
    axis.set_ylabel("ATM call price")
    axis.set_title("Option Price vs Maturity (K/S = 1)")
    axis.legend(frameon=False)
    path = figure_dir / FIGURE_NAMES[1]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    styles = ("-", "--", ":")
    for tau_value, linestyle in zip(maturities, styles, strict=True):
        dense_tau_slice = np.full_like(dense_strikes, tau_value)
        classical = _classical_prices(
            source,
            spot=spot,
            variance_slow=variance_slow,
            variance_fast=variance_fast,
            tau=dense_tau_slice,
            strike=dense_strikes,
            rate=rate,
            carry=carry,
            node_count=config.pricing_node_count,
        )
        pinn = _predict(
            model,
            _features(
                spot,
                variance_slow,
                variance_fast,
                dense_tau_slice,
                dense_strikes,
                rate,
                carry,
            ),
        )
        axis.plot(
            dense_strikes,
            np.abs(pinn - classical),
            color="black",
            linestyle=linestyle,
            label=f"{round(tau_value * 365):d} days",
        )
    axis.set_xlabel("strike K")
    axis.set_ylabel("|PINN - reference|")
    axis.set_title("Absolute Pricing Error vs Strike")
    axis.legend(frameon=False)
    path = figure_dir / FIGURE_NAMES[2]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    epochs = [row["epoch"] for row in train_rows]
    for key, label, linestyle in (
        ("train_pde_loss", "L_PDE", "-"),
        ("train_boundary_loss", "L_B", "--"),
        ("train_terminal_loss", "L_T", "-."),
        ("train_data_loss", "L_data", ":"),
    ):
        axis.plot(epochs, [max(row[key], np.finfo(float).tiny) for row in train_rows], color="black", linestyle=linestyle, label=label)
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss (log scale)")
    axis.set_title("Training Losses vs Epoch")
    axis.legend(frameon=False)
    path = figure_dir / FIGURE_NAMES[3]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    val_epochs = [row["epoch"] for row in validation_rows]
    for key, label, linestyle in (
        ("validation_price_rmse", "price RMSE", "-"),
        ("validation_price_mae", "price MAE", "--"),
        ("validation_nrmse", "normalized RMSE", ":"),
    ):
        axis.plot(val_epochs, [row[key] for row in validation_rows], color="black", linestyle=linestyle, label=label)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation error")
    axis.set_title("Validation Price Error vs Epoch")
    axis.legend(frameon=False)
    path = figure_dir / FIGURE_NAMES[4]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    pde_points = sample_pde_points(
        config.training.pde_batch_size,
        config=config,
        parameter_source=source,
        seed=config.seed + config.training.collocation_seed_offset + config.training.evaluation_seed_offset,
    )
    _, residual = pde_loss(model, pde_points, scale_floor=config.losses.pde_scale_floor)
    residual_values = residual.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].hist(residual_values, bins=32, color="0.35", edgecolor="black")
    axes[0].set_xlabel("raw PDE residual")
    axes[0].set_ylabel("count")
    axes[0].set_title("Residual distribution")
    axes[1].scatter(pde_points.spot.detach().cpu().numpy(), np.abs(residual_values), color="black", s=10)
    axes[1].set_xlabel("spot S")
    axes[1].set_ylabel("|PDE residual|")
    axes[1].set_title("Residual magnitude vs spot")
    fig.suptitle("PDE Residual Diagnostics")
    path = figure_dir / FIGURE_NAMES[5]
    _save(fig, path, dpi=config.evaluation.figure_dpi)
    paths.append(path)

    if len(paths) != 6 or any(not path.exists() for path in paths):
        raise AssertionError("V1 figure contract requires six written figures")
    return paths


__all__ = ["FIGURE_NAMES", "make_figures"]
