"""Train and evaluate the standalone inverse Black--Scholes PINN."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

try:
    from .market import dense_evaluation_grid, generate_synthetic_market, load_market_csv, market_to_normalized
    from .model import BlackScholesPINN, Domain, LossWeights
except ImportError:  # Allows: python black_scholes_pinn/train.py
    from market import dense_evaluation_grid, generate_synthetic_market, load_market_csv, market_to_normalized
    from model import BlackScholesPINN, Domain, LossWeights


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260827
    true_sigma: float = 0.20
    sigma_initial: float = 0.35
    rate: float = 0.03
    dividend: float = 0.01
    hidden_width: int = 96
    hidden_layers: int = 5
    collocation_points: int = 8192
    adam_batch_size: int = 1024
    warmup_steps: int = 2500
    adam_steps: int = 5000
    adam_learning_rate: float = 1.0e-3
    sigma_learning_rate: float = 5.0e-3
    lbfgs_max_iterations: int = 1400
    noise_fraction: float = 0.0

    def validate(self) -> None:
        positive = {
            "true_sigma": self.true_sigma,
            "sigma_initial": self.sigma_initial,
            "hidden_width": self.hidden_width,
            "hidden_layers": self.hidden_layers,
            "collocation_points": self.collocation_points,
            "adam_batch_size": self.adam_batch_size,
            "warmup_steps": self.warmup_steps,
            "adam_steps": self.adam_steps,
            "adam_learning_rate": self.adam_learning_rate,
            "sigma_learning_rate": self.sigma_learning_rate,
        }
        if min(positive.values()) <= 0:
            raise ValueError("all training sizes, rates, and volatility values must be positive")
        if self.lbfgs_max_iterations < 0 or self.noise_fraction < 0.0:
            raise ValueError("lbfgs iterations and noise must be non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sobol_points(
    count: int,
    domain: Domain,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    unit = engine.draw(count).to(device=device, dtype=dtype)
    x = domain.x_min + (domain.x_max - domain.x_min) * unit[:, :1]
    # Avoid tau=0 at PDE points because the payoff has a kink there.
    tau_floor = 1.0e-4
    tau = tau_floor + (domain.tau_max - tau_floor) * unit[:, 1:2]
    return x.requires_grad_(requires_grad), tau.requires_grad_(requires_grad)


def tensor_column(values: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype).reshape(-1, 1)


def compute_losses(
    model: BlackScholesPINN,
    collocation_x: torch.Tensor,
    collocation_tau: torch.Tensor,
    market_x: torch.Tensor,
    market_tau: torch.Tensor,
    market_price: torch.Tensor,
    terminal_x: torch.Tensor,
    boundary_tau: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    residual = model.pde_residual(collocation_x, collocation_tau)
    pde_loss = residual.square().mean()
    market_loss = F.mse_loss(model(market_x, market_tau), market_price)
    terminal_loss, boundary_loss = model.condition_errors(terminal_x, boundary_tau)
    total = (
        weights.pde * pde_loss
        + weights.market * market_loss
        + weights.boundary * boundary_loss
        + weights.terminal * terminal_loss
    )
    return {
        "total": total,
        "pde": pde_loss,
        "market": market_loss,
        "boundary": boundary_loss,
        "terminal": terminal_loss,
    }


def detached_row(step: int, phase: str, model: BlackScholesPINN, losses: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "phase": phase,
        "step": step,
        "total_loss": float(losses["total"].detach()),
        "pde_loss": float(losses["pde"].detach()),
        "market_loss": float(losses["market"].detach()),
        "boundary_loss": float(losses["boundary"].detach()),
        "terminal_loss": float(losses["terminal"].detach()),
        "calibrated_sigma": float(model.sigma.detach()),
    }


def train(
    config: TrainConfig,
    *,
    output_directory: str | Path,
    market_csv: str | Path | None = None,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Train without ever providing true_sigma to the PINN loss or model."""
    config.validate()
    weights = LossWeights()
    weights.validate()
    domain = Domain()
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    dtype = torch.float64

    if market_csv is None:
        market = generate_synthetic_market(
            true_sigma=config.true_sigma,
            rate=config.rate,
            dividend=config.dividend,
            seed=config.seed,
            noise_fraction=config.noise_fraction,
        )
        evaluation_sigma: float | None = config.true_sigma
        market_source = "synthetic_analytical_quotes"
    else:
        market = load_market_csv(market_csv)
        evaluation_sigma = None
        market_source = str(Path(market_csv).resolve())

    rate = float(market["rate"].iloc[0])
    dividend = float(market["dividend"].iloc[0])
    x_values, tau_values, normalized_prices = market_to_normalized(market)
    if x_values.min() < domain.x_min or x_values.max() > domain.x_max:
        raise ValueError("market log-moneyness lies outside the configured PINN domain")
    if tau_values.max() > domain.tau_max:
        raise ValueError("market maturity lies outside the configured PINN domain")

    # Notice that evaluation_sigma/true_sigma is intentionally absent here.
    model = BlackScholesPINN(
        domain=domain,
        rate=rate,
        dividend=dividend,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        sigma_initial=config.sigma_initial,
    ).to(device=device, dtype=dtype)

    market_x = tensor_column(x_values, device, dtype)
    market_tau = tensor_column(tau_values, device, dtype)
    market_price = tensor_column(normalized_prices, device, dtype)
    full_x, full_tau = sobol_points(
        config.collocation_points,
        domain,
        seed=config.seed + 1,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    terminal_x = torch.linspace(
        domain.x_min, domain.x_max, 257, device=device, dtype=dtype
    ).reshape(-1, 1)
    boundary_tau = torch.linspace(
        0.0, domain.tau_max, 257, device=device, dtype=dtype
    ).reshape(-1, 1)

    network_parameters = [parameter for name, parameter in model.named_parameters() if name != "raw_sigma"]
    optimizer = torch.optim.Adam(
        [
            {"params": network_parameters, "lr": config.adam_learning_rate},
            {"params": [model.raw_sigma], "lr": config.sigma_learning_rate},
        ]
    )
    history: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    started = time.perf_counter()

    # Stage 1: learn a smooth price field before exposing the inverse parameter
    # to PDE gradients. This prevents the well-known premature sigma collapse.
    model.train()
    for step in range(1, config.warmup_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        market_loss = F.mse_loss(model(market_x, market_tau), market_price)
        terminal_loss, boundary_loss = model.condition_errors(terminal_x, boundary_tau)
        warmup_total = (
            weights.market * market_loss
            + weights.boundary * boundary_loss
            + weights.terminal * terminal_loss
        )
        warmup_total.backward()
        torch.nn.utils.clip_grad_norm_(network_parameters, max_norm=10.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.warmup_steps:
            warmup_losses = {
                "total": warmup_total,
                "pde": warmup_total.new_tensor(float("nan")),
                "market": market_loss,
                "boundary": boundary_loss,
                "terminal": terminal_loss,
            }
            row = detached_row(step, "price_warmup", model, warmup_losses)
            history.append(row)
            if step == 1 or step % 250 == 0 or step == config.warmup_steps:
                print(
                    f"Warmup {step:5d}/{config.warmup_steps}: "
                    f"loss={row['total_loss']:.3e}, sigma(frozen)={row['calibrated_sigma']:.6f}"
                )

    # Obtain a physics-only starting value from the learned field derivatives.
    # This contains no true volatility or analytical prices.
    initialization_x = full_x.detach().clone().requires_grad_(True)
    initialization_tau = full_tau.detach().clone().requires_grad_(True)
    pde_sigma_initial = estimate_sigma_from_price_field(model, initialization_x, initialization_tau)
    set_model_sigma(model, pde_sigma_initial)
    print(f"PDE derivative initialization: sigma={float(model.sigma.detach()):.6f}")

    # Stage 2: jointly optimize the field and inverse volatility with the PDE.
    for step in range(1, config.adam_steps + 1):
        indices = torch.randint(
            config.collocation_points,
            (min(config.adam_batch_size, config.collocation_points),),
            generator=generator,
        ).to(device)
        batch_x = full_x[indices].detach().clone().requires_grad_(True)
        batch_tau = full_tau[indices].detach().clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(
            model,
            batch_x,
            batch_tau,
            market_x,
            market_tau,
            market_price,
            terminal_x,
            boundary_tau,
            weights,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.adam_steps:
            row = detached_row(step, "adam", model, losses)
            history.append(row)
            if step == 1 or step % 250 == 0 or step == config.adam_steps:
                print(
                    f"Adam {step:5d}/{config.adam_steps}: "
                    f"loss={row['total_loss']:.3e}, sigma={row['calibrated_sigma']:.6f}"
                )

    if config.lbfgs_max_iterations:
        lbfgs_x = full_x.detach().clone().requires_grad_(True)
        lbfgs_tau = full_tau.detach().clone().requires_grad_(True)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=config.lbfgs_max_iterations,
            max_eval=int(config.lbfgs_max_iterations * 1.25),
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            lbfgs.zero_grad(set_to_none=True)
            current = compute_losses(
                model,
                lbfgs_x,
                lbfgs_tau,
                market_x,
                market_tau,
                market_price,
                terminal_x,
                boundary_tau,
                weights,
            )
            current["total"].backward()
            evaluations += 1
            if evaluations == 1 or evaluations % 25 == 0:
                history.append(detached_row(evaluations, "lbfgs", model, current))
            return current["total"]

        lbfgs.step(closure)
        final_train_x = full_x.detach().clone().requires_grad_(True)
        final_train_tau = full_tau.detach().clone().requires_grad_(True)
        final_losses = compute_losses(
            model,
            final_train_x,
            final_train_tau,
            market_x,
            market_tau,
            market_price,
            terminal_x,
            boundary_tau,
            weights,
        )
        history.append(detached_row(evaluations, "lbfgs_final", model, final_losses))
        print(
            f"L-BFGS evaluations={evaluations}: loss={float(final_losses['total'].detach()):.3e}, "
            f"sigma={float(model.sigma.detach()):.6f}"
        )

    joint_sigma = float(model.sigma.detach())
    finalizer_x, finalizer_tau = sobol_points(
        65536,
        domain,
        seed=config.seed + 701,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    robust_sigma, finalizer_points = estimate_sigma_robust_from_price_field(
        model, finalizer_x, finalizer_tau
    )
    set_model_sigma(model, robust_sigma)
    print(
        f"Robust PDE finalization: sigma={robust_sigma:.8f} "
        f"from {finalizer_points} stable points"
    )

    elapsed = time.perf_counter() - started
    model.eval()
    artifacts = evaluate_and_save(
        model=model,
        market=market,
        evaluation_sigma=evaluation_sigma,
        output_path=output_path,
        config=config,
        weights=weights,
        history=history,
        elapsed_seconds=elapsed,
        market_source=market_source,
        device=device,
        dtype=dtype,
        joint_sigma_before_finalization=joint_sigma,
        finalizer_points=finalizer_points,
    )
    return {"model": model, **artifacts}


def evaluate_and_save(
    *,
    model: BlackScholesPINN,
    market: pd.DataFrame,
    evaluation_sigma: float | None,
    output_path: Path,
    config: TrainConfig,
    weights: LossWeights,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
    market_source: str,
    device: torch.device,
    dtype: torch.dtype,
    joint_sigma_before_finalization: float | None = None,
    finalizer_points: int | None = None,
) -> dict[str, Any]:
    market = market.copy()
    market_x, market_tau, _ = market_to_normalized(market)
    with torch.no_grad():
        normalized_prediction = model(
            tensor_column(market_x, device, dtype), tensor_column(market_tau, device, dtype)
        ).cpu().numpy().ravel()
    market["pinn_price"] = normalized_prediction * market["strike"].to_numpy()
    market["absolute_error"] = np.abs(market["pinn_price"] - market["call_price"])
    market.to_csv(output_path / "market_fit.csv", index=False)

    metrics: dict[str, Any] = {
        "status": "trained",
        "method": "inverse physics-informed neural network",
        "calibrated_sigma": float(model.sigma.detach().cpu()),
        "sigma_initial": config.sigma_initial,
        "market_price_mae": float(market["absolute_error"].mean()),
        "market_price_rmse": float(np.sqrt(np.mean(np.square(market["absolute_error"])))),
        "market_price_max_absolute_error": float(market["absolute_error"].max()),
        "training_seconds": elapsed_seconds,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "market_source": market_source,
        "market_quotes": len(market),
        "collocation_points": config.collocation_points,
        "analytical_price_used_in_training_loss": False,
        "volatility_supervision_used": False,
        "joint_optimizer_sigma_before_finalization": joint_sigma_before_finalization,
        "calibration_finalizer": "robust pointwise PDE inversion on stable central-domain points",
        "calibration_finalizer_points": finalizer_points,
        "terminal_condition_enforcement": "explicit PINN loss",
        "boundary_condition_enforcement": "explicit PINN loss",
    }

    if evaluation_sigma is not None:
        evaluation = dense_evaluation_grid(
            rate=model.rate, dividend=model.dividend, true_sigma=evaluation_sigma
        )
        eval_x = np.log(evaluation["spot"].to_numpy() / evaluation["strike"].to_numpy())
        eval_tau = evaluation["tau"].to_numpy()
        with torch.no_grad():
            eval_normalized = model(
                tensor_column(eval_x, device, dtype), tensor_column(eval_tau, device, dtype)
            ).cpu().numpy().ravel()
        evaluation["pinn_price"] = eval_normalized * evaluation["strike"].to_numpy()
        evaluation["absolute_error"] = np.abs(
            evaluation["pinn_price"] - evaluation["reference_price"]
        )
        evaluation["relative_error"] = evaluation["absolute_error"] / np.maximum(
            evaluation["reference_price"], 0.50
        )
        evaluation.to_csv(output_path / "dense_predictions.csv", index=False)
        sigma_error = abs(metrics["calibrated_sigma"] - evaluation_sigma)
        metrics.update(
            {
                "true_sigma_evaluation_only": evaluation_sigma,
                "sigma_absolute_error": sigma_error,
                "sigma_relative_error_percent": 100.0 * sigma_error / evaluation_sigma,
                "dense_price_mae": float(evaluation["absolute_error"].mean()),
                "dense_price_rmse": float(
                    np.sqrt(np.mean(np.square(evaluation["absolute_error"])))
                ),
                "dense_price_max_absolute_error": float(evaluation["absolute_error"].max()),
                "dense_price_mean_relative_error_percent": float(
                    100.0 * evaluation["relative_error"].mean()
                ),
            }
        )
        save_surface_plot(evaluation, output_path / "price_comparison.png")

    test_x, test_tau = sobol_points(
        4096,
        model.domain,
        seed=config.seed + 99,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    residual = model.pde_residual(test_x, test_tau).detach().cpu().numpy().ravel()
    metrics["pde_residual_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["pde_residual_max_absolute"] = float(np.max(np.abs(residual)))

    terminal_x = torch.linspace(
        model.domain.x_min, model.domain.x_max, 501, device=device, dtype=dtype
    ).reshape(-1, 1)
    boundary_tau = torch.linspace(
        0.0, model.domain.tau_max, 501, device=device, dtype=dtype
    ).reshape(-1, 1)
    terminal_loss, boundary_loss = model.condition_errors(terminal_x, boundary_tau)
    metrics["terminal_condition_rmse"] = math.sqrt(float(terminal_loss.detach()))
    metrics["boundary_condition_rmse"] = math.sqrt(float(boundary_loss.detach()))

    pd.DataFrame(history).to_csv(output_path / "training_history.csv", index=False)
    save_loss_plot(history, output_path / "training_history.png")
    save_calibration_plot(
        history, evaluation_sigma, output_path / "volatility_calibration.png"
    )
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run_metadata = {
        "config": asdict(config),
        "loss_weights": weights.to_dict(),
        "model": model.metadata(),
        "optimizer_sequence": ["Adam", "L-BFGS"],
        "pde": "c_tau - 0.5*sigma^2*c_xx - (r-q-0.5*sigma^2)*c_x + r*c = 0",
    }
    with (output_path / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": model.metadata(),
            "config": asdict(config),
            "metrics": metrics,
        },
        output_path / "black_scholes_pinn.pt",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return {"metrics": metrics, "output_directory": output_path}


def save_loss_plot(history: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 5))
    for column, label in (("total_loss", "total"), ("pde_loss", "PDE"), ("market_loss", "market")):
        axis.semilogy(np.arange(len(frame)), np.maximum(frame[column], 1.0e-18), label=label)
    axis.set(xlabel="logged optimization point", ylabel="loss", title="Black–Scholes inverse PINN training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def estimate_sigma_from_price_field(
    model: BlackScholesPINN,
    x: torch.Tensor,
    tau: torch.Tensor,
) -> float:
    """Least-squares PDE estimate using only neural price-field derivatives."""
    price = model(x, tau)
    c_tau = torch.autograd.grad(price, tau, torch.ones_like(price), create_graph=True, retain_graph=True)[0]
    c_x = torch.autograd.grad(price, x, torch.ones_like(price), create_graph=True, retain_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, torch.ones_like(c_x), create_graph=False, retain_graph=True)[0]
    a = c_tau - (model.rate - model.dividend) * c_x + model.rate * price
    b = 0.5 * (c_x - c_xx)
    sigma_squared = -torch.sum(a.detach() * b.detach()) / torch.sum(b.detach().square()).clamp_min(1.0e-14)
    estimate = math.sqrt(max(float(sigma_squared), model.sigma_min * model.sigma_min))
    return min(max(estimate, model.sigma_min + 1.0e-6), model.sigma_max - 1.0e-6)


def estimate_sigma_robust_from_price_field(
    model: BlackScholesPINN,
    x: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[float, int]:
    """Robustly invert the PDE where volatility is well identified.

    Pointwise inversion becomes unstable for tiny gamma, extreme moneyness, and
    the payoff kink near expiry. The fixed mask below excludes those regions on
    numerical-identifiability grounds; it contains no reference volatility.
    """
    price = model(x, tau)
    c_tau = torch.autograd.grad(
        price, tau, torch.ones_like(price), create_graph=True, retain_graph=True
    )[0]
    c_x = torch.autograd.grad(
        price, x, torch.ones_like(price), create_graph=True, retain_graph=True
    )[0]
    c_xx = torch.autograd.grad(
        c_x, x, torch.ones_like(c_x), create_graph=False, retain_graph=True
    )[0]
    curvature = c_xx - c_x
    numerator = 2.0 * (
        c_tau - (model.rate - model.dividend) * c_x + model.rate * price
    )
    local_variance = numerator / curvature.clamp_min(1.0e-14)
    mask = (
        (x.abs() <= 0.25)
        & (tau >= 0.50)
        & (curvature >= 0.02)
        & (local_variance >= model.sigma_min**2)
        & (local_variance <= model.sigma_max**2)
    )
    stable_variances = local_variance.detach()[mask]
    if stable_variances.numel() < 100:
        raise RuntimeError("too few stable PDE points for robust volatility finalization")
    estimate = float(torch.sqrt(torch.median(stable_variances)))
    return estimate, int(stable_variances.numel())


def set_model_sigma(model: BlackScholesPINN, sigma: float) -> None:
    unit = (sigma - model.sigma_min) / (model.sigma_max - model.sigma_min)
    unit = min(max(unit, 1.0e-9), 1.0 - 1.0e-9)
    raw = math.log(unit / (1.0 - unit))
    with torch.no_grad():
        model.raw_sigma.fill_(raw)


def save_calibration_plot(
    history: list[dict[str, Any]], evaluation_sigma: float | None, path: Path
) -> None:
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["calibrated_sigma"].to_numpy(), label="PINN calibrated volatility")
    if evaluation_sigma is not None:
        axis.axhline(
            evaluation_sigma,
            color="black",
            linestyle="--",
            label="reference (evaluation only)",
        )
    axis.set(xlabel="logged optimization point", ylabel="volatility", title="Inverse-parameter convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_surface_plot(evaluation: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    selected = evaluation.iloc[::3]
    scatter = axes[0].scatter(
        selected["strike"], selected["tau"], c=selected["pinn_price"], cmap="viridis", s=18
    )
    axes[0].set(xlabel="strike", ylabel="maturity", title="PINN call-price surface")
    figure.colorbar(scatter, ax=axes[0], label="call price")
    axes[1].scatter(
        evaluation["reference_price"], evaluation["pinn_price"], s=8, alpha=0.55
    )
    low = min(evaluation["reference_price"].min(), evaluation["pinn_price"].min())
    high = max(evaluation["reference_price"].max(), evaluation["pinn_price"].max())
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set(
        xlabel="analytical reference price",
        ylabel="PINN price",
        title="Out-of-sample price agreement",
    )
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "high_accuracy_run",
    )
    parser.add_argument("--market-csv", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--true-sigma", type=float, default=TrainConfig.true_sigma)
    parser.add_argument("--sigma-initial", type=float, default=TrainConfig.sigma_initial)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--adam-steps", type=int, default=TrainConfig.adam_steps)
    parser.add_argument("--lbfgs-iterations", type=int, default=TrainConfig.lbfgs_max_iterations)
    parser.add_argument("--collocation-points", type=int, default=TrainConfig.collocation_points)
    parser.add_argument("--noise-fraction", type=float, default=TrainConfig.noise_fraction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        seed=args.seed,
        true_sigma=args.true_sigma,
        sigma_initial=args.sigma_initial,
        warmup_steps=args.warmup_steps,
        adam_steps=args.adam_steps,
        lbfgs_max_iterations=args.lbfgs_iterations,
        collocation_points=args.collocation_points,
        noise_fraction=args.noise_fraction,
    )
    train(
        config,
        output_directory=args.output,
        market_csv=args.market_csv,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
