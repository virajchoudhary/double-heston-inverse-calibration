from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dheston.calibration.transforms import scale_parameters_to_unit_torch
from dheston.pricing.heston import FourierConfig, price_double_heston_torch


def _expand_batch_to_points(batch: dict[str, Any], parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mask = batch["mask"]
    max_points = mask.shape[1]
    expanded_params = parameters.unsqueeze(1).expand(-1, max_points, -1)[mask]
    spot = batch["spot"].unsqueeze(1).expand(-1, max_points)[mask]
    strike = batch["strike"][mask]
    tau = batch["tau"][mask]
    rate = batch["rate"].unsqueeze(1).expand(-1, max_points)[mask]
    dividend = batch["dividend"].unsqueeze(1).expand(-1, max_points)[mask]
    is_call = batch["is_call"][mask]
    market_price = batch["market_price"][mask]
    return expanded_params, spot, strike, tau, rate, dividend, is_call, market_price


def predict_surface_prices(parameters: torch.Tensor, batch: dict[str, Any], pricing_config: FourierConfig) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_params, spot, strike, tau, rate, dividend, is_call, market_price = _expand_batch_to_points(batch, parameters)
    predicted = price_double_heston_torch(spot, strike, tau, rate, dividend, is_call, expanded_params, pricing_config)
    return predicted, market_price


def parameter_supervision_loss(predicted: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return predicted.new_tensor(0.0)
    predicted_unit = scale_parameters_to_unit_torch(predicted)
    target_unit = scale_parameters_to_unit_torch(target)
    return F.mse_loss(predicted_unit, target_unit)


def price_reconstruction_loss(predicted_prices: torch.Tensor, market_prices: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(predicted_prices / spot.clamp_min(1.0), market_prices / spot.clamp_min(1.0))


def ordering_penalty(parameters: torch.Tensor) -> torch.Tensor:
    return torch.relu(parameters[:, 6] - parameters[:, 1]).pow(2).mean()


def boundary_penalty(predicted_prices: torch.Tensor, batch: dict[str, Any], parameters: torch.Tensor) -> torch.Tensor:
    expanded_params, spot, strike, tau, rate, dividend, is_call, _ = _expand_batch_to_points(batch, parameters)
    del expanded_params
    discounted_spot = spot * torch.exp(-dividend * tau)
    discounted_strike = strike * torch.exp(-rate * tau)
    intrinsic_call = torch.clamp(discounted_spot - discounted_strike, min=0.0)
    intrinsic_put = torch.clamp(discounted_strike - discounted_spot, min=0.0)
    lower = torch.where(is_call > 0.5, intrinsic_call, intrinsic_put)
    upper = torch.where(is_call > 0.5, discounted_spot, discounted_strike)
    below = torch.relu(lower - predicted_prices)
    above = torch.relu(predicted_prices - upper)
    return (below.pow(2) + above.pow(2)).mean()


def _safe_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if not outputs.requires_grad or not inputs.requires_grad:
        return torch.zeros_like(inputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradients is None:
        return torch.zeros_like(inputs)
    return gradients


def pde_residual_loss(
    parameters: torch.Tensor,
    batch: dict[str, Any],
    pricing_config: FourierConfig,
    max_points: int,
) -> torch.Tensor:
    if max_points <= 0:
        return parameters.new_tensor(0.0)
    indices = batch["mask"].nonzero(as_tuple=False)
    if indices.numel() == 0:
        return parameters.new_tensor(0.0)
    if indices.shape[0] > max_points:
        step = max(1, indices.shape[0] // max_points)
        indices = indices[::step][:max_points]

    surface_index = indices[:, 0]
    point_index = indices[:, 1]

    chosen_params = parameters[surface_index]
    spot = batch["spot"][surface_index].detach().clone().requires_grad_(True)
    strike = batch["strike"][surface_index, point_index]
    tau = batch["tau"][surface_index, point_index].detach().clone().requires_grad_(True)
    rate = batch["rate"][surface_index]
    dividend = batch["dividend"][surface_index]
    is_call = batch["is_call"][surface_index, point_index]

    prices = price_double_heston_torch(spot, strike, tau, rate, dividend, is_call, chosen_params, pricing_config)

    d_tau = _safe_grad(prices, tau)
    delta = _safe_grad(prices, spot)
    gamma = _safe_grad(delta, spot)

    v01 = chosen_params[:, 0]
    kappa1 = chosen_params[:, 1]
    theta1 = chosen_params[:, 2]
    sigma1 = chosen_params[:, 3]
    rho1 = chosen_params[:, 4]
    v02 = chosen_params[:, 5]
    kappa2 = chosen_params[:, 6]
    theta2 = chosen_params[:, 7]
    sigma2 = chosen_params[:, 8]
    rho2 = chosen_params[:, 9]

    d_v01 = _safe_grad(prices, v01)
    d_v02 = _safe_grad(prices, v02)
    cross_sv01 = _safe_grad(delta, v01)
    cross_sv02 = _safe_grad(delta, v02)
    d2_v01 = _safe_grad(d_v01, v01)
    d2_v02 = _safe_grad(d_v02, v02)

    diffusion = 0.5 * (v01 + v02) * spot.square() * gamma
    drift = (rate - dividend) * spot * delta - rate * prices
    factor_one = kappa1 * (theta1 - v01) * d_v01 + rho1 * sigma1 * v01 * spot * cross_sv01 + 0.5 * sigma1.square() * v01 * d2_v01
    factor_two = kappa2 * (theta2 - v02) * d_v02 + rho2 * sigma2 * v02 * spot * cross_sv02 + 0.5 * sigma2.square() * v02 * d2_v02
    residual = d_tau - (diffusion + drift + factor_one + factor_two)
    scale = prices.detach().abs().clamp_min(1.0)
    return torch.mean((residual / scale).pow(2))


def build_loss_components(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    pricing_config: FourierConfig,
    loss_weights: dict[str, float],
    pde_points: int,
) -> dict[str, torch.Tensor]:
    parameters = model_output["params"]
    predicted_prices, market_prices = predict_surface_prices(parameters, batch, pricing_config)
    expanded_spot = batch["spot"].unsqueeze(1).expand_as(batch["market_price"])[batch["mask"]]

    param_loss = parameter_supervision_loss(parameters, batch.get("target_params"))
    price_loss = price_reconstruction_loss(predicted_prices, market_prices, expanded_spot)
    order_loss = ordering_penalty(parameters)
    bounds_loss = boundary_penalty(predicted_prices, batch, parameters)
    if loss_weights.get("lambda_pde", 0.0) > 0:
        pde_loss = pde_residual_loss(parameters, batch, pricing_config, max_points=pde_points)
    else:
        pde_loss = parameters.new_tensor(0.0)

    total = (
        loss_weights.get("lambda_param", 0.0) * param_loss
        + loss_weights.get("lambda_price", 0.0) * price_loss
        + loss_weights.get("lambda_order", 0.0) * order_loss
        + loss_weights.get("lambda_boundary", 0.0) * bounds_loss
        + loss_weights.get("lambda_pde", 0.0) * pde_loss
    )
    return {
        "total": total,
        "parameter": param_loss,
        "price": price_loss,
        "order": order_loss,
        "boundary": bounds_loss,
        "pde": pde_loss,
    }
