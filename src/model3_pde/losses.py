"""Loss primitives for the pre-training Model 3 foundation."""

from __future__ import annotations

import torch

from .operator import PDEState, double_heston_pde_residual


def pde_residual_loss(
    prices: torch.Tensor,
    state: PDEState,
    parameters: torch.Tensor,
    *,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
) -> torch.Tensor:
    """Mean squared PDE residual scaled by price magnitude."""
    residual = double_heston_pde_residual(
        prices,
        state,
        parameters,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    scale = prices.detach().abs().clamp_min(1.0)
    return torch.mean((residual / scale).square())


def discounted_arbitrage_bounds(
    state: PDEState,
    strike: torch.Tensor,
    *,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    is_call: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tight European no-arbitrage bounds in dollar-price units."""
    discounted_spot = state.spot * torch.exp(-dividend_yield * state.maturity)
    discounted_strike = strike * torch.exp(-risk_free_rate * state.maturity)
    call_lower = torch.clamp(discounted_spot - discounted_strike, min=0.0)
    put_lower = torch.clamp(discounted_strike - discounted_spot, min=0.0)
    lower = torch.where(is_call.bool(), call_lower, put_lower)
    upper = torch.where(is_call.bool(), discounted_spot, discounted_strike)
    return lower, upper


def terminal_payoff_loss(
    prices: torch.Tensor,
    state_at_zero: PDEState,
    strike: torch.Tensor,
    *,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Mean squared mismatch against undiscounted intrinsic payoff at tau=0."""
    if bool(torch.any(state_at_zero.maturity != 0)):
        raise ValueError("terminal loss requires maturity exactly zero")
    payoff = torch.where(
        is_call.bool(),
        torch.clamp(state_at_zero.spot - strike, min=0.0),
        torch.clamp(strike - state_at_zero.spot, min=0.0),
    )
    return torch.mean((prices - payoff).square())


def arbitrage_boundary_loss(
    prices: torch.Tensor,
    state: PDEState,
    strike: torch.Tensor,
    *,
    risk_free_rate: torch.Tensor,
    dividend_yield: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Squared violation of hard European no-arbitrage bounds."""
    lower, upper = discounted_arbitrage_bounds(
        state,
        strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
    )
    below = torch.relu(lower - prices)
    above = torch.relu(prices - upper)
    return torch.mean(below.square() + above.square())


def masked_normalized_price_loss(
    predicted_prices: torch.Tensor,
    observed_prices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE for prices already divided by surface spot."""
    if predicted_prices.shape != observed_prices.shape or predicted_prices.shape != mask.shape:
        raise ValueError("predicted prices, observed prices, and mask must have equal shape")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not bool(torch.any(mask)):
        raise ValueError("at least one observed slot is required")
    residuals = (predicted_prices[mask] - observed_prices[mask]).square()
    return torch.mean(residuals)
