from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.model3_pde.collocation import (
    sample_collocation_states,
    sample_eligible_contract_slot_indices,
)
from src.model3_pde.losses import (
    arbitrage_boundary_loss,
    pde_residual_loss,
    terminal_payoff_loss,
)
from src.model3_pde.model import Model3PDESystem
from src.model3_pde.collocation import (
    CollocationDomain,
    sample_conditioned_collocation_states,
)
from src.model3_pde.operator import PDEState, double_heston_pde_residual


REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_parameters(point_count: int = 4) -> torch.Tensor:
    vector = torch.tensor(
        [0.80, 0.060, 0.250, -0.40, 0.050, 3.00, 0.030, 0.350, 0.20, 0.020],
        dtype=torch.float64,
    )
    return vector.repeat(point_count, 1)


def make_state(point_count: int = 4) -> PDEState:
    return PDEState(
        spot=torch.linspace(80.0, 120.0, point_count, dtype=torch.float64, requires_grad=True),
        variance_slow=torch.linspace(0.02, 0.08, point_count, dtype=torch.float64, requires_grad=True),
        variance_fast=torch.linspace(0.01, 0.04, point_count, dtype=torch.float64, requires_grad=True),
        maturity=torch.linspace(10.0 / 365.0, 60.0 / 365.0, point_count, dtype=torch.float64, requires_grad=True),
    )


def rates(state: PDEState, value: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full_like(state.spot, value),
        torch.full_like(state.spot, value / 2.0),
    )


def test_quadratic_artificial_solution_matches_manual_pde_coefficients() -> None:
    state = make_state(3)
    risk_free, dividend = rates(state, 0.06)
    parameters = valid_parameters(3)
    kappa_s, theta_s, sigma_s, rho_s, _ = (parameters[:, i] for i in range(5))
    kappa_f, theta_f, sigma_f, rho_f, _ = (parameters[:, i] for i in range(5, 10))

    coefficients = {
        "spot": 0.37,
        "spot_squared": -0.11,
        "variance_slow": 0.83,
        "variance_slow_squared": -0.29,
        "variance_fast": -0.41,
        "variance_fast_squared": 0.23,
        "mixed_spot_slow": 0.57,
        "mixed_spot_fast": -0.67,
        "tau": 1.19,
    }
    prices = (
        coefficients["spot"] * state.spot
        + coefficients["spot_squared"] * state.spot.square()
        + coefficients["variance_slow"] * state.variance_slow
        + coefficients["variance_slow_squared"] * state.variance_slow.square()
        + coefficients["variance_fast"] * state.variance_fast
        + coefficients["variance_fast_squared"] * state.variance_fast.square()
        + coefficients["mixed_spot_slow"] * state.spot * state.variance_slow
        + coefficients["mixed_spot_fast"] * state.spot * state.variance_fast
        + coefficients["tau"] * state.maturity
    )
    v_s_derivative = (
        coefficients["variance_slow"]
        + 2 * coefficients["variance_slow_squared"] * state.variance_slow
        + coefficients["mixed_spot_slow"] * state.spot
    )
    v_f_derivative = (
        coefficients["variance_fast"]
        + 2 * coefficients["variance_fast_squared"] * state.variance_fast
        + coefficients["mixed_spot_fast"] * state.spot
    )
    spot_derivative = (
        coefficients["spot"]
        + 2 * coefficients["spot_squared"] * state.spot
        + coefficients["mixed_spot_slow"] * state.variance_slow
        + coefficients["mixed_spot_fast"] * state.variance_fast
    )
    expected = coefficients["tau"] - (
        (risk_free - dividend)
        * state.spot
        * spot_derivative
        + kappa_s * (theta_s - state.variance_slow) * v_s_derivative
        + kappa_f * (theta_f - state.variance_fast) * v_f_derivative
        + (state.variance_slow + state.variance_fast)
        * state.spot.square()
        * coefficients["spot_squared"]
        + rho_s
        * sigma_s
        * state.variance_slow
        * state.spot
        * coefficients["mixed_spot_slow"]
        + rho_f
        * sigma_f
        * state.variance_fast
        * state.spot
        * coefficients["mixed_spot_fast"]
        + sigma_s.square()
        * state.variance_slow
        * coefficients["variance_slow_squared"]
        + sigma_f.square()
        * state.variance_fast
        * coefficients["variance_fast_squared"]
    ) + risk_free * prices
    actual = double_heston_pde_residual(
        prices,
        state,
        parameters,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-12)


def test_affine_forward_solution_has_zero_pde_residual() -> None:
    state = make_state()
    risk_free, dividend = rates(state, 0.03)
    prices = (
        state.spot * torch.exp(-dividend * state.maturity)
        - 100.0 * torch.exp(-risk_free * state.maturity)
        + 0.0 * state.spot.square()
        + 0.0 * state.spot.square() * state.variance_slow
        + 0.0 * state.spot.square() * state.variance_fast
        + 0.0 * state.variance_slow.square()
        + 0.0 * state.variance_fast.square()
    )
    residual = double_heston_pde_residual(
        prices,
        state,
        valid_parameters(),
        risk_free_rate=risk_free,
        dividend_yield=dividend,
    )
    assert torch.isfinite(residual).all()
    assert float(residual.detach().abs().max()) <= 1.0e-12


def test_operator_rejects_silent_non_differentiable_state() -> None:
    state = make_state()
    detached_spot = state.spot.detach()
    detached_variance_slow = state.variance_slow.detach()
    detached_variance_fast = state.variance_fast.detach()
    detached_maturity = state.maturity.detach()
    with pytest.raises(ValueError, match="autograd leaves"):
        bad_state = PDEState(
            spot=detached_spot,
            variance_slow=detached_variance_slow,
            variance_fast=detached_variance_fast,
            maturity=detached_maturity,
        )
        risk_free, dividend = rates(bad_state, 0.03)
        prices = bad_state.spot.square()
        double_heston_pde_residual(
            prices,
            bad_state,
            valid_parameters(),
            risk_free_rate=risk_free,
            dividend_yield=dividend,
        )


def test_collocation_sampling_is_deterministic_and_in_domain() -> None:
    domain = CollocationDomain(
        spot_minimum=50.0,
        spot_maximum=150.0,
        variance_slow_minimum=0.01,
        variance_slow_maximum=0.15,
        variance_fast_minimum=0.005,
        variance_fast_maximum=0.10,
        maturity_minimum=7.0 / 365.0,
        maturity_maximum=180.0 / 365.0,
    )
    left, _ = sample_collocation_states(domain, point_count=8, seed=3407)
    right, _ = sample_collocation_states(domain, point_count=8, seed=3407)
    for field in ("spot", "variance_slow", "variance_fast", "maturity"):
        assert torch.equal(getattr(left, field), getattr(right, field))
    assert torch.all(left.spot > 50.0) and torch.all(left.spot < 150.0)
    assert left.spot.dtype == torch.float64


def test_terminal_boundary_and_scaled_pde_losses_have_expected_shapes() -> None:
    state = make_state()
    risk_free, dividend = rates(state, 0.03)
    strike = torch.full_like(state.spot, 100.0)
    is_call = torch.tensor([True, False, True, False], dtype=torch.bool)
    lower = torch.zeros_like(state.spot)
    upper = torch.where(is_call, state.spot, strike)
    bounded_prices = lower + (upper - lower) * 0.25
    boundary = arbitrage_boundary_loss(
        bounded_prices,
        state,
        strike,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        is_call=is_call,
    )
    assert float(boundary.detach()) == 0.0

    affine_prices = (
        state.spot * torch.exp(-dividend * state.maturity)
        - strike * torch.exp(-risk_free * state.maturity)
        + 0.0 * state.spot.square()
        + 0.0 * state.spot.square() * state.variance_slow
        + 0.0 * state.spot.square() * state.variance_fast
        + 0.0 * state.variance_slow.square()
        + 0.0 * state.variance_fast.square()
    )
    physics = pde_residual_loss(
        affine_prices,
        state,
        valid_parameters(),
        risk_free_rate=risk_free,
        dividend_yield=dividend,
    )
    assert float(physics.detach()) <= 1.0e-24

    zero_state = PDEState(
        spot=state.spot.detach().clone().requires_grad_(True),
        variance_slow=state.variance_slow.detach().clone().requires_grad_(True),
        variance_fast=state.variance_fast.detach().clone().requires_grad_(True),
        maturity=torch.zeros_like(state.maturity).requires_grad_(True),
    )
    payoff = torch.where(is_call, torch.clamp(zero_state.spot - strike, min=0), torch.clamp(strike - zero_state.spot, min=0))
    terminal_error = terminal_payoff_loss(payoff + 0.01, zero_state, strike, is_call=is_call)
    assert abs(float(terminal_error.detach()) - 0.0001) < 1.0e-14


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_model_system_smoke_has_finite_gradients_without_training() -> None:
    torch.manual_seed(42)
    system = Model3PDESystem()
    assert next(system.parameters()).dtype == torch.float64
    features = torch.randn((2, 100), dtype=torch.float64)
    parameters = system.predict_parameters(features)
    state = make_state(2)
    risk_free, dividend = rates(state, 0.03)
    strike = torch.full_like(state.spot, 100.0)
    is_call = torch.tensor([True, False], dtype=torch.bool)
    prices = system.predict_prices(
        state,
        strike=strike,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        is_call=is_call,
        parameters=parameters,
    )
    physics = pde_residual_loss(
        prices,
        state,
        parameters,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
    )
    physics.backward()
    pricing_grads = [parameter.grad for parameter in system.pricing.parameters()]
    inverse_grads = [parameter.grad for parameter in system.inverse.parameters()]
    assert all(grad is None or torch.isfinite(grad).all() for grad in pricing_grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in inverse_grads)
    assert any(grad is not None and torch.any(grad != 0) for grad in pricing_grads)


def test_forward_map_enforces_terminal_payoff_exactly() -> None:
    torch.manual_seed(42)
    system = Model3PDESystem()
    features = torch.randn((3, 100), dtype=torch.float64)
    parameters = system.predict_parameters(features)
    zero_state = PDEState(
        spot=torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64, requires_grad=True),
        variance_slow=torch.tensor([0.04, 0.05, 0.06], dtype=torch.float64, requires_grad=True),
        variance_fast=torch.tensor([0.02, 0.03, 0.04], dtype=torch.float64, requires_grad=True),
        maturity=torch.zeros(3, dtype=torch.float64, requires_grad=True),
    )
    strike = torch.full_like(zero_state.spot, 100.0)
    risk_free = torch.full_like(zero_state.spot, 0.03)
    dividend = torch.full_like(zero_state.spot, 0.01)
    is_call = torch.tensor([True, False, True], dtype=torch.bool)
    prices = system.pricing(
        zero_state,
        strike=strike,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        is_call=is_call,
        parameters=parameters,
    )
    expected = torch.where(
        is_call,
        torch.clamp(zero_state.spot - strike, min=0.0),
        torch.clamp(strike - zero_state.spot, min=0.0),
    )
    assert torch.allclose(prices.detach(), expected, atol=1.0e-14, rtol=0.0)


def test_forward_map_retains_upper_no_arbitrage_representability() -> None:
    torch.manual_seed(42)
    system = Model3PDESystem()
    with torch.no_grad():
        system.pricing.network[-1].weight.zero_()
        system.pricing.network[-1].bias.fill_(50.0)
    state = make_state(2)
    strike = torch.full_like(state.spot, 100.0)
    risk_free, dividend = rates(state, 0.03)
    is_call = torch.tensor([True, False], dtype=torch.bool)
    prices = system.pricing(
        state,
        strike=strike,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        is_call=is_call,
        parameters=valid_parameters(2),
    )
    discounted_spot = state.spot * torch.exp(-dividend * state.maturity)
    discounted_strike = strike * torch.exp(-risk_free * state.maturity)
    call_lower = torch.clamp(discounted_spot - discounted_strike, min=0.0)
    put_lower = torch.clamp(discounted_strike - discounted_spot, min=0.0)
    lower = torch.where(is_call, call_lower, put_lower)
    upper = torch.where(is_call, discounted_spot, discounted_strike)
    payoff = torch.where(
        is_call,
        torch.clamp(state.spot - strike, min=0.0),
        torch.clamp(strike - state.spot, min=0.0),
    )
    relative_maturity = state.maturity / (7.0 / 365.0)
    terminal_weight = torch.where(
        relative_maturity < 1.0,
        (1.0 - relative_maturity).clamp_min(0.0) ** 4,
        torch.zeros_like(relative_maturity),
    )
    expected = lower + (upper - lower) - (upper - payoff) * terminal_weight
    assert torch.allclose(prices.detach(), expected, atol=1.0e-12)
    interval_fraction = (prices.detach() - lower) / (upper - lower)
    assert bool(torch.all(interval_fraction > 0.99))


def test_conditioned_collocations_are_deterministic_surface_major_and_in_domain() -> None:
    spots = torch.tensor([100.0, 120.0], dtype=torch.float64)
    slow_theta = torch.tensor([0.06, 0.08], dtype=torch.float64)
    fast_theta = torch.tensor([0.03, 0.04], dtype=torch.float64)
    kwargs = dict(
        observed_spots=spots,
        theta_slow=slow_theta,
        theta_fast=fast_theta,
        variance_slow_ceiling=1.0,
        variance_fast_ceiling=1.5,
        points_per_surface=3,
        seed=3407,
        contract_masks=torch.ones((2, 6), dtype=torch.bool),
    )
    left, left_sources, left_slots = sample_conditioned_collocation_states(**kwargs)
    right, right_sources, right_slots = sample_conditioned_collocation_states(**kwargs)
    assert torch.equal(left_sources, right_sources)
    assert torch.equal(left_sources, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64))
    for field in ("spot", "variance_slow", "variance_fast", "maturity"):
        assert torch.equal(getattr(left, field), getattr(right, field))
    assert bool(torch.all(left.variance_slow >= 0.05 * slow_theta.repeat_interleave(3)))
    assert bool(torch.all(left.variance_fast <= 2.0 * fast_theta.repeat_interleave(3)))
    assert torch.equal(left_slots, right_slots)
    assert bool(torch.all(left_slots >= 0)) and bool(torch.all(left_slots < 6))


def test_contract_slots_use_only_each_surfaces_eligible_canonical_slots() -> None:
    masks = torch.tensor(
        [
            [True, False, True, False, True, False],
            [False, True, False, True, False, False],
        ],
        dtype=torch.bool,
    )
    slots = sample_eligible_contract_slot_indices(
        masks, points_per_surface=64, seed=3407
    ).reshape(2, 64)
    assert torch.equal(torch.unique(slots[0]), torch.tensor([0, 2, 4]))
    assert torch.equal(torch.unique(slots[1]), torch.tensor([1, 3]))


def test_pretraining_protocol_freeze_matches_selected_architecture() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/model3_pde_protocol.yaml").read_text(encoding="utf-8"))
    assert config["protocol"]["status"] == "PRE_TRAINING_DESIGN_FREEZE"
    assert config["protocol"]["base_main_git_sha"] == (
        "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
    )
    assert config["model3_definition"]["architecture"]["kind"] == (
        "CONDITIONAL_FORWARD_PRICING_PINN_PLUS_INVERSE_ENCODER"
    )
    assert config["data"]["frozen_dataset_sha256"] == (
        "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
    )
    assert config["losses"]["scaled_double_heston_pde_residual_mse"] == 0.10
    assert config["protocol"]["version"] == "1.1"
    assert config["collocation"]["contract_slot_policy"]["eligible_slots"] == (
        "observed_unmasked_canonical_r2_slots"
    )
    assert config["collocation"]["contract_slot_policy"]["indexing"] == (
        "contract[surface_index, canonical_slot_index]"
    )
    assert config["losses"]["terminal_payoff_mse"] == 0.0
    assert config["losses"]["hard_no_arbitrage_boundary_penalty"] == 0.0
    assert config["collocation"]["terminal_blend"] == (
        "c2_polynomial_below_seven_days_zero_at_or_above_support"
    )
    assert config["losses"]["boundary_policy"] == (
        "c2_terminal_blend_below_support_full_bounded_base_at_or_above_support"
    )
    tau_domain = config["collocation"]["tau_domain_years"]
    assert all(isinstance(value, float) for value in tau_domain)
    assert tau_domain[0] == pytest.approx(7.0 / 365.0)
    assert tau_domain[1] == pytest.approx(180.0 / 365.0)
    assert isinstance(config["collocation"]["terminal_support_cutoff_years"], float)
    assert config["collocation"]["terminal_support_cutoff_years"] == pytest.approx(
        7.0 / 365.0
    )
    assert config["collocation"]["deterministic_generator_seed"] == 3407
    assert config["research_run_design"]["seeds"] == [11, 22, 33]
    research = config["research_run_design"]
    assert research["max_epochs"] == 120
    assert research["early_stopping_patience"] == 15
    assert research["checkpoint_rule"] == "minimum_validation_total_loss_only"
    assert research["optimizer"] == "adamw"
    assert research["learning_rate"] == pytest.approx(0.0002)
    assert research["weight_decay"] == pytest.approx(0.00001)
    assert research["batch_size"] == 32
    assert config["anti_leakage"]["real_market_weight_update"] == "forbidden"


def test_float32_r2_boundary_is_explicitly_upcast_to_physics_float64() -> None:
    torch.manual_seed(42)
    system = Model3PDESystem()
    assert next(system.parameters()).dtype == torch.float64
    features = torch.randn((2, 100), dtype=torch.float32)
    parameters = system.predict_parameters(features)
    assert parameters.dtype == torch.float64


def test_operator_rejects_non_leaf_state_tensor() -> None:
    state = make_state()
    non_leaf_spot = state.spot.square()
    with pytest.raises(ValueError, match="autograd leaves"):
        PDEState(
            spot=non_leaf_spot,
            variance_slow=state.variance_slow,
            variance_fast=state.variance_fast,
            maturity=state.maturity,
        )


def test_canonical_torch_mirror_satisfies_pde_operator_consistency() -> None:
    from src.torch_double_heston import price_double_heston_surface_batch_vectorized

    state = make_state(1)
    parameters = valid_parameters(1)
    parameters[:, 4] = state.variance_slow
    parameters[:, 9] = state.variance_fast
    risk_free, dividend = rates(state, 0.03)
    prices = price_double_heston_surface_batch_vectorized(
        parameters,
        state.spot,
        torch.full_like(state.spot, 100.0).unsqueeze(1),
        state.maturity.unsqueeze(1),
        risk_free,
        dividend,
        [["call"]],
        node_count=16,
    ).reshape(-1)
    residual = double_heston_pde_residual(
        prices,
        state,
        parameters,
        risk_free_rate=risk_free,
        dividend_yield=dividend,
    )
    scale = prices.detach().abs().clamp_min(1.0)
    assert torch.isfinite(residual).all()
    assert float((residual / scale).detach().abs().max()) <= 1.0e-4
