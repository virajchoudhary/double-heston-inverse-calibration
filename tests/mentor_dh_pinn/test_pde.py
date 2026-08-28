from __future__ import annotations

import torch
from torch import nn

from src.mentor_dh_pinn.collocation import PDEPoints
from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.losses import pde_loss, pde_residual
from src.mentor_dh_pinn.parameter_source import select_first_eligible_train_record
from src.mentor_dh_pinn.collocation import sample_pde_points
from src.model3_pde.operator import PDEState, double_heston_pde_residual


class AffineForward(nn.Module):
    """A known zero-PDE call field for the tau-forward operator."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spot, _, _, tau, _, rate, carry = features.unbind(dim=1)
        return (spot * torch.exp(-carry * tau) - torch.exp(-rate * tau)).unsqueeze(1)


def test_pde_matches_canonical_tau_forward_affine_solution() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record(
        "data/final_r2_clean_10000/surfaces.jsonl"
    )
    points = sample_pde_points(4, config=config, parameter_source=source, seed=3407)
    residual = pde_residual(AffineForward(), points)
    assert torch.isfinite(residual).all()
    assert float(residual.detach().abs().max()) < 1.0e-11
    loss, _ = pde_loss(AffineForward(), points)
    assert float(loss.detach()) < 1.0e-20


def test_pde_point_class_rejects_non_float64_or_non_leaf_state() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record(
        "data/final_r2_clean_10000/surfaces.jsonl"
    )
    points = sample_pde_points(2, config=config, parameter_source=source, seed=3407)
    try:
        PDEPoints(
            points.spot.square(),
            points.variance_slow,
            points.variance_fast,
            points.tau,
            points.strike,
            points.rate,
            points.carry,
            points.parameters,
        )
    except ValueError as error:
        assert "differentiable leaves" in str(error)
    else:
        raise AssertionError("non-leaf state must be rejected")


def test_network_first_and_second_derivative_path_is_finite() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record("data/final_r2_clean_10000/surfaces.jsonl")
    points = sample_pde_points(3, config=config, parameter_source=source, seed=3408)
    from src.mentor_dh_pinn.model import DoubleHestonForwardPINN
    model = DoubleHestonForwardPINN(
        feature_min=config.domain.feature_min, feature_max=config.domain.feature_max
    )
    residual = pde_residual(model, points)
    assert torch.isfinite(residual).all()


def test_slow_fast_coefficients_are_distinct_and_no_variance_cross_term_is_added() -> None:
    spot = torch.tensor([1.1], dtype=torch.float64, requires_grad=True)
    slow = torch.tensor([0.07], dtype=torch.float64, requires_grad=True)
    fast = torch.tensor([0.03], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    state = PDEState(spot=spot, variance_slow=slow, variance_fast=fast, maturity=tau)
    parameters = torch.tensor(
        [[0.8, 0.12, 0.21, -0.6, 0.07, 3.4, 0.04, 0.49, 0.25, 0.03]],
        dtype=torch.float64,
    )
    price = (
        slow * fast
        + 0.0 * slow.square()
        + 0.0 * fast.square()
        + 0.0 * spot.square()
        + 0.0 * spot * slow
        + 0.0 * spot * fast
        + 0.0 * tau
    )
    rate = torch.tensor([0.05], dtype=torch.float64)
    carry = torch.tensor([0.01], dtype=torch.float64)
    actual = double_heston_pde_residual(
        price, state, parameters, risk_free_rate=rate, dividend_yield=carry
    )
    expected = -(
        parameters[:, 0] * (parameters[:, 1] - slow) * fast
        + parameters[:, 5] * (parameters[:, 6] - fast) * slow
    ) + rate * price
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-14)


def test_pde_sampler_preserves_float64_leaves_on_selected_device() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record("data/final_r2_clean_10000/surfaces.jsonl")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        points = sample_pde_points(
            2, config=config, parameter_source=source, seed=3410, device=device
        )
        for tensor in (*points.features.unbind(dim=1), points.parameters):
            assert tensor.dtype == torch.float64
            assert tensor.device.type == device
        assert all(
            field.is_leaf and field.requires_grad
            for field in (
                points.spot, points.variance_slow, points.variance_fast,
                points.tau, points.strike, points.rate, points.carry,
            )
        )
