from __future__ import annotations

import torch

from src.mentor_dh_pinn.collocation import (
    BoundaryPoints,
    sample_high_s_boundary_points,
    sample_low_s_boundary_points,
    sample_terminal_points,
)
from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.parameter_source import select_first_eligible_train_record
from src.mentor_dh_pinn.losses import (
    LossComponents,
    data_loss,
    high_s_boundary_loss,
    low_s_boundary_loss,
    terminal_loss,
    weighted_total_loss,
)


class PayoffModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.clamp(features[:, 0] - features[:, 4], min=0.0).unsqueeze(1)


class ZeroModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((features.shape[0], 1), dtype=torch.float64)


class HighBoundaryModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spot, tau, strike, rate, carry = (
            features[:, 0], features[:, 3], features[:, 4], features[:, 5], features[:, 6]
        )
        return (spot * torch.exp(-carry * tau) - strike * torch.exp(-rate * tau)).unsqueeze(1)


def test_data_loss_is_spot_normalized_and_weighted_total_is_explicit() -> None:
    predicted = torch.tensor([2.0, 4.0], dtype=torch.float64)
    reference = torch.tensor([1.0, 2.0], dtype=torch.float64)
    spot = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert torch.isclose(data_loss(predicted, reference, spot), torch.tensor(1.0, dtype=torch.float64))
    components = LossComponents(*(torch.tensor(float(i), dtype=torch.float64) for i in range(1, 6)))
    assert torch.isclose(weighted_total_loss(components), torch.tensor(15.0, dtype=torch.float64))
    weights = {"data": 1.0, "pde": 1.0, "boundary": 1.0, "terminal": 1.0}
    assert torch.isclose(weighted_total_loss(components, weights), components.total)


def test_boundary_class_keeps_low_and_high_targets_separate() -> None:
    fields = [torch.ones(3, dtype=torch.float64) for _ in range(7)]
    low = BoundaryPoints(*fields, target=torch.zeros(3, dtype=torch.float64), name="low_s")
    high = BoundaryPoints(
        *fields,
        target=torch.full((3,), 0.5, dtype=torch.float64),
        name="high_s",
    )
    assert low.name == "low_s"
    assert high.name == "high_s"
    assert not torch.equal(low.target, high.target)


def test_boundary_samplers_use_explicit_low_and_high_spots() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record(
        "data/final_r2_clean_10000/surfaces.jsonl"
    )
    low = sample_low_s_boundary_points(
        3, config=config, parameter_source=source, seed=3407
    )
    high = sample_high_s_boundary_points(
        3, config=config, parameter_source=source, seed=3407
    )
    assert torch.all(low.spot == config.domain.boundary_spot_low)
    assert torch.all(high.spot == config.domain.boundary_spot_high)
    low_loss, _, _ = low_s_boundary_loss(ZeroModel(), low)
    high_loss, _, _ = high_s_boundary_loss(HighBoundaryModel(), high)
    assert float(low_loss) == 0.0
    assert float(high_loss.detach()) < 1.0e-28


def test_terminal_exact_payoff_is_zero_and_wrong_prediction_is_positive() -> None:
    config = load_baseline_config().with_overrides(
        train_count=2, validation_count=2, test_count=2, max_epochs=1, patience=1
    )
    source = select_first_eligible_train_record("data/final_r2_clean_10000/surfaces.jsonl")
    points = sample_terminal_points(8, config=config, parameter_source=source, seed=3407)
    exact, _, _ = terminal_loss(PayoffModel(), points)
    wrong, _, _ = terminal_loss(ZeroModel(), points)
    assert float(exact.detach()) == 0.0
    assert float(wrong.detach()) > 0.0
