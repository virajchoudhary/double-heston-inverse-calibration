from __future__ import annotations

import pytest
import torch

from models.pinn_model import DoubleHestonConstraintMap, PhysicsInformedInverseCalibrator
from src.constraints import validate_parameters


def test_pinn_returns_valid_batch_by_ten() -> None:
    model = PhysicsInformedInverseCalibrator(
        input_size=108,
        hidden_sizes=(64, 64),
        dropout=0.0,
    )
    output = model(torch.zeros(4, 108))
    assert output.shape == (4, 10)
    for row in output.detach().cpu().numpy():
        assert validate_parameters(row)["is_valid"]


def test_constraint_map_respects_double_heston_contract() -> None:
    mapping = DoubleHestonConstraintMap()
    unconstrained = torch.randn(32, 10)
    constrained = mapping(unconstrained)
    for row in constrained.detach().cpu().numpy():
        diagnostics = validate_parameters(row)
        assert diagnostics["is_valid"], diagnostics["violations"]


@pytest.mark.parametrize("shape", [(108,), (4, 107), (4, 108, 1)])
def test_pinn_rejects_malformed_input(shape: tuple[int, ...]) -> None:
    model = PhysicsInformedInverseCalibrator(
        input_size=108,
        hidden_sizes=(32, 32),
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="Expected features"):
        model(torch.zeros(shape))
