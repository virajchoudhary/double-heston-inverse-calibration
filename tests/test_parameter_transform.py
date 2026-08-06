from __future__ import annotations

import pytest
import torch

from ann_inverse_calibration.models.parameter_transform import (
    BoundedParameterTransform,
    TargetStandardizer,
)


def test_parameter_transform_stays_strictly_within_bounds() -> None:
    lower = torch.zeros(10)
    upper = torch.arange(1.0, 11.0)
    transform = BoundedParameterTransform(lower, upper)
    unconstrained = torch.linspace(-3.0, 3.0, 20).reshape(2, 10)
    bounded = transform(unconstrained)
    assert torch.all(bounded > lower)
    assert torch.all(bounded < upper)
    torch.testing.assert_close(transform.inverse(bounded), unconstrained)


def test_parameter_transform_inverse_rejects_boundary() -> None:
    transform = BoundedParameterTransform(torch.zeros(10), torch.ones(10))
    with pytest.raises(ValueError, match="strictly inside"):
        transform.inverse(torch.zeros(1, 10))


def test_standardized_target_mode_round_trip() -> None:
    targets = torch.arange(30.0).reshape(3, 10)
    standardizer = TargetStandardizer().fit(targets)
    torch.testing.assert_close(
        standardizer.inverse_transform(standardizer.transform(targets)), targets
    )
