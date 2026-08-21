from __future__ import annotations

import pytest
import torch

from models.ann_model import ANNInverseCalibrator


def test_ann_returns_batch_by_ten() -> None:
    model = ANNInverseCalibrator(input_size=108)
    output = model(torch.zeros(4, 108))
    assert output.shape == (4, 10)


@pytest.mark.parametrize("shape", [(108,), (4, 107), (4, 108, 1)])
def test_ann_rejects_malformed_input(shape: tuple[int, ...]) -> None:
    model = ANNInverseCalibrator(input_size=108)
    with pytest.raises(ValueError, match="Expected features"):
        model(torch.zeros(shape))
