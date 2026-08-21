from __future__ import annotations

import numpy as np
import torch

from src.double_heston import price_double_heston_surface
from src.torch_double_heston import price_double_heston_surface_batch


VALID_PARAMETERS = np.array(
    [0.8, 0.04, 0.20, -0.45, 0.05, 3.0, 0.02, 0.25, -0.25, 0.025],
    dtype=np.float64,
)
SPOT = 100.0
RATE = 0.05
DIVIDEND = 0.01


def test_torch_surface_matches_numpy_surface() -> None:
    strikes = np.array([85.0, 95.0, 100.0, 110.0], dtype=np.float64)
    maturities = np.array([0.2, 0.5, 1.0, 1.8], dtype=np.float64)
    option_types = ["call", "put", "call", "put"]
    expected = price_double_heston_surface(
        SPOT,
        strikes,
        maturities,
        RATE,
        DIVIDEND,
        option_types,
        VALID_PARAMETERS,
    )
    actual = price_double_heston_surface_batch(
        torch.tensor(np.expand_dims(VALID_PARAMETERS, axis=0), dtype=torch.float64),
        torch.tensor([SPOT], dtype=torch.float64),
        torch.tensor(np.expand_dims(strikes, axis=0), dtype=torch.float64),
        torch.tensor(np.expand_dims(maturities, axis=0), dtype=torch.float64),
        torch.tensor([RATE], dtype=torch.float64),
        torch.tensor([DIVIDEND], dtype=torch.float64),
        [option_types],
    )
    np.testing.assert_allclose(
        actual.detach().cpu().numpy()[0],
        expected,
        rtol=0.0,
        atol=1e-8,
    )


def test_torch_pricer_backpropagates_to_parameters() -> None:
    parameters = torch.tensor(
        np.expand_dims(VALID_PARAMETERS, axis=0),
        dtype=torch.float64,
        requires_grad=True,
    )
    prices = price_double_heston_surface_batch(
        parameters,
        torch.tensor([SPOT], dtype=torch.float64),
        torch.tensor([[90.0, 100.0, 110.0]], dtype=torch.float64),
        torch.tensor([[0.25, 0.75, 1.25]], dtype=torch.float64),
        torch.tensor([RATE], dtype=torch.float64),
        torch.tensor([DIVIDEND], dtype=torch.float64),
        [["call", "put", "call"]],
    )
    loss = prices.sum()
    loss.backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert torch.any(parameters.grad.abs() > 0.0)
