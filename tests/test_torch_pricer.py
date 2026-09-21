"""Cross-platform regression tests for the differentiable Fourier teacher."""

import math

import numpy as np
import torch

from src.double_heston import price_double_heston_call
from src.double_heston_reference import reference_double_heston_call
from src.mentor_dh_pinn.torch_pricer import price_call


SUBNORMAL_LOG1P_PARAMETERS = np.array(
    [
        0.6278737771824936,
        0.11574142466032664,
        0.20566191103560955,
        -0.0671157254877468,
        0.04698819188707379,
        4.115698567795603,
        0.15624404187519605,
        0.7554620538906811,
        -0.029848632648244666,
        0.021307640409021857,
    ],
    dtype=np.float64,
)
SUBNORMAL_LOG1P_SPOT = math.exp(-1.7743945416859939)
SUBNORMAL_LOG1P_TAU = 2.0


def test_subnormal_complex_log1p_row_matches_independent_pricers() -> None:
    """Proposal 34 must not become NaN at the largest 128-node GL abscissa."""
    parameters = torch.tensor(
        SUBNORMAL_LOG1P_PARAMETERS, dtype=torch.float64, requires_grad=True
    )
    actual = price_call(
        parameters,
        torch.tensor(SUBNORMAL_LOG1P_SPOT, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(SUBNORMAL_LOG1P_TAU, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        node_count=128,
    )
    expected = price_double_heston_call(
        SUBNORMAL_LOG1P_SPOT,
        1.0,
        SUBNORMAL_LOG1P_TAU,
        0.0,
        0.0,
        SUBNORMAL_LOG1P_PARAMETERS,
        node_count=128,
    )
    reference, diagnostics = reference_double_heston_call(
        SUBNORMAL_LOG1P_SPOT,
        1.0,
        SUBNORMAL_LOG1P_TAU,
        0.0,
        0.0,
        SUBNORMAL_LOG1P_PARAMETERS,
    )

    assert torch.isfinite(actual)
    assert diagnostics["reliable"] is True
    value = float(actual.detach())
    np.testing.assert_allclose(value, expected, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(value, reference, rtol=0.0, atol=2e-12)

    actual.backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
