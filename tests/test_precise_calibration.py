"""A meaningful exact-engine recovery and invalid-input check, independent of checkpoints."""
import numpy as np
import pytest
import torch
from src.mentor_dh_pinn.params_v2 import encode
from src.mentor_dh_pinn.precise_calibration import exact_polish
from src.mentor_dh_pinn.torch_pricer import price_call


def test_polish_reduces_actual_quote_error_without_truth_argument():
    p = np.array([.8, .06, .22, -.6, .08, 6., .035, .4, -.25, .045])
    t = np.repeat(np.array([.1, .5, 1., 2.]), 7)
    geo = dict(spot=np.ones(len(t)), strike=np.tile(np.linspace(.85, 1.15, 7), 4),
               tau=t, rate=np.full(len(t), .05), carry=np.full(len(t), .01))
    with torch.no_grad():
        y = price_call(torch.tensor(p), *(torch.tensor(v) for v in geo.values()), node_count=64).numpy()
    z0 = encode(p) + np.linspace(-.04, .04, 10)
    result = exact_polish(geo, y, z0, node_count=64, n_starts=1, max_nfev=80)
    assert result["objective"] < 1e-13
    assert len(result["params"]) == 10
    assert result["starts"][0]["nfev"] <= 80
    with pytest.raises(ValueError, match="finite"):
        exact_polish(geo, np.full_like(y, np.nan), z0)
