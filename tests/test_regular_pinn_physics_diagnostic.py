"""Fresh-point diagnostic keeps nonfinite failures and is chunk-independent."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import draw_points

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("physics_diagnostic", ROOT / "scripts/mentor_dh_pinn/check_regular_pinn_physics.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_diagnostic_is_chunk_independent_and_reports_failures():
    model = RegularVariancePINN(factors=2, width=8, depth=1)
    points = draw_points(12, 2, 907699, collocation=True)
    first = diagnostic.evaluate_points(model, points, chunk=4)
    second = diagnostic.evaluate_points(model, points, chunk=12)
    a = first["dimensionless_pde_residual"]["all_points"]["rmse_all"]
    b = second["dimensionless_pde_residual"]["all_points"]["rmse_all"]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    assert first["all_diagnostics_finite"]
    assert first["terminal_condition"]["passed_roundoff_tolerance"]
    assert first["same_weights_float32_float64_iv_parity"]["passed"]
    failed = diagnostic.distribution([1.0, np.nan])
    assert failed["count"] == 2 and failed["nonfinite"] == 1
    assert failed["rmse_all"] is None


def test_native_deep_checkpoint_diagnostic_retains_residual_branches():
    import torch
    from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual
    from src.mentor_dh_pinn.regular_pinn_data import coordinates
    torch.manual_seed(911)
    model=TorchRegularVariancePINN(factors=2,width=8,depth=5,residual_blocks=6)
    with torch.no_grad():
        for _,last in model.extra:last.weight.normal_(0,.02)
    model.requires_grad_(False)
    points=draw_points(12,2,909199,collocation=True)
    result=diagnostic.evaluate_points(model,points,chunk=4)
    c,p=coordinates(torch.tensor(points,dtype=torch.float64),2,torch)
    expected=float(residual(model,c,p)[0].square().mean().sqrt().detach())
    np.testing.assert_allclose(result['dimensionless_pde_residual']['all_points']['rmse_all'],expected,rtol=1e-12)
    assert result['hidden_layers']==17
    assert result['same_weights_float32_float64_iv_parity']['status']=='not_applicable'
    assert result['all_diagnostics_finite']
