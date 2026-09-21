"""Independent float64 price-PDE and training checks for the regular PINN."""
import math

import numpy as np
import pytest
import torch

from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual


def example(factors):
    torch.manual_seed(400 + factors)
    model = TorchRegularVariancePINN(factors=factors, width=10, depth=2)
    with torch.no_grad():
        model.head.weight.mul_(0.08)
        model.head.bias.fill_(0.01)
    if factors == 1:
        coords = [[0.04, 0.09, 0.25], [-0.07, 0.13, 0.7]]
        structural = [[1.2, 0.08, 0.3, -0.55]]
    else:
        coords = [[0.04, 0.05, 0.04, 0.25], [-0.07, 0.08, 0.05, 0.7]]
        structural = [[0.8, 0.04, 0.19, -0.5], [4.5, 0.05, 0.35, -0.3]]
    return model, torch.tensor(coords, dtype=torch.float64), torch.tensor(structural, dtype=torch.float64)


@pytest.mark.parametrize("factors", [1, 2])
def test_transformed_residual_matches_raw_black_price_pde(factors):
    model, coords, structural = example(factors)
    coords.requires_grad_(True)
    transformed, diagnostics = residual(model, coords, structural)
    call = model.price(coords, structural)

    def d(value):
        return torch.autograd.grad(value.sum(), coords, create_graph=True)[0]

    first = d(call)
    second_x = d(first[:, 0])
    raw = first[:, -1] - 0.5 * coords[:, 1:-1].sum(-1) * (second_x[:, 0] - first[:, 0])
    for i in range(factors):
        j = i + 1
        kappa, theta, sigma, rho = structural[i]
        raw = (raw - kappa * (theta - coords[:, j]) * first[:, j]
               - rho * sigma * coords[:, j] * second_x[:, j]
               - 0.5 * sigma**2 * coords[:, j] * d(first[:, j])[:, j])
    w = diagnostics["w"]
    d2 = coords[:, 0] / torch.sqrt(w) - 0.5 * torch.sqrt(w)
    c_w = torch.exp(-0.5 * d2**2) / (math.sqrt(2.0 * math.pi) * 2.0 * torch.sqrt(w))
    torch.testing.assert_close(transformed, raw / (c_w * diagnostics["scale"]), atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("factors", [1, 2])
def test_zero_vol_of_vol_limit_and_terminal_payoff(factors):
    model, coords, structural = example(factors)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    structural[:, 2] = 0.0
    result, _ = residual(model, coords, structural)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=2e-13, rtol=0.0)
    terminal = coords.clone()
    terminal[:, -1] = 0.0
    expected = torch.clamp_min(torch.expm1(coords[:, 0]), 0.0)
    torch.testing.assert_close(model.price(terminal, structural), expected, atol=0.0, rtol=0.0)


def test_pde_loss_backpropagates_to_weights_and_matches_finite_difference():
    model, coords, structural = example(2)
    result, _ = residual(model, coords, structural)
    loss = result.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    analytic = model.head.weight.grad[0, 0].item()
    assert abs(analytic) > 1e-9
    original = model.head.weight[0, 0].item()
    losses = []
    for delta in (1e-5, -1e-5):
        with torch.no_grad():
            model.head.weight[0, 0] = original + delta
        losses.append(residual(model, coords, structural)[0].square().mean().item())
    with torch.no_grad():
        model.head.weight[0, 0] = original
    np.testing.assert_allclose(analytic, (losses[0] - losses[1]) / 2e-5, atol=2e-9, rtol=2e-6)
    assert not coords.requires_grad  # The caller's input was not mutated.


def test_residual_parity_with_mlx_and_frozen_weight_copy():
    mx = pytest.importorskip("mlx.core")
    from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN, residual as mlx_residual

    mx.random.seed(77)
    source = RegularVariancePINN(factors=2, width=10, depth=2)
    source.head.weight = 0.025 * mx.random.normal(source.head.weight.shape)
    source.head.bias = mx.array([0.01])
    mx.eval(source.parameters())
    model = TorchRegularVariancePINN.from_mlx(source)
    _, coords, structural = example(2)
    actual, diagnostics = residual(model, coords, structural)
    expected, mlx_diagnostics = mlx_residual(source, mx.array(coords.numpy(), dtype=mx.float32),
                                             mx.array(structural.numpy(), dtype=mx.float32))
    np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected), atol=2e-6, rtol=2e-4)
    for key in ("ell", "w", "l_tau", "convexity"):
        np.testing.assert_allclose(diagnostics[key].detach().numpy(), np.asarray(mlx_diagnostics[key]), atol=2e-5, rtol=2e-4)
    assert not any(p.requires_grad for p in model.parameters())
