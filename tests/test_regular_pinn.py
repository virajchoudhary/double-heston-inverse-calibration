"""Independent PDE and derivative checks for the regular one-/two-factor PINN."""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten

from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN, residual


def _example(factors):
    if factors == 1:
        coords = mx.array([[0.04, 0.09, 0.25], [-0.07, 0.13, 0.7]])
        params = mx.array([[1.2, 0.08, 0.3, -0.55]])
    else:
        coords = mx.array([[0.04, 0.05, 0.04, 0.25], [-0.07, 0.08, 0.05, 0.7]])
        params = mx.array([[0.8, 0.04, 0.19, -0.5], [4.5, 0.05, 0.35, -0.3]])
    model = RegularVariancePINN(factors=factors, width=12, depth=2)
    return model, coords, params


@pytest.mark.parametrize("factors", [1, 2])
def test_zero_vol_of_vol_backbone_solves_pde(factors):
    model, coords, params = _example(factors)
    params[:, 2] = 0.0
    value, diagnostics = residual(model, coords, params)
    np.testing.assert_allclose(np.array(value), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.array(diagnostics["convexity"]), 1.0, atol=1e-6)
    assert model.features(coords, params).shape == (2, 13 if factors == 1 else 24)


@pytest.mark.parametrize("factors", [1, 2])
def test_transformed_pde_matches_independent_black_price_derivatives(factors):
    mx.random.seed(300 + factors)
    model, coords, params = _example(factors)
    model.head.weight = 0.03 * mx.random.normal(model.head.weight.shape)
    model.head.bias = mx.array([0.02])
    transformed, diagnostics = residual(model, coords, params)
    # This route differentiates Black call prices themselves; it does not use
    # the log-variance chain-rule expression implemented in residual().
    gradient = mx.grad(lambda state: mx.sum(model.price(state, params)))
    first = gradient(coords)
    second_x = mx.grad(lambda state: mx.sum(gradient(state)[..., 0]))(coords)
    variance = mx.sum(coords[:, 1:-1], axis=-1)
    raw = first[:, -1] - 0.5 * variance * (second_x[:, 0] - first[:, 0])
    for i in range(factors):
        j = i + 1
        vv = mx.grad(lambda state: mx.sum(gradient(state)[..., j]))(coords)[:, j]
        kappa, theta, sigma, rho = [params[i, k] for k in range(4)]
        raw = (raw - kappa * (theta - coords[:, j]) * first[:, j]
               - rho * sigma * coords[:, j] * second_x[:, j]
               - 0.5 * sigma**2 * coords[:, j] * vv)
    w = diagnostics["w"]
    d2 = coords[:, 0] / mx.sqrt(w) - 0.5 * mx.sqrt(w)
    black_total_variance_vega = mx.exp(-0.5 * d2**2) / math.sqrt(2 * math.pi) / (2 * mx.sqrt(w))
    independently_scaled = raw / (black_total_variance_vega * diagnostics["scale"])
    np.testing.assert_allclose(np.array(transformed), np.array(independently_scaled), atol=6e-6, rtol=2e-4)


def test_second_state_derivatives_allow_finite_nonzero_training_gradients():
    model, coords, params = _example(2)
    model.head.weight = mx.full(model.head.weight.shape, 0.01)

    def loss(network):
        r, _ = residual(network, coords, params)
        return mx.mean(r**2) + mx.mean((network.iv(coords, params) - 0.32)**2)

    value, gradients = nn.value_and_grad(model, loss)(model)
    mx.eval(value, gradients)
    leaves = [np.array(v) for _, v in tree_flatten(gradients)]
    assert np.isfinite(float(value))
    assert leaves and all(np.isfinite(v).all() for v in leaves)
    assert sum(float(np.abs(v).sum()) for v in leaves) > 0


def test_exact_terminal_payoff_and_pointwise_parameter_derivatives():
    model, coords, params = _example(2)
    model.head.weight = mx.full(model.head.weight.shape, 0.04)
    terminal = mx.array(coords)
    terminal[:, -1] = 0.0
    expected = np.maximum(np.exp(np.array(terminal[:, 0])) - 1.0, 0.0)
    np.testing.assert_allclose(np.array(model.price(terminal, params)), expected, atol=1e-7)
    batched = mx.broadcast_to(params, (2, 2, 4))
    single_iv = np.array(model.iv(coords[0], batched[0]))
    np.testing.assert_allclose(np.array(model.iv(coords, batched))[0], single_iv, atol=1e-7)
    parameter_gradient = mx.grad(lambda p: mx.sum(model.correction(coords, p)))(batched)
    assert np.isfinite(np.array(parameter_gradient)).all()
    assert np.max(np.abs(np.array(parameter_gradient))) > 0


def test_compiled_double_pde_gradient_matches_eager():
    model,coords,params=_example(2)
    model.head.weight=mx.full(model.head.weight.shape,.02)
    def loss(state,structural):
        r,_=residual(model,state,structural)
        return mx.mean(r*r)
    vg=nn.value_and_grad(model,loss)
    expected,eg=vg(coords,params)
    compiled=mx.compile(vg,inputs=model.state,outputs=model.state)
    actual,ag=compiled(coords,params);mx.eval(actual,ag,expected,eg)
    np.testing.assert_allclose(np.array(actual),np.array(expected),atol=2e-6,rtol=1e-4)
    for (_,a),(_,b) in zip(tree_flatten(ag),tree_flatten(eg)):
        np.testing.assert_allclose(np.array(a),np.array(b),atol=2e-5,rtol=1e-3)
