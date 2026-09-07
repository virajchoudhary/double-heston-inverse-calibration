"""Calibration must depend only on calibration quotes and the frozen ANN."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import coordinates

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("assess_regular_pinn", ROOT / "scripts/mentor_dh_pinn/assess_regular_pinn.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


def _surface():
    model = RegularVariancePINN(factors=1, width=8, depth=1)
    x = np.linspace(-0.15, 0.15, 12)
    tau = np.linspace(0.08, 1.5, 12)
    unit = np.array([0.45, 0.55, 0.4, 0.35, 0.5])
    query = np.column_stack([x, np.log(tau), np.broadcast_to(unit, (len(x), 5))])
    state, structural = coordinates(mx.array(query, dtype=mx.float32), 1, mx)
    observed_iv = np.asarray(model.iv(state, structural), float)
    return model, x, tau, observed_iv


def test_holdout_values_never_enter_calibration():
    model, x, tau, observed = _surface()
    fit_mask = np.arange(len(x)) % 3 != 2
    first = assessment.fit_network(model, x, tau, observed, fit_mask=fit_mask, starts=2, seed=33, max_nfev=30)
    altered_x, altered_tau, altered_iv = [a.copy() for a in (x, tau, observed)]
    altered_x[~fit_mask] = np.nan
    altered_tau[~fit_mask] = -1e20
    altered_iv[~fit_mask] = 1e100
    second = assessment.fit_network(model, altered_x, altered_tau, altered_iv,
                                    fit_mask=fit_mask, starts=2, seed=33, max_nfev=30)
    assert first["status"] == second["status"] == "fitted"
    np.testing.assert_array_equal(first["unit"], second["unit"])
    assert first["calibration_iv_sse"] == second["calibration_iv_sse"]
    assert first["calibration_iv_sse"] < 1e-9


def test_fitting_never_calls_fourier_pricer(monkeypatch):
    import src.mentor_dh_pinn.regular_pinn_data as data
    import src.mentor_dh_pinn.torch_pricer as pricer

    def forbidden(*args, **kwargs):
        raise AssertionError("Exact pricing must not enter network calibration")

    monkeypatch.setattr(assessment, "exact_prices", forbidden)
    monkeypatch.setattr(data, "exact_prices", forbidden)
    monkeypatch.setattr(pricer, "price_call", forbidden)
    monkeypatch.setattr(pricer, "price_call_single", forbidden)
    model, x, tau, observed = _surface()
    result = assessment.fit_network(model, x, tau, observed, starts=1, seed=3, max_nfev=25)
    assert result["status"] == "fitted"
    assert len(result["starts"]) == 1
    assert result["starts"][0]["sse"] <= result["starts"][0]["initial_sse"]


def test_invalid_calibration_quote_is_rejected():
    model, x, tau, observed = _surface()
    observed[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        assessment.fit_network(model, x, tau, observed, starts=1)


def test_failed_starts_are_retained_and_frozen_weights_do_not_change(monkeypatch):
    from mlx.utils import tree_flatten
    model, x, tau, observed = _surface()
    before = [np.asarray(value).copy() for _, value in tree_flatten(model.parameters())]
    def failed_solver(*args, **kwargs):
        raise FloatingPointError("injected failure")
    monkeypatch.setattr(assessment, "least_squares", failed_solver)
    result = assessment.fit_network(model, x, tau, observed, starts=3)
    assert result["status"] == "all_starts_failed"
    assert len(result["starts"]) == 3
    assert all("injected failure" in row["error"] for row in result["starts"])
    assert "physical" not in result
    for previous, (_, current) in zip(before, tree_flatten(model.parameters())):
        np.testing.assert_array_equal(previous, np.asarray(current))


def test_failed_recovery_validation_is_scored_as_failure(monkeypatch):
    import scripts.mentor_dh_pinn.train_regular_pinn as training
    import scripts.mentor_dh_pinn.assess_regular_pinn as evaluator
    monkeypatch.setattr(evaluator, "fit_network", lambda *a, **kw: {
        "status": "all_starts_failed", "starts": []})
    q = np.zeros((126, 7)); q[:, 1] = np.log(0.5)
    score, records = training.recovery_validation(None, 1, [(q, {"w": np.ones(126)}, np.ones(5))])
    assert np.isinf(score)
    assert records[0]["success"] is False


def test_float64_parameter_jacobian_matches_difference_and_scoring(monkeypatch):
    model, x, tau, observed = _surface()
    model.head.weight = mx.full(model.head.weight.shape, 0.025)
    scipy_solver = assessment.least_squares
    checked = []

    def checked_solver(fun, initial, *, jac, **kwargs):
        analytical = jac(initial)
        differences = []
        for j in range(len(initial)):
            delta = np.zeros_like(initial)
            delta[j] = 1e-5
            differences.append((fun(initial + delta) - fun(initial - delta)) / (2e-5))
        np.testing.assert_allclose(analytical, np.stack(differences, axis=-1), atol=2e-8, rtol=2e-5)
        np.testing.assert_allclose(fun(initial) + observed,
                                   assessment._network_iv(model, x, tau, initial), atol=1e-14)
        assert analytical.dtype == np.float64
        checked.append(True)
        return scipy_solver(fun, initial, jac=jac, **kwargs)

    monkeypatch.setattr(assessment, "least_squares", checked_solver)
    result = assessment.fit_network(model, x, tau, observed, starts=1, seed=3, max_nfev=10)
    assert result["status"] == "fitted" and checked == [True]
    assert "float64" in result["inference"]


def test_price_gate_depends_on_heldout_error():
    x = np.zeros(4)
    tau = np.full(4, 0.5)
    true_iv = np.full(4, 0.3)
    truth = assessment.black_call(x, true_iv**2 * tau)
    holdout = np.array([False, False, False, True])
    prediction = truth.copy()
    prediction[-1] += 1.5e-5
    score, _ = assessment._score_price_iv(prediction, truth, true_iv, x, tau, holdout)
    assert score["all_price_rmse_spot"] < 1e-5
    assert score["holdout_price_rmse_spot"] > 1e-5
    assert score["price_gate"] is False


def test_compiled_training_step_supports_nested_pde_and_parameter_derivatives():
    from functools import partial
    import mlx.nn as nn
    import mlx.optimizers as optim
    from src.mentor_dh_pinn.regular_pinn import residual

    model = RegularVariancePINN(factors=2, width=8, depth=2)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=0.001, weight_decay=1e-6)
    state = [model.state, optimizer.state, mx.random.state]
    query = mx.array(np.column_stack([[-0.03, 0.07], np.log([0.2, 0.7]),
                                     np.full((2, 10), 0.45)]), dtype=mx.float32)

    def correction(q):
        coords, structural = coordinates(q, 2, mx)
        return model.correction(coords, structural)

    def loss(q):
        coords, structural = coordinates(q, 2, mx)
        pde, _ = residual(model, coords, structural)
        sensitivity = mx.grad(lambda points: mx.sum(correction(points)))(q)[:, 2:]
        return mx.mean(pde**2) + mx.mean((correction(q) - 0.05)**2) + mx.mean(sensitivity**2)

    value_grad = nn.value_and_grad(model, loss)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(q):
        value, gradient = value_grad(q)
        optimizer.update(model, gradient)
        return value

    first = step(query)
    mx.eval(state, first)
    second = step(query)
    mx.eval(state, second)
    assert np.isfinite(float(first)) and np.isfinite(float(second))
    assert np.any(np.asarray(model.head.weight) != 0.0)
