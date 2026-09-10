import json

import mlx.core as mx
import numpy as np
import pytest
import torch

from scripts.mentor_dh_pinn.finetune_regular_lbfgs import directional_check
from scripts.mentor_dh_pinn.regular_lbfgs_jacobian_loss import (
    FrozenJacobianObjective, neural_iv_unit_jacobian, training_geometry,
)
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import coordinates, decode_unit, draw_points, expected_variance
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def inputs():
    rng = np.random.default_rng(895)
    q = draw_points(4, 2, 896)
    training = {'q': q, 'g': .003 * q[:, 2:].sum(axis=1),
                'dg_du': np.full((len(q), 10), .003)}
    physics = draw_points(4, 2, 897, collocation=True)
    sq = np.broadcast_to(q[:3, None, :], (3, 12, 12)).copy()
    sq[..., 0] = np.linspace(-.12, .12, 12)
    sq[..., 1] = np.log(np.linspace(.1, 1.8, 12))
    iv = np.sqrt(expected_variance(sq, 2)) * np.exp(.006 + .002 * sq[..., 0])
    b = rng.normal(size=(3, 10, 12)) * 50
    b[-1, -1] = 0.
    surfaces = {'q': sq, 'iv': iv, 'preconditioner': b,
                'usable': np.ones(3, dtype=bool), 'rank': np.array([10, 10, 9])}
    mx.random.seed(898)
    model = RegularVariancePINN(2, width=8, depth=2)
    model.head.weight = .002 * mx.random.normal(model.head.weight.shape)
    return model, training, physics, np.full(10, .02), surfaces


def test_training_transform_and_rank_projector_identities():
    *_, surfaces = inputs()
    units = surfaces['q'][:, 0, 2:]
    b = surfaces['preconditioner']
    t, p = training_geometry(units, b, surfaces['rank'], 2)
    np.testing.assert_allclose(p @ p, p, atol=3e-15)
    np.testing.assert_allclose(p, p.transpose(0, 2, 1), atol=1e-15)
    np.testing.assert_allclose(p @ b, b, atol=5e-13)
    np.testing.assert_allclose(p, b @ np.linalg.pinv(b), atol=5e-14)
    np.testing.assert_allclose(np.trace(p, axis1=1, axis2=2), [10, 10, 9], atol=1e-13)
    for unit, transform in zip(units, t):
        physical = decode_unit(unit, 2)
        tolerance = .05 * np.abs(physical)
        tolerance[3::5] = .05
        epsilon = 1e-5
        columns = [(decode_unit(unit + epsilon * transform[:, j], 2)
                    - decode_unit(unit - epsilon * transform[:, j], 2)) / (2 * epsilon)
                   for j in range(10)]
        np.testing.assert_allclose(np.stack(columns, axis=1), np.diag(tolerance), atol=2e-9)
    with pytest.raises(ValueError, match='rank'):
        training_geometry(units, b, np.full(3, 10), 2)


def test_neural_jacobian_holds_geometry_fixed():
    model, _, _, _, surfaces = inputs()
    q = mx.array(surfaces['q'].astype(np.float32))
    derivative = np.asarray(neural_iv_unit_jacobian(model, q))
    # Weak correlation sensitivities are below float32 IV differencing
    # resolution. The same weights in float64 provide the independent FD check.
    reference = TorchRegularVariancePINN.from_mlx(model)
    tq = torch.tensor(np.asarray(q).copy(), dtype=torch.float64)
    step = 1e-4
    for j in (0, 4, 8):
        bump = torch.zeros_like(tq)
        bump[..., j + 2] = step
        plus = reference.iv(*coordinates(tq + bump, 2, torch)).detach().numpy()
        minus = reference.iv(*coordinates(tq - bump, 2, torch)).detach().numpy()
        np.testing.assert_allclose(derivative[..., j], (plus - minus) / (2 * step),
                                   rtol=1e-4, atol=1e-7)


def test_compiled_loss_gradient_chunks_and_frozen_coefficient(monkeypatch):
    from src.mentor_dh_pinn import regular_pinn_data, torch_pricer
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference pricing must not enter the training objective')
    monkeypatch.setattr(regular_pinn_data, 'exact_prices', forbidden)
    monkeypatch.setattr(torch_pricer, 'price_call', forbidden)
    model, training, physics, scale, surfaces = inputs()
    full = FrozenJacobianObjective(model, training, physics, scale, chunk=4,
                                   surfaces=surfaces, surface_chunk=3)
    chunks = FrozenJacobianObjective(model, training, physics, scale, chunk=2,
                                     surfaces=surfaces, surface_chunk=1)
    x = full.vector()
    a, ag = full(x)
    b, bg = chunks(x)
    np.testing.assert_allclose(a, b, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(ag, bg, rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(full.jacobian_coefficient, chunks.jacobian_coefficient, rtol=2e-5)
    meta = full.jacobian_metadata
    assert full.bias_coefficient == 0.
    np.testing.assert_allclose(meta['initial_weighted_jacobian_gradient_norm'],
                               .5 * meta['initial_base_plus_group_price_gradient_norm'])
    snapshot = json.dumps(full.surface_metadata, sort_keys=True)
    coefficient = full.jacobian_coefficient
    calls = full.calls
    value, gradient = full(x + .001)
    assert full.calls == calls + 1
    assert full.last[0] == value
    np.testing.assert_array_equal(full.last[1], gradient)
    assert full.jacobian_coefficient == coefficient
    assert json.dumps(full.surface_metadata, sort_keys=True) == snapshot
    full.assign(x)
    assert min(check['relative_error'] for check in directional_check(full, x)) < .01
