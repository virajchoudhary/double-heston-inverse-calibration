import json

import mlx.core as mx
import numpy as np
import pytest

from scripts.mentor_dh_pinn.finetune_regular_lbfgs import directional_check
from scripts.mentor_dh_pinn.regular_lbfgs_surface_loss import (
    FrozenSurfaceObjective, stable_iv_error,
)
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import draw_points, expected_variance


def inputs():
    # Analytic fixtures exercise objective plumbing without any assessment file
    # or Fourier reference pricing. They are not model-recovery evidence.
    rng = np.random.default_rng(891)
    q = draw_points(6, 2, 892)
    training = {'q': q, 'g': .003 * q[:, 2:].sum(axis=1),
                'dg_du': np.full((len(q), 10), .003)}
    physics = draw_points(6, 2, 893, collocation=True)
    sq = np.broadcast_to(q[:5, None, :], (5, 12, 12)).copy()
    sq[..., 0] = np.linspace(-.12, .12, 12)
    sq[..., 1] = np.log(np.linspace(.1, 1.8, 12))
    g = .006 + .002 * sq[..., 0] + .001 * np.exp(sq[..., 1])
    iv = np.sqrt(expected_variance(sq, 2)) * np.exp(g)
    surfaces = {'q': sq, 'iv': iv, 'preconditioner': rng.normal(size=(5, 10, 12)) * 50,
                'usable': np.ones(5, dtype=bool), 'rank': np.full(5, 10)}
    mx.random.seed(894)
    model = RegularVariancePINN(2, width=8, depth=2)
    model.head.weight = .002 * mx.random.normal(model.head.weight.shape)
    return model, training, physics, np.full(10, .02), surfaces


def test_quadratic_surface_chunk_gradient_and_frozen_coefficient():
    model, training, physics, scale, surfaces = inputs()
    full = FrozenSurfaceObjective(model, training, physics, scale, chunk=6,
                                  surfaces=surfaces, surface_chunk=5)
    chunks = FrozenSurfaceObjective(model, training, physics, scale, chunk=3,
                                    surfaces=surfaces, surface_chunk=2)
    x = full.vector()
    a, ag = full(x)
    b, bg = chunks(x)
    np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(ag, bg, rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(full.bias_coefficient, chunks.bias_coefficient, rtol=2e-5)
    meta = full.surface_metadata
    np.testing.assert_allclose(meta['initial_weighted_bias_gradient_norm'],
                               .5 * meta['initial_base_plus_group_price_gradient_norm'])
    assert meta['usable_surfaces'] == 5
    assert meta['training_quotes'] == 60
    frozen = json.dumps(meta, sort_keys=True)
    coefficient = full.bias_coefficient
    before_calls = full.calls
    value, gradient = full(x + .001)
    assert full.calls == before_calls + 1
    assert full.last[0] == value
    np.testing.assert_array_equal(full.last[1], gradient)
    assert full.bias_coefficient == coefficient
    assert json.dumps(full.surface_metadata, sort_keys=True) == frozen
    full.assign(x)
    checks = directional_check(full, x)
    assert min(check['relative_error'] for check in checks) < .01


def test_zero_share_is_price_control_and_unusable_rows_stay_excluded():
    model, training, physics, scale, surfaces = inputs()
    surfaces['usable'][-1] = False
    surfaces['iv'][-1] = np.nan
    objective = FrozenSurfaceObjective(model, training, physics, scale, chunk=6,
                                       surfaces=surfaces, bias_gradient_share=0., surface_chunk=4)
    x = objective.vector()
    a, ag = objective(x)
    # Changing B cannot change the control objective when its coefficient is zero.
    altered = {**surfaces, 'preconditioner': surfaces['preconditioner'] * 10}
    control = FrozenSurfaceObjective(model, training, physics, scale, chunk=6,
                                     surfaces=altered, bias_gradient_share=0., surface_chunk=4)
    b, bg = control(x)
    assert objective.bias_coefficient == 0
    assert objective.surface_metadata['excluded_unusable_surfaces'] == 1
    np.testing.assert_allclose(a, b, rtol=1e-6)
    np.testing.assert_allclose(ag, bg, rtol=1e-6, atol=1e-7)


def test_stable_error_matches_backbone_identity_and_retains_small_errors():
    g = np.array([.006, -.025, .08], dtype=np.float32)
    target = np.array([.00599999, -.02500001, .08000001], dtype=np.float32)
    iv = np.array([.25, .4, .3], dtype=np.float32)
    actual = np.asarray(stable_iv_error(mx.array(g), mx.array(target), mx.array(iv)))
    expected = iv.astype(float) * np.expm1(g.astype(float) - target.astype(float))
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-15)
    backbone_root = iv.astype(float) * np.exp(-target.astype(float))
    np.testing.assert_allclose(expected, backbone_root * np.exp(g.astype(float)) - iv,
                               atol=1e-16)
    assert np.all(actual != 0)


@pytest.mark.parametrize('share', [-1., float('nan')])
def test_invalid_bias_share_rejected(share):
    model, training, physics, scale, surfaces = inputs()
    with pytest.raises(ValueError, match='bias_gradient_share'):
        FrozenSurfaceObjective(model, training, physics, scale, surfaces=surfaces,
                                bias_gradient_share=share)
