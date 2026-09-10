"""Fixed training-only inverse-sensitivity consistency for the regular PINN.

This is the grouped-price control plus a Jacobian loss, not a combination with
the quadratic parameter-bias loss. It runs no reference pricer or calibration.
"""
import hashlib
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import torch

from scripts.mentor_dh_pinn.regular_lbfgs_surface_loss import FrozenSurfaceObjective
from scripts.mentor_dh_pinn.train_regular_pinn import array
from src.mentor_dh_pinn.regular_pinn_data import coordinates, decode_unit


def training_geometry(units, preconditioners, ranks, factors):
    """Float64 tolerance-coordinate maps and retained-space projectors.

    T maps physical recovery-tolerance shifts to unit-coordinate shifts. The
    column space of each stored B determines P; the recorded rank is verified
    against B before truncation. No price or IV Jacobian teacher is evaluated.
    """
    units, b = np.asarray(units, dtype=float), np.asarray(preconditioners, dtype=float)
    ranks = np.asarray(ranks)
    d = 5 * factors
    if (units.ndim != 2 or units.shape[1] != d or b.ndim != 3
            or b.shape[:2] != (len(units), d) or ranks.shape != (len(units),)
            or not np.isfinite(units).all() or not np.isfinite(b).all()
            or not np.isfinite(ranks).all() or np.any((units < 0) | (units > 1))
            or np.any(ranks != np.floor(ranks)) or np.any((ranks < 0) | (ranks > d))):
        raise ValueError("Invalid training parameter geometry")
    transforms, projectors = [], []
    for unit, matrix, rank in zip(units, b, ranks.astype(int)):
        u = torch.tensor(unit, dtype=torch.float64, requires_grad=True)
        p = decode_unit(u, factors, torch).detach().numpy()
        dp = torch.autograd.functional.jacobian(lambda z: decode_unit(z, factors, torch), u).numpy()
        tolerance = .05 * np.abs(p)
        tolerance[3::5] = .05
        transforms.append(np.linalg.solve(dp, np.diag(tolerance)))
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        numerical_rank = int(np.sum(singular > singular[0] * 1e-10)) if singular[0] else 0
        if numerical_rank != rank:
            raise ValueError("Stored retained rank does not match training preconditioner")
        retained = left[:, :rank]
        projectors.append(retained @ retained.T)
    transforms, projectors = np.asarray(transforms), np.asarray(projectors)
    if not np.isfinite(transforms).all():
        raise ValueError("Nonfinite training tolerance transform")
    return transforms, projectors


def neural_iv_unit_jacobian(model, q):
    """Pointwise IV derivatives; x and log(tau) are held fixed."""
    flat = q.reshape(-1, q.shape[-1])
    derivative = mx.grad(lambda z: mx.sum(model.iv(*coordinates(z, model.factors, mx))))(flat)
    return derivative[:, 2:].reshape(*q.shape[:2], 5 * model.factors)


class FrozenJacobianObjective(FrozenSurfaceObjective):
    def __init__(self, model, training, physics, scale, chunk=1024,
                 pde_weight=.2, sensitivity=.2, *, surfaces,
                 jacobian_gradient_share=.5, surface_chunk=64):
        if not math.isfinite(jacobian_gradient_share) or jacobian_gradient_share < 0:
            raise ValueError("jacobian_gradient_share must be finite and nonnegative")
        super().__init__(model, training, physics, scale, chunk, pde_weight, sensitivity,
                         surfaces=surfaces, bias_gradient_share=0., surface_chunk=surface_chunk)
        usable = np.asarray(surfaces['usable'])
        q = np.asarray(surfaces['q'], dtype=float)[usable]
        b = np.asarray(surfaces['preconditioner'], dtype=float)[usable]
        ranks = np.asarray(surfaces['rank'])[usable]
        transforms, projectors = training_geometry(q[:, 0, 2:], b, ranks, self.factors)
        self.st, self.sp = array(transforms), array(projectors)
        mx.eval(self.st, self.sp)
        count, d = len(q), 5 * self.factors

        def jacobian_loss(q, preconditioner, transform, projector):
            jacobian = neural_iv_unit_jacobian(model, q)
            error = preconditioner @ (jacobian @ transform) - projector
            return mx.sum(error**2) / (count * d * d)

        self.jacobian_vg = mx.compile(nn.value_and_grad(model, jacobian_loss),
                                      inputs=model.state, outputs=model.state)
        initial = self.vector()
        base_value, base_gradient = super().__call__(initial)
        jacobian_value, jacobian_gradient = self._jacobian_value_gradient()
        reference_norm = float(np.linalg.norm(base_gradient))
        jacobian_norm = float(np.linalg.norm(jacobian_gradient))
        if jacobian_gradient_share > 0 and jacobian_norm == 0:
            raise ValueError("Cannot normalize a zero initial Jacobian-loss gradient")
        self.jacobian_coefficient = (jacobian_gradient_share * reference_norm / jacobian_norm
                                     if jacobian_gradient_share > 0 else 0.)
        if not math.isfinite(self.jacobian_coefficient):
            raise FloatingPointError("Nonfinite frozen Jacobian coefficient")
        self.jacobian_metadata = {
            'loss': 'mean((B @ (J_neural_unit @ T) - P)**2)',
            'jacobian_gradient_share': float(jacobian_gradient_share),
            'jacobian_coefficient': self.jacobian_coefficient,
            'initial_base_plus_group_price_gradient_norm': reference_norm,
            'initial_unweighted_jacobian_gradient_norm': jacobian_norm,
            'initial_weighted_jacobian_gradient_norm': self.jacobian_coefficient * jacobian_norm,
            'initial_base_plus_group_price_objective': base_value,
            'initial_unweighted_jacobian_objective': jacobian_value,
            'coefficient_policy': 'Computed once from initial training gradients; fixed thereafter',
            'quadratic_bias_coefficient': self.bias_coefficient,
            'transform': 'solve(dp/du, diag(0.05*abs(p) for positive parameters; 0.05 for rho))',
            'projector': 'Orthogonal projector onto retained column space of training B; stored rank checked',
            'transform_float64_sha256': hashlib.sha256(transforms.tobytes()).hexdigest(),
            'projector_float64_sha256': hashlib.sha256(projectors.tobytes()).hexdigest(),
            'maximum_projector_idempotence_error': float(np.max(np.abs(projectors @ projectors - projectors))),
            'usable_surfaces': count, 'all_usable_surfaces_every_evaluation': True,
            'interpretation': ('Training-only derivative consistency, not recovered parameters. '
                               'Truncated directions are not restored. No Fourier pricing, clipping, '
                               'coefficient adaptation, or exposed assessment cases.'),
        }
        self.surface_metadata['jacobian_consistency'] = self.jacobian_metadata
        self.calls, self.last = 0, None

    def _jacobian_value_gradient(self):
        value = mx.array(0.)
        gradient = mx.zeros((sum(size for _, _, size in self.layout),))
        for start in range(0, len(self.sq), self.surface_chunk):
            sl = slice(start, start + self.surface_chunk)
            val, gr = self.jacobian_vg(self.sq[sl], self.sb[sl], self.st[sl], self.sp[sl])
            value = value + val
            gradient = gradient + mx.concatenate([v.reshape(-1) for _, v in tree_flatten(gr)])
            mx.eval(value, gradient)
        result = float(value), np.asarray(gradient, dtype=float)
        if not math.isfinite(result[0]) or not np.isfinite(result[1]).all():
            raise FloatingPointError("Nonfinite training Jacobian objective/gradient")
        return result

    def __call__(self, vector):
        base_value, base_gradient = super().__call__(vector)
        value, gradient = self._jacobian_value_gradient()
        self.last = (base_value + self.jacobian_coefficient * value,
                     base_gradient + self.jacobian_coefficient * gradient)
        if not math.isfinite(self.last[0]) or not np.isfinite(self.last[1]).all():
            raise FloatingPointError("Nonfinite combined training Jacobian objective/gradient")
        return self.last
