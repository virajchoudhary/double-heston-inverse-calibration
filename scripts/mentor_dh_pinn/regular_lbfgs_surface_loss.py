"""Fixed grouped TRAINING-surface losses for the regular price-PDE PINN.

The quadratic preconditioned IV error is a local parameter-bias proxy, not
recovered parameters. No calibration, reference pricer, or assessment is run.
"""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from scripts.mentor_dh_pinn.finetune_regular_lbfgs import FrozenObjective
from scripts.mentor_dh_pinn.train_regular_pinn import array
from src.mentor_dh_pinn.regular_pinn_data import coordinates, expected_variance


def stable_iv_error(predicted_g, target_g, target_iv):
    """IV error without subtracting nearly equal IV levels.

    For a shared expected-variance backbone, IV = sqrt(vbar) * exp(g), so
    this is algebraically identical to predicted_IV - target_IV. Reference
    corrections/backbones are computed in float64 before their MLX conversion.
    """
    return target_iv * mx.expm1(predicted_g - target_g)


class FrozenSurfaceObjective(FrozenObjective):
    def __init__(self, model, training, physics, scale, chunk=1024,
                 pde_weight=.2, sensitivity=.2, *, surfaces,
                 bias_gradient_share=.5, surface_chunk=64):
        if not math.isfinite(bias_gradient_share) or bias_gradient_share < 0:
            raise ValueError("bias_gradient_share must be finite and nonnegative")
        if not isinstance(surface_chunk, int) or surface_chunk < 1:
            raise ValueError("surface_chunk must be a positive integer")
        q, iv, b = (np.asarray(surfaces[k], dtype=float)
                    for k in ('q', 'iv', 'preconditioner'))
        usable = np.asarray(surfaces['usable'])
        ranks = np.asarray(surfaces['rank'])
        nparam = 5 * model.factors
        if (q.ndim != 3 or q.shape[2] != nparam + 2 or not q.shape[1]
                or iv.shape != q.shape[:2]
                or b.shape != (len(q), nparam, q.shape[1])
                or usable.shape != (len(q),) or usable.dtype != np.bool_
                or ranks.shape != (len(q),)):
            raise ValueError("Invalid grouped training surface shapes/mask")
        count = int(usable.sum())
        if not count:
            raise ValueError("No usable grouped training surfaces")
        selected_ranks = ranks[usable]
        q, iv, b = q[usable], iv[usable], b[usable]
        if (not all(np.isfinite(a).all() for a in (q, iv, b, selected_ranks))
                or np.any(iv <= 0) or np.any(q[..., 2:] < 0)
                or np.any(q[..., 2:] > 1)
                or not np.all(q[..., 2:] == q[:, :1, 2:])
                or np.any(selected_ranks != np.floor(selected_ranks))
                or np.any(selected_ranks < 0) or np.any(selected_ranks > nparam)):
            raise ValueError("Invalid usable training values, parameter groups, or ranks")
        backbone = expected_variance(q, model.factors)
        if not np.isfinite(backbone).all() or np.any(backbone <= 0):
            raise ValueError("Training surfaces require a positive finite variance backbone")
        target_g = np.log(iv / np.sqrt(backbone))
        if not np.isfinite(target_g).all():
            raise ValueError("Nonfinite reference correction")

        super().__init__(model, training, physics, scale, chunk, pde_weight, sensitivity)
        self.surface_chunk = surface_chunk
        self.sq, self.siv, self.sg, self.sb = map(array, (q, iv, target_g, b))
        mx.eval(self.sq, self.siv, self.sg, self.sb)

        def grouped_loss(q, target_iv, target_g, preconditioner, price_weight, bias_weight):
            c, p = coordinates(q.reshape(-1, q.shape[-1]), self.factors, mx)
            g = model.correction(c, p).reshape(target_iv.shape)
            error = stable_iv_error(g, target_g, target_iv)
            price = mx.sum((error / .01)**2) / (count * iv.shape[1])
            shift = (preconditioner @ error[..., None])[..., 0]
            bias = mx.sum(shift**2) / (count * nparam)
            return price_weight * price + bias_weight * bias

        self.surface_vg = mx.compile(nn.value_and_grad(model, grouped_loss),
                                     inputs=model.state, outputs=model.state)
        initial = self.vector()
        base_value, base_gradient = super().__call__(initial)
        price_value, price_gradient = self._surface_value_gradient(.1, 0.)
        bias_value, bias_gradient = self._surface_value_gradient(0., 1.)
        reference_norm = float(np.linalg.norm(base_gradient + price_gradient))
        bias_norm = float(np.linalg.norm(bias_gradient))
        if bias_gradient_share > 0 and bias_norm == 0:
            raise ValueError("Cannot normalize a zero initial quadratic-bias gradient")
        self.bias_coefficient = (bias_gradient_share * reference_norm / bias_norm
                                 if bias_gradient_share > 0 else 0.)
        if not math.isfinite(self.bias_coefficient):
            raise FloatingPointError("Nonfinite frozen bias coefficient")
        self.surface_metadata = {
            'candidate_surfaces': len(usable), 'usable_surfaces': count,
            'excluded_unusable_surfaces': int((~usable).sum()),
            'quotes_per_surface': iv.shape[1], 'training_quotes': int(iv.size),
            'usable_rank_counts': {str(r): int(np.sum(selected_ranks == r))
                                   for r in range(nparam + 1)},
            'surface_chunk': surface_chunk, 'all_usable_surfaces_every_evaluation': True,
            'price_scale': .01, 'group_price_weight': .1,
            'bias_loss': 'mean((B @ IV_error)**2)',
            'bias_gradient_share': float(bias_gradient_share),
            'bias_coefficient': self.bias_coefficient,
            'initial_base_plus_group_price_gradient_norm': reference_norm,
            'initial_unweighted_bias_gradient_norm': bias_norm,
            'initial_weighted_bias_gradient_norm': self.bias_coefficient * bias_norm,
            'initial_base_objective': base_value,
            'initial_weighted_group_price_objective': price_value,
            'initial_unweighted_bias_objective': bias_value,
            'coefficient_policy': 'Computed once at initialization from training gradients; fixed thereafter',
            'residual': 'target_IV * expm1(predicted_g - target_g); target_g uses float64 backbone',
            'interpretation': ('Local linear training proxy, not actual recovery. Truncated singular '
                               'directions receive no bias supervision. No clipping or adaptive weights.'),
        }
        # Initialization normalization is separate from optimizer evaluations.
        self.calls, self.last = 0, None

    def _surface_value_gradient(self, price_weight, bias_weight):
        value = mx.array(0.)
        gradient = mx.zeros((sum(size for _, _, size in self.layout),))
        for start in range(0, len(self.sq), self.surface_chunk):
            sl = slice(start, start + self.surface_chunk)
            val, gr = self.surface_vg(self.sq[sl], self.siv[sl], self.sg[sl], self.sb[sl],
                                      array(price_weight), array(bias_weight))
            value = value + val
            gradient = gradient + mx.concatenate([v.reshape(-1) for _, v in tree_flatten(gr)])
            mx.eval(value, gradient)
        result = float(value), np.asarray(gradient, dtype=float)
        if not math.isfinite(result[0]) or not np.isfinite(result[1]).all():
            raise FloatingPointError("Nonfinite grouped training objective/gradient")
        return result

    def __call__(self, vector):
        base_value, base_gradient = super().__call__(vector)
        value, gradient = self._surface_value_gradient(.1, self.bias_coefficient)
        self.last = base_value + value, base_gradient + gradient
        if not math.isfinite(self.last[0]) or not np.isfinite(self.last[1]).all():
            raise FloatingPointError("Nonfinite combined training objective/gradient")
        return self.last
