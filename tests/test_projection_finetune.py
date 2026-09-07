"""Analytic posterior checks and failure handling; no checkpoint or market data needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mentor_dh_pinn.finetune_projection import ProjectionFineTuner


DTYPE = torch.float64


def model(d=2, **kw):
    def pricer(params, spot, strike, tau, rate, carry, **kwargs):
        return params[:, :1] * 0 + torch.zeros_like(spot)
    return ProjectionFineTuner(nn.Identity(), lambda z: list(z.unbind(-1)), pricer,
                              torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE), **kw)


def batch():
    t = lambda values: torch.tensor([values], dtype=DTYPE)
    return {"spot": t([1., float("nan")]), "strike": t([1., 0.]),
            "tau": t([.1, float("nan")]), "rate": t([0., float("nan")]),
            "carry": t([0., float("nan")]), "price": t([0., float("nan")]),
            "vega": t([1., float("nan")]), "quote_sigma": t([1., float("nan")]),
            "mask": t([1., 0.])}


class ProjectionFineTuneTests(unittest.TestCase):
    def test_flat_likelihood_width_has_correct_optimum_and_collapse_barrier(self):
        m, b = model(10), batch()
        values = []
        for scale in (.1, 1., 2.):
            log_scale = torch.tensor(scale, dtype=DTYPE).log().requires_grad_()
            mu = torch.zeros((1, 10), dtype=DTYPE)
            L = torch.eye(10, dtype=DTYPE).unsqueeze(0) * log_scale.exp()
            loss, logs = m.self_consistency_loss(b, mu, L)
            slope = torch.autograd.grad(loss, log_scale)[0]
            self.assertAlmostEqual(float(loss.detach()), 5 * (scale**2 - 1 - 2 * torch.log(
                torch.tensor(scale, dtype=DTYPE)).item()), places=10)
            self.assertAlmostEqual(float(slope), 10 * (scale**2 - 1), places=10)
            self.assertEqual(float(logs["sc_expected_nll"]), 0.)
            values.append(float(loss.detach()))
        self.assertEqual(values[1], 0.)
        L = torch.eye(10, dtype=DTYPE).unsqueeze(0) * 1e-12
        tiny, _ = m.self_consistency_loss(b, torch.zeros((1, 10), dtype=DTYPE), L)
        self.assertGreater(float(tiny), values[0] + 200.)

    def test_gaussian_kl_matches_distribution_reference_with_correlations(self):
        m = model(2)
        m.prior_mean.copy_(torch.tensor([.2, -.3], dtype=DTYPE))
        m.prior_sd.copy_(torch.tensor([2., .7], dtype=DTYPE))
        mu = torch.tensor([[.1, .6]], dtype=DTYPE, requires_grad=True)
        L = torch.tensor([[[.4, 0.], [.2, .9]]], dtype=DTYPE, requires_grad=True)
        loss, _ = m.self_consistency_loss(batch(), mu, L)
        q = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        prior = torch.distributions.MultivariateNormal(m.prior_mean,
                                                       scale_tril=torch.diag(m.prior_sd))
        reference = torch.distributions.kl_divergence(q, prior).mean()
        torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
        for g in torch.autograd.grad(loss, (mu, L)):
            self.assertTrue(torch.isfinite(g).all())

    def test_informative_direction_shrinks_to_known_posterior_flat_direction_does_not(self):
        # z~N(0,I), y=2*z[0]+N(0,.5²): posterior variance is (1/17, 1).
        m, b = model(2, n_consistency_draws=2), batch()
        m._price = lambda z, _: (2 * z[:, :1]).expand(-1, 2)
        b["quote_sigma"][0, 0] = .5
        mu = torch.zeros((1, 2), dtype=DTYPE, requires_grad=True)
        for factor in (.5, 1., 2.):
            log_sd = torch.tensor([factor / 17**.5, 1.], dtype=DTYPE).log().requires_grad_()
            L = torch.diag_embed(log_sd.exp()).unsqueeze(0)
            # These exact second moments remove Monte Carlo tolerance from the test.
            eps = [torch.ones_like(mu), -torch.ones_like(mu)]
            with patch("torch.randn_like", side_effect=eps):
                loss, _ = m.self_consistency_loss(b, mu, L)
            grad_mu, grad_width = torch.autograd.grad(loss, (mu, log_sd))
            torch.testing.assert_close(grad_mu, torch.zeros_like(mu), atol=1e-12, rtol=0)
            torch.testing.assert_close(grad_width,
                                       torch.tensor([factor**2 - 1., 0.], dtype=DTYPE),
                                       atol=1e-12, rtol=1e-12)

    def test_invalid_observed_prediction_rejects_whole_batch(self):
        m, b = model(), batch()
        m._price = lambda z, _: torch.full((1, 2), float("nan"), dtype=DTYPE)
        mu = torch.zeros((1, 2), dtype=DTYPE)
        L = torch.eye(2, dtype=DTYPE).unsqueeze(0)
        projection, p = m.projection_loss(b, mu)
        elbo, e = m.self_consistency_loss(b, mu, L)
        self.assertTrue(torch.isinf(projection) and torch.isinf(p["proj_iv_rmse"]))
        self.assertTrue(torch.isinf(elbo))
        self.assertEqual(float(p["proj_unpriceable"]), 1.)
        self.assertEqual(float(e["sc_invalid"]), 1.)

    def test_invalid_padding_is_excluded_before_pricing_and_arithmetic(self):
        m, b = model(), batch()
        def pricer(params, spot, strike, tau, rate, carry, **kwargs):
            self.assertTrue(all(torch.isfinite(t).all()
                                for t in (spot, strike, tau, rate, carry)))
            return params[:, :1] + torch.log(spot / strike)
        m.exact_fourier_pricer = pricer
        mu = torch.zeros((1, 2), dtype=DTYPE, requires_grad=True)
        L = torch.eye(2, dtype=DTYPE).unsqueeze(0).requires_grad_()
        projection, _ = m.projection_loss(b, mu)
        elbo, _ = m.self_consistency_loss(b, mu, L)
        self.assertTrue(torch.isfinite(projection + elbo))
        for g in torch.autograd.grad(projection + elbo, (mu, L)):
            self.assertTrue(torch.isfinite(g).all())

    def test_empty_or_invalid_observed_quotes_are_not_successful_losses(self):
        mu, L = torch.zeros((1, 2), dtype=DTYPE), torch.eye(2, dtype=DTYPE).unsqueeze(0)
        for key, value in (("mask", 0.), ("price", float("nan"))):
            m, b = model(), batch()
            b[key][0, 0] = value
            self.assertTrue(torch.isinf(m.projection_loss(b, mu)[0]))
            self.assertTrue(torch.isinf(m.self_consistency_loss(b, mu, L)[0]))

    def test_disabled_objectives_are_not_executed(self):
        m = model(w_anchor=0., w_projection=0., w_self_consistency=0., n_consistency_draws=0)
        with patch.object(m, "_encode", side_effect=AssertionError("disabled")), \
             patch.object(m, "recovery_loss", side_effect=AssertionError("disabled")):
            loss, _ = m.compute_losses({}, {})
        self.assertEqual(float(loss), 0.)
        self.assertEqual(m._loss_ema, {})
        m = model(w_anchor=1., w_projection=0., w_self_consistency=0., n_consistency_draws=0)
        with patch.object(m, "_encode", side_effect=AssertionError("disabled")), \
             patch.object(m, "recovery_loss", return_value=(torch.tensor(2.), {})):
            loss, _ = m.compute_losses({}, {})
        self.assertEqual(float(loss), 1.)

    def test_nonfinite_loss_cannot_poison_ema_and_scale_uses_absolute_magnitude(self):
        m = model(w_anchor=1., w_projection=0., w_self_consistency=0.)
        m._balance("anchor", torch.tensor(-2.))
        for invalid in (float("nan"), float("inf")):
            with patch.object(m, "recovery_loss", return_value=(torch.tensor(invalid), {})):
                loss, _ = m.compute_losses({}, {})
            self.assertTrue(torch.isinf(loss))
            self.assertEqual(m._loss_ema, {"anchor": 2.})
        self.assertEqual(float(m._balance("anchor", torch.tensor(2.))), 1.)
        self.assertEqual(m._loss_ema["anchor"], 2.)


if __name__ == "__main__":
    unittest.main()
