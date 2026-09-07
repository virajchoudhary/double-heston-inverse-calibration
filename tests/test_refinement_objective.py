"""Regression checks for quote weighting, accepted steps and local sensitivity reporting."""
import unittest

import torch
from torch import nn

from src.mentor_dh_pinn.unified import UnifiedCalibrator, quote_scale


class ToyCalibrator(UnifiedCalibrator):
    """Analytic prices isolate optimizer behavior from Fourier quadrature accuracy."""

    def __init__(self, squared=False, invalid_first=False):
        nn.Module.__init__(self)
        self.refine_steps = 4
        self.squared, self.invalid_first = squared, invalid_first

    def _prices(self, z, batch):
        p = z[:, :batch["price"].shape[1]]
        p = p.square() if self.squared else p
        if self.invalid_first:
            bad = torch.zeros_like(p, dtype=torch.bool)
            bad[0, 0] = True
            p = torch.where(bad, torch.full_like(p, float("nan")), p)
        return p


def batch(price):
    p = torch.tensor(price, dtype=torch.float64)
    return {"price": p, "mask": torch.ones_like(p), "spot": torch.ones_like(p),
            "noise_level": torch.full((len(p),), 0.01, dtype=torch.float64),
            "quote_sigma": torch.ones_like(p)}


class RefinementObjectiveTests(unittest.TestCase):
    def test_explicit_quote_noise_controls_objective_and_rejects_bad_scale(self):
        b = batch([[2., 3.]])
        b["quote_sigma"] = torch.tensor([[2., 0.5]], dtype=torch.float64)
        z = torch.zeros((1, 10), dtype=torch.float64)
        L = torch.eye(10, dtype=torch.float64)[None]
        m = ToyCalibrator()
        self.assertEqual(float(m.refinement_objective(z, z, L, b)), 37.)
        self.assertTrue(torch.equal(quote_scale(b), b["quote_sigma"]))
        b["quote_sigma"][0, 0] = 0.
        with self.assertRaises(ValueError):
            quote_scale(b)
        b["mask"][0, 0] = 0.
        self.assertEqual(float(quote_scale(b)[0, 0]), 1.)

    def test_each_surface_descends_and_history_scores_returned_point(self):
        b = batch([[0.04, 0.16], [0.25, 0.36]])
        z0 = torch.full((2, 10), 0.01, dtype=torch.float64)
        L = torch.eye(10, dtype=torch.float64).expand(2, -1, -1)
        m = ToyCalibrator(squared=True)
        previous = m.refinement_objective(z0, z0, L, b, prior_weight=0.)
        for n in range(1, 7):
            with torch.no_grad():
                z, history = m.refine(z0, L, b, steps=n, prior_weight=0.)
                actual = m.refinement_objective(z, z0, L, b, prior_weight=0.)
            self.assertTrue(bool((actual <= previous).all()), (previous, actual))
            self.assertAlmostEqual(history[-1], float(actual.mean()), places=14)
            self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))
            previous = actual
        self.assertLess(float(actual.max()), 1e-9)

    def test_one_failed_active_quote_freezes_entire_surface(self):
        b = batch([[0.2, 0.3], [0.2, 0.3]])
        z0 = torch.zeros((2, 10), dtype=torch.float64)
        L = torch.eye(10, dtype=torch.float64).expand(2, -1, -1)
        m = ToyCalibrator(invalid_first=True)
        with torch.no_grad():
            z, _ = m.refine(z0, L, b, steps=3, prior_weight=0.)
            objective = m.refinement_objective(z, z0, L, b, prior_weight=0.)
        self.assertTrue(torch.equal(z[0], z0[0]))
        self.assertTrue(torch.isinf(objective[0]))
        self.assertLess(float(objective[1]), 1e-12)

    def test_gradients_survive_accepted_steps_and_holdout_is_unused(self):
        b = batch([[0.2, 0.3]])
        z0 = torch.zeros((1, 10), dtype=torch.float64, requires_grad=True)
        L = torch.eye(10, dtype=torch.float64)[None].requires_grad_()
        m = ToyCalibrator()
        z, _ = m.refine(z0, L, b, steps=2, create_graph=True)
        z.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(z0.grad).all()))
        self.assertTrue(bool(torch.isfinite(L.grad).all()))
        b["holdout_price"] = torch.tensor([[float("nan"), 1e100]])
        b["params"] = torch.full((1, 10), float("nan"))
        b["clean"] = torch.full_like(b["price"], float("nan"))
        with torch.no_grad():
            z2, _ = m.refine(z0.detach(), L.detach(), b, steps=2)
        torch.testing.assert_close(z.detach(), z2, atol=1e-14, rtol=1e-14)

    def test_svd_reports_scaled_local_rank_and_explicit_weak_basis(self):
        b = batch([[0.2, 0.3]])
        b["quote_sigma"][0, 1] = 1e4
        z = torch.zeros((1, 10), dtype=torch.float64)
        d = ToyCalibrator().local_identifiability(z, b, threshold=1e-6)[0]
        self.assertEqual(d["rank"], 1)
        self.assertEqual(len(d["weak_basis"]), 9)
        self.assertAlmostEqual(d["relative_information"][1], 1e-8)
        self.assertEqual(d["status"], "local_practical_sensitivity")
        scale = torch.ones(10, dtype=torch.float64)
        scale[1] = 1e4
        self.assertEqual(ToyCalibrator().local_identifiability(z, b, scale)[0]["rank"], 2)

    def test_exact_fourier_refinement_retains_differentiable_path(self):
        from src.mentor_dh_pinn.params_v2 import encode
        m = UnifiedCalibrator(d_model=16, enc_blocks=0, rounds=1, node_count=48)
        truth = torch.tensor(encode([1., .09, .3, -.5, .08, 5., .06, .4, -.2, .05]),
                             dtype=torch.float64)[None]
        b = batch([[0.] * 9])
        b.update(strike=torch.tensor([[.9, 1., 1.1] * 3], dtype=torch.float64),
                 tau=torch.tensor([[.1] * 3 + [.5] * 3 + [1.] * 3], dtype=torch.float64),
                 rate=torch.full((1, 9), .05, dtype=torch.float64),
                 carry=torch.full((1, 9), .01, dtype=torch.float64))
        with torch.no_grad():
            b["price"] = m._prices(truth, b)
        b["quote_sigma"] = torch.full_like(b["price"], .001)
        mu = (truth + .02).requires_grad_()
        L = (torch.eye(10, dtype=torch.float64)[None] * .5).requires_grad_()
        z, _ = m.refine(mu, L, b, steps=1, create_graph=True)
        z.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(mu.grad).all()))
        self.assertTrue(bool(torch.isfinite(L.grad).all()))
        with torch.no_grad():
            inference, _ = m.refine(mu.detach(), L.detach(), b, steps=1)
        torch.testing.assert_close(z.detach(), inference, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
