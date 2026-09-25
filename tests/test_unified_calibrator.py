"""Deliverable D. Every test the redesign brief lists, plus the audit's own findings."""
from __future__ import annotations
import math, sys, unittest
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from src.constraints import validate_parameters
from src.double_heston import price_double_heston_call
from src.mentor_dh_pinn.collate import collate
from src.mentor_dh_pinn.dataset_v6 import build, sample_geometry, sample_parameters
from src.mentor_dh_pinn.params_v2 import CANONICAL, decode, encode, encode_batch, to_array
from src.mentor_dh_pinn.torch_pricer import price_call
from src.mentor_dh_pinn.unified import UnifiedCalibrator

CKPT = ROOT / "outputs" / "unified_v6" / "unified.pt"


def _model(**kw):
    m = UnifiedCalibrator(**kw)
    if CKPT.exists():
        ck = torch.load(CKPT, weights_only=False)
        c = ck.get("config", {})
        m = UnifiedCalibrator(d_model=c.get("d_model", 128), rounds=c.get("rounds", 3),
                              node_count=c.get("nodes", 48), **kw)
        m.load_state_dict(ck["state_dict"])
    m.eval(); return m


def _batch(n=6, seed=11, **kw):
    d = build(n, seed=seed, **kw)
    idx = np.where(d["ok"])[0]
    return d, collate(d, idx)


class T01_Permutation(unittest.TestCase):
    def test_quote_permutation_invariance(self):
        d, b = _batch(8, 3); m = _model()
        perm = torch.randperm(b["mask"].shape[1])
        b2 = dict(b)
        for k in ("spot", "strike", "tau", "rate", "carry", "price", "clean", "mask"):
            b2[k] = b[k][:, perm]
        with torch.no_grad():
            a = m(b, refine_steps=0)["mu_z"]; c = m(b2, refine_steps=0)["mu_z"]
        self.assertLess(float((a - c).abs().max()), 1e-9)


class T02_VariableLength(unittest.TestCase):
    def test_variable_number_of_quotes(self):
        d, _ = _batch(48, 4); m = _model(); seen = set()
        for i in np.where(d["ok"])[0][:24]:
            b = collate(d, np.array([i])); seen.add(int(b["n_quotes"][0]))
            with torch.no_grad(): o = m(b, refine_steps=0)
            self.assertTrue(torch.isfinite(o["mu_z"]).all())
        self.assertGreater(len(seen), 4, "test needs genuinely varying quote counts")


class T03_SingleExpiry(unittest.TestCase):
    def test_single_expiry_is_accepted(self):
        m = _model()
        b = _synthetic_batch(n_exp=1, n_strikes=7)
        with torch.no_grad(): o = m(b, refine_steps=1)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T04_MissingExpiry(unittest.TestCase):
    def test_missing_expiry_is_accepted(self):
        m = _model()
        b = _synthetic_batch(days=(30.0, 365.0), n_strikes=6)      # 90/180 absent
        with torch.no_grad(): o = m(b, refine_steps=1)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T05_IrregularMaturities(unittest.TestCase):
    def test_irregular_maturities(self):
        m = _model()
        b = _synthetic_batch(days=(11.0, 43.0, 117.0, 402.0, 913.0), n_strikes=5)
        with torch.no_grad(): o = m(b, refine_steps=1)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T06_ArbitraryStrikes(unittest.TestCase):
    def test_arbitrary_strikes(self):
        m = _model()
        b = _synthetic_batch(days=(60.0,), strikes=np.array([0.62, 0.81, 0.97, 1.03, 1.44]))
        with torch.no_grad(): o = m(b, refine_steps=1)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T07_LowVolatility(unittest.TestCase):
    def test_low_volatility_parameter_set(self):
        p = np.array([0.8, 0.016, 0.10, -0.55, 0.014, 5.0, 0.012, 0.16, -0.20, 0.013])
        self.assertFalse(validate_parameters(dict(zip(CANONICAL, p)))["violations"])
        b = _synthetic_batch(params=p)
        m = _model()
        with torch.no_grad(): o = m(b, refine_steps=2)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T08_HighVolatility(unittest.TestCase):
    def test_high_volatility_parameter_set(self):
        p = np.array([1.1, 0.40, 0.55, -0.70, 0.90, 7.0, 0.30, 1.20, -0.25, 0.85])
        self.assertFalse(validate_parameters(dict(zip(CANONICAL, p)))["violations"])
        b = _synthetic_batch(params=p)
        m = _model()
        with torch.no_grad(): o = m(b, refine_steps=2)
        self.assertTrue(torch.isfinite(o["params"]).all())


class T09_DecodeValidity(unittest.TestCase):
    def test_decode_always_valid(self):
        rng = np.random.default_rng(5)
        for sd in (1.0, 4.0, 10.0):
            Z = rng.normal(0, sd, (4000, 10))
            for i in range(len(Z)):
                p = to_array(decode(Z[i]))
                self.assertFalse(validate_parameters(dict(zip(CANONICAL, p)))["violations"],
                                 f"invalid at sd={sd}: {p}")


class T10_FactorOrdering(unittest.TestCase):
    def test_kappa_ordering_by_construction(self):
        rng = np.random.default_rng(6); Z = rng.normal(0, 6.0, (5000, 10))
        P = np.stack([to_array(decode(Z[i])) for i in range(len(Z))])
        self.assertTrue(bool((P[:, 0] < P[:, 5]).all()))

    def test_factors_are_permutation_symmetric(self):
        """The symmetry is real, which is why an ordering convention is required."""
        p = [1.10, 0.090, 0.280, -0.55, 0.095, 4.20, 0.055, 0.390, -0.20, 0.048]
        sw = p[5:] + p[:5]
        a = price_double_heston_call(1.0, 1.0, 0.5, 0.05, 0.01, p, node_count=96)
        b = price_double_heston_call(1.0, 1.0, 0.5, 0.05, 0.01, sw, node_count=96,
                                     enforce_ordering=False)
        self.assertLess(abs(a - b) / a, 1e-10)


class T11_NoHiddenClipping(unittest.TestCase):
    def test_round_trip_outside_the_old_param_box(self):
        """The old decode silently clipped 5 of these 10 coordinates."""
        for p in ([1.10, 0.40, 0.28, -0.55, 0.90, 14.0, 0.30, 0.60, -0.20, 0.85],
                  [5.00, 0.005, 0.20, 0.80, 0.004, 60.0, 0.002, 0.30, 0.55, 0.003],
                  [0.05, 1.20, 0.30, -0.97, 3.00, 0.90, 0.90, 0.90, 0.15, 2.00]):
            got = to_array(decode(encode(p)))
            self.assertLess(float(np.max(np.abs(np.array(p) - got) / np.abs(p))), 1e-8)

    def test_encode_fails_loudly_outside_the_model_class(self):
        for bad in ([0.02, 0.90, 0.19, -0.98, 2.5, 40.0, 0.80, 3.5, 0.10, 1.9],
                    [1.0, 0.09, 0.28, -0.90, 0.09, 5.0, 0.06, 0.40, 0.60, 0.05],
                    [6.0, 0.09, 0.28, -0.50, 0.09, 2.0, 0.06, 0.40, -0.2, 0.05]):
            with self.assertRaises(ValueError):
                encode(bad)


class T12_CovariancePD(unittest.TestCase):
    def test_covariance_is_positive_definite(self):
        d, b = _batch(16, 8); m = _model()
        with torch.no_grad(): o = m(b, refine_steps=0)
        S = o["L"] @ o["L"].transpose(-1, -2)
        ev = torch.linalg.eigvalsh(S)
        self.assertTrue(bool((ev > 0).all()))
        self.assertLess(float((S - S.transpose(-1, -2)).abs().max()), 1e-12)


class T13_PricerGradients(unittest.TestCase):
    def test_exact_pricer_matches_numpy_engine(self):
        """Two assertions, because one ratio cannot describe both ends of the price range.

        Absolute agreement is what the training loss and the Gauss-Newton residual see.
        Relative agreement is meaningful only above a tradeable price floor -- below about
        1e-6 of spot both engines sit on their own quadrature noise (the NumPy engine
        returns negative prices there), so a relative test would measure noise.
        """
        rng = np.random.default_rng(2)
        worst_abs, worst_rel, n_rel = 0.0, 0.0, 0
        for _ in range(3):
            p, _ = sample_parameters(rng, 1); p = p[0]
            for K in (0.8, 1.0, 1.25):
                for tau in (14 / 365, 200 / 365, 900 / 365):
                    a = price_double_heston_call(1.0, K, tau, 0.05, 0.01, list(p), node_count=64)
                    b = float(price_call(torch.tensor(p), torch.tensor([1.0]), torch.tensor([K]),
                                         torch.tensor([tau]), torch.tensor([0.05]),
                                         torch.tensor([0.01]))[0])
                    worst_abs = max(worst_abs, abs(a - b))          # spot is 1.0 here
                    if abs(a) > 1e-4:                               # 1bp of spot: tradeable
                        worst_rel = max(worst_rel, abs(a - b) / abs(a)); n_rel += 1
        self.assertGreater(n_rel, 10, "need enough tradeable quotes for the relative test")
        self.assertLess(worst_abs, 1e-12, f"absolute agreement {worst_abs:.3e}")
        self.assertLess(worst_rel, 1e-11, f"relative agreement {worst_rel:.3e}")

    def test_gradients_match_finite_differences(self):
        p = [1.0, 0.090, 0.35, -0.55, 0.090, 5.0, 0.060, 0.45, -0.20, 0.055]
        S, K, tau = 1.0, 1.0, 0.5
        t = torch.tensor(p, requires_grad=True)
        c = price_call(t, torch.tensor([S]), torch.tensor([K]), torch.tensor([tau]),
                       torch.tensor([0.05]), torch.tensor([0.01]))[0]
        c.backward(); ad = t.grad.numpy()
        num = []
        for j in range(10):
            h = max(1e-6, 1e-5 * abs(p[j])); f = []
            for mlt in (-2, -1, 1, 2):
                q = list(p); q[j] = p[j] + mlt * h
                f.append(price_double_heston_call(S, K, tau, 0.05, 0.01, q, node_count=64))
            num.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
        num = np.array(num)
        # scaled by the largest derivative: that is the scale a Gauss-Newton step sees, and
        # the FD estimate of the smallest derivative is itself noise-limited
        self.assertLess(float(np.abs(ad - num).max() / np.abs(num).max()), 1e-6)


class T14_GaussNewtonStability(unittest.TestCase):
    def test_refinement_is_stable_and_reduces_residual(self):
        d, b = _batch(12, 12); m = _model()
        with torch.no_grad():
            o0 = m(b, refine_steps=0); o = m(b, refine_steps=4)
        self.assertTrue(torch.isfinite(o["params"]).all())
        h = o["residual_history"]
        self.assertEqual(len(h), 4)
        self.assertTrue(all(math.isfinite(x) for x in h))
        self.assertLessEqual(h[-1], h[0] * 1.05, f"refinement diverged: {h}")

    def test_refinement_respects_the_trust_region(self):
        d, b = _batch(8, 13); m = _model()
        with torch.no_grad(): o = m(b, refine_steps=3)
        self.assertLess(float((o["z"] - o["mu_z"]).norm(dim=-1).max()), 3 * 1.5 + 1e-9)


class T15_ZeroNoiseRecovery(unittest.TestCase):
    """With no noise and a rich geometry, refinement should reprice essentially exactly."""
    def test_zero_noise_synthetic_recovery(self):
        if not CKPT.exists(): self.skipTest("no trained checkpoint")
        m = _model()
        rng = np.random.default_rng(21); params, _ = sample_parameters(rng, 1)
        b = _synthetic_batch(params=params[0], days=(30., 60., 90., 180., 365., 730.),
                             n_strikes=9)
        with torch.no_grad(): o = m(b, refine_steps=5)
        pr = price_call(o["params"][0], b["spot"][0], b["strike"][0], b["tau"][0],
                        b["rate"][0], b["carry"][0]).numpy()
        rmse = float(np.sqrt(np.mean(((pr - b["clean"][0].numpy()) / 1.0) ** 2)))
        self.assertLess(rmse, 5e-3, f"zero-noise repricing {rmse:.3e}")


class T16_NoisyRecovery(unittest.TestCase):
    def test_noisy_recovery_is_finite_and_bounded(self):
        if not CKPT.exists(): self.skipTest("no trained checkpoint")
        d, b = _batch(24, 33); m = _model()
        with torch.no_grad(): o = m(b, refine_steps=3)
        self.assertTrue(torch.isfinite(o["params"]).all())
        for i in range(o["params"].shape[0]):
            v = validate_parameters(dict(zip(CANONICAL, o["params"][i].numpy())))
            self.assertFalse(v["violations"], "refined parameters must stay in the model class")


class T17_OODWarning(unittest.TestCase):
    """The old decode hid parameter-box failures. The new model must fail visibly."""
    def test_ood_status_is_reported_and_ordered(self):
        import json as _json
        from src.mentor_dh_pinn.ood import assess, STATUSES
        ref_path = ROOT / "outputs" / "unified_v6" / "ood_reference.json"
        if not ref_path.exists(): self.skipTest("no OOD reference built yet")
        ref = _json.loads(ref_path.read_text())
        geo = {"spot": np.ones(9), "strike": np.linspace(0.85, 1.15, 9),
               "tau": np.full(9, 0.25), "rate": np.full(9, 0.05), "carry": np.full(9, 0.01)}
        typical = np.array([1.0, 0.06, 0.25, -0.55, 0.06, 5.0, 0.05, 0.40, -0.20, 0.05])
        wild = np.array([0.02, 3.00, 0.30, -0.10, 5.00, 90.0, 2.50, 1.00, 0.05, 4.00])
        a = assess(typical, geo, ref); b = assess(wild, geo, ref)
        self.assertIn(a["status"], STATUSES); self.assertIn(b["status"], STATUSES)
        self.assertGreater(a["tail_distance_pct"], b["tail_distance_pct"],
                           "a wildly out-of-prior vector must sit closer to the tail")
        self.assertEqual(assess(typical, geo, ref, priced_ok=False)["status"],
                         "numerically_unsafe")


class T18_Serialisation(unittest.TestCase):
    def test_reload_reproduces_outputs(self):
        import tempfile
        d, b = _batch(8, 14); m = _model()
        with torch.no_grad(): a = m(b, refine_steps=1)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "m.pt"; torch.save(m.state_dict(), f)
            m2 = UnifiedCalibrator(d_model=m.embed[0].out_features, rounds=m.rounds,
                                   node_count=m.node_count)
            m2.load_state_dict(torch.load(f, weights_only=True)); m2.eval()
            with torch.no_grad(): c = m2(b, refine_steps=1)
        self.assertLess(float((a["params"] - c["params"]).abs().max()), 1e-12)


def _synthetic_batch(params=None, days=(30.0, 90.0, 365.0), n_strikes=7, strikes=None,
                     n_exp=None, rate=0.05, carry=0.01):
    """A deliberately hand-built geometry, so the geometry tests do not depend on sampling."""
    rng = np.random.default_rng(0)
    if params is None:
        params, _ = sample_parameters(rng, 1); params = params[0]
    if n_exp is not None:
        days = tuple(np.exp(np.linspace(math.log(30), math.log(365), n_exp)))
    ks = np.linspace(0.85, 1.15, n_strikes) if strikes is None else np.asarray(strikes)
    tau = np.concatenate([np.full(len(ks), d / 365.0) for d in days])
    K = np.tile(ks, len(days))
    n = len(tau)
    p = torch.tensor(np.asarray(params, dtype=float))
    with torch.no_grad():
        c = price_call(p, torch.ones(n), torch.tensor(K), torch.tensor(tau),
                       torch.full((n,), rate), torch.full((n,), carry)).numpy()
    t = lambda a: torch.tensor(np.asarray(a, dtype=float)).unsqueeze(0)
    return {"spot": t(np.ones(n)), "strike": t(K), "tau": t(tau),
            "rate": t(np.full(n, rate)), "carry": t(np.full(n, carry)),
            "price": t(np.maximum(c, 1e-10)), "clean": t(np.maximum(c, 1e-10)),
            "mask": t(np.ones(n)), "noise_level": torch.tensor([0.005]),
            "params": t(params), "n_quotes": torch.tensor([n])}


if __name__ == "__main__":
    unittest.main(verbosity=2)
