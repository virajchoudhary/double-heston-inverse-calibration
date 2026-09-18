import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amend01 import fit_bs_expiry, predict_bs_expiry
from engine import black

B = {'bs_vol': [.05, 5.]}


def test_one_vol_per_expiry_despite_jittered_tau():
    rng = np.random.default_rng(1); e = np.repeat(['a', 'b', 'c'], 6); x = np.tile(np.linspace(-.2, .2, 6), 3)
    tau = np.repeat([.05, .2, .6], 6) + rng.uniform(0, 1e-4, 18); y = black(x, tau, np.repeat([.5, .7, .9], 6))
    f = fit_bs_expiry(e, x, tau, y, np.ones(18), B)
    assert f['expiries'] == ['a', 'b', 'c'] and len(f['volatility']) == 3
    np.testing.assert_allclose(f['volatility'], [.5, .7, .9], atol=1e-6)


def test_heldout_expiry_interpolates_between_expiry_knots_and_smile_is_flat():
    e = np.repeat(['a', 'b', 'c'], 5); x = np.tile(np.linspace(-.2, .2, 5), 3); tau = np.repeat([.05, .2, .6], 5)
    y = black(x, tau, .6 + .3 * x ** 2 * 10)            # a real smile: BS cannot fit it exactly in-sample
    cal = e != 'b'; f = fit_bs_expiry(e[cal], x[cal], tau[cal], y[cal], np.ones(cal.sum()), B)
    p = predict_bs_expiry(f, e, x, tau)
    assert np.abs(p[cal] - y[cal]).max() > 1e-4           # in-sample error is not zero
    w = np.interp(.2, f['knots'], np.array(f['knots']) * np.array(f['volatility']) ** 2)
    np.testing.assert_allclose(p[~cal], black(x[~cal], .2, np.sqrt(w / .2)), atol=1e-12)
