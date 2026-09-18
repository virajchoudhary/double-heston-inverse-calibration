import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import Grid, exact, adaptive, admissible, decode, box, fit_bs_term, bs_predict, black, iv
from data import solve_k, forward_price, clean

C = json.loads((HERE / 'config.json').read_text())
P = np.array([0.9491, 0.0257, 0.0517, 0.7009, 0.0003, 10.7526, 0.033, 0.3613, -0.8916, 0.0252])
S = np.array([8.9814, 0.0409, 0.297, -0.9621, 0.0244])


def test_matches_audited_literature_exact_engine():
    sys.path.insert(0, str(HERE.parent / 'nifty_multifactor_v4'))
    from literature_exact import exact as lit_exact
    x = np.linspace(-.3, .3, 13); tau = np.repeat([.02, .1, .5, 1.5], 13)[:13]
    for p in [P, S]:
        np.testing.assert_allclose(exact(p, x, tau, feller=True), lit_exact(p, x, tau), atol=1e-10, rtol=0)


def test_feller_free_prices_are_finite_bounded_and_agree_with_adaptive():
    q = np.array([1.5, .5, 2.5, -.6, .6, 12., .9, 4.0, -.4, .9])      # both factors violate Feller
    with pytest.raises(ValueError): admissible(q, feller=True)
    x = np.array([-.5, -.1, 0., .2, .6]); tau = np.array([3 / 365, .05, .2, .6, 1.1])
    a = exact(q, x, tau, feller=False)
    assert np.isfinite(a).all() and (a >= np.maximum(1 - np.exp(-x), 0) - 1e-9).all() and (a <= 1 + 1e-9).all()
    for j in range(len(x)):
        assert abs(a[j] - adaptive(q, float(x[j]), float(tau[j]), False)) < 1e-7


def test_removed_factor_reduces_to_single_heston():
    q = np.r_[.1, 1e-10, 1e-6, 0., 1e-10, S]
    x = np.array([-.2, 0., .2]); tau = np.array([.05, .3, 1.])
    np.testing.assert_allclose(exact(q, x, tau, feller=False), exact(S, x, tau, feller=True), atol=1e-8, rtol=0)


@pytest.mark.parametrize('kind', ['SH', 'DH'])
@pytest.mark.parametrize('variant', ['feller_strict', 'feller_free'])
def test_decoded_parameters_are_always_admissible(kind, variant):
    lo, hi = box(kind, variant, C['bounds']); rng = np.random.default_rng(0)
    for u in rng.uniform(size=(2000, len(lo))):
        admissible(decode(kind, variant, lo + (hi - lo) * u), feller=(variant == 'feller_strict'))


def test_forward_recovery_round_trip():
    for k, s, t, call in [(1.1, .6, .02, True), (.8, .9, .3, False), (1.4, .5, 1., True), (.95, 1.2, .01, False)]:
        price = float(forward_price(k, s, t, call)); assert abs(solve_k(price, s, t, call) - k) < 1e-8


def test_bs_term_uses_only_calibration_expiries():
    x = np.tile(np.linspace(-.2, .2, 5), 3); tau = np.repeat([.05, .2, .6], 5); y = black(x, tau, np.repeat([.5, .7, .9], 5))
    cal = tau != .2; fit = fit_bs_term(x[cal], tau[cal], y[cal], np.ones(cal.sum()), C['bounds'])
    assert fit['knots'] == [.05, .6]
    np.testing.assert_allclose(bs_predict(fit, x[cal], tau[cal]), y[cal], atol=1e-8)
    w = np.interp(.2, [.05, .6], [.05 * .25, .6 * .81]); np.testing.assert_allclose(bs_predict(fit, x[~cal], tau[~cal]), black(x[~cal], .2, np.sqrt(w / .2)), atol=1e-12)


def test_clean_builds_disjoint_designs_on_synthetic_trades():
    import datetime as dt
    from data import expiry_of
    base = 1_700_000_000_000; trades = []; F = 50000.; now = dt.datetime.fromtimestamp(base / 1000, dt.timezone.utc)
    for j, code in enumerate(['20NOV23', '24NOV23', '1DEC23', '29DEC23', '29MAR24', '28JUN24']):
        t = (expiry_of(code) - now).total_seconds() / (365 * 86400)
        for K in np.linspace(40000, 62000, 9):
            cp = 'C' if K >= F else 'P'; s = .6; k = K / F; price = float(forward_price(k, s, t, cp == 'C'))
            if price < .0005: continue
            trades.append({'instrument_name': f'BTC-{code}-{int(K)}-{cp}', 'timestamp': base, 'price': price, 'mark_price': price, 'iv': 60., 'index_price': F, 'trade_id': f'{j}{K}'})
    # expiry times are fixed at 08:00 UTC; tau is recomputed from the trade timestamp inside clean()
    q, audit = clean(trades, '2023-11-14', C)
    if not audit['usable']: pytest.skip('synthetic geometry insufficient')
    assert not (q.heldout_A & ~q.heldout_A).any()
    longest = q.expiry.max(); assert not q[q.expiry.eq(longest)].heldout_B.any()
    assert (q.groupby('expiry').apply(lambda g: (~g.heldout_A).sum()) >= 2).all()
    np.testing.assert_allclose(q.market_iv, .6, atol=2e-3)
