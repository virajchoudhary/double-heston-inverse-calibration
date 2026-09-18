import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from data_sol import solve_f, parse, clean, forward_price, expiry_of
C = json.loads((HERE / 'config.json').read_text())


def test_parse_decimal_strike():
    assert parse('SOL_USDC-18SEP26-65-C') == ('18SEP26', 65.0, 'C') and parse('SOL_USDC-2OCT26-72d5-P') == ('2OCT26', 72.5, 'P')


@pytest.mark.parametrize('call', [True, False])
def test_forward_round_trip(call):
    F, K, s, t = 2.37, 2.1 if not call else 2.6, .9, .07
    p = F * forward_price(K / F, s, t, call); assert abs(solve_f(p, K, s, t, call, F * 1.01) - F) < 1e-9


def test_clean_on_synthetic_trades():
    base = 1_750_000_000_000; now = dt.datetime.fromtimestamp(base / 1000, dt.timezone.utc); F = 150.0; trades = []
    for code in ['20JUN25', '27JUN25', '25JUL25', '29AUG25', '26SEP25']:
        t = (expiry_of(code) - now).total_seconds() / (365 * 86400)
        for K in np.round(np.linspace(115, 190, 9), 0):
            cp = 'C' if K >= F else 'P'; p = F * float(forward_price(K / F, .7, t, cp == 'C'))
            if p < .2: continue
            trades.append({'instrument_name': f"SOL_USDC-{code}-{str(K).replace('.', 'd')}-{cp}", 'timestamp': base, 'price': p, 'mark_price': p, 'iv': 70., 'index_price': F})
    q, a = clean(trades, '2025-06-15', C)
    assert a['usable'] and not q[q.expiry.eq(q.expiry.max())].heldout_B.any()
    np.testing.assert_allclose(q.market_iv, .7, atol=1e-6); np.testing.assert_allclose(q.forward, F, rtol=1e-8)
