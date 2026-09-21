import numpy as np
import pandas as pd
from scripts.mentor_dh_pinn.audit_crisis_option_coverage import parity_gate


def test_heldout_quotes_do_not_enter_carry_estimation():
    strike = np.arange(80., 121., 5.)
    puts = 30. + .1 * strike
    pairs = pd.DataFrame({'CE': puts + .995 * (100 - strike), 'PE': puts}, index=strike)
    before = parity_gate(pairs, 30 / 365)
    assert before is not None
    assert abs(before['forward_from_anchor_parity'] - 100) < 1e-10
    assert abs(before['discount_from_anchor_parity'] - .995) < 1e-10
    corrupt = pairs.copy()
    corrupt.loc[corrupt.index[np.arange(len(pairs)) % 3 == 1], 'CE'] += 1000
    assert parity_gate(corrupt, 30 / 365) == before


def test_impossible_discount_is_rejected_not_repaired():
    k = np.arange(80., 121., 5.)
    pairs = pd.DataFrame({'CE': 50 + .5 * (100 - k), 'PE': np.full(len(k), 50.)}, index=k)
    assert parity_gate(pairs, 30 / 365) is None
