"""Regression checks for quote leakage, contract filtering, and portable corpus packing."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.mentor_dh_pinn.build_real_corpus import _pack, _stratified_indices, save, stock_surfaces
from src.mentor_dh_pinn import nifty_panel as nifty


def quote_fixture():
    rows = []
    tau = 91/365
    forward, discount = 100*np.exp(.04*tau), np.exp(-.05*tau)
    for strike in np.arange(80., 122., 2.):
        call = discount*nifty.black76(forward, strike, tau, .25)
        put = call-discount*(forward-strike)
        for opt, price in (("CE", call), ("PE", put)):
            rows.append({"TradDt": "2025-01-01", "XpryDt": "2025-04-02",
                         "TckrSymb": "NIFTY", "OptnTp": opt, "StrkPric": strike,
                         "SttlmPric": price, "UndrlygPric": 100.,
                         "TtlTradgVol": 10., "OpnIntrst": 10000., "dte": 91})
    return pd.DataFrame(rows)


def test_nifty_carry_ignores_both_options_at_heldout_strikes():
    quotes = quote_fixture()
    with patch.object(nifty, "read_quotes", return_value=quotes):
        original = nifty.surface("2025-01-01", holdout_fold=2)
    held_strikes = set(original.loc[original.holdout_mask, "strike"])
    altered = quotes.copy()
    altered.loc[altered.StrkPric.isin(held_strikes) & altered.OptnTp.eq("CE"), "SttlmPric"] += .1
    with patch.object(nifty, "read_quotes", return_value=altered):
        result = nifty.surface("2025-01-01", holdout_fold=2)
    for key in ("forward", "discount", "rate", "dividend", "parity_nrmse"):
        np.testing.assert_array_equal(original[key], result[key])
    assert result.carry_fit_scope.eq("calibration_strikes").all()
    assert result.holdout_mask.any() and (~result.holdout_mask).any()


def test_read_quotes_rejects_conflicts_and_filters_individual_open_interest(tmp_path):
    quotes = quote_fixture().drop(columns="dte")
    quotes.loc[0, "OpnIntrst"] = 9999
    path = nifty.bhav_path("2025-01-01", tmp_path)
    path.parent.mkdir()
    quotes.to_csv(path, index=False, compression="zip")
    result = nifty.read_quotes("2025-01-01", bhav_dir=tmp_path)
    assert len(result) == len(quotes)-1
    assert result.OpnIntrst.ge(10000).all()
    conflict = quotes.iloc[[1]].copy()
    conflict.SttlmPric += .1
    pd.concat([quotes, conflict]).to_csv(path, index=False, compression="zip")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        nifty.read_quotes("2025-01-01", bhav_dir=tmp_path)


def packed_fixture():
    strikes = np.tile(np.linspace(.85, 1.15, 12), 3)
    tau = np.repeat([.1, .3, 1.], 12)
    iv = .25 + .2*np.square(np.log(strikes))
    prices = np.array([nifty.black76(1., k, t, v) for k, t, v in zip(strikes, tau, iv)])
    held = np.arange(len(tau)) % 3 == 2
    return np.ones(len(tau)), strikes, tau, prices, iv, held


def test_heldout_price_iv_poison_does_not_change_calibration_weights_or_noise():
    fwd, strike, tau, price, iv, held = packed_fixture()
    original = _pack(fwd, strike, tau, price, iv, "FIXTURE|2025-01-01", held)
    iv2, price2 = iv.copy(), price.copy()
    iv2[held] *= 1.5
    price2[held] *= 1.01
    changed = _pack(fwd, strike, tau, price2, iv2, "FIXTURE|2025-01-01", held)
    for key in ("vega", "quote_sigma", "iv_noise", "spot", "strike", "holdout_mask"):
        np.testing.assert_array_equal(original[key], changed[key])
    np.testing.assert_array_equal(original["price"][~held], changed["price"][~held])


def test_quote_cap_preserves_every_expiry_and_fold():
    tau = np.repeat(np.arange(1., 13.), 30)
    strike = np.tile(np.linspace(.8, 1.2, 30), 12)
    held = np.arange(len(tau)) % 3 == 2
    selected = _stratified_indices(strike, tau, held)
    assert len(selected) == 100 and len(np.unique(selected)) == 100
    assert {(t, h) for t, h in zip(tau[selected], held[selected])} == {
        (t, h) for t in np.unique(tau) for h in (True, False)}


def test_corpus_preserves_mask_and_rejects_duplicate_contracts(tmp_path):
    fwd, strike, tau, price, iv, held = packed_fixture()
    rec = _pack(fwd, strike, tau, price, iv, "FIXTURE|2025-01-01", held)
    save([rec], tmp_path/"quotes.npz")
    data = np.load(tmp_path/"quotes.npz", allow_pickle=False)
    np.testing.assert_array_equal(data["holdout_mask"][0].astype(bool), held)
    assert data["mask"].sum() == len(tau)
    strike[1] = strike[0]
    with pytest.raises(ValueError, match="Duplicate contracts"):
        _pack(fwd, strike, tau, price, iv, "FIXTURE|2025-01-01", held)


def test_calendar_exclusion_removes_all_symbols_on_same_date():
    fwd, strike, tau, price, iv, held = packed_fixture()
    panel = pd.DataFrame({"forward": fwd, "strike": strike, "maturity": tau,
                          "market_price_adjusted": price, "discount_factor": 1.,
                          "is_call": True, "market_iv": iv, "fold": "calibration",
                          "split": "train", "trade_date": pd.Timestamp("2025-01-01"),
                          "symbol": np.where(np.arange(len(tau)) < 18, "A", "B")})
    assert stock_surfaces(panel, "train", {"2025-01-01"}) == []
