from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scripts.run_ntpc_single_stock_pilot as pilot
from src.constants import PARAMETER_NAMES
from src.double_heston_reference import reference_double_heston_option


def test_exact_dte_and_year_fraction() -> None:
    assert [pilot.dte_and_time(pilot.VALUATION_DATE, expiry)[0] for expiry in pilot.EXPIRIES] == [13, 41, 76]
    assert [pilot.dte_and_time(pilot.VALUATION_DATE, expiry)[1] for expiry in pilot.EXPIRIES] == pytest.approx([13 / 365, 41 / 365, 76 / 365])


def test_log_moneyness_is_exact_and_validated() -> None:
    assert pilot.log_moneyness(345.0, 344.35) == pytest.approx(math.log(345.0 / 344.35))
    with pytest.raises(ValueError):
        pilot.log_moneyness(0.0, 344.35)


def test_row_selection_is_deterministic_and_fail_closed() -> None:
    first_all, first_selected, first_futures = pilot.build_option_dataset()
    second_all, second_selected, second_futures = pilot.build_option_dataset()
    pd.testing.assert_frame_equal(first_all, second_all, check_exact=True)
    pd.testing.assert_frame_equal(first_selected, second_selected, check_exact=True)
    pd.testing.assert_frame_equal(first_futures, second_futures, check_exact=True)
    assert len(first_all) == 146
    assert first_selected["sample_role"].value_counts().to_dict() == {"CALIBRATION": 12, "HOLDOUT": 7}
    missing = first_selected.loc[
        (first_selected["expiry_date"] == "2026-07-28")
        & (first_selected["option_type"] == "call")
        & (first_selected["target_log_moneyness"] == -0.10)
    ]
    assert missing.empty
    assert np.max(np.abs(first_selected["target_log_moneyness"] - first_selected["log_moneyness"])) <= 0.05 + 1e-12


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_iv_inversion_round_trip(option_type: str) -> None:
    forward, strike, maturity, discount, sigma = 345.55, 350.0, 13 / 365, 0.9981, 0.24
    price = pilot.forward_black_price(forward, strike, maturity, discount, sigma, option_type)
    recovered = pilot.implied_volatility(price, forward, strike, maturity, discount, option_type)
    assert recovered == pytest.approx(sigma, rel=0, abs=1e-10)


def test_iv_rejects_no_arbitrage_violation() -> None:
    with pytest.raises(ValueError, match="outside no-arbitrage"):
        pilot.implied_volatility(400.0, 345.55, 350.0, 13 / 365, 0.9981, "call")


def test_black_scholes_baseline_is_one_parameter_and_deterministic() -> None:
    _, selected, _ = pilot.build_option_dataset()
    calibration = selected.loc[selected["sample_role"] == "CALIBRATION"]
    holdout = selected.loc[selected["sample_role"] == "HOLDOUT"]
    first, first_predictions = pilot.fit_black_scholes(calibration, holdout)
    second, second_predictions = pilot.fit_black_scholes(calibration, holdout)
    assert first["parameter_count"] == 1
    assert first["sigma"] == pytest.approx(second["sigma"], rel=0, abs=1e-12)
    pd.testing.assert_frame_equal(first_predictions, second_predictions, check_exact=False, rtol=0, atol=1e-12)


def test_heston_pricing_validity_parity_convergence_and_reference() -> None:
    _, selected, _ = pilot.build_option_dataset()
    call_row = selected.loc[(selected["expiry_date"] == "2026-08-25") & (selected["option_type"] == "call") & (selected["target_log_moneyness"] == 0.0)].iloc[0]
    put_row = call_row.copy()
    put_row["option_type"] = "put"
    parameters = np.asarray([1.4, 0.045, 0.24, -0.45, 0.05])
    call_64 = pilot.price_heston_option(call_row, parameters, node_count=64)
    call_96 = pilot.price_heston_option(call_row, parameters, node_count=96)
    put_64 = pilot.price_heston_option(put_row, parameters, node_count=64)
    assert np.isfinite([call_64, call_96, put_64]).all()
    parity = float(call_row["spot"] * math.exp(-float(call_row["futures_implied_carry"]) * float(call_row["T"])) - float(call_row["strike"]) * math.exp(-float(call_row["continuous_rate"]) * float(call_row["T"])))
    assert call_64 - put_64 == pytest.approx(parity, abs=1e-10)
    assert call_64 == pytest.approx(call_96, abs=2e-5)
    # The repository reference independently implements both the affine transform
    # and adaptive Fourier quadrature.  A negligible second factor supplies the
    # canonical Double Heston contract while converging to this one-factor case.
    independent_parameters = np.asarray([
        parameters[0], parameters[1], parameters[2], parameters[3], parameters[4],
        8.0, 1e-10, 1e-5, 0.0, 1e-10,
    ])
    reference, diagnostics = reference_double_heston_option(
        float(call_row["spot"]),
        float(call_row["strike"]),
        float(call_row["T"]),
        float(call_row["continuous_rate"]),
        float(call_row["futures_implied_carry"]),
        "call",
        independent_parameters,
    )
    assert diagnostics["reliable"] is True
    assert call_64 == pytest.approx(reference, abs=3e-5)


def test_double_heston_total_variance_and_half_lives() -> None:
    parameters = np.asarray([1.2, 0.04, 0.2, -0.35, 0.03, 3.0, 0.02, 0.2, -0.25, 0.02])
    maturity = 41 / 365
    slow = pilot.expected_average_variance(parameters[0], parameters[1], parameters[4], maturity)
    fast = pilot.expected_average_variance(parameters[5], parameters[6], parameters[9], maturity)
    assert slow + fast == pytest.approx(sum([slow, fast]), rel=0, abs=1e-15)
    assert parameters[4] + parameters[9] == pytest.approx(0.05)
    assert pilot.half_life(parameters[0])[0] == pytest.approx(math.log(2) / parameters[0])
    assert pilot.half_life(parameters[5])[1] == pytest.approx(365 * math.log(2) / parameters[5])


def test_realized_volatility_formula() -> None:
    closes = np.asarray([100.0, 101.0, 99.0, 102.0])
    annualized, count, realized_variance = pilot.realized_volatility(closes)
    returns = np.diff(np.log(closes))
    assert count == 3
    assert realized_variance == pytest.approx(float(np.sum(returns**2)))
    assert annualized == pytest.approx(math.sqrt(252 / 3 * float(np.sum(returns**2))))


def test_no_future_return_leakage_and_holdout_separation() -> None:
    _, selected, _ = pilot.build_option_dataset()
    calibration = selected.loc[selected["sample_role"] == "CALIBRATION"]
    holdout = selected.loc[selected["sample_role"] == "HOLDOUT"]
    identities = ["expiry_date", "option_type", "strike"]
    assert set(map(tuple, calibration[identities].to_numpy())).isdisjoint(set(map(tuple, holdout[identities].to_numpy())))
    assert "history" not in inspect.signature(pilot.fit_black_scholes).parameters
    assert "history" not in inspect.signature(pilot.fit_stochastic_model).parameters
    assert max(pilot.HISTORY_DATES) == pilot.EXPIRIES[0]
    assert pilot.EXPIRIES[1] > pilot.AS_OF_DATE and pilot.EXPIRIES[2] > pilot.AS_OF_DATE


def test_source_files_and_manifest_hashes_when_manifest_exists() -> None:
    fo_csv, fo_zip = pilot._fo_paths()
    assert fo_csv.is_file() and fo_zip.is_file()
    assert pilot.sha256(fo_csv) == hashlib.sha256(fo_csv.read_bytes()).hexdigest().upper()
    rate_observation = json.loads(pilot.RISK_FREE_OBSERVATION_PATH.read_text(encoding="utf-8"))
    assert rate_observation["source_authority"] == "Reserve Bank of India"
    assert rate_observation["source_identifier"] == "RBI Press Release 2026-2027/672"
    assert rate_observation["release_date"] == "2026-07-15"
    assert rate_observation["cutoff_price"] == 98.688
    assert rate_observation["simple_annual_yield_decimal"] == pilot.RISK_FREE_SIMPLE_YIELD
    if not pilot.MANIFEST_PATH.exists():
        pytest.skip("manifest is created by the canonical pilot run")
    manifest = json.loads(pilot.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["underlying"] == "NTPC"
    assert manifest["valuation_date"] == "2026-07-15"
    for relative, expected in manifest["tracked_artifact_hashes"].items():
        assert pilot.sha256(pilot.REPOSITORY_ROOT / relative) == expected
    for relative, expected in manifest["artifact_hashes"].items():
        assert pilot.sha256(pilot.OUTPUT_ROOT / relative) == expected


def test_canonical_parameter_order_is_unchanged() -> None:
    assert tuple(PARAMETER_NAMES) == (
        "kappa_slow",
        "theta_slow",
        "sigma_slow",
        "rho_slow",
        "v0_slow",
        "kappa_fast",
        "theta_fast",
        "sigma_fast",
        "rho_fast",
        "v0_fast",
    )
