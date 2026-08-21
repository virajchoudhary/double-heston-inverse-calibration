from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import yaml

from scripts.create_market_data_audit_stage_a_structure import create_structure
from src.market_data_audit import (
    derive_option_metrics,
    futures_implied_carry,
    load_audit_config,
    load_stage_a_surface,
    summarize_option_coverage,
    validate_required_columns,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "market_data_audit_stage_a.yaml"


def _three_expiry_mock() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valuation_date": ["2026-07-01"] * 3,
            "underlying": ["INFY"] * 3,
            "expiry_date": ["2026-09-24", "2026-07-30", "2026-08-27"],
            "strike": [110.0, 90.0, 100.0],
            "option_type": ["call", "call", "put"],
            "bid": [3.0, 11.0, 5.0],
            "ask": [5.0, 13.0, 7.0],
            "volume": [0.0, 10.0, 2.0],
            "open_interest": [0.0, 20.0, 5.0],
        }
    )


def test_one_underlying_date_with_three_expiries_is_one_surface() -> None:
    derived = derive_option_metrics(_three_expiry_mock(), spot=100.0)
    summary = summarize_option_coverage(derived)

    assert len(summary) == 1
    assert summary.loc[0, "surface_count"] == 1
    assert summary.loc[0, "expiry_count"] == 3
    assert bool(summary.loc[0, "complete_near_mid_far"])


def test_dte_and_expiry_buckets_follow_actual_expiry_order() -> None:
    derived = derive_option_metrics(_three_expiry_mock(), spot=100.0)
    by_expiry = derived.set_index(derived["expiry_date"].dt.strftime("%Y-%m-%d"))

    assert by_expiry.loc["2026-07-30", "dte"] == 29
    assert math.isclose(by_expiry.loc["2026-07-30", "T"], 29.0 / 365.0)
    assert by_expiry.loc["2026-07-30", "expiry_bucket"] == "near"
    assert by_expiry.loc["2026-08-27", "expiry_bucket"] == "mid"
    assert by_expiry.loc["2026-09-24", "expiry_bucket"] == "far"


def test_log_moneyness_mid_normalization_and_spread() -> None:
    derived = derive_option_metrics(_three_expiry_mock(), spot=100.0)
    first = derived.iloc[0]

    assert math.isclose(first["k_over_s"], 1.1)
    assert math.isclose(first["log_k_over_s"], math.log(1.1))
    assert math.isclose(first["mid"], 4.0)
    assert math.isclose(first["normalized_mid"], 0.04)
    assert math.isclose(first["relative_bid_ask_spread"], 0.5)


def test_price_usability_is_independent_of_zero_volume_and_open_interest() -> None:
    derived = derive_option_metrics(_three_expiry_mock(), spot=100.0)
    zero_activity = derived.iloc[0]

    assert bool(zero_activity["price_usable"])
    assert not bool(zero_activity["volume_positive"])
    assert not bool(zero_activity["open_interest_positive"])


def test_futures_implied_carry_helper() -> None:
    expected = 0.06 - math.log(103.0 / 100.0) / 0.5
    actual = futures_implied_carry(
        risk_free_rate=0.06,
        futures_price=103.0,
        spot=100.0,
        maturity_years=0.5,
    )
    assert math.isclose(actual, expected)


def test_derived_calculations_do_not_mutate_raw_input() -> None:
    raw = _three_expiry_mock()
    before = raw.copy(deep=True)

    derived = derive_option_metrics(raw, spot=100.0)

    assert_frame_equal(raw, before)
    assert derived is not raw
    assert "dte" not in raw.columns
    assert np.isfinite(derived["log_k_over_s"]).all()


def test_yaml_driven_required_column_validator() -> None:
    raw = _three_expiry_mock()
    validate_required_columns(raw, "options")

    with pytest.raises(ValueError, match="ask"):
        validate_required_columns(raw.drop(columns="ask"), "options")


def test_config_keeps_representation_open_and_nifty_reference_only() -> None:
    config = load_audit_config(CONFIG)

    assert config["universe"]["reference"] == {
        "underlying": "NIFTY",
        "reference_only": True,
        "ranked": False,
    }
    assert config["representation_status"] == "PROVISIONAL"
    assert config["representation_decision"] == "OPEN"
    assert config["replacement_representation"] is None
    assert not config["replacement_54_feature_representation_defined"]
    assert not config["replacement_57_feature_representation_defined"]
    assert not config["other_replacement_neural_representation_defined"]


def test_structure_generator_creates_manifests_but_no_fake_workbooks(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "stage_a"

    created = create_structure(CONFIG, output_root)
    created_again = create_structure(CONFIG, output_root)

    assert len(created) == 27
    assert created_again == []
    assert (
        output_root
        / "candidates"
        / "it"
        / "INFY"
        / "2026-07-01"
        / "collection_manifest.yaml"
    ).is_file()
    assert (
        output_root
        / "reference"
        / "NIFTY"
        / "2026-07-22"
        / "collection_manifest.yaml"
    ).is_file()
    assert not list(output_root.rglob("*.xlsx"))


def test_loader_reads_mock_workbooks_and_validates_one_manifest_identity(
    tmp_path: Path,
) -> None:
    surface_directory = tmp_path / "INFY" / "2026-07-01"
    surface_directory.mkdir(parents=True)
    options = _three_expiry_mock()
    futures = pd.DataFrame(
        {
            "valuation_date": ["2026-07-01"],
            "underlying": ["INFY"],
            "expiry_date": ["2026-07-30"],
            "futures_price": [101.0],
        }
    )
    spot = pd.DataFrame(
        {
            "valuation_date": ["2026-07-01"],
            "underlying": ["INFY"],
            "spot": [100.0],
        }
    )
    options.to_excel(surface_directory / "options_raw.xlsx", index=False)
    futures.to_excel(surface_directory / "futures_raw.xlsx", index=False)
    spot.to_excel(surface_directory / "spot_raw.xlsx", index=False)
    manifest = {
        "stage": "A",
        "collection_status": "COLLECTED_FOR_TEST",
        "underlying": "INFY",
        "valuation_date": "2026-07-01",
        "reference_only": False,
        "surface_definition": "one_underlying_date_with_all_near_mid_far_expiry_slices",
        "expected_files": [
            "options_raw.xlsx",
            "futures_raw.xlsx",
            "spot_raw.xlsx",
            "collection_manifest.yaml",
        ],
    }
    (surface_directory / "collection_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    loaded = load_stage_a_surface(surface_directory)

    assert len(loaded.options) == 3
    assert len(loaded.futures) == 1
    assert len(loaded.spot) == 1
    assert loaded.manifest["underlying"] == "INFY"
