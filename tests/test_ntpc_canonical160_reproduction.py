"""Regression contracts for NTPC canonical-160 artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import investigate_ntpc_canonical160_reproduction as forensic
from scripts import run_ntpc_dh_stability_reparameterization as experiment
from src.ntpc_pricing_input_contract import canonicalize_pricing_inputs


FIXTURES = Path(__file__).parent / "fixtures"


def test_reference_stability_uses_exactly_66_unique_unordered_pairs() -> None:
    frame = pd.read_csv(FIXTURES / "ntpc_canonical160_reviewed_vectors.csv")
    result, pairs = forensic.reference_stability(frame, forensic.HARD_BOUNDS)
    assert result["solution_count"] == 12
    assert result["unordered_pair_count"] == 66
    assert len(pairs) == 66
    assert (pairs["left_start_id"] < pairs["right_start_id"]).all()
    assert not (pairs["left_start_id"] == pairs["right_start_id"]).any()
    assert not pairs.duplicated(["left_start_id", "right_start_id"]).any()
    assert result["median_pairwise_range_scaled_distance"] == pytest.approx(0.35733879424203197, abs=1e-15)
    assert result["maximum_pairwise_range_scaled_distance"] == pytest.approx(0.5641491074467359, abs=1e-15)
    assert result["maximum_range_scaled_distance_from_best"] == pytest.approx(0.4910854918863381, abs=1e-15)
    assert result["materially_displaced_start_count"] == 11
    assert result["cluster_count"] == 7


def test_strict_start_id_join_detects_first_material_divergence() -> None:
    reviewed = pd.read_csv(FIXTURES / "ntpc_canonical160_reviewed_vectors.csv")
    replay = pd.read_csv(FIXTURES / "ntpc_canonical160_replay_vectors.csv")
    compared = forensic.compare_starts(reviewed, replay, forensic.HARD_BOUNDS)
    assert compared["start_id"].tolist() == list(range(12))
    assert compared.loc[compared["start_id"] == 0, "classification"].item() == "NUMERICAL_TOLERANCE_ONLY"
    assert compared.loc[compared["start_id"] == 3, "classification"].item() == "MATERIALLY_DIFFERENT_SOLUTION"
    assert compared.loc[compared["start_id"] == 6, "classification"].item() == "MATERIALLY_DIFFERENT_SOLUTION"
    assert compared.loc[compared["start_id"] == 6, "range_scaled_rms_difference"].item() == pytest.approx(
        0.318066732342736, abs=1e-15
    )
    assert compared.loc[compared["classification"] == "MATERIALLY_DIFFERENT_SOLUTION", "start_id"].tolist() == [3, 4, 5, 6, 8, 9, 11]


def test_start_population_hash_is_frozen() -> None:
    frame = pd.read_csv(FIXTURES / "ntpc_canonical160_start_hashes.csv").sort_values("start_id")
    digest = hashlib.sha256("|".join(frame["canonical_start_sha256"]).encode()).hexdigest().upper()
    assert frame["start_id"].tolist() == list(range(12))
    assert digest == "3AC1C30FF1B5416987D2103EA70B9262BBB8B4991F18F7A06C98E3A41C86ABA1"


def test_pricing_inputs_reconstruct_binary_time_from_dte() -> None:
    frame = pd.DataFrame(
        {
            "DTE": [13, 41, 76],
            "risk_free_simple_yield": [0.053324] * 3,
            "spot": [344.35] * 3,
            "matched_futures_price": [345.0, 346.0, 347.0],
            "T": [float(f"{13 / 365:.16g}"), float(f"{41 / 365:.16g}"), float(f"{76 / 365:.16g}")],
            "discount_factor": [0.0] * 3,
            "continuous_rate": [0.0] * 3,
            "futures_implied_carry": [0.0] * 3,
        }
    )
    restored = canonicalize_pricing_inputs(frame)
    np.testing.assert_array_equal(restored["T"].to_numpy(), np.asarray([13 / 365, 41 / 365, 76 / 365]))
    assert not np.array_equal(frame["T"].to_numpy(), restored["T"].to_numpy())
    assert np.isfinite(restored[["discount_factor", "continuous_rate", "futures_implied_carry"]]).all().all()


def test_fixture_provenance_hashes_are_explicit() -> None:
    expected = {
        "ntpc_canonical160_reviewed_vectors.csv": forensic.REVIEWED_FIXTURE_SHA256,
        "ntpc_canonical160_replay_vectors.csv": forensic.REPLAY_FIXTURE_SHA256,
        "ntpc_canonical160_start_hashes.csv": forensic.START_FIXTURE_SHA256,
    }
    for name, digest in expected.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest().upper() == digest


def test_csv_replay_is_canonicalized_before_reparameterized_optimization(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "DTE": [13], "risk_free_simple_yield": [0.053324], "spot": [344.35],
            "matched_futures_price": [345.0], "T": [float(f"{13 / 365:.16g}")],
            "discount_factor": [0.0], "continuous_rate": [0.0], "futures_implied_carry": [0.0],
        }
    )
    path = tmp_path / "selected_options.csv"
    source.to_csv(path, index=False)
    loaded = experiment.load_frozen_selected_options(path)
    assert loaded.loc[0, "T"] == 13 / 365
    assert loaded.loc[0, "T"] != pd.read_csv(path).loc[0, "T"]


def test_forensic_run_rejects_unreviewed_input_bytes(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("start_id\n0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forensic input provenance mismatch"):
        forensic.run(wrong, wrong, wrong, wrong, tmp_path / "output")


def test_causal_probe_validation_requires_t_only_reproduction_and_identical_repeats() -> None:
    reviewed = {"maximum_abs_difference_from_reviewed": 2e-16,
                "maximum_abs_difference_from_replay": 1.0}
    replay = {"maximum_abs_difference_from_reviewed": 1.0,
              "maximum_abs_difference_from_replay": 9e-17}
    cases = {
        "rebuilt_in_memory": reviewed,
        "csv_with_rebuilt_T": reviewed,
        **{
            f"csv_with_rebuilt_{field}": replay
            for field in (
                "log_moneyness", "discount_factor", "continuous_rate",
                "futures_implied_carry", "market_implied_volatility",
            )
        },
        "csv_loaded": replay,
    }
    probe = {
        "max_nfev": 160,
        "start_id": 6,
        "cases": cases,
        "csv_repeat_fits": [{"parameter_sha256": "SAME"}] * 3,
    }
    result = forensic.validate_causal_probe(probe)
    assert result["csv_repeat_count"] == 3
    probe["cases"]["csv_with_rebuilt_T"] = replay
    with pytest.raises(RuntimeError, match="rebuilt T did not reproduce"):
        forensic.validate_causal_probe(probe)
