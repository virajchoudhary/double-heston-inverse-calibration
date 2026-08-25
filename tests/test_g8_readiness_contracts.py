from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.g8_readiness.acquisition import (
    G8AcquisitionLocked,
    intake_official_nse,
    normalize_rbi_auction,
)
from src.g8_readiness.checkpoints import checkpoint_readiness_manifest
from src.g8_readiness.contracts import (
    DATE_FLOOR,
    SCAN_END,
    canonical_slot_roles,
    continuous_rate,
    discount_factor,
    forward_black_price,
    futures_implied_carry,
    implied_volatility,
    validate_g8_valuation_date,
)
from src.g8_readiness.model3 import evaluate_model3_inclusion
from src.g8_readiness.pipeline import assert_final_evaluation_gate, assert_future_acquisition_gate
from src.g8_readiness.pipeline import run_synthetic_end_to_end_replay


def test_date_floor_and_window_are_hard() -> None:
    assert validate_g8_valuation_date("2026-09-30") == DATE_FLOOR
    with pytest.raises(Exception, match="precedes frozen floor"):
        validate_g8_valuation_date("2026-09-29")
    with pytest.raises(Exception, match="scan end"):
        validate_g8_valuation_date("2027-01-01")


def test_rate_carry_math_matches_frozen_formulas() -> None:
    maturity = 45 / 365.0
    discount = discount_factor(0.0525, maturity)
    rate = continuous_rate(0.0525, maturity)
    forward = 101.0
    carry = futures_implied_carry(100.0, forward, maturity, 0.0525)[1]
    assert discount == pytest.approx(1.0 / (1.0 + 0.0525 * maturity))
    assert rate == pytest.approx(-math_log(discount) / maturity)
    assert carry == pytest.approx(rate - math_log(forward / 100.0) / maturity)


def math_log(value: float) -> float:
    import math
    return math.log(value)


def test_forward_black_round_trip_rejects_impossible_price() -> None:
    price = forward_black_price(102.0, 100.0, 30 / 365.0, 0.995, 0.25, "call")
    iv = implied_volatility(price, 102.0, 100.0, 30 / 365.0, 0.995, "call")
    assert iv == pytest.approx(0.25)
    with pytest.raises(Exception, match="bracketed"):
        implied_volatility(-1.0, 102.0, 100.0, 30 / 365.0, 0.995, "call")


def test_role_partition_is_exact() -> None:
    roles = canonical_slot_roles()
    assert int(roles["pricing_family_calibration"].sum()) == 12
    assert int(roles["pricing_family_holdout"].sum()) == 8
    assert not (roles["pricing_family_calibration"] & roles["pricing_family_holdout"]).any()


def _udiff_zip(trading_date: str) -> bytes:
    columns = [
        "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN",
        "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric",
        "OptnTp", "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric",
        "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst",
        "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd",
        "SsnId", "NewBrdLotQty", "Rmks", "Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4",
    ]
    row = {column: "" for column in columns}
    row.update({"TradDt": trading_date, "BizDt": trading_date, "Sgmt": "FO", "Src": "NSE"})
    frame = pd.DataFrame([row])
    csv = frame.to_csv(index=False).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "BhavCopy_NSE_FO_0_0_0_20261001_F_0000.csv",
            csv,
        )
    return buffer.getvalue()


def test_acquisition_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(G8AcquisitionLocked):
        intake_official_nse(
            "FO", "2026-10-01", store_root=tmp_path, authorize_acquisition=False,
            downloader=lambda url: b"",
        )


def test_fixture_intake_preserves_provenance_and_duplicate_identity(tmp_path: Path) -> None:
    content = _udiff_zip("01-Oct-2026")
    timestamp = "2026-10-01T18:00:00+00:00"

    def downloader(url: str) -> bytes:
        assert url == (
            "https://nsearchives.nseindia.com/content/fo/"
            "BhavCopy_NSE_FO_0_0_0_20261001_F_0000.csv.zip"
        )
        return content

    first = intake_official_nse(
        "FO", "2026-10-01", store_root=tmp_path, authorize_acquisition=True,
        downloader=downloader, retrieval_timestamp_utc=timestamp,
    )
    second = intake_official_nse(
        "FO", "2026-10-01", store_root=tmp_path, authorize_acquisition=True,
        downloader=lambda _url: (_ for _ in ()).throw(AssertionError("network retry forbidden")),
    )
    assert first.csv_sha256 == second.csv_sha256
    assert first.retrieval_timestamp_utc == second.retrieval_timestamp_utc == timestamp
    provenance = json.loads(next(tmp_path.rglob(".provenance.*.json")).read_text())
    assert provenance["zip_sha256"] == hashlib.sha256(content).hexdigest()


def test_corrupt_archive_fails_closed_and_is_retained(tmp_path: Path) -> None:
    corrupt = b"not a zip"
    with pytest.raises(Exception, match="invalid ZIP"):
        intake_official_nse(
            "CM", "2026-10-02", store_root=tmp_path, authorize_acquisition=True,
            downloader=lambda _url: corrupt, retrieval_timestamp_utc="fixed",
        )
    rejected = list((tmp_path / "rejected").glob("*.json"))
    assert len(rejected) == 1
    payload = json.loads(rejected[0].read_text())
    assert payload["retained"] is True
    assert payload["archive_sha256"] == hashlib.sha256(corrupt).hexdigest()


def test_future_rbi_observation_is_rejected() -> None:
    html = (
        '<div data-release-id="R" data-observation-date="2026-11-01" '
        'data-cutoff-price="99" data-yield-percent="5"></div>'
    )
    with pytest.raises(Exception, match="future RBI"):
        normalize_rbi_auction(
            html,
            official_url="https://www.rbi.org.in/result",
            latest_permitted_observation_date=date(2026, 10, 31),
        )


def test_checkpoint_manifest_reports_absent_not_failed() -> None:
    manifest = checkpoint_readiness_manifest()
    assert manifest["overall_status"] == "CHECKPOINT_ARTIFACTS_NOT_STAGED"
    assert manifest["all_checks_passed"] is False
    assert len(manifest["results"]) == 6
    assert all(item["status"] == "MISSING" for item in manifest["results"])


def test_model3_remote_state_is_not_frozen() -> None:
    decision = evaluate_model3_inclusion(None)
    assert decision["label"] == "MODEL3_NOT_FROZEN_NOT_EVALUATED"


def test_default_gates_cannot_be_crossed() -> None:
    with pytest.raises(RuntimeError, match="default invocation"):
        assert_future_acquisition_gate(authorize_g8_acquisition=False, valuation_date="2026-10-01", current_date=date(2026, 12, 1))
    with pytest.raises(Exception, match="calendar blocker"):
        assert_future_acquisition_gate(authorize_g8_acquisition=True, valuation_date="2026-10-01", current_date=date(2026, 8, 26))
    with pytest.raises(Exception, match="default invocation"):
        assert_final_evaluation_gate(authorize_g8_final_evaluation=False, selected_data_manifest={}, current_date=date(2026, 12, 1))
    with pytest.raises(Exception, match="real selected-data seal"):
        assert_final_evaluation_gate(authorize_g8_final_evaluation=True, selected_data_manifest={"classification": "SYNTHETIC_G8_PIPELINE_FIXTURE"}, current_date=date(2027, 1, 1))


def test_synthetic_end_to_end_replay_stays_fixture_only(tmp_path: Path) -> None:
    result = run_synthetic_end_to_end_replay(output_root=tmp_path)
    assert result["SYNTHETIC_G8_PIPELINE_FIXTURE"] is True
    assert result["NOT_REAL_MARKET_DATA"] is True
    assert result["NOT_A_RESEARCH_RESULT"] is True
    assert result["surfaces_constructed"] == 8
    assert result["scan_target_reached"] is True
    assert len(result["scan_selected_dates"]) == 2
    assert result["selected_data_status"].startswith("SYNTHETIC_G8_SELECTED_DATA_FIXTURE")
    assert result["pricing_family_winner"] == "NO_CLEAR_PRICING_FAMILY_WINNER"
    assert result["traditional_start_strategies"] == [
        "neutral_transform_midpoint", "deterministic_broad_start"
    ]
    assert result["checkpoint_overall_status"] == "CHECKPOINT_ARTIFACTS_NOT_STAGED"
    assert result["model3_label"] == "MODEL3_NOT_FROZEN_NOT_EVALUATED"
    assert result["pre_acquisition_freeze_status"] == "G8_READINESS_PREFLIGHT_NOT_SEALED"
    assert result["real_market_data_accessed"] is False
    assert result["research_result_computed"] is False
