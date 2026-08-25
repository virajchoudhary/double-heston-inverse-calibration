"""Current-locked orchestration plus a fully synthetic end-to-end replay."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..nse_stage_a import UDIFF_COLUMNS
from ..r2_representation.surface import R2Surface
from .acquisition import NSEArchiveRecord, RbiRateRecord, normalize_rbi_auction
from .checkpoints import checkpoint_readiness_manifest
from .contracts import DATE_FLOOR, PRIMARY_SYMBOLS, SCAN_END, discount_factor, forward_black_price
from .contracts import canonical_slot_roles
from .harness import (
    aggregate_pricing_family,
    calibrate_real_g8_traditional,
    fit_pricing_family_surface,
    pricing_rows_from_surface,
)
from .manifests import artifact_identity, build_pre_acquisition_freeze, build_selected_data_freeze
from .model3 import evaluate_model3_inclusion
from .scanner import full_window_backup_replacements, scan_common_dates
from .surfaces import build_g8_r2_surface


class CurrentDateAcquisitionLocked(RuntimeError):
    pass


class FinalEvaluationLocked(RuntimeError):
    pass


def assert_future_acquisition_gate(
    *,
    authorize_g8_acquisition: bool,
    valuation_date: date | str,
    current_date: date | None = None,
) -> date:
    value = valuation_date if type(valuation_date) is date else date.fromisoformat(str(valuation_date))
    today = current_date or date.today()
    if not authorize_g8_acquisition:
        raise RuntimeError("default invocation cannot acquire G8 market data")
    if value < DATE_FLOOR:
        raise RuntimeError(f"valuation {value.isoformat()} precedes frozen floor")
    if today < DATE_FLOOR:
        raise CurrentDateAcquisitionLocked(
            f"calendar blocker remains: current date {today.isoformat()} precedes {DATE_FLOOR.isoformat()}"
        )
    if value > today:
        raise CurrentDateAcquisitionLocked(
            f"future valuation evidence is unavailable: requested {value.isoformat()}, current {today.isoformat()}"
        )
    return value


def assert_final_evaluation_gate(
    *,
    authorize_g8_final_evaluation: bool,
    selected_data_manifest: dict[str, Any],
    current_date: date | None = None,
) -> None:
    today = current_date or date.today()
    if not authorize_g8_final_evaluation:
        raise FinalEvaluationLocked("default invocation cannot run final G8 evaluation")
    if selected_data_manifest.get("classification") != "REAL_G8_SELECTED_DATA":
        raise FinalEvaluationLocked("real final evaluation requires a real selected-data seal")
    if selected_data_manifest.get("status") != "REAL_G8_SELECTED_DATA_FROZEN":
        raise FinalEvaluationLocked("selected-data manifest is not sealed")
    if today < DATE_FLOOR:
        raise FinalEvaluationLocked("current-date calendar blocker remains")


def _udiff_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: row.get(column, "") for column in UDIFF_COLUMNS} for row in rows],
        columns=list(UDIFF_COLUMNS),
    )


def _fixture_market_frames(symbol: str, valuation: date, spot: float, simple_yield: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cm_rows = [
        {
            "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}",
            "Sgmt": "CM", "Src": "NSE", "FinInstrmTp": "EQ", "FinInstrmId": "1",
            "TckrSymb": symbol, "SctySrs": "EQ", "ClsPric": f"{spot:.2f}",
        }
    ]
    fo_rows: list[dict[str, Any]] = []
    instrument_id = 10
    continuous_rate = -math.log(discount_factor(simple_yield, 1.0))
    carry = 0.01
    for offset in (30, 60):
        expiry = valuation + timedelta(days=offset)
        maturity = offset / 365.0
        forward = spot * math.exp((continuous_rate - carry) * maturity)
        discount = discount_factor(simple_yield, maturity)
        fo_rows.append(
            {
                "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}",
                "Sgmt": "FO", "Src": "NSE", "FinInstrmTp": "STF", "FinInstrmId": str(instrument_id),
                "TckrSymb": symbol, "XpryDt": f"{expiry:%d-%b-%Y}",
                "FininstrmActlXpryDt": f"{expiry:%d-%b-%Y}", "ClsPric": f"{forward:.6f}",
                "UndrlygPric": f"{spot:.2f}", "OpnIntrst": "100", "TtlTradgVol": "100",
                "TtlNbOfTxsExctd": "10",
            }
        )
        instrument_id += 1
        for target in (-0.10, -0.05, 0.0, 0.05, 0.10):
            strike = spot * math.exp(target)
            for suffix, option_type in (("CE", "call"), ("PE", "put")):
                price = forward_black_price(forward, strike, maturity, discount, 0.25, option_type)
                fo_rows.append(
                    {
                        "TradDt": f"{valuation:%d-%b-%Y}", "BizDt": f"{valuation:%d-%b-%Y}",
                        "Sgmt": "FO", "Src": "NSE", "FinInstrmTp": "STO",
                        "FinInstrmId": str(instrument_id), "TckrSymb": symbol,
                        "XpryDt": f"{expiry:%d-%b-%Y}",
                        "FininstrmActlXpryDt": f"{expiry:%d-%b-%Y}",
                        "StrkPric": f"{strike:.10f}", "OptnTp": suffix,
                        "ClsPric": f"{price:.8f}", "UndrlygPric": f"{spot:.2f}",
                        "OpnIntrst": "100", "TtlTradgVol": "100", "TtlNbOfTxsExctd": "10",
                    }
                )
                instrument_id += 1
    return _udiff_frame(cm_rows), _udiff_frame(fo_rows)


def _fixture_rate(valuation: date, sequence: int) -> RbiRateRecord:
    observation = valuation - timedelta(days=7)
    html = (
        '<div class="auction-result" '
        f'data-release-id="FIXTURE-{observation:%Y%m%d}-{sequence}" '
        f'data-observation-date="{observation.isoformat()}" '
        'data-cutoff-price="98.7000" data-yield-percent="5.2500"></div>'
    )
    record = normalize_rbi_auction(
        html,
        official_url="https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=FIXTURE",
        latest_permitted_observation_date=valuation,
    )
    object.__setattr__(record, "source_sha256", hashlib.sha256(html.encode()).hexdigest())
    return record


def _mock_pricing_functions() -> dict[str, Any]:
    def return_observed(frame: pd.DataFrame, _parameters: np.ndarray) -> np.ndarray:
        return frame["observed_price"].to_numpy(float)

    return {name: return_observed for name in ("STANDARD_HESTON", "DOUBLE_HESTON")}


def _mock_inverse_pricer(*_args: Any, **_kwargs: Any) -> np.ndarray:
    # Signature is completed by calibrate_real_g8_traditional's residual closure.
    raise AssertionError("unused mock")


def run_synthetic_end_to_end_replay(output_root: Path | str | None = None) -> dict[str, Any]:
    """Exercise the orchestration only; every market byte here is fabricated."""
    labels = {
        "SYNTHETIC_G8_PIPELINE_FIXTURE": True,
        "NOT_REAL_MARKET_DATA": True,
        "NOT_A_RESEARCH_RESULT": True,
    }
    dates = (DATE_FLOOR, DATE_FLOOR + timedelta(days=10))
    rates_by_date = {value: _fixture_rate(value, index + 1) for index, value in enumerate(dates)}
    support: dict[date, dict[str, bool]] = {}
    surfaces: list[R2Surface] = []
    archive_records: list[NSEArchiveRecord] = []
    rate_records: list[RbiRateRecord] = []
    construction_reports: list[dict[str, Any]] = []
    construction_failures: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="g8_fixture_", dir=output_root) as temporary:
        root = Path(temporary)
        for date_index, value in enumerate(dates):
            support[value] = {}
            for symbol_index, symbol in enumerate(PRIMARY_SYMBOLS):
                spot = 100.0 + 5.0 * symbol_index + date_index
                cm, fo = _fixture_market_frames(symbol, value, spot, 0.0525)
                try:
                    available_rates = {
                        date.fromisoformat(record.observation_date): record for observed, record in rates_by_date.items()
                        if type(observed) is date and observed <= value
                    }
                    release_calendar = {
                        record.release_identifier: date.fromisoformat(record.observation_date)
                        for observed, record in rates_by_date.items()
                        if type(observed) is date and observed <= value
                    }
                    surface, report = build_g8_r2_surface(
                        symbol,
                        value,
                        cm,
                        fo,
                        available_rates,
                        data_classification="SYNTHETIC_G8_PIPELINE_FIXTURE",
                        official_release_calendar=release_calendar,
                        development_contract_keys=set(),
                    )
                    surfaces.append(surface)
                    construction_reports.append(report)
                    support[value][symbol] = True
                except Exception as exc:
                    support[value][symbol] = False
                    construction_failures.append(
                        {
                            "valuation_date": value.isoformat(),
                            "symbol": symbol,
                            "reason": str(exc),
                        }
                    )
            for market in ("CM", "FO"):
                source_frame = cm if market == "CM" else fo
                csv_bytes = source_frame.to_csv(index=False).encode()
                filename = f"BhavCopy_NSE_{market}_0_0_0_{value:%Y%m%d}_F_0000.csv.zip"
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(filename.removesuffix(".zip"), csv_bytes)
                archive_bytes = zip_buffer.getvalue()
                archive_records.append(
                    NSEArchiveRecord(
                        market=market,
                        trading_date=value.isoformat(),
                        official_url=(
                            "https://nsearchives.nseindia.com/content/"
                            f"{'cm' if market == 'CM' else 'fo'}/{filename}"
                        ),
                        retrieval_timestamp_utc=f"SYNTHETIC_FIXTURE_{date_index}",
                        original_filename=filename,
                        byte_size=len(archive_bytes),
                        zip_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                        zip_integrity_result=True,
                        member_filename=filename.removesuffix(".zip"),
                        csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                        encoding="UTF-8",
                        delimiter=",",
                        archive_path=root / f"never_materialized_{market}.zip",
                        extracted_csv_path=root / f"never_materialized_{market}.csv",
                    )
                )
            rate_records.append(rates_by_date[value])

        scan = scan_common_dates(support, expected_calendar_dates=dates)
        backup_decisions, replacements = full_window_backup_replacements(
            support,
            expected_scanned_dates=dates,
        )
        roles = {
            name: [int(index) for index, valid in enumerate(mask_array) if valid]
            for name, mask_array in canonical_slot_roles().items()
        }
        selected_manifest = build_selected_data_freeze(
            surfaces=surfaces,
            archive_records=archive_records,
            rate_records=rate_records,
            scan_result=scan,
            backup_decisions=backup_decisions,
            observation_role_mappings=roles,
        )
        pricing_runs = []
        for surface in surfaces[:2]:
            rows = pricing_rows_from_surface(surface)
            pricing_runs.append(fit_pricing_family_surface(
                rows,
                source_surface=surface,
                pricing_functions=_mock_pricing_functions(),
                max_nfev=3,
            ))
        family_aggregate = aggregate_pricing_family(pricing_runs)
        traditional_runs = []
        for surface in surfaces[:1]:
            def observed_pricer(_spot, _strikes, _maturities, _rates, _carries, _types, _parameters, *, node_count=64):
                return np.asarray(surface.prices, dtype=float)[np.asarray(surface.mask, dtype=bool)] * surface.spot

            result = calibrate_real_g8_traditional(surface, max_nfev=3, pricer=observed_pricer)
            traditional_runs.append(
                {
                    "surface_id": surface.surface_id,
                    "start_count": len(result.starts),
                    "strategies": [row["start_strategy"] for row in result.starts],
                    "representative_parameters": result.representative.tolist(),
                    "runtime_seconds_all_starts": result.wall_seconds_all_starts,
                }
            )
        checkpoint_manifest = checkpoint_readiness_manifest()
        model3_decision = evaluate_model3_inclusion(None, acquisition_has_begun=False)
        prefreeze = build_pre_acquisition_freeze(
            protocol_commit="7eecc7188c54f9d4505d32ccf5c51069a4c3a97c",
            config_path=Path("configs/g8_final_real_market.yaml"),
            checkpoint_manifest=checkpoint_manifest,
            independent_review_verdict=None,
            model3_decision=model3_decision,
            tool_identities={
                "acquisition": {"path": Path("src/g8_readiness/acquisition.py"), "sha256": artifact_identity(Path("src/g8_readiness/acquisition.py"))["sha256"]},
                "surface_builder": {"path": Path("src/g8_readiness/surfaces.py"), "sha256": artifact_identity(Path("src/g8_readiness/surfaces.py"))["sha256"]},
                "evaluation_harness": {"module": "src/g8_readiness/harness", "sha256": artifact_identity(Path("src/g8_readiness/harness.py"))["sha256"]},
            },
            current_date=date.today(),
            protocol_frozen=True,
            seal=False,
        )
    return {
        **labels,
        "surfaces_constructed": len(surfaces),
        "construction_failures": construction_failures,
        "scan_selected_dates": [value.isoformat() for value in scan.selected_dates],
        "scan_target_reached": scan.reached_target,
        "backup_replacements": replacements,
        "selected_data_status": selected_manifest["status"],
        "selected_data_manifest_sha256": selected_manifest["manifest_sha256"],
        "pricing_family_winner": family_aggregate["winner_label"],
        "traditional_start_strategies": traditional_runs[0]["strategies"] if traditional_runs else [],
        "checkpoint_overall_status": checkpoint_manifest["overall_status"],
        "model3_label": model3_decision["label"],
        "pre_acquisition_freeze_status": prefreeze["status"],
        "real_market_data_accessed": False,
        "research_result_computed": False,
    }
