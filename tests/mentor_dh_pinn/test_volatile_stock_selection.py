from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.mentor_dh_pinn.volatile_stock_selection import (
    REPORT_COLUMNS,
    VolatileStockAuditError,
    candidate_volatility,
    load_spec,
    parse_security_report,
    propose_split,
    validate_history,
)


CONFIG = Path("configs/mentor_dh_pinn/volatile_stock_selection_v1.yaml")


def test_frozen_mentor_methodology_and_universe_parse() -> None:
    spec = load_spec(CONFIG)
    assert (spec.history_start, spec.history_end) == (date(2021, 1, 1), date(2026, 7, 21))
    assert spec.option_dates == (date(2026, 7, 1), date(2026, 7, 15), date(2026, 7, 22))
    assert spec.calendar_months == 3 and spec.minimum_returns == 50 and spec.annualization_days == 252
    assert len(spec.symbols) == 10 and spec.series == "EQ"


def test_security_report_schema_and_date_fail_closed() -> None:
    header = ",".join(REPORT_COLUMNS)
    row = "X,EQ,04-Jan-2021,100,101,102,99,101,101,100.5,10,1,2,5,50"
    parsed = parse_security_report(f"{header}\n{row}\n".encode(), date(2021, 1, 4))
    assert parsed.loc[0, "CLOSE_PRICE"] == 101
    with pytest.raises(VolatileStockAuditError, match="date identity"):
        parse_security_report(f"{header}\n{row}\n".encode(), date(2021, 1, 5))


def test_conflicting_duplicate_symbol_date_is_rejected() -> None:
    spec = replace(load_spec(CONFIG), symbols=("X",))
    rows = [_history_row("X", date(2021, 1, 4), 100, 101), _history_row("X", date(2021, 1, 4), 100, 102)]
    with pytest.raises(VolatileStockAuditError, match="Conflicting duplicate"):
        validate_history(pd.DataFrame(rows), spec)


def test_calendar_month_window_uses_sample_std_and_strict_prevaluation_close() -> None:
    spec = replace(load_spec(CONFIG), symbols=("X",), history_start=date(2021, 1, 1), history_end=date(2021, 6, 30), minimum_returns=2)
    rows = []
    value = date(2021, 1, 1); close = 100.0
    while value <= date(2021, 6, 30):
        if value.weekday() < 5:
            new_close = close * (1.01 if len(rows) % 2 == 0 else 0.995)
            rows.append(_history_row("X", value, close, new_close)); close = new_close
        value += timedelta(days=1)
    history = validate_history(pd.DataFrame(rows), spec)
    result = candidate_volatility(history, "X", date(2021, 6, 15), spec)
    assert result["window_start"] == date(2021, 3, 15)
    assert result["window_end"] == date(2021, 6, 14)
    assert result["returns"] == result["closes"] - 1
    assert result["rolling_count"] > 0


def test_previous_close_discontinuity_invalidates_candidate_window() -> None:
    spec = replace(load_spec(CONFIG), symbols=("X",), history_start=date(2021, 1, 1), history_end=date(2021, 6, 30), minimum_returns=2)
    rows = []
    value = date(2021, 3, 1); close = 100.0
    while value < date(2021, 6, 1):
        if value.weekday() < 5:
            rows.append(_history_row("X", value, close, close + 1)); close += 1
        value += timedelta(days=1)
    rows[10]["PREV_CLOSE"] += 10
    history = validate_history(pd.DataFrame(rows), spec)
    with pytest.raises(VolatileStockAuditError, match="discontinuous"):
        candidate_volatility(history, "X", date(2021, 6, 1), spec)


def _history_row(symbol: str, value: date, previous: float, close: float) -> dict[str, object]:
    return {
        "SYMBOL": symbol, "SERIES": "EQ", "DATE1": value, "PREV_CLOSE": previous,
        "OPEN_PRICE": close, "HIGH_PRICE": close, "LOW_PRICE": close, "LAST_PRICE": close,
        "CLOSE_PRICE": close, "AVG_PRICE": close, "TTL_TRD_QNTY": 1, "TURNOVER_LACS": 1,
        "NO_OF_TRADES": 1, "DELIV_QTY": 1, "DELIV_PER": 100,
    }

def test_unknown_config_field_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.read_text(encoding="utf-8") + "unknown_field: true\n", encoding="utf-8")
    with pytest.raises(VolatileStockAuditError, match="Config fields drifted"):
        load_spec(path)


def test_official_ooxml_variant_parses_under_same_schema() -> None:
    from io import BytesIO

    row = {column: 1 for column in REPORT_COLUMNS}
    row.update(SYMBOL="X", SERIES="EQ", DATE1="04-Jan-2021")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([row], columns=REPORT_COLUMNS).to_excel(writer, index=False)
    parsed = parse_security_report(buffer.getvalue(), date(2021, 1, 4))
    assert tuple(parsed.columns) == REPORT_COLUMNS


def test_strike_quantile_split_is_deterministic_and_disjoint() -> None:
    spec = load_spec(CONFIG)
    rows = []
    for expiry in (date(2026, 7, 28), date(2026, 8, 25)):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            rows.append({"expiry": expiry, "StrkPric": strike, "K_over_S": strike / 100.0})
    active = pd.DataFrame(rows)
    first = propose_split(active, spec)
    second = propose_split(active.sample(frac=1.0, random_state=7), spec)
    pd.testing.assert_frame_equal(first, second)
    assert list(first["role"]).count("calibration") == 6
    assert list(first["role"]).count("holdout") == 4
    assert not first.duplicated(["expiry", "StrkPric"]).any()
