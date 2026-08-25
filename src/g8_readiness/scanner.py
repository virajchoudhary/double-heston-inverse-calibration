"""Deterministic structural scan and full-window backup policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from .contracts import BACKUP_SYMBOLS_BY_PRIMARY, PRIMARY_SYMBOLS, SCAN_END, SCAN_START, G8ReadinessError, validate_g8_valuation_date


@dataclass(frozen=True)
class ScanFailure:
    valuation_date: date
    symbol: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    selected_dates: tuple[date, ...]
    failures: tuple[ScanFailure, ...]
    reached_target: bool
    complete_window_scanned: bool


def scan_common_dates(
    structural_support: Mapping[date | str, Mapping[str, bool]],
    *,
    active_symbols: tuple[str, ...] = PRIMARY_SYMBOLS,
    target_common_dates: int = 2,
) -> ScanResult:
    """Scan ascending dates and stop after two common dates without model access."""
    if tuple(active_symbols) != PRIMARY_SYMBOLS and not set(PRIMARY_SYMBOLS).issuperset(active_symbols):
        raise G8ReadinessError("active symbols must remain within the frozen universe")
    normalized: dict[date, Mapping[str, bool]] = {}
    for raw_date, supports in structural_support.items():
        value = validate_g8_valuation_date(raw_date)
        normalized[value] = supports
    selected: list[date] = []
    failures: list[ScanFailure] = []
    ordered_dates = sorted(normalized)
    for value in ordered_dates:
        date_failures: list[ScanFailure] = []
        for symbol in active_symbols:
            supported = bool(normalized[value].get(symbol, False))
            if not supported:
                date_failures.append(
                    ScanFailure(value, symbol, "STRUCTURAL_SUPPORT_FALSE")
                )
        failures.extend(date_failures)
        if not date_failures:
            selected.append(value)
            if len(selected) >= target_common_dates:
                break
    reached_target = len(selected) >= target_common_dates
    return ScanResult(
        selected_dates=tuple(selected),
        failures=tuple(failures),
        reached_target=reached_target,
        complete_window_scanned=not reached_target,
    )


@dataclass(frozen=True)
class BackupDecision:
    primary_symbol: str
    backup_symbol: str
    primary_support_count: int
    trigger: str


def full_window_backup_replacements(
    structural_support: Mapping[date | str, Mapping[str, bool]],
) -> tuple[tuple[BackupDecision, ...], dict[str, str]]:
    """Replace a primary only after zero eligible surfaces across the full window."""
    counts = {symbol: 0 for symbol in PRIMARY_SYMBOLS}
    for raw_date, supports in structural_support.items():
        value = validate_g8_valuation_date(raw_date)
        if value < SCAN_START or value > SCAN_END:
            continue
        for symbol in PRIMARY_SYMBOLS:
            if bool(supports.get(symbol, False)):
                counts[symbol] += 1
    decisions = tuple(
        BackupDecision(
            primary_symbol=symbol,
            backup_symbol=BACKUP_SYMBOLS_BY_PRIMARY[symbol],
            primary_support_count=0,
            trigger="PRIMARY_ZERO_ELIGIBLE_SURFACES_COMPLETE_WINDOW",
        )
        for symbol in PRIMARY_SYMBOLS
        if counts[symbol] == 0
    )
    replacements = {decision.primary_symbol: decision.backup_symbol for decision in decisions}
    return decisions, replacements
