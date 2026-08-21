from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = [
    "surface_key",
    "symbol",
    "trade_date",
    "expiry_date",
    "option_type",
    "strike_price",
    "time_to_expiry_years",
    "underlying_value",
    "observed_option_price",
    "log_moneyness",
    "strike_to_spot",
    "number_of_contracts",
    "open_interest",
    "is_model_ready",
    "model_input_status",
]


@dataclass
class SurfaceTemplate:
    surface_key: str
    symbol: str
    trade_date: str
    expiry_date: str
    option_type: str
    spot: float
    rate: float
    dividend: float
    strikes: np.ndarray
    tau: np.ndarray
    log_moneyness: np.ndarray
    strike_to_spot: np.ndarray
    is_call: np.ndarray
    contracts: np.ndarray
    open_interest: np.ndarray

    def feature_matrix(self, prices: np.ndarray) -> np.ndarray:
        normalized_price = prices / max(self.spot, 1.0)
        return np.column_stack(
            [
                normalized_price,
                self.log_moneyness,
                np.sqrt(np.maximum(self.tau, 1e-8)),
                self.is_call,
                self.strike_to_spot,
                np.log1p(self.contracts),
                np.log1p(self.open_interest),
            ]
        ).astype(np.float32)


@dataclass
class SurfaceRecord:
    template: SurfaceTemplate
    prices: np.ndarray
    source: str
    target_params: np.ndarray | None = None

    @property
    def features(self) -> np.ndarray:
        return self.template.feature_matrix(self.prices)


def read_option_rows(dataset_path: str, symbols: list[str] | None = None, model_ready_only: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path, usecols=REQUIRED_COLUMNS)
    frame = frame.dropna(subset=["strike_price", "time_to_expiry_years", "underlying_value", "observed_option_price", "log_moneyness", "strike_to_spot"])
    if model_ready_only:
        ready = frame["is_model_ready"].astype(str).str.lower().isin(["true", "t", "1"])
        frame = frame.loc[ready].copy()
    if symbols:
        symbol_set = {symbol.upper() for symbol in symbols}
        frame = frame.loc[frame["symbol"].str.upper().isin(symbol_set)].copy()
    frame = frame.loc[(frame["strike_price"] > 0) & (frame["underlying_value"] > 0) & (frame["observed_option_price"] > 0) & (frame["time_to_expiry_years"] > 0)].copy()
    return frame


def _downsample_surface(group: pd.DataFrame, max_surface_points: int) -> pd.DataFrame:
    if len(group) <= max_surface_points:
        return group
    ranked = group.assign(distance_to_atm=np.abs(group["log_moneyness"]))
    return ranked.sort_values(["distance_to_atm", "strike_price"]).head(max_surface_points).sort_values(["log_moneyness", "strike_price"])


def build_surface_records(
    frame: pd.DataFrame,
    risk_free_rate: float,
    dividend_yield: float,
    min_surface_points: int,
    max_surface_points: int,
) -> list[SurfaceRecord]:
    records: list[SurfaceRecord] = []
    for surface_key, group in frame.groupby("surface_key", sort=False):
        group = group.sort_values(["log_moneyness", "strike_price"])
        group = _downsample_surface(group, max_surface_points)
        if len(group) < min_surface_points:
            continue
        option_type = str(group["option_type"].iloc[0]).upper()
        is_call = np.ones(len(group), dtype=np.float32) if option_type == "CE" else np.zeros(len(group), dtype=np.float32)
        template = SurfaceTemplate(
            surface_key=str(surface_key),
            symbol=str(group["symbol"].iloc[0]),
            trade_date=str(group["trade_date"].iloc[0]),
            expiry_date=str(group["expiry_date"].iloc[0]),
            option_type=option_type,
            spot=float(group["underlying_value"].median()),
            rate=float(risk_free_rate),
            dividend=float(dividend_yield),
            strikes=group["strike_price"].to_numpy(dtype=np.float32),
            tau=group["time_to_expiry_years"].to_numpy(dtype=np.float32),
            log_moneyness=group["log_moneyness"].to_numpy(dtype=np.float32),
            strike_to_spot=group["strike_to_spot"].to_numpy(dtype=np.float32),
            is_call=is_call,
            contracts=group["number_of_contracts"].to_numpy(dtype=np.float32),
            open_interest=group["open_interest"].to_numpy(dtype=np.float32),
        )
        records.append(
            SurfaceRecord(
                template=template,
                prices=group["observed_option_price"].to_numpy(dtype=np.float32),
                source="real",
            )
        )
    return records


def verify_zero_leakage(splits: dict[str, list[SurfaceRecord]]) -> bool:
    """Programmatically verify zero date overlap and zero surface key overlap across splits."""
    train_dates = {r.template.trade_date for r in splits.get("train", [])}
    val_dates = {r.template.trade_date for r in splits.get("validation", [])}
    test_dates = {r.template.trade_date for r in splits.get("test", [])}

    train_keys = {r.template.surface_key for r in splits.get("train", [])}
    val_keys = {r.template.surface_key for r in splits.get("validation", [])}
    test_keys = {r.template.surface_key for r in splits.get("test", [])}

    assert len(train_dates & val_dates) == 0, f"Date leakage between train and validation: {train_dates & val_dates}"
    assert len(train_dates & test_dates) == 0, f"Date leakage between train and test: {train_dates & test_dates}"
    assert len(val_dates & test_dates) == 0, f"Date leakage between validation and test: {val_dates & test_dates}"

    assert len(train_keys & val_keys) == 0, f"Surface key leakage between train and validation: {train_keys & val_keys}"
    assert len(train_keys & test_keys) == 0, f"Surface key leakage between train and test: {train_keys & test_keys}"
    assert len(val_keys & test_keys) == 0, f"Surface key leakage between validation and test: {val_keys & test_keys}"

    return True


def split_records_chronologically(
    records: list[SurfaceRecord],
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[SurfaceRecord]]:
    if not records:
        raise ValueError("No records were available for splitting.")
    unique_dates = sorted({record.template.trade_date for record in records})
    n_dates = len(unique_dates)

    if n_dates >= 3:
        train_end = max(1, min(int(n_dates * train_fraction), n_dates - 2))
        val_target = int(n_dates * (train_fraction + validation_fraction))
        validation_end = max(train_end + 1, min(val_target, n_dates - 1))
    elif n_dates == 2:
        train_end = 1
        validation_end = 1
    else:
        train_end = 1
        validation_end = 1

    train_dates = set(unique_dates[:train_end])
    validation_dates = set(unique_dates[train_end:validation_end])
    test_dates = set(unique_dates[validation_end:])

    splits = {
        "train": [record for record in records if record.template.trade_date in train_dates],
        "validation": [record for record in records if record.template.trade_date in validation_dates],
        "test": [record for record in records if record.template.trade_date in test_dates],
    }
    verify_zero_leakage(splits)
    return splits



def cap_records(records: list[SurfaceRecord], cap: int | None, seed: int) -> list[SurfaceRecord]:
    if cap is None or cap <= 0 or len(records) <= cap:
        return records
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=cap, replace=False))
    return [records[index] for index in indices]


class SurfaceDataset(Dataset):
    def __init__(self, records: list[SurfaceRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        template = record.template
        return {
            "surface_key": template.surface_key,
            "symbol": template.symbol,
            "trade_date": template.trade_date,
            "features": record.features.astype(np.float32),
            "market_price": record.prices.astype(np.float32),
            "spot": np.float32(template.spot),
            "strike": template.strikes.astype(np.float32),
            "tau": template.tau.astype(np.float32),
            "rate": np.float32(template.rate),
            "dividend": np.float32(template.dividend),
            "is_call": template.is_call.astype(np.float32),
            "target_params": None if record.target_params is None else record.target_params.astype(np.float32),
        }


def pad_surface_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_points = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]

    features = torch.zeros(batch_size, max_points, feature_dim, dtype=torch.float32)
    market_price = torch.zeros(batch_size, max_points, dtype=torch.float32)
    strike = torch.zeros(batch_size, max_points, dtype=torch.float32)
    tau = torch.zeros(batch_size, max_points, dtype=torch.float32)
    is_call = torch.zeros(batch_size, max_points, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
    spot = torch.zeros(batch_size, dtype=torch.float32)
    rate = torch.zeros(batch_size, dtype=torch.float32)
    dividend = torch.zeros(batch_size, dtype=torch.float32)
    target_params = []
    has_targets = True

    surface_keys: list[str] = []
    symbols: list[str] = []
    trade_dates: list[str] = []

    for batch_index, item in enumerate(batch):
        points = item["features"].shape[0]
        features[batch_index, :points] = torch.from_numpy(item["features"])
        market_price[batch_index, :points] = torch.from_numpy(item["market_price"])
        strike[batch_index, :points] = torch.from_numpy(item["strike"])
        tau[batch_index, :points] = torch.from_numpy(item["tau"])
        is_call[batch_index, :points] = torch.from_numpy(item["is_call"])
        mask[batch_index, :points] = True
        spot[batch_index] = float(item["spot"])
        rate[batch_index] = float(item["rate"])
        dividend[batch_index] = float(item["dividend"])
        if item["target_params"] is None:
            has_targets = False
        else:
            target_params.append(torch.from_numpy(item["target_params"]))
        surface_keys.append(item["surface_key"])
        symbols.append(item["symbol"])
        trade_dates.append(item["trade_date"])

    result: dict[str, Any] = {
        "features": features,
        "market_price": market_price,
        "strike": strike,
        "tau": tau,
        "is_call": is_call,
        "mask": mask,
        "spot": spot,
        "rate": rate,
        "dividend": dividend,
        "surface_key": surface_keys,
        "symbol": symbols,
        "trade_date": trade_dates,
    }
    result["target_params"] = torch.stack(target_params) if has_targets and target_params else None
    return result

