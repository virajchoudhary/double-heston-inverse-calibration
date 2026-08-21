from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/default_experiment.json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        defaults = json.load(handle)
    if default_path == DEFAULT_CONFIG_PATH:
        return defaults
    with Path(path).open("r", encoding="utf-8") as handle:
        override = json.load(handle)
    return _deep_merge(defaults, override)

