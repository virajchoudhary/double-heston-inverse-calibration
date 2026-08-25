"""Fail-closed validation helpers for the frozen G8 protocol.

The tool deliberately knows the protocol and exclusion registry, but not any
future market values or model outcomes.  ``validate-config`` performs static
checks.  ``check-candidate`` checks a proposed symbol/date identity without
downloading, pricing, calibrating, or printing market observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "g8_final_real_market.yaml"
EXPECTED_BASE = "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
EXPECTED_CONFIG_SHA256 = "d6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37"
EXPECTED_REPRESENTATION = "FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE"
PRIMARY_SYMBOLS = ("NTPC", "CIPLA", "INFY", "HDFCBANK")
STAGE_A_SYMBOLS = (
    "NTPC", "POWERGRID", "SUNPHARMA", "CIPLA",
    "INFY", "TCS", "ICICIBANK", "HDFCBANK",
)
STAGE_A_DATES = ("2026-07-01", "2026-07-15", "2026-07-22")
POWER_EXTENSION_DATES = ("2026-07-08", "2026-07-29")
CANONICAL_PARAMETER_NAMES = (
    "kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
    "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast",
)
EXPECTED_METHOD_PAYLOAD_MODELS = {
    "MODEL1_ANN": "model1_ordinary_ann",
    "MODEL2_CONSTRAINT_REPRICING_INFORMED": "model2_constraint_repricing_informed",
}


class G8ProtocolValidationError(ValueError):
    """A frozen protocol invariant is absent or violated."""


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G8ProtocolValidationError("G8 config root must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8ProtocolValidationError(message)


def validate_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate the frozen scientific/config invariants without market access."""
    config = _load_mapping(path)
    _require(config.get("schema_version") == "1.0", "unexpected schema_version")
    _require(config.get("contract_name") == "G8_FINAL_REAL_MARKET_PROTOCOL", "wrong contract")
    _require(config.get("protocol_version") == "1.0", "wrong protocol version")
    _require(config.get("status") == "FROZEN_PENDING_UNTOUCHED_DATA_ACQUISITION", "protocol is not frozen-pending-acquisition")
    _require(config.get("base_commit") == EXPECTED_BASE, "unexpected base commit")
    execution = config.get("execution", {})
    _require(execution.get("authorized_now") is False, "G8 execution must not be authorized by this config")
    _require(execution.get("data_acquisition_authorized_now") is False, "G8 acquisition must not be authorized by this config")
    _require(execution.get("final_evaluation_started") is False, "final evaluation marker must be false")
    _require(execution.get("neural_weight_updates_on_real_data") is False, "real-data weight updates must be false")
    _require(execution.get("architecture_or_hyperparameter_tuning_on_g8") is False, "G8 tuning must be false")

    exclusions = config.get("development_exclusions", {})
    _require(exclusions.get("hard_floor_for_all_g8_valuation_dates") == "2026-09-30", "G8 date floor changed")
    _require(exclusions.get("overlap_policy") == "FAIL_CLOSED_ON_SYMBOL_DATE_OR_CONTRACT_PROVENANCE_OVERLAP", "overlap policy is not fail closed")
    original = exclusions.get("stage_a_original", {})
    _require(set(original.get("symbols", ())) == set(STAGE_A_SYMBOLS), "Stage A exclusion universe drifted")
    _require(set(original.get("valuation_dates", ())) == set(STAGE_A_DATES), "Stage A exclusion dates drifted")
    extension = exclusions.get("power_tiebreak_extension", {})
    _require(set(extension.get("symbols", ())) == {"NTPC", "POWERGRID"}, "Power tie-break exclusion drifted")
    _require(set(extension.get("valuation_dates", ())) == set(POWER_EXTENSION_DATES), "Power tie-break dates drifted")
    pilot = exclusions.get("ntpc_single_stock_pilot", {})
    _require(pilot.get("classification") == "DEVELOPMENT_PILOT_NEVER_FINAL_EVALUATION", "NTPC pilot mislabeled")
    _require(pilot.get("valuation_date") == "2026-07-15", "NTPC pilot date drifted")
    _require(pilot.get("realized_close_window", {}).get("last_date_inclusive") == "2026-07-28", "NTPC realized-window exclusion drifted")

    universe = config.get("eligible_universe", {})
    _require(tuple(universe.get("primary_symbols", ())) == PRIMARY_SYMBOLS, "primary universe/order drifted")
    selection = config.get("deterministic_selection", {})
    _require(selection.get("search_start_inclusive") == "2026-09-30", "selection start changed")
    _require(selection.get("target_common_dates") == 2, "common-date count changed")
    _require(selection.get("maximum_surfaces") == 8, "maximum surface count changed")
    _require(selection.get("stop_after_target_common_dates_without_inspecting_model_results") is True, "selection may inspect outcomes")

    construction = config.get("surface_construction", {})
    _require(construction.get("representation") == EXPECTED_REPRESENTATION, "noncanonical representation")
    _require(construction.get("representation_version") == "1.0", "representation version changed")
    _require(construction.get("nominal_slot_count") == 20, "nominal slot count changed")
    _require(construction.get("masked_slots") == "explicit_false_with_zero_placeholder_no_imputation", "mask policy changed")
    roles = config.get("observation_roles", {})
    _require(roles.get("pricing_model_family_calibration", {}).get("target_log_moneyness") == [-0.05, 0.0, 0.05], "family calibration role changed")
    _require(roles.get("pricing_model_family_holdout", {}).get("target_log_moneyness") == [-0.10, 0.10], "family holdout role changed")
    inverse = config.get("inverse_method_comparison", {})
    _require(inverse.get("shared_rules", {}).get("no_real_weight_updates") is True, "inverse weight-update guard changed")
    _require(inverse.get("interpretation", {}).get("real_parameter_winner_permitted") is False, "real truth-winner guard changed")

    blocker = config.get("data_blocker", {})
    _require(blocker.get("status") == "BLOCKED_PENDING_PROTOCOL_COMPLIANT_ACQUISITION", "untouched-data blocker is absent")
    actual_hash = sha256_file(path)
    _require(
        actual_hash == EXPECTED_CONFIG_SHA256,
        f"G8 config hash drift: expected {EXPECTED_CONFIG_SHA256}, got {actual_hash}",
    )
    checkpoints = (
        config.get("inverse_method_comparison", {})
        .get("shared_rules", {})
        .get("checkpoint_restore_contract", {})
        .get("required_best_validation_checkpoints", [])
    )
    _require(len(checkpoints) == 6, "exactly six neural checkpoints must be bound")
    _require(len({item.get("sha256") for item in checkpoints}) == 6, "neural checkpoint hashes must be unique")
    return config


def is_development_observation(symbol: str, valuation_date: str) -> tuple[bool, str]:
    """Return whether a proposed valuation identity overlaps known development use."""
    try:
        day = date.fromisoformat(valuation_date)
    except ValueError as exc:
        raise G8ProtocolValidationError(f"invalid ISO date: {valuation_date}") from exc
    normalized_symbol = symbol.strip().upper()
    floor = date.fromisoformat("2026-09-30")
    if day < floor:
        return True, "BEFORE_G8_DATE_FLOOR"
    if normalized_symbol == "NIFTY":
        return True, "NIFTY_REFERENCE_ONLY_PROHIBITED"
    if normalized_symbol in STAGE_A_SYMBOLS and valuation_date in STAGE_A_DATES:
        return True, "STAGE_A_ORIGINAL_DEVELOPMENT_PANEL"
    if normalized_symbol in {"NTPC", "POWERGRID"} and valuation_date in POWER_EXTENSION_DATES:
        return True, "POWER_TIEBREAK_DEVELOPMENT_PANEL"
    if normalized_symbol == "NTPC" and date.fromisoformat("2026-07-15") <= day <= date.fromisoformat("2026-07-28"):
        return True, "NTPC_SINGLE_STOCK_PILOT_REALIZED_OR_VALUATION_WINDOW"
    return False, ""


def check_candidate(symbol: str, valuation_date: str) -> dict[str, str | bool]:
    """Check one identity; no market data is read and no outcome is exposed."""
    excluded, reason = is_development_observation(symbol, valuation_date)
    return {
        "symbol": symbol.strip().upper(),
        "valuation_date": valuation_date,
        "development_excluded": excluded,
        "reason": reason,
        "contract_key_overlap_checked": False,
        "market_data_read": False,
        "model_executed": False,
    }


def verify_checkpoint_registry(config: dict[str, Any]) -> dict[str, Any]:
    """Verify frozen neural restoration artifacts; never run or evaluate a model."""
    contract = config["inverse_method_comparison"]["shared_rules"][
        "checkpoint_restore_contract"
    ]
    dataset_path = REPOSITORY_ROOT / contract["dataset_path"]
    dataset_hash = sha256_file(dataset_path) if dataset_path.is_file() else None
    dataset_ok = dataset_hash == contract["dataset_sha256"]
    results: list[dict[str, Any]] = []

    for expected in contract["required_best_validation_checkpoints"]:
        path = REPOSITORY_ROOT / str(expected["path"])
        result: dict[str, Any] = {
            "method": expected["method"],
            "seed": expected["seed"],
            "path": expected["path"],
            "status": "MISSING" if not path.is_file() else "READY_FOR_HASH_CHECK",
            "file_exists": path.is_file(),
            "sha256_matches": False,
            "loaded": False,
            "provenance_matches": False,
        }
        if not path.is_file():
            results.append(result)
            continue
        actual_hash = sha256_file(path)
        result["actual_sha256"] = actual_hash
        result["sha256_matches"] = actual_hash == expected["sha256"]
        if not result["sha256_matches"]:
            result["status"] = "HASH_MISMATCH"
            results.append(result)
            continue

        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        standardizer = payload.get("target_standardizer", {})
        parameter_order = tuple(payload.get("parameter_order", ()))
        provenance_ok = all(
            (
                isinstance(standardizer, dict) and "mean" in standardizer and "scale" in standardizer,
                payload.get("run_kind") == "RESEARCH",
                payload.get("model") == EXPECTED_METHOD_PAYLOAD_MODELS[expected["method"]],
                int(payload.get("seed", -1)) == int(expected["seed"]),
                payload.get("git_sha") == expected["git_sha"],
                isinstance(payload.get("spec"), dict) and bool(payload["spec"]),
                parameter_order == CANONICAL_PARAMETER_NAMES,
                payload.get("test_set_used_for_selection") is False,
                payload.get("selection_data") == "validation_only",
            )
        )
        result.update(
            {
                "status": "PASS" if provenance_ok else "PROVENANCE_MISMATCH",
                "loaded": True,
                "standardizer_state_present": isinstance(standardizer, dict)
                and "mean" in standardizer
                and "scale" in standardizer,
                "provenance_matches": provenance_ok,
            }
        )
        results.append(result)

    passed = dataset_ok and bool(results) and all(item["status"] == "PASS" for item in results)
    return {
        "command": "check-checkpoints",
        "all_checks_passed": passed,
        "dataset_path": contract["dataset_path"],
        "dataset_sha256_matches": dataset_ok,
        "checkpoint_count": len(results),
        "results": results,
        "market_data_read": False,
        "pricing_executed": False,
        "calibration_executed": False,
        "evaluation_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config")
    candidate = subparsers.add_parser("check-candidate")
    candidate.add_argument("symbol")
    candidate.add_argument("valuation_date", help="YYYY-MM-DD")
    subparsers.add_parser("check-checkpoints")
    args = parser.parse_args()

    if args.command == "check-checkpoints":
        config = validate_config(args.config)
        payload = verify_checkpoint_registry(config)
    elif args.command == "validate-config":
        config = validate_config(args.config)
        payload = {
            "valid": True,
            "contract_name": config["contract_name"],
            "status": config["status"],
            "config_sha256": sha256_file(args.config),
            "market_data_read": False,
            "model_executed": False,
        }
    else:
        validate_config(args.config)
        payload = check_candidate(args.symbol, args.valuation_date)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("valid", payload.get("all_checks_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
