"""Canonical checkpoint staging/readiness without retraining or replacement."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml

from .contracts import G8ReadinessError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "g8_final_real_market.yaml"
CheckpointLoader = Callable[[Path], dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_loader(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _canonical_parameter_order() -> tuple[str, ...]:
    from ..constants import PARAMETER_NAMES

    return tuple(PARAMETER_NAMES)


def _payload_model(expected_method: str) -> str:
    return {
        "MODEL1_ANN": "model1_ordinary_ann",
        "MODEL2_CONSTRAINT_REPRICING_INFORMED": "model2_constraint_repricing_informed",
    }[expected_method]


def checkpoint_readiness_manifest(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    path_overrides: dict[tuple[str, int], Path] | None = None,
    loader: CheckpointLoader | None = None,
) -> dict[str, Any]:
    """Return a machine-readable staging state without executing a model."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config["inverse_method_comparison"]["shared_rules"]["checkpoint_restore_contract"]
    dataset_path = REPOSITORY_ROOT / contract["dataset_path"]
    dataset_hash = _sha256_file(dataset_path) if dataset_path.is_file() else None
    dataset_ok = dataset_hash == contract.get("dataset_sha256")
    overrides = path_overrides or {}
    results: list[dict[str, Any]] = []

    for expected in contract["required_best_validation_checkpoints"]:
        method = expected["method"]
        seed = int(expected["seed"])
        path = overrides.get((method, seed), REPOSITORY_ROOT / expected["path"])
        item: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "path": str(path),
            "expected_registry_sha256": expected["sha256"],
            "recorded_git_sha_required": expected["git_sha"],
            "file_exists": path.is_file(),
            "status": "MISSING",
            "file_sha256": None,
            "hash_approved": False,
            "successful_load_after_hash_approval": False,
            "run_kind": None,
            "recorded_git_sha": None,
            "architecture_present": False,
            "architecture_identity": None,
            "canonical_parameter_order": None,
            "target_standardizer_state_present": False,
            "dataset_identity_bound": dataset_ok,
            "provenance_matches": False,
        }
        if path.is_file():
            actual_hash = _sha256_file(path)
            item.update(
                {
                    "file_sha256": actual_hash,
                    "status": "READY_FOR_LOAD",
                    "hash_approved": actual_hash == expected["sha256"],
                }
            )
            if item["hash_approved"]:
                try:
                    payload = (loader or _default_loader)(path)
                    standardizer = payload.get("target_standardizer", {})
                    parameter_order = tuple(payload.get("parameter_order", ()))
                    spec = payload.get("spec")
                    provenance_matches = all(
                        (
                            isinstance(standardizer, dict)
                            and "mean" in standardizer
                            and "scale" in standardizer,
                            payload.get("run_kind") == "RESEARCH",
                            payload.get("model") == _payload_model(method),
                            int(payload.get("seed", -1)) == seed,
                            payload.get("git_sha") == expected["git_sha"],
                            isinstance(spec, dict) and bool(spec),
                            parameter_order == _canonical_parameter_order(),
                            payload.get("test_set_used_for_selection") is False,
                            payload.get("selection_data") == "validation_only",
                        )
                    )
                    item.update(
                        {
                            "status": "PASS" if provenance_matches else "PROVENANCE_MISMATCH",
                            "successful_load_after_hash_approval": True,
                            "run_kind": payload.get("run_kind"),
                            "recorded_git_sha": payload.get("git_sha"),
                            "architecture_present": isinstance(spec, dict) and bool(spec),
                            "architecture_identity": (
                                hashlib.sha256(
                                    json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
                                ).hexdigest()
                                if isinstance(spec, dict)
                                else None
                            ),
                            "canonical_parameter_order": list(parameter_order),
                            "target_standardizer_state_present": (
                                isinstance(standardizer, dict)
                                and "mean" in standardizer
                                and "scale" in standardizer
                            ),
                            "provenance_matches": provenance_matches,
                        }
                    )
                except Exception as exc:
                    item.update(
                        {
                            "status": "LOAD_FAILED",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            else:
                item["status"] = "HASH_MISMATCH"
        results.append(item)

    statuses = {item["status"] for item in results}
    if "MISSING" in statuses:
        overall = "CHECKPOINT_ARTIFACTS_NOT_STAGED"
    elif statuses <= {"PASS"}:
        overall = "CHECKPOINT_ARTIFACTS_STAGED_AND_VERIFIED"
    elif statuses <= {"PASS", "READY_FOR_LOAD"}:
        overall = "CHECKPOINT_HASHES_NOT_APPROVED"
    else:
        overall = "CHECKPOINT_IDENTITY_INVALID"
    return {
        "schema_version": "g8.checkpoint_readiness/1",
        "config_path": str(config_path),
        "overall_status": overall,
        "all_checks_passed": overall == "CHECKPOINT_ARTIFACTS_STAGED_AND_VERIFIED",
        "dataset_path": str(dataset_path),
        "dataset_sha256_matches": dataset_ok,
        "checkpoint_count": len(results),
        "results": results,
        "market_data_read": False,
        "pricing_executed": False,
        "calibration_executed": False,
        "evaluation_executed": False,
        "neural_weights_updated": False,
    }


def stage_canonical_checkpoint(
    expected: dict[str, Any],
    *,
    source_path: Path,
    approve_expected_hash: bool,
) -> Path:
    """Copy an externally supplied byte-identical checkpoint into its registry path."""
    if not approve_expected_hash:
        raise G8ReadinessError("external checkpoint transfer requires explicit expected-hash approval")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_hash = _sha256_file(source)
    if actual_hash != expected["sha256"]:
        raise G8ReadinessError(
            f"source checkpoint SHA mismatch: expected {expected['sha256']}, got {actual_hash}"
        )
    destination = REPOSITORY_ROOT / str(expected["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_hash = _sha256_file(destination)
        if existing_hash != expected["sha256"]:
            raise G8ReadinessError(f"destination exists with noncanonical content: {destination}")
        return destination
    temporary = destination.with_name(destination.name + ".staging")
    shutil.copyfile(source, temporary)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
