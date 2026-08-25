"""Sealed evaluation infrastructure for the frozen OOD/boundary cohorts.

This module deliberately contains no research result.  The default execution
path cannot open the frozen research cohorts; research evaluation requires an
explicit future authorization with both CLI flags described in
``configs/ood_boundary_evaluation_ready.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import PARAMETER_NAMES
from .constraints import validate_parameters
from .calibrate_double_heston import calibrate_double_heston
from .ood_boundary_protocol import parameter_vector_hash
from .r2_primary.calibration import FROZEN_SETTINGS, select_representatives
from .r2_primary.dataset import R2PrimaryDataset, R2SurfaceItem, build_r2_features
from .r2_primary.evaluation import (
    constraint_validity_metrics,
    parameter_recovery_metrics,
    reprice_normalized,
    repricing_metrics,
    train_split_scaling,
)
from .r2_primary.training import MODEL1_SPEC, MODEL2_SPEC, load_run, predict_parameters
from .r2_representation.serialization import payload_to_surface, validate_payload
from .utils import set_deterministic_seed


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_PATH = ROOT / "configs" / "ood_boundary_protocol.yaml"
EVALUATION_CONFIG_PATH = ROOT / "configs" / "ood_boundary_evaluation_ready.yaml"
PRIMARY_CONFIG_PATH = ROOT / "configs" / "r2_primary_comparison_FINAL.yaml"
PRIMARY_DATASET_PATH = ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
PRIMARY_MANIFEST_PATH = (
    ROOT / "evidence" / "r2_primary_comparison_20260823" / "training_run_manifest.json"
)
COHORT_ROOT = ROOT / "evidence" / "ood_boundary_protocol_v1"
COHORT_REPLAY_ROOT = ROOT / "evidence" / "ood_boundary_protocol_v1_replay"
READY_ROOT = ROOT / "evidence" / "ood_boundary_evaluation_ready"
MODEL3_MANIFEST_PATH = ROOT / "checkpoints" / "model3_research" / "manifest.json"

RESEARCH_COHORT_AUTHORIZATION_PHRASE = "AUTHORIZE_FROZEN_OOD_RESEARCH_EVALUATION_V1"
NEURAL_SEEDS = (11, 22, 33)
ACTIVE_PARAMETER_COHORTS = (
    "boundary_challenge",
    "distribution_shift",
    "maturity_conditioning_shift",
    "incomplete_observation",
)
TRADITIONAL_SUBSET_COUNTS = {
    "boundary_challenge": 15,
    "distribution_shift": 15,
    "maturity_conditioning_shift": 15,
    "incomplete_observation": 15,
}
RESULT_SCHEMA_VERSION = "1.0"


class EvaluationSealError(RuntimeError):
    """Raised when an operation would violate the sealed-cohort contract."""


class ResearchEvaluationLocked(EvaluationSealError):
    """Raised when frozen research evaluation is not explicitly authorized."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def deterministic_json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return (text + "\n").encode("utf-8")


def _jsonable_metrics(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: _jsonable_metrics(item)
            for key, item in value.items()
            if key != "per_surface_rmse"
        }
    if isinstance(value, list):
        return [_jsonable_metrics(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(deterministic_json_bytes(payload))
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n", float_format="%.17g")
    return sha256_file(target)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def _hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def load_ready_config() -> dict[str, Any]:
    config = yaml.safe_load(EVALUATION_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("contract_name") != "OOD_BOUNDARY_EVALUATION_READY":
        raise EvaluationSealError("unexpected evaluation-ready config")
    if config.get("status") != "EVALUATION_INFRASTRUCTURE_SEALED_NO_RESULTS":
        raise EvaluationSealError("evaluation infrastructure is not sealed")
    identities = freeze_identities()
    if config["frozen_protocol"]["config_sha256"] != identities[
        "protocol_config_sha256"
    ]:
        raise EvaluationSealError("evaluation config references a stale OOD protocol")
    if config["frozen_protocol"]["cohort_manifest_sha256"] != identities[
        "primary_cohort_manifest_sha256"
    ]:
        raise EvaluationSealError("evaluation config references stale cohort evidence")
    if config["frozen_protocol"]["all_research_surfaces_sha256"] != identities[
        "all_research_surfaces_jsonl_sha256"
    ]:
        raise EvaluationSealError("evaluation config references stale research cohort")
    frozen_authority = yaml.safe_load(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if identities["primary_config_sha256"].lower() != frozen_authority[
        "authority"
    ]["primary_comparison_config"]["sha256"].lower():
        raise EvaluationSealError("frozen OOD protocol's primary-config pin changed")
    if sha256_file(PRIMARY_MANIFEST_PATH) != config["frozen_protocol"][
        "primary_training_manifest_sha256"
    ]:
        raise EvaluationSealError("primary training manifest identity is not pinned")
    lock_requirements = config["research_execution_lock"]["research_requires_all_of"]
    exact_confirmation = next(
        item["exact_confirmation"]
        for item in lock_requirements
        if isinstance(item, dict) and "exact_confirmation" in item
    )
    if exact_confirmation != (
        RESEARCH_COHORT_AUTHORIZATION_PHRASE
    ):
        raise EvaluationSealError("research authorization phrase drift")
    return config


def check_research_authorization(
    *, cohort: str, authorize: bool, confirmation: str
) -> dict[str, Any]:
    """Authorize research only after three explicit, non-default CLI choices."""
    if cohort != "research":
        raise ResearchEvaluationLocked("only --cohort development-fixture is available now")
    if not authorize:
        raise ResearchEvaluationLocked(
            "frozen research evaluation requires --authorize-frozen-evaluation"
        )
    if confirmation != RESEARCH_COHORT_AUTHORIZATION_PHRASE:
        raise ResearchEvaluationLocked(
            "frozen research evaluation confirmation phrase is missing or incorrect"
        )
    return {
        "mechanism": "explicit_cohort_authorize_flag_exact_phrase",
        "authorization_phrase_sha256": sha256_bytes(
            confirmation.encode("utf-8")
        ),
        "environment_variable_used": False,
        "authorized_at_utc": datetime.now(UTC).isoformat(),
    }


def freeze_identities() -> dict[str, Any]:
    """Return structural identities and hashes without computing any metric."""
    from .ood_boundary_protocol import validate_generated_output

    primary_validation = validate_generated_output(COHORT_ROOT)
    replay_validation = validate_generated_output(COHORT_REPLAY_ROOT)
    manifest_path = COHORT_ROOT / "manifest.json"
    replay_report_path = COHORT_ROOT / "replay_report.json"
    replay_report = read_json(replay_report_path)
    if replay_report.get("identical") is not True:
        raise EvaluationSealError("authoritative OOD cohort replay is not identical")
    artifact_files = {
        "clean_surfaces_jsonl_sha256": "clean_surfaces.jsonl",
        "incomplete_surfaces_jsonl_sha256": "incomplete_surfaces.jsonl",
        "all_research_surfaces_jsonl_sha256": "all_research_surfaces.jsonl",
        "development_sanity_panel_csv_sha256": "development_sanity_panel.csv",
        "selected_parameters_csv_sha256": "selected_parameters.csv",
    }
    hashes = {
        key: sha256_file(COHORT_ROOT / filename)
        for key, filename in artifact_files.items()
    }
    counts = dict(primary_validation["manifest"]["counts"])
    counts["development_sanity_parameter_rows"] = 12
    counts["development_sanity_surfaces"] = 0
    expected_counts = {
        "boundary_challenge": 120,
        "distribution_shift": 120,
        "maturity_conditioning_shift": 120,
        "incomplete_observation": 60,
        "serialized_research_total": 420,
        "clean_pricing_calls": 360,
        "pricing_failure_count": 0,
        "development_sanity_rows": 12,
        "development_sanity_parameter_rows": 12,
        "development_sanity_surfaces": 0,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise EvaluationSealError(f"frozen cohort count mismatch: {counts}")
    assert_no_frozen_result_artifacts()
    return {
        "protocol_config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "primary_config_sha256": sha256_file(PRIMARY_CONFIG_PATH),
        "r2_dataset_sha256": sha256_file(PRIMARY_DATASET_PATH),
        "primary_cohort_manifest_sha256": sha256_file(manifest_path),
        "replay_report_sha256": sha256_file(replay_report_path),
        **hashes,
        "counts": counts,
        "replay_identical": True,
        "structural_validation_passed": bool(
            primary_validation["clean"] == 360
            and primary_validation["incomplete"] == 60
            and replay_validation["clean"] == 360
        ),
        "research_model_metrics_present": False,
    }


def assert_no_frozen_result_artifacts() -> None:
    """Fail if prediction/metric artifacts entered either immutable cohort tree."""
    forbidden_tokens = ("prediction", "predictions", "method_metrics", "result_metrics")
    for root in (COHORT_ROOT, COHORT_REPLAY_ROOT):
        for path in root.rglob("*"):
            if path.is_file() and any(token in path.name.lower() for token in forbidden_tokens):
                raise EvaluationSealError(
                    f"unexpected frozen-research outcome artifact already exists: {path}"
                )


@dataclass(frozen=True)
class OODCohort:
    kind: str
    path: Path
    sha256: str
    items: list[R2SurfaceItem]
    records: list[dict[str, Any]]
    surface_id_order_sha256: str
    parent_items: dict[str, R2SurfaceItem]

    @property
    def truths(self) -> np.ndarray:
        return np.stack([item.targets for item in self.items])


def _record_to_ood_item(record: Mapping[str, Any], split: str) -> R2SurfaceItem:
    from .constants import PARAMETER_NAMES

    validate_payload(record)
    metadata = record["metadata"]
    stored_parameters = metadata["parameters_canonical_order"]
    targets = np.asarray(
        [float(stored_parameters[name]) for name in PARAMETER_NAMES],
        dtype=np.float64,
    )
    diagnostics = validate_parameters(targets)
    if not diagnostics["is_valid"]:
        raise EvaluationSealError("frozen cohort truth violates canonical constraints")
    surface = payload_to_surface(record)
    rates = surface.rates_array()
    carries = surface.carries_array()
    if not bool(np.all(rates == rates[0])) or not bool(np.all(carries == carries[0])):
        # The current synthetic cohorts are rank-constant.  A future per-rank
        # extension requires a separate pricer adapter rather than silent use.
        raise EvaluationSealError("non-rank-constant rate/carry conditioning rejected")
    mask = surface.mask_array()
    normalized_prices = np.where(mask, surface.prices_array(), 0.0)
    strikes = surface.spot * np.exp(
        np.asarray([key.target_log_moneyness for key in surface.slot_keys])
    )
    user_metadata = metadata["user_metadata"]
    return R2SurfaceItem(
        surface_id=str(record["surface_id"]),
        split=split,
        features=build_r2_features(record),
        targets=targets,
        mask=mask,
        dollar_prices=normalized_prices * surface.spot,
        normalized_prices=normalized_prices,
        strikes=strikes,
        maturities=surface.maturities_array(),
        option_types=[key.option_type for key in surface.slot_keys],
        spot=float(surface.spot),
        rate=float(rates[0]),
        carry=float(carries[0]),
        parameter_vector_hash=str(user_metadata.get("parameter_vector_hash", "")),
    )


def _surface_order_hash(items: list[R2SurfaceItem]) -> str:
    payload = "\n".join(item.surface_id for item in items) + "\n"
    return sha256_bytes(payload.encode("utf-8"))


def load_development_fixture(path: str | Path) -> OODCohort:
    source = Path(path)
    records = [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]
    items = [_record_to_ood_item(record, "development_fixture") for record in records]
    if len(items) < 3 or not all(sum(item.mask) >= 10 for item in items):
        raise EvaluationSealError("invalid development fixture")
    return OODCohort(
        kind="development_fixture",
        path=source.resolve(),
        sha256=sha256_file(source),
        items=items,
        records=records,
        surface_id_order_sha256=_surface_order_hash(items),
        parent_items={},
    )


def load_frozen_research_cohort(
    *, authorized: bool, identity_manifest_path: str | Path = READY_ROOT / "freeze_identity.json"
) -> OODCohort:
    """Load research data only after authorization and identity validation."""
    load_ready_config()
    if not authorized:
        raise ResearchEvaluationLocked("frozen research cohort loading is locked")
    identity = read_json(identity_manifest_path)
    if identity.get("status") != "FROZEN_OOD_STRUCTURAL_IDENTITY_VERIFIED":
        raise EvaluationSealError("frozen OOD identity manifest is not approved")
    source = COHORT_ROOT / "all_research_surfaces.jsonl"
    if sha256_file(source) != identity["artifacts"]["all_research_surfaces_jsonl_sha256"]:
        raise EvaluationSealError("frozen research cohort hash changed")
    records = [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]
    items = [_record_to_ood_item(record, "frozen_research") for record in records]
    order_hash = _surface_order_hash(items)
    if order_hash != identity["surface_id_order_sha256"]:
        raise EvaluationSealError("frozen research surface ordering changed")
    clean_items = [item for item in items if len(item.mask) == sum(item.mask)]
    parent_items = {item.surface_id: item for item in clean_items}
    return OODCohort(
        kind="frozen_research",
        path=source.resolve(),
        sha256=sha256_file(source),
        items=items,
        records=records,
        surface_id_order_sha256=order_hash,
        parent_items=parent_items,
    )


def _safe_readiness_root(output_directory: str | Path) -> Path:
    resolved = Path(output_directory).resolve()
    protected = [COHORT_ROOT.resolve(), COHORT_REPLAY_ROOT.resolve()]
    if any(resolved == target or target in resolved.parents for target in protected):
        raise EvaluationSealError(
            f"refusing to write readiness artifacts under frozen evidence: {resolved}"
        )
    return resolved


def prepare_freeze_identity(output_directory: str | Path = READY_ROOT) -> Path:
    _safe_readiness_root(output_directory)
    identities = freeze_identities()
    records = [
        json.loads(line)
        for line in (COHORT_ROOT / "all_research_surfaces.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    ids = [str(record["surface_id"]) for record in records]
    if len(ids) != 420 or len(set(ids)) != 420:
        raise EvaluationSealError("cannot establish unique immutable surface order")
    payload = {
        "schema_version": "1.0",
        "status": "FROZEN_OOD_STRUCTURAL_IDENTITY_VERIFIED",
        "result_status": "NO_METHOD_RESULTS_OPENED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_is_rng_input": False,
        "git_commit_sha": _git_head(),
        "surface_id_order_sha256": sha256_bytes(
            ("\n".join(ids) + "\n").encode("utf-8")
        ),
        "surface_ids_are_sealed": True,
        "artifacts": {
            key: value
            for key, value in identities.items()
            if key.endswith("_sha256")
        },
        "counts": identities["counts"],
        "replay_identical": identities["replay_identical"],
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    return write_json(output / "freeze_identity.json", payload)


def prepare_reference_scaling(output_directory: str | Path = READY_ROOT) -> Path:
    _safe_readiness_root(output_directory)
    dataset = R2PrimaryDataset.from_jsonl(
        PRIMARY_DATASET_PATH, splits={"train"}
    )
    scaling = train_split_scaling(dataset)
    if any(entry["range"] <= 0 or entry["std"] <= 0 for entry in scaling.values()):
        raise EvaluationSealError("invalid ID reference scaling")
    payload = {
        "schema_version": "1.0",
        "status": "TRAIN_SPLIT_ONLY_REFERENCE_SCALING",
        "source_path": PRIMARY_DATASET_PATH.as_posix(),
        "source_sha256": sha256_file(PRIMARY_DATASET_PATH),
        "split": "train_only",
        "parameter_order": list(PARAMETER_NAMES),
        "scaling": scaling,
        "note": "truth-panel scaling only; contains no model outcome",
    }
    return write_json(Path(output_directory) / "reference_scaling.json", payload)


def load_reference_scaling(
    path: str | Path = READY_ROOT / "reference_scaling.json",
) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    if payload.get("status") != "TRAIN_SPLIT_ONLY_REFERENCE_SCALING":
        raise EvaluationSealError("invalid reference-scaling artifact")
    if payload["source_sha256"] != sha256_file(PRIMARY_DATASET_PATH):
        raise EvaluationSealError("reference scaling source hash mismatch")
    if list(payload["parameter_order"]) != PARAMETER_NAMES:
        raise EvaluationSealError("reference scaling parameter-order mismatch")
    return payload["scaling"]


def checkpoint_catalog() -> dict[str, Any]:
    catalog = {"model1": {}, "model2": {}}
    manifest = read_json(PRIMARY_MANIFEST_PATH)
    expected_manifest_hash = load_ready_config()["frozen_protocol"][
        "primary_training_manifest_sha256"
    ]
    if sha256_file(PRIMARY_MANIFEST_PATH) != expected_manifest_hash:
        raise EvaluationSealError("primary training manifest hash mismatch")
    for method in ("model1", "model2"):
        for seed in NEURAL_SEEDS:
            seed_key = str(seed)
            if seed_key not in manifest[method]:
                raise EvaluationSealError(f"primary manifest lacks {method} seed {seed}")
            entry = manifest[method][seed_key]
            relative = Path("checkpoints") / "r2_primary_comparison" / f"{method}_seed{seed}"
            checkpoint_path = ROOT / relative / "best_validation_checkpoint.pt"
            present = checkpoint_path.is_file()
            actual_hash = sha256_file(checkpoint_path) if present else None
            catalog[method][seed] = {
                "seed": seed,
                "relative_path": relative.as_posix(),
                "checkpoint_path": checkpoint_path,
                "present": present,
                "expected_checkpoint_sha256": entry["checkpoint_sha256"],
                "actual_checkpoint_sha256": actual_hash,
                "identity_matches": bool(
                    present and actual_hash == entry["checkpoint_sha256"]
                ),
                "training_git_sha": entry["git_sha"],
                "run_kind_expected": entry.get("run_kind", "RESEARCH"),
            }
    return catalog


def checkpoint_readiness(output_directory: str | Path = READY_ROOT) -> Path:
    _safe_readiness_root(output_directory)
    catalog = checkpoint_catalog()
    missing = [
        f"{method}/{seed}"
        for method, seeds in catalog.items()
        for seed, entry in seeds.items()
        if not entry["present"]
    ]
    mismatched = [
        f"{method}/{seed}"
        for method, seeds in catalog.items()
        for seed, entry in seeds.items()
        if entry["present"] and not entry["identity_matches"]
    ]
    payload = {
        "schema_version": "1.0",
        "status": (
            "READY_ALL_SEEDS_PRESENT_AND_HASH_MATCHED"
            if not missing and not mismatched
            else "BLOCKED_CHECKPOINTS_NOT_AVAILABLE_LOCALLY"
        ),
        "required_seeds": list(NEURAL_SEEDS),
        "selection_policy": "all_seeds_or_method_is_blocked_no_cherry_picking",
        "missing_local_checkpoints": missing,
        "hash_mismatches": mismatched,
        "catalog": {
            method: {
                str(seed): {
                    **entry,
                    "checkpoint_path": str(entry["checkpoint_path"]),
                }
                for seed, entry in seeds.items()
            }
            for method, seeds in catalog.items()
        },
    }
    return write_json(Path(output_directory) / "checkpoint_readiness.json", payload)


def _validate_neural_payload(
    method: str, seed: int, payload: Mapping[str, Any], catalog_entry: Mapping[str, Any]
) -> None:
    expected_spec = MODEL1_SPEC if method == "model1" else MODEL2_SPEC
    actual_spec = payload.get("spec", {})
    identity_fields = ("hidden_sizes", "activation", "dropout")
    if any(actual_spec.get(field) != expected_spec[field] for field in identity_fields):
        raise EvaluationSealError(f"{method} seed {seed}: architecture identity mismatch")
    if payload.get("seed") != seed:
        raise EvaluationSealError(f"{method}: checkpoint seed mismatch")
    if payload.get("run_kind") != "RESEARCH":
        raise EvaluationSealError(f"{method}: non-research checkpoint refused")
    if payload.get("git_sha") != catalog_entry["training_git_sha"]:
        raise EvaluationSealError(f"{method}: training Git SHA mismatch")
    if tuple(payload.get("parameter_order", [])) != PARAMETER_NAMES:
        raise EvaluationSealError("checkpoint parameter-order mismatch")
    standardizer = payload.get("target_standardizer", {})
    mean = torch.as_tensor(standardizer.get("mean", []))
    scale = torch.as_tensor(standardizer.get("scale", []))
    if mean.shape != (10,) or scale.shape != (10,):
        raise EvaluationSealError("checkpoint standardizer shape mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise EvaluationSealError("checkpoint standardizer is non-finite")
    if bool((scale <= 0).any()):
        raise EvaluationSealError("checkpoint standardizer has non-positive scale")


def run_neural_adapter(
    method: str,
    cohort: OODCohort,
    *,
    device: str = "cpu",
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]], dict[str, Any]]:
    """Run every required frozen seed or fail closed without partial selection."""
    if method not in {"model1", "model2"}:
        raise ValueError("neural method must be model1 or model2")
    catalog = checkpoint_catalog()[method]
    blockers = []
    for seed in NEURAL_SEEDS:
        entry = catalog[seed]
        if not entry["present"]:
            blockers.append(f"missing:{entry['relative_path']}")
        elif not entry["identity_matches"]:
            blockers.append(f"hash_mismatch:{entry['relative_path']}")
    predictions: dict[int, np.ndarray] = {}
    runtimes: dict[int, dict[str, float]] = {}
    if blockers:
        return predictions, runtimes, {
            "status": "NOT_AVAILABLE_BLOCKED_CHECKPOINTS",
            "blockers": blockers,
            "checkpoint_identities": {
                str(seed): {
                    "present": catalog[seed]["present"],
                    "expected_checkpoint_sha256": catalog[seed][
                        "expected_checkpoint_sha256"
                    ],
                    "actual_checkpoint_sha256": catalog[seed][
                        "actual_checkpoint_sha256"
                    ],
                    "training_git_sha": catalog[seed]["training_git_sha"],
                }
                for seed in NEURAL_SEEDS
            },
        }
    resolved_device = torch.device(device)
    set_deterministic_seed(min(NEURAL_SEEDS))
    for seed in NEURAL_SEEDS:
        run = load_run(
            ROOT / "checkpoints" / "r2_primary_comparison" / f"{method}_seed{seed}",
            method,
        )
        _validate_neural_payload(method, seed, run["payload"], catalog[seed])
        indices = list(range(len(cohort.items)))
        started = time.perf_counter()
        predicted = predict_parameters(
            run["model"],
            cohort.items,  # type: ignore[arg-type]
            indices,
            standardizer=run["standardizer"] if method == "model1" else None,
            device=str(resolved_device),
        )
        elapsed = time.perf_counter() - started
        if predicted.shape != (len(cohort.items), 10):
            raise EvaluationSealError("aligned prediction shape failure")
        predictions[seed] = predicted
        runtimes[seed] = {
            "full_split_inference_seconds": elapsed,
            "per_surface_inference_ms_amortized": 1000.0 * elapsed / len(indices),
            "device": str(resolved_device),
        }
    return predictions, runtimes, {
        "status": "COMPLETE",
        "blockers": [],
        "training_protocol_config": PRIMARY_CONFIG_PATH.as_posix(),
        "checkpoint_identities": {
            str(seed): {
                "seed": seed,
                "checkpoint_sha256": catalog[seed]["actual_checkpoint_sha256"],
                "training_git_sha": catalog[seed]["training_git_sha"],
                "training_config_path": PRIMARY_CONFIG_PATH.as_posix(),
                "training_config_sha256": sha256_file(PRIMARY_CONFIG_PATH),
            }
            for seed in NEURAL_SEEDS
        },
    }


MODEL3_REQUIRED_MANIFEST_FIELDS = (
    "status",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "training_git_sha",
    "training_config_sha256",
    "training_dataset_sha256",
    "seeds",
    "parameter_output_contract",
    "inference",
    "approved_for_frozen_ood_evaluation",
)


def model3_readiness(output_directory: str | Path = READY_ROOT) -> Path:
    _safe_readiness_root(output_directory)
    if not MODEL3_MANIFEST_PATH.is_file():
        status = "WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS"
        blockers = ["manifest_absent"]
    else:
        try:
            manifest = read_json(MODEL3_MANIFEST_PATH)
            missing = [
                field for field in MODEL3_REQUIRED_MANIFEST_FIELDS if field not in manifest
            ]
            if missing:
                status = "BLOCKED_INVALID_MODEL3_MANIFEST"
                blockers = [f"missing_field:{field}" for field in missing]
            elif manifest["status"] != "FROZEN_RESEARCH_CHECKPOINT_APPROVED":
                status = "WAITING_FOR_FROZEN_RESEARCH_CHECKPOINTS"
                blockers = [f"status:{manifest['status']}"]
            elif manifest.get("approved_for_frozen_ood_evaluation") is not True:
                status = "BLOCKED_NOT_APPROVED_FOR_FROZEN_OOD"
                blockers = ["approval_false"]
            elif manifest.get("parameter_output_contract") != list(PARAMETER_NAMES):
                status = "BLOCKED_PARAMETER_CONTRACT_MISMATCH"
                blockers = ["parameter_order"]
            elif list(manifest.get("seeds", [])) != list(NEURAL_SEEDS):
                status = "BLOCKED_MODEL3_SEED_CONTRACT_MISMATCH"
                blockers = ["required_seeds_11_22_33"]
            elif manifest.get("training_dataset_sha256") != sha256_file(
                PRIMARY_DATASET_PATH
            ):
                status = "BLOCKED_MODEL3_TRAINING_DATA_IDENTITY_MISMATCH"
                blockers = ["training_dataset_not_frozen_r2_train_contract"]
            else:
                status = "READY_PENDING_RUNTIME_VALIDATION"
                blockers = []
        except Exception as error:
            status = "BLOCKED_INVALID_MODEL3_MANIFEST"
            blockers = [f"{type(error).__name__}:{error}"]
    payload = {
        "schema_version": "1.0",
        "status": status,
        "current_repository_truth": (
            "No Stage-A execution, Stage-B training, or frozen Model3 research "
            "checkpoint exists. Pilot/smoke weights are ineligible."
        ),
        "blockers": blockers,
        "fake_predictions_allowed": False,
    }
    return write_json(Path(output_directory) / "model3_readiness.json", payload)


def evenly_spaced_indices(count: int, selected_count: int) -> list[int]:
    if count <= 0 or selected_count <= 0 or selected_count > count:
        raise ValueError("invalid subset size")
    return np.linspace(0, count - 1, selected_count, dtype=int).tolist()


def materialize_traditional_subset(
    output_directory: str | Path = READY_ROOT,
) -> Path:
    _safe_readiness_root(output_directory)
    source = COHORT_ROOT / "all_research_surfaces.jsonl"
    selections = expected_traditional_selections(source)
    if len(selections) != 60 or len({row["surface_id"] for row in selections}) != 60:
        raise EvaluationSealError("traditional subset cardinality failure")
    payload = {
        "schema_version": "1.0",
        "status": "DETERMINISTIC_SUBSET_MATERIALIZED_NO_EVALUATION",
        "source_path": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(source),
        "surface_id_order_sha256": sha256_bytes(
            ("\n".join(row["surface_id"] for row in selections) + "\n").encode("utf-8")
        ),
        "selection_config": TRADITIONAL_SUBSET_COUNTS,
        "total_selected": len(selections),
        "starts_per_surface": FROZEN_SETTINGS["start_count"],
        "selections": selections,
        "method_outputs_present": False,
    }
    return write_json(Path(output_directory) / "traditional_subset_manifest.json", payload)


def load_traditional_subset(
    path: str | Path = READY_ROOT / "traditional_subset_manifest.json",
) -> list[dict[str, Any]]:
    payload = read_json(path)
    if payload.get("status") != "DETERMINISTIC_SUBSET_MATERIALIZED_NO_EVALUATION":
        raise EvaluationSealError("traditional subset was not materialized cleanly")
    if payload["source_sha256"] != sha256_file(COHORT_ROOT / "all_research_surfaces.jsonl"):
        raise EvaluationSealError("traditional subset source hash mismatch")
    if len(payload["selections"]) != 60:
        raise EvaluationSealError("traditional subset does not contain exactly 60 rows")
    expected = expected_traditional_selections(
        COHORT_ROOT / "all_research_surfaces.jsonl"
    )
    if payload["selections"] != expected:
        raise EvaluationSealError("traditional subset drifts from declared selection rule")
    recorded_order_hash = payload["surface_id_order_sha256"]
    actual_order_hash = sha256_bytes(
        (
            "\n".join(row["surface_id"] for row in payload["selections"]) + "\n"
        ).encode("utf-8")
    )
    if recorded_order_hash != actual_order_hash:
        raise EvaluationSealError("traditional subset order hash mismatch")
    return payload["selections"]


def expected_traditional_selections(source: str | Path) -> list[dict[str, Any]]:
    by_cohort: dict[str, list[tuple[int, str]]] = {
        cohort: [] for cohort in ACTIVE_PARAMETER_COHORTS
    }
    for index, line in enumerate(Path(source).open(encoding="utf-8")):
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record["metadata"]["user_metadata"]
        cohort = metadata["cohort"]
        if cohort not in by_cohort:
            raise EvaluationSealError(f"unknown traditional-subset cohort: {cohort}")
        by_cohort[cohort].append((index, str(record["surface_id"])))
    selections: list[dict[str, Any]] = []
    for cohort in ACTIVE_PARAMETER_COHORTS:
        entries = by_cohort[cohort]
        indices = evenly_spaced_indices(len(entries), TRADITIONAL_SUBSET_COUNTS[cohort])
        for selected_index in indices:
            source_index, surface_id = entries[selected_index]
            selections.append(
                {
                    "cohort": cohort,
                    "surface_id": surface_id,
                    "source_row_index": source_index,
                    "within_cohort_index": selected_index,
                    "selection_rule": "evenly_spaced_indices_starting_at_zero",
                }
            )
    return selections


def _settings_fingerprint(
    cohort_hash: str, subset_manifest_hash: str, max_nfev_override: int | None
) -> str:
    settings = {
        "frozen_settings": FROZEN_SETTINGS,
        "cohort_sha256": cohort_hash,
        "subset_manifest_sha256": subset_manifest_hash,
        "max_nfev_override": max_nfev_override,
        "development_override_allowed": max_nfev_override is not None,
    }
    return sha256_bytes(deterministic_json_bytes(settings))


def _traditional_worker(payload: dict[str, Any]) -> dict[str, Any]:
    item_dict = payload["item"]
    item = R2SurfaceItem(**item_dict)
    valid = np.asarray(item.mask, dtype=bool)
    max_nfev_override = payload.get("max_nfev_override")
    started = time.perf_counter()
    frame = calibrate_double_heston(
        spot=item.spot,
        strikes=item.strikes[valid],
        maturities=item.maturities[valid],
        risk_free_rate=item.rate,
        dividend_yield=item.carry,
        option_types=list(np.asarray(item.option_types)[valid]),
        observed_prices=item.dollar_prices[valid],
        known_parameters=item.targets,
        bounds_path=FROZEN_SETTINGS["bounds_path"],
        node_count=FROZEN_SETTINGS["node_count"],
        max_nfev=(
            FROZEN_SETTINGS["max_nfev"]
            if max_nfev_override is None
            else int(max_nfev_override)
        ),
        seed=FROZEN_SETTINGS["start_seed"],
    )
    return {
        "surface_id": item.surface_id,
        "usable_quote_count": int(valid.sum()),
        "masked_quote_count": int((~valid).sum()),
        "wall_seconds_all_starts": time.perf_counter() - started,
        "rows": frame.to_dict(orient="records"),
    }


def execute_traditional_calibration(
    cohort: OODCohort,
    output_directory: str | Path,
    *,
    workers: int = 4,
    max_nfev_override: int | None = None,
) -> dict[str, Any]:
    """Run only the materialized subset; resume without changing settings."""
    if max_nfev_override is not None and cohort.kind != "development_fixture":
        raise EvaluationSealError("max_nfev override is allowed only for development fixture")
    subset_rows = (
        [
            row
            for row in load_traditional_subset()
            if row["surface_id"] in {item.surface_id for item in cohort.items}
        ]
        if cohort.kind == "frozen_research"
        else [
            {
                "cohort": cohort.kind,
                "surface_id": item.surface_id,
                "source_row_index": index,
            }
            for index, item in enumerate(cohort.items[:1])
        ]
    )
    if not subset_rows:
        raise EvaluationSealError("no traditional subset rows selected")
    expected_ids = [row["surface_id"] for row in subset_rows]
    id_to_item = {item.surface_id: item for item in cohort.items}
    if len(id_to_item) != len(cohort.items):
        raise EvaluationSealError("duplicate cohort surface IDs")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    journal_path = output / "traditional_starts_journal.jsonl"
    fingerprint = _settings_fingerprint(
        cohort.sha256,
        sha256_file(READY_ROOT / "traditional_subset_manifest.json"),
        max_nfev_override,
    )
    completed: dict[str, dict[str, Any]] = {}
    if journal_path.exists():
        for line_number, line in enumerate(journal_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("settings_fingerprint") != fingerprint:
                raise EvaluationSealError(
                    f"journal settings mismatch at line {line_number}; no altered retry"
                )
            surface_id = entry.get("surface_id")
            if surface_id not in expected_ids or surface_id in completed:
                raise EvaluationSealError("invalid duplicate/unknown journal entry")
            completed[surface_id] = entry

    pending_ids = [surface_id for surface_id in expected_ids if surface_id not in completed]
    pending_payloads = [
        {
            "item": id_to_item[surface_id].__dict__,
            "surface_id": surface_id,
            "max_nfev_override": max_nfev_override,
        }
        for surface_id in pending_ids
    ]
    with journal_path.open("a", encoding="utf-8") as journal:
        if workers == 1:
            results = map(_traditional_worker, pending_payloads)
            for payload, result in zip(pending_payloads, results, strict=True):
                result["settings_fingerprint"] = fingerprint
                journal.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                completed[result["surface_id"]] = result
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_traditional_worker, payload): payload
                    for payload in pending_payloads
                }
                for future in as_completed(futures):
                    result = future.result()
                    result["settings_fingerprint"] = fingerprint
                    journal.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    completed[result["surface_id"]] = result

    ordered_entries = [completed[surface_id] for surface_id in expected_ids]
    all_rows = [row for entry in ordered_entries for row in entry["rows"]]
    for surface_id, entry in zip(expected_ids, ordered_entries, strict=True):
        for row in entry["rows"]:
            row["surface_id"] = surface_id
            row["wall_seconds_all_starts"] = entry["wall_seconds_all_starts"]
    starts_frame = pd.DataFrame(all_rows)
    starts_csv = output / "traditional_starts.csv"
    starts_hash = write_csv(starts_frame, starts_csv)
    representatives = select_representatives(starts_frame)
    representative_columns = ["surface_id", *PARAMETER_NAMES]
    renamed = representatives.rename(
        columns={f"predicted_{name}": name for name in PARAMETER_NAMES}
    )
    representative_hash = write_csv(
        renamed[representative_columns], output / "traditional_representatives.csv"
    )
    if "success" not in starts_frame:
        raise EvaluationSealError("traditional start rows lack success provenance")
    success_flags = starts_frame["success"].fillna(False).astype(bool)
    per_surface_success = (
        starts_frame.assign(success=success_flags)
        .groupby("surface_id", sort=True)["success"]
        .all()
    )
    complete = len(completed) == len(expected_ids) and all(
        len(entry["rows"]) == FROZEN_SETTINGS["start_count"]
        for entry in ordered_entries
    )
    summary = {
        "schema_version": "1.0",
        "status": "COMPLETE" if complete else "PARTIAL_RESUMABLE",
        "cohort_kind": cohort.kind,
        "expected_surface_count": len(expected_ids),
        "completed_surface_count": len(completed),
        "starts_per_surface": FROZEN_SETTINGS["start_count"],
        "settings_fingerprint": fingerprint,
        "max_nfev_override": max_nfev_override,
        "scientifically_comparable": max_nfev_override is None,
        "start_success_rate": float(success_flags.mean()),
        "start_failure_rate": float(1.0 - success_flags.mean()),
        "surfaces_with_all_starts_successful_rate": float(per_surface_success.mean()),
        "execution_failure_rate_exceeds_five_percent": bool(
            (1.0 - success_flags.mean()) > 0.05
        ),
        "masked_quotes_imputed": False,
        "artifact_hashes": {
            "journal_sha256": sha256_file(journal_path),
            "starts_csv_sha256": starts_hash,
            "representatives_csv_sha256": representative_hash,
        },
    }
    write_json(output / "traditional_summary.json", summary)
    return summary


def _prediction_frame(
    cohort: OODCohort, predicted: np.ndarray, failures: list[dict[str, Any]]
) -> pd.DataFrame:
    failed_by_index = {int(row["index"]): row.get("error_type", "FAILED") for row in failures}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(cohort.items):
        row: dict[str, Any] = {"surface_id": item.surface_id}
        if index in failed_by_index:
            row.update({name: np.nan for name in PARAMETER_NAMES})
            row["prediction_success"] = False
            row["failure_type"] = failed_by_index[index]
        else:
            row.update(dict(zip(PARAMETER_NAMES, map(float, predicted[index]), strict=True)))
            row["prediction_success"] = bool(np.isfinite(predicted[index]).all())
            row["failure_type"] = "" if row["prediction_success"] else "NONFINITE_PREDICTION"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_prediction_matrix(
    cohort: OODCohort,
    predicted: np.ndarray,
    scaling: Mapping[str, Mapping[str, float]],
    indices: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate using the existing primary-comparison metric implementations."""
    selected = list(range(len(cohort.items))) if indices is None else list(indices)
    items = [cohort.items[index] for index in selected]
    selected_predictions = predicted[indices] if indices is not None else predicted
    finite = np.isfinite(selected_predictions).all(axis=1)
    failures = [
        {"index": int(index), "error_type": "NONFINITE_PREDICTION"}
        for index in np.flatnonzero(~finite)
    ]
    success_predicted = selected_predictions[finite]
    success_truth = np.stack(
        [item.targets for item in items], axis=0
    )[finite]
    recovery = (
        parameter_recovery_metrics(success_truth, success_predicted, dict(scaling))
        if len(success_predicted)
        else {"error": "no successful finite predictions"}
    )
    validity = (
        constraint_validity_metrics(success_predicted)
        if len(success_predicted)
        else {"constraint_validity_rate": 0.0}
    )
    repriced = np.full((len(items), 20), np.nan, dtype=np.float64)
    pricing_failures: list[dict[str, Any]] = []
    for position, index in enumerate(selected):
        if not finite[position]:
            continue
        input_item = cohort.items[int(index)]
        target_item = cohort.parent_items.get(input_item.surface_id, input_item)
        dummy_dataset = R2PrimaryDataset([target_item])
        try:
            repriced[position] = reprice_normalized(
                dummy_dataset,
                [0],
                selected_predictions[position][None, :],
                node_count=FROZEN_SETTINGS["node_count"],
            )[0]
        except Exception as error:
            pricing_failures.append(
                {
                    "index": int(index),
                    "surface_id": input_item.surface_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    observed_clean = np.stack(
        [
            cohort.parent_items.get(item.surface_id, item).normalized_prices
            for item in items
        ]
    )
    pricing_success = finite & np.isfinite(repriced).all(axis=1)
    repricing = (
        repricing_metrics(observed_clean[pricing_success], repriced[pricing_success])
        if pricing_success.any()
        else {"status": "no_successful_repricing_rows"}
    )
    identifiability = None
    if pricing_success.any():
        from .r2_primary.evaluation import identifiability_aware_metrics

        identifiability = identifiability_aware_metrics(
            observed_clean[pricing_success],
            repriced[pricing_success],
            np.stack([item.targets for item in items], axis=0)[pricing_success],
            selected_predictions[pricing_success],
        dict(scaling),
        )
    from .r2_primary.evaluation import _range_scaled_error_matrix

    parameter_per_surface_mse = (
        (
            _range_scaled_error_matrix(
                np.stack([item.targets for item in items], axis=0)[finite],
                selected_predictions[finite],
                dict(scaling),
            )
            ** 2
        ).mean(axis=1)
    )
    squared_errors = (repriced[pricing_success] - observed_clean[pricing_success]) ** 2
    repricing_per_surface_rmse = np.sqrt(np.nanmean(squared_errors, axis=1))
    summary: dict[str, Any] = {
        "attempted_surface_count": len(items),
        "successful_prediction_count": int(finite.sum()),
        "prediction_failure_count": int((~finite).sum()),
        "pricing_failure_count": len(pricing_failures),
        "pricing_failures_retained": pricing_failures,
        "parameter_recovery": recovery,
        "constraint_validity": validity,
        "clean_latent_repricing": repricing,
        "_per_surface_parameter_mse": parameter_per_surface_mse,
        "_per_surface_repricing_rmse": repricing_per_surface_rmse,
    }
    global_incomplete_indices = {
        index
        for index, item in enumerate(cohort.items)
        if item.surface_id not in cohort.parent_items
    }
    local_incomplete_positions = [
        position
        for position, index in enumerate(selected)
        if index in global_incomplete_indices
    ]
    if identifiability is not None:
        summary["identifiability_aware"] = identifiability
    if cohort.kind == "frozen_research" and cohort.parent_items:
        incomplete_items = [
            (position, cohort.items[index])
            for position, index in enumerate(selected)
            if index in global_incomplete_indices
        ]
        observed_masked = np.full_like(observed_clean, np.nan)
        for index, item in incomplete_items:
            active = item.mask
            observed_masked[index, active] = observed_clean[index, active]
        mask_pricing_success = pricing_success[local_incomplete_positions]
        masked_repricing = (
            repricing_metrics(
                observed_masked[local_incomplete_positions][mask_pricing_success],
                repriced[local_incomplete_positions][mask_pricing_success],
            )
            if local_incomplete_positions and mask_pricing_success.any()
            else {"status": "no_successful_repricing_rows"}
        )
        summary["observed_slot_repricing_diagnostic"] = masked_repricing
    return summary


def load_id_baseline_summary(method: str) -> dict[str, Any]:
    """Load hash-pinned primary test summaries for a future unlocked comparison."""
    frozen = yaml.safe_load(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    artifacts = frozen["authority"]["primary_baseline_artifacts"]
    expected = {
        "parameter_metrics": "parameter_metrics.json",
        "repricing_metrics": "repricing_metrics.json",
        "validity_metrics": "validity_metrics.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for label, filename in expected.items():
        identity = artifacts[label]
        path = ROOT / identity["path"]
        if not path.is_file() or sha256_file(path).lower() != identity[
            "sha256"
        ].lower():
            raise EvaluationSealError(f"ID baseline artifact identity failed: {label}")
        loaded[label] = read_json(path)
    aliases = {
        "model1": ("model1_seed_mean", "model1"),
        "model2": ("model2_seed_mean", "model2"),
        "traditional_calibration": ("traditional_calibration",),
    }
    if method not in aliases:
        raise ValueError(f"no ID baseline alias for {method}")
    parameter_key = next(
        (alias for alias in aliases[method] if alias in loaded["parameter_metrics"]),
        None,
    )
    repricing_key = next(
        (alias for alias in aliases[method] if alias in loaded["repricing_metrics"]),
        None,
    )
    validity_key = next(
        (alias for alias in aliases[method] if alias in loaded["validity_metrics"]),
        None,
    )
    if parameter_key is None or repricing_key is None or validity_key is None:
        raise EvaluationSealError(f"ID baseline method absent: {method}")
    return {
        "parameter_recovery": loaded["parameter_metrics"][parameter_key],
        "clean_latent_repricing": loaded["repricing_metrics"][repricing_key],
        "constraint_validity": loaded["validity_metrics"][validity_key],
    }


def bootstrap_materiality(
    ood_values: np.ndarray,
    id_baseline_value: float,
    *,
    statistic: str = "mean",
    seed: int = 20260829,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    ratio_floor: float = 1.0e-8,
    material_ratio: float = 1.25,
) -> dict[str, Any]:
    """Bootstrap the exact aggregate statistic against a fixed ID baseline."""
    if statistic not in {"mean", "root_mean_of_surface_mse"}:
        raise ValueError("unsupported bootstrap statistic")
    values = np.asarray(ood_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise EvaluationSealError("bootstrap requires one row per eligible surface")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sampled_statistics = (
        values[indices].mean(axis=1)
        if statistic == "mean"
        else np.sqrt(values[indices].mean(axis=1))
    )
    point_statistic = (
        float(values.mean())
        if statistic == "mean"
        else float(np.sqrt(values.mean()))
    )
    ratios = sampled_statistics / max(float(id_baseline_value), ratio_floor)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(ratios, [alpha, 1.0 - alpha])
    return {
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "ratio_interval": [float(lower), float(upper)],
        "interval_spans_materiality": bool(lower <= material_ratio <= upper),
        "statistic": statistic,
        "point_ratio": point_statistic / max(float(id_baseline_value), ratio_floor),
        "id_baseline_treated_as_fixed": True,
    }


def degradation_against_id_baseline(
    ood_summary: Mapping[str, Any],
    id_baseline_summary: Mapping[str, Any],
    *,
    ratio_floor: float = 1.0e-8,
    material_ratio: float = 1.25,
    material_validity_failure_increase: float = 0.05,
) -> dict[str, Any]:
    """Apply the frozen post-hoc degradation rule to one aligned cohort summary."""

    def metric(summary: Mapping[str, Any], key: str) -> float:
        if key == "constraint_validity_failure_rate":
            return 1.0 - float(
                summary["constraint_validity"]["constraint_validity_rate"]
            )
        if key.startswith("parameter_recovery."):
            leaf = key.split(".", 1)[1]
            return float(summary["parameter_recovery"]["aggregate"][leaf])
        return float(summary["clean_latent_repricing"][key])

    keys = (
        "parameter_recovery.range_scaled_parameter_rmse",
        "parameter_recovery.standardized_parameter_rmse",
        "normalized_price_rmse_mean",
        "constraint_validity_failure_rate",
    )
    result: dict[str, Any] = {}
    for key in keys:
        ood_value = metric(ood_summary, key)
        baseline_value = metric(id_baseline_summary, key)
        entry: dict[str, Any] = {
            "ood_value": ood_value,
            "id_baseline_value": baseline_value,
        }
        if key == "constraint_validity_failure_rate":
            entry["absolute_increase"] = ood_value - baseline_value
            entry["material"] = bool(
                entry["absolute_increase"] > material_validity_failure_increase
            )
        else:
            entry["degradation_ratio"] = max(ood_value, ratio_floor) / max(
                baseline_value, ratio_floor
            )
            entry["material"] = bool(entry["degradation_ratio"] > material_ratio)
        result[key] = entry
    result["decision"] = (
        "MATERIAL_DEGRADATION_INDICATED"
        if any(
            isinstance(value, dict) and value.get("material") is True
            for value in result.values()
        )
        else "NO_MATERIAL_DEGRADATION_BY_FROZEN_RULE"
    )
    result["claim_boundary"] = (
        "descriptive degradation only; repricing success is never recovery proof"
    )
    return result


def aggregate_neural_seeds(
    predictions: Mapping[int, np.ndarray],
    headline_by_seed: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    """Retain every seed and report canonical cross-seed stability."""
    from .r2_primary.evaluation import stability_metrics

    seeds = sorted(predictions)
    if len(seeds) < 2:
        raise EvaluationSealError("seed aggregation requires at least two seeds")
    stacked = np.stack([predictions[seed] for seed in seeds], axis=0)
    return {
        "status": "ALL_SEEDS_RETAINED",
        "seeds": seeds,
        "seed_mean_prediction": stacked.mean(axis=0),
        "stability": stability_metrics(dict(predictions), dict(headline_by_seed)),
    }


def _write_metric_artifact(
    output_directory: str | Path,
    method: str,
    seed: int | None,
    cohort: OODCohort,
    predicted: np.ndarray,
    scaling: Mapping[str, Mapping[str, float]],
    runtime: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any], bool]:
    stem = method if seed is None else f"{method}_seed{seed}"
    prediction_frame = _prediction_frame(cohort, predicted, [])
    prediction_path = output_directory / f"{stem}_predictions.csv"
    prediction_hash = write_csv(frame=prediction_frame, path=prediction_path)
    metrics = evaluate_prediction_matrix(cohort, predicted, scaling)
    per_surface_parameter_mse = metrics.pop("_per_surface_parameter_mse")
    per_surface_repricing_rmse = metrics.pop("_per_surface_repricing_rmse")
    cohort_indices: dict[str, list[int]] = {}
    for index, item in enumerate(cohort.items):
        record = cohort.records[index]
        cohort_name = str(record["metadata"]["user_metadata"].get(
            "cohort", record["metadata"]["user_metadata"].get("missingness_pattern")
        ))
        cohort_indices.setdefault(cohort_name, []).append(index)
    cohort_summaries = {
        name: evaluate_prediction_matrix(
            cohort, predicted, scaling, indices=indices
        )
        for name, indices in sorted(cohort_indices.items())
    }
    for summary in cohort_summaries.values():
        summary.pop("_per_surface_parameter_rmse", None)
        summary.pop("_per_surface_parameter_mse", None)
        summary.pop("_per_surface_repricing_rmse", None)
    metrics = _jsonable_metrics(metrics)
    metrics_payload = {
        "method": method,
        "seed": seed,
        "run_kind": (
            "DEVELOPMENT_FIXTURE_NON_RESULT"
            if cohort.kind == "development_fixture"
            else "FROZEN_OOD_RESEARCH_RESULT"
        ),
        "metrics": metrics,
        "cohort_metrics": _jsonable_metrics(cohort_summaries),
    }
    if cohort.kind == "frozen_research":
        baseline = load_id_baseline_summary(method)
        degradation = degradation_against_id_baseline(metrics, baseline)
        degradation["bootstrap"] = {
            "parameter_recovery.range_scaled_parameter_rmse": (
                bootstrap_materiality(
                    per_surface_parameter_mse,
                    baseline["parameter_recovery"]["aggregate"][
                        "range_scaled_parameter_rmse"
                    ],
                    statistic="root_mean_of_surface_mse",
                )
            ),
            "normalized_price_rmse_mean": bootstrap_materiality(
                per_surface_repricing_rmse,
                baseline["clean_latent_repricing"]["normalized_price_rmse_mean"],
            ),
        }
        metrics_payload["degradation"] = _jsonable_metrics(degradation)
    metrics_path = write_json(output_directory / f"{stem}_metrics.json", metrics_payload)
    metrics_hash = sha256_file(metrics_path)
    runtime_hash = None
    if runtime is not None:
        runtime_path = write_json(
            output_directory / f"{stem}_runtime.json",
            {"runtime": runtime},
        )
        runtime_hash = sha256_file(runtime_path)
    hashes = {
        f"{stem}_predictions_csv_sha256": prediction_hash,
        f"{stem}_metrics_json_sha256": metrics_hash,
    }
    if runtime_hash is not None:
        hashes[f"{stem}_runtime_json_sha256"] = runtime_hash
    bootstrap_inconclusive = False
    method_status: dict[str, Any] = {"status": "COMPLETE"}
    if cohort.kind == "frozen_research":
        degradation = metrics_payload.get("degradation", {})
        bootstrap = degradation.get("bootstrap", {})
        bootstrap_inconclusive = any(
            entry.get("interval_spans_materiality") is True
            for entry in bootstrap.values()
        )
        method_status["bootstrap_inconclusive"] = bootstrap_inconclusive
    return hashes, method_status, bootstrap_inconclusive


def _safe_output_directory(output: str | Path) -> Path:
    resolved = Path(output).resolve()
    protected = [
        COHORT_ROOT.resolve(),
        COHORT_REPLAY_ROOT.resolve(),
        PRIMARY_DATASET_PATH.parent.resolve(),
        (ROOT / "evidence" / "final_r2_clean_10000").resolve(),
    ]
    if any(resolved == target or target in resolved.parents for target in protected):
        raise EvaluationSealError(f"refusing to write evaluation output under frozen evidence: {resolved}")
    existing_names = {path.name for path in resolved.iterdir()} if resolved.exists() else set()
    if existing_names - {".staging_development_fixture"}:
        raise EvaluationSealError(f"refusing to overwrite nonempty evaluation output: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_development_fixture(output_directory: str | Path) -> tuple[Path, dict[str, Any]]:
    """Build a tiny nonresearch fixture unrelated to frozen cohort truths."""
    vectors = [
        [0.55, 0.06, 0.18, -0.35, 0.04, 2.50, 0.05, 0.35, 0.15, 0.03],
        [1.10, 0.10, 0.30, 0.20, 0.08, 5.00, 0.07, 0.60, -0.40, 0.06],
    ]
    from .r2_representation import R2Conditioning, build_synthetic_surface

    conditionings = [
        R2Conditioning(
            date_id="DEV_FIXTURE_A",
            spot=100.0,
            expiry_dates=("DEV_RANK_1", "DEV_RANK_2"),
            dte=(30, 90),
            rates=(0.02, 0.02),
            carries=(0.01, 0.01),
        ),
        R2Conditioning(
            date_id="DEV_FIXTURE_B",
            spot=100.0,
            expiry_dates=("DEV_RANK_1", "DEV_RANK_2"),
            dte=(45, 150),
            rates=(0.02, 0.02),
            carries=(0.01, 0.01),
        ),
    ]
    surfaces = []
    for index, (vector, conditioning) in enumerate(zip(vectors, conditionings, strict=True)):
        vector_array = np.asarray(vector, dtype=np.float64)
        if not validate_parameters(vector_array)["is_valid"]:
            raise EvaluationSealError("development fixture vector invalid")
        surfaces.append(
            build_synthetic_surface(
                vector_array,
                conditioning,
                surface_id=f"DEV_FIXTURE_{index:04d}",
                metadata={
                    "dataset_status": "DEVELOPMENT_FIXTURE_NOT_RESEARCH_RESULT",
                    "real_market_inputs_used": False,
                },
                node_count=64,
            )
        )
    from .ood_boundary_protocol import make_incomplete_surface

    incomplete = make_incomplete_surface(
        surfaces[-1],
        pattern="central_three_moneyness",
        surface_id="DEV_FIXTURE_0001_INCOMPLETE_000000",
        sequence_index=0,
    )
    surfaces.append(incomplete)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fixture_path = output / "development_fixture_surfaces.jsonl"
    payloads = []
    from .r2_representation import surface_to_payload

    for surface in surfaces:
        payload = surface_to_payload(surface)
        validate_payload(payload)
        payload["metadata"]["user_metadata"]["run_kind"] = "DEVELOPMENT_FIXTURE_NOT_RESEARCH_RESULT"
        payloads.append(payload)
    content = b"".join(deterministic_json_bytes(payload) for payload in payloads)
    fixture_path.write_bytes(content)
    items = [_record_to_ood_item(payload, "development_fixture") for payload in payloads]
    fixture_ids = {item.surface_id for item in items}
    fixture_hashes = {
        parameter_vector_hash(
            dict(zip(PARAMETER_NAMES, item.targets, strict=True))
        )
        for item in items
    }
    research_ids: set[str] = set()
    research_hashes: set[str] = set()
    with (COHORT_ROOT / "all_research_surfaces.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            research_ids.add(str(record["surface_id"]))
            stored_parameters = record["metadata"]["parameters_canonical_order"]
            canonical_research_vector = {
                name: float(stored_parameters[name]) for name in PARAMETER_NAMES
            }
            research_hashes.add(
                parameter_vector_hash(
                    canonical_research_vector
                )
            )
    overlap = bool((fixture_ids & research_ids) or (fixture_hashes & research_hashes))
    identity = {
        "schema_version": "1.0",
        "status": "DEVELOPMENT_FIXTURE_ONLY_NOT_RESEARCH_DATA",
        "surface_count": len(payloads),
        "sha256": sha256_bytes(content),
        "surface_id_order_sha256": _surface_order_hash(items),
        "overlaps_frozen_research_surfaces": overlap,
        "overlapping_surface_id_count": len(fixture_ids & research_ids),
        "overlapping_parameter_vector_count": len(fixture_hashes & research_hashes),
    }
    if overlap:
        raise EvaluationSealError("development fixture overlaps frozen research identities")
    write_json(output / "development_fixture_identity.json", identity)
    return fixture_path, identity


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in {"result_manifest.json", "replay_report.json"}
    }


def verify_result_intake(
    result_directory: str | Path, *, require_complete: bool = True
) -> dict[str, Any]:
    directory = Path(result_directory)
    manifest = read_json(directory / "result_manifest.json")
    required = {
        "schema_version",
        "status",
        "run_kind",
        "requested_methods",
        "protocol_config_sha256",
        "evaluation_config_sha256",
        "cohort_manifest_sha256",
        "cohort_file_sha256",
        "command",
        "git_commit_sha",
        "environment",
        "hardware",
        "artifact_hashes",
        "implementation_hashes",
        "prediction_surface_id_order_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise EvaluationSealError(f"result manifest missing fields: {missing}")
    expected_identities = {
        "protocol_config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "evaluation_config_sha256": sha256_file(EVALUATION_CONFIG_PATH),
        "cohort_manifest_sha256": sha256_file(COHORT_ROOT / "manifest.json"),
        "cohort_file_sha256": sha256_file(
            COHORT_ROOT / "all_research_surfaces.jsonl"
        ),
    }
    for key in ("protocol_config_sha256", "evaluation_config_sha256"):
        if manifest[key] != expected_identities[key]:
            raise EvaluationSealError(f"stale result identity: {key}")
    if manifest["run_kind"] == "frozen_research":
        for key in ("cohort_manifest_sha256", "cohort_file_sha256"):
            if manifest[key] != expected_identities[key]:
                raise EvaluationSealError(f"stale frozen cohort identity: {key}")
    actual = _artifact_hashes(directory)
    recorded = manifest["artifact_hashes"]
    requested_methods = list(manifest["requested_methods"])
    if not requested_methods:
        raise EvaluationSealError("result manifest requests no methods")
    missing_statuses = sorted(set(requested_methods) - set(manifest["method_statuses"]))
    extra_statuses = sorted(set(manifest["method_statuses"]) - set(requested_methods))
    if missing_statuses or extra_statuses:
        raise EvaluationSealError(
            f"method status mismatch: missing={missing_statuses}, extra={extra_statuses}"
        )
    incomplete_requested = [
        method
        for method in requested_methods
        if manifest["method_statuses"][method].get("status") != "COMPLETE"
    ]
    if manifest["status"] == "COMPLETE" and incomplete_requested:
        raise EvaluationSealError(
            f"contradictory COMPLETE result; incomplete methods={incomplete_requested}"
        )
    if manifest["status"] != "COMPLETE" and require_complete:
        raise EvaluationSealError(
            f"partial/blocked/inconclusive result refused: {manifest['status']}"
        )
    mismatches = sorted(key for key in actual if recorded.get(key) != actual[key])
    mismatches.extend(
        key
        for key in sorted(set(recorded) - set(actual))
        if key not in mismatches
    )
    missing_required: list[str] = []
    for method in requested_methods:
        if method in {"model1", "model2"} and manifest["method_statuses"][
            method
        ].get("status") == "COMPLETE":
            required_files = [
                f"{method}_seed{seed}{suffix}"
                for seed in NEURAL_SEEDS
                for suffix in ("_predictions.csv", "_metrics.json", "_runtime.json")
            ]
            missing_required.extend(
                filename for filename in required_files if filename not in recorded
            )
        elif method == "truth_pipeline" and manifest["method_statuses"][method].get(
            "status"
        ) == "COMPLETE":
            missing_required.extend(
                filename
                for filename in (
                    "truth_pipeline_predictions.csv",
                    "truth_pipeline_metrics.json",
                )
                if filename not in recorded
            )
        elif method == "traditional_calibration" and manifest["method_statuses"][
            method
        ].get("status") == "COMPLETE":
            missing_required.extend(
                filename
                for filename in (
                    "traditional_starts_journal.jsonl",
                    "traditional_starts.csv",
                    "traditional_representatives.csv",
                    "traditional_summary.json",
                )
                if filename not in recorded
            )
        elif method == "model3" and manifest["method_statuses"][method].get(
            "status"
        ) == "COMPLETE":
            missing_required.extend(
                filename
                for filename in (
                    "model3_predictions.csv",
                    "model3_metrics.json",
                    "model3_runtime.json",
                )
                if filename not in recorded
            )
    if missing_required:
        raise EvaluationSealError(f"missing required result artifacts: {missing_required}")
    if mismatches:
        raise EvaluationSealError(f"result artifact hash mismatches: {mismatches}")
    if manifest["run_kind"] == "frozen_research" and manifest.get(
        "authorization_record", {}
    ).get("mechanism") != "explicit_cohort_authorize_flag_exact_phrase":
        raise EvaluationSealError("research result lacks explicit authorization record")
    prediction_order_hash = manifest.get("prediction_surface_id_order_sha256")
    if not prediction_order_hash:
        raise EvaluationSealError("result manifest has no prediction-order identity")
    for path in directory.rglob("*_predictions.csv"):
        frame = pd.read_csv(path)
        order_hash = sha256_bytes(
            ("\n".join(frame["surface_id"].astype(str)) + "\n").encode("utf-8")
        )
        if order_hash != prediction_order_hash:
            raise EvaluationSealError(f"prediction row misalignment: {path.name}")
    return {"manifest": manifest, "artifact_count": len(actual), "hashes_verified": True}


def run_evaluation(
    *,
    cohort_kind: str,
    methods: list[str],
    output_directory: str | Path,
    authorize: bool = False,
    confirmation: str = "",
    include_traditional: bool = False,
    traditional_workers: int = 1,
    max_nfev_override: int | None = None,
) -> dict[str, Any]:
    """Execute development fixture now, or sealed research later."""
    if cohort_kind not in {"development_fixture", "research"}:
        raise ValueError("cohort_kind must be development_fixture or research")
    output = _safe_output_directory(output_directory)
    requested_methods = list(dict.fromkeys(methods))
    if "truth_pipeline" in requested_methods and cohort_kind == "research":
        raise EvaluationSealError("truth_pipeline is forbidden on frozen research cohorts")
    authorization_record = None
    if cohort_kind == "research":
        authorization_record = check_research_authorization(
            cohort=cohort_kind, authorize=authorize, confirmation=confirmation
        )
        cohort = load_frozen_research_cohort(authorized=True)
    else:
        if authorize or confirmation:
            raise EvaluationSealError("development fixture must not carry research authorization flags")
        staging = Path(output_directory).resolve() / ".staging_development_fixture"
        if staging.exists():
            raise EvaluationSealError("refusing to overwrite development fixture staging")
        fixture_path, _ = build_development_fixture(staging)
        cohort = load_development_fixture(fixture_path)
    scaling = load_reference_scaling()
    if not requested_methods:
        raise EvaluationSealError("at least one method must be requested")
    unknown = set(requested_methods) - {"model1", "model2", "model3", "truth_pipeline"}
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if include_traditional:
        requested_methods.append("traditional_calibration")
    artifact_hashes: dict[str, str] = {}
    method_statuses: dict[str, Any] = {}
    bootstrap_inconclusive = False
    if "truth_pipeline" in requested_methods:
        predicted = cohort.truths
        hashes, truth_status, _ = _write_metric_artifact(
            output, "truth_pipeline", None, cohort, predicted, scaling
        )
        artifact_hashes.update(hashes)
        method_statuses["truth_pipeline"] = {"status": "COMPLETE"}
        method_statuses["truth_pipeline"].update(truth_status)
    for method in ("model1", "model2"):
        if method not in requested_methods:
            continue
        predictions, runtimes, status = run_neural_adapter(method, cohort)
        if status["status"] != "COMPLETE":
            method_statuses[method] = status
            continue
        for seed in NEURAL_SEEDS:
            hashes, neural_metric_status, seed_inconclusive = _write_metric_artifact(
                output,
                method,
                seed,
                cohort,
                predictions[seed],
                scaling,
                runtimes[seed],
            )
            artifact_hashes.update(hashes)
            bootstrap_inconclusive |= seed_inconclusive
            status.setdefault("per_seed", {})[str(seed)] = neural_metric_status
        method_statuses[method] = status
    if "model3" in requested_methods:
        readiness_path = model3_readiness(output)
        readiness = read_json(readiness_path)
        method_statuses["model3"] = {
            "status": readiness["status"],
            "blockers": readiness["blockers"],
            "fake_predictions_allowed": False,
        }
    traditional_summary = None
    if include_traditional:
        traditional_summary = execute_traditional_calibration(
            cohort,
            output,
            workers=traditional_workers,
            max_nfev_override=max_nfev_override,
        )
        method_statuses["traditional_calibration"] = {
            "status": traditional_summary["status"],
            "scientifically_comparable": traditional_summary[
                "scientifically_comparable"
            ],
            "execution_failure_rate_exceeds_five_percent": (
                traditional_summary["execution_failure_rate_exceeds_five_percent"]
            ),
        }

    # Capture fixture provenance and adapter-readiness sidecars too, so strict
    # intake verifies every byte that participates in the result bundle.
    artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"result_manifest.json", "replay_report.json"}
    }

    core_complete = all(
        status.get("status") == "COMPLETE"
        for name, status in method_statuses.items()
        if name in requested_methods
    )
    traditional_failure_inconclusive = bool(
        traditional_summary
        and traditional_summary.get(
            "execution_failure_rate_exceeds_five_percent"
        )
    )
    if not core_complete:
        status = "PARTIAL_OR_BLOCKED"
    elif bootstrap_inconclusive:
        status = "INCONCLUSIVE_BOOTSTRAP_MATERIALITY"
    elif traditional_failure_inconclusive:
        status = "INCONCLUSIVE_TRADITIONAL_FAILURE_RATE"
    else:
        status = "COMPLETE"
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "run_kind": cohort.kind,
        "requested_methods": requested_methods,
        "method_statuses": method_statuses,
        "protocol_config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "evaluation_config_sha256": sha256_file(EVALUATION_CONFIG_PATH),
        "cohort_manifest_sha256": sha256_file(COHORT_ROOT / "manifest.json"),
        "cohort_file_sha256": cohort.sha256,
        "surface_id_order_sha256": cohort.surface_id_order_sha256,
        "prediction_surface_id_order_sha256": cohort.surface_id_order_sha256,
        "authorization_record": authorization_record,
        "command": "RECORDED_BY_CLI",
        "git_commit_sha": _git_head(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_is_rng_input": False,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "hardware": _hardware(),
        "implementation_hashes": {
            "harness_source_sha256": sha256_file(Path(__file__)),
            "cli_source_sha256": sha256_file(ROOT / "scripts" / "run_ood_boundary_evaluation.py"),
            "evaluation_config_sha256": sha256_file(EVALUATION_CONFIG_PATH),
        },
        "artifact_hashes": artifact_hashes,
        "traditional_summary": traditional_summary,
    }
    if cohort_kind == "research":
        manifest["command"] = (
            "python -m scripts.run_ood_boundary_evaluation run --cohort research "
            "--methods ... --authorize-frozen-evaluation "
            "--confirmation <exact-configured-phrase> "
            f"--output {output}"
        )
    else:
        manifest["command"] = (
            "python -m scripts.run_ood_boundary_evaluation run-development-smoke "
            f"--output {output}"
        )
    write_json(output / "result_manifest.json", manifest)
    intake = verify_result_intake(output, require_complete=False)
    return {
        "output": output,
        "status": status,
        "method_statuses": method_statuses,
        "intake": intake,
    }


def compare_replay(left_directory: str | Path, right_directory: str | Path) -> dict[str, Any]:
    left, right = Path(left_directory), Path(right_directory)
    ignored_names = {"result_manifest.json", "replay_report.json"}
    ignored_suffixes = ("_runtime.json",)
    def ignored(path: Path) -> bool:
        return path.name in ignored_names or path.name.endswith(ignored_suffixes)

    names_left = {
        path.relative_to(left).as_posix()
        for path in left.rglob("*")
        if path.is_file() and not ignored(path)
    }
    names_right = {
        path.relative_to(right).as_posix()
        for path in right.rglob("*")
        if path.is_file() and not ignored(path)
    }
    comparisons = {
        relative: sha256_file(left / relative) == sha256_file(right / relative)
        for relative in sorted(names_left & names_right)
    }
    return {
        "deterministic_core_identical": all(comparisons.values()) and names_left == names_right,
        "comparisons": comparisons,
        "left_only": sorted(names_left - names_right),
        "right_only": sorted(names_right - names_left),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-freeze")
    prepare = subparsers.add_parser("prepare-readiness")
    prepare.add_argument("--overwrite", action="store_true")
    development = subparsers.add_parser("run-development-smoke")
    development.add_argument("--methods", default="truth_pipeline")
    development.add_argument("--include-traditional", action="store_true")
    development.add_argument("--workers", type=int, default=1)
    development.add_argument("--max-nfev-override", type=int, default=1)
    development.add_argument("--output", type=Path, required=True)
    research = subparsers.add_parser("run")
    research.add_argument("--cohort", choices=["research"], required=True)
    research.add_argument("--methods", required=True)
    research.add_argument("--include-traditional", action="store_true")
    research.add_argument("--workers", type=int, default=4)
    research.add_argument("--authorize-frozen-evaluation", action="store_true")
    research.add_argument("--confirmation", default="")
    research.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify-freeze":
        print(json.dumps(freeze_identities(), sort_keys=True))
        return 0
    if arguments.command == "prepare-readiness":
        output = READY_ROOT
        if output.exists() and any(output.iterdir()) and not arguments.overwrite:
            parser.error("readiness directory is nonempty; pass --overwrite only for deliberate regeneration")
        output.mkdir(parents=True, exist_ok=True)
        artifacts = [
            prepare_freeze_identity(output),
            prepare_reference_scaling(output),
            checkpoint_readiness(output),
            model3_readiness(output),
            materialize_traditional_subset(output),
        ]
        print(json.dumps({path.name: sha256_file(path) for path in artifacts}, sort_keys=True))
        return 0
    if arguments.command == "run-development-smoke":
        methods = [method.strip() for method in arguments.methods.split(",") if method.strip()]
        result = run_evaluation(
            cohort_kind="development_fixture",
            methods=methods,
            output_directory=arguments.output,
            include_traditional=arguments.include_traditional,
            traditional_workers=arguments.workers,
            max_nfev_override=arguments.max_nfev_override,
        )
    else:
        methods = [method.strip() for method in arguments.methods.split(",") if method.strip()]
        result = run_evaluation(
            cohort_kind="research",
            methods=methods,
            output_directory=arguments.output,
            authorize=arguments.authorize_frozen_evaluation,
            confirmation=arguments.confirmation,
            include_traditional=arguments.include_traditional,
            traditional_workers=arguments.workers,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(result["output"]),
                "method_statuses": result["method_statuses"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
