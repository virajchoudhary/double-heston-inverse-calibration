"""Strict contracts for frozen Model 3 Stage-B research checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"
PROTOCOL_CONFIG_PATH = REPO_ROOT / "configs" / "model3_pde_protocol.yaml"
PROTOCOL_VERSION = "1.1"
STAGE_B_RUN_KIND = "MODEL3_STAGE_B_RESEARCH_FROZEN"
REQUIRED_SEEDS = (11, 22, 33)
MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 15
EXPECTED_DATASET_SHA256 = (
    "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6"
)
FROZEN_SETTINGS = {
    "train_limit": 7500,
    "validation_limit": 1250,
    "epochs": MAX_EPOCHS,
    "batch_size": 32,
    "interior_points": 32,
    "terminal_points": 8,
    "learning_rate": 0.0002,
    "weight_decay": 0.00001,
    "device": "cuda",
    "smoke_mode": False,
    "patience": EARLY_STOPPING_PATIENCE,
}
FROZEN_LOSS_WEIGHTS = {
    "parameter": 1.0,
    "reconstruction": 1.0,
    "pde_residual": 0.10,
    "terminal_diagnostic": 0.0,
    "boundary_penalty": 0.0,
}
TRAIN_HISTORY_FIELDS = (
    "epoch", "parameter_loss", "reconstruction_loss", "pde_residual_loss",
    "total_loss", "finite_gradients", "gradient_norm", "pde_residual_rms",
    "pde_residual_max_scaled_rms", "terminal_payoff_max_abs",
    "duration_seconds", "accelerator_memory_allocated_bytes",
    "accelerator_memory_reserved_bytes",
)
VALIDATION_HISTORY_FIELDS = (
    "epoch", "validation_parameter_loss", "validation_reconstruction_loss",
    "validation_pde_residual_loss", "validation_total_loss",
)
PHYSICS_FIELDS = (
    "epoch", "split", "batch_index", "surface_count",
    "collocation_point_count", "residual_mean", "residual_max_abs",
    "terminal_payoff_max_abs",
)
GRADIENT_FIELDS = ("epoch", "batch_index", "finite_gradients", "gradient_norm")
REQUIRED_ARTIFACTS = (
    "checkpoint.pt", "optimizer.pt", "epoch_metadata.json",
    "train_history.csv", "validation_history.csv",
    "physics_diagnostics.csv", "gradient_diagnostics.csv",
    "environment_provenance.json", "artifact_manifest.json",
)
CHECKPOINT_CONTRACT_FIELDS = frozenset(
    {
        "schema_version", "experiment_id", "source_git_sha", "protocol_name",
        "protocol_version", "config_sha256", "final_r2_dataset_sha256", "seed",
        "train_population_sha256", "train_population_count",
        "validation_population_sha256", "validation_population_count",
        "test_not_opened", "run_kind", "settings", "loss_weights",
        "best_epoch", "epochs_completed", "checkpoint_selection_rule",
        "checkpoint_sha256", "training_history_sha256",
        "validation_history_sha256", "physics_diagnostics_sha256",
        "gradient_diagnostics_sha256", "environment_provenance_sha256",
        "artifact_manifest_sha256", "completion_status",
        "training_host", "training_platform",
    }
)
FREEZE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "manifest_kind", "experiment_id", "source_git_sha",
        "protocol_name", "protocol_version", "config_sha256",
        "final_r2_dataset_sha256", "seeds", "seed_contracts",
        "all_three_seeds_verified", "research_evaluation_authorized",
        "test_not_opened",
    }
)


class CheckpointContractError(ValueError):
    """A checkpoint or freeze manifest failed the sealed contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def read_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CheckpointContractError(f"non-finite JSON literal {value} in {path}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise CheckpointContractError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointContractError(f"{path} must contain a JSON object")
    return value


def deterministic_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _population_signature(signature: Mapping[str, Any], split: str) -> tuple[str, int]:
    try:
        population = signature[split]
        ids = [str(value) for value in population["surface_ids"]]
        hashes = [str(value) for value in population["parameter_vector_hashes"]]
        count = int(population["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointContractError(f"invalid {split} population signature") from error
    if count != len(ids) or len(ids) != len(set(ids)) or len(ids) != len(hashes):
        raise CheckpointContractError(f"invalid {split} population cardinality")
    payload = {"surface_ids": ids, "parameter_vector_hashes": hashes}
    return deterministic_digest(payload), count


def _read_frame(path: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != list(fields):
        raise CheckpointContractError(f"history schema mismatch in {path}")
    return frame


def _read_epoch_history(path: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    frame = _read_frame(path, fields)
    epochs = frame["epoch"].astype(int).tolist()
    expected = list(range(1, len(epochs) + 1))
    if not epochs or epochs != expected or max(epochs) > MAX_EPOCHS:
        raise CheckpointContractError(f"incomplete or invalid epoch history: {path}")
    return frame


def _read_batch_history(path: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    return _read_frame(path, fields)


def _tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _tensor_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _tensor_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _verify_artifact_manifest(run_root: Path) -> None:
    manifest_path = run_root / "artifact_manifest.json"
    payload = read_json(manifest_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CheckpointContractError("invalid Stage-B artifact manifest")
    actual = {
        path.relative_to(run_root).as_posix(): path
        for path in run_root.rglob("*")
        if path.is_file()
    }
    actual.pop("artifact_manifest.json", None)
    failures: list[str] = []
    for relative, metadata in artifacts.items():
        path = actual.pop(relative, None)
        if path is None:
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != metadata.get("sha256"):
            failures.append(f"hash_mismatch:{relative}")
    failures.extend(f"extra:{relative}" for relative in actual)
    if failures:
        raise CheckpointContractError("; ".join(sorted(failures)))


def _validate_completion(
    validation_history: pd.DataFrame, best_epoch: int
) -> str:
    values = validation_history["validation_total_loss"].to_numpy(dtype=float)
    completed = len(values)
    if not np.isfinite(values).all():
        raise CheckpointContractError("non-finite validation history")
    canonical_best = int(np.argmin(values)) + 1
    if canonical_best != best_epoch:
        raise CheckpointContractError("best epoch does not select minimum validation loss")
    if completed == MAX_EPOCHS:
        return "COMPLETE_MAX_EPOCHS"
    if best_epoch + EARLY_STOPPING_PATIENCE <= completed:
        return "COMPLETE_EARLY_STOPPED"
    raise CheckpointContractError("partial/interrupted Stage-B run cannot satisfy contract")


def build_seed_contract(
    run_root: str | Path,
    *,
    seed: int,
    experiment_id: str,
    expected_train_population_sha256: str | None = None,
    expected_validation_population_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one raw Stage-B run and emit its immutable identity contract."""
    root = Path(run_root)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise CheckpointContractError(f"missing Stage-B artifacts: {missing}")
    _verify_artifact_manifest(root)

    train_frame = _read_epoch_history(root / "train_history.csv", TRAIN_HISTORY_FIELDS)
    validation_frame = _read_epoch_history(
        root / "validation_history.csv", VALIDATION_HISTORY_FIELDS
    )
    physics_frame = _read_batch_history(
        root / "physics_diagnostics.csv", PHYSICS_FIELDS
    )
    gradient_frame = _read_batch_history(
        root / "gradient_diagnostics.csv", GRADIENT_FIELDS
    )
    if len(train_frame) != len(validation_frame):
        raise CheckpointContractError("training/validation histories have unequal epochs")
    epochs_completed = len(validation_frame)

    environment = read_json(root / "environment_provenance.json")
    metadata = read_json(root / "epoch_metadata.json")
    checkpoint_payload = torch.load(
        root / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    optimizer_payload = torch.load(
        root / "optimizer.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, dict) or not isinstance(optimizer_payload, dict):
        raise CheckpointContractError("invalid torch checkpoint payloads")
    stored_metadata = checkpoint_payload.get("metadata")
    if stored_metadata != metadata or stored_metadata != optimizer_payload.get("metadata"):
        raise CheckpointContractError("checkpoint/optimizer/metadata identities differ")

    required_identity = {
        "run_kind": STAGE_B_RUN_KIND,
        "protocol_name": "MODEL3_GENUINE_PDE_DOUBLE_HESTON",
        "protocol_version": PROTOCOL_VERSION,
        "allowed_splits": ["train", "validation"],
        "forbidden_split": "test",
        "tracked_git_dirty": False,
    }
    mismatches = [
        f"{key}:{stored_metadata.get(key)!r}"
        for key, value in required_identity.items()
        if stored_metadata.get(key) != value
    ]
    if mismatches:
        raise CheckpointContractError(f"identity mismatch: {mismatches}")
    settings = dict(stored_metadata["settings"])
    scientific_settings = {
        key: settings.get(key)
        for key in FROZEN_SETTINGS
        if key not in {"learning_rate", "weight_decay"}
    }
    for numeric_key in ("learning_rate", "weight_decay"):
        value = settings.get(numeric_key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CheckpointContractError(f"invalid {numeric_key}")
        scientific_settings[numeric_key] = float(value)
    if scientific_settings != FROZEN_SETTINGS or int(settings["seed"]) != seed:
        raise CheckpointContractError("Stage-B settings/seed mismatch")
    if dict(stored_metadata["loss_weights"]) != FROZEN_LOSS_WEIGHTS:
        raise CheckpointContractError("frozen Model3 loss weights mismatch")
    required_environment = {
        "device_selected": FROZEN_SETTINGS["device"],
        "deterministic_algorithms": True,
        "float64_physics_boundary": True,
        "real_market_inputs_used": False,
        "issue34_numeric_outcomes_used": False,
    }
    environment_mismatches = [
        f"{key}:{environment.get(key)!r}"
        for key, value in required_environment.items()
        if environment.get(key) != value
    ]
    if environment_mismatches:
        raise CheckpointContractError(
            f"environment provenance mismatch: {environment_mismatches}"
        )

    train_population_sha, train_count = _population_signature(
        stored_metadata["subset_signature"], "train"
    )
    validation_population_sha, validation_count = _population_signature(
        stored_metadata["subset_signature"], "validation"
    )
    expected_train = FROZEN_SETTINGS["train_limit"]
    expected_validation = FROZEN_SETTINGS["validation_limit"]
    if train_count != expected_train or validation_count != expected_validation:
        raise CheckpointContractError("Stage-B population size mismatch")
    if expected_train_population_sha256 is not None and train_population_sha != expected_train_population_sha256:
        raise CheckpointContractError("training population differs from canonical dataset")
    if expected_validation_population_sha256 is not None and validation_population_sha != expected_validation_population_sha256:
        raise CheckpointContractError("validation population differs from canonical dataset")
    train_ids = set(stored_metadata["subset_signature"]["train"]["surface_ids"])
    validation_ids = set(stored_metadata["subset_signature"]["validation"]["surface_ids"])
    if train_ids & validation_ids:
        raise CheckpointContractError("training/validation populations overlap")

    best_epoch = int(checkpoint_payload["best_epoch"])
    completion_status = _validate_completion(validation_frame, best_epoch)
    if (
        int(checkpoint_payload["completed_epoch"]) != best_epoch
        or int(optimizer_payload["completed_epoch"]) != best_epoch
    ):
        raise CheckpointContractError("selected checkpoint is not the historical best state")
    if not _tensor_equal(
        checkpoint_payload.get("model_state_dict"),
        checkpoint_payload.get("best_model_state_dict"),
    ) or not _tensor_equal(
        optimizer_payload.get("optimizer_state_dict"),
        checkpoint_payload.get("best_optimizer_state_dict"),
    ):
        raise CheckpointContractError("exported best-state mismatch")
    if not bool(train_frame["finite_gradients"].astype(bool).all()) or not bool(
        gradient_frame["finite_gradients"].astype(bool).all()
    ):
        raise CheckpointContractError("non-finite gradient diagnostics")
    physics_train = physics_frame["split"].astype(str).eq("train")
    batches_per_epoch = math.ceil(expected_train / FROZEN_SETTINGS["batch_size"])
    expected_physics_rows = batches_per_epoch * epochs_completed
    if int(physics_train.sum()) != expected_physics_rows or len(physics_frame) != expected_physics_rows:
        raise CheckpointContractError("physics diagnostics do not cover every training batch")
    if len(gradient_frame) != expected_physics_rows:
        raise CheckpointContractError("gradient diagnostics do not cover every training batch")

    artifact_hash_sources = {
        "checkpoint_sha256": "checkpoint.pt",
        "training_history_sha256": "train_history.csv",
        "validation_history_sha256": "validation_history.csv",
        "physics_diagnostics_sha256": "physics_diagnostics.csv",
        "gradient_diagnostics_sha256": "gradient_diagnostics.csv",
        "environment_provenance_sha256": "environment_provenance.json",
        "artifact_manifest_sha256": "artifact_manifest.json",
    }
    artifact_hashes = {
        field: sha256_file(root / relative)
        for field, relative in artifact_hash_sources.items()
    }
    artifact_header = read_json(root / "artifact_manifest.json")
    training_host = artifact_header.get("host")
    training_platform = artifact_header.get("platform")
    if not isinstance(training_host, str) or not training_host:
        raise CheckpointContractError("artifact manifest lacks training host")
    if not isinstance(training_platform, str) or not training_platform:
        raise CheckpointContractError("artifact manifest lacks training platform")
    contract: dict[str, Any] = {
        "schema_version": "MODEL3_RESEARCH_CHECKPOINT_CONTRACT_V1",
        "experiment_id": str(experiment_id),
        "source_git_sha": str(stored_metadata["git_sha"]),
        "protocol_name": required_identity["protocol_name"],
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": str(stored_metadata["config_sha256"]),
        "final_r2_dataset_sha256": str(stored_metadata["dataset_sha256"]),
        "seed": int(seed),
        "train_population_sha256": train_population_sha,
        "train_population_count": train_count,
        "validation_population_sha256": validation_population_sha,
        "validation_population_count": validation_count,
        "test_not_opened": True,
        "run_kind": STAGE_B_RUN_KIND,
        "settings": FROZEN_SETTINGS,
        "loss_weights": dict(stored_metadata["loss_weights"]),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "checkpoint_selection_rule": "minimum_validation_total_loss_only",
        **artifact_hashes,
        "completion_status": completion_status,
        "training_host": training_host,
        "training_platform": training_platform,
    }
    if set(contract) != CHECKPOINT_CONTRACT_FIELDS:
        raise AssertionError("checkpoint schema drifted without a version change")
    return contract


def verify_freeze_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify an independently serialized three-seed freeze manifest."""
    if set(payload) != FREEZE_MANIFEST_FIELDS:
        raise CheckpointContractError("freeze-manifest schema mismatch")
    expected_top = {
        "schema_version": "MODEL3_THREE_SEED_FREEZE_MANIFEST_V1",
        "manifest_kind": "FINAL_MODEL3_RESEARCH_CHECKPOINT_FREEZE",
        "all_three_seeds_verified": True,
        "research_evaluation_authorized": False,
        "test_not_opened": True,
    }
    failures = [
        f"{key}:{payload.get(key)!r}"
        for key, value in expected_top.items()
        if payload.get(key) != value
    ]
    if failures:
        raise CheckpointContractError(f"freeze manifest header mismatch: {failures}")
    seeds_value = payload["seeds"]
    seed_contracts = payload["seed_contracts"]
    if seeds_value != list(REQUIRED_SEEDS) or not isinstance(seed_contracts, dict):
        raise CheckpointContractError("freeze manifest must contain exactly seeds 11/22/33")
    if set(seed_contracts) != {str(seed) for seed in REQUIRED_SEEDS}:
        raise CheckpointContractError("missing or duplicate seed contracts")
    parsed: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    shared_fields = (
        "experiment_id", "source_git_sha", "protocol_name", "protocol_version",
        "config_sha256", "final_r2_dataset_sha256", "train_population_sha256",
        "train_population_count", "validation_population_sha256",
        "validation_population_count", "settings", "loss_weights",
        "checkpoint_selection_rule", "run_kind",
    )
    first = seed_contracts[str(REQUIRED_SEEDS[0])]
    for raw_seed, contract in seed_contracts.items():
        try:
            if not isinstance(contract, Mapping) or set(contract) != CHECKPOINT_CONTRACT_FIELDS:
                raise CheckpointContractError("checkpoint contract schema mismatch")
            if int(contract["seed"]) != int(raw_seed) or int(raw_seed) not in REQUIRED_SEEDS:
                raise CheckpointContractError("wrong/duplicate seed identity")
            differences = [
                field for field in shared_fields if contract.get(field) != first.get(field)
            ]
            if differences:
                raise CheckpointContractError(f"cross-seed identity differs: {differences}")
            if contract["final_r2_dataset_sha256"] != EXPECTED_DATASET_SHA256:
                raise CheckpointContractError("frozen R2 dataset identity mismatch")
            if contract["config_sha256"] != sha256_file(PROTOCOL_CONFIG_PATH):
                raise CheckpointContractError("Model3 protocol config identity mismatch")
            if contract["protocol_version"] != PROTOCOL_VERSION or contract["run_kind"] != STAGE_B_RUN_KIND:
                raise CheckpointContractError("wrong protocol/run kind")
            if contract["test_not_opened"] is not True:
                raise CheckpointContractError("checkpoint did not declare test closure")
            if not str(contract["completion_status"]).startswith("COMPLETE_"):
                raise CheckpointContractError("incomplete checkpoint")
            if int(contract["best_epoch"]) <= 0 or int(contract["epochs_completed"]) < int(contract["best_epoch"]):
                raise CheckpointContractError("invalid selected checkpoint epoch")
            parsed[int(raw_seed)] = dict(contract)
        except (CheckpointContractError, KeyError, TypeError, ValueError) as error:
            errors.append(f"seed_{raw_seed}: {error}")
    if errors:
        raise CheckpointContractError("; ".join(errors))
    return {
        "valid": True,
        "seeds": list(REQUIRED_SEEDS),
        "source_git_sha": payload["source_git_sha"],
        "experiment_id": payload["experiment_id"],
        "config_sha256": payload["config_sha256"],
        "final_r2_dataset_sha256": payload["final_r2_dataset_sha256"],
        "seed_epochs": {str(seed): parsed[seed]["best_epoch"] for seed in REQUIRED_SEEDS},
    }


def build_freeze_manifest(
    output_roots: Mapping[int, str | Path],
    *,
    experiment_id: str,
    expected_train_population_sha256: str | None = None,
    expected_validation_population_sha256: str | None = None,
) -> dict[str, Any]:
    contracts = {
        str(seed): build_seed_contract(
            output_roots[seed],
            seed=seed,
            experiment_id=experiment_id,
            expected_train_population_sha256=expected_train_population_sha256,
            expected_validation_population_sha256=expected_validation_population_sha256,
        )
        for seed in REQUIRED_SEEDS
    }
    first = contracts[str(REQUIRED_SEEDS[0])]
    manifest: dict[str, Any] = {
        "schema_version": "MODEL3_THREE_SEED_FREEZE_MANIFEST_V1",
        "manifest_kind": "FINAL_MODEL3_RESEARCH_CHECKPOINT_FREEZE",
        "experiment_id": experiment_id,
        "source_git_sha": first["source_git_sha"],
        "protocol_name": first["protocol_name"],
        "protocol_version": first["protocol_version"],
        "config_sha256": first["config_sha256"],
        "final_r2_dataset_sha256": first["final_r2_dataset_sha256"],
        "seeds": list(REQUIRED_SEEDS),
        "seed_contracts": contracts,
        "all_three_seeds_verified": True,
        "research_evaluation_authorized": False,
        "test_not_opened": True,
    }
    verify_freeze_manifest(manifest)
    return manifest


def render_report(verification: Mapping[str, Any]) -> str:
    lines = [
        "MODEL3 THREE-SEED RESEARCH FREEZE: PASS",
        f"experiment_id={verification['experiment_id']}",
        f"source_git_sha={verification['source_git_sha']}",
        f"config_sha256={verification['config_sha256']}",
        f"dataset_sha256={verification['final_r2_dataset_sha256']}",
    ]
    for seed in REQUIRED_SEEDS:
        lines.append(f"seed={seed} best_epoch={verification['seed_epochs'][str(seed)]}")
    lines.append("frozen_clean_test=NOT_AUTHORIZED_BY_THIS_REPORT")
    return "\n".join(lines) + "\n"
