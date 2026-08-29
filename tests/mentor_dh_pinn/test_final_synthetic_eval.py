from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.final_synthetic_eval import (
    EXPECTED_CHECKPOINT_MATRIX,
    EXPECTED_PHASE1_DATASET_SHA256,
    FINAL_CLAIM_SCHEMA,
    FINAL_DATASET_SCHEMA,
    FINAL_EVAL_COUNT,
    FINAL_EVAL_SEED,
    PRIMARY_RUN_IDS,
    RUN_SUMMARY_FIELDS,
    SECONDARY_RUN_IDS,
    CheckpointIdentity,
    FinalSyntheticDataset,
    generate_final_synthetic_dataset,
    load_final_synthetic_eval_spec,
    run_final_synthetic_eval,
    summarize_final_synthetic_eval,
    validate_frozen_checkpoint_provenance,
)
from src.mentor_dh_pinn.parameter_source import (
    EXPECTED_FIRST_PARAMETER_HASH,
    EXPECTED_FIRST_PARAMETER_VECTOR,
    EXPECTED_FIRST_SURFACE_ID,
    FROZEN_SURFACES_SHA256,
    ParameterSource,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/mentor_dh_pinn/final_synthetic_eval_v1.yaml"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_hash(values) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _source() -> ParameterSource:
    return ParameterSource(
        vector=EXPECTED_FIRST_PARAMETER_VECTOR.copy(),
        record={"surface_id": EXPECTED_FIRST_SURFACE_ID},
        dataset_path="frozen/surfaces.jsonl",
        dataset_sha256=FROZEN_SURFACES_SHA256,
        parameter_hash=EXPECTED_FIRST_PARAMETER_HASH,
        surface_id=EXPECTED_FIRST_SURFACE_ID,
        split="train",
    )


def _checkpoint_tuple(setting) -> tuple:
    return (
        setting.run_id,
        setting.role,
        setting.variant,
        setting.seed,
        setting.path,
        setting.lambda_pde,
        setting.lambda_boundary,
        setting.lambda_terminal,
        setting.lambda_data,
    )


def test_protocol_freezes_final_seed_count_and_checkpoint_roles() -> None:
    spec = load_final_synthetic_eval_spec(CONFIG_PATH)
    assert spec.schema_version == "mentor_dh_pinn_final_synthetic_eval_v1"
    assert spec.final_eval_seed == FINAL_EVAL_SEED == 73129
    assert spec.final_eval_count == FINAL_EVAL_COUNT == 4096
    assert spec.final_eval_seed not in {3407, 11, 22, 33}
    assert tuple(_checkpoint_tuple(item) for item in spec.checkpoints) == EXPECTED_CHECKPOINT_MATRIX
    assert tuple(item.run_id for item in spec.checkpoints if item.role == "primary") == PRIMARY_RUN_IDS
    assert tuple(
        item.run_id for item in spec.checkpoints if item.role == "secondary_ablation"
    ) == SECONDARY_RUN_IDS
    assert {item.variant for item in spec.checkpoints} == {"equal", "optimized"}
    assert {
        (item.lambda_pde, item.lambda_boundary, item.lambda_terminal, item.lambda_data)
        for item in spec.checkpoints
    } == {(1.0, 1.0, 1.0, 1.0), (0.1, 0.1, 1.0, 1.0)}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("final_eval_seed", 3407, "seed is frozen"),
        ("final_eval_seed", 73130, "seed is frozen"),
        ("final_eval_count", 4095, "count is frozen"),
        ("expected_phase1_dataset_sha256", "bad", "dataset SHA anchor"),
    ),
)
def test_protocol_rejects_seed_count_and_identity_drift(
    tmp_path: Path, field: str, value, message: str
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw[field] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_final_synthetic_eval_spec(changed)


def test_final_cohort_is_deterministic_split_free_and_new(monkeypatch, tmp_path: Path) -> None:
    import src.mentor_dh_pinn.final_synthetic_eval as module

    spec = load_final_synthetic_eval_spec(CONFIG_PATH)
    baseline = load_baseline_config()
    source = _source()
    monkeypatch.setattr(
        module,
        "_price_call",
        lambda spot, strike, tau, rate, carry, parameters, node_count: (
            spot + strike + tau + rate + carry + parameters[4] + parameters[9]
        ),
    )
    monkeypatch.setattr(module, "_git_sha", lambda root: "git-sha")
    left = generate_final_synthetic_dataset(
        spec, baseline, source, tmp_path / "left", repo_root=tmp_path
    )
    right = generate_final_synthetic_dataset(
        spec, baseline, source, tmp_path / "right", repo_root=tmp_path
    )
    assert left.size == right.size == 4096
    np.testing.assert_array_equal(left.features, right.features)
    np.testing.assert_array_equal(left.reference_prices, right.reference_prices)
    np.testing.assert_array_equal(left.sample_ids, right.sample_ids)
    assert left.manifest["dataset_npz_sha256"] == right.manifest["dataset_npz_sha256"]
    assert left.manifest["sample_id_hash"] == right.manifest["sample_id_hash"]
    assert left.manifest["sample_id_hash"] not in {"train-hash", "validation-hash", "test-hash"}
    assert left.manifest["seed"] == 73129
    assert left.manifest["final_holdout_only"] is True
    for forbidden_api in ("indices", "split_id_hashes", "train", "validation", "test"):
        assert not hasattr(left, forbidden_api)
    with pytest.raises(ValueError):
        left.features[0, 0] = 0.0


def _write_phase_artifacts(root: Path, spec, baseline, source) -> dict:
    (root / "configs/mentor_dh_pinn").mkdir(parents=True, exist_ok=True)
    (root / spec.phase2d_config_path).write_text(
        (REPO_ROOT / spec.phase2d_config_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    phase1_dataset = {
        "dataset_sha256": EXPECTED_PHASE1_DATASET_SHA256,
        "split_id_hashes": {
            "train": "train-hash",
            "validation": "validation-hash",
            "test": "test-hash",
        },
        "parameter_source": source.provenance(),
    }
    phase1_checkpoint = {
        "config": baseline.to_dict(),
        "seed": 3407,
        "weights": {"pde": 1.0, "boundary": 1.0, "terminal": 1.0, "data": 1.0},
        "best_epoch": 909,
        "best_validation_nrmse": 0.0012545129490993494,
        "dataset_identity": phase1_dataset,
        "provenance": source.provenance(),
        "git_sha": "phase1-git",
        "model_state_dict": {},
    }
    phase1_path = root / spec.phase1_checkpoint_path
    phase1_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phase1_checkpoint, phase1_path)
    phase2d_root = root / spec.phase2d_output_dir
    phase2d_root.mkdir(parents=True, exist_ok=True)
    (phase2d_root / "multiseed_aggregate_summary.json").write_text(
        json.dumps(
            {
                "optimized_confirmed": False,
                "inferential_significance_test_performed": False,
                "seeds": [11, 22, 33],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for setting in spec.checkpoints[1:]:
        expected_config = baseline.to_dict()
        expected_config["seed"] = setting.seed
        expected_config["losses"].update(
            pde_lambda=setting.lambda_pde,
            boundary_lambda=setting.lambda_boundary,
            terminal_lambda=setting.lambda_terminal,
            data_lambda=setting.lambda_data,
        )
        checkpoint = {
            "config": expected_config,
            "seed": setting.seed,
            "weights": setting.weights,
            "dataset_identity": phase1_dataset,
            "provenance": source.provenance(),
            "git_sha": f"git-{setting.run_id}",
            "model_state_dict": {},
        }
        checkpoint_path = root / setting.path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        result = {
            "run_id": setting.run_id,
            "variant": setting.variant,
            "seed": setting.seed,
            "lambda_pde": setting.lambda_pde,
            "lambda_boundary": setting.lambda_boundary,
            "lambda_terminal": setting.lambda_terminal,
            "lambda_data": setting.lambda_data,
            "cohort_dataset_sha256": EXPECTED_PHASE1_DATASET_SHA256,
            "train_split_id_hash": "train-hash",
            "validation_split_id_hash": "validation-hash",
            "git_sha": f"git-{setting.run_id}",
        }
        (checkpoint_path.parent / "multiseed_result.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
    return phase1_checkpoint


def test_phase1_and_phase2d_provenance_drift_fail_closed(tmp_path: Path) -> None:
    spec = load_final_synthetic_eval_spec(CONFIG_PATH)
    baseline = load_baseline_config()
    source = _source()
    phase1 = _write_phase_artifacts(tmp_path, spec, baseline, source)
    identities, dataset_identity = validate_frozen_checkpoint_provenance(
        spec, baseline, source, repo_root=tmp_path
    )
    assert [item.setting.run_id for item in identities] == [
        "EQ3407", "EQ11", "EQ22", "EQ33", "OPT11", "OPT22", "OPT33"
    ]
    assert dataset_identity["dataset_sha256"] == EXPECTED_PHASE1_DATASET_SHA256

    phase1["dataset_identity"] = {
        **phase1["dataset_identity"],
        "dataset_sha256": "changed",
    }
    torch.save(phase1, tmp_path / spec.phase1_checkpoint_path)
    with pytest.raises(ValueError, match="generated dataset SHA mismatch"):
        validate_frozen_checkpoint_provenance(spec, baseline, source, repo_root=tmp_path)

    _write_phase_artifacts(tmp_path, spec, baseline, source)
    aggregate_path = tmp_path / spec.phase2d_output_dir / "multiseed_aggregate_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["optimized_confirmed"] = True
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ValueError, match="optimized_confirmed == false"):
        validate_frozen_checkpoint_provenance(spec, baseline, source, repo_root=tmp_path)


def _fake_final_dataset(spec, source, root: Path) -> FinalSyntheticDataset:
    features = np.ones((spec.final_eval_count, 7), dtype=np.float64)
    references = np.zeros(spec.final_eval_count, dtype=np.float64)
    sample_ids = np.asarray(
        [f"mentor_dh_pinn_final_v1_{index:06d}" for index in range(spec.final_eval_count)]
    )
    dataset_path = root / spec.dataset_filename
    np.savez_compressed(
        dataset_path,
        features=features,
        reference_prices=references,
        sample_ids=sample_ids,
    )
    sample_hash = _sample_hash(sample_ids)
    manifest = {
        "schema_version": FINAL_DATASET_SCHEMA,
        "seed": spec.final_eval_seed,
        "count": spec.final_eval_count,
        "feature_names": [
            "spot", "variance_slow", "variance_fast", "tau", "strike", "rate", "carry"
        ],
        "frozen_surfaces_sha256": FROZEN_SURFACES_SHA256,
        "parameter_hash": EXPECTED_FIRST_PARAMETER_HASH,
        "source_surface_id": EXPECTED_FIRST_SURFACE_ID,
        "domain": {},
        "baseline_config_sha256": "baseline-config-sha",
        "pricing_node_count": 64,
        "sample_ids": sample_ids.tolist(),
        "dataset_filename": spec.dataset_filename,
        "dataset_npz_sha256": _file_sha256(dataset_path),
        "sample_id_hash": sample_hash,
        "generation_code_git_sha": "current-git",
        "final_holdout_only": True,
        "used_for_training": False,
        "used_for_validation": False,
    }
    (root / spec.dataset_manifest_filename).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FinalSyntheticDataset(features, references, sample_ids, source, manifest)


def _metric(
    setting,
    diagnostics,
    value: float,
    dataset_sha: str = "final-dataset-sha",
    sample_hash: str = "final-sample-hash",
) -> dict:
    return {
        "schema_version": "mentor_dh_pinn_final_checkpoint_metrics_v1",
        "run_id": setting.run_id,
        "role": setting.role,
        "variant": setting.variant,
        "seed": setting.seed,
        "lambda_pde": setting.lambda_pde,
        "lambda_boundary": setting.lambda_boundary,
        "lambda_terminal": setting.lambda_terminal,
        "lambda_data": setting.lambda_data,
        "checkpoint_path": setting.path,
        "checkpoint_sha256": f"sha-{setting.run_id}",
        "checkpoint_git_sha": f"git-{setting.run_id}",
        "final_dataset_sha256": dataset_sha,
        "sample_id_hash": sample_hash,
        "diagnostic_identity": diagnostics,
        "price_rmse": value + 0.001,
        "price_mae": value + 0.002,
        "price_nrmse": value,
        "pde_rms": value + 0.003,
        "pde_max_abs": value + 0.004,
        "terminal_rmse": value + 0.005,
        "terminal_max_abs": value + 0.006,
        "boundary_low_s_rmse": value + 0.007,
        "boundary_high_s_rmse": value + 0.008,
        "inference_seconds_total": value + 1.0,
        "inference_seconds_per_contract": value + 0.0001,
        "all_finite": True,
    }


def test_runner_uses_same_holdout_and_physics_and_is_one_shot(monkeypatch, tmp_path: Path) -> None:
    import src.mentor_dh_pinn.final_synthetic_eval as module

    spec = load_final_synthetic_eval_spec(CONFIG_PATH)
    baseline = load_baseline_config()
    source = _source()
    identities = tuple(
        CheckpointIdentity(
            setting=setting,
            path=tmp_path / setting.path,
            sha256=f"sha-{setting.run_id}",
            payload={"git_sha": f"git-{setting.run_id}", "model_state_dict": {}},
        )
        for setting in spec.checkpoints
    )
    phase1_identity = {
        "split_id_hashes": {
            "train": "train-hash",
            "validation": "validation-hash",
            "test": "test-hash",
        }
    }
    monkeypatch.setattr(module, "load_baseline_config", lambda path: baseline)
    monkeypatch.setattr(module, "select_first_eligible_train_record", lambda *a, **k: source)
    monkeypatch.setattr(
        module,
        "validate_frozen_checkpoint_provenance",
        lambda *a, **k: (identities, phase1_identity),
    )
    monkeypatch.setattr(module, "_git_sha", lambda root: "current-git")

    def fake_generate(specification, config, parameter_source, output_dir, **kwargs):
        return _fake_final_dataset(specification, parameter_source, Path(output_dir))

    diagnostics_identity = {
        "final_eval_seed": 73129,
        "pde_sha256": "pde-sha",
        "terminal_sha256": "terminal-sha",
        "low_boundary_sha256": "low-sha",
        "high_boundary_sha256": "high-sha",
    }
    diagnostics = SimpleNamespace(identity=diagnostics_identity)
    monkeypatch.setattr(module, "generate_final_synthetic_dataset", fake_generate)
    monkeypatch.setattr(module, "build_common_diagnostics", lambda *a, **k: diagnostics)
    observed: list[tuple[int, int, str]] = []

    def fake_evaluate(identity, dataset, supplied_diagnostics, config, **kwargs):
        observed.append((id(dataset.features), id(supplied_diagnostics), identity.setting.run_id))
        return _metric(
            identity.setting,
            diagnostics_identity,
            0.01 + len(observed) / 1000.0,
            dataset.manifest["dataset_npz_sha256"],
            dataset.manifest["sample_id_hash"],
        )

    monkeypatch.setattr(module, "_evaluate_checkpoint", fake_evaluate)
    outputs = run_final_synthetic_eval(CONFIG_PATH, repo_root=tmp_path)
    assert set(outputs) == {
        "claim", "dataset_manifest", "run_summary", "primary_aggregate",
        "secondary_ablation", "evaluation_manifest",
    }
    assert [item[2] for item in observed] == [item.run_id for item in spec.checkpoints]
    assert len({item[0] for item in observed}) == 1
    assert len({item[1] for item in observed}) == 1
    claim = json.loads(outputs["claim"].read_text(encoding="utf-8"))
    assert claim["primary_run_ids"] == list(PRIMARY_RUN_IDS)
    assert claim["secondary_ablation_run_ids"] == list(SECONDARY_RUN_IDS)
    with pytest.raises(FileExistsError, match="requires explicit recovery"):
        run_final_synthetic_eval(CONFIG_PATH, repo_root=tmp_path)


def test_source_and_scripts_have_no_training_or_old_test_dependency() -> None:
    paths = (
        REPO_ROOT / "src/mentor_dh_pinn/final_synthetic_eval.py",
        REPO_ROOT / "scripts/mentor_dh_pinn/run_final_synthetic_eval.py",
        REPO_ROOT / "scripts/mentor_dh_pinn/summarize_final_synthetic_eval.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "from .trainer" not in text
    assert "train_baseline" not in text
    assert "evaluate_test_once" not in text
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert ".indices(\"test\")" not in text
    assert "model_selection_performed_on_final_holdout\": True" not in text


def _write_completed_summary_inputs(tmp_path: Path, values: dict[str, float]):
    spec = load_final_synthetic_eval_spec(CONFIG_PATH)
    output_root = tmp_path / spec.output_root
    metrics_root = output_root / spec.metrics_subdirectory
    metrics_root.mkdir(parents=True)
    sample_ids = np.asarray(
        [f"mentor_dh_pinn_final_v1_{index:06d}" for index in range(4096)]
    )
    dataset_path = output_root / spec.dataset_filename
    np.savez_compressed(
        dataset_path,
        features=np.ones((4096, 7)),
        reference_prices=np.zeros(4096),
        sample_ids=sample_ids,
    )
    dataset_sha = _file_sha256(dataset_path)
    sample_hash = _sample_hash(sample_ids)
    dataset_manifest = {
        "schema_version": FINAL_DATASET_SCHEMA,
        "seed": 73129,
        "count": 4096,
        "feature_names": [
            "spot", "variance_slow", "variance_fast", "tau", "strike", "rate", "carry"
        ],
        "frozen_surfaces_sha256": FROZEN_SURFACES_SHA256,
        "parameter_hash": EXPECTED_FIRST_PARAMETER_HASH,
        "source_surface_id": EXPECTED_FIRST_SURFACE_ID,
        "domain": {},
        "baseline_config_sha256": "baseline-config-sha",
        "pricing_node_count": 64,
        "sample_ids": sample_ids.tolist(),
        "dataset_filename": spec.dataset_filename,
        "dataset_npz_sha256": dataset_sha,
        "sample_id_hash": sample_hash,
        "generation_code_git_sha": "current-git",
        "final_holdout_only": True,
        "used_for_training": False,
        "used_for_validation": False,
    }
    (output_root / spec.dataset_manifest_filename).write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostics = {"common": "physics"}
    claim = {
        "schema_version": FINAL_CLAIM_SCHEMA,
        "final_seed": 73129,
        "final_cohort_sha256": dataset_sha,
        "sample_id_hash": sample_hash,
        "current_git_sha": "current-git",
        "checkpoint_identities": [
            {
                "run_id": setting.run_id,
                "role": setting.role,
                "variant": setting.variant,
                "seed": setting.seed,
                "checkpoint_sha256": f"sha-{setting.run_id}",
                "checkpoint_git_sha": f"git-{setting.run_id}",
                "weights": setting.weights,
            }
            for setting in spec.checkpoints
        ],
        "primary_run_ids": list(PRIMARY_RUN_IDS),
        "secondary_ablation_run_ids": list(SECONDARY_RUN_IDS),
        "diagnostic_identity": diagnostics,
        "final_loss_choice": "equal",
        "loss_choice_frozen_before_final_evaluation": True,
    }
    (output_root / spec.claim_filename).write_text(
        json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for setting in spec.checkpoints:
        payload = _metric(
            setting,
            diagnostics,
            values[setting.run_id],
            dataset_sha,
            sample_hash,
        )
        (metrics_root / f"{setting.run_id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return spec, output_root


def test_summaries_use_all_primary_runs_and_secondary_cannot_select(monkeypatch, tmp_path: Path) -> None:
    values = {
        "EQ3407": 0.010,
        "EQ11": 0.020,
        "EQ22": 0.030,
        "EQ33": 0.040,
        "OPT11": 0.0001,
        "OPT22": 0.0002,
        "OPT33": 0.0003,
    }
    spec, output_root = _write_completed_summary_inputs(tmp_path, values)
    first = summarize_final_synthetic_eval(CONFIG_PATH, repo_root=tmp_path)
    first_bytes = {name: path.read_bytes() for name, path in first.items()}
    second = summarize_final_synthetic_eval(CONFIG_PATH, repo_root=tmp_path)
    assert {name: path.read_bytes() for name, path in second.items()} == first_bytes

    with first["run_summary"].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == RUN_SUMMARY_FIELDS
    assert [row["run_id"] for row in rows] == [item.run_id for item in spec.checkpoints]
    aggregate = json.loads(first["primary_aggregate"].read_text(encoding="utf-8"))
    assert aggregate["primary_run_ids"] == list(PRIMARY_RUN_IDS)
    assert aggregate["checkpoint_count"] == 4
    assert set(aggregate["individual_checkpoints"]) == set(PRIMARY_RUN_IDS)
    assert aggregate["price_nrmse"]["mean"] == pytest.approx(0.025)
    assert aggregate["final_loss_choice"] == "equal"
    assert aggregate["loss_choice_frozen_before_final_evaluation"] is True
    assert aggregate["model_selection_performed_on_final_holdout"] is False
    assert aggregate["final_holdout_used_for_training"] is False
    assert aggregate["final_holdout_used_for_validation"] is False
    assert aggregate["inferential_significance_test_performed"] is False

    with first["secondary_ablation"].open("r", newline="", encoding="utf-8") as handle:
        secondary = list(csv.DictReader(handle))
    assert [row["run_id"] for row in secondary] == list(SECONDARY_RUN_IDS)
    assert all(row["secondary_ablation_only"] == "true" for row in secondary)
    assert all(row["may_reopen_model_selection"] == "false" for row in secondary)
    manifest = json.loads(first["evaluation_manifest"].read_text(encoding="utf-8"))
    assert manifest["final_loss_choice"] == "equal"
    assert manifest["may_reopen_model_selection"] is False

    (output_root / spec.metrics_subdirectory / "EQ33.json").unlink()
    with pytest.raises(ValueError, match="incomplete Phase 3A result"):
        summarize_final_synthetic_eval(CONFIG_PATH, repo_root=tmp_path)
