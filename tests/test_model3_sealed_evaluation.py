"""Fast sealed-evaluation fixtures; no frozen research data is opened."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from src.model3_evaluation.adapter import (
    Model3CheckpointError,
    Model3EvaluationAdapter,
)
from src.model3_evaluation.contracts import (
    CHECKPOINT_CONTRACT_FIELDS,
    FREEZE_MANIFEST_FIELDS,
    REQUIRED_SEEDS,
    build_freeze_manifest,
    build_seed_contract,
    verify_freeze_manifest,
)
from src.model3_evaluation.g8_adapter import G8AdapterError, validate_g8_request
from src.model3_evaluation.g8_adapter import G8_INCLUSION_CONDITION
from src.model3_evaluation.harness import (
    load_frozen_baseline_evidence,
    run_clean_evaluation,
)
from src.model3_evaluation.locking import FrozenTestLockedError, require_frozen_test_authorization
from src.model3_evaluation.noise_adapter import (
    FROZEN_NOISE_PROTOCOL_CONFIG_SHA256,
    NoiseAdapterError,
    validate_noise_request,
)
from src.model3_evaluation.ood_adapter import (
    OOD_COHORT_CONTENT_SHA256,
    OOD_PROTOCOL_CONFIG_SHA256,
    OODAdapterError,
    validate_ood_request,
)
from src.model3_evaluation.paper_export import PaperExportError, export_publication_tables
from src.r2_primary.dataset import R2PrimaryDataset, R2SurfaceItem
from src.r2_primary.evaluation import reprice_normalized, stability_metrics


BEST_EPOCH = 2
EPOCHS = 17  # best epoch + frozen patience proves the early-stop completion rule


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_frame(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _metadata(seed: int, train_limit: int, validation_limit: int) -> dict[str, Any]:
    return {
        "run_kind": "MODEL3_STAGE_B_RESEARCH_FROZEN",
        "settings": {
            "dataset": "data/final_r2_clean_10000/surfaces.jsonl",
            "output_root": f"outputs/model3_stage_b_seed_{seed}",
            "train_limit": train_limit,
            "validation_limit": validation_limit,
            "epochs": 120,
            "batch_size": 2 if train_limit == 4 else 32,
            "interior_points": 1 if train_limit == 4 else 32,
            "terminal_points": 1 if train_limit == 4 else 8,
            "learning_rate": 0.0002,
            "weight_decay": 0.00001,
            "device": "cuda",
            "seed": seed,
            "smoke_mode": False,
            "patience": 15,
            "run_kind": "MODEL3_STAGE_B_RESEARCH_FROZEN",
        },
        "loss_weights": {
            "parameter": 1.0, "reconstruction": 1.0, "pde_residual": 0.10,
            "terminal_diagnostic": 0.0, "boundary_penalty": 0.0,
        },
        "subset_signature": {
            "train": {
                "count": train_limit,
                "surface_ids": [f"train-{index}" for index in range(train_limit)],
                "parameter_vector_hashes": [f"train-hash-{index}" for index in range(train_limit)],
            },
            "validation": {
                "count": validation_limit,
                "surface_ids": [f"valid-{index}" for index in range(validation_limit)],
                "parameter_vector_hashes": [f"valid-hash-{index}" for index in range(validation_limit)],
            },
        },
        "git_sha": "a" * 40,
        "config_sha256": hashlib.sha256(
            Path("configs/model3_pde_protocol.yaml").read_bytes()
        ).hexdigest(),
        "dataset_sha256": "148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6",
        "protocol_name": "MODEL3_GENUINE_PDE_DOUBLE_HESTON",
        "protocol_version": "1.1",
        "allowed_splits": ["train", "validation"],
        "forbidden_split": "test",
        "tracked_git_dirty": False,
        "tracked_git_status": [],
    }


def _checkpoint_payload(metadata: dict[str, Any], seed: int) -> dict[str, Any]:
    state = {"fixture_weight": torch.tensor([float(seed)], dtype=torch.float64)}
    standardizer = {"mean": torch.zeros(10), "scale": torch.ones(10)}
    return {
        "completed_epoch": BEST_EPOCH,
        "model_state_dict": state,
        "optimizer_state_dict": state,
        "target_standardizer": standardizer,
        "best_validation_loss": 0.75,
        "best_epoch": BEST_EPOCH,
        "rng_states": {"python_random": (0, (0,), None), "cpu": torch.zeros(1, dtype=torch.uint8)},
        "best_model_state_dict": state,
        "best_optimizer_state_dict": state,
        "best_target_standardizer": standardizer,
        "best_rng_states": {"python_random": (0, (0,), None), "cpu": torch.zeros(1, dtype=torch.uint8)},
        "metadata": metadata,
    }


def _make_run(
    root: Path,
    *,
    seed: int,
    epochs: int = EPOCHS,
    stage_a: bool = False,
) -> None:
    root.mkdir(parents=True)
    metadata = _metadata(seed, train_limit=4, validation_limit=2)
    if stage_a:
        metadata["run_kind"] = "MODEL3_STAGE_A_DEVELOPMENT_PILOT_NOT_RESEARCH_RESULT"
        metadata["settings"]["run_kind"] = metadata["run_kind"]

    validation_rows = [
        {
            "epoch": epoch,
            "validation_parameter_loss": 1.0,
            "validation_reconstruction_loss": 1.0,
            "validation_pde_residual_loss": 1.0,
            "validation_total_loss": 0.75 if epoch == BEST_EPOCH else 1.0 + epoch / 1000,
        }
        for epoch in range(1, epochs + 1)
    ]
    train_rows = [
        {
            **{
                field: (True if field == "finite_gradients" else float(index + epoch / 100))
            for index, field in enumerate((
                "parameter_loss", "reconstruction_loss", "pde_residual_loss",
                "total_loss", "gradient_norm", "pde_residual_rms",
                "pde_residual_max_scaled_rms", "terminal_payoff_max_abs",
                "duration_seconds", "accelerator_memory_allocated_bytes",
                "accelerator_memory_reserved_bytes",
            ))
            },
            "epoch": epoch,
            "finite_gradients": True,
        }
        for epoch in range(1, epochs + 1)
    ]
    _write_csv(root / "train_history.csv", _history_frame(
        ("epoch", "parameter_loss", "reconstruction_loss", "pde_residual_loss", "total_loss",
         "finite_gradients", "gradient_norm", "pde_residual_rms",
         "pde_residual_max_scaled_rms", "terminal_payoff_max_abs", "duration_seconds",
         "accelerator_memory_allocated_bytes", "accelerator_memory_reserved_bytes"),
        train_rows,
    ))
    _write_csv(root / "validation_history.csv", _history_frame(
        ("epoch", "validation_parameter_loss", "validation_reconstruction_loss",
         "validation_pde_residual_loss", "validation_total_loss"),
        validation_rows,
    ))
    batch_count = math.ceil(4 / 2)
    physics_rows = [
        {
            "epoch": epoch, "split": "train", "batch_index": batch_index,
            "surface_count": 2, "collocation_point_count": 2,
            "residual_mean": 1.0, "residual_max_abs": 1.0,
            "terminal_payoff_max_abs": 0.0,
        }
        for epoch in range(1, epochs + 1)
        for batch_index in range(batch_count)
    ]
    gradient_rows = [
        {"epoch": epoch, "batch_index": batch_index, "finite_gradients": True, "gradient_norm": 1.0}
        for epoch in range(1, epochs + 1)
        for batch_index in range(batch_count)
    ]
    _write_csv(root / "physics_diagnostics.csv", _history_frame(
        ("epoch", "split", "batch_index", "surface_count", "collocation_point_count",
         "residual_mean", "residual_max_abs", "terminal_payoff_max_abs"), physics_rows,
    ))
    _write_csv(root / "gradient_diagnostics.csv", _history_frame(
        ("epoch", "batch_index", "finite_gradients", "gradient_norm"), gradient_rows,
    ))
    environment = {
        "host": f"fixture-host-{seed}", "device_selected": "cuda",
        "cuda_available": True, "torch_threads": 1,
        "deterministic_algorithms": True, "float64_physics_boundary": True,
        "real_market_inputs_used": False, "issue34_numeric_outcomes_used": False,
    }
    (root / "environment_provenance.json").write_text(
        json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = _checkpoint_payload(metadata, seed)
    torch.save(checkpoint, root / "checkpoint.pt")
    torch.save(
        {
            "completed_epoch": BEST_EPOCH,
            "optimizer_state_dict": checkpoint["optimizer_state_dict"],
            "metadata": metadata,
        },
        root / "optimizer.pt",
    )
    (root / "epoch_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = {
        path.relative_to(root).as_posix(): {"sha256": _sha(path)}
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "host": f"fixture-host-{seed}",
                "platform": "fixture-platform",
                "artifacts": artifacts,
            },
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def three_fixture_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[int, Path]:
    from src.model3_evaluation import contracts as contract_module

    small_settings = {
        "train_limit": 4, "validation_limit": 2, "epochs": 120, "batch_size": 2,
        "interior_points": 1, "terminal_points": 1, "learning_rate": 0.0002,
        "weight_decay": 0.00001, "device": "cuda", "smoke_mode": False,
        "patience": 15,
    }
    monkeypatch.setattr(contract_module, "FROZEN_SETTINGS", small_settings)
    roots = {}
    for seed in REQUIRED_SEEDS:
        root = tmp_path / f"seed-{seed}"
        _make_run(root, seed=seed)
        roots[seed] = root
    return roots


def test_checkpoint_and_three_seed_manifest_schemas(three_fixture_runs):
    manifest = build_freeze_manifest(three_fixture_runs, experiment_id="FIXTURE_ONLY")
    assert set(manifest) == FREEZE_MANIFEST_FIELDS
    for seed in REQUIRED_SEEDS:
        assert set(manifest["seed_contracts"][str(seed)]) == CHECKPOINT_CONTRACT_FIELDS
    verification = verify_freeze_manifest(manifest)
    assert verification["seeds"] == [11, 22, 33]
    assert verification["seed_epochs"] == {str(seed): BEST_EPOCH for seed in REQUIRED_SEEDS}


def test_gate_rejects_mixed_training_populations(three_fixture_runs):
    valid = build_freeze_manifest(three_fixture_runs, experiment_id="FIXTURE_ONLY")
    mixed = json.loads(json.dumps(valid))
    mixed["seed_contracts"]["22"]["train_population_sha256"] = "1" * 64
    with pytest.raises(Exception, match="cross-seed identity differs"):
        verify_freeze_manifest(mixed)


def test_gate_rejects_checkpoint_population_not_matching_dataset(three_fixture_runs):
    with pytest.raises(Exception, match="training population differs"):
        build_seed_contract(
            three_fixture_runs[11],
            seed=11,
            experiment_id="FIXTURE_ONLY",
            expected_train_population_sha256="f" * 64,
            expected_validation_population_sha256="e" * 64,
        )


def test_gate_rejects_config_identity_drift(three_fixture_runs):
    valid = build_freeze_manifest(three_fixture_runs, experiment_id="FIXTURE_ONLY")
    drifted = json.loads(json.dumps(valid))
    drifted["seed_contracts"]["33"]["config_sha256"] = "2" * 64
    with pytest.raises(Exception, match="cross-seed identity differs"):
        verify_freeze_manifest(drifted)


def test_gate_rejects_wrong_missing_duplicate_and_tampered_seeds(tmp_path, monkeypatch):
    from src.model3_evaluation import contracts as contract_module
    monkeypatch.setattr(contract_module, "FROZEN_SETTINGS", {
        "train_limit": 4, "validation_limit": 2, "epochs": 120, "batch_size": 2,
        "interior_points": 1, "terminal_points": 1, "learning_rate": 0.0002,
        "weight_decay": 0.00001, "device": "cuda", "smoke_mode": False,
        "patience": 15,
    })
    roots = {seed: tmp_path / str(seed) for seed in REQUIRED_SEEDS}
    for seed in REQUIRED_SEEDS:
        _make_run(roots[seed], seed=seed)
    valid = build_freeze_manifest(roots, experiment_id="FIXTURE_ONLY")

    wrong = json.loads(json.dumps(valid))
    wrong["seed_contracts"]["22"]["seed"] = 11
    with pytest.raises(Exception, match="wrong/duplicate"):
        verify_freeze_manifest(wrong)
    missing = json.loads(json.dumps(valid))
    missing["seed_contracts"].pop("33")
    with pytest.raises(Exception, match="missing or duplicate"):
        verify_freeze_manifest(missing)
    duplicate = json.loads(json.dumps(valid))
    duplicate["seeds"] = [11, 22, 22]
    with pytest.raises(Exception, match="exactly seeds"):
        verify_freeze_manifest(duplicate)
    checkpoint = roots[11] / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="hash_mismatch:checkpoint.pt"):
        build_seed_contract(roots[11], seed=11, experiment_id="FIXTURE_ONLY")


def test_gate_rejects_stage_a_incomplete_config_dataset_and_partial(tmp_path, monkeypatch):
    from src.model3_evaluation import contracts as contract_module
    small_settings = {
        "train_limit": 4, "validation_limit": 2, "epochs": 120, "batch_size": 2,
        "interior_points": 1, "terminal_points": 1, "learning_rate": 0.0002,
        "weight_decay": 0.00001, "device": "cuda", "smoke_mode": False,
        "patience": 15,
    }
    monkeypatch.setattr(contract_module, "FROZEN_SETTINGS", small_settings)
    roots = {seed: tmp_path / str(seed) for seed in REQUIRED_SEEDS}
    for seed in REQUIRED_SEEDS:
        _make_run(roots[seed], seed=seed)
    _make_run(roots[11].parent / "stage-a", seed=11, stage_a=True)
    with pytest.raises(Exception, match="identity mismatch"):
        from src.model3_evaluation.contracts import build_seed_contract
        build_seed_contract(
            roots[11].parent / "stage-a", seed=11, experiment_id="FIXTURE_ONLY"
        )

    partial_root = tmp_path / "partial"
    _make_run(partial_root, seed=11, epochs=BEST_EPOCH + 1)
    with pytest.raises(Exception, match="partial/interrupted"):
        from src.model3_evaluation.contracts import build_seed_contract
        build_seed_contract(partial_root, seed=11, experiment_id="FIXTURE_ONLY")

    config_changed = {seed: roots[seed] for seed in REQUIRED_SEEDS}
    valid = build_freeze_manifest(config_changed, experiment_id="FIXTURE_ONLY")
    dataset_bad = json.loads(json.dumps(valid))
    dataset_bad["seed_contracts"]["11"]["final_r2_dataset_sha256"] = "0" * 64
    with pytest.raises(Exception, match="cross-seed identity differs"):
        verify_freeze_manifest(dataset_bad)


def test_default_clean_runner_is_locked_before_dataset_access(tmp_path):
    sentinel = tmp_path / "dataset-that-must-not-be-opened.jsonl"
    output = tmp_path / "locked-output"
    with pytest.raises(FrozenTestLockedError, match="--authorize-frozen-test-evaluation"):
        run_clean_evaluation(
            freeze_manifest_path=tmp_path / "absent-manifest.json",
            checkpoint_roots={seed: tmp_path / str(seed) for seed in REQUIRED_SEEDS},
            output_root=output,
            exact_command="fixture command",
            authorize_frozen_test_evaluation=False,
            dataset_path=sentinel,
        )
    assert not output.exists()
    assert not sentinel.exists()


def test_valid_freeze_still_requires_explicit_authorization_flag(three_fixture_runs, tmp_path):
    manifest = build_freeze_manifest(three_fixture_runs, experiment_id="FIXTURE_ONLY")
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FrozenTestLockedError):
        require_frozen_test_authorization(manifest_path, authorized=False)
    verified = require_frozen_test_authorization(manifest_path, authorized=True)
    assert verified["valid"] is True


def test_adapter_preserves_order_dtype_and_rejects_stage_a(tmp_path):
    items = []
    for row_index, split in enumerate(("train", "validation")):
        target = np.linspace(0.2, 2.0, 10) * (row_index + 1)
        mask = np.array([True] * 18 + [False] * 2)
        prices = np.linspace(0.9, 1.1, 20)
        items.append(R2SurfaceItem(
            surface_id=f"dev-{split}", split=split,
            features=np.linspace(-1.0, 1.0, 100, dtype=np.float32),
            targets=target,
            mask=mask,
            dollar_prices=prices * 100.0,
            normalized_prices=np.where(mask, prices, 0.0),
            strikes=np.linspace(90.0, 110.0, 20),
            maturities=np.linspace(0.03, 0.4, 20),
            option_types=["call"] * 20,
            spot=100.0, rate=0.05, carry=0.01,
            parameter_vector_hash=f"{split}-hash",
        ))
    dataset = R2PrimaryDataset(items)
    indices = [1, 0]
    adapter = _adapter_from_random_system(tmp_path, dataset, expected_seed=22)
    first = adapter.predict_parameters(dataset, indices, seed_identity=22)
    second = adapter.predict_parameters(dataset, indices, seed_identity=22)
    assert first.shape == second.shape == (2, 10)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(Model3CheckpointError):
        adapter.predict_parameters(dataset, indices, seed_identity=11)


def _adapter_from_random_system(tmp_path: Path, dataset: R2PrimaryDataset, expected_seed: int):
    from models.parameter_transform import TargetStandardizer
    from src.model3_pde.model import Model3PDESystem

    torch.manual_seed(expected_seed)
    system = Model3PDESystem()
    standardizer = TargetStandardizer().fit(dataset.targets)
    standardized = standardizer.transform(dataset.targets)
    metadata = {
        "run_kind": "MODEL3_STAGE_B_RESEARCH_FROZEN",
        "settings": {"seed": expected_seed},
    }
    payload = {
        "completed_epoch": 7,
        "best_epoch": 7,
        "model_state_dict": system.state_dict(),
        "target_standardizer": standardizer.state_dict(),
        "metadata": metadata,
        "_fixture_standardized_shape": tuple(standardized.shape),
    }
    checkpoint = tmp_path / "development-checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return Model3EvaluationAdapter(checkpoint, expected_seed=expected_seed)


def test_adapter_aligns_predictions_to_surface_rows(tmp_path):
    def item(surface_id: str) -> R2SurfaceItem:
        target = np.linspace(0.2, 2.0, 10)
        if surface_id == "second":
            target = target + 0.01
        mask = np.array([True] * 18 + [False] * 2)
        prices = np.linspace(0.9, 1.1, 20)
        return R2SurfaceItem(
            surface_id=surface_id,
            split="validation",
            features=np.linspace(-1.0, 1.0, 100, dtype=np.float32),
            targets=target,
            mask=mask,
            dollar_prices=prices * 100.0,
            normalized_prices=np.where(mask, prices, 0.0),
            strikes=np.linspace(90.0, 110.0, 20),
            maturities=np.linspace(0.03, 0.4, 20),
            option_types=["call"] * 20,
            spot=100.0,
            rate=0.05,
            carry=0.01,
            parameter_vector_hash=surface_id,
        )

    dataset = R2PrimaryDataset([item("first"), item("second")])
    adapter = _adapter_from_random_system(tmp_path / "alignment", dataset, 11)
    forward = adapter.predict_parameters(dataset, [0, 1])
    backward = adapter.predict_parameters(dataset, [1, 0])
    np.testing.assert_allclose(forward[0], backward[1], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(forward[1], backward[0], rtol=0.0, atol=1e-15)


def test_canonical_metric_reuse_row_alignment_and_masking():
    from src.r2_primary.evaluation import constraint_validity_metrics, repricing_metrics
    from models.pinn_model import PhysicsInformedInverseCalibrator

    observed = np.array([[0.10, np.nan], [0.20, 0.30]])
    predicted_prices = np.array([[0.12, 99.0], [0.19, 0.31]])
    metrics = repricing_metrics(observed, predicted_prices)
    assert metrics["normalized_price_rmse_mean"] == pytest.approx((0.02 + 0.01) / 2)
    truth = PhysicsInformedInverseCalibrator(input_size=1)(torch.zeros(2, 1)).detach().numpy()
    validity = constraint_validity_metrics(truth)
    assert validity["constraint_validity_rate"] == 1.0
    predictions_a = {"11": np.zeros((2, 10)), "22": np.ones((2, 10))}
    headline = {"11": {"metric": 1.0}, "22": {"metric": 3.0}}
    dispersion = stability_metrics(predictions_a, headline)
    assert dispersion["headline_metric_cross_seed_std"]["metric"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_frozen_baseline_evidence_is_hash_pinned():
    baselines = load_frozen_baseline_evidence()
    assert baselines["original_manifest_byte_hash_mismatch_count"] > 0
    assert len(baselines["manifest"]["files"]) >= 10
    assert set(baselines["neural_seed_results"]["method"].unique()) == {"model1", "model2"}
    assert set(baselines["neural_seed_results"]["seed"].unique()) == {11, 22, 33}


def test_future_adapters_fail_closed_on_tuning_or_execution_leakage():
    noise_base = {
        "schema": "MODEL3_ISSUE34_NOISE_EVALUATION_REQUEST_V1",
        "three_seed_freeze_manifest_sha256": "a" * 64,
        "clean_evaluation_manifest_sha256": "b" * 64,
        "clean_evaluation_completion_state": "COMPLETE",
        "issue34_protocol_git_sha": "c" * 64,
        "issue34_cohort_manifest_sha256": "d" * 64,
        "issue34_protocol_config_sha256": FROZEN_NOISE_PROTOCOL_CONFIG_SHA256,
        "paired_cohort_identity_required": True,
        "model3_tuning_allowed": False,
        "noisy_observation_repricing_required": True,
        "clean_latent_repricing_required": True,
        "parameter_recovery_required": True,
        "degradation_curves_required": True,
    }
    assert validate_noise_request(noise_base)["executed_by_this_adapter"] is False
    bad_noise = {**noise_base, "model3_tuning_allowed": True}
    with pytest.raises(NoiseAdapterError):
        validate_noise_request(bad_noise)

    ood_base = {
        "schema": "MODEL3_OOD_EVALUATION_REQUEST_V1",
        "three_seed_freeze_manifest_sha256": "a" * 64,
        "ood_protocol_git_sha": "b" * 64,
        "ood_cohort_manifest_sha256": "c" * 64,
        "ood_protocol_config_sha256": OOD_PROTOCOL_CONFIG_SHA256,
        "ood_cohort_content_sha256": OOD_COHORT_CONTENT_SHA256,
        "cohort_generation_owned_by_model3_layer": False,
        "checkpoint_identity_required": True,
        "row_surface_alignment_required": True,
        "model3_tuning_allowed": False,
        "result_intake_schema": "MODEL3_OOD_RESULT_INTAKE_V1",
    }
    assert validate_ood_request(ood_base)["executed_by_this_adapter"] is False
    with pytest.raises(OODAdapterError):
        validate_ood_request({**ood_base, "cohort_generation_owned_by_model3_layer": True})

    g8_base = {
        "schema": "MODEL3_G8_EVALUATION_REQUEST_V1",
        "g8_protocol_git_sha": "a" * 40,
        "g8_result_intake_schema": "MODEL3_G8_RESULT_INTAKE_V1",
        "real_market_weight_updates_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "hyperparameter_tuning_allowed": False,
        "pricing_family_and_inverse_method_comparisons_separate": True,
        "frozen_checkpoint_identity_required": True,
        "model3_inclusion_condition": G8_INCLUSION_CONDITION,
    }
    assert validate_g8_request(g8_base)["weight_update_quarantine"] == "ENFORCED"
    with pytest.raises(G8AdapterError):
        validate_g8_request({**g8_base, "real_market_weight_updates_allowed": True})
    with pytest.raises(G8AdapterError, match="boolean false"):
        validate_g8_request({**g8_base, "training_allowed": 0})


def test_partial_result_export_and_overwrite_are_refused(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "evaluation_status.json").write_text('{"status":"PARTIAL"}', encoding="utf-8")
    manifest = {
        "schema": "MODEL3_CLEAN_EVALUATION_RESULT_MANIFEST_V1",
        "completion_state": "PARTIAL",
        "artifact_hashes": {},
    }
    (result / "final_evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PaperExportError, match="partial"):
        export_publication_tables(result, tmp_path / "export")
    with pytest.raises(PaperExportError, match="overwrite"):
        export_publication_tables(result, result)


def test_failed_export_creates_collision_safe_partial_marker(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    from src.model3_evaluation import paper_export as export_module

    result = tmp_path / "result"
    result.mkdir()
    (result / "evaluation_status.json").write_text(
        '{"status":"COMPLETE","research_metrics_complete":true}', encoding="utf-8"
    )
    manifest = {
        "schema": "MODEL3_CLEAN_EVALUATION_RESULT_MANIFEST_V1",
        "completion_state": "COMPLETE",
        "artifact_hashes": {},
    }
    (result / "final_evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fail_after_validation(*args, **kwargs):
        raise OSError("fixture export failure")

    monkeypatch.setattr(export_module.shutil, "copyfile", fail_after_validation)
    monkeypatch.setattr(export_module.pd, "read_csv", fail_after_validation)
    destination = tmp_path / "failed-export"
    with pytest.raises(OSError, match="fixture export failure"):
        export_publication_tables(result, destination)
    assert (destination / "EXPORT_PARTIAL_FAILED_CLOSED").read_text() == "do_not_publish\n"
