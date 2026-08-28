from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.full_horizon_confirmation import (
    COMPARISON_FIELDS,
    EXPECTED_CANDIDATE,
    EXPECTED_PHASE1_REFERENCE,
    load_full_horizon_confirmation_spec,
    run_full_horizon_confirmation,
    summarize_full_horizon_confirmation,
)
from src.mentor_dh_pinn.trainer import TrainingResult


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION_CONFIG = (
    REPO_ROOT / "configs/mentor_dh_pinn/full_horizon_confirmation_v1.yaml"
)


def _candidate_tuple(setting) -> tuple[str, float, float, float, float]:
    return (
        setting.run_id,
        setting.lambda_pde,
        setting.lambda_boundary,
        setting.lambda_terminal,
        setting.lambda_data,
    )


def test_confirmation_config_parses_exact_candidate_controls_and_reference() -> None:
    spec = load_full_horizon_confirmation_spec(CONFIRMATION_CONFIG)
    assert _candidate_tuple(spec.candidate) == EXPECTED_CANDIDATE
    reference = spec.phase1_reference
    assert (
        reference.run_id,
        reference.lambda_pde,
        reference.lambda_boundary,
        reference.lambda_terminal,
        reference.lambda_data,
        reference.best_epoch,
        reference.best_validation_nrmse,
    ) == EXPECTED_PHASE1_REFERENCE
    assert (spec.seed, spec.max_epochs, spec.patience) == (3407, 1000, 100)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("candidate", "lambda_pde", 0.11, "frozen C1 setting"),
        (None, "max_epochs", 999, "controls are frozen"),
        ("phase1_reference", "best_epoch", 908, "reference has drifted"),
    ),
)
def test_confirmation_config_fails_closed_on_protocol_drift(
    tmp_path: Path, section: str | None, field: str, value, message: str
) -> None:
    raw = yaml.safe_load(CONFIRMATION_CONFIG.read_text(encoding="utf-8"))
    target = raw if section is None else raw[section]
    target[field] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_full_horizon_confirmation_spec(changed)


def test_runner_reuses_phase1_cohort_never_touches_test_and_refuses_completed_run(
    monkeypatch, tmp_path: Path
) -> None:
    import src.mentor_dh_pinn.full_horizon_confirmation as module

    baseline = load_baseline_config()
    requested_splits: list[str] = []
    identity = {
        "dataset_sha256": "phase1-dataset-sha",
        "split_id_hashes": {
            "train": "phase1-train-sha",
            "validation": "phase1-validation-sha",
        },
    }

    class DatasetWithoutTestAccess:
        manifest = identity

        def indices(self, name: str):
            assert name != "test"
            requested_splits.append(name)
            return [0]

    dataset = DatasetWithoutTestAccess()
    checkpoint = {
        "config": baseline.to_dict(),
        "weights": {"pde": 1.0, "boundary": 1.0, "terminal": 1.0, "data": 1.0},
        "best_epoch": 909,
        "best_validation_nrmse": 0.0012545129490993494,
        "dataset_identity": identity,
    }
    monkeypatch.setattr(module, "load_baseline_config", lambda path: baseline)
    monkeypatch.setattr(module, "load_synthetic_dataset", lambda path, *, config: dataset)
    monkeypatch.setattr(module, "_load_phase1_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(module, "DoubleHestonForwardPINN", lambda **kwargs: object())
    monkeypatch.setattr(module, "_git_sha", lambda root: "git-sha")

    def fake_train(model, supplied_dataset, output_dir, **kwargs):
        assert supplied_dataset is dataset
        assert kwargs["cohort_config"] is baseline
        assert kwargs["config"].losses.weights == {
            "pde": 0.1, "boundary": 0.1, "terminal": 1.0, "data": 1.0
        }
        assert kwargs["config"].training.max_epochs == 1000
        assert kwargs["config"].training.patience == 100
        supplied_dataset.indices("train")
        supplied_dataset.indices("validation")
        output = Path(output_dir)
        output.mkdir(parents=True)
        checkpoint_path = output / "checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        train_history = output / "train_history.csv"
        fields = (
            "epoch", "validation_price_rmse", "validation_price_mae",
            "pde_residual_rms", "terminal_rmse", "boundary_low_rmse",
            "boundary_high_rmse", "finite_gradients",
        )
        with train_history.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": 12,
                    "validation_price_rmse": 0.002,
                    "validation_price_mae": 0.0015,
                    "pde_residual_rms": 0.003,
                    "terminal_rmse": 0.004,
                    "boundary_low_rmse": 0.005,
                    "boundary_high_rmse": 0.006,
                    "finite_gradients": True,
                }
            )
        validation_history = output / "validation_history.csv"
        validation_history.write_text("epoch,validation_nrmse\n12,0.001\n", encoding="utf-8")
        return TrainingResult(
            output_dir=output,
            checkpoint_path=checkpoint_path,
            best_epoch=12,
            epochs_completed=20,
            best_validation_nrmse=0.001,
            train_history_path=train_history,
            validation_history_path=validation_history,
        )

    monkeypatch.setattr(module, "train_baseline", fake_train)
    result_path = run_full_horizon_confirmation(CONFIRMATION_CONFIG, repo_root=tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert requested_splits == ["train", "validation"]
    assert payload["run_id"] == "C1"
    assert payload["epochs_completed"] == 20
    assert payload["cohort_dataset_sha256"] == "phase1-dataset-sha"
    assert payload["train_split_id_hash"] == "phase1-train-sha"
    assert payload["validation_split_id_hash"] == "phase1-validation-sha"
    assert payload["git_sha"] == "git-sha"
    assert payload["finite_gradients"] is True
    assert not (result_path.parent / "test_metrics.json").exists()
    with pytest.raises(FileExistsError, match="completed Phase 2C run already exists"):
        run_full_horizon_confirmation(CONFIRMATION_CONFIG, repo_root=tmp_path)
    source = (REPO_ROOT / "src/mentor_dh_pinn/full_horizon_confirmation.py").read_text(
        encoding="utf-8"
    )
    runner = (
        REPO_ROOT / "scripts/mentor_dh_pinn/run_full_horizon_confirmation.py"
    ).read_text(encoding="utf-8")
    assert "evaluate_test_once" not in source + runner
    assert "mentor_dh_pinn.evaluation" not in source + runner

    checkpoint["dataset_identity"] = {
        **identity,
        "dataset_sha256": "changed-dataset-sha",
    }
    result_path.unlink()
    with pytest.raises(ValueError, match="cohort dataset or train/validation split identity"):
        run_full_horizon_confirmation(CONFIRMATION_CONFIG, repo_root=tmp_path)


def test_comparison_summary_is_deterministic(tmp_path: Path) -> None:
    spec = load_full_horizon_confirmation_spec(CONFIRMATION_CONFIG)
    run_dir = tmp_path / spec.output_root / "C1"
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": "C1",
        "lambda_pde": 0.1,
        "lambda_boundary": 0.1,
        "lambda_terminal": 1.0,
        "lambda_data": 1.0,
        "best_validation_nrmse": 0.001,
        "phase1_best_epoch": 909,
        "phase1_best_validation_nrmse": 0.0012545129490993494,
        "validation_rmse": 0.002,
        "validation_mae": 0.0015,
        "pde_rms": 0.003,
        "terminal_rmse": 0.004,
        "boundary_low_rmse": 0.005,
        "boundary_high_rmse": 0.006,
    }
    (run_dir / "confirmation_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first = summarize_full_horizon_confirmation(CONFIRMATION_CONFIG, repo_root=tmp_path)
    first_bytes = first.read_bytes()
    second = summarize_full_horizon_confirmation(CONFIRMATION_CONFIG, repo_root=tmp_path)
    assert second.read_bytes() == first_bytes
    with first.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == COMPARISON_FIELDS
    assert rows[0]["run_id"] == "C1"
    assert rows[0]["comparison"] == "improvement"
    assert float(rows[0]["absolute_difference"]) == pytest.approx(
        0.0012545129490993494 - 0.001
    )
    assert float(rows[0]["percentage_improvement"]) > 0
