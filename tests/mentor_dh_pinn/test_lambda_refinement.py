from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.lambda_refinement import (
    EXPECTED_PHASE2A_WINNER,
    EXPECTED_REFINEMENT_LAMBDAS,
    load_lambda_refinement_spec,
    run_lambda_refinement,
    summarize_lambda_refinement,
)
from src.mentor_dh_pinn.lambda_sweep import SUMMARY_FIELDS, build_run_config
from src.mentor_dh_pinn.trainer import TrainingResult


REPO_ROOT = Path(__file__).resolve().parents[2]
REFINEMENT_CONFIG = REPO_ROOT / "configs/mentor_dh_pinn/lambda_refinement_v1.yaml"


def _setting_tuple(setting) -> tuple[str, float, float, float, float]:
    return (
        setting.run_id,
        setting.lambda_pde,
        setting.lambda_boundary,
        setting.lambda_terminal,
        setting.lambda_data,
    )


def test_refinement_config_parses_exact_matrix_and_a9_anchor() -> None:
    spec = load_lambda_refinement_spec(REFINEMENT_CONFIG)
    assert _setting_tuple(spec.phase2a_winner) == EXPECTED_PHASE2A_WINNER
    assert tuple(_setting_tuple(run) for run in spec.runs) == EXPECTED_REFINEMENT_LAMBDAS
    assert _setting_tuple(spec.runs[0])[1:] == EXPECTED_PHASE2A_WINNER[1:]
    assert (spec.seed, spec.max_epochs, spec.patience) == (3407, 300, 50)
    baseline = load_baseline_config(REPO_ROOT / spec.base_config_path)
    for setting in spec.runs:
        config = build_run_config(baseline, spec, setting)
        assert config.training.max_epochs == 300
        assert config.training.patience == 50
        assert config.losses.weights == {
            "pde": setting.lambda_pde,
            "boundary": setting.lambda_boundary,
            "terminal": 1.0,
            "data": 1.0,
        }


def test_refinement_config_fails_closed_on_matrix_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(REFINEMENT_CONFIG.read_text(encoding="utf-8"))
    raw["runs"][1]["lambda_pde"] = 0.051
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen B0-B8 matrix"):
        load_lambda_refinement_spec(changed)


def test_runner_reuses_cohort_selects_runs_and_never_touches_test(
    monkeypatch, tmp_path: Path
) -> None:
    import src.mentor_dh_pinn.lambda_refinement as refinement_module

    baseline = load_baseline_config()
    requested_splits: list[str] = []

    class DatasetWithoutTestAccess:
        manifest = {
            "dataset_sha256": "phase1-cohort-sha",
            "split_id_hashes": {
                "train": "phase1-train-sha",
                "validation": "phase1-validation-sha",
            },
        }

        def indices(self, name: str):
            assert name != "test"
            requested_splits.append(name)
            return [0]

    dataset = DatasetWithoutTestAccess()
    monkeypatch.setattr(refinement_module, "load_baseline_config", lambda path: baseline)
    monkeypatch.setattr(
        refinement_module,
        "load_synthetic_dataset",
        lambda path, *, config: dataset,
    )
    monkeypatch.setattr(refinement_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(
        refinement_module,
        "DoubleHestonForwardPINN",
        lambda **kwargs: object(),
    )

    trained_ids: list[str] = []

    def fake_train(model, supplied_dataset, output_dir, **kwargs):
        assert supplied_dataset is dataset
        assert kwargs["cohort_config"] is baseline
        supplied_dataset.indices("train")
        supplied_dataset.indices("validation")
        output = Path(output_dir)
        trained_ids.append(output.name)
        output.mkdir(parents=True)
        return TrainingResult(
            output_dir=output,
            checkpoint_path=output / "checkpoint.pt",
            best_epoch=5,
            epochs_completed=8,
            best_validation_nrmse=0.2,
            train_history_path=output / "train_history.csv",
            validation_history_path=output / "validation_history.csv",
        )

    monkeypatch.setattr(refinement_module, "train_baseline", fake_train)
    results = run_lambda_refinement(
        REFINEMENT_CONFIG,
        repo_root=tmp_path,
        selected_run_ids=["B3", "B1"],
    )
    assert trained_ids == ["B1", "B3"]
    assert requested_splits == ["train", "validation", "train", "validation"]
    assert [path.parent.name for path in results] == ["B1", "B3"]
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["cohort_dataset_sha256"] == "phase1-cohort-sha"
        assert payload["train_split_id_hash"] == "phase1-train-sha"
        assert payload["validation_split_id_hash"] == "phase1-validation-sha"
        assert not (path.parent / "test_metrics.json").exists()
        assert not (path.parent / "test_evaluation_claim.json").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        run_lambda_refinement(
            REFINEMENT_CONFIG,
            repo_root=tmp_path,
            selected_run_ids=["B1"],
        )
    with pytest.raises(ValueError, match="unknown refinement run IDs"):
        run_lambda_refinement(
            REFINEMENT_CONFIG,
            repo_root=tmp_path,
            selected_run_ids=["B9"],
        )
    source = (REPO_ROOT / "src/mentor_dh_pinn/lambda_refinement.py").read_text(
        encoding="utf-8"
    )
    runner = (REPO_ROOT / "scripts/mentor_dh_pinn/run_lambda_refinement.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_test_once" not in source + runner
    assert "mentor_dh_pinn.evaluation" not in source + runner


def _write_fake_run(run_dir: Path, setting, index: int) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        **setting.as_dict(),
        "best_epoch": 2,
        "best_validation_nrmse": 0.2 + index / 100.0,
        "training_seconds": 20.0 + index,
    }
    (run_dir / "refinement_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "epoch", "validation_price_rmse", "validation_price_mae",
        "validation_nrmse", "pde_residual_rms", "terminal_rmse",
        "boundary_low_rmse", "boundary_high_rmse", "finite_gradients",
    )
    with (run_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for epoch in (1, 2):
            value = 9.0 if epoch == 1 else 0.1 + index / 100.0
            writer.writerow(
                {
                    "epoch": epoch,
                    "validation_price_rmse": value,
                    "validation_price_mae": value + 0.01,
                    "validation_nrmse": value + 0.02,
                    "pde_residual_rms": value + 0.03,
                    "terminal_rmse": value + 0.04,
                    "boundary_low_rmse": value + 0.05,
                    "boundary_high_rmse": value + 0.06,
                    "finite_gradients": True,
                }
            )


def test_refinement_summary_is_deterministic_and_b_ordered(tmp_path: Path) -> None:
    spec = load_lambda_refinement_spec(REFINEMENT_CONFIG)
    root = tmp_path / spec.output_root
    for index, setting in reversed(list(enumerate(spec.runs))):
        _write_fake_run(root / setting.run_id, setting, index)
    first = summarize_lambda_refinement(REFINEMENT_CONFIG, repo_root=tmp_path)
    first_bytes = first.read_bytes()
    second = summarize_lambda_refinement(REFINEMENT_CONFIG, repo_root=tmp_path)
    assert second.read_bytes() == first_bytes
    with first.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [row["run_id"] for row in rows] == [f"B{index}" for index in range(9)]
    assert all(row["finite_gradients"] == "true" for row in rows)
