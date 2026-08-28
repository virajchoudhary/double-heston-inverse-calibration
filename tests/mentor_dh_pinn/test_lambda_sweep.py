from __future__ import annotations

import csv
import json
from pathlib import Path

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.lambda_sweep import (
    EXPECTED_LAMBDAS,
    SUMMARY_FIELDS,
    build_run_config,
    load_lambda_sweep_spec,
    run_lambda_sweep,
    summarize_lambda_sweep,
)
from src.mentor_dh_pinn.trainer import TrainingResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_CONFIG = REPO_ROOT / "configs/mentor_dh_pinn/lambda_sweep_v1.yaml"


def test_lambda_configs_parse_with_exact_screening_settings() -> None:
    spec = load_lambda_sweep_spec(SWEEP_CONFIG)
    observed = tuple(
        (run.run_id, run.lambda_pde, run.lambda_boundary, run.lambda_terminal, run.lambda_data)
        for run in spec.runs
    )
    assert observed == EXPECTED_LAMBDAS
    assert spec.seed == 3407
    assert spec.max_epochs == 300
    assert spec.patience == 50
    baseline = load_baseline_config(REPO_ROOT / spec.base_config_path)
    for setting in spec.runs:
        run_config = build_run_config(baseline, spec, setting)
        assert run_config.seed == 3407
        assert run_config.training.max_epochs == 300
        assert run_config.training.patience == 50
        assert run_config.losses.weights == {
            "pde": setting.lambda_pde,
            "boundary": setting.lambda_boundary,
            "terminal": setting.lambda_terminal,
            "data": setting.lambda_data,
        }


def test_runner_uses_only_phase1_train_validation_cohort_and_never_evaluates_test(
    monkeypatch, tmp_path: Path
) -> None:
    import src.mentor_dh_pinn.lambda_sweep as sweep_module

    baseline = load_baseline_config()
    requested_splits: list[str] = []

    class DatasetWithoutTestAccess:
        manifest = {
            "dataset_sha256": "cohort-sha",
            "split_id_hashes": {"train": "train-sha", "validation": "validation-sha"},
        }

        def indices(self, name: str):
            assert name != "test"
            requested_splits.append(name)
            return [0]

    dataset = DatasetWithoutTestAccess()

    monkeypatch.setattr(sweep_module, "load_baseline_config", lambda path: baseline)
    monkeypatch.setattr(
        sweep_module,
        "load_synthetic_dataset",
        lambda path, *, config: dataset,
    )
    monkeypatch.setattr(sweep_module, "seed_everything", lambda seed: None)

    class FakeModel:
        pass

    monkeypatch.setattr(sweep_module, "DoubleHestonForwardPINN", lambda **kwargs: FakeModel())

    def fake_train(model, supplied_dataset, output_dir, **kwargs):
        assert supplied_dataset is dataset
        assert kwargs["cohort_config"] is baseline
        assert kwargs["config"].losses.weights == {
            "data": 1.0, "pde": 0.1, "boundary": 1.0, "terminal": 1.0
        }
        supplied_dataset.indices("train")
        supplied_dataset.indices("validation")
        output = Path(output_dir)
        output.mkdir(parents=True)
        return TrainingResult(
            output_dir=output,
            checkpoint_path=output / "checkpoint.pt",
            best_epoch=7,
            epochs_completed=9,
            best_validation_nrmse=0.25,
            train_history_path=output / "train_history.csv",
            validation_history_path=output / "validation_history.csv",
        )

    monkeypatch.setattr(sweep_module, "train_baseline", fake_train)
    results = run_lambda_sweep(
        SWEEP_CONFIG,
        repo_root=tmp_path,
        selected_run_ids=["A1"],
    )
    assert requested_splits == ["train", "validation"]
    assert results == [
        tmp_path / "outputs/mentor_dh_pinn_lambda_sweep/A1/sweep_result.json"
    ]
    assert not (results[0].parent / "test_metrics.json").exists()
    assert not (results[0].parent / "test_evaluation_claim.json").exists()
    runner_text = (REPO_ROOT / "scripts/mentor_dh_pinn/run_lambda_sweep.py").read_text(
        encoding="utf-8"
    )
    module_text = (REPO_ROOT / "src/mentor_dh_pinn/lambda_sweep.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_test_once" not in runner_text + module_text
    assert "mentor_dh_pinn.evaluation" not in runner_text + module_text


def _write_fake_run(run_dir: Path, setting, index: int) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        **setting.as_dict(),
        "best_epoch": 2,
        "best_validation_nrmse": 0.1 + index / 100.0,
        "training_seconds": 10.0 + index,
    }
    (run_dir / "sweep_result.json").write_text(
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
        writer.writerow(
            {
                "epoch": 1,
                "validation_price_rmse": 9,
                "validation_price_mae": 9,
                "validation_nrmse": 9,
                "pde_residual_rms": 9,
                "terminal_rmse": 9,
                "boundary_low_rmse": 9,
                "boundary_high_rmse": 9,
                "finite_gradients": True,
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "validation_price_rmse": 0.2 + index / 100.0,
                "validation_price_mae": 0.15 + index / 100.0,
                "validation_nrmse": 0.1 + index / 100.0,
                "pde_residual_rms": 0.3 + index / 100.0,
                "terminal_rmse": 0.4 + index / 100.0,
                "boundary_low_rmse": 0.5 + index / 100.0,
                "boundary_high_rmse": 0.6 + index / 100.0,
                "finite_gradients": True,
            }
        )


def test_summary_table_is_deterministic_and_ordered(tmp_path: Path) -> None:
    spec = load_lambda_sweep_spec(SWEEP_CONFIG)
    sweep_root = tmp_path / spec.output_root
    for index, setting in reversed(list(enumerate(spec.runs))):
        _write_fake_run(sweep_root / setting.run_id, setting, index)
    first = summarize_lambda_sweep(SWEEP_CONFIG, repo_root=tmp_path)
    first_bytes = first.read_bytes()
    second = summarize_lambda_sweep(SWEEP_CONFIG, repo_root=tmp_path)
    assert second.read_bytes() == first_bytes
    with first.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [row["run_id"] for row in rows] == [f"A{index}" for index in range(1, 10)]
    assert all(row["finite_gradients"] == "true" for row in rows)
