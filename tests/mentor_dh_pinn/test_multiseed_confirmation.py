from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest
import yaml

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.multiseed_confirmation import (
    EXPECTED_COHORT_DATASET_SHA256,
    EXPECTED_RUN_MATRIX,
    EXPECTED_SEEDS,
    PAIR_SUMMARY_FIELDS,
    RESULT_FIELDS,
    RUN_SUMMARY_FIELDS,
    build_multiseed_run_config,
    load_multiseed_confirmation_spec,
    run_multiseed_confirmation,
    summarize_multiseed_confirmation,
)
from src.mentor_dh_pinn.trainer import TrainingResult


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/mentor_dh_pinn/multiseed_confirmation_v1.yaml"


def _run_tuple(run) -> tuple:
    return (
        run.run_id,
        run.variant,
        run.seed,
        run.lambda_pde,
        run.lambda_boundary,
        run.lambda_terminal,
        run.lambda_data,
    )


def test_config_freezes_seed_matrix_variants_controls_and_cohort() -> None:
    spec = load_multiseed_confirmation_spec(CONFIG_PATH)
    assert spec.seeds == EXPECTED_SEEDS == (11, 22, 33)
    assert 3407 not in spec.seeds
    assert tuple(_run_tuple(run) for run in spec.runs) == EXPECTED_RUN_MATRIX
    assert {run.variant for run in spec.runs} == {"equal", "optimized"}
    assert {
        (run.lambda_pde, run.lambda_boundary, run.lambda_terminal, run.lambda_data)
        for run in spec.runs
    } == {(1.0, 1.0, 1.0, 1.0), (0.1, 0.1, 1.0, 1.0)}
    assert (spec.max_epochs, spec.patience) == (1000, 100)
    assert spec.expected_cohort_dataset_sha256 == EXPECTED_COHORT_DATASET_SHA256
    baseline = load_baseline_config(REPO_ROOT / spec.base_config_path)
    for run in spec.runs:
        config = build_multiseed_run_config(baseline, spec, run)
        assert config.seed == run.seed
        assert config.losses.weights == run.weights
        assert config.training == baseline.training


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (("seeds", None, [11, 22, 3407]), "seeds are frozen"),
        (("runs", 1, {"lambda_pde": 0.11}), "frozen matrix"),
        (("max_epochs", None, 999), "controls are frozen"),
        (("expected_cohort_dataset_sha256", None, "bad"), "dataset SHA has drifted"),
    ),
)
def test_config_fails_closed_on_drift(tmp_path: Path, mutation, message: str) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    field, index, value = mutation
    if field == "runs":
        raw[field][index].update(value)
    else:
        raw[field] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_multiseed_confirmation_spec(changed)


def _checkpoint(baseline, dataset_identity: dict) -> dict:
    return {
        "config": baseline.to_dict(),
        "weights": {"pde": 1.0, "boundary": 1.0, "terminal": 1.0, "data": 1.0},
        "best_epoch": 909,
        "best_validation_nrmse": 0.0012545129490993494,
        "dataset_identity": dataset_identity,
    }


def _write_history(path: Path, epoch: int, value: float) -> None:
    fields = (
        "epoch", "validation_price_rmse", "validation_price_mae",
        "pde_residual_rms", "terminal_rmse", "boundary_low_rmse",
        "boundary_high_rmse", "finite_gradients",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "validation_price_rmse": value + 0.001,
                "validation_price_mae": value + 0.002,
                "pde_residual_rms": value + 0.003,
                "terminal_rmse": value + 0.004,
                "boundary_low_rmse": value + 0.005,
                "boundary_high_rmse": value + 0.006,
                "finite_gradients": True,
            }
        )


def test_runner_reuses_phase1_cohort_selects_runs_and_never_touches_test(
    monkeypatch, tmp_path: Path
) -> None:
    import src.mentor_dh_pinn.multiseed_confirmation as module

    baseline = load_baseline_config()
    requested_splits: list[str] = []
    identity = {
        "dataset_sha256": EXPECTED_COHORT_DATASET_SHA256,
        "split_id_hashes": {"train": "train-sha", "validation": "validation-sha"},
    }

    class DatasetWithoutTestAccess:
        manifest = identity

        def indices(self, name: str):
            assert name != "test"
            requested_splits.append(name)
            return [0]

    dataset = DatasetWithoutTestAccess()
    monkeypatch.setattr(module, "load_baseline_config", lambda path: baseline)
    monkeypatch.setattr(module, "load_synthetic_dataset", lambda path, *, config: dataset)
    monkeypatch.setattr(module, "_load_phase1_checkpoint", lambda path: _checkpoint(baseline, identity))
    monkeypatch.setattr(module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(module, "DoubleHestonForwardPINN", lambda **kwargs: object())
    monkeypatch.setattr(module, "_git_sha", lambda root: "git-sha")
    trained: list[tuple[str, int]] = []

    def fake_train(model, supplied_dataset, output_dir, **kwargs):
        assert supplied_dataset is dataset
        assert kwargs["cohort_config"] is baseline
        supplied_dataset.indices("train")
        supplied_dataset.indices("validation")
        output = Path(output_dir)
        output.mkdir(parents=True)
        trained.append((output.name, kwargs["config"].seed))
        checkpoint_path = output / "checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        train_history = output / "train_history.csv"
        _write_history(train_history, 7, 0.01)
        validation_history = output / "validation_history.csv"
        validation_history.write_text("epoch,validation_nrmse\n7,0.01\n", encoding="utf-8")
        return TrainingResult(
            output_dir=output,
            checkpoint_path=checkpoint_path,
            best_epoch=7,
            epochs_completed=9,
            best_validation_nrmse=0.01,
            train_history_path=train_history,
            validation_history_path=validation_history,
        )

    monkeypatch.setattr(module, "train_baseline", fake_train)
    results = run_multiseed_confirmation(
        CONFIG_PATH,
        repo_root=tmp_path,
        selected_run_ids=["OPT11", "EQ11"],
    )
    assert trained == [("EQ11", 11), ("OPT11", 11)]
    assert requested_splits == ["train", "validation", "train", "validation"]
    assert [path.parent.name for path in results] == ["EQ11", "OPT11"]
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["cohort_dataset_sha256"] == EXPECTED_COHORT_DATASET_SHA256
        assert payload["train_split_id_hash"] == "train-sha"
        assert payload["validation_split_id_hash"] == "validation-sha"
        assert payload["git_sha"] == "git-sha"
        assert not (path.parent / "test_metrics.json").exists()
    with pytest.raises(FileExistsError, match="completed Phase 2D run already exists"):
        run_multiseed_confirmation(
            CONFIG_PATH,
            repo_root=tmp_path,
            selected_run_ids=["EQ11"],
        )
    with pytest.raises(ValueError, match="unknown Phase 2D run IDs"):
        run_multiseed_confirmation(
            CONFIG_PATH,
            repo_root=tmp_path,
            selected_run_ids=["EQ44"],
        )
    source = (REPO_ROOT / "src/mentor_dh_pinn/multiseed_confirmation.py").read_text(
        encoding="utf-8"
    )
    scripts = "".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "scripts/mentor_dh_pinn/run_multiseed_confirmation.py",
            REPO_ROOT / "scripts/mentor_dh_pinn/summarize_multiseed_confirmation.py",
        )
    )
    assert "evaluate_test_once" not in source + scripts
    assert "mentor_dh_pinn.evaluation" not in source + scripts


def _write_complete_result(root: Path, setting, nrmse: float, pde_rms: float) -> None:
    run_dir = root / setting.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (run_dir / "train_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    (run_dir / "validation_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    payload = {
        **setting.as_dict(),
        "best_epoch": 5,
        "epochs_completed": 8,
        "best_validation_nrmse": nrmse,
        "validation_rmse": nrmse + 0.001,
        "validation_mae": nrmse + 0.002,
        "pde_rms": pde_rms,
        "terminal_rmse": 0.004,
        "boundary_low_rmse": 0.005,
        "boundary_high_rmse": 0.006,
        "finite_gradients": True,
        "training_seconds": 10.0 + setting.seed,
        "cohort_dataset_sha256": EXPECTED_COHORT_DATASET_SHA256,
        "train_split_id_hash": "train-sha",
        "validation_split_id_hash": "validation-sha",
        "git_sha": "git-sha",
    }
    (run_dir / "multiseed_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_summaries_are_deterministic_paired_and_aggregate_correct(tmp_path: Path) -> None:
    spec = load_multiseed_confirmation_spec(CONFIG_PATH)
    output_root = tmp_path / spec.output_root
    metrics = {
        "EQ11": (0.010, 0.020), "OPT11": (0.008, 0.030),
        "EQ22": (0.020, 0.040), "OPT22": (0.018, 0.060),
        "EQ33": (0.030, 0.050), "OPT33": (0.031, 0.090),
    }
    for setting in reversed(spec.runs):
        _write_complete_result(output_root, setting, *metrics[setting.run_id])
    first_paths = summarize_multiseed_confirmation(CONFIG_PATH, repo_root=tmp_path)
    first_bytes = tuple(path.read_bytes() for path in first_paths)
    second_paths = summarize_multiseed_confirmation(CONFIG_PATH, repo_root=tmp_path)
    assert tuple(path.read_bytes() for path in second_paths) == first_bytes

    with first_paths[0].open("r", newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))
    assert tuple(run_rows[0]) == RUN_SUMMARY_FIELDS
    assert [row["run_id"] for row in run_rows] == [run.run_id for run in spec.runs]

    with first_paths[1].open("r", newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert tuple(pair_rows[0]) == PAIR_SUMMARY_FIELDS
    assert [int(row["seed"]) for row in pair_rows] == [11, 22, 33]
    assert float(pair_rows[0]["percentage_improvement"]) == pytest.approx(20.0)
    assert float(pair_rows[0]["pde_ratio_opt_vs_equal"]) == pytest.approx(1.5)
    assert [row["optimized_wins"] for row in pair_rows] == ["true", "true", "false"]
    assert all(row["physics_gate_pass"] == "true" for row in pair_rows)

    aggregate = json.loads(first_paths[2].read_text(encoding="utf-8"))
    equal_values = [0.010, 0.020, 0.030]
    optimized_values = [0.008, 0.018, 0.031]
    percentages = [20.0, 10.0, 100.0 * (0.030 - 0.031) / 0.030]
    assert aggregate["seeds"] == [11, 22, 33]
    assert aggregate["equal_mean_validation_nrmse"] == pytest.approx(statistics.mean(equal_values))
    assert aggregate["equal_std_validation_nrmse"] == pytest.approx(statistics.pstdev(equal_values))
    assert aggregate["optimized_mean_validation_nrmse"] == pytest.approx(
        statistics.mean(optimized_values)
    )
    assert aggregate["optimized_std_validation_nrmse"] == pytest.approx(
        statistics.pstdev(optimized_values)
    )
    assert aggregate["mean_paired_percentage_improvement"] == pytest.approx(
        statistics.mean(percentages)
    )
    assert aggregate["median_paired_percentage_improvement"] == pytest.approx(
        statistics.median(percentages)
    )
    assert aggregate["optimized_win_count"] == 2
    assert aggregate["physics_gate_pass_count"] == 3
    assert aggregate["optimized_confirmed"] is True
    assert aggregate["inferential_significance_test_performed"] is False


def test_summary_fails_closed_on_incomplete_result(tmp_path: Path) -> None:
    spec = load_multiseed_confirmation_spec(CONFIG_PATH)
    output_root = tmp_path / spec.output_root
    for setting in spec.runs:
        _write_complete_result(output_root, setting, 0.01, 0.02)
    (output_root / "OPT33" / "validation_history.csv").unlink()
    with pytest.raises(ValueError, match="incomplete Phase 2D result"):
        summarize_multiseed_confirmation(CONFIG_PATH, repo_root=tmp_path)
