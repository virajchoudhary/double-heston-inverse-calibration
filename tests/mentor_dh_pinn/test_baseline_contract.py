from __future__ import annotations

from pathlib import Path
import copy
import csv
import os

import pytest
import torch
import yaml

from src.mentor_dh_pinn.config import baseline_config_from_mapping, load_baseline_config
from src.mentor_dh_pinn.evaluation import evaluate_test_once
from src.mentor_dh_pinn.model import DoubleHestonForwardPINN
from src.mentor_dh_pinn.synthetic_data import generate_synthetic_dataset, load_synthetic_dataset
from src.mentor_dh_pinn.trainer import seed_everything, train_baseline, validate_checkpoint_identities


def test_architecture_and_one_epoch_checkpoint_contract(tmp_path: Path) -> None:
    config = load_baseline_config().with_overrides(
        train_count=4, validation_count=3, test_count=2, max_epochs=1, patience=1
    )
    dataset = generate_synthetic_dataset(tmp_path, config=config)
    model = DoubleHestonForwardPINN(
        feature_min=config.domain.feature_min,
        feature_max=config.domain.feature_max,
    )
    assert model.parameter_count == 67201
    result = train_baseline(model, dataset, tmp_path, config=config)
    assert result.best_epoch == 1
    assert result.checkpoint_path.exists()
    assert result.train_history_path.exists()
    assert result.validation_history_path.exists()
    assert result.checkpoint_path.name == "checkpoint.pt"
    with result.train_history_path.open(newline="", encoding="utf-8") as handle:
        fields = set(next(csv.DictReader(handle)))
    assert {
        "epoch", "train_total_loss", "train_pde_loss", "train_boundary_loss",
        "train_boundary_low_loss", "train_boundary_high_loss", "train_terminal_loss",
        "train_data_loss", "validation_price_rmse", "validation_price_mae",
        "validation_nrmse", "pde_residual_rms", "terminal_rmse",
        "boundary_low_rmse", "boundary_high_rmse", "gradient_norm",
        "finite_gradients", "duration_seconds",
    } <= fields
    reloaded = load_synthetic_dataset(tmp_path, config=config)
    evaluation = evaluate_test_once(result.checkpoint_path, reloaded, tmp_path, config=config)
    assert evaluation.metrics["test_evaluated_once"] is True
    assert evaluation.metrics["test_count"] == 2
    assert evaluation.summary_path.exists()
    assert (tmp_path / "test_evaluation_claim.json").exists()
    original_metrics = evaluation.metrics_path.read_bytes()
    with pytest.raises(FileExistsError, match="repeated test evaluation is disabled"):
        evaluate_test_once(result.checkpoint_path, reloaded, tmp_path, config=config)
    assert evaluation.metrics_path.read_bytes() == original_metrics

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    changed = config.with_overrides(train_count=5)
    with pytest.raises(ValueError, match="dataset/config split counts"):
        validate_checkpoint_identities(checkpoint, dataset, changed)
    tampered = copy.deepcopy(checkpoint)
    tampered["dataset_identity"]["parameter_hash"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint/dataset"):
        validate_checkpoint_identities(tampered, dataset, config)


def test_yaml_contains_frozen_scientific_contract() -> None:
    path = Path(__file__).resolve().parents[2] / "configs/mentor_dh_pinn/baseline_v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["experiment_id"] == "mentor_dh_pinn_baseline_v1"
    assert raw["seed"] == 3407
    assert raw["option_type"] == "call"
    assert raw["pricing_node_count"] == 64
    assert raw["data"]["train_count"] == 4096
    assert raw["data"]["validation_count"] == 1024
    assert raw["data"]["test_count"] == 1024
    assert raw["network"]["input_size"] == 7
    assert raw["network"]["hidden_layers"] == 5
    assert raw["network"]["hidden_width"] == 128
    assert raw["losses"]["data_lambda"] == 1.0
    assert raw["losses"]["pde_lambda"] == 1.0
    assert raw["losses"]["boundary_lambda"] == 1.0
    assert raw["losses"]["terminal_lambda"] == 1.0
    assert "test_isolation_declaration" in raw["scientific_contract"]
    assert raw["evaluation"]["slice_maturity_days"] == [30, 90, 180]
    assert raw["evaluation"]["marker_maturity_days"] == [7, 30, 60, 90, 120, 180]


def test_yaml_schema_and_scientific_contract_fail_closed() -> None:
    raw = load_baseline_config().to_dict()
    typo = copy.deepcopy(raw)
    typo["training"]["learning_ratte"] = typo["training"].pop("learning_rate")
    with pytest.raises(ValueError, match="unknown=.*learning_ratte.*missing=.*learning_rate"):
        baseline_config_from_mapping(typo)
    missing = copy.deepcopy(raw)
    del missing["evaluation"]["slice_rate"]
    with pytest.raises(ValueError, match="missing=.*slice_rate"):
        baseline_config_from_mapping(missing)
    changed = copy.deepcopy(raw)
    changed["scientific_contract"]["test_policy"] = "evaluate repeatedly"
    with pytest.raises(ValueError, match="unexpected scientific contract"):
        baseline_config_from_mapping(changed)


def test_cuda_runtime_contract_is_explicit_and_deterministic() -> None:
    seed_everything(3407)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    repo_root = Path(__file__).resolve().parents[2]
    notebook = (repo_root / "notebooks/mentor_dh_pinn/01_Double_Heston_PINN_Baseline_Kaggle.ipynb").read_text(encoding="utf-8")
    for token in (
        "torch==2.7.1", "cu126", "torch.version.cuda == '12.6'", "sm_60",
        "CUBLAS_WORKSPACE_CONFIG", "'--device', 'cuda'",
    ):
        assert token in notebook
    evaluator = (repo_root / "scripts/mentor_dh_pinn/evaluate_baseline.py").read_text(encoding="utf-8")
    assert "--allow-repeat" not in evaluator
