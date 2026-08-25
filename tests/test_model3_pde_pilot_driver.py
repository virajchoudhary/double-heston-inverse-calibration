from __future__ import annotations

import importlib.util
from dataclasses import asdict
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def load_driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_model3_pde_pilot.py"
    spec = importlib.util.spec_from_file_location("run_model3_pde_pilot_under_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError("pilot driver module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_settings(tmp_path: Path, **overrides):
    driver = load_driver()
    values = {
        "dataset": tmp_path / "surfaces.jsonl",
        "output_root": tmp_path / "run",
        **overrides,
    }
    return driver.PilotSettings(**values), driver


def fake_dataset(surface_count: int = 4):
    generator = torch.Generator().manual_seed(42)
    return SimpleNamespace(
        features=torch.randn((surface_count, 100), generator=generator),
        targets=torch.stack(
            [
                torch.tensor(
                    [
                        0.5 + row * 0.01,
                        0.05 + row * 0.001,
                        0.20 + row * 0.002,
                        -0.10 + row * 0.01,
                        0.04 + row * 0.001,
                        2.00 + row * 0.05,
                        0.03 + row * 0.0007,
                        0.30 + row * 0.003,
                        0.10 + row * 0.001,
                        0.02 + row * 0.002,
                    ],
                    dtype=torch.float64,
                )
                for row in range(surface_count)
            ]
        ),
        masks=torch.ones((surface_count, 20), dtype=torch.bool),
        items=[
            SimpleNamespace(
                surface_id=f"surface_{row}",
                split="train",
                parameter_vector_hash=f"hash_{row}",
                spot=100.0,
                rate=0.02,
                carry=0.01,
                strikes=torch.linspace(80.0, 120.0, 20).numpy(),
                maturities=torch.full((20,), 30.0 / 365.0).numpy(),
                option_types=["call"] * 10 + ["put"] * 10,
                normalized_prices=torch.linspace(0.01, 0.12, 20).numpy(),
            )
            for row in range(surface_count)
        ],
    )


def test_cli_defaults_match_frozen_stage_a():
    driver = load_driver()
    parser = driver.build_argument_parser()
    args = parser.parse_args(
        [
            "--dataset", "data/final_r2_clean_10000/surfaces.jsonl",
            "--output-root", "outputs/model3_pde_development_pilot",
        ]
    )
    assert args.train_limit == 240
    assert args.validation_limit == 40
    assert args.seed == 4207
    assert args.epochs == 3
    assert args.batch_size == 16
    assert args.interior_points == 16
    assert args.terminal_points == 8
    assert args.learning_rate == pytest.approx(0.0002)
    assert args.weight_decay == pytest.approx(0.00001)
    assert args.device == "cpu"
    assert not args.smoke


def test_settings_reject_invalid_execution_values(tmp_path):
    with pytest.raises(ValueError, match="strictly positive"):
        make_settings(tmp_path, epochs=0)


def test_loader_rejects_test_split(monkeypatch):
    driver = load_driver()
    forbidden = SimpleNamespace(indices_for_split=lambda split: [0], items=[])
    monkeypatch.setattr(
        driver.R2PrimaryDataset,
        "from_jsonl",
        classmethod(lambda cls, path, splits=None: forbidden),
    )
    with pytest.raises(RuntimeError, match="test-split"):
        driver.load_pilot_dataset(Path("unused.jsonl"), train_limit=1, validation_limit=1)


def test_deterministic_subsets_are_first_in_stored_split_order(monkeypatch):
    driver = load_driver()
    dataset = fake_dataset(6)
    splits = ["train", "train", "train", "validation", "validation", "validation"]
    for index, split in enumerate(splits):
        dataset.items[index].split = split
    dataset.indices_for_split = lambda split: [
        index for index, value in enumerate(splits) if value == split
    ]
    monkeypatch.setattr(
        driver.R2PrimaryDataset,
        "from_jsonl",
        classmethod(lambda cls, path, splits=None: dataset),
    )
    loaded, train_indices, validation_indices = driver.load_pilot_dataset(
        Path("unused.jsonl"), train_limit=2, validation_limit=2
    )
    assert loaded is dataset
    assert train_indices == [0, 1]
    assert validation_indices == [3, 4]


def test_target_scaling_uses_only_requested_training_rows():
    driver = load_driver()
    dataset = fake_dataset(6)
    train_indices = [0, 2]
    standardizer = driver.fit_train_only_standardizer(dataset, train_indices)
    expected_mean = dataset.targets[train_indices].mean(dim=0)
    expected_scale = dataset.targets[train_indices].std(dim=0, unbiased=False)
    assert torch.equal(standardizer.mean, expected_mean)
    assert torch.equal(standardizer.scale, expected_scale)


def test_system_and_optimizer_have_frozen_shapes_and_float64():
    driver = load_driver()
    system = driver.build_system()
    optimizer = driver.build_optimizer(system, learning_rate=0.0002, weight_decay=0.00001)
    assert next(system.parameters()).dtype == torch.float64
    assert all(group["lr"] == pytest.approx(0.0002) for group in optimizer.param_groups)
    assert all(
        group["weight_decay"] == pytest.approx(0.00001) for group in optimizer.param_groups
    )
    assert type(optimizer).__name__ == "AdamW"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loss_component_wiring_and_optimizer_step_are_finite(tmp_path):
    settings, driver = make_settings(
        tmp_path,
        train_limit=2,
        validation_limit=2,
        batch_size=2,
        interior_points=1,
        terminal_points=1,
    )
    dataset = fake_dataset(4)
    system = driver.build_system()
    standardizer = driver.fit_train_only_standardizer(dataset, [0, 2])
    metrics = driver.evaluate_batch(
        system,
        dataset,
        [0, 2],
        standardizer,
        settings,
        epoch=1,
        optimizer=None,
    )
    expected_total = (
        metrics["parameter_loss"]
        + metrics["reconstruction_loss"]
        + 0.10 * metrics["pde_residual_loss"]
    )
    assert metrics["total_loss"] == pytest.approx(expected_total, rel=1e-12)
    assert all(
        torch.isfinite(torch.tensor(value))
        for key, value in metrics.items()
        if key.endswith("loss")
    )
    assert torch.isfinite(torch.tensor(metrics["pde_residual_rms"]))
    assert torch.isfinite(torch.tensor(metrics["pde_residual_max_scaled_rms"]))
    assert metrics["reconstruction_loss"] > 0.0
    # The fixture stores spot-normalized targets; a dollar-price comparison would
    # inflate this term by roughly 10,000 for spot 100.
    assert metrics["reconstruction_loss"] < 1.0

    optimizer = driver.build_optimizer(system, learning_rate=0.0002, weight_decay=0.00001)
    stepped = driver.evaluate_batch(
        system,
        dataset,
        [0, 2],
        standardizer,
        settings,
        epoch=2,
        optimizer=optimizer,
    )
    assert stepped["finite_gradients"] is True
    assert stepped["gradient_norm"] > 0.0


def test_checkpoint_identity_rejects_mismatch():
    driver = load_driver()
    stored = {"metadata": {"run_kind": "old"}}
    with pytest.raises(RuntimeError, match="resume identity mismatch"):
        driver.validate_resume_identity(stored["metadata"], {"run_kind": "new"})


def test_required_output_schema_is_declared():
    driver = load_driver()
    assert len(driver.REQUIRED_ARTIFACTS) == 8
    assert set(driver.REQUIRED_ARTIFACTS) == {
        "checkpoint.pt",
        "optimizer.pt",
        "epoch_metadata.json",
        "train_history.csv",
        "validation_history.csv",
        "physics_diagnostics.csv",
        "gradient_diagnostics.csv",
        "environment_provenance.json",
    }
    assert set(driver.LOSS_WEIGHTS) == {
        "parameter", "reconstruction", "pde_residual", "terminal_diagnostic", "boundary_penalty"
    }
    assert driver.LOSS_WEIGHTS["parameter"] == 1.0
    assert driver.LOSS_WEIGHTS["reconstruction"] == 1.0
    assert driver.LOSS_WEIGHTS["pde_residual"] == 0.10
    assert driver.LOSS_WEIGHTS["terminal_diagnostic"] == 0.0
    assert driver.LOSS_WEIGHTS["boundary_penalty"] == 0.0


def test_smoke_mode_has_one_batch_at_most_one_step():
    driver = load_driver()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        settings = driver.smoke_settings(root / "surfaces.jsonl", root / "output")
        assert settings.train_limit == 2
        assert settings.validation_limit == 2
        assert settings.epochs == 1
        assert settings.batch_size == 2
        assert settings.interior_points == 1
        assert settings.terminal_points == 1
        assert settings.smoke_mode is True


def test_driver_has_no_real_market_or_issue34_dependency():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_model3_pde_pilot.py").read_text(encoding="utf-8")
    assert "issue34_numeric_outcomes_used\": False".replace('\\"', '"') in source
    assert 'real_market_inputs_used": False' in source
    assert "evidence/r2_noise_robustness" not in source
    assert "positive_noise" not in source.lower()
    assert "test-noise" not in source.lower()
    assert driver_allowed_splits_source(source)


def driver_allowed_splits_source(source: str) -> bool:
    return 'ALLOWED_SPLITS = frozenset({"train", "validation"})' in source and 'FORBIDDEN_SPLIT = "test"' in source
