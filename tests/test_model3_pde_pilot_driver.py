from __future__ import annotations

import importlib.util
import copy
from dataclasses import asdict
import random
import numpy as np
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
        masks=torch.stack(
            [
                torch.tensor(
                    [row % 2 == slot % 2 for slot in range(20)], dtype=torch.bool
                )
                for row in range(surface_count)
            ]
        ),
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


def test_device_leaf_creation_keeps_target_coordinates_as_leaves():
    driver = load_driver()
    source = torch.tensor([80.0, 100.0], dtype=torch.float64, requires_grad=True)
    moved_without_leaf_pattern = source.to(device=torch.device("meta"))
    leaf = driver._device_leaf(source.detach().cpu(), torch.device("meta"))
    assert moved_without_leaf_pattern.is_leaf is False
    assert leaf.device.type == "meta"
    assert leaf.is_leaf is True
    assert leaf.requires_grad is True
    assert leaf.dtype == torch.float64


def test_interior_and_terminal_contracts_use_surface_slot_matrix_indexing(tmp_path):
    settings, driver = make_settings(
        tmp_path,
        train_limit=2,
        validation_limit=2,
        batch_size=2,
        interior_points=5,
        terminal_points=3,
    )
    dataset = fake_dataset(4)
    system = driver.build_system()
    standardizer = driver.fit_train_only_standardizer(dataset, [0, 2])
    original_predict_prices = system.predict_prices
    calls = []

    def record_predict_prices(state, *, strike, risk_free_rate, dividend_yield, is_call, parameters):
        calls.append(
            {
                "strike": strike.detach().cpu().clone(),
                "is_call": is_call.detach().cpu().clone(),
            }
        )
        return original_predict_prices(
            state,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            is_call=is_call,
            parameters=parameters,
        )

    system.predict_prices = record_predict_prices
    metrics = driver.evaluate_batch(
        system,
        dataset,
        [0, 1],
        standardizer,
        settings,
        epoch=7,
        optimizer=None,
    )
    assert len(calls) == 3
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in ("total_loss", "pde_residual_rms"))
    interior_sources = torch.repeat_interleave(torch.tensor([0, 1]), settings.interior_points)
    # The exact seeded slots are checked structurally against the two matrices:
    # every selected strike/type pair must exist on its originating surface.
    for call_index, point_count in ((1, settings.interior_points), (2, settings.terminal_points)):
        sources = torch.repeat_interleave(torch.tensor([0, 1]), point_count)
        for row in range(point_count * 2):
            surface = int(sources[row])
            eligible_strikes = np.asarray(dataset.items[surface].strikes)[dataset.masks[surface]]
            eligible_calls = np.asarray(dataset.items[surface].option_types)[dataset.masks[surface]]
            strike_matches = torch.isclose(
                calls[call_index]["strike"][row],
                torch.as_tensor(eligible_strikes, dtype=torch.float64),
            )
            assert bool(strike_matches.any())
            selected_type = "call" if bool(calls[call_index]["is_call"][row]) else "put"
            assert selected_type in eligible_calls.tolist()


def test_rng_states_round_trip_exactly():
    driver = load_driver()
    torch.manual_seed(11)
    random.seed(13)
    np.random.seed(17)
    states = driver.capture_rng_states()
    torch.rand(8)
    random.random()
    np.random.random()
    driver.restore_rng_states(copy.deepcopy(states))
    left = torch.rand(8, dtype=torch.float64)
    right_random = random.random()
    right_numpy = np.random.random()
    driver.restore_rng_states(states)
    assert torch.equal(left, torch.rand(8, dtype=torch.float64))
    assert right_random == random.random()
    assert right_numpy == np.random.random()


def test_history_validation_accepts_multi_batch_and_rejects_cardinality_errors():
    driver = load_driver()
    epoch_rows = [{"epoch": epoch} for epoch in range(1, 4)]
    batch_rows = [
        {"epoch": epoch, "batch_index": batch}
        for epoch in range(1, 4)
        for batch in range(3)
    ]
    driver.validate_history_consistency(
        {"train": copy.deepcopy(epoch_rows), "validation": copy.deepcopy(epoch_rows)},
        {"physics": copy.deepcopy(batch_rows), "gradient": copy.deepcopy(batch_rows)},
        next_epoch=4,
        expected_batch_counts={"physics": 3, "gradient": 3},
    )
    duplicate = copy.deepcopy(batch_rows)
    duplicate.append({"epoch": 3, "batch_index": 2})
    with pytest.raises(RuntimeError, match="cardinality"):
        driver.validate_history_consistency(
            {"train": epoch_rows, "validation": epoch_rows},
            {"physics": duplicate},
            next_epoch=4,
            expected_batch_counts={"physics": 3},
        )


def test_checkpoint_pair_rejects_epoch_and_identity_mismatch():
    driver = load_driver()
    metadata = {"identity": "fixed"}
    optimizer_state = {
        "state": {},
        "param_groups": [{"lr": 0.0002, "weight_decay": 0.00001}],
    }
    checkpoint = {
        "completed_epoch": 2,
        "metadata": metadata,
        "optimizer_state_dict": copy.deepcopy(optimizer_state),
    }
    export = {
        "completed_epoch": 2,
        "metadata": metadata,
        "optimizer_state_dict": copy.deepcopy(optimizer_state),
    }
    driver.validate_checkpoint_pair(checkpoint, export)
    stale_export = {**export, "completed_epoch": 1}
    with pytest.raises(RuntimeError, match="epoch mismatch"):
        driver.validate_checkpoint_pair(checkpoint, stale_export)
    other_identity = {**export, "metadata": {"identity": "changed"}}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        driver.validate_checkpoint_pair(checkpoint, other_identity)
    changed_export = copy.deepcopy(export)
    changed_export["optimizer_state_dict"]["param_groups"][0]["lr"] = 999.0
    with pytest.raises(RuntimeError, match="state mismatch"):
        driver.validate_checkpoint_pair(checkpoint, changed_export)


def test_real_stage_a_rejects_dirty_tree_but_smoke_records_it(monkeypatch):
    driver = load_driver()
    monkeypatch.setattr(driver, "current_git_sha", lambda: "fixed-sha")
    monkeypatch.setattr(
        driver,
        "sha256_file",
        lambda path: driver.FROZEN_DATASET_SHA256,
    )
    monkeypatch.setattr(
        driver,
        "current_git_dirty_state",
        lambda: {"git_dirty": True, "git_status_tracked": ("M scripts/example.py",)},
    )
    with pytest.raises(RuntimeError, match="clean tracked working tree"):
        driver.build_run_identity(Path("unused.jsonl"), smoke_mode=False)
    smoke_identity = driver.build_run_identity(Path("unused.jsonl"), smoke_mode=True)
    payload = smoke_identity.payload()
    assert payload["tracked_git_dirty"] is True
    assert payload["tracked_git_status"] == ["M scripts/example.py"]


def test_multi_batch_resume_restores_rng_and_matches_uninterrupted_run(monkeypatch, tmp_path):
    driver = load_driver()
    dataset = fake_dataset(4)

    def four_row_loader(dataset_path, *, train_limit, validation_limit):
        return dataset, [0, 1, 2, 3], [2, 3]

    def high_dropout_system():
        system = driver.Model3PDESystem()
        for module in system.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.5
        system.train()
        return system

    monkeypatch.setattr(
        driver,
        "build_run_identity",
        lambda dataset_path, *, smoke_mode: driver.RunIdentity(
            git_sha="test-sha",
            config_sha256="test-config",
            dataset_sha256="test-data",
            tracked_git_dirty=True,
            tracked_git_status=("M tests/example.py",),
        ),
    )
    monkeypatch.setattr(driver, "load_pilot_dataset", four_row_loader)
    monkeypatch.setattr(driver, "build_system", high_dropout_system)
    common = {
        "dataset": tmp_path / "surfaces.jsonl",
        "train_limit": 4,
        "validation_limit": 2,
        "seed": 4207,
        "batch_size": 2,
        "interior_points": 1,
        "terminal_points": 1,
        "device": "cpu",
        "smoke_mode": True,
    }
    uninterrupted = driver.run_pilot(
        driver.PilotSettings(output_root=tmp_path / "uninterrupted", epochs=2, **common)
    )
    resumed_settings = driver.PilotSettings(
        output_root=tmp_path / "resumed", epochs=2, **common
    )
    driver.run_pilot(resumed_settings, _end_epoch_override=1)
    torch.rand(128)
    np.random.random()
    random.random()
    continued_settings = driver.PilotSettings(
        output_root=tmp_path / "resumed", epochs=2, **common
    )
    resumed = driver.run_pilot(continued_settings)
    uninterrupted_state = torch.load(
        tmp_path / "uninterrupted" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_state = torch.load(
        tmp_path / "resumed" / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert uninterrupted["completed_epoch"] == resumed["completed_epoch"] == 2
    for key, value in uninterrupted_state["model_state_dict"].items():
        torch.testing.assert_close(
            resumed_state["model_state_dict"][key],
            value,
            rtol=0.0,
            atol=0.0,
            check_dtype=True,
        )
    assert resumed_state["completed_epoch"] == 2
    stale_export = torch.load(
        tmp_path / "resumed" / "optimizer.pt", map_location="cpu", weights_only=False
    )
    stale_export["completed_epoch"] = 1
    torch.save(stale_export, tmp_path / "resumed" / "optimizer.pt")
    with pytest.raises(RuntimeError, match="epoch mismatch"):
        driver.run_pilot(continued_settings)


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
