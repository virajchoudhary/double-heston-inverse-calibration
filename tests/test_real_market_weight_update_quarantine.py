"""Regression tests for the Archive-2 real-market weight-update quarantine (Issue #20).

Canonical policy: primary neural weight learning is SYNTHETIC-ONLY; real market
observations are reserved for frozen-model evaluation. These tests prove that
normal execution of the Archive-2 entrypoint (``train_double_heston.py``) can no
longer reach a real-market ``train_stage`` call, an optimizer, or any weight
mutation without the explicit ``--allow-noncanonical-real-weight-updates``
opt-in, and that the canonical stack is untouched by the quarantine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train_double_heston as trainer
from dheston.pricing.heston import FourierConfig
from dheston.real_market_policy import (
    ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG,
    RealMarketWeightUpdateQuarantineError,
    resolve_real_market_epochs,
)


LIMIT = trainer.CONTINUOUS_EPOCH_LIMIT

CANONICAL_TRAINING_FILES = [
    "src/train.py",
    "src/train_pinn.py",
    "src/run_smoke_test.py",
    "src/run_pinn_improved_benchmark.py",
    "src/run_pinn_synthetic_baseline.py",
    "src/run_pinn_two_stage_baseline.py",
    "models/ann_model.py",
    "models/pinn_model.py",
    "models/parameter_transform.py",
]


def _resolve(config_real_epochs: int, continuous: bool, allow: bool) -> int:
    return resolve_real_market_epochs(
        config_real_epochs=config_real_epochs,
        continuous_requested=continuous,
        allow_noncanonical_real_weight_updates=allow,
        continuous_epoch_limit=LIMIT,
    )


def _synthetic_only_config(real_epochs: int = 0) -> dict:
    loss_weights = {
        "lambda_param": 1.0,
        "lambda_price": 0.25,
        "lambda_pde": 0.0,
        "lambda_order": 0.0,
        "lambda_boundary": 0.05,
    }
    return {
        "dataset_path": "unused.csv",
        "output_root": "experiments",
        "random_seed": 7,
        "market_inputs": {"risk_free_rate": 0.06, "dividend_yield": 0.0},
        "filters": {
            "symbols": [],
            "min_surface_points": 5,
            "max_surface_points": 32,
            "model_ready_only": True,
            "surface_cap_per_split": {},
        },
        "split": {"train_fraction": 0.7, "validation_fraction": 0.15},
        "pricing": {},
        "synthetic": {},
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "real_epochs": real_epochs,
            "learning_rate": 0.001,
            "hidden_dim": 8,
            "dropout": 0.0,
            "pde_points_per_batch": 0,
        },
        "losses": {
            "ordinary": dict(loss_weights),
            "physics": dict(loss_weights),
            "real_finetune": {
                "lambda_param": 0.0,
                "lambda_price": 1.0,
                "lambda_pde": 0.0,
                "lambda_order": 0.0,
                "lambda_boundary": 0.1,
            },
        },
    }


class _FakeLoader:
    def __init__(self, batch: dict) -> None:
        self.batch = batch

    def __iter__(self):
        yield self.batch


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, allow_real_stage: bool = False) -> dict:
    """Monkeypatch the actual training/update boundary so any guarded-path slip fails fast."""
    calls: dict = {"stages": [], "dataloaders_built": False, "loaders": {}, "completed_run_saved": False}
    batch = {"features": torch.zeros(2, 11), "mask": torch.ones(2, 4, dtype=torch.bool)}

    def fake_load_config(path=None):
        return _synthetic_only_config()

    def fake_build_dataloaders(config):
        calls["dataloaders_built"] = True
        calls["loaders"] = {
            name: _FakeLoader(batch)
            for name in ("train_synth", "valid_synth", "test_synth", "train_real", "valid_real", "test_real")
        }
        return calls["loaders"], {"real_train_surfaces": 0}, FourierConfig()

    def fake_train_stage(model, train_loader, valid_loader, optimizer, device, pricing_config, loss_weights, **kwargs):
        stage_name = kwargs["stage_name"]
        if stage_name == "real_finetune" and not allow_real_stage:
            raise AssertionError(
                "GUARD FAILURE: real_finetune train_stage reached without the explicit "
                "--allow-noncanonical-real-weight-updates opt-in"
            )
        calls["stages"].append((stage_name, train_loader))
        stage_log = trainer.serialize_stage_log([], float("inf"), "completed", kwargs["epochs"] + 1, kwargs["epochs"])
        return stage_log, trainer.model_state_to_cpu(model)

    def fake_evaluate_loader(model, loader, device, pricing_config):
        return {"price_mae_normalized": 0.0}

    def fake_save_completed_run(*args, **kwargs):
        calls["completed_run_saved"] = True

    monkeypatch.setattr(trainer, "load_config", fake_load_config)
    monkeypatch.setattr(trainer, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(trainer, "train_stage", fake_train_stage)
    monkeypatch.setattr(trainer, "evaluate_loader", fake_evaluate_loader)
    monkeypatch.setattr(trainer, "save_completed_run", fake_save_completed_run)
    return calls


def _run_main(monkeypatch, tmp_path: Path, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["train_double_heston.py", "--run-dir", str(tmp_path / "run"), *argv])
    trainer.main()


def _write_resume_run(tmp_path: Path, config: dict, *, status: str, stage_name: str) -> Path:
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    (run_dir / trainer.CONFIG_SNAPSHOT_FILENAME).write_text(json.dumps(config), encoding="utf-8")
    model = trainer.DeepSurfaceInverseModel(
        input_dim=11,
        hidden_dim=int(config["training"]["hidden_dim"]),
        dropout=float(config["training"]["dropout"]),
    )
    torch.save(
        {
            "mode": "ordinary",
            "stage_name": stage_name,
            "next_epoch": 1,
            "total_epochs": 1,
            "status": status,
            "model_state_dict": trainer.model_state_to_cpu(model),
            "optimizer_state_dict": None,
            "best_model_state_dict": None,
            "best_validation_total": float("inf"),
            "training_log": [],
            "stage_logs": {},
        },
        run_dir / trainer.CHECKPOINT_FILENAME,
    )
    return run_dir


# ---------------------------------------------------------------------------
# Policy resolver matrix
# ---------------------------------------------------------------------------


def test_synthetic_only_resolution_allowed_by_default() -> None:
    assert _resolve(config_real_epochs=0, continuous=False, allow=False) == 0


def test_config_real_epochs_positive_rejected_without_explicit_opt_in() -> None:
    for continuous in (False, True):
        with pytest.raises(RealMarketWeightUpdateQuarantineError):
            _resolve(config_real_epochs=1, continuous=continuous, allow=False)


def test_continuous_alone_never_authorizes_real_market_weight_updates() -> None:
    assert _resolve(config_real_epochs=0, continuous=True, allow=False) == 0


def test_explicit_opt_in_mechanics() -> None:
    # Opt-in + continuous reproduces the historical continuous real training,
    # but only as a double explicit opt-in.
    assert _resolve(config_real_epochs=0, continuous=True, allow=True) == LIMIT
    # Opt-in honors the configured real epochs; the flag authorizes but does
    # not itself request real training (0 stays 0).
    assert _resolve(config_real_epochs=3, continuous=False, allow=True) == 3
    assert _resolve(config_real_epochs=0, continuous=False, allow=True) == 0


def test_negative_real_epochs_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        _resolve(config_real_epochs=-1, continuous=False, allow=False)


def test_quarantine_error_message_is_explicit_about_noncanonical_real_market_weight_updates() -> None:
    with pytest.raises(RealMarketWeightUpdateQuarantineError) as excinfo:
        _resolve(config_real_epochs=1, continuous=True, allow=False)
    message = str(excinfo.value)
    for token in ("NONCANONICAL", "EXPERIMENTAL", "REAL-MARKET", "WEIGHT UPDATES", "SYNTHETIC-ONLY", ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG):
        assert token in message


def test_opt_in_flag_defaults_off() -> None:
    argv = sys.argv
    try:
        sys.argv = ["train_double_heston.py"]
        args = trainer.parse_args()
    finally:
        sys.argv = argv
    assert args.allow_noncanonical_real_weight_updates is False
    assert args.continuous is False


def test_no_vague_override_flag_names() -> None:
    source = (PROJECT_ROOT / "train_double_heston.py").read_text(encoding="utf-8")
    for vague_flag in ("--force", "--unsafe", "--experimental"):
        assert vague_flag not in source


# ---------------------------------------------------------------------------
# Trainer-level CLI routes (fail-closed before any training work)
# ---------------------------------------------------------------------------


def test_normal_synthetic_run_completes(monkeypatch, tmp_path, capsys) -> None:
    calls = _install_fakes(monkeypatch)
    _run_main(monkeypatch, tmp_path, ["--mode", "ordinary"])
    assert [stage for stage, _ in calls["stages"]] == ["synthetic"]
    assert calls["completed_run_saved"] is True


def test_default_config_real_epochs_zero_run_completes(monkeypatch, tmp_path) -> None:
    # _install_fakes loads a config with real_epochs = 0: the default execution path.
    calls = _install_fakes(monkeypatch)
    _run_main(monkeypatch, tmp_path, [])
    assert [stage for stage, _ in calls["stages"]] == ["synthetic"]


def test_config_real_epochs_positive_blocks_before_any_training_work(monkeypatch, tmp_path) -> None:
    calls = _install_fakes(monkeypatch)

    def fake_load_config(path=None):
        return _synthetic_only_config(real_epochs=1)

    monkeypatch.setattr(trainer, "load_config", fake_load_config)
    with pytest.raises(RealMarketWeightUpdateQuarantineError):
        _run_main(monkeypatch, tmp_path, [])
    # Guard fired before dataloaders, before any train_stage call, and before
    # any optimizer/backward/weight mutation could run.
    assert calls["dataloaders_built"] is False
    assert calls["stages"] == []
    run_dir = tmp_path / "run"
    assert list(run_dir.iterdir()) == []  # not even a config snapshot was written


def test_continuous_with_real_epochs_zero_stays_synthetic_only(monkeypatch, tmp_path) -> None:
    calls = _install_fakes(monkeypatch)
    _run_main(monkeypatch, tmp_path, ["--continuous"])
    assert [stage for stage, _ in calls["stages"]] == ["synthetic"]


def test_continuous_with_real_epochs_positive_is_rejected(monkeypatch, tmp_path) -> None:
    calls = _install_fakes(monkeypatch)

    def fake_load_config(path=None):
        return _synthetic_only_config(real_epochs=1)

    monkeypatch.setattr(trainer, "load_config", fake_load_config)
    with pytest.raises(RealMarketWeightUpdateQuarantineError):
        _run_main(monkeypatch, tmp_path, ["--continuous"])
    assert calls["dataloaders_built"] is False
    assert calls["stages"] == []


def test_explicit_opt_in_runs_real_stage_with_banner(monkeypatch, tmp_path, capsys) -> None:
    calls = _install_fakes(monkeypatch, allow_real_stage=True)

    def fake_load_config(path=None):
        return _synthetic_only_config(real_epochs=2)

    monkeypatch.setattr(trainer, "load_config", fake_load_config)
    _run_main(monkeypatch, tmp_path, [ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG])
    assert [stage for stage, _ in calls["stages"]] == ["synthetic", "real_finetune"]
    # The real stage actually received the real-market training loader.
    assert calls["stages"][1][1] is calls["loaders"]["train_real"]
    assert "NONCANONICAL_EXPERIMENTAL_REAL_MARKET_WEIGHT_UPDATES" in capsys.readouterr().out


def test_resume_interrupted_real_stage_with_continuous_is_quarantined(monkeypatch, tmp_path) -> None:
    calls = _install_fakes(monkeypatch)
    run_dir = _write_resume_run(tmp_path, _synthetic_only_config(real_epochs=1), status="interrupted", stage_name="real_finetune")
    monkeypatch.setattr(sys, "argv", ["train_double_heston.py", "--mode", "ordinary", "--run-dir", str(run_dir), "--continuous"])
    with pytest.raises(RealMarketWeightUpdateQuarantineError):
        trainer.main()
    assert calls["dataloaders_built"] is False
    assert calls["stages"] == []


def test_resume_completed_continuous_reentry_blocked_without_opt_in(monkeypatch, tmp_path, capsys) -> None:
    calls = _install_fakes(monkeypatch)
    run_dir = _write_resume_run(tmp_path, _synthetic_only_config(real_epochs=0), status="completed", stage_name="synthetic")
    monkeypatch.setattr(sys, "argv", ["train_double_heston.py", "--mode", "ordinary", "--run-dir", str(run_dir), "--continuous"])
    trainer.main()
    assert calls["stages"] == []
    out = capsys.readouterr().out
    assert "already_completed" in out
    assert "blocked_by_real_market_weight_update_quarantine" in out


# ---------------------------------------------------------------------------
# Config/evidence hygiene and canonical-stack isolation
# ---------------------------------------------------------------------------


def test_live_default_configs_are_synthetic_only() -> None:
    for name in ("configs/default_experiment.json", "configs/smoke_experiment.json"):
        data = json.loads((PROJECT_ROOT / name).read_text(encoding="utf-8"))
        assert data["training"]["real_epochs"] == 0, name


def test_archive2_historical_configs_preserved_and_now_quarantined() -> None:
    for name in ("configs/archive2_default_experiment.json", "configs/archive2_smoke_experiment.json"):
        data = json.loads((PROJECT_ROOT / name).read_text(encoding="utf-8"))
        # Historical record intact: the archived experiment configs still document
        # the real fine-tuning they once ran, and are rejected without opt-in.
        assert data["training"]["real_epochs"] == 1, name
        with pytest.raises(RealMarketWeightUpdateQuarantineError):
            _resolve(config_real_epochs=data["training"]["real_epochs"], continuous=False, allow=False)


def test_canonical_training_files_do_not_depend_on_archive2() -> None:
    for rel in CANONICAL_TRAINING_FILES:
        source = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert "dheston" not in source, f"{rel} must not depend on Archive-2"
        assert "pde_residual_loss" not in source, f"{rel} must not use the Archive-2 PDE loss"


def test_pde_residual_loss_remains_confined_to_archive2() -> None:
    offenders: list[str] = []
    model3_pde_root = SRC_ROOT / "model3_pde"
    model3_pilot_driver = PROJECT_ROOT / "scripts" / "run_model3_pde_pilot.py"
    model3_contract = SRC_ROOT / "model3_evaluation" / "contracts.py"
    for root in (SRC_ROOT, PROJECT_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if (
                "__pycache__" in path.parts
                or "dheston" in path.parts
                or model3_pde_root in path.parents
                or path == model3_pilot_driver
                or path == model3_contract
            ):
                continue
            if "pde_residual_loss" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
    assert model3_pde_root.is_dir()
    assert model3_pilot_driver.is_file()
