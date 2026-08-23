"""Execution-placement plumbing tests for the R2 primary comparison CLI.

Proves the GPU repair changes EXECUTION PLACEMENT ONLY:
- ``--device`` exists with safe explicit choices {cpu, cuda}, default cpu;
- the CLI forwards the requested device into train_model1/train_model2 on
  both the research and smoke paths (CUDA never auto-selected);
- ``_repricing_loss`` keeps every differentiable-pricer input tensor on the
  prediction device with the frozen float64 dtype;
- the CPU numeric path is bit-for-bit unchanged;
- frozen hyperparameters/splits are untouched (no test-split use anywhere).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.r2_primary import training as training_module
from src.r2_primary.dataset import R2PrimaryDataset
from src.r2_primary.training import (
    MODEL1_SPEC,
    MODEL2_SPEC,
    _build_argument_parser,
    _repricing_loss,
)
from src.torch_double_heston import (
    price_double_heston_surface_batch_vectorized as real_pricer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"


def _tiny_dataset() -> R2PrimaryDataset:
    """Small multi-split dataset (mirrors the implementation-test fixture)."""
    from src.r2_primary.dataset import _record_to_item

    needed = {"train": 24, "validation": 12, "test": 8}
    items: list = []
    counts = dict.fromkeys(needed, 0)
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            split = record["metadata"]["user_metadata"]["split"]
            if counts[split] < needed[split]:
                items.append(_record_to_item(record))
                counts[split] += 1
            if all(counts[name] >= needed[name] for name in needed):
                break
    return R2PrimaryDataset(items)


# ---------------------------------------------------------------------------
# frozen settings must not move with the plumbing repair
# ---------------------------------------------------------------------------


def test_frozen_hyperparameters_unchanged_by_device_repair() -> None:
    assert MODEL1_SPEC == {
        "hidden_sizes": [512, 256, 128, 64],
        "activation": "relu",
        "dropout": 0.10,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 20,
    }
    assert MODEL2_SPEC == {
        "hidden_sizes": [512, 512, 256, 256, 128],
        "activation": "gelu",
        "dropout": 0.05,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "batch_size": 64,
        "max_epochs": 200,
        "patience": 20,
        "parameter_loss_weight": 1.0,
        "repricing_loss_weight": 1.0,
        "pricing_node_count": 64,
        "repricing_compute_dtype": "float64",
    }


def test_training_splits_still_exclude_test_split() -> None:
    dataset = _tiny_dataset()
    train = set(dataset.indices_for_split("train"))
    validation = set(dataset.indices_for_split("validation"))
    test = set(dataset.indices_for_split("test"))
    assert test and train and validation
    assert not (train | validation) & test


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_exposes_device_with_safe_choices_and_cpu_default() -> None:
    parser = _build_argument_parser()
    default_args = parser.parse_args(["--model", "model2", "--seed", "11"])
    assert default_args.device == "cpu"
    cuda_args = parser.parse_args(
        ["--model", "model2", "--seed", "11", "--device", "cuda"]
    )
    assert cuda_args.device == "cuda"
    cpu_args = parser.parse_args(
        ["--model", "model1", "--seed", "22", "--device", "cpu"]
    )
    assert cpu_args.device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "model2", "--seed", "11", "--device", "mps"])


def _run_cli_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cli_extra: list[str],
    smoke: bool,
    expected_device: str,
    captured: dict,
) -> None:
    def fake_trainer(dataset, seed, output, **kwargs):
        captured["dataset"] = dataset
        captured["seed"] = seed
        captured["output"] = output
        captured["kwargs"] = kwargs
        return {"best_epoch": 1, "epochs_completed": 1, "runtime_seconds": 0.0}

    monkeypatch.setattr(
        training_module.R2PrimaryDataset,
        "from_jsonl",
        staticmethod(lambda path: _tiny_dataset()),
    )
    target = "train_model2" if "model2" in cli_extra[cli_extra.index("--model") + 1] else "train_model1"
    monkeypatch.setattr(training_module, target, fake_trainer)
    argv = ["prog", *cli_extra, "--output", str(tmp_path / "out")]
    if smoke:
        argv.append("--smoke")
    monkeypatch.setattr("sys.argv", argv)
    assert training_module.main() == 0
    assert captured["kwargs"]["device"] == expected_device


def test_research_model2_cli_forwards_cuda_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    _run_cli_forwarding(
        monkeypatch,
        tmp_path,
        cli_extra=["--model", "model2", "--seed", "11", "--device", "cuda"],
        smoke=False,
        expected_device="cuda",
        captured=captured,
    )


def test_research_model1_cli_forwards_cuda_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    _run_cli_forwarding(
        monkeypatch,
        tmp_path,
        cli_extra=["--model", "model1", "--seed", "33", "--device", "cuda"],
        smoke=False,
        expected_device="cuda",
        captured=captured,
    )


def test_smoke_model2_cli_forwards_device_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    _run_cli_forwarding(
        monkeypatch,
        tmp_path,
        cli_extra=["--model", "model2", "--seed", "11", "--device", "cuda"],
        smoke=True,
        expected_device="cuda",
        captured=captured,
    )


def test_default_invocation_remains_cpu_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for smoke in (False, True):
        captured: dict = {}
        _run_cli_forwarding(
            monkeypatch,
            tmp_path,
            cli_extra=["--model", "model2", "--seed", "11"],
            smoke=smoke,
            expected_device="cpu",
            captured=captured,
        )


# ---------------------------------------------------------------------------
# _repricing_loss single-device guarantee
# ---------------------------------------------------------------------------


def test_repricing_loss_keeps_every_pricing_tensor_on_prediction_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    for device_name in devices:
        observed_calls: list[dict] = []

        def spying_pricer(parameters, spots, strikes, maturities, rates, carries, option_types, *, node_count):
            observed_calls.append(
                {
                    "parameters": parameters,
                    "spots": spots,
                    "strikes": strikes,
                    "maturities": maturities,
                    "rates": rates,
                    "carries": carries,
                    "node_count": node_count,
                }
            )
            return torch.zeros(
                (parameters.shape[0], 20), dtype=torch.float64, device=parameters.device
            )

        monkeypatch.setattr(training_module, "price_double_heston_surface_batch_vectorized", spying_pricer)
        dataset = _tiny_dataset()
        items = [dataset.items[i] for i in dataset.indices_for_split("train")[:4]]
        prediction_device = torch.device(device_name)
        predicted = torch.randn(4, 10, dtype=torch.float32, device=prediction_device)
        loss = _repricing_loss(predicted, items, node_count=64)
        assert len(observed_calls) == 1
        call = observed_calls[0]
        for name in ("spots", "strikes", "maturities", "rates", "carries"):
            tensor = call[name]
            assert tensor.device == prediction_device, name
            assert tensor.dtype == torch.float64, name
        assert call["node_count"] == 64
        assert loss.device == prediction_device
        assert torch.isfinite(loss)


def test_repricing_loss_cpu_numerics_bitwise_unchanged() -> None:
    """The device kwarg must not perturb the CPU path vs the pre-repair code."""
    dataset = _tiny_dataset()
    items = [dataset.items[i] for i in dataset.indices_for_split("validation")[:3]]
    predicted = torch.tensor(
        np.stack([item.targets for item in items]), dtype=torch.float64
    )

    repaired = _repricing_loss(predicted, items, node_count=64)

    # verbatim pre-repair construction: CPU tensors, same dtypes/order
    spots = torch.tensor([item.spot for item in items], dtype=torch.float64)
    strikes = torch.tensor(np.stack([item.strikes for item in items]), dtype=torch.float64)
    maturities = torch.tensor(np.stack([item.maturities for item in items]), dtype=torch.float64)
    rates = torch.tensor([item.rate for item in items], dtype=torch.float64)
    carries = torch.tensor([item.carry for item in items], dtype=torch.float64)
    option_types = [list(item.option_types) for item in items]
    observed = torch.tensor(
        np.stack([item.normalized_prices for item in items]), dtype=torch.float64
    )
    mask = torch.tensor(np.stack([item.mask for item in items]), dtype=torch.float64)
    repriced = real_pricer(
        predicted.to(torch.float64),
        spots,
        strikes,
        maturities,
        rates,
        carries,
        option_types,
        node_count=64,
    )
    normalized = repriced / spots.unsqueeze(1)
    active = mask
    denominator = active.sum().clamp_min(1.0)
    reference = torch.sum(((normalized - observed) ** 2) * active) / denominator

    assert torch.equal(repaired, reference)
    assert repaired.dtype == torch.float64


# ---------------------------------------------------------------------------
# end-to-end placement (skips cleanly without CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_model2_training_runs_end_to_end_on_cuda(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    result = training_module.train_model2(
        dataset, 11, tmp_path / "m2cuda", max_epochs=1, max_train_surfaces=16, device="cuda"
    )
    assert result["history"][0]["epoch"] == 1
    for key in ("train_total_loss", "validation_total_loss", "validation_repricing_loss"):
        assert np.isfinite(result["history"][0][key])
    checkpoint = torch.load(
        tmp_path / "m2cuda" / "best_validation_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["device_used"] == "cuda"
    assert checkpoint["spec"]["repricing_compute_dtype"] == "float64"
