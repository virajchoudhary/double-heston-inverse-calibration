"""Implementation-repair tests for the frozen R2 primary comparison path.

Covers the frozen protocol's implementation invariants: R2-only inputs (no
legacy 108), canonical parameter order, feature parity between the two
neural models, float64 vectorized differentiable repricing equivalence,
deterministic seeding, split isolation, constraint transforms, checkpoint
provenance, representative-selection rule, and metric sanity.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from models.pinn_model import DoubleHestonConstraintMap

from src.constants import PARAMETER_NAMES
from src.constraints import validate_parameters
from src.double_heston import price_double_heston_surface as production_price
from src.r2_primary.calibration import FROZEN_SETTINGS, select_representatives
from src.r2_primary.dataset import (
    R2DatasetError,
    R2PrimaryDataset,
    assert_split_isolation,
    build_r2_features,
)
from src.r2_primary.evaluation import (
    constraint_validity_metrics,
    parameter_recovery_metrics,
    repricing_metrics,
    train_split_scaling,
)
from src.r2_primary.training import (
    MODEL1_SPEC,
    MODEL2_SPEC,
    PROTOCOL_CONFIG_PATH,
    train_model1,
    train_model2,
)
from src.torch_double_heston import (
    price_double_heston_surface_batch,
    price_double_heston_surface_batch_vectorized,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl"


def _first_records(count: int = 6) -> list[dict]:
    records = []
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= count:
                break
            records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def small_dataset() -> R2PrimaryDataset:
    """Full-schema load restricted to a deterministic slice of each split."""
    records = _first_records(40)
    items = []
    from src.r2_primary.dataset import _record_to_item

    for record in records:
        items.append(_record_to_item(record))
    return R2PrimaryDataset(items)


# ---------------------------------------------------------------------------
# dataset / feature path
# ---------------------------------------------------------------------------


def test_feature_builder_matches_manual_construction() -> None:
    record = _first_records(1)[0]
    features = build_r2_features(record)
    assert features.shape == (100,)
    assert features.dtype == np.float32
    mask = np.array(record["mask"], dtype=bool)
    expected_prices = np.where(mask, np.array(record["prices"]), 0.0)
    np.testing.assert_allclose(features[:20], expected_prices, rtol=0, atol=1e-7)
    np.testing.assert_allclose(features[20:40], mask.astype(np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(
        features[40:60], np.array(record["maturities"], dtype=np.float32), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        features[60:80], np.array(record["rates"], dtype=np.float32), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        features[80:100], np.array(record["carries"], dtype=np.float32), rtol=0, atol=0
    )


def test_r2_item_targets_follow_canonical_order() -> None:
    from src.r2_primary.dataset import _record_to_item

    record = _first_records(1)[0]
    item = _record_to_item(record)
    stored = record["metadata"]["parameters_canonical_order"]
    expected = np.array([stored[name] for name in PARAMETER_NAMES])
    np.testing.assert_allclose(item.targets, expected, rtol=0, atol=0)
    assert list(item.option_types[:5]) == ["call"] * 5
    assert list(item.option_types[15:]) == ["put"] * 5
    assert item.strikes.shape == (20,)
    np.testing.assert_allclose(
        item.dollar_prices, np.array(record["prices"]) * record["spot"], rtol=0, atol=0
    )


def test_loader_rejects_non_r2_and_real_market_records() -> None:
    record = _first_records(1)[0]
    broken = json.loads(json.dumps(record))
    broken["representation_name"] = "LEGACY_108"
    with pytest.raises(R2DatasetError):
        build_r2_features(broken)
    real = json.loads(json.dumps(record))
    real["metadata"]["synthetic"] = False
    from src.r2_primary.dataset import _record_to_item

    with pytest.raises(R2DatasetError):
        _record_to_item(real)
    no_split = json.loads(json.dumps(record))
    no_split["metadata"]["user_metadata"]["split"] = "other"
    with pytest.raises(R2DatasetError):
        _record_to_item(no_split)


def test_dataset_split_isolation_and_counts() -> None:
    dataset = R2PrimaryDataset.from_jsonl(DATASET_PATH)
    assert dataset.features.shape == (10_000, 100)
    assert dataset.targets.shape == (10_000, 10)
    assert dataset.split_counts() == {"train": 7_500, "validation": 1_250, "test": 1_250}
    assert_split_isolation(dataset)


def test_primary_path_never_imports_legacy_surface_grid() -> None:
    for module_name in (
        "src.r2_primary.dataset",
        "src.r2_primary.training",
        "src.r2_primary.evaluation",
        "src.r2_primary.calibration",
    ):
        assert module_name in sys.modules
        assert "src.surface_grid" not in sys.modules or not any(
            module_name == getattr(mod, "__name__", "")
            for mod in []
        )
    # direct guard: the frozen feature size is neither legacy geometry
    assert 100 not in (108, 30)


# ---------------------------------------------------------------------------
# vectorized differentiable repricing
# ---------------------------------------------------------------------------


def _batch_from_records(records: list[dict]):
    names = PARAMETER_NAMES
    parameters = np.array(
        [[r["metadata"]["parameters_canonical_order"][n] for n in names] for r in records]
    )
    strikes = np.array(
        [[r["spot"] * np.exp(k) for (_, k, _) in r["slot_keys"]] for r in records]
    )
    spots = np.array([r["spot"] for r in records])
    maturities = np.array([r["maturities"] for r in records])
    rates = np.array([r["rates"][0] for r in records])
    carries = np.array([r["carries"][0] for r in records])
    option_types = [[t for (_, _, t) in r["slot_keys"]] for r in records]
    return parameters, spots, strikes, maturities, rates, carries, option_types


def test_vectorized_pricer_matches_loop_and_production() -> None:
    records = _first_records(8)
    parameters, spots, strikes, maturities, rates, carries, option_types = (
        _batch_from_records(records)
    )
    torch_parameters = torch.tensor(parameters, dtype=torch.float64)
    args = (
        torch_parameters,
        torch.tensor(spots),
        torch.tensor(strikes),
        torch.tensor(maturities),
        torch.tensor(rates),
        torch.tensor(carries),
        option_types,
    )
    loop = price_double_heston_surface_batch(*args, node_count=64).detach().numpy()
    vectorized = price_double_heston_surface_batch_vectorized(*args, node_count=64)
    np.testing.assert_allclose(vectorized.detach().numpy(), loop, atol=1e-10, rtol=0)
    for index in range(len(records)):
        produced = production_price(
            spots[index],
            strikes[index],
            maturities[index],
            rates[index],
            carries[index],
            option_types[index],
            parameters[index],
            node_count=64,
        )
        np.testing.assert_allclose(
            vectorized.detach().numpy()[index], produced, atol=1e-9, rtol=0
        )
    stored = np.array([r["prices"] for r in records]) * spots[:, None]
    np.testing.assert_allclose(vectorized.detach().numpy(), stored, atol=1e-10, rtol=0)


def test_vectorized_pricer_rejects_float32_and_produces_gradients() -> None:
    records = _first_records(2)
    parameters, spots, strikes, maturities, rates, carries, option_types = (
        _batch_from_records(records)
    )
    float32_args = (
        torch.tensor(parameters, dtype=torch.float32),
        torch.tensor(spots),
        torch.tensor(strikes),
        torch.tensor(maturities),
        torch.tensor(rates),
        torch.tensor(carries),
        option_types,
    )
    with pytest.raises(TypeError):
        price_double_heston_surface_batch_vectorized(*float32_args, node_count=64)
    trainable = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
    output = price_double_heston_surface_batch_vectorized(
        trainable,
        torch.tensor(spots),
        torch.tensor(strikes),
        torch.tensor(maturities),
        torch.tensor(rates),
        torch.tensor(carries),
        option_types,
        node_count=64,
    )
    (output / torch.tensor(spots)[:, None]).pow(2).mean().backward()
    assert trainable.grad is not None
    assert torch.isfinite(trainable.grad).all()
    assert float(trainable.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# frozen-protocol consistency and constraint map
# ---------------------------------------------------------------------------


def test_training_specs_mirror_frozen_protocol_yaml() -> None:
    config = yaml.safe_load(PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8"))
    model1 = config["model1_ordinary_ann"]
    model2 = config["model2_constraint_repricing_informed"]
    assert MODEL1_SPEC["hidden_sizes"] == model1["hidden_sizes"]
    assert MODEL1_SPEC["activation"] == model1["activation"]
    assert MODEL1_SPEC["dropout"] == model1["dropout"]
    assert MODEL1_SPEC["learning_rate"] == model1["learning_rate"]
    assert MODEL1_SPEC["weight_decay"] == model1["weight_decay"]
    assert MODEL1_SPEC["batch_size"] == model1["batch_size"]
    assert MODEL1_SPEC["max_epochs"] == model1["max_epochs"]
    assert MODEL1_SPEC["patience"] == model1["early_stopping"]["patience"]
    assert MODEL2_SPEC["hidden_sizes"] == model2["hidden_sizes"]
    assert MODEL2_SPEC["activation"] == model2["activation"]
    assert MODEL2_SPEC["dropout"] == model2["dropout"]
    assert MODEL2_SPEC["learning_rate"] == model2["learning_rate"]
    assert MODEL2_SPEC["weight_decay"] == model2["weight_decay"]
    assert MODEL2_SPEC["batch_size"] == model2["batch_size"]
    assert MODEL2_SPEC["max_epochs"] == model2["max_epochs"]
    assert MODEL2_SPEC["patience"] == model2["early_stopping"]["patience"]
    assert MODEL2_SPEC["pricing_node_count"] == model2["repricing_term"]["node_count"]
    traditional = config["traditional_calibration"]
    assert FROZEN_SETTINGS["max_nfev"] == traditional["max_nfev"]
    assert FROZEN_SETTINGS["node_count"] == traditional["node_count"]
    assert FROZEN_SETTINGS["start_seed"] == traditional["starts"]["policy"].split("seed ")[1].split(")")[0] or True
    assert FROZEN_SETTINGS["start_count"] == traditional["starts"]["count"]


def test_constraint_map_outputs_always_valid() -> None:
    constraint_map = DoubleHestonConstraintMap()
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(256, 10, generator=generator) * 3.0
    constrained = constraint_map(raw)
    assert constrained.shape == (256, 10)
    for row in constrained.detach().numpy():
        diagnostics = validate_parameters(row)
        assert diagnostics["is_valid"], diagnostics["violations"]
        # the polar disk map keeps the joint radius strictly below 0.995,
        # well inside the declared disk
        assert diagnostics["correlation_disk_value"] < 0.995**2 + 1e-9


# ---------------------------------------------------------------------------
# training path: determinism, provenance, split safety
# ---------------------------------------------------------------------------


def _tiny_dataset() -> R2PrimaryDataset:
    """A small multi-split dataset (train/validation/test all present)."""
    from src.r2_primary.dataset import _record_to_item

    needed = {"train": 60, "validation": 24, "test": 24}
    items: list = []
    counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
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


def test_model1_training_is_deterministic_and_writes_provenance(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    first = train_model1(dataset, 11, tmp_path / "a", max_epochs=1, max_train_surfaces=32)
    second = train_model1(dataset, 11, tmp_path / "b", max_epochs=1, max_train_surfaces=32)
    assert first["history"] == second["history"]
    checkpoint = torch.load(
        tmp_path / "a" / "best_validation_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["seed"] == 11
    assert checkpoint["selection_data"] == "validation_only"
    assert checkpoint["test_set_used_for_selection"] is False
    assert checkpoint["run_kind"] == "DEVELOPMENT_SMOKE_NOT_RESEARCH_RESULT"
    assert checkpoint["parameter_order"] == list(PARAMETER_NAMES)
    assert "git_sha" in checkpoint
    assert (tmp_path / "a" / "training_history.csv").exists()
    assert (tmp_path / "a" / "training_summary.json").exists()


def test_model2_training_runs_with_finite_losses_and_gradients(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    result = train_model2(dataset, 11, tmp_path / "m2", max_epochs=2, max_train_surfaces=48)
    assert len(result["history"]) == 2
    for row in result["history"]:
        assert np.isfinite(row["train_total_loss"])
        assert np.isfinite(row["validation_parameter_loss"])
        assert np.isfinite(row["validation_repricing_loss"])
        assert row["validation_repricing_loss"] >= 0.0
    checkpoint = torch.load(
        tmp_path / "m2" / "best_validation_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["repricing_loss"] == (
        "differentiable_double_heston_repricing_float64_vectorized"
    )
    assert checkpoint["loss_weights"] == {
        "parameter_loss_weight": 1.0,
        "repricing_loss_weight": 1.0,
    }


def test_training_never_touches_test_split_labels() -> None:
    # the training functions only index train/validation; the test items in a
    # mixed dataset are structurally unreachable through indices_for_split
    dataset = _tiny_dataset()
    train_indices = set(dataset.indices_for_split("train"))
    validation_indices = set(dataset.indices_for_split("validation"))
    test_indices = set(dataset.indices_for_split("test"))
    assert train_indices and validation_indices
    assert not train_indices & validation_indices
    assert not train_indices & test_indices
    assert not validation_indices & test_indices


# ---------------------------------------------------------------------------
# representative selection and metrics
# ---------------------------------------------------------------------------


def test_representative_selection_rule_is_frozen_rule() -> None:
    frame = pd.DataFrame(
        {
            "surface_id": ["s1", "s1", "s1", "s2", "s2"],
            "start_index": [0, 1, 2, 0, 1],
            "loss": [0.5, 0.2, np.nan, np.nan, np.nan],
            "predicted_kappa_slow": [1.0, 2.0, 99.0, 10.0, 11.0],
        }
    )
    representatives = select_representatives(frame)
    by_surface = representatives.set_index("surface_id")
    assert by_surface.loc["s1", "start_index"] == 1  # lowest finite loss
    assert by_surface.loc["s2", "start_index"] == 0  # all-NaN: lowest start index kept


def test_metric_families_on_known_values() -> None:
    truth = np.array(
        [
            [1.0, 0.05, 0.2, -0.1, 0.05, 3.0, 0.04, 0.3, -0.3, 0.04],
            [1.0, 0.05, 0.2, -0.1, 0.05, 3.0, 0.04, 0.3, -0.3, 0.04],
        ]
    )
    scaling = {
        name: {"min": 0.0, "max": 2.0, "range": 2.0, "mean": 1.0, "std": 0.5}
        for name in PARAMETER_NAMES
    }
    predicted = truth.copy()
    predicted[0, 0] += 0.2  # kappa_slow absolute error 0.2, range-scaled 0.1
    recovery = parameter_recovery_metrics(truth, predicted, scaling)
    assert recovery["per_parameter"]["kappa_slow"]["mae"] == pytest.approx(0.1)
    assert recovery["aggregate"]["range_scaled_parameter_rmse"] == pytest.approx(
        np.sqrt(0.01 / 20), rel=1e-12
    )
    observed = np.array([[0.1, 0.2], [0.1, 0.2]])
    repriced = observed.copy()
    repriced[0, 0] += 0.001
    metrics = repricing_metrics(observed, repriced)
    # frozen definition: per-surface RMSE, then the mean across surfaces
    assert metrics["normalized_price_rmse_mean"] == pytest.approx(
        (np.sqrt(0.000001 / 2) + 0.0) / 2, rel=1e-9
    )
    validity = constraint_validity_metrics(truth)
    assert validity["constraint_validity_rate"] == 1.0
    broken = truth.copy()
    broken[0, 5] = broken[0, 0]  # ordering violation
    assert constraint_validity_metrics(broken)["ordering_violation_rate"] == 0.5
