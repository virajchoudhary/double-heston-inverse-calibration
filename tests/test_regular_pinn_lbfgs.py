import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.mentor_dh_pinn.finetune_regular_pinn_lbfgs import (
    create_output_directory,
    expected_data_hashes,
    fine_tune,
    load_torch_checkpoint,
    loss_components,
    save_checkpoint,
    select_batches,
    surface_jacobian_loss,
)
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint
from scripts.mentor_dh_pinn.evaluate_regular_pinn_development import development_surfaces
from src.mentor_dh_pinn.regular_pinn_data import draw_points
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def checkpoint(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    config = {"factors": 2, "width": 8, "depth": 1, "sensitivity": 0.2, "weight_pde": 0.2}
    (directory / "config.json").write_text(json.dumps(config))
    model = TorchRegularVariancePINN(factors=2, width=8, depth=1)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()},
              str(directory / "model.safetensors"))
    return directory


def dataset(tmp_path, model):
    directory = tmp_path / "data"
    directory.mkdir()
    q = draw_points(24, 2, 123)
    qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    from src.mentor_dh_pinn.regular_pinn_data import coordinates
    coords, structural = coordinates(qt, 2, torch)
    correction = model.correction(coords, structural)
    gradient = torch.autograd.grad(correction.sum(), qt)[0][:, 2:]
    target = correction.detach().numpy() + 0.01
    np.savez(directory / "train.npz", q=q, g=target, dg_du=gradient.detach().numpy(),
             usable=np.ones(len(q), dtype=bool))
    np.savez(directory / "collocation.npz", q=draw_points(20, 2, 124, collocation=True))
    return directory


def test_checkpoint_load_and_assessor_round_trip_are_float64(tmp_path):
    source = checkpoint(tmp_path)
    model, _, _ = load_torch_checkpoint(source)
    assert all(parameter.dtype == torch.float64 and parameter.requires_grad
               for parameter in model.parameters())
    output = tmp_path / "torch-output"
    output.mkdir()
    config = {"factors": 2, "width": 8, "depth": 1,
              "framework": "pytorch", "dtype": "float64"}
    (output / "config.json").write_text(json.dumps(config))
    save_checkpoint(model, output / "model.safetensors")
    loaded, info = load_checkpoint(output)
    assert isinstance(loaded, TorchRegularVariancePINN)
    assert info["config"]["dtype"] == "float64"
    assert all(value.dtype == torch.float64 for value in load_file(
        str(output / "model.safetensors"), device="cpu"
    ).values())


def test_loss_rejects_non_float64_inputs():
    model = TorchRegularVariancePINN(factors=2, width=4, depth=1)
    q = torch.ones((2, 12), dtype=torch.float32)
    with pytest.raises(TypeError, match="float64"):
        loss_components(model, q, torch.ones(2), torch.ones((2, 10)), q,
                        torch.ones(10), sensitivity_weight=0.2, pde_weight=0.2)


def test_deterministic_lbfgs_updates_weights_and_decreases_loss(tmp_path):
    source = checkpoint(tmp_path)
    model, _, _ = load_torch_checkpoint(source)
    data = dataset(tmp_path, model)
    selected = select_batches(data, seed=99, anchor_batch=12, pde_batch=8)
    batches = selected[:5]
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial, final, history, finite = fine_tune(
        model, batches, sensitivity_weight=0.0, pde_weight=0.0,
        lr=0.5, max_iter=6, history_size=5, tolerance_grad=1e-12, tolerance_change=1e-15,
    )
    assert finite and history
    assert final["total"] < initial["total"]
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


def test_batch_selection_is_deterministic_and_validates_size(tmp_path):
    source = checkpoint(tmp_path)
    model, _, _ = load_torch_checkpoint(source)
    data = dataset(tmp_path, model)
    first = select_batches(data, seed=7, anchor_batch=8, pde_batch=6)
    second = select_batches(data, seed=7, anchor_batch=8, pde_batch=6)
    assert first[5] == second[5]
    for left, right in zip(first[:5], second[:5]):
        torch.testing.assert_close(left, right)
    with pytest.raises(ValueError, match="exceeds"):
        select_batches(data, seed=7, anchor_batch=100, pde_batch=6)


def test_output_directory_must_be_new(tmp_path):
    output = tmp_path / "result"
    create_output_directory(output)
    with pytest.raises(FileExistsError, match="must not already exist"):
        create_output_directory(output)


def test_expected_data_hashes_are_read_from_source_manifest(tmp_path):
    source = tmp_path / "source-manifest"
    source.mkdir()
    manifest = {
        "input_sha256": {
            "some/old/train.npz": "train-hash",
            "some/old/validation.npz": "validation-hash",
            "some/old/collocation.npz": "collocation-hash",
        }
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    expected, digest = expected_data_hashes(source)
    assert expected == {"train.npz": "train-hash", "collocation.npz": "collocation-hash"}
    assert digest


def test_development_surface_contract_is_fixed():
    surfaces = development_surfaces()
    assert len(surfaces) == 4
    assert all(q.shape == (126, 12) for q, _, _ in surfaces)
    assert all(truth.shape == (10,) for _, _, truth in surfaces)
    assert all(labels["usable"].all() for _, labels, _ in surfaces)


def test_surface_jacobian_loss_is_zero_for_its_own_teacher():
    model = TorchRegularVariancePINN(factors=2, width=4, depth=1)
    q = torch.tensor(draw_points(6, 2, 888), dtype=torch.float64).reshape(2, 3, 12)
    flat = q.reshape(-1, 12).detach().requires_grad_(True)
    from src.mentor_dh_pinn.regular_pinn_data import coordinates
    coords, structural = coordinates(flat, 2, torch)
    iv = model.iv(coords, structural)
    teacher = torch.autograd.grad(iv.sum(), flat)[0][:, 2:].reshape(2, 3, 10)
    coordinate, weak = surface_jacobian_loss(
        model, q, teacher, weak_directions=2, normalization_floor=1e-4
    )
    torch.testing.assert_close(coordinate, torch.zeros_like(coordinate), atol=1e-24, rtol=0)
    torch.testing.assert_close(weak, torch.zeros_like(weak), atol=1e-24, rtol=0)
