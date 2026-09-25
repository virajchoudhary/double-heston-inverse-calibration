#!/usr/bin/env python3
"""Deterministic float64 L-BFGS fine-tuning for the regular Double Heston PINN."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mentor_dh_pinn.regular_pinn_data import coordinates
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def expected_data_hashes(checkpoint_directory: Path) -> tuple[dict[str, str], str | None]:
    path = checkpoint_directory / "manifest.json"
    if not path.exists():
        return {}, None
    manifest = json.loads(path.read_text())
    expected = {}
    for recorded_path, digest in manifest.get("input_sha256", {}).items():
        name = Path(recorded_path).name
        if name in ("train.npz", "collocation.npz"):
            expected[name] = digest
    return expected, sha256(path)


def load_torch_checkpoint(checkpoint: Path) -> tuple[TorchRegularVariancePINN, dict, Path]:
    checkpoint = checkpoint.resolve()
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weights = directory / "model.safetensors" if checkpoint.is_dir() else checkpoint
    config = json.loads((directory / "config.json").read_text())
    if config.get("factors") != 2 or config.get("residual_blocks", 0):
        raise ValueError("Phase 1 supports only the regular one-network Double Heston checkpoint")
    keys = ("factors", "width", "depth", "tau_min", "tau_max", "x_half_width", "correction_limit")
    model = TorchRegularVariancePINN(**{key: config[key] for key in keys if key in config})
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model.double().train().requires_grad_(True)
    if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
        raise TypeError("Every network parameter must be float64")
    return model, config, weights


def select_batches(data: Path, *, seed: int, anchor_batch: int, pde_batch: int):
    train_path, collocation_path = data / "train.npz", data / "collocation.npz"
    with np.load(train_path) as source:
        usable = np.asarray(source["usable"], dtype=bool)
        valid = np.flatnonzero(usable)
        if anchor_batch > len(valid):
            raise ValueError("anchor-batch exceeds usable training rows")
        rng = np.random.default_rng(seed)
        anchor_indices = np.sort(rng.choice(valid, size=anchor_batch, replace=False))
        q = np.asarray(source["q"][anchor_indices], dtype=np.float64)
        g = np.asarray(source["g"][anchor_indices], dtype=np.float64)
        dg_du = np.asarray(source["dg_du"][anchor_indices], dtype=np.float64)
        all_dg_du = np.asarray(source["dg_du"][usable], dtype=np.float64)
    with np.load(collocation_path) as source:
        pool = np.asarray(source["q"], dtype=np.float64)
        if pde_batch > len(pool):
            raise ValueError("pde-batch exceeds collocation rows")
        pde_indices = np.sort(rng.choice(len(pool), size=pde_batch, replace=False))
        pde_q = pool[pde_indices]
    scale = np.maximum(np.sqrt(np.mean(all_dg_du**2, axis=0)), 0.02)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    return (
        tensor(q),
        tensor(g),
        tensor(dg_du),
        tensor(pde_q),
        tensor(scale),
        {
            "anchor_count": int(anchor_batch),
            "anchor_indices_sha256": index_sha256(anchor_indices),
            "pde_count": int(pde_batch),
            "pde_indices_sha256": index_sha256(pde_indices),
        },
    )


def select_jacobian_surfaces(path: Path, *, seed: int, count: int):
    with np.load(path) as source:
        if "jacobian" not in source:
            raise ValueError("Jacobian surface data must contain the teacher Jacobian")
        usable = np.flatnonzero(np.asarray(source["usable"], dtype=bool))
        if count > len(usable):
            raise ValueError("jacobian-batch exceeds usable teacher surfaces")
        indices = np.sort(np.random.default_rng(seed).choice(usable, size=count, replace=False))
        q = np.asarray(source["q"][indices], dtype=np.float64)
        jacobian = np.asarray(source["jacobian"][indices], dtype=np.float64)
    return (
        torch.as_tensor(q, dtype=torch.float64),
        torch.as_tensor(jacobian, dtype=torch.float64),
        {
            "jacobian_surface_count": int(count),
            "jacobian_surface_indices_sha256": index_sha256(indices),
        },
    )


def surface_jacobian_loss(
    model: TorchRegularVariancePINN,
    queries: torch.Tensor,
    teacher_jacobian: torch.Tensor,
    *,
    weak_directions: int,
    normalization_floor: float,
):
    if queries.dtype != torch.float64 or teacher_jacobian.dtype != torch.float64:
        raise TypeError("Jacobian supervision must remain float64")
    shape = queries.shape
    q = queries.detach().reshape(-1, shape[-1]).requires_grad_(True)
    coords, structural = coordinates(q, model.factors, torch)
    predicted = model.iv(coords, structural)
    learned = torch.autograd.grad(predicted.sum(), q, create_graph=True)[0][..., 2:]
    learned = learned.reshape(shape[0], shape[1], -1)
    scales = teacher_jacobian.square().mean(dim=1).sqrt().clamp_min(normalization_floor)
    difference = (learned - teacher_jacobian) / scales[:, None, :]
    coordinate_loss = difference.square().mean()
    weak_loss = torch.zeros((), dtype=torch.float64)
    if weak_directions:
        count = min(weak_directions, teacher_jacobian.shape[-1])
        normalized_teacher = teacher_jacobian / scales[:, None, :]
        _, singular, right = torch.linalg.svd(normalized_teacher, full_matrices=False)
        weak_vectors = right[:, -count:, :].detach()
        projected = torch.matmul(difference, weak_vectors.transpose(-1, -2))
        weak_scale = (singular[:, -count:] / math.sqrt(shape[1])).clamp_min(
            normalization_floor
        ).detach()
        weak_loss = (projected / weak_scale[:, None, :]).square().mean()
    return coordinate_loss, weak_loss


def loss_components(
    model: TorchRegularVariancePINN,
    anchor_q: torch.Tensor,
    anchor_g: torch.Tensor,
    anchor_dg_du: torch.Tensor,
    pde_q: torch.Tensor,
    sensitivity_scale: torch.Tensor,
    *,
    sensitivity_weight: float,
    pde_weight: float,
    jacobian_queries: torch.Tensor | None = None,
    teacher_jacobian: torch.Tensor | None = None,
    jacobian_weight: float = 0.0,
    weak_weight: float = 0.0,
    weak_directions: int = 2,
    jacobian_normalization_floor: float = 1e-4,
) -> dict[str, torch.Tensor]:
    if any(value.dtype != torch.float64 for value in
           (anchor_q, anchor_g, anchor_dg_du, pde_q, sensitivity_scale)):
        raise TypeError("All fine-tuning inputs must be float64")
    q = anchor_q.detach().requires_grad_(True)
    coords, structural = coordinates(q, model.factors, torch)
    predicted = model.correction(coords, structural)
    anchor = torch.mean(((predicted - anchor_g) / 0.05) ** 2)
    predicted_grad = torch.autograd.grad(predicted.sum(), q, create_graph=True)[0][..., 2:]
    sensitivity = torch.mean(((predicted_grad - anchor_dg_du) / sensitivity_scale) ** 2)

    pde_coords, pde_structural = coordinates(pde_q, model.factors, torch)
    pde_residual, diagnostics = residual(model, pde_coords, pde_structural)
    root_w = torch.sqrt(diagnostics["w"])
    d2 = pde_coords[:, 0] / root_w - 0.5 * root_w
    relevance = torch.clamp_min(torch.exp(-0.5 * d2.square()), 0.01).detach()
    absolute = pde_residual.abs()
    huber = torch.where(absolute < 0.1, 0.5 * pde_residual.square(),
                        0.1 * (absolute - 0.05)) / 0.005
    physics = torch.sum(relevance * huber) / torch.sum(relevance)
    constraints = torch.clamp_min(-diagnostics["convexity"], 0).square().mean()
    constraints = constraints + torch.clamp_min(
        -diagnostics["w"] * diagnostics["l_tau"], 0
    ).square().mean()
    jacobian = weak = torch.zeros((), dtype=torch.float64)
    if jacobian_weight or weak_weight:
        if jacobian_queries is None or teacher_jacobian is None:
            raise ValueError("Jacobian weights require fixed teacher surface data")
        jacobian, weak = surface_jacobian_loss(
            model, jacobian_queries, teacher_jacobian,
            weak_directions=weak_directions,
            normalization_floor=jacobian_normalization_floor,
        )
    total = (anchor + pde_weight * physics + sensitivity_weight * sensitivity
             + 0.05 * constraints + jacobian_weight * jacobian + weak_weight * weak)
    return {
        "total": total,
        "anchor": anchor,
        "physics": physics,
        "sensitivity": sensitivity,
        "constraints": constraints,
        "jacobian": jacobian,
        "weak_jacobian": weak,
    }


def fine_tune(
    model: TorchRegularVariancePINN,
    batches,
    *,
    sensitivity_weight: float,
    pde_weight: float,
    lr: float,
    max_iter: int,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    jacobian_batches=None,
    jacobian_weight: float = 0.0,
    weak_weight: float = 0.0,
    weak_directions: int = 2,
    jacobian_normalization_floor: float = 1e-4,
):
    anchor_q, anchor_g, anchor_dg_du, pde_q, sensitivity_scale = batches
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    evaluations = []
    finite_gradients = True

    def evaluate():
        return loss_components(
            model, anchor_q, anchor_g, anchor_dg_du, pde_q, sensitivity_scale,
            sensitivity_weight=sensitivity_weight, pde_weight=pde_weight,
            jacobian_queries=None if jacobian_batches is None else jacobian_batches[0],
            teacher_jacobian=None if jacobian_batches is None else jacobian_batches[1],
            jacobian_weight=jacobian_weight,
            weak_weight=weak_weight,
            weak_directions=weak_directions,
            jacobian_normalization_floor=jacobian_normalization_floor,
        )

    initial = {key: float(value.detach()) for key, value in evaluate().items()}

    def closure():
        nonlocal finite_gradients
        optimizer.zero_grad(set_to_none=True)
        parts = evaluate()
        if not all(torch.isfinite(value) for value in parts.values()):
            raise FloatingPointError("Non-finite L-BFGS loss")
        parts["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        finite_gradients = finite_gradients and all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        if not finite_gradients:
            raise FloatingPointError("Non-finite or missing L-BFGS gradient")
        evaluations.append({key: float(value.detach()) for key, value in parts.items()})
        return parts["total"]

    optimizer.step(closure)
    final = {key: float(value.detach()) for key, value in evaluate().items()}
    return initial, final, evaluations, finite_gradients


def save_checkpoint(model: TorchRegularVariancePINN, path: Path) -> None:
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if any(value.dtype != torch.float64 for value in state.values()):
        raise TypeError("Refusing to save a checkpoint containing non-float64 tensors")
    save_file(state, str(path), metadata={"framework": "pytorch", "dtype": "float64"})


def create_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError("Output path must not already exist")
    path.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=917017)
    parser.add_argument("--anchor-batch", type=int, default=4096)
    parser.add_argument("--pde-batch", type=int, default=512)
    parser.add_argument("--sensitivity-weight", type=float, default=0.2)
    parser.add_argument("--pde-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--tolerance-change", type=float, default=1e-12)
    parser.add_argument("--jacobian-data", type=Path)
    parser.add_argument("--jacobian-batch", type=int, default=0)
    parser.add_argument("--jacobian-weight", type=float, default=0.0)
    parser.add_argument("--weak-weight", type=float, default=0.0)
    parser.add_argument("--weak-directions", type=int, default=2)
    parser.add_argument("--jacobian-normalization-floor", type=float, default=1e-4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output path must not already exist")
    if min(args.anchor_batch, args.pde_batch, args.max_iter, args.history_size) < 1:
        parser.error("batch sizes and L-BFGS iteration/history sizes must be positive")
    if min(args.sensitivity_weight, args.pde_weight, args.lr) < 0 or args.lr == 0:
        parser.error("loss weights must be nonnegative and learning rate must be positive")
    if min(args.jacobian_batch, args.jacobian_weight, args.weak_weight) < 0:
        parser.error("Jacobian batch and weights must be nonnegative")
    if (args.jacobian_weight or args.weak_weight) and (args.jacobian_data is None or args.jacobian_batch < 1):
        parser.error("Positive Jacobian weights require --jacobian-data and --jacobian-batch")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model, source_config, weights = load_torch_checkpoint(args.checkpoint)
    selected = select_batches(
        args.data, seed=args.seed, anchor_batch=args.anchor_batch, pde_batch=args.pde_batch
    )
    batches, selection = selected[:5], selected[5]
    jacobian_batches = None
    if args.jacobian_data is not None:
        jq, jj, jacobian_selection = select_jacobian_surfaces(
            args.jacobian_data, seed=args.seed + 1, count=args.jacobian_batch
        )
        jacobian_batches = (jq, jj)
        selection.update(jacobian_selection)
    before_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    initial, final, evaluations, finite_gradients = fine_tune(
        model,
        batches,
        sensitivity_weight=args.sensitivity_weight,
        pde_weight=args.pde_weight,
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=args.history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        jacobian_batches=jacobian_batches,
        jacobian_weight=args.jacobian_weight,
        weak_weight=args.weak_weight,
        weak_directions=args.weak_directions,
        jacobian_normalization_floor=args.jacobian_normalization_floor,
    )
    seconds = time.perf_counter() - started
    changed = any(not torch.equal(before_state[key], value) for key, value in model.state_dict().items())
    if not changed:
        raise RuntimeError("L-BFGS did not update any network weight")

    create_output_directory(args.out)
    architecture_keys = (
        "factors", "width", "depth", "tau_min", "tau_max", "x_half_width",
        "correction_limit", "architecture", "selection", "calibration",
    )
    output_config = {
        **{key: source_config[key] for key in architecture_keys if key in source_config},
        "data": str(args.data),
        "out": str(args.out),
        "framework": "pytorch",
        "dtype": "float64",
        "optimizer": "torch.optim.LBFGS",
        "fine_tuned_from": str(weights),
        "fine_tune_seed": args.seed,
        "lbfgs": {
            "lr": args.lr,
            "max_iter": args.max_iter,
            "history_size": args.history_size,
            "tolerance_grad": args.tolerance_grad,
            "tolerance_change": args.tolerance_change,
            "line_search_fn": "strong_wolfe",
        },
        "anchor_batch": args.anchor_batch,
        "pde_batch": args.pde_batch,
        "jacobian_data": str(args.jacobian_data) if args.jacobian_data else None,
        "jacobian_batch": args.jacobian_batch,
        "jacobian_weight": args.jacobian_weight,
        "weak_weight": args.weak_weight,
        "weak_directions": args.weak_directions,
        "jacobian_normalization_floor": args.jacobian_normalization_floor,
        "sensitivity_weight": args.sensitivity_weight,
        "pde_weight": args.pde_weight,
        "pretraining_config": source_config,
    }
    (args.out / "config.json").write_text(json.dumps(output_config, indent=2))
    save_checkpoint(model, args.out / "model.safetensors")
    sources = [
        Path(__file__),
        ROOT / "src/mentor_dh_pinn/regular_pinn_torch.py",
        ROOT / "src/mentor_dh_pinn/regular_pinn_data.py",
    ]
    actual_data_hashes = {
        "train.npz": sha256(args.data / "train.npz"),
        "collocation.npz": sha256(args.data / "collocation.npz"),
    }
    expected_hashes, source_manifest_sha256 = expected_data_hashes(weights.parent)
    manifest = {
        "status": "complete",
        "source_checkpoint": str(weights),
        "source_checkpoint_sha256": sha256(weights),
        "source_config_sha256": sha256(weights.parent / "config.json"),
        "source_manifest_sha256": source_manifest_sha256,
        "data_sha256": actual_data_hashes,
        "source_training_data_compatibility": {
            "expected_sha256": expected_hashes,
            "exact_hash_match": bool(expected_hashes)
            and all(actual_data_hashes.get(name) == digest for name, digest in expected_hashes.items()),
            "interpretation": (
                "False means the fine-tuning data is a regenerated cohort, not the byte-identical "
                "pretraining artifact; checkpoint and current data provenance remain separate."
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "seed": args.seed,
        "dtype": "torch.float64",
        "deterministic_algorithms": True,
        "fixed_batch_selection": selection,
        "optimizer": output_config["lbfgs"],
        "loss_weights": {
            "anchor": 1.0,
            "pde": args.pde_weight,
            "sensitivity": args.sensitivity_weight,
            "constraints": 0.05,
            "jacobian": args.jacobian_weight,
            "weak_jacobian": args.weak_weight,
        },
        "initial_loss": initial,
        "final_loss": final,
        "loss_decreased": bool(final["total"] < initial["total"]),
        "finite_gradients": finite_gradients,
        "weights_changed": changed,
        "closure_evaluations": len(evaluations),
        "seconds": seconds,
    }
    if args.jacobian_data:
        manifest["data_sha256"]["jacobian_surfaces.npz"] = sha256(args.jacobian_data)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "history.json").write_text(json.dumps(evaluations, indent=2))
    (args.out / "complete.json").write_text(json.dumps({
        "status": "complete",
        "optimizer": "torch.optim.LBFGS",
        "dtype": "float64",
        "initial_total_loss": initial["total"],
        "final_total_loss": final["total"],
        "loss_decreased": manifest["loss_decreased"],
        "finite_gradients": finite_gradients,
        "weights_changed": changed,
        "closure_evaluations": len(evaluations),
        "seconds": seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
