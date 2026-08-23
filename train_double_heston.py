"""Archive-2 (donor/experimental) Double Heston inverse-model trainer.

NONCANONICAL ENTRYPOINT. This trainer is Archive-2 donor code, not the canonical
research stack. Canonical policy (docs/OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md,
section 5): primary neural weight learning is SYNTHETIC-ONLY and real market data
is reserved for frozen-model evaluation. Real-market neural weight updates
(``training.real_epochs > 0`` or continuous real re-entry) are quarantined
fail-closed behind the explicit ``--allow-noncanonical-real-weight-updates``
opt-in and are disabled by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dheston.config import load_config
from dheston.real_market_policy import (
    ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG,
    resolve_real_market_epochs,
)
from dheston.data.surfaces import SurfaceDataset, build_surface_records, cap_records, pad_surface_batch, read_option_rows, split_records_chronologically
from dheston.data.synthetic import build_synthetic_records
from dheston.evaluation.metrics import mae, parameter_summary, rmse
from dheston.models.losses import build_loss_components, predict_surface_prices
from dheston.models.networks import DeepSurfaceInverseModel
from dheston.pricing.heston import FourierConfig


CHECKPOINT_FILENAME = "latest_checkpoint.pt"
MARKER_FILENAME = "training_marker.json"
METRICS_FILENAME = "metrics.json"
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.json"


CONTINUOUS_EPOCH_LIMIT = 999_999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ordinary or physics-aware Double Heston inverse model.")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON overriding configs/default_experiment.json")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--mode", type=str, choices=["ordinary", "physics"], default="physics")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol subset")
    parser.add_argument("--run-dir", type=str, default=None, help="Stable experiment directory to save checkpoints and resume from")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint inside --run-dir")
    parser.add_argument("--continuous", action="store_true", help="Auto-resume orchestration: resume from the latest checkpoint inside --run-dir if one exists and keep saving per-epoch checkpoints. This flag does NOT authorize real-market neural weight updates; real fine-tuning additionally requires --allow-noncanonical-real-weight-updates.")
    parser.add_argument(
        "--allow-noncanonical-real-weight-updates",
        action="store_true",
        help=(
            "NONCANONICAL EXPERIMENTAL opt-in: allow REAL-MARKET neural WEIGHT UPDATES "
            "(the real_finetune stage: training.real_epochs > 0, or --continuous real training). "
            "Disabled by default. Real-market weight updating conflicts with the canonical "
            "synthetic-only research protocol (docs/OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md, "
            "section 5) and is not part of the research baseline."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def average_history(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    keys = history[0].keys()
    return {key: float(np.mean([item[key] for item in history])) for key in keys}


def model_state_to_cpu(model: DeepSurfaceInverseModel) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def clone_state_dict(state_dict: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if state_dict is None:
        return None
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def normalize_best_validation(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def serialize_stage_log(history: list[dict[str, float]], best_validation_total: float, status: str, next_epoch: int, total_epochs: int) -> dict[str, Any]:
    return {
        "history": history,
        "best_validation_total": normalize_best_validation(best_validation_total),
        "status": status,
        "next_epoch": int(next_epoch),
        "total_epochs": int(total_epochs),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_FILENAME


def marker_path(run_dir: Path) -> Path:
    return run_dir / MARKER_FILENAME


def metrics_path(run_dir: Path) -> Path:
    return run_dir / METRICS_FILENAME


def config_snapshot_path(run_dir: Path) -> Path:
    return run_dir / CONFIG_SNAPSHOT_FILENAME


def save_training_state(
    run_dir: Path,
    *,
    mode: str,
    stage_name: str,
    next_epoch: int,
    total_epochs: int,
    status: str,
    model_state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any] | None,
    best_model_state_dict: dict[str, torch.Tensor] | None,
    best_validation_total: float,
    training_log: list[dict[str, float]],
    stage_logs: dict[str, Any],
) -> None:
    payload = {
        "mode": mode,
        "stage_name": stage_name,
        "next_epoch": int(next_epoch),
        "total_epochs": int(total_epochs),
        "status": status,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "best_model_state_dict": best_model_state_dict,
        "best_validation_total": float(best_validation_total) if np.isfinite(best_validation_total) else float("inf"),
        "training_log": training_log,
        "stage_logs": stage_logs,
    }
    torch.save(payload, checkpoint_path(run_dir))
    write_json(
        marker_path(run_dir),
        {
            "mode": mode,
            "stage_name": stage_name,
            "next_epoch": int(next_epoch),
            "total_epochs": int(total_epochs),
            "status": status,
            "checkpoint_file": str(checkpoint_path(run_dir)),
            "updated_at": payload["saved_at"],
        },
    )


def load_training_state(run_dir: Path) -> dict[str, Any]:
    checkpoint_file = checkpoint_path(run_dir)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")
    return torch.load(checkpoint_file, map_location="cpu", weights_only=False)


def resolve_run_dir(args: argparse.Namespace, output_root: str) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    return (PROJECT_ROOT / output_root / f"{timestamp_string()}_{args.mode}").resolve()


def maybe_load_resume_config(run_dir: Path) -> dict[str, Any]:
    snapshot = config_snapshot_path(run_dir)
    if not snapshot.exists():
        raise FileNotFoundError(f"Cannot resume because {snapshot} does not exist.")
    with snapshot.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_stage(
    model: DeepSurfaceInverseModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pricing_config: FourierConfig,
    loss_weights: dict[str, float],
    epochs: int,
    pde_points: int,
    run_dir: Path,
    mode: str,
    stage_name: str,
    stage_logs: dict[str, Any],
    resume_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    best_state = model_state_to_cpu(model)
    best_validation_loss = float("inf")
    training_log: list[dict[str, float]] = []
    start_epoch = 1

    if resume_state is not None and resume_state.get("stage_name") == stage_name:
        training_log = list(resume_state.get("training_log", []))
        best_validation_loss = float(resume_state.get("best_validation_total", float("inf")))
        best_state = clone_state_dict(resume_state.get("best_model_state_dict")) or model_state_to_cpu(model)
        start_epoch = int(resume_state.get("next_epoch", 1))

    if start_epoch > epochs:
        return serialize_stage_log(training_log, best_validation_loss, "completed", start_epoch, epochs), best_state

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_history: list[dict[str, float]] = []
        try:
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["features"], batch["mask"])
                losses = build_loss_components(output, batch, pricing_config, loss_weights, pde_points)
                losses["total"].backward()
                optimizer.step()
                train_history.append({name: float(value.detach().cpu()) for name, value in losses.items()})
        except KeyboardInterrupt:
            partial_stage_log = serialize_stage_log(training_log, best_validation_loss, "interrupted", epoch, epoch if epochs >= CONTINUOUS_EPOCH_LIMIT else epochs)
            interrupted_logs = dict(stage_logs)
            interrupted_logs[stage_name] = partial_stage_log
            save_training_state(
                run_dir,
                mode=mode,
                stage_name=stage_name,
                next_epoch=epoch,
                total_epochs=epoch if epochs >= CONTINUOUS_EPOCH_LIMIT else epochs,
                status="interrupted",
                model_state_dict=model_state_to_cpu(model),
                optimizer_state_dict=optimizer.state_dict(),
                best_model_state_dict=clone_state_dict(best_state),
                best_validation_total=best_validation_loss,
                training_log=training_log,
                stage_logs=interrupted_logs,
            )
            raise

        model.eval()
        valid_history: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = move_batch_to_device(batch, device)
                output = model(batch["features"], batch["mask"])
                losses = build_loss_components(output, batch, pricing_config, loss_weights, pde_points=0)
                valid_history.append({name: float(value.detach().cpu()) for name, value in losses.items()})

        train_mean = average_history(train_history)
        valid_mean = average_history(valid_history)
        training_log.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_mean.items()},
                **{f"valid_{key}": value for key, value in valid_mean.items()},
            }
        )

        if valid_mean.get("total", float("inf")) < best_validation_loss:
            best_validation_loss = valid_mean["total"]
            best_state = model_state_to_cpu(model)

        stage_status = "running" if epoch < epochs else "stage_complete"
        display_total_epochs = epoch if epochs >= CONTINUOUS_EPOCH_LIMIT else epochs
        current_stage_log = serialize_stage_log(training_log, best_validation_loss, stage_status, epoch + 1, display_total_epochs)
        checkpoint_logs = dict(stage_logs)
        checkpoint_logs[stage_name] = current_stage_log
        save_training_state(
            run_dir,
            mode=mode,
            stage_name=stage_name,
            next_epoch=epoch + 1,
            total_epochs=display_total_epochs,
            status=stage_status,
            model_state_dict=model_state_to_cpu(model),
            optimizer_state_dict=optimizer.state_dict(),
            best_model_state_dict=clone_state_dict(best_state),
            best_validation_total=best_validation_loss,
            training_log=training_log,
            stage_logs=checkpoint_logs,
        )
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "total_real_epochs": len(training_log),
                    "train_total": train_mean.get("total"),
                    "valid_total": valid_mean.get("total"),
                    "run_dir": str(run_dir),
                }
            )
        )

    display_total = len(training_log) if epochs >= CONTINUOUS_EPOCH_LIMIT else epochs
    return serialize_stage_log(training_log, best_validation_loss, "completed", epochs + 1, display_total), best_state


def evaluate_loader(
    model: DeepSurfaceInverseModel,
    loader: DataLoader,
    device: torch.device,
    pricing_config: FourierConfig,
) -> dict[str, Any]:
    model.eval()
    price_predictions = []
    price_targets = []
    parameter_predictions = []
    parameter_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch["features"], batch["mask"])
            predicted_prices, market_prices = predict_surface_prices(output["params"], batch, pricing_config)
            expanded_spot = batch["spot"].unsqueeze(1).expand_as(batch["market_price"])[batch["mask"]]
            price_predictions.append((predicted_prices / expanded_spot.clamp_min(1.0)).cpu())
            price_targets.append((market_prices / expanded_spot.clamp_min(1.0)).cpu())
            if batch["target_params"] is not None:
                parameter_predictions.append(output["params"].cpu())
                parameter_targets.append(batch["target_params"].cpu())

    predicted_prices_np = torch.cat(price_predictions).numpy()
    target_prices_np = torch.cat(price_targets).numpy()
    metrics: dict[str, Any] = {
        "price_mae_normalized": mae(predicted_prices_np, target_prices_np),
        "price_rmse_normalized": rmse(predicted_prices_np, target_prices_np),
    }

    if parameter_predictions and parameter_targets:
        predicted_params = torch.cat(parameter_predictions)
        target_params = torch.cat(parameter_targets)
        metrics.update(parameter_summary(predicted_params, target_params))

    return metrics


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, int], FourierConfig]:
    filters = config["filters"]
    rows = read_option_rows(
        dataset_path=config["dataset_path"],
        symbols=filters.get("symbols") or None,
        model_ready_only=bool(filters.get("model_ready_only", True)),
    )
    market_inputs = config["market_inputs"]
    real_records = build_surface_records(
        rows,
        risk_free_rate=float(market_inputs["risk_free_rate"]),
        dividend_yield=float(market_inputs["dividend_yield"]),
        min_surface_points=int(filters["min_surface_points"]),
        max_surface_points=int(filters["max_surface_points"]),
    )
    split_config = config["split"]
    split_records = split_records_chronologically(real_records, split_config["train_fraction"], split_config["validation_fraction"])
    caps = filters.get("surface_cap_per_split", {})
    seed = int(config["random_seed"])
    train_real = cap_records(split_records["train"], caps.get("train"), seed=seed)
    valid_real = cap_records(split_records["validation"], caps.get("validation"), seed=seed + 1)
    test_real = cap_records(split_records["test"], caps.get("test"), seed=seed + 2)

    pricing_config = FourierConfig(**config["pricing"])
    synthetic_cfg = config["synthetic"]
    train_synth = build_synthetic_records(train_real, synthetic_cfg["train_samples"], synthetic_cfg["noise_std"], seed=seed + 10, pricing_config=pricing_config)
    valid_synth = build_synthetic_records(valid_real, synthetic_cfg["validation_samples"], synthetic_cfg["noise_std"], seed=seed + 20, pricing_config=pricing_config)
    test_synth = build_synthetic_records(test_real, synthetic_cfg["test_samples"], synthetic_cfg["noise_std"], seed=seed + 30, pricing_config=pricing_config)

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    loaders = {
        "train_synth": DataLoader(SurfaceDataset(train_synth), batch_size=batch_size, shuffle=True, collate_fn=pad_surface_batch),
        "valid_synth": DataLoader(SurfaceDataset(valid_synth), batch_size=batch_size, shuffle=False, collate_fn=pad_surface_batch),
        "test_synth": DataLoader(SurfaceDataset(test_synth), batch_size=batch_size, shuffle=False, collate_fn=pad_surface_batch),
        "train_real": DataLoader(SurfaceDataset(train_real), batch_size=batch_size, shuffle=True, collate_fn=pad_surface_batch),
        "valid_real": DataLoader(SurfaceDataset(valid_real), batch_size=batch_size, shuffle=False, collate_fn=pad_surface_batch),
        "test_real": DataLoader(SurfaceDataset(test_real), batch_size=batch_size, shuffle=False, collate_fn=pad_surface_batch),
    }
    counts = {
        "real_train_surfaces": len(train_real),
        "real_validation_surfaces": len(valid_real),
        "real_test_surfaces": len(test_real),
        "synthetic_train_surfaces": len(train_synth),
        "synthetic_validation_surfaces": len(valid_synth),
        "synthetic_test_surfaces": len(test_synth),
    }
    return loaders, counts, pricing_config


def save_completed_run(
    run_dir: Path,
    *,
    mode: str,
    model: DeepSurfaceInverseModel,
    stage_logs: dict[str, Any],
    config: dict[str, Any],
    counts: dict[str, int],
    synthetic_metrics: dict[str, Any],
    real_metrics: dict[str, Any],
) -> None:
    torch.save(model.state_dict(), run_dir / f"{mode}_model.pt")
    write_json(
        metrics_path(run_dir),
        {
            "mode": mode,
            "synthetic_metrics": synthetic_metrics,
            "real_metrics": real_metrics,
            "counts": counts,
            "stage_logs": stage_logs,
        },
    )
    save_training_state(
        run_dir,
        mode=mode,
        stage_name="completed",
        next_epoch=0,
        total_epochs=0,
        status="completed",
        model_state_dict=model_state_to_cpu(model),
        optimizer_state_dict=None,
        best_model_state_dict=model_state_to_cpu(model),
        best_validation_total=float("inf"),
        training_log=[],
        stage_logs=stage_logs,
    )
    write_json(config_snapshot_path(run_dir), config)


def main() -> None:
    args = parse_args()

    base_config = load_config(args.config)
    run_dir = resolve_run_dir(args, base_config["output_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.continuous:
        args.resume = args.resume or checkpoint_path(run_dir).exists()

    if args.resume:
        if args.run_dir is None:
            raise ValueError("--resume requires --run-dir so the script knows which checkpoint to load.")
        config = maybe_load_resume_config(run_dir)
        resume_state = load_training_state(run_dir)
        if resume_state.get("mode") != args.mode:
            raise ValueError(f"Checkpoint mode {resume_state.get('mode')} does not match requested mode {args.mode}.")
    else:
        if checkpoint_path(run_dir).exists():
            raise FileExistsError(f"{run_dir} already contains a checkpoint. Use --resume or --continuous.")
        config = base_config
        if args.dataset is not None:
            config["dataset_path"] = args.dataset
        if args.symbols:
            config["filters"]["symbols"] = args.symbols
        resume_state = None

    # Fail-closed quarantine: resolve real-market weight-update authorization before
    # any config snapshot write, dataloader, model, optimizer, or training loop runs.
    allow_noncanonical_real_weight_updates = bool(args.allow_noncanonical_real_weight_updates)
    real_epochs = resolve_real_market_epochs(
        config_real_epochs=int(config["training"].get("real_epochs", 0)),
        continuous_requested=bool(args.continuous),
        allow_noncanonical_real_weight_updates=allow_noncanonical_real_weight_updates,
        continuous_epoch_limit=CONTINUOUS_EPOCH_LIMIT,
    )

    if not args.resume:
        write_json(config_snapshot_path(run_dir), config)

    set_seed(int(config["random_seed"]))

    training_cfg = config["training"]
    device = torch.device(training_cfg["device"])
    pde_points = int(training_cfg["pde_points_per_batch"])

    loaders, counts, pricing_config = build_dataloaders(config)
    sample_batch = next(iter(loaders["train_synth"]))
    input_dim = sample_batch["features"].shape[-1]
    model = DeepSurfaceInverseModel(
        input_dim=input_dim,
        hidden_dim=int(training_cfg["hidden_dim"]),
        dropout=float(training_cfg["dropout"]),
    ).to(device)

    stage_logs: dict[str, Any] = {}
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        stage_logs = dict(resume_state.get("stage_logs", {}))
        if resume_state.get("status") == "completed":
            if not args.continuous or not allow_noncanonical_real_weight_updates:
                print(
                    json.dumps(
                        {
                            "status": "already_completed",
                            "run_dir": str(run_dir),
                            "metrics_file": str(metrics_path(run_dir)),
                            "marker_file": str(marker_path(run_dir)),
                            **(
                                {"continuous_real_reentry": "blocked_by_real_market_weight_update_quarantine"}
                                if args.continuous
                                else {}
                            ),
                        },
                        indent=2,
                    )
                )
                return
            # Explicit NONCANONICAL opt-in: re-enter continuous real training from the
            # completed model. Carry forward the old real_finetune history and
            # continue epoch count.
            old_real_log = stage_logs.get("real_finetune", {})
            old_history = old_real_log.get("history", [])
            old_best = old_real_log.get("best_validation_total", float("inf"))
            resume_state["stage_name"] = "real_finetune"
            resume_state["next_epoch"] = len(old_history) + 1
            resume_state["total_epochs"] = CONTINUOUS_EPOCH_LIMIT
            resume_state["status"] = "ready"
            resume_state["training_log"] = list(old_history)
            resume_state["best_validation_total"] = old_best if old_best is not None else float("inf")
            resume_state["best_model_state_dict"] = model_state_to_cpu(model)
            resume_state["optimizer_state_dict"] = None
            print(json.dumps({"status": "continuing_real_training", "run_dir": str(run_dir)}))

    try:
        synthetic_resume = resume_state if resume_state is not None and resume_state.get("stage_name") == "synthetic" else None
        real_resume = resume_state if resume_state is not None and resume_state.get("stage_name") == "real_finetune" else None
        synthetic_loss_weights = config["losses"][args.mode]

        if real_resume is None:
            synthetic_optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
            if synthetic_resume is not None and synthetic_resume.get("optimizer_state_dict") is not None:
                synthetic_optimizer.load_state_dict(synthetic_resume["optimizer_state_dict"])
            stage_logs["synthetic"], best_synthetic_state = train_stage(
                model,
                loaders["train_synth"],
                loaders["valid_synth"],
                synthetic_optimizer,
                device,
                pricing_config,
                synthetic_loss_weights,
                epochs=int(training_cfg["epochs"]),
                pde_points=pde_points,
                run_dir=run_dir,
                mode=args.mode,
                stage_name="synthetic",
                stage_logs=stage_logs,
                resume_state=synthetic_resume,
            )
            model.load_state_dict(best_synthetic_state)

            if real_epochs > 0:
                save_training_state(
                    run_dir,
                    mode=args.mode,
                    stage_name="real_finetune",
                    next_epoch=1,
                    total_epochs=real_epochs,
                    status="ready",
                    model_state_dict=model_state_to_cpu(model),
                    optimizer_state_dict=None,
                    best_model_state_dict=model_state_to_cpu(model),
                    best_validation_total=float("inf"),
                    training_log=[],
                    stage_logs=stage_logs,
                )

        if real_epochs > 0:
            print(
                json.dumps(
                    {
                        "status": "NONCANONICAL_EXPERIMENTAL_REAL_MARKET_WEIGHT_UPDATES",
                        "opt_in_flag": ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG,
                        "real_epochs": real_epochs,
                        "note": "Real-market neural weight updating is not part of the canonical synthetic-only research baseline.",
                    },
                    indent=2,
                )
            )
            real_optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]) * 0.5)
            if real_resume is not None and real_resume.get("optimizer_state_dict") is not None:
                real_optimizer.load_state_dict(real_resume["optimizer_state_dict"])
            stage_logs["real_finetune"], best_real_state = train_stage(
                model,
                loaders["train_real"],
                loaders["valid_real"],
                real_optimizer,
                device,
                pricing_config,
                config["losses"]["real_finetune"],
                epochs=real_epochs,
                pde_points=pde_points,
                run_dir=run_dir,
                mode=args.mode,
                stage_name="real_finetune",
                stage_logs=stage_logs,
                resume_state=real_resume,
            )
            model.load_state_dict(best_real_state)

        synthetic_metrics = evaluate_loader(model, loaders["test_synth"], device, pricing_config)
        real_metrics = evaluate_loader(model, loaders["test_real"], device, pricing_config)
        save_completed_run(
            run_dir,
            mode=args.mode,
            model=model,
            stage_logs=stage_logs,
            config=config,
            counts=counts,
            synthetic_metrics=synthetic_metrics,
            real_metrics=real_metrics,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "run_dir": str(run_dir),
                    "checkpoint_file": str(checkpoint_path(run_dir)),
                    "marker_file": str(marker_path(run_dir)),
                    "resume_command_hint": f"python3 train_double_heston.py --mode {args.mode} --run-dir '{run_dir}' --continuous (real-stage resume additionally requires {ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG})",
                },
                indent=2,
            )
        )
        return

    print(
        json.dumps(
            {
                "mode": args.mode,
                "run_dir": str(run_dir),
                "checkpoint_file": str(checkpoint_path(run_dir)),
                "marker_file": str(marker_path(run_dir)),
                "metrics_file": str(metrics_path(run_dir)),
                "synthetic_metrics": synthetic_metrics,
                "real_metrics": real_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
