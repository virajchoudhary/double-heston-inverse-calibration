"""Run the Mentor Double Heston forward PINN training stage (no test use)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mentor_dh_pinn.config import load_baseline_config
from src.mentor_dh_pinn.model import DoubleHestonForwardPINN
from src.mentor_dh_pinn.synthetic_data import (
    generate_synthetic_dataset,
    load_synthetic_dataset,
)
from src.mentor_dh_pinn.trainer import seed_everything, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mentor_dh_pinn_baseline_v1"),
    )
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="reuse an existing manifest/dataset in output-dir",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a small deterministic 2-epoch development run",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _smoke_or_explicit_config(args: argparse.Namespace):
    config = load_baseline_config(args.config)
    overrides: dict[str, int] = {}
    if args.smoke:
        overrides.update(train_count=32, validation_count=8, test_count=8, max_epochs=2, patience=2)
    if args.train_count is not None:
        overrides["train_count"] = args.train_count
    if args.validation_count is not None:
        overrides["validation_count"] = args.validation_count
    if args.test_count is not None:
        overrides["test_count"] = args.test_count
    if args.epochs is not None:
        overrides["max_epochs"] = args.epochs
    if args.patience is not None:
        overrides["patience"] = args.patience
    return config.with_overrides(**overrides) if overrides else config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _smoke_or_explicit_config(args)
    output_dir = args.output_dir
    if args.skip_data_generation:
        dataset = load_synthetic_dataset(output_dir, config=config)
    else:
        dataset = generate_synthetic_dataset(output_dir, config=config)
    # Seed before module construction so initial weights are part of the
    # reproducible V1 run, not merely the optimizer/collocation streams.
    seed_everything(config.seed)
    model = DoubleHestonForwardPINN(
        feature_min=config.domain.feature_min,
        feature_max=config.domain.feature_max,
    )
    result = train_baseline(model, dataset, output_dir, config=config, device=args.device)
    payload = result.as_dict()
    payload["network_parameter_count"] = model.parameter_count
    payload["dataset_sha256"] = dataset.manifest.get("dataset_sha256")
    with (output_dir / "run_result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
