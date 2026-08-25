"""Emit independent frozen Stage-B launch commands; never execute them."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPO_ROOT / "scripts" / "run_model3_pde_pilot.py"
FROZEN_SHARED = {
    "dataset": REPO_ROOT / "data" / "final_r2_clean_10000" / "surfaces.jsonl",
    "train_limit": 7500,
    "validation_limit": 1250,
    "epochs": 120,
    "batch_size": 32,
    "interior_points": 32,
    "terminal_points": 8,
    "learning_rate": 0.0002,
    "weight_decay": 0.00001,
    "device": "cuda",
    "patience": 15,
    "run_kind": "MODEL3_STAGE_B_RESEARCH_FROZEN",
}
SEEDS = (11, 22, 33)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_driver_module():
    spec = importlib.util.spec_from_file_location("model3_stage_b_driver", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Model 3 driver could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package(seed: int) -> dict[str, Any]:
    driver = load_driver_module()
    output_root = REPO_ROOT / "outputs" / f"model3_stage_b_seed_{seed}"
    settings = driver.PilotSettings(output_root=output_root, seed=seed, **FROZEN_SHARED)
    portable_settings = {
        **asdict(settings),
        "dataset": "data/final_r2_clean_10000/surfaces.jsonl",
        "output_root": output_root.relative_to(REPO_ROOT).as_posix(),
    }
    arguments = [
        "PYTHONPATH=.",
        "python",
        "scripts/run_model3_pde_pilot.py",
        "--dataset",
        "data/final_r2_clean_10000/surfaces.jsonl",
        "--output-root",
        output_root.relative_to(REPO_ROOT).as_posix(),
        "--train-limit",
        str(FROZEN_SHARED["train_limit"]),
        "--validation-limit",
        str(FROZEN_SHARED["validation_limit"]),
        "--seed",
        str(seed),
        "--epochs",
        str(FROZEN_SHARED["epochs"]),
        "--batch-size",
        str(FROZEN_SHARED["batch_size"]),
        "--interior-points",
        str(FROZEN_SHARED["interior_points"]),
        "--terminal-points",
        str(FROZEN_SHARED["terminal_points"]),
        "--learning-rate",
        str(FROZEN_SHARED["learning_rate"]),
        "--weight-decay",
        str(FROZEN_SHARED["weight_decay"]),
        "--device",
        FROZEN_SHARED["device"],
        "--patience",
        str(FROZEN_SHARED["patience"]),
        "--run-kind",
        FROZEN_SHARED["run_kind"],
    ]
    return {
        "schema": "MODEL3_STAGE_B_SEED_PACKAGE_V1",
        "execution_status": "PREPARED_NOT_EXECUTED",
        "seed": seed,
        "output_root": output_root.relative_to(REPO_ROOT).as_posix(),
        "settings": portable_settings,
        "launch_command": arguments,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON path; never overwrite")
    arguments = parser.parse_args(argv)
    packages = [package(seed) for seed in SEEDS]
    payload = {
        "schema": "MODEL3_STAGE_B_THREE_SEED_PACKAGES_V1",
        "status": "PREPARED_NOT_EXECUTED",
        "test_split": "CLOSED",
        "packages": packages,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
