"""Sealed post-training evaluation and evidence-intake layer for Model 3."""

from .adapter import Model3EvaluationAdapter
from .contracts import (
    EXPECTED_DATASET_SHA256,
    REQUIRED_SEEDS,
    build_seed_contract,
    verify_freeze_manifest,
)

__all__ = [
    "EXPECTED_DATASET_SHA256",
    "Model3EvaluationAdapter",
    "REQUIRED_SEEDS",
    "build_seed_contract",
    "verify_freeze_manifest",
]
