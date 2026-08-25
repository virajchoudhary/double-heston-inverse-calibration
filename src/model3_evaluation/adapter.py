"""Deterministic inference adapter for frozen Model 3 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.parameter_transform import TargetStandardizer
from src.model3_pde.model import Model3PDESystem
from src.r2_primary.dataset import R2PrimaryDataset


class Model3CheckpointError(ValueError):
    """A checkpoint cannot be safely loaded for inference."""


class Model3EvaluationAdapter:
    """Load one immutable best-validation Stage-B checkpoint for inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.expected_seed = int(expected_seed)
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise Model3CheckpointError("checkpoint payload must be a dictionary")
        metadata = payload.get("metadata", {})
        settings = metadata.get("settings", {})
        try:
            stored_seed = int(settings["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise Model3CheckpointError("checkpoint lacks a valid seed identity") from error
        if stored_seed != self.expected_seed:
            raise Model3CheckpointError(
                f"checkpoint seed {stored_seed} does not match requested {self.expected_seed}"
            )
        if metadata.get("run_kind") != "MODEL3_STAGE_B_RESEARCH_FROZEN":
            raise Model3CheckpointError("Stage-A or development checkpoints are rejected")
        if int(payload.get("completed_epoch", -1)) != int(payload.get("best_epoch", -2)):
            raise Model3CheckpointError("checkpoint is not an exported best-validation state")
        system = Model3PDESystem()
        try:
            system.load_state_dict(payload["model_state_dict"], strict=True)
        except (KeyError, TypeError, RuntimeError) as error:
            raise Model3CheckpointError(f"incompatible Model3 state dict: {error}") from error
        standardizer = TargetStandardizer()
        standardizer_state = payload.get("target_standardizer")
        if not isinstance(standardizer_state, dict) or set(standardizer_state) != {"mean", "scale"}:
            raise Model3CheckpointError("checkpoint lacks a fitted target standardizer")
        standardizer.mean = standardizer_state["mean"].detach().cpu()
        standardizer.scale = standardizer_state["scale"].detach().cpu()
        self.system = system.to(device=self.device)
        self.system.eval()
        self.standardizer = standardizer
        self.metadata = metadata

    @property
    def seed(self) -> int:
        return self.expected_seed

    @torch.inference_mode()
    def predict_parameters(
        self,
        dataset: R2PrimaryDataset,
        indices: list[int],
        *,
        seed_identity: int | None = None,
    ) -> np.ndarray:
        """Return physical parameters in canonical order for exact input rows."""
        if seed_identity is not None and seed_identity != self.expected_seed:
            raise Model3CheckpointError("prediction seed identity mismatch")
        if len(indices) != len(set(indices)) or any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError("indices must be unique and within the supplied development dataset")
        features = torch.as_tensor(
            np.stack([dataset.items[index].features for index in indices]),
            dtype=torch.float32,
            device=self.device,
        )
        standardized = self.system.predict_parameters(features)
        physical = self.standardizer.inverse_transform(standardized)
        output = physical.detach().cpu().to(dtype=torch.float64).numpy()
        if output.shape != (len(indices), 10) or not np.isfinite(output).all():
            raise Model3CheckpointError("Model3 produced invalid parameter predictions")
        return output

    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)
