"""PyTorch dataset for one fixed-length vector per complete option surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import CALL_OPTION, NOT_RESEARCH_DATA, PARAMETER_COUNT, PARAMETER_NAMES, PUT_OPTION
from .surface_grid import expected_input_size


class SurfaceParameterDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]]):
    """Fixed-size features with ten targets and preserved surface identity."""

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        surface_ids: Sequence[str],
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.surface_ids = [str(value) for value in surface_ids]
        self.metadata = list(metadata or [{} for _ in self.surface_ids])
        if self.features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if self.targets.shape != (len(self.features), PARAMETER_COUNT):
            raise ValueError(
                f"targets must have shape ({len(self.features)}, {PARAMETER_COUNT})"
            )
        if len(self.surface_ids) != len(self.features) or len(self.metadata) != len(
            self.features
        ):
            raise ValueError("features, targets, surface_ids, and metadata lengths differ")
        if len(set(self.surface_ids)) != len(self.surface_ids):
            raise ValueError("Each dataset item must have a unique surface_id")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.targets).all():
            raise ValueError("features and targets must be finite")

    def __len__(self) -> int:
        return len(self.surface_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]:
        return (
            self.features[index],
            self.targets[index],
            self.surface_ids[index],
            self.metadata[index],
        )

    @classmethod
    def from_surface_frame(
        cls,
        frame: pd.DataFrame,
        allow_not_research_data: bool = False,
    ) -> "SurfaceParameterDataset":
        """Group complete rows without allowing a surface to cross splits."""
        required = {
            "surface_id",
            "split",
            "option_type",
            "maturity_days",
            "log_moneyness",
            "normalized_price",
            "mask",
            "data_status",
            *PARAMETER_NAMES,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Surface frame is missing columns: {missing}")
        statuses = set(frame["data_status"].astype(str))
        if NOT_RESEARCH_DATA in statuses and not allow_not_research_data:
            raise ValueError(
                "NOT_RESEARCH_DATA cannot be loaded without allow_not_research_data=True"
            )
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        surface_ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        option_order = pd.CategoricalDtype([CALL_OPTION, PUT_OPTION], ordered=True)
        for surface_id, group in frame.groupby("surface_id", sort=True):
            if group["split"].nunique() != 1:
                raise ValueError(f"Surface {surface_id} appears in multiple splits")
            ordered = group.assign(
                option_type=group["option_type"].astype(option_order)
            ).sort_values(["option_type", "maturity_days", "log_moneyness"])
            if len(ordered) != expected_input_size():
                raise ValueError(
                    f"Surface {surface_id} has {len(ordered)} rows, expected {expected_input_size()}"
                )
            mask = ordered["mask"].astype(bool).to_numpy()
            feature = ordered["normalized_price"].to_numpy(dtype=np.float32)
            feature = np.where(mask, feature, 0.0)
            target_rows = ordered[PARAMETER_NAMES].drop_duplicates()
            if len(target_rows) != 1:
                raise ValueError(f"Surface {surface_id} has inconsistent target parameters")
            features.append(feature)
            targets.append(target_rows.iloc[0].to_numpy(dtype=np.float32))
            surface_ids.append(str(surface_id))
            metadata.append(
                {
                    "split": str(ordered["split"].iloc[0]),
                    "data_status": str(ordered["data_status"].iloc[0]),
                    "spot": float(ordered["spot"].iloc[0]),
                    "generation_seed": int(ordered["generation_seed"].iloc[0]),
                    "noise_level": float(ordered["noise_level"].iloc[0]),
                }
            )
        return cls(np.stack(features), np.stack(targets), surface_ids, metadata)

    def indices_for_split(self, split: str) -> list[int]:
        """Return indices for exactly one stored surface-level split label."""
        return [
            index
            for index, item in enumerate(self.metadata)
            if item.get("split") == split
        ]
