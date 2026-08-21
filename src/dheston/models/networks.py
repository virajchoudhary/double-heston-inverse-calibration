from __future__ import annotations

import torch
from torch import nn

from dheston.calibration.transforms import constrain_parameter_tensor


class DeepSurfaceInverseModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.point_encoder(features)
        expanded_mask = mask.unsqueeze(-1)
        encoded = encoded * expanded_mask
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled_mean = encoded.sum(dim=1) / lengths
        pooled_max = encoded.masked_fill(~expanded_mask, -1e9).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        summary = torch.cat([pooled_mean, pooled_max], dim=-1)
        raw = self.head(summary)
        params = constrain_parameter_tensor(raw)
        return {"raw": raw, "params": params}

