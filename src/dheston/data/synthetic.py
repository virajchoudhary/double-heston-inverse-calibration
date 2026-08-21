from __future__ import annotations

import numpy as np

from dheston.calibration.transforms import sample_parameter_vector
from dheston.data.surfaces import SurfaceRecord
from dheston.pricing.heston import FourierConfig, price_double_heston_numpy


def build_synthetic_records(
    templates: list[SurfaceRecord],
    sample_count: int,
    noise_std: float,
    seed: int,
    pricing_config: FourierConfig,
) -> list[SurfaceRecord]:
    if not templates:
        raise ValueError("Cannot generate synthetic records from an empty template list.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(templates), size=sample_count)
    synthetic_records: list[SurfaceRecord] = []

    for template_index in indices:
        base_record = templates[int(template_index)]
        template = base_record.template
        parameters = sample_parameter_vector(rng)
        prices = price_double_heston_numpy(
            spot=np.full_like(template.strikes, template.spot, dtype=np.float64),
            strikes=template.strikes.astype(np.float64),
            tau=template.tau.astype(np.float64),
            rates=np.full_like(template.strikes, template.rate, dtype=np.float64),
            dividends=np.full_like(template.strikes, template.dividend, dtype=np.float64),
            is_call=template.is_call.astype(np.float64),
            parameters=parameters,
            config=pricing_config,
        )
        if noise_std > 0:
            prices = prices * (1.0 + rng.normal(0.0, noise_std, size=prices.shape))
        synthetic_records.append(
            SurfaceRecord(
                template=template,
                prices=np.maximum(prices, 1e-6).astype(np.float32),
                source="synthetic",
                target_params=parameters.astype(np.float32),
            )
        )

    return synthetic_records

