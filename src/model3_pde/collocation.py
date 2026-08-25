"""Deterministic state-domain sampling for lightweight PDE checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .operator import PDEState


@dataclass(frozen=True)
class CollocationDomain:
    """Finite interior state box used by diagnostics and pilot sampling."""

    spot_minimum: float
    spot_maximum: float
    variance_slow_minimum: float
    variance_slow_maximum: float
    variance_fast_minimum: float
    variance_fast_maximum: float
    maturity_minimum: float
    maturity_maximum: float

    def __post_init__(self) -> None:
        pairs = (
            ("spot", self.spot_minimum, self.spot_maximum),
            ("slow variance", self.variance_slow_minimum, self.variance_slow_maximum),
            ("fast variance", self.variance_fast_minimum, self.variance_fast_maximum),
            ("maturity", self.maturity_minimum, self.maturity_maximum),
        )
        for name, minimum, maximum in pairs:
            if not minimum < maximum or minimum <= 0:
                raise ValueError(f"{name} domain must satisfy 0 < minimum < maximum")

    def sample(self, point_count: int, generator: torch.Generator) -> PDEState:
        if point_count <= 0:
            raise ValueError("point_count must be strictly positive")

        def uniform(minimum: float, maximum: float) -> torch.Tensor:
            draw = torch.rand((point_count,), generator=generator, dtype=torch.float64)
            return minimum + (maximum - minimum) * draw

        spot = uniform(self.spot_minimum, self.spot_maximum).requires_grad_(True)
        variance_slow = uniform(
            self.variance_slow_minimum, self.variance_slow_maximum
        ).requires_grad_(True)
        variance_fast = uniform(
            self.variance_fast_minimum, self.variance_fast_maximum
        ).requires_grad_(True)
        maturity = uniform(self.maturity_minimum, self.maturity_maximum).requires_grad_(True)
        return PDEState(
            spot=spot,
            variance_slow=variance_slow,
            variance_fast=variance_fast,
            maturity=maturity,
        )


def sample_collocation_states(
    domain: CollocationDomain,
    *,
    point_count: int,
    seed: int,
) -> tuple[PDEState, torch.Generator]:
    """Return deterministic float64 states and the generator used to make them."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return domain.sample(point_count, generator), generator


def sample_eligible_contract_slot_indices(
    contract_masks: torch.Tensor,
    *,
    points_per_surface: int,
    seed: int,
) -> torch.Tensor:
    """Draw canonical observed-slot indices independently for every point."""
    if contract_masks.ndim != 2 or contract_masks.dtype != torch.bool:
        raise ValueError("contract_masks must be a boolean (surface, slot) matrix")
    if points_per_surface <= 0:
        raise ValueError("points_per_surface must be strictly positive")
    counts = contract_masks.cpu().sum(dim=1)
    if bool(torch.any(counts == 0)):
        raise ValueError("every surface requires at least one observed contract slot")
    surface_count = contract_masks.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.rand((surface_count, points_per_surface), generator=generator, dtype=torch.float64)
    local_indices = (draws * counts.unsqueeze(1)).to(dtype=torch.int64)
    local_indices = torch.minimum(local_indices, (counts - 1).unsqueeze(1))
    ordered_observed_slots = torch.argsort(
        (~contract_masks.cpu()).to(dtype=torch.uint8), dim=1, stable=True
    )
    row_indices = torch.arange(surface_count, dtype=torch.int64).unsqueeze(1)
    return ordered_observed_slots[row_indices, local_indices]


def sample_conditioned_collocation_states(
    *,
    observed_spots: torch.Tensor,
    theta_slow: torch.Tensor,
    theta_fast: torch.Tensor,
    contract_masks: torch.Tensor,
    variance_slow_ceiling: float,
    variance_fast_ceiling: float,
    points_per_surface: int,
    seed: int,
) -> tuple[PDEState, torch.Tensor, torch.Tensor]:
    """Sample a deterministic interior box around each observed surface.

    Variance lower/upper multiples follow the frozen protocol.  The result is
    flattened surface-major.  ``source_indices`` maps every point back to its
    surface, while ``contract_slot_indices`` maps that point to one eligible
    observed ``(surface, slot)`` contract in canonical R2 order.
    """
    if observed_spots.ndim != 1 or theta_slow.shape != observed_spots.shape or theta_fast.shape != observed_spots.shape:
        raise ValueError("observed_spots, theta_slow, and theta_fast must have equal one-dimensional shapes")
    if observed_spots.dtype != torch.float64 or theta_slow.dtype != torch.float64 or theta_fast.dtype != torch.float64:
        raise TypeError("surface conditioning must be float64")
    if points_per_surface <= 0:
        raise ValueError("points_per_surface must be strictly positive")
    if variance_slow_ceiling <= 0 or variance_fast_ceiling <= 0:
        raise ValueError("variance ceilings must be strictly positive")
    if not torch.isfinite(observed_spots).all() or bool(torch.any(observed_spots <= 0)):
        raise ValueError("observed spots must be finite and strictly positive")
    if (
        not torch.isfinite(theta_slow).all()
        or not torch.isfinite(theta_fast).all()
        or bool(torch.any(theta_slow <= 0))
        or bool(torch.any(theta_fast <= 0))
    ):
        raise ValueError("theta values must be finite and strictly positive")

    surface_count = observed_spots.shape[0]
    spots_cpu = observed_spots.detach().cpu()
    theta_slow_cpu = theta_slow.detach().cpu()
    theta_fast_cpu = theta_fast.detach().cpu()
    contract_masks_cpu = contract_masks.detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        draw = torch.rand((surface_count, points_per_surface), generator=generator, dtype=torch.float64)
        return (minimum.unsqueeze(1) + (maximum - minimum).unsqueeze(1) * draw).reshape(-1)

    slow_minimum = 0.05 * theta_slow_cpu
    slow_maximum = torch.minimum(
        2.0 * theta_slow_cpu,
        torch.full_like(theta_slow_cpu, variance_slow_ceiling),
    )
    fast_minimum = 0.05 * theta_fast_cpu
    fast_maximum = torch.minimum(
        2.0 * theta_fast_cpu,
        torch.full_like(theta_fast_cpu, variance_fast_ceiling),
    )
    if bool(torch.any(slow_maximum <= slow_minimum)) or bool(torch.any(fast_maximum <= fast_minimum)):
        raise ValueError("conditioned variance domains must have positive width")

    spot_minimum = 0.5 * spots_cpu
    spot_maximum = 1.5 * spots_cpu
    maturity_minimum = torch.full_like(spots_cpu, 7.0 / 365.0)
    maturity_maximum = torch.full_like(spots_cpu, 180.0 / 365.0)
    state = PDEState(
        spot=uniform(spot_minimum, spot_maximum).requires_grad_(True),
        variance_slow=uniform(slow_minimum, slow_maximum).requires_grad_(True),
        variance_fast=uniform(fast_minimum, fast_maximum).requires_grad_(True),
        maturity=uniform(maturity_minimum, maturity_maximum).requires_grad_(True),
    )
    source_indices = torch.arange(surface_count, dtype=torch.int64).repeat_interleave(points_per_surface)
    contract_slot_indices = sample_eligible_contract_slot_indices(
        contract_masks_cpu,
        points_per_surface=points_per_surface,
        seed=seed + 6151,
    ).reshape(-1)
    return state, source_indices, contract_slot_indices
