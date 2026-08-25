"""Lightweight, pre-training foundation for genuine PDE-informed Model 3."""

from .collocation import (
    CollocationDomain,
    sample_collocation_states,
    sample_conditioned_collocation_states,
    sample_eligible_contract_slot_indices,
)
from .losses import (
    arbitrage_boundary_loss,
    masked_normalized_price_loss,
    pde_residual_loss,
    terminal_payoff_loss,
)
from .model import ConditionalDoubleHestonPriceNetwork, Model3PDESystem
from .operator import PDEState, double_heston_pde_residual

__all__ = [
    "ConditionalDoubleHestonPriceNetwork",
    "CollocationDomain",
    "Model3PDESystem",
    "PDEState",
    "arbitrage_boundary_loss",
    "double_heston_pde_residual",
    "masked_normalized_price_loss",
    "pde_residual_loss",
    "sample_collocation_states",
    "sample_conditioned_collocation_states",
    "sample_eligible_contract_slot_indices",
    "terminal_payoff_loss",
]
