"""Fail-closed quarantine policy for real-market neural weight updates (Archive-2).

Canonical research policy (docs/OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md,
section 5): primary model weight learning is SYNTHETIC-ONLY, and real market
observations are reserved for frozen-model evaluation. Real-market fine-tuning
of neural-network weights is NONCANONICAL and EXPERIMENTAL behavior: it is
disabled by default and is reachable only through an explicit, unmistakable
opt-in (``--allow-noncanonical-real-weight-updates``). It is not part of the
canonical research baseline.

This module is Archive-2's own policy boundary. Canonical code must not import
it or depend on it.
"""

from __future__ import annotations

ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG = "--allow-noncanonical-real-weight-updates"

POLICY_REFERENCE = "docs/OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md (section 5)"


class RealMarketWeightUpdateQuarantineError(RuntimeError):
    """Raised when normal execution would update neural weights on real market data."""


def quarantine_error_message(config_real_epochs: int, continuous_requested: bool) -> str:
    triggers = []
    if config_real_epochs > 0:
        triggers.append(f"training.real_epochs = {config_real_epochs} in the resolved config")
    if continuous_requested:
        triggers.append("--continuous (real-market re-entry / continuous training request)")
    return "\n".join(
        [
            "REAL-MARKET NEURAL WEIGHT UPDATES ARE QUARANTINED (fail-closed).",
            "",
            "This invocation would fine-tune neural-network weights on real market",
            f"observations. Triggered by: {'; '.join(triggers)}.",
            "",
            "Such real-market weight updating is NONCANONICAL and EXPERIMENTAL.",
            "Canonical policy: primary weight learning is SYNTHETIC-ONLY and real",
            "market data is used for frozen-model evaluation only",
            f"({POLICY_REFERENCE}).",
            "",
            "To run this noncanonical experiment anyway, pass the explicit opt-in:",
            f"    {ALLOW_NONCANONICAL_REAL_WEIGHT_UPDATES_FLAG}",
            "The opt-in is DISABLED BY DEFAULT and is not part of the research baseline.",
        ]
    )


def resolve_real_market_epochs(
    *,
    config_real_epochs: int,
    continuous_requested: bool,
    allow_noncanonical_real_weight_updates: bool,
    continuous_epoch_limit: int,
) -> int:
    """Return the effective real fine-tune epoch count under the fail-closed policy.

    Matrix (``allow`` = explicit noncanonical opt-in flag):

    - config real_epochs > 0 without the opt-in -> RealMarketWeightUpdateQuarantineError
      (raised before any dataloader, model, optimizer, or training loop runs);
    - ``--continuous`` alone never authorizes real-market weight updates: with
      real_epochs == 0 and no opt-in it resolves to 0 (harmless auto-resume
      orchestration only);
    - opt-in + ``--continuous`` -> continuous_epoch_limit (historical continuous
      real training, now double-opt-in);
    - opt-in without ``--continuous`` -> config_real_epochs (0 stays 0: the flag
      authorizes but does not itself request real training).
    """
    if config_real_epochs < 0:
        raise ValueError("config_real_epochs must be >= 0")
    if config_real_epochs > 0 and not allow_noncanonical_real_weight_updates:
        raise RealMarketWeightUpdateQuarantineError(
            quarantine_error_message(config_real_epochs, continuous_requested)
        )
    if allow_noncanonical_real_weight_updates and continuous_requested:
        return int(continuous_epoch_limit)
    if allow_noncanonical_real_weight_updates:
        return int(config_real_epochs)
    return 0
