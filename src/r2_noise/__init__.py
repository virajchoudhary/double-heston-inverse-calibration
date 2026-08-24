"""Frozen observation-noise robustness machinery for the R2 primary methods.

Implements exactly the contract frozen in
``configs/r2_noise_robustness_FINAL.yaml`` /
``docs/R2_NOISE_ROBUSTNESS_PROTOCOL.md``.  No research result is produced by
this package; it provides deterministic derivation and selection primitives
whose behaviour is pinned by ``tests/test_r2_noise_contract.py``.
"""
