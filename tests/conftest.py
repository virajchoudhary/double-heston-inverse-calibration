"""Put the single-Heston sources on sys.path for the test suite.

Without this, `pytest tests` aborts at COLLECTION: several test modules do a bare
top-level `import pinn_heston_core` / `from single_heston import ...` because they were
written when those files sat at the root of the single-Heston working directory. A
collection error stops the whole run, not just the affected module, so every other test
in the suite silently fails to execute.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("single_heston_pinn/src", "single_heston_pinn/research"):
    p = ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
