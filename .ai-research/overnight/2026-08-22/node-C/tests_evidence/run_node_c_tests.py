"""Dependency-free runner for tests/test_node_c_pde_physics_audit.py.

The audit machine's torch-capable interpreter (3.9) has no pytest, and the
compute policy forbids environment mutation. This runner installs a minimal
pytest shim (approx + mark.parametrize bookkeeping), imports the real test
module, and executes every parametrised case, recording results as JSON.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
TEST_FILE = PROJECT_ROOT / "tests" / "test_node_c_pde_physics_audit.py"


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        tol = max(self.abs or 0.0, abs(self.expected) * (self.rel or 0.0))
        return abs(other - self.expected) <= tol + 1e-300

    def __req__(self, other):
        return self.__eq__(other)


class _Mark:
    def parametrize(self, argnames, argvalues):
        names = [a.strip() for a in argnames.split(",")]
        new_values = [v if isinstance(v, tuple) else (v,) for v in argvalues]

        def decorator(fn):
            existing = getattr(fn, "_node_c_params", None)
            if existing is None:
                fn._node_c_params = (names, new_values)
            else:
                # compose stacked decorators as the cross product, like pytest
                old_names, old_values = existing
                fn._node_c_params = (
                    old_names + names,
                    [old + new for old in old_values for new in new_values],
                )
            return fn

        return decorator


class _PytestShim:
    approx = staticmethod(lambda expected, rel=None, abs=None: _Approx(expected, rel, abs))
    mark = _Mark()
    staticmethod = staticmethod


sys.modules.setdefault("pytest", _PytestShim)

spec = importlib.util.spec_from_file_location("node_c_tests", TEST_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

results = []
failures = 0
for name in sorted(dir(module)):
    if not name.startswith("test_"):
        continue
    fn = getattr(module, name)
    cases = getattr(fn, "_node_c_params", None)
    if cases is None:
        case_list = [()]
    else:
        names, values = cases
        # pytest binds parametrize arguments by NAME; honor that here.
        case_list = [dict(zip(names, v if isinstance(v, tuple) else (v,))) for v in values]
    for case in case_list:
        label = f"{name}{'[' + ','.join(repr(c) for c in (case.values() if isinstance(case, dict) else case)) + ']' if case else ''}"
        try:
            fn(**case) if isinstance(case, dict) else fn(*case)
            results.append({"test": label, "status": "PASS"})
            print(f"PASS {label}")
        except Exception:
            failures += 1
            results.append({"test": label, "status": "FAIL", "error": traceback.format_exc(limit=3)})
            print(f"FAIL {label}")
            print(traceback.format_exc(limit=3))

summary = {"total": len(results), "passed": len(results) - failures, "failed": failures, "results": results}
out = Path(__file__).with_name("pytest_equivalent_results.json")
out.write_text(json.dumps(summary, indent=2))
print(f"\n{summary['passed']}/{summary['total']} passed -> {out}")
sys.exit(1 if failures else 0)
