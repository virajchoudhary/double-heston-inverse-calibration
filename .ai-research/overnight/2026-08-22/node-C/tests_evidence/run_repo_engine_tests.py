"""Run the repository's own canonical-engine tests under Python 3.9.

The audit machine's torch-capable interpreter is 3.9; the repo targets 3.10+
(`typing.TypeAlias`, PEP 604 alias values, `zip(..., strict=True)`). This
runner loads `src/double_heston.py` with three typing-only replacements and
one `zip(strict=)` removal (semantics-preserving for the equal-length inputs
the tests provide), provides a minimal pytest shim (approx / mark.parametrize
by name / raises with match), and executes
`tests/test_double_heston_engine.py` unmodified. Cross-interpreter
reproducibility datapoint against Node A's >=3.10 pytest run.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

_patch_sites = 0


def load_patched_src_module(module_name: str, filename: str):
    """Load a src/ module with 3.10-isms neutralised: typing-only alias lines
    and `zip(..., strict=True)` (semantics-preserving for the equal-length
    inputs involved). All pricing mathematics executes unmodified."""
    global _patch_sites
    source = (SRC_ROOT / filename).read_text()
    for old, new in (
        ("from typing import TypeAlias", "TypeAlias = object"),
        ("ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]", "ParameterInput = None"),
        ("ComplexResult: TypeAlias = complex | np.ndarray", "ComplexResult = None"),
    ):
        if source.count(old):
            source = source.replace(old, new)
    _patch_sites += source.count(", strict=True)")
    source = source.replace(", strict=True)", ")")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(module_name, SRC_ROOT / filename)
    )
    module.__dict__["__package__"] = "src"
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


import src  # noqa: E402

load_patched_src_module("src.double_heston", "double_heston.py")
load_patched_src_module("src.pricing_interface", "pricing_interface.py")


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.abs or 0.0, abs(self.expected) * (self.rel or 0.0)) + 1e-300

    def __req__(self, other):
        return self.__eq__(other)


class _RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected = expected_exception
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"{self.expected.__name__} not raised")
        if not issubclass(exc_type, self.expected):
            return False  # propagate unexpected exception
        if self.match is not None:
            assert re.search(self.match, str(exc_value)), f"pattern {self.match!r} not found in {exc_value!r}"
        self.value = exc_value
        return True


class _Mark:
    def parametrize(self, argnames, argvalues):
        if isinstance(argnames, (list, tuple)):
            names = [str(a) for a in argnames]
        else:
            names = [a.strip() for a in argnames.split(",")]
        values = [v if isinstance(v, tuple) else (v,) for v in argvalues]

        def decorator(fn):
            existing = getattr(fn, "_node_c_params", None)
            if existing is None:
                fn._node_c_params = (names, values)
            else:
                old_names, old_values = existing
                fn._node_c_params = (old_names + names, [o + n for o in old_values for n in values])
            return fn

        return decorator

    def __getattr__(self, name):
        def decorator(*args, **kwargs):
            return (args[0] if args and callable(args[0]) else (lambda f: f))

        return decorator


class _PytestShim:
    approx = staticmethod(lambda expected, rel=None, abs=None: _Approx(expected, rel, abs))

    @staticmethod
    @contextmanager
    def raises(expected_exception, match=None):
        ctx = _RaisesContext(expected_exception, match)
        with ctx:
            yield ctx

    mark = _Mark()


sys.modules.setdefault("pytest", _PytestShim())

spec = importlib.util.spec_from_file_location(
    "repo_engine_tests", PROJECT_ROOT / "tests" / "test_double_heston_engine.py"
)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

passed = failed = 0
failures = []
for name in sorted(dir(test_module)):
    if not name.startswith("test_"):
        continue
    fn = getattr(test_module, name)
    cases = getattr(fn, "_node_c_params", None)
    if cases is None:
        case_list = [{}]
    else:
        names, values = cases
        case_list = [dict(zip(names, v)) for v in values]
    for case in case_list:
        label = f"{name}{list(case.values()) if case else ''}"
        try:
            fn(**case)
            passed += 1
            print(f"PASS {label}")
        except Exception:
            failed += 1
            failures.append(label)
            print(f"FAIL {label}")
            traceback.print_exc(limit=2)

print(f"\nzip(strict=) sites patched: {_patch_sites}")
print(f"RESULT: {passed}/{passed + failed} repo engine tests passed on py3.9 + documented shim; failures: {failures}")
sys.exit(1 if failed else 0)
