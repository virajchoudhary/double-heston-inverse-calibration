"""Export publication tables only from a sealed COMPLETE Model3 result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model3_evaluation.paper_export import export_publication_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root")
    parser.add_argument("output_root")
    arguments = parser.parse_args(argv)
    paths = export_publication_tables(arguments.result_root, arguments.output_root)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
