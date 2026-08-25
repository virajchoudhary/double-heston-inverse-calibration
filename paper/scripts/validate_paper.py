#!/usr/bin/env python3
"""Lightweight structural and provenance checks for the synthesis layer."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent

REQUIRED_SECTIONS = [
    "00_abstract",
    "01_introduction",
    "02_motivation_and_related_work",
    "03_model",
    "04_inverse_problem",
    "05_methodology",
    "06_r2_representation",
    "07_dataset_construction",
    "08_methods_comparison",
    "09_model3_protocol",
    "10_experimental_design",
    "11_completed_results",
    "12_identifiability_discussion",
    "13_limitations_and_pending_work",
    "14_conclusion",
    "90_appendix",
]

REQUIRED_TABLES = {
    "canonical_engine_benchmark",
    "evidence_classification",
    "primary_comparison",
    "real_market_development_comparison",
    "r2_representation_selection",
    "completed_neural_noise",
}

REQUIRED_FIGURES = {
    "speed_accuracy_tradeoff.pdf",
    "fit_is_not_recovery.pdf",
    "completed_neural_noise.pdf",
}

ENVIRONMENTS = ["document", "abstract", "table", "tabular", "figure", "equation", "align"]


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing file {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def check_ascii(path: Path, text: str) -> None:
    non_ascii = [(index + 1, char) for index, char in enumerate(text) if ord(char) > 127]
    if non_ascii:
        line = text.count("\n", 0, non_ascii[0][0]) + 1
        fail(f"non-ASCII character in {path.relative_to(ROOT)} near line {line}")


def check_balanced_environments(text: str, path: Path) -> None:
    for env in ENVIRONMENTS:
        begins = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", text))
        ends = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", text))
        if begins != ends:
            fail(
                f"unbalanced {env} environments in {path.relative_to(ROOT)}: "
                f"{begins} begin versus {ends} end"
            )


def referenced_inputs(text: str) -> set[str]:
    values = set(re.findall(r"\\input\{([^}]+)\}", text))
    return {value if value.endswith(".tex") else value + ".tex" for value in values}


def referenced_figures(text: str) -> set[str]:
    return set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text))


def referenced_labels(text: str) -> set[str]:
    return set(re.findall(r"\\ref\{([^}]+)\}", text))


def defined_labels(text: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def main() -> None:
    main_tex_path = PAPER / "main.tex"
    main_tex = read(main_tex_path)
    check_ascii(main_tex_path, main_tex)
    check_balanced_environments(main_tex, main_tex_path)

    for section in REQUIRED_SECTIONS:
        path = PAPER / "sections" / f"{section}.tex"
        text = read(path)
        check_ascii(path, text)
        check_balanced_environments(text, path)
        if f"\\input{{sections/{section}}}" not in main_tex:
            fail(f"required section not included in main.tex: {section}")

    for name in REQUIRED_TABLES:
        path = PAPER / "generated" / "tables" / f"{name}.tex"
        text = read(path)
        check_ascii(path, text)
        check_balanced_environments(text, path)

    for name in REQUIRED_FIGURES:
        path = PAPER / "generated" / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing or empty generated figure {path.relative_to(ROOT)}")

    inventory = read(PAPER / "RESULTS_INVENTORY.md")
    check_ascii(PAPER / "RESULTS_INVENTORY.md", inventory)
    for name in REQUIRED_TABLES | REQUIRED_FIGURES:
        if name not in inventory:
            fail(f"asset missing from RESULTS_INVENTORY.md: {name}")

    all_tex_paths = [main_tex_path] + list((PAPER / "sections").glob("*.tex"))
    all_tex_paths.extend((PAPER / "generated" / "tables").glob("*.tex"))
    all_text = {path: read(path) for path in all_tex_paths}
    for path, text in all_text.items():
        check_ascii(path, text)
        check_balanced_environments(text, path)

    inputs = set()
    figures = set()
    labels = set()
    refs = set()
    for path, text in all_text.items():
        inputs |= referenced_inputs(text)
        figures |= referenced_figures(text)
        labels |= defined_labels(text)
        refs |= referenced_labels(text)

    for relative in inputs:
        if not (PAPER / relative).is_file():
            fail(f"LaTeX input does not exist: {relative}")
    for relative in figures:
        if not (PAPER / "figures" / relative).is_file():
            fail(f"LaTeX figure does not exist: {relative}")
    undefined = refs - labels
    if undefined:
        fail(f"undefined LaTeX references: {sorted(undefined)}")
    duplicates = len(labels) - len(set(labels))
    if duplicates != 0:
        fail("duplicate labels detected")

    for commit in [
        "72ad8e1aa845ec4c6f0fc61fc526df75438639bb",
        "e58b36d1ff6cbf39115d46698ad0fb45b1ac1b8b",
        "02c2a2cbc2498d5c4ce1e914e7c3d22693a55fc9",
    ]:
        if commit not in inventory:
            fail(f"pinned commit missing from RESULTS_INVENTORY.md: {commit}")

    sources = read(PAPER / "generated" / "asset_sources.txt")
    if "noise_commit=e58b36d1ff6cbf39115d46698ad0fb45b1ac1b8b" not in sources:
        fail("generated asset provenance does not pin the noise branch commit")

    tex_sources = all_tex_paths
    for path in tex_sources:
        text = all_text[path]
        if re.search(r"\\cite\{[^}]+\}", text):
            fail(f"citations must remain placeholder-only in this foundation: {path}")

    print("PAPER_VALIDATION_PASSED")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
