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


def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    text = read(path)
    check_ascii(path, text)
    entries: dict[str, dict[str, str]] = {}
    matches = list(re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    if not matches:
        fail("no BibTeX entries found")
    for index, match in enumerate(matches):
        key = match.group(2)
        if key in entries:
            fail(f"duplicate BibTeX key: {key}")
        start = match.end()
        next_entry = re.search(r"\n(?=@\w+\s*\{)", text[start:])
        end = start + next_entry.start() if next_entry else len(text)
        if end < 0:
            end = len(text)
        body = text[start:end]
        depth = 0
        for char in body:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        # The delimiter that ends this BibTeX entry is intentionally excluded.
        if depth != -1:
            fail(f"unbalanced braces in BibTeX entry {key}: net depth {depth}")
        fields = {}
        for field_match in re.finditer(r"^\s*(\w+)\s*=\s*(.+?),?\s*$", body, flags=re.MULTILINE):
            fields[field_match.group(1).lower()] = field_match.group(2).strip()
        entries[key] = {"type": match.group(1).lower(), **fields}
    return entries


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for command in ["citep", "citet", "citealp", "citeauthor", "citeyear", "cite"]:
        for match in re.findall(rf"[\\]{command}\{{([^}}]+)\}}", text):
            keys.update(item.strip() for item in match.split(","))
    return keys


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

    bib_path = PAPER / "references.bib"
    bib = parse_bibtex(bib_path)
    literature_inventory = read(PAPER / "LITERATURE_INVENTORY.md")
    check_ascii(PAPER / "LITERATURE_INVENTORY.md", literature_inventory)

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
    cited = set()
    for path, text in all_text.items():
        inputs |= referenced_inputs(text)
        figures |= referenced_figures(text)
        labels |= defined_labels(text)
        refs |= referenced_labels(text)
        cited |= citation_keys(text)

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

    unresolved = cited - set(bib)
    if unresolved:
        fail(f"undefined BibTeX keys: {sorted(unresolved)}")
    uncited = set(bib) - cited
    if uncited:
        fail(f"BibTeX entries not cited in manuscript: {sorted(uncited)}")
    for key, entry in bib.items():
        for required_field in ["title", "author", "year"]:
            if required_field not in entry:
                fail(f"BibTeX entry {key} lacks verified metadata field: {required_field}")
        venue_present = any(field in entry for field in ["journal", "booktitle", "howpublished"])
        identifier_present = any(field in entry for field in ["doi", "url", "eid"])
        if not venue_present or not identifier_present:
            fail(f"BibTeX entry {key} lacks source/venue or stable identifier")
        if key not in literature_inventory:
            fail(f"BibTeX key missing from LITERATURE_INVENTORY.md: {key}")

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
    if "model3_protocol_commit=f34a4d334f1703ed6e31fa60782c43ac7290ef3d" not in sources:
        fail("generated asset provenance does not pin current Model3 readiness commit")

    related_work = all_text[PAPER / "sections" / "02_motivation_and_related_work.tex"]
    if "\\placeholder{" in related_work or "without citations" in related_work.lower():
        fail("literature placeholders remain in the related-work section")
    combined_paper_text = "\n".join(all_text.values())
    claim_pattern = re.compile(r"\b(this is the first|novel|state-of-the-art|outperforms?)\b", re.IGNORECASE)
    unsupported_claims = None
    for path, text in all_text.items():
        match = claim_pattern.search(text)
        if match:
            unsupported_claims = match
            break
    if unsupported_claims:
        fail(f"unsupported claim language found: {unsupported_claims.group(0)}")
    pending_boundaries = [
        "No Model3 training result exists",
        "No OOD result is claimed",
        "No G8 result is claimed",
        "Positive-level traditional subset calibration remains pending",
    ]
    for boundary in pending_boundaries:
        if boundary.lower() not in combined_paper_text.lower():
            fail(f"pending-result boundary missing: {boundary}")
    if re.search(r"\bModel\s*2\s+(?:is|as)\s+(?:a|the)\s+PINN\b", combined_paper_text, flags=re.IGNORECASE):
        fail("Model2 is incorrectly described as a PINN")

    print("PAPER_VALIDATION_PASSED")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
