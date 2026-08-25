#!/usr/bin/env python3
"""Generate paper tables and figures from committed completed evidence."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"
TABLES = GENERATED / "tables"
FIGURES = GENERATED / "figures"
LATEX_FIGURES = PAPER / "figures"

EVIDENCE = ROOT / "evidence" / "r2_primary_comparison_20260823"
BENCHMARK = ROOT / "outputs" / "double_heston_benchmark"
R2_DIAGNOSTICS = ROOT / "evidence" / "g2_r2_r3_20260822"

BASE_COMMIT = "72ad8e1aa845ec4c6f0fc61fc526df75438639bb"
NOISE_COMMIT = "e58b36d1ff6cbf39115d46698ad0fb45b1ac1b8b"
MODEL3_PROTOCOL_COMMIT = "f34a4d334f1703ed6e31fa60782c43ac7290ef3d"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n")
    if any(ord(char) > 127 for char in text):
        raise ValueError(f"Non-ASCII output generated: {path}")
    path.write_text(text, encoding="ascii", newline="\n")


def copy_figure(path: Path) -> None:
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    destination = LATEX_FIGURES / path.name
    destination.write_bytes(path.read_bytes())


def sci(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    exponent = int(f"{value:.{digits}e}".split("e")[1])
    mantissa = value / (10**exponent)
    if abs(exponent) >= 5 or (abs(value) < 1e-4 and value != 0):
        return rf"${mantissa:.{digits}f} \times 10^{{{exponent}}}$"
    if abs(value) < 1:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if abs(value) < 100:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:,.2f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}\\%"


def latex_table(
    name: str,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
    column_spec: str,
    note: str | None = None,
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.extend(["\\par\\vspace{0.35em}", f"\\begin{{minipage}}{{0.94\\linewidth}}\\small {note}\\end{{minipage}}"])
    lines.extend(["\\end{table}", ""])
    write_text(TABLES / f"{name}.tex", "\n".join(lines))


def git_text(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def canonical_engine_table() -> None:
    data = json.loads((BENCHMARK / "benchmark_summary.json").read_text(encoding="utf-8"))
    n64 = data["node_comparisons"]["64"]
    n96 = data["node_comparisons"]["96"]
    rows = [
        ["Controlled cases passing", f'{n64["passing_cases"]}/36', f'{n96["passing_cases"]}/36'],
        ["RMSE versus reference", sci(n64["rmse"]), sci(n96["rmse"])],
        ["MAE versus reference", sci(n64["mae"]), sci(n96["mae"])],
        ["Maximum absolute difference", sci(n64["maximum_absolute_difference"]), sci(n96["maximum_absolute_difference"])],
        ["No-arbitrage failures", str(data["no_arbitrage_failures"]["64"]), str(data["no_arbitrage_failures"]["96"])],
        ["Parity maximum error", sci(data["parity"]["max_errors"]["production_64"], 2), sci(data["parity"]["max_errors"]["production_96"], 2)],
    ]
    latex_table(
        "canonical_engine_benchmark",
        "Independent adaptive-quadrature validation of the production Double Heston pricer.",
        "tab:canonical-engine-benchmark",
        ["Diagnostic", "64-node production", "96-node production"],
        rows,
        "lrr",
        note="The reference independently implements the affine characteristic function and uses adaptive SciPy quadrature; it does not import the production pricer.",
    )


def primary_comparison_table() -> None:
    frame = pd.read_csv(EVIDENCE / "synthetic_test_comparison.csv")
    lookup = {
        "model1_seed_mean": frame[frame["method"] == "model1_seed_mean"].iloc[0],
        "model2_seed_mean": frame[frame["method"] == "model2_seed_mean"].iloc[0],
        "traditional_calibration": frame[frame["method"] == "traditional_calibration"].iloc[0],
    }
    neural = pd.read_csv(EVIDENCE / "neural_seed_results.csv")
    runtime = json.loads((EVIDENCE / "runtime_metrics.json").read_text(encoding="utf-8"))
    m1_seconds = 0.001 * neural[neural["method"] == "model1"]["per_surface_inference_ms"].mean()
    m2_seconds = 0.001 * neural[neural["method"] == "model2"]["per_surface_inference_ms"].mean()
    traditional_seconds = runtime["traditional_calibration"]["per_surface_calibration_seconds_mean"]

    metric_specs = [
        ("Range-scaled parameter RMSE", "range_scaled_param_rmse", sci),
        ("Standardized parameter RMSE", "standardized_param_rmse", sci),
        ("v0 total MAE", "v0_total_mae", sci),
        ("theta total MAE", "theta_total_mae", sci),
        ("Constraint validity", "constraint_validity_rate", pct),
        ("Mean repricing nRMSE", "repricing_normalized_rmse_mean", sci),
        ("Repricing p95 nRMSE", "repricing_normalized_rmse_p95", sci),
        ("Repricing $\\leq 10^{-4}$", "repricing_success_rate_le_1e-4", pct),
        ("Repricing $\\leq 10^{-3}$", "repricing_success_rate_le_1e-3", pct),
        ("Parameter RMSE $\\leq 0.25$", "param_recovery_rate_le_0.25", pct),
    ]
    rows = []
    for display, column, formatter in metric_specs:
        rows.append([display] + [formatter(lookup[method][column]) for method in lookup])
    rows.append([
        "Mean evaluation seconds per surface",
        sci(m1_seconds, 3),
        sci(m2_seconds, 3),
        sci(traditional_seconds, 3),
    ])
    latex_table(
        "primary_comparison",
        "Frozen synthetic-test comparison on 1{,}250 untouched known-truth surfaces.",
        "tab:primary-comparison",
        ["Metric", "Model 1 ANN", "Model 2 informed", "Traditional"],
        rows,
        "lccc",
        note="Neural values are means over seeds 11, 22, and 33. Traditional uses its frozen representative start. Inference and calibration runtimes are hardware-specific provenance and are not a quality score.",
    )


def real_market_development_table() -> None:
    doc = (ROOT / "docs" / "NTPC_SINGLE_STOCK_CALIBRATION.md").read_text(encoding="utf-8")
    wanted = {"BLACK_SCHOLES", "HESTON", "DOUBLE_HESTON"}
    rows = []
    for line in doc.splitlines():
        if not line.startswith("| ") or len(line.strip("|").split("|")) < 11:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0].upper() in wanted:
            rows.append([
                cells[0].replace("_", "\\_"),
                cells[1],
                cells[2],
                cells[3],
                cells[8],
                cells[9],
                cells[10],
            ])
    if len(rows) != 3:
        raise ValueError(f"Expected three NTPC development rows, found {len(rows)}")
    latex_table(
        "real_market_development_comparison",
        "Development-market BS/Heston/DH calibration pilot for NTPC on 2026-07-15.",
        "tab:real-market-development",
        ["Model", "Parameters", "Calibration price RMSE", "Holdout price RMSE", "Calibration IV RMSE", "Holdout IV RMSE", "Runtime (s)"],
        rows,
        "lcccccc",
        note="This is a development diagnostic, not a final real-market result. The declared winner rule returned NO\\_CLEAR\\_WINNER; fitted DH parameters are not claimed to be true market parameters.",
    )


def r2_selection_table() -> None:
    data = json.loads((R2_DIAGNOSTICS / "diagnostics_summary.json").read_text(encoding="utf-8"))
    levels = [("0.0000", "0\\%"), ("0.0050", "0.50\\%"), ("0.0100", "1.00\\%"), ("0.0200", "2.00\\%")]
    rows = []
    for key, display in levels:
        r2 = data["aggregates"]["R2"][key]
        r3 = data["aggregates"]["R3"][key]
        classification = data["comparative_assessment_by_noise"][key]["classification"]
        rows.append([
            display,
            sci(r2["median_best_parameter_rmse_scaled"]),
            sci(r3["median_best_parameter_rmse_scaled"]),
            sci(r2["median_best_repricing_rmse_relative"]),
            sci(r3["median_best_repricing_rmse_relative"]),
            classification.replace("_", "\\_"),
        ])
    latex_table(
        "r2_representation_selection",
        "Predeclared R2/R3 representation-selection diagnostics across 20 truths.",
        "tab:r2-selection",
        ["Noise", "R2 parameter RMSE", "R3 parameter RMSE", "R2 repricing", "R3 repricing", "Decision band"],
        rows,
        "lccccc",
        note="Parameter errors are range-scaled best-start medians; repricing entries are relative-RMSE medians. The frozen stopping rule selected R2 because R3's clean advantage was absent under realistic noise.",
    )


def evidence_classification_table() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "evidence" / "NTPC_SINGLE_STOCK_PILOT_MANIFEST.json").read_text(encoding="utf-8")
    )
    dataset = json.loads((ROOT / "data" / "final_r2_clean_10000" / "manifest.json").read_text(encoding="utf-8"))
    primary = json.loads((EVIDENCE / "FINAL_EVALUATION_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    rows = [
        [
            "Canonical engine validation",
            "Canonical",
            BASE_COMMIT[:12],
            "\\texttt{outputs/double\\_heston\\_benchmark/benchmark\\_summary.json}",
            "Independent adaptive-quadrature agreement",
        ],
        [
            "Development NTPC model pilot",
            "Development",
            BASE_COMMIT[:12],
            "\\texttt{docs/evidence/NTPC\\_SINGLE\\_STOCK\\_PILOT\\_MANIFEST.json}",
            str(manifest["status"]["winner"]).replace("_", "\\_"),
        ],
        [
            "Frozen R2 representation",
            "Canonical",
            BASE_COMMIT[:12],
            "\\texttt{docs/R2\\_REPRESENTATION\\_CONTRACT.md}",
            "Selected by sealed R2/R3 rule",
        ],
        [
            "Final 10k synthetic dataset",
            "Canonical",
            dataset["generation_git_sha"][:12],
            "\\texttt{data/final\\_r2\\_clean\\_10000/manifest.json}",
            f'Full replay: {str(dataset["replay_status"]).replace("_", "\\_")}',
        ],
        [
            "Primary method comparison",
            "Canonical",
            BASE_COMMIT[:12],
            "\\texttt{evidence/r2\\_primary\\_comparison\\_20260823/FINAL\\_EVALUATION\\_EVIDENCE\\_MANIFEST.json}",
            f'First test read: {str(primary["first_synthetic_test_read"]).lower()}',
        ],
        [
            "Neural positive-noise evaluations",
            "Complete subset of noise protocol",
            NOISE_COMMIT[:12],
            "\\texttt{origin/codex/r2-noise-recovery:evidence/r2\\_noise\\_robustness/neural/MANIFEST.json}",
            "Full neural population; traditional positives pending",
        ],
        [
            "Positive-noise traditional calibration",
            "Active/pending",
            NOISE_COMMIT[:12],
            "\\texttt{origin/codex/r2-noise-recovery:docs/R2\\_NOISE\\_CLOUD\\_EXECUTION\\_HANDOFF.md}",
            "No positive-level final CSV exists",
        ],
        [
            "Model3 PDE methodology",
            "Frozen protocol only",
            MODEL3_PROTOCOL_COMMIT[:12],
            "\\texttt{origin/research/model3-pde-protocol:docs/MODEL3\\_PDE\\_PROTOCOL.md}",
            "No Model3 training result exists",
        ],
        [
            "Pre-R2 G2 diagnostics and 108 grid",
            "Superseded/historical",
            BASE_COMMIT[:12],
            "\\texttt{docs/RESULTS\\_TO\\_DATE.md}; \\texttt{docs/G2\\_R2\\_R3...}",
            "Motivation and method development only",
        ],
        [
            "Model2 local CPU seed-11 replication",
            "Non-primary execution replica",
            BASE_COMMIT[:12],
            "\\texttt{evidence/r2\\_primary\\_comparison\\_20260823/training\\_run\\_manifest.json}",
            "Excluded from primary cohort",
        ],
    ]
    latex_table(
        "evidence_classification",
        "Central classification of manuscript evidence at the synthesis boundary.",
        "tab:evidence-classification",
        ["Item", "Classification", "Commit", "Artifact", "Boundary"],
        rows,
        "p{0.20\\linewidth}p{0.13\\linewidth}p{0.08\\linewidth}p{0.27\\linewidth}p{0.19\\linewidth}",
        note=(
            "The complete claim-to-artifact mapping is maintained in "
            "\\texttt{paper/RESULTS\\_INVENTORY.md}; this appendix table is a compact audit view."
        ),
    )


def neural_noise_tables_and_figure() -> None:
    raw = git_text(NOISE_COMMIT, "evidence/r2_noise_robustness/neural/all_neural_seed_headline.csv")
    frame = pd.read_csv(io.StringIO(raw))
    grouped = frame.groupby(["method", "noise_level_label"], sort=False).mean(numeric_only=True).reset_index()
    level_order = ["0%", "0.10%", "0.25%", "0.50%", "1.00%"]
    grouped["noise_level_label"] = pd.Categorical(grouped["noise_level_label"], categories=level_order, ordered=True)
    grouped = grouped.sort_values(["method", "noise_level_label"])

    rows = []
    for level in level_order:
        m1 = grouped[(grouped["method"] == "model1") & (grouped["noise_level_label"] == level)].iloc[0]
        m2 = grouped[(grouped["method"] == "model2") & (grouped["noise_level_label"] == level)].iloc[0]
        rows.append([
            level.replace("%", "\\%"),
            sci(m1["range_scaled_parameter_rmse"]),
            sci(m1["clean_latent_price_rmse_mean"]),
            sci(m2["range_scaled_parameter_rmse"]),
            sci(m2["clean_latent_price_rmse_mean"]),
            pct((m1["parameter_recovery_le_0_25"] + m2["parameter_recovery_le_0_25"]) / 2),
        ])
    latex_table(
        "completed_neural_noise",
        "Completed neural full-test-population observation-noise evaluations.",
        "tab:completed-neural-noise",
        ["Noise", "M1 parameter RMSE", "M1 clean-latent nRMSE", "M2 parameter RMSE", "M2 clean-latent nRMSE", "Mean recovery $\\leq0.25$"],
        rows,
        "lccccc",
        note=(
            "Seed-mean values over seeds 11, 22, and 33 for all 1,250 test surfaces at "
            "each level, read from pinned commit "
            + NOISE_COMMIT[:12]
            + ". Positive-level traditional subset completion remains pending and is excluded here."
        ),
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x_values = [0.0, 0.001, 0.0025, 0.005, 0.01]
    styles = {"model1": ("s", "-"), "model2": ("o", "--")}
    labels = {"model1": "Model 1 ANN", "model2": "Model 2 informed"}
    for method, (marker, linestyle) in styles.items():
        subset = grouped[grouped["method"] == method]
        ax.plot(
            x_values,
            subset["parameter_recovery_le_0_25"],
            marker=marker,
            linestyle=linestyle,
            color="black",
            linewidth=1.2,
            markersize=4.5,
            label=labels[method],
        )
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xticks(x_values)
    ax.set_xticklabels(["0", "0.10%", "0.25%", "0.50%", "1.00%"])
    ax.set_xlabel("Observation-noise level")
    ax.set_ylabel("Parameter RMSE $\\leq 0.25$")
    ax.grid(True, which="both", color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "completed_neural_noise.pdf", format="pdf")
    plt.close(fig)


def speed_accuracy_figure() -> None:
    frame = pd.read_csv(EVIDENCE / "synthetic_test_comparison.csv")
    neural = pd.read_csv(EVIDENCE / "neural_seed_results.csv")
    runtime = json.loads((EVIDENCE / "runtime_metrics.json").read_text(encoding="utf-8"))
    points = []
    for method, display, marker in [
        ("model1_seed_mean", "Model 1 ANN", "s"),
        ("model2_seed_mean", "Model 2 informed", "o"),
    ]:
        row = frame[frame["method"] == method].iloc[0]
        seconds = 0.001 * neural[neural["method"] == method.removesuffix("_seed_mean")]["per_surface_inference_ms"].mean()
        points.append((seconds, row["range_scaled_param_rmse"], display, marker))
    traditional_row = frame[frame["method"] == "traditional_calibration"].iloc[0]
    traditional_seconds = runtime["traditional_calibration"]["per_surface_calibration_seconds_mean"]
    points.append((traditional_seconds, traditional_row["range_scaled_param_rmse"], "Traditional calibration", "^"))

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    offsets = {"Model 1 ANN": (-0.05, 0.004), "Model 2 informed": (0.05, -0.010), "Traditional calibration": (-0.02, 0.006)}
    for seconds, error, label, marker in points:
        ax.scatter(seconds, error, marker=marker, s=48, facecolor="white", edgecolor="black", linewidth=1.1, zorder=3)
        dx, dy = offsets[label]
        ax.annotate(label, (seconds * (1 + dx * 0.05), error + dy), fontsize=9, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Mean evaluation seconds per surface (hardware-specific)")
    ax.set_ylabel("Range-scaled parameter RMSE")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.set_xlim(1e-6, 2e3)
    fig.tight_layout()
    fig.savefig(FIGURES / "speed_accuracy_tradeoff.pdf", format="pdf")
    plt.close(fig)


def fit_recovery_figure() -> None:
    frame = pd.read_csv(EVIDENCE / "synthetic_test_comparison.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    specs = [
        ("model1_seed_mean", "Model 1 ANN", "s"),
        ("model2_seed_mean", "Model 2 informed", "o"),
        ("traditional_calibration", "Traditional", "^"),
    ]
    for method, label, marker in specs:
        row = frame[frame["method"] == method].iloc[0]
        ax.scatter(
            row["repricing_normalized_rmse_mean"],
            row["range_scaled_param_rmse"],
            marker=marker,
            s=52,
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean repricing normalized RMSE")
    ax.set_ylabel("Range-scaled parameter RMSE")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fit_is_not_recovery.pdf", format="pdf")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    canonical_engine_table()
    primary_comparison_table()
    real_market_development_table()
    r2_selection_table()
    evidence_classification_table()
    neural_noise_tables_and_figure()
    speed_accuracy_figure()
    fit_recovery_figure()
    for source in FIGURES.glob("*.pdf"):
        copy_figure(source)
    write_text(GENERATED / "asset_sources.txt", "\n".join([
        f"base_commit={BASE_COMMIT}",
        f"noise_commit={NOISE_COMMIT}",
        f"model3_protocol_commit={MODEL3_PROTOCOL_COMMIT}",
        "",
    ]))


if __name__ == "__main__":
    main()
