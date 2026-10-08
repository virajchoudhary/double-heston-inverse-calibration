# Overleaf project — read this first

## What this is, and what it is not

Your paper's `.tex` source does not exist anywhere on this machine; only
`NQComp2027_Anonymous.pdf` does. **This project is therefore NOT your paper with
edits applied.** It is a compilable scaffold that already contains everything I
could verify against committed artifacts, and marks with `%%% PASTE` the prose
that exists only inside your PDF.

**This project was not compiled locally** — there is no TeX toolchain on this
machine. The preamble uses only `IEEEtran`, `graphicx`, `booktabs`, `amsmath`,
`amssymb`, `url` and `hyperref`, all standard on Overleaf, but you should
compile it and fix anything the log flags.

## How to use it

1. Upload the whole folder to Overleaf ("New Project" → "Upload Project").
2. Set the compiler to **pdfLaTeX**.
3. Open `main.tex` and work through each `%%% PASTE` marker, copying the
   corresponding prose out of your PDF.
4. Resolve the Table III decision below.
5. Paste your remaining references into `thebibliography`; five are already there.

## The one blocking issue: Table III

Table III in your PDF reports all six arms "on the common panel". Only two of
those rows — `SHARED` and `ARCH_A` — match `final_fidelity/`. The values for
`A0`, `CONTROL`, `ARCH_B` and `ARCH_C` appear in **no committed artifact**, and
the only ablation CSV that exists is the *development* panel, which gives
different numbers for those same arms.

You have two honest options:

* **Preferred.** Re-run the four arms on the final panel, export the CSV, then
  delete the development-panel table in `main.tex` and `\input{table3_final_panel.tex}`
  after filling its starred cells.
* **Fallback.** Keep the table already in `main.tex`. It uses development-panel
  values for all six arms, so it is internally consistent, and its caption states
  that the headline number is from a different panel. Note that `ARCH_A` then
  reads $9.872901\times10^{-6}$, not $9.610126\times10^{-6}$.

Do not leave the mixed panel as published.

## One number to confirm

`main.tex` Table II carries the inference timings printed in your PDF (47.61 /
55.14 / 74.50 ms). The committed `data/efficiency.csv` reports a **different**
quantity: 11.14 ms (baseline) and 16.72 ms (ARCH_A) per 1024 quotes. Confirm
which measure Table II reports and state the unit. The parameter counts
(275,684 → 320,805) agree across both sources and are safe.

## Figures

`figures/` holds the four referenced by `main.tex`:

| file | what it is |
|---|---|
| `fig_regions.png` | per-region change in price RMSE, including the −1.67% regression |
| `fig_seeds.png` | seed robustness, mean over seeds 17 and 43 with the range |
| `fig_efficiency.png` | accuracy against inference cost |
| `fig_price_curves.png` | DH-PINN against exact Double Heston, C vs S, with residuals |

`figures/optional/` holds extras you may want instead of or alongside these:
architecture diagrams, training losses, the ARCH_A curve panel, the
Black-Scholes reference family, the four-model comparison, C vs τ, the
fast-heavy/slow-heavy contrast, and four normalised diagnostic panels.

## Data

`data/` holds the CSVs behind every number in the tables, so a reviewer asking
for provenance can be answered directly. `DATA_PROVENANCE.md` explains what
data was used across the whole project — the paper is 100% synthetic by design,
and the market datasets sit in separate experiments.

`PAPER_PATCHES.tex` is the standalone block list, kept for reference; everything
in it is already incorporated into `main.tex`.
