# ICDEC 2026 submission — Overleaf project

Upload this folder to Overleaf (**New Project → Upload Project**) and set the
compiler to **pdfLaTeX**. It compiles on upload: `llncs.cls` is part of CTAN and
TeX Live, so Overleaf already has it and no class file needs to be shipped.

## Format requirements I verified (not assumed)

| Requirement | Value | Source |
|---|---|---|
| Proceedings series | Springer **Lecture Notes in Networks and Systems**, ISSN 2367-3370, Scopus-indexed | ICDEC 2024/2025 volumes on SpringerLink; COMSYS call for papers |
| Document class | `llncs` (Springer Proceedings LaTeX2e) | Springer author instructions for LNNS proceedings |
| Reference style | LNCS/`splncs04` style — already applied in `thebibliography` | same |
| Length | minimum 8, **target ~10**, maximum 12 pages incl. figures, tables, references | ICDEC 2025 call for papers |
| Submission | Microsoft CMT | ICDEC 2025 call for papers |
| Plagiarism | Springer policy applies; papers can be rejected at any stage | ICDEC 2025 call for papers |

The current draft is about 4,200 words with 2 figures and 2 tables, which
typesets to roughly 10-11 pages in `llncs` — inside the window. Check the page
count after your author block is filled in, and trim Section 7 first if needed.

## Before you submit

1. **Replace the author block.** Lines marked `% ---- REPLACE` near the top:
   names, ORCIDs, institute, email.
2. **Fill in the venue name.** The paper says "a single venue" throughout. If you
   are content to name Deribit, replace those mentions; if not, leave as is.
3. **Verify four references.** I am confident about the classical citations
   (Heston, Bakshi–Cao–Chen, Christoffersen–Heston–Jacobs, Carr–Madan,
   Lord–Kahl, Cui et al., Wilcoxon, Efron, Cameron et al., Nosek et al.).
   Please confirm exact volume/issue/pages for the four crypto-options papers:
   Madan et al. (2019), Hou et al. (2020), and the two Alexander & Imeraj papers.
4. **Check the figure date.** The captions say "a representative date". The
   underlying figures are Bitcoin, 27 July 2024. Name the date if you prefer.

## A note on session fit

The poster advertises **Special Session SS-02: Generative AI, LLMs and Secure
Intelligent Systems**. This paper is an empirical model-comparison study in
option pricing and does **not** fit SS-02. It fits the **main ICDEC conference**,
whose tracks include AI/ML, big data and data mining, and intelligent data
analytics. Submit to the main track, not SS-02, unless you reframe the work
around the neural surrogate component of the wider project.


## If a figure does not appear

Fixed in this version, but for reference: the first build used `[t]` placement on both
figures and both tables. With four large `[t]`-only floats competing near the end of
Section 5, LaTeX can defer a float past the end of the document and drop it, logging
`LaTeX Warning: Float(s) lost`. This version uses `[htbp]`, relaxes the float fractions,
and adds a `\FloatBarrier` before the Discussion section so both figures are flushed
before Section 6 begins.

The PNGs were also flattened from RGBA to RGB (an alpha channel can render oddly under
pdfLaTeX) and capped at 2200 px on the long edge, which keeps them well above print
resolution at the printed width while halving compile time.

If a figure still fails to appear, search the Overleaf log for `Float(s) lost` or
`Overfull \vbox` and tell me what it says.

## Figures

Three figures are used in the paper:

| File | Figure | Content |
|---|---|---|
| `figures/fig1_C_vs_S.png` | Fig. 1 | Call price against underlying price, three calibrated models against real quotes |
| `figures/fig2_C_vs_tau.png` | Fig. 2 | Call price against time to maturity |
| `figures/fig3_statistical_evidence.png` | Fig. 3 | (a) empirical CDF of held-out error, (b) per-date forest plot over 50 test dates |

### Why not every plot is in the paper

The venue caps submissions at **12 pages including figures, tables and
references**. The paper currently sits near 11 pages with three figures and two
tables. Adding more would push it over the limit and risk desk rejection, so the
remaining plots are kept in `figures/alternates/` rather than inserted:

| File | What it is | Use it if |
|---|---|---|
| `PANEL20d_mono.png`, `PANELTAU105_mono.png` | monochrome versions of Figs. 1 and 2 | the proceedings print in greyscale |
| `CLEAR_C_vs_S.png`, `CLEAR_C_vs_tau.png` | three-expiry versions of Figs. 1 and 2 | a reviewer asks for more expiries |
| `BTC6_statistical_evidence.png` | the full four-panel version of Fig. 3 (adds box plot and binned bias) | you have spare pages |
| `BTC5_range_of_outcomes.png` | best / typical / worst date side by side | a reviewer questions cherry-picking |

To swap any of them in, change the filename inside the relevant
`\includegraphics` call. To add one, copy an existing `figure` block and check
the page count afterwards.

### A caution on Fig. 3

Panel (b) reports Double Heston ahead on **45 of 50** dates with a mean
advantage of +0.58. That is a *descriptive* statistic pooling shock and calm
dates and using mean paired absolute error. It is **not** the pre-registered
endpoint, which is the by-regime cluster-level test in Table 2 (23/29 on shock
dates, 19/21 on calm). The caption and the surrounding text say so explicitly.
Do not quote 45/50 as the headline result -- quote Table 2.

## What makes this paper defensible

- Dates were chosen from volatility-index data **before any option data was
  downloaded**; the manifests record it.
- Two independent significance criteria were required, and this **changed the
  conclusions** — two results that pass a Wilcoxon test alone are reported as
  non-wins.
- A defect in our own Black-Scholes baseline is disclosed in Section 4.5. The
  correction made the competing model **stronger** (9.63 → 8.13) and both sets
  of numbers are reported.
- Negative results are in the main results table, including the Solana cell
  where the one-factor model is significantly better.
