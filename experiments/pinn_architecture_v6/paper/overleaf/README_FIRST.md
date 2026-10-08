# Physics-Informed Neural Architectures for Double-Heston Option Pricing

Upload this folder to Overleaf (**New Project -> Upload Project**) and set the
compiler to **pdfLaTeX**. `main.tex` is the complete paper. There are no
placeholders.

## What changed from the earlier manuscript

The title is unchanged. The **study behind it is different**: the paper is now
built on the `dh_pinn_v5` work, not the earlier v6 ARCH_A study.

This was not a numbers swap, because the two are not comparable:

| | earlier v6 study | this paper (v5) |
|---|---|---|
| final panel | 4,096 pts, seed 206104 | **16,384 pts, seed 93202** |
| development | 4,096 | 8,192, seed 93201 |
| baseline | v5 SHARED, 275,684 params | locked v4 C3, 269,825 params |
| selected | ARCH_A, 320,805 params | **A5 gated, 282,632 params** |
| headline | 9.24% price-RMSE reduction | **1.61x fidelity (1.062e-5 -> 6.610e-6)** |

Different panels, different baselines, different repositories. The earlier
9.24% figure and its bootstrap interval belong to the other study and do not
appear here.

Sections I and II reuse the earlier manuscript's prose, which is your own
writing, with the related-work paragraph extended to cite adaptive activations
since the ablation now isolates them.

## The paper's argument

1. A six-stage ablation at an identical budget shows **residual connections are
   the dominant ingredient** (3.1x), adaptive activations add 21%, gradient
   balancing trades price accuracy for IV accuracy, adaptive collocation repairs
   the physics residual, and the gated block wins on every metric.
2. The winner, selected on development data alone, is **1.61x more accurate**
   on a panel opened once — with its costs stated: 4.7% more parameters, 1.7x
   slower inference, and a **1.8x worse PDE residual**.
3. That fidelity gain moves real S&P 500 error by **0.003 of 5.85** volatility
   points, because network error is ~1.3% of model misspecification.
4. A 4,545-parameter residual head cuts held-out market error by two thirds —
   but the two-factor prior **beats Black-Scholes (p=0.018) and does not beat
   Single Heston (p=0.34)**. Reported as a negative result.

The paper is stronger for point 4, not weaker. A reviewer who sees a
well-controlled negative result alongside a positive one trusts both more.

## Figures

| file | source |
|---|---|
| `fig1_ablation.png` | redrawn from the report's ablation table |
| `fig2_old_vs_new.png` | redrawn from the fidelity table |
| `fig3_architecture.png` | the study's own diagram, unmodified |
| `fig4_market_benchmark.png` | redrawn from the SPX table |
| `fig5_track_b.png` | redrawn from the hybrid table |
| `figA1_C_vs_S.png` | call price against underlying price |
| `figA2_C_vs_tau.png` | call price against time to maturity |

Figures 1, 2, 4 and 5 were redrawn rather than reused: the originals had
overlapping tick labels. Every value comes from `data/v5_study_report.md`, and
`data/make_v5_figures.py` regenerates them.

The maturity axis in `figA2` **decreases** left to right, expiry at the right
edge, following the usual textbook convention. The data are unchanged and the
relation is still the standard one, dC/dtau >= 0, which is the same statement
as dC/dt <= 0 since tau = T - t.

## Appendices

**Appendix A** carries exactly two figures, C against S and C against tau, and
validates that the surrogate is a valid option price: across 58 curves, no
violations of monotonicity in S, of `C >= max(S-K,0)`, or of `C <= S`. Earlier
drafts also carried normalised and fast/slow panels; those have been removed at
the author's request and the underlying data remain in `data/`.

**Appendix B** is explicitly marked as outside the frozen protocol. It covers
the five Deribit markets and the IWM surface, with both IWM disclosures intact.

To make the paper purely synthetic, delete Appendix B: remove from
`\section{Breadth of the Market Observation}` to `\begin{thebibliography}`.

## Not compiled here

There is no TeX toolchain on this machine, so `main.tex` has **not** been
compiled. Static checks pass: zero non-ASCII bytes, balanced environments,
401/401 braces, even `$` count, all 13 figures resolve with no unused files, no
undefined `\ref`, and all 26 `\bibitem` keys cited. Compile on Overleaf and fix
whatever the log reports; float placement is the likeliest complaint.
