# What was fixed — three papers

All three compile from a clean extraction of the shipped zip with
**0 errors, 0 undefined references, 0 missing files, 0 overfull boxes**,
every figure embedded as a real image, and no placeholder boxes.

| | pages | figures | status |
|---|---|---|---|
| **A** capstone-4 (llncs) | 17 | 6 | compiles clean |
| **B** ICDEC (llncs) | 13 | 4 | compiles clean |
| **C** PINN (IEEEtran) | 8 | 7 | compiles clean, anonymised |

---

## Correction to my own earlier report

I told you Paper B's 13 pages broke "ICDEC's 12-page maximum". **That was wrong.**
It came from my own README quoting the *2025* call. The **2026** call
(icdec.org/2026/call-for-paper, checked today) states a **minimum of 10 pages and
no maximum**. 13 pages is fine. I trimmed ~700 words before catching this; the
trims tightened the prose and I kept them, but they were not required.

I also told you the Design-B held-out total was 5,660. It is **5,214**
(BTC 2,223 / ETH 1,509 / SOL 658 / XRP 269 / HYPE 555) — I had wrongly counted
BTC's 446 validation-stage quotes. Corrected everywhere, including Paper A's README.

---

## PAPER B — 14 fixes

**Integrity**
1. Title: "Pre-Registered Evidence" → "Pre-Specified Held-Out Evidence".
2. All 10 body uses of "pre-registered" → "pre-specified". Left untouched: the
   Nosek citation, its title, and "independently pre-registered experiment"
   (correct usage).
3. Added an explicit terminology note: the freeze was self-administered, not a
   registry deposit, and no external preregistration is claimed.
4. The manifest flag is no longer cited as if it were a measurement. New text
   states what is actually true: the freeze routine *aborts if a surface
   directory already exists* and writes the manifest + SHA-256 before any
   download, the flag records that invariant, and on disk the manifest precedes
   the first stored surface in every market.

**The Design A claim — the most serious fix**
5. main.tex:314 said *"Design A agrees with it in sign and in significance in
   every cell and is omitted for space."* Re-derived: **6 of 11 cells disagree**
   (4 in significance, 2 in sign). Sentence removed.
6. Design A is now **reported in full**, merged into the main results table so
   the two designs sit side by side. DH clears the criterion in **7 of 9** cells
   under Design A vs **4 of 9** under Design B.
7. Design A was described as withholding "a random subset of quotes". The frozen
   config defines it as **deterministic odd-ranked strikes**. Corrected.
8. Added the interpretation, which makes the paper *stronger*: under Design A
   every maturity is still calibrated on, so no separation of time scales is
   required and the extra factor helps almost everywhere; under Design B a
   maturity must be priced unseen, which demands identifying two speeds, and
   that only works where enough expiries exist. The divergence between designs
   is now presented as the clearest single piece of evidence for the mechanism.

**Statistics**
9. "On Solana calm dates the one-factor model is *significantly better*" —
   re-derived the reverse one-sided Wilcoxon: **p = 0.1742**. The paper refuses
   DH wins that pass only one test (XRP, HYPE) but was accepting an SH win that
   passes only one. Now applied symmetrically in all four places it appeared
   (Introduction, Results, Discussion, Conclusion).
10. Added the multiplicity result: all **4/4** declared wins survive
    Holm–Bonferroni and Benjamini–Hochberg; noted that at 10 clusters the
    smallest attainable one-sided p is 2⁻¹⁰ = 0.00098, so the BTC/ETH shock rows
    are at the test's floor, not anomalously extreme.

**Factual**
11. 16,186 relabelled from "held-out quotes" to "option observations" in the
    abstract, Table 1 header and Conclusion; the true held-out total (5,214,
    with per-market breakdown) now stated.
12. Disclosed the **market-dependent inclusion filter**: `min_expiries_per_date`
    = 5 for BTC/ETH but 4 for SOL/XRP/HYPE. This directly conditions the paper's
    explanatory variable and was previously undisclosed.
13. Figure caption said "a held-out Bitcoin expiry". The plotted expiry
    (2024-08-16, maturity rank 2 of 8) has `heldout_B = False` — it is a
    **calibration** expiry and all three models are in-sample on it. Caption now
    says so, and notes the "4.7×" banner is computed over 7 of the expiry's 20
    quotes.
14. "positive on 45 of the 50 test dates" — unreproducible. Artifacts give
    **42/50**, which is also what the figure's own embedded title says. Corrected.

**Also:** author block filled in (was `First Author` / `0000-0000-0000-0000` /
`your.email@institute.edu`); two figures merged side by side; README_FIRST.md
page-limit row corrected.

---

## PAPER A — 6 fixes

1. **Your name was misspelled**: "Dhurva Ambhaikar" → "Dhruva Ambhaikar".
2. Same figure-caption fix as Paper B (calibration expiry, not held-out; 7 of 20 quotes).
3. Section heading "Data and Pre-Registered Design" → "Pre-Specified".
4. Table 1 caption now gives the real held-out total (5,214, per market) and
   discloses the market-dependent expiry filter.
5. Added the Holm/BH multiplicity result and the 2⁻¹⁰ floor note.
6. Added the market-dependent filter paragraph to Limitations.
7. README.txt corrected 5,660 → 5,214.

Paper A was already *correct* on Design A — it says Design A "is not tabulated
here and no claim about its cell-level agreement is made." That is honest, and
is exactly where Paper B went wrong.

---

## PAPER C — 16 fixes, all independently verified first

Four verifier agents re-derived every finding against the artifacts before any
edit. **Two findings were refuted and I did not act on them** — notably the
alarming "279 convexity violations", which are all confined to six
`BS_PINN_own_convention` rows the paper's sentence already excludes, and whose
"−43% of peak" was a 100× unit error (the column is already a percentage; the
true worst is −0.434%). Over the 52 rows the claim actually covers, the worst
convexity residual is −9.7e-10 — literally "of order 1e-9", as written.

Applied: network-output text/code mismatch (the correction exponentiates, it
does not multiply); the stopping-rule claim (0.0738 > 0.07, it never fired);
the single-trading-day disclosure for all 10,706 SPX quotes; "two to three
orders of magnitude" → the true 70× and 87×; Table II development-panel
labelling; "58 curves" → 34 distinct on the r=q=0 benchmark; "30
adjacent-maturity comparisons" → 20 distinct; Appendix B now cites its source
study and flags it as under submission; Appendix B's held-out mislabel; the
identifiability over-claim; and five bibliography corrections (Zhang initials
and bogus page range, Hainaut/Casas initial, Bae/Kang/Lee initials, Cuomo's
"Schiano Di Cola", Ruf page range). `DATA_PROVENANCE.md` described the **v6**
study while the paper is built on **v5** — reconciled.

---

# STILL NEEDS YOUR DECISION — I did not decide these for you

**1. Papers A and B are the same paper. You must pick one.** 47 identical
sentences, 34% 8-gram overlap, a 185-word verbatim shared passage, identical
Table 1, identical Design-B results, and a pixel-identical figure. Submitting
both is duplicate publication under Springer/IEEE policy. Fixing both does not
fix this — only choosing does.

**2. Paper A's title.** It claims physics-informed inverse calibration and
stable parameter recovery; the paper does neither, and its own Limitations
section says so. Either retitle (e.g. *"When Does a Second Variance Factor Pay?
Held-Out Evidence from Five Cryptocurrency Option Markets"* — but that is now
Paper B's title, which is the point), or fold A into B. My recommendation:
**submit B, drop A.** B is the honest version of the same work.

**3. Paper B's author list.** I filled the placeholder block with Paper A's five
authors and ORCIDs, on the assumption it is the same team. Confirm before
submitting.

**4. "Dhurva Bamb".** I corrected *your* name from your git identity. I cannot
verify the second author's — if it should also be "Dhruva", fix it in both papers.

**5. Paper C's venue.** It is 8 pages in IEEEtran and correctly anonymised, so
it is not aimed at ICDEC (minimum 10, llncs, not blind). Confirm the target.
