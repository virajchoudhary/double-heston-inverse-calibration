# Publication readiness — verdict

**Audited:** 3 papers, 10 dimensions, 65 findings (17 blocker, 27 major, 21 minor).
**Status of the audit itself:** the fan-out was cut short by a usage limit. 6 of 10
dimensions completed; **stats-methodology, build-format, figures and numbers-A did not
finish**, and the adversarial verification stage did not run at all. The findings below
are therefore *unverified auditor output*, except the ones I personally re-derived,
which are marked **[VERIFIED]**. Treat the rest as leads to check, not facts.

---

## Verdict

| Paper | Verdict |
|---|---|
| **A** — capstone-4, *Physics-Informed Inverse Calibration…* | **NOT READY** |
| **B** — ICDEC, *…Pre-Registered Evidence…* | **NOT READY** |
| **C** — PINN, *…Controlled Study* | **NOT READY** (was my earlier call of "strongest"; it has 1 blocker + 12 major) |

**Nothing here is submittable today.** The gating problem is not formatting — it is that
three papers report overlapping work, two of them are the same study, and several
headline claims are not supported by the artifacts.

---

## Blockers that cut across all three

**1. Papers A and B are the same paper. [VERIFIED]**
- 47 identical sentences (24% of A's sentences); auditor measured 42% on its own slice
- 34% 8-gram overlap in both directions
- ~185-word verbatim shared passage (the whole Introduction opening)
- Table 1 identical; Table 2 identical in all nine rows
- `fig1_C_vs_S.png` **pixel-identical** (md5 of raw pixel data matches exactly)

Submitting both = duplicate publication under Springer/IEEE policy. Pick one.

**2. No paper discloses the other two.** No companion citation, no "under review
elsewhere" statement. Paper C's Appendix B reproduces A/B's five-market results and
conclusion with no citation at all.

---

## Paper B — 10 blockers

**[VERIFIED] The "Pre-Registered" title claim is not supported.**
- Freeze precedes first data download by **17–31 seconds** in every market — one script run
- `run.py:38` writes `'option_data_fetched_before_freeze': False` as a **hardcoded Python
  literal**. The paper cites this as evidence. It is an assertion, not a measurement.
- Mitigating: `freeze()` does `assert not (OUT/'surfaces').exists()`, and `manifest.sha256`
  exists — so the ordering claim is genuinely substantiated, just **self-administered**
- No OSF/AsPredicted/Zenodo record anywhere in the project
- Paper A explicitly declines the word for the identical study. B should too.

**[VERIFIED] The 16,186 "held-out" mislabel — and my earlier number was wrong.**
16,186 is the TOTAL observation count. The true Design-B held-out total is **5,214**
(BTC 2,223 / ETH 1,509 / SOL 658 / XRP 269 / HYPE 555), *not* the 5,660 I reported
earlier — I had wrongly included BTC's 446 validation-stage quotes. Note 2,223 is
exactly the BTC figure both papers already quote in their Figure 1 text.
Appears in B's **abstract**, Table 1 header and Conclusion.

**[VERIFIED] The explanatory variable is censored by a market-dependent filter.**
`min_expiries_per_date = 5` for BTC/ETH but `= 4` for SOL/XRP/HYPE.
The paper's headline is "the two markets that quote the most maturities are the two
where the second factor pays." That grouping is **partly built into the inclusion rule**,
and neither paper discloses it. This is the most damaging finding in the audit.

**Unverified but high-priority (auditor, not re-checked by me):**
- Design A reportedly disagrees with Design B in 6 cells and *reverses* the central
  finding; the paper says it agrees "in every cell"
- Figures 1 and 2 reportedly plot a **calibration** expiry while captioned as held-out
- SOL-calm "one-factor significantly better" reportedly fails the paper's own two-part
  criterion (reverse Wilcoxon p=0.174)
- Undisclosed post-freeze amendment to XRP calm-date selection, inherited by SOL
- Table 2's SH−DH column may be the per-date mean, not the per-cluster mean the caption claims
- Author block is still `First Author` / `0000-0000-0000-0000` / `your.email@institute.edu` **[VERIFIED]**
- 13 pages vs ICDEC's 12-page maximum **[VERIFIED]**

---

## Paper A

- **Title overclaims** — asserts physics-informed inverse calibration and stable parameter
  recovery; its own Limitations and Conclusion say it does neither
- **[VERIFIED] Your name is misspelled.** Paper A says **"Dhurva Ambhaikar"**; your git
  identity is **"Dhruva Ambhaikar"**. All five author names should be re-checked.
- Section heading "Data and Pre-Registered Design" contradicts its own disclaimer
- My shipped `README.txt` quotes the 5,660 figure — superseded by 5,214 above
- (One auditor finding — "Paper A does not exist" — is invalid; the scratchpad was cleared
  mid-run and I restored it.)

---

## Paper C — I was wrong to call this the strongest

Clean compile, 8 pages, properly anonymised, opaque alpha channels, 437–754 dpi — all
still true. But the audit found substantive problems:

- **Blocker:** Appendix B reproduces A/B's results with no citation
- Network output misspecified: text says the correction *multiplies* the baseline vol,
  the code *exponentiates* it — and Fig. 3 agrees with the code, not the text
- The predeclared stopping rule never fired (0.0738 > 0.07), yet §V-D says work was
  "exhausted by the stopping rule"
- Appendix A's "58 model-by-maturity curves" overcounts by 45% (only 40 exist)
- Appendix A's convexity claim conceals **279 convexity violations**, worst gamma
  −3.97e-4 (−43% of peak), not "of order 1e-9"
- All 10,706 SPX quotes behind the central "market ceiling" result come from **one
  trading day**; never stated
- Three citations have wrong author initials (Zhang → C. not Y.; Hainaut & Casas → A.
  not J.; Bae/Kang/Lee → S./M.)
- Missing the published Correction to Zhang et al. (2025)

---

## The one piece of good news [VERIFIED]

The multiplicity problem both papers disclose but don't address **is not a problem**:

- No p-value is below its attainable minimum (exact one-sided Wilcoxon floor = 1/2ⁿ).
  BTC-shock and ETH-shock sit exactly *at* 1/1024 — saturated at 10 clusters, not fabricated.
- **Holm–Bonferroni** at α=0.05: all **4/4** claimed DH wins survive
- **Benjamini–Hochberg** FDR: the same 4 discoveries

Add one sentence stating this and a referee's first attack becomes a selling point.

---

## What to do, in order

1. **Decide: A or B, not both.** They are one paper.
2. **Check the `min_expiries_per_date` 5-vs-4 filter.** If the headline survives it,
   disclose it. If it doesn't, the central claim needs rewriting. Do this first —
   it determines whether there is a paper at all.
3. **Re-verify the Design A agreement claim** against the artifacts.
4. **Check whether Figures 1–2 plot a calibration or a held-out expiry.**
5. Drop "Pre-Registered" from B's title; describe it as an internally frozen protocol.
6. Fix 16,186 → 5,214 held-out (or relabel as total observations) in every location.
7. Fix the author names and fill in B's author block.
8. Cut B to ≤12 pages.
9. Retitle A, or merge it into B.
10. Fix Paper C's text/code mismatch, stopping-rule claim, convexity claim, one-day SPX
    disclosure, and the three citation errors.

---

# ADDENDUM — verification of the blockers the cut-short audit left unchecked

I re-derived the four decisive "unverified" items myself. **Three of the four confirm,
and two are worse than the auditor stated.**

## 1. [VERIFIED — CONFIRMED] Paper B's Design A claim is false

main.tex:314 states: *"Design~A agrees with it in sign and in significance in every cell
and is omitted for space."* Re-derived from the artifacts, **6 of 11 cells disagree**:

| Cell | Design B (reported) | Design A (omitted) | disagreement |
|---|---|---|---|
| HYPE all | DH does NOT beat SH (p=0.0287) | DH **DOES** beat SH (p=0.0402) | significance |
| HYPE other | NOT (p=0.0239) | **DOES** (p=0.0402) | significance |
| SOL shock | NOT (p=0.234) | **DOES** (p=0.0234) | significance |
| XRP shock | NOT (p=0.0371) | **DOES** (p=0.00195) | significance |
| SOL calm | −0.481 favours SH | **+0.137 favours DH** | sign |
| XRP calm | −0.188 favours SH | **+0.496 favours DH** | sign |

This is the most serious finding in the audit. The paper's entire thesis is that the
second factor pays **only** in the two markets with the richest maturity coverage.
Under Design A it also pays on SOL shock, XRP shock and HYPE — **the maturity-coverage
story collapses under the secondary design.** The paper omits that analysis and
justifies the omission with a claim that is demonstrably untrue.

## 2. [VERIFIED — CONFIRMED, AND WORSE] The shared C-vs-S figure is in-sample, not held-out

`fig1_C_vs_S.png` is Figure 2 of Paper A and Figure 1 of Paper B (byte-identical).
Generated by `btc_multifactor_v1/plot_single_clean.py`, which has **no held-out filter**.

- Date 2024-07-27, expiry 2024-08-16, maturity rank **2 of 8**
- `heldout_B = False` → **this is a CALIBRATION expiry. All three models are in-sample on it.**
- Design B holds out odd ranks (1, 3, 5); rank 2 is even, so it was fitted on.

Paper A captions it *"Illustrative **held-out** Bitcoin surface at one expiry."* That is wrong.

Worse: the "Double Heston is 4.7x more accurate" banner printed inside the figure is
computed over **7 of the expiry's 20 quotes** — only those inside the moneyness window
[0.71, 0.88] hardcoded in the script. On the full expiry the ratio is smaller.

**This figure is in the Overleaf zip I sent you.** Do not submit either paper with it
until the caption is corrected or the figure is regenerated on a genuinely held-out expiry.

## 3. [VERIFIED] Market-dependent filter on the explanatory variable

`min_expiries_per_date = 5` for BTC/ETH, `= 4` for SOL/XRP/HYPE. Undisclosed in both papers.

## 4. [VERIFIED] The freeze flag is a hardcoded literal

`btc_multifactor_v1/run.py:38` writes `'option_data_fetched_before_freeze': False` as a
Python constant. Partly mitigated: `freeze()` asserts `not (OUT/'surfaces').exists()`,
and the on-disk mtimes do show freeze-before-fetch. Self-administered, not independent.

---

## Revised bottom line

The ordering in "What to do" above changes. **Item 1 is now: resolve Design A.**
Either report it honestly — which means the headline claim must be substantially
weakened — or justify in the text why Design B alone is decisive. The current sentence
at main.tex:314 cannot stay in any form.
