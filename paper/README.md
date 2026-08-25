# Paper synthesis lane

This directory is documentation-only synthesis. It does not define, rerun, or
modify scientific experiments. Numerical statements are generated from committed
evidence or read from explicit immutable Git references; `RESULTS_INVENTORY.md`
is the authoritative map from manuscript claims to sources.

## Status boundary

- Canonical: frozen R2 representation, final 10k synthetic dataset, primary
  Traditional/ANN/Model2 comparison, canonical engine validation.
- Development: NTPC BS/Heston/DH pilot and earlier G2 diagnostics used for
  motivation and method development.
- Historical/superseded: the fixed 108 grid, pre-R2 G2 geometry decisions, and
  non-primary execution replicas.
- Active/pending: positive-noise traditional subset completion, OOD/boundary,
  G8 real-market evaluation, and all Model3 training results.

## Reproduce generated assets

From the repository root:

```powershell
C:\Python313\python.exe paper\scripts\generate_results_assets.py
C:\Python313\python.exe paper\scripts\validate_paper.py
```

The generator writes only under `paper/generated`, then byte-copies the PDFs into
`paper/figures` because that is the manuscript's graphics path. It reads committed
evidence in the current worktree and uses `git show` against pinned commit hashes for
neural noise evidence that lives on another branch. It never checks out or merges that
branch.

PDF compilation was not available on the audit host (`pdflatex`, `latexmk`, and
Tectonic were absent). The structural validator checks required sections, labels,
balanced environments, generated assets, inventory coverage, and ASCII source.

## Compile when LaTeX is available

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
