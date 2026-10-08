CAPSTONE-4 OVERLEAF PROJECT -- COMPLETE AND SELF-CONTAINED
==========================================================
Physics-Informed Inverse Calibration of the Double-Heston Model
for Stable Option-Surface Parameter Recovery

HOW TO USE
----------
Overleaf -> New Project -> Upload Project -> select this .zip
Set the main document to main.tex if Overleaf does not pick it
automatically (Menu -> Main document).
Compiler: pdfLaTeX (Overleaf's default). No bibtex run is needed.

VERIFIED BEFORE SHIPPING
------------------------
Compiled locally with TeX Live 2026 / pdfLaTeX, three passes:
  - 0 errors
  - 0 undefined references
  - 0 undefined citations
  - 0 missing files
  - 0 overfull boxes
  - Output: 16 pages, all 6 figures embedded as real images
    (473-559 ppi), no placeholder boxes anywhere.

FILES
-----
main.tex                    The paper. Pure ASCII, no special encoding.
llncs.cls                   Springer LNCS class, bundled so the project
                            compiles even if the class is unavailable.
fig_error_ecdf_panel.png    Fig 1a  quote-level absolute IV error ECDF
fig_error_date_panel.png    Fig 1b  ranked date-level MAE advantage
fig1_C_vs_S.png             Fig 2   held-out BTC surface, F/K vs C/K
fig_range_best.png          Fig 3 top     2025-01-15, 97 quotes
fig_range_typical.png       Fig 3 bottom  2025-09-02, 133 quotes
fig_range_worst.png         Fig 4         2024-10-20, 60 quotes

WHAT WENT WRONG BEFORE
----------------------
The previous upload contained only figures and no .tex file, so
Overleaf had nothing to compile ("job aborted, file error in nonstop
mode"). Separately, five of the six figures did not exist in the
project, which is why they appeared as boxes with filenames printed
inside them. Both problems are fixed here: main.tex is present and
every figure it references is included.

ONE SUBSTANTIVE CORRECTION
--------------------------
In Table 1 the third column was headed "Held-out quotes" with the
values 6,171 / 3,619 / 2,687 / 1,249 / 2,460 (total 16,186).
Checked against the artifacts, 16,186 is the TOTAL number of option
observations on the evaluated surfaces. The Design B held-out subset
is 5,214 quotes (BTC 2,223, ETH 1,509, SOL 658, XRP 269, HYPE 555).
The column header has been changed to "Option observations", the
caption now states that the Design B held-out subset is smaller, and
the two places in the body text that said "held-out quotes/option
observations" for the 16,186 figure (Sections 5.3 and 8) now say
"option observations". The abstract already said "option
observations" and was left unchanged.
Every other number in Tables 1 and 2 was re-derived from the
artifacts and reproduces exactly.

REMOVING THE LOCAL TeX INSTALL (optional)
-----------------------------------------
A sandboxed TeX Live (TinyTeX) was installed to verify this compile.
It lives entirely in ~/Library/TinyTeX and needed no admin rights.
Delete that folder if you do not want it:  rm -rf ~/Library/TinyTeX
