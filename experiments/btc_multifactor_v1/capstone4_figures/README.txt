FIGURES FOR capstone-4  (Physics-Informed Inverse Calibration of the Double-Heston Model)

WHY YOUR FIGURES WERE EMPTY BOXES
---------------------------------
Your compiled PDF draws a rectangle with a filename printed inside it, e.g.

    [                                    ]
    [   fig_error_ecdf_panel.png         ]
    [                                    ]

That is what graphicx does when the image file cannot be found. It is not a
float-placement problem and not a compiler problem: the five files simply did
not exist in the project. The manuscript was revised and the figures renamed
and split, but the new image files were never generated.

FILES IN THIS ARCHIVE — drop all five next to main.tex (or into your figures/
folder if \graphicspath points there). Names match your .tex exactly.

  fig_error_ecdf_panel.png    Fig. 1(a)  quote-level absolute IV error, ECDF
                              50 dates, 2,223 held-out quotes
                              medians: Black-Scholes 2.95, Single Heston 1.39,
                              Double Heston 0.93 volatility points

  fig_error_date_panel.png    Fig. 1(b)  ranked date-level mean |error|
                              advantage of Double over Single Heston, with 95%
                              uncertainty bars; DH ahead on 45 of 50 dates,
                              mean +0.58, cluster-bootstrap CI [0.42, 0.75]

  fig_range_best.png          Fig. 3 top     15 January 2025, 97 quotes
                              BS 11.10  SH 3.64  DH 0.95
  fig_range_typical.png       Fig. 3 bottom   2 September 2025, 133 quotes
                              BS 5.82   SH 1.96  DH 1.41
  fig_range_worst.png         Fig. 4         20 October 2024, 60 quotes
                              BS 6.00   SH 2.97  DH 3.17  <- Single Heston wins

All are RGB (no alpha channel, which can render as a black box under pdfLaTeX)
and capped at 2000 px on the long edge.

ABOUT FIGURE 2
--------------
Your PDF has a caption for Fig. 2 ("Illustrative held-out Bitcoin surface at one
expiry, displayed in F/K and C/K coordinates") but the PDF contains no
placeholder box for it, so its expected filename could not be recovered from the
compiled output. spare_for_fig2/ contains that figure in colour and monochrome.
Rename whichever you want to the filename your .tex uses, or tell me the name
and I will regenerate it under that name.

PROVENANCE
----------
Every value comes from the frozen experiment artifacts. Black-Scholes is the
corrected AMENDMENT_01 per-expiry fit, not the flawed v1 per-quote version.
Nothing was refitted to produce these figures.
