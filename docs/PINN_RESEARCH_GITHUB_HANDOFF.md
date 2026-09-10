# PINN research GitHub handoff — 2026-09-10

This update preserves the regular, deeper, factor-structured and compositional
PINN experiments, including failed recovery results. It incorporates the remote
Single Heston research commit `b0b461ac012a16894078f7906cd9efb50e204a9c`
without replacing its files.

Start with `docs/REGULAR_PINN_HANDOFF.md` and
`outputs/regular_pinn_recovery/composition_balanced_report910101/REPORT.md`.
The latest balanced composition passes 50/120 individual parameter gates and
0/12 complete cases. The earlier composition passes 57/120 and 0/12. Accurate
ten-parameter recovery remains unsolved; these are exposed synthetic development
results, not an unseen or NSE market validation claim.

## Included and excluded artifacts

Code, protocols, parameter tables, plots, source snapshots, JSON audits, test XML
and selected `model.pt` weights are included for all new research variants,
including unsuccessful pilots. Existing safetensors artifacts are preserved.
The Git attributes preserve research evidence bytes rather than normalizing
line endings that could invalidate recorded hashes.

The existing ignore policy continues to exclude generated `.npz` training and
diagnostic arrays, intermediate `.pt` checkpoints, optimizer states, caches and
logs. They remain on the originating machine; this push is not a complete backup
of that machine. Refer to the variant protocols and archived configs for
generation/training commands and seeds. A full artifact audit may require those
local arrays and intermediate states; a fresh clone alone does not contain them.
No new raw NSE dataset or database export is part of this update.

Historical notes saying work was local/unpushed describe their original run
state. Archived evidence and test outcomes are not rewritten by this packaging
update. The broader checkout's recorded failures remain disclosed in the reports.
