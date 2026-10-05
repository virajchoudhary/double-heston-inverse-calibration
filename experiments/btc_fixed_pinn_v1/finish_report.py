"""Reporting-only compatibility repair; all frozen sources/scores unchanged.

Frozen run.py completed scores/integrity, then failed plotting fidelity['id']:
the saved identifier is fidelity['scenario']. Add its alias only in memory.
"""
import json
from pathlib import Path
import pandas as pd
from run import OUT, sha, read, verify, make_report, save

protocol = verify()
score_files = ['predictions.csv', 'metrics.csv', 'daily_metrics.csv', 'fidelity.json',
               'scoring_quotes.csv', 'cleaning_audit.json', 'forward_anchors.csv',
               'leakage_probes.json', 'integrity.json']
before = {name: sha(OUT / name) for name in score_files}
fidelity = [{**r, 'id': r['scenario']} for r in read(OUT / 'fidelity.json')]
make_report(protocol, pd.read_csv(OUT/'scoring_quotes.csv'), pd.read_csv(OUT/'metrics.csv'), fidelity)
after = {name: sha(OUT / name) for name in score_files}
assert before == after
verify()
save(OUT / 'reporting_repair.json', {
    'scope': 'Report rendering only: in-memory alias id=scenario; no scores, settings, source code or weights altered.',
    'score_hashes_before_equal_after': True, 'score_sha256': after,
    'reporting_script_sha256': sha(Path(__file__)),
})
with (OUT/'REPORT.md').open('a') as f:
    f.write('\n## Reporting-only repair\n\nThe scoring run completed and saved all scores/integrity checks, then hit a plotting identifier mismatch (`id` versus `scenario`). `finish_report.py` renders the same saved results with an in-memory alias. All frozen sources and score hashes remain unchanged; see `reporting_repair.json`.\n')
print('Report generated; all frozen source, checkpoint and score hashes unchanged.')
