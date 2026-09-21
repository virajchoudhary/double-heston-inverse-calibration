"""Freeze manifest for the untouched final NIFTY test. REPORTING/AUDIT ONLY.

create          Hash every component the final test depends on. Refuses unless the controlled
                experiment is locked and evaluated and the final interval is still unopened.
verify          Re-check every hash. Run immediately before `run.py fetch-final`, after it, and
                before any final-test reporting.
record-opening  Record when the final interval was opened, against the freeze manifest hash.
"""
import argparse
from run import OUT, HERE, ROOT, verify as verify_protocol, read, save, sha, stamp

CODE = ['run.py', 'literature_exact.py', 'report.py', 'supplement.py', 'test_experiment.py', 'teacher_quality.py',
        'pinn_development_check.py', 'controlled_analysis.py', 'market_tables.py', 'final_freeze.py', 'AMENDMENT.md', 'CONTRACT_AUDIT.md']
REPO = ['src/double_heston.py', 'src/double_heston_reference.py', 'src/constraints.py', 'src/mentor_dh_pinn/regular_pinn_torch.py',
        'src/mentor_dh_pinn/regular_pinn_data.py', 'src/mentor_dh_pinn/torch_pricer.py', 'scripts/mentor_dh_pinn/nifty_fixed_crisis.py',
        'configs/nifty_fixed_crisis.json']


def components(c):
    f = {'protocol manifest': OUT / 'manifest.json', 'protocol manifest sha256': OUT / 'manifest.sha256', 'scenario bank': OUT / 'scenario_bank.json',
         'config (final dates, quote filters, SH/BS/PINN settings, metrics)': HERE / 'config.json',
         'market selection (SH 14%, DH 14%, BS_TERM)': OUT / 'market_selection.json', 'validation quotes': OUT / 'market_validation/clean_quotes.csv',
         'evaluation lock (checkpoints, baselines)': OUT / 'evaluation_lock.json', 'controlled results': OUT / 'controlled_results.json',
         'exact-pricer validation': OUT / 'exact_validation.json', 'exact-pricer test output': OUT / 'exact_tests.txt',
         'teacher quality': OUT / 'teacher_quality/teacher_quality.json', 'PINN development check': OUT / 'pinn_development_check/development_check.json',
         'reporting addendum manifest': OUT / 'reporting_addendum_manifest.json'}
    for s in c['pinn']['seeds']:
        f[f'PINN checkpoint seed {s}'] = OUT / f'pinn_s{s}/weights.pt'; f[f'PINN completion record seed {s}'] = OUT / f'pinn_s{s}/completed.json'
    for p in CODE: f[f'code/{p}'] = HERE / p
    for p in REPO: f[f'repo/{p}'] = ROOT / p
    return f


def create():
    c = verify_protocol(); m = read(OUT / 'manifest.json')
    assert m['market_final_opened'] is False
    for p in ['market_final_download.json', 'market_final', 'market_final_predictions.csv', 'market_final_status.json']:
        assert not (OUT / p).exists(), f'final interval already touched: {p}'
    assert (OUT / 'evaluation_lock.json').exists() and (OUT / 'controlled_results.json').exists()
    path = OUT / 'final_test_freeze_manifest.json'; assert not path.exists()
    files = components(c); missing = [k for k, p in files.items() if not p.exists()]; assert not missing, missing
    lock = read(OUT / 'evaluation_lock.json')
    save(path, {'utc': stamp(), 'purpose': 'Every component the untouched final NIFTY test depends on, frozen before any final file is downloaded or read.',
        'final_interval': c['market_final'], 'validation_interval': c['market_validation'], 'selected': read(OUT / 'market_selection.json')['selected'],
        'quote_filters': c['market_quote_filters'], 'single_heston_protocol': c['single_fit'], 'black_scholes_definition': c['bs'],
        'pinn': {k: c['pinn'][k] for k in ['architecture', 'width', 'depth', 'seeds', 'checkpoint']}, 'state_adaptive': c['state_adaptive'],
        'metrics': c['metrics'], 'market_selection_objective': c['market_selection_objective'],
        'checkpoint_sha256_via_evaluation_lock': lock['models'], 'baseline_sha256_via_evaluation_lock': lock['baselines'],
        'post_freeze_command_sequence': ['python final_freeze.py verify', 'python run.py fetch-final', 'python final_freeze.py verify',
            'python final_freeze.py record-opening', 'python run.py market-final', 'python run.py report', 'python supplement.py audit',
            'python supplement.py report', 'python market_tables.py --stage final'],
        'rules_after_opening': ['no retraining', 'no parameter changes', 'no quote-filter changes', 'no metric changes', 'no date removal',
            'no new scenarios presented as preregistered', 'a genuine software bug requires preserving the failed run and a versioned correction'],
        'file_sha256': {k: sha(p) for k, p in files.items()}})
    (OUT / 'final_test_freeze_manifest.sha256').write_text(sha(path) + '\n')
    print('FINAL-TEST FREEZE CREATED', sha(path), len(files), 'components', flush=True)


def verify_freeze():
    path = OUT / 'final_test_freeze_manifest.json'
    assert sha(path) == (OUT / 'final_test_freeze_manifest.sha256').read_text().strip(), 'freeze manifest itself changed'
    m = read(path); c = verify_protocol(); files = components(c)
    changed = [k for k, p in files.items() if not p.exists() or sha(p) != m['file_sha256'].get(k)]
    assert not changed, f'Frozen final-test component changed: {changed}'
    print('FINAL-TEST FREEZE VERIFIED:', len(files), 'components unchanged', flush=True)
    return m


def record_opening():
    verify_freeze(); path = OUT / 'final_test_opened.json'; assert not path.exists()
    dl = read(OUT / 'market_final_download.json'); st = read(OUT / 'market_final_status.json')
    save(path, {'utc': stamp(), 'freeze_manifest_sha256': sha(OUT / 'final_test_freeze_manifest.json'), 'status': st,
                'downloaded_dates': [r['date'] for r in dl['rows'] if r['status'] == 'downloaded'],
                'unavailable_dates': [{'date': r['date'], 'status': r['status']} for r in dl['rows'] if r['status'] != 'downloaded']})
    print('FINAL INTERVAL OPENING RECORDED', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['create', 'verify', 'record-opening'])
    {'create': create, 'verify': verify_freeze, 'record-opening': record_opening}[ap.parse_args().command]()
