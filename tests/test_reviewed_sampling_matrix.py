from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
import yaml
import json
from src import audit_reviewed_sampling as reviewed
from src.constants import PARAMETER_NAMES
from src.synthetic_dataset import assign_reviewed_distribution_splits

def test_determinism_and_seed_sensitivity() -> None:
    config = reviewed.load_reviewed_config()
    first = reviewed.sample_distribution('interior_train', 64, seed=7, config=config)
    second = reviewed.sample_distribution('interior_train', 64, seed=7, config=config)
    different = reviewed.sample_distribution('interior_train', 64, seed=8, config=config)
    pd.testing.assert_frame_equal(first, second)
    assert not first[PARAMETER_NAMES].equals(different[PARAMETER_NAMES])

def test_constraints_transform_polar_and_interior_margin_gate() -> None:
    config = reviewed.load_reviewed_config()
    frame = reviewed.sample_distribution('interior_train', 128, config=config)
    hard = config['hard_constraints']
    assert frame['hard_bounds_valid'].all()
    assert (frame['kappa_fast'] - frame['kappa_slow'] >= 0.8).all()
    assert (frame['slow_feller_margin'] > 0).all()
    assert (frame['fast_feller_margin'] > 0).all()
    assert (frame['correlation_disk_value'] < 1).all()
    assert (frame[['rho_slow', 'rho_fast']].abs() <= 0.95).all().all()
    row = frame.iloc[0]
    assert row.slow_feller_margin == pytest.approx(reviewed.canonical_feller_margin(row.kappa_slow, row.theta_slow, row.sigma_slow))
    for parameter in PARAMETER_NAMES:
        assert (frame[parameter] >= float(hard[parameter]['lower'])).all()
        assert (frame[parameter] <= float(hard[parameter]['upper'])).all()
    accepted = frame.loc[frame['accepted']]
    assert not accepted.empty
    assert not accepted[['accepted_any_boundary_near', 'accepted_hard_bound_near', 'accepted_feller_near', 'accepted_weak_slow_fast_separation', 'accepted_correlation_disk_near']].any().any()

def test_challenge_ood_and_complete_surface_split_isolation() -> None:
    config = reviewed.load_reviewed_config()
    challenge = reviewed.sample_challenge(80, config=config)
    predicates = {'near_feller': 'accepted_feller_near', 'weak_separation': 'accepted_weak_slow_fast_separation', 'near_hard_bound': 'accepted_hard_bound_near', 'near_correlation_disk': 'accepted_correlation_disk_near'}
    for regime, predicate in predicates.items():
        assert challenge.loc[challenge['regime'] == regime, predicate].all()
    ood = reviewed.sample_distribution('ood_test', 80, config=config)
    reviewed.validate_ood_support(ood, config)
    ids = [f'surface-{index}' for index in range(20)]
    eligible = assign_reviewed_distribution_splits(ids, 'wide_valid_train', seed=3)
    assert set(eligible) == set(ids) and set(eligible.values()) == {'train', 'validation', 'test'}
    assert set(assign_reviewed_distribution_splits(ids, 'boundary_challenge').values()) == {'challenge_excluded'}
    assert set(assign_reviewed_distribution_splits(ids, 'ood_test').values()) == {'ood_test'}

def test_config_noise_metadata_malformed_range_and_priced_subset(tmp_path: Path) -> None:
    config = reviewed.load_reviewed_config()
    assert config['noise_tests']['levels'] == [0, 0.005, 0.01, 0.02]
    policy_keys = {
        'exclude_any_boundary_near',
        'near_threshold',
        'weak_separation_threshold',
        'source',
        'rationale',
        'status',
        'provisional',
        'reviewed',
    }
    for distribution in ('interior_train', 'wide_valid_train', 'boundary_challenge', 'ood_test'):
        assert set(config[distribution]['acceptance_margin_policy']) == policy_keys
    for distribution in ('interior_train', 'wide_valid_train', 'boundary_challenge', 'ood_test'):
        for parameter in PARAMETER_NAMES:
            assert {'lower', 'upper', 'source', 'rationale', 'status', 'provisional', 'reviewed'} <= set(config[distribution]['parameter_ranges'][parameter])
    config['interior_train']['parameter_ranges']['kappa_slow']['upper'] = 0.1
    malformed = tmp_path / 'bad.yaml'
    malformed.write_text(yaml.safe_dump(config), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid range'):
        reviewed.load_reviewed_config(malformed)
    good = reviewed.load_reviewed_config()
    records = reviewed._priced(reviewed.sample_distribution('interior_train', 4, config=good).query('accepted'), cap=1)
    assert len(records) == 1 and records[0]['finite']
    assert 'dummy_surface_generator' not in Path(reviewed.__file__).read_text(encoding='utf-8')

def test_reviewed_config_population_and_price_cap_contract() -> None:
    config = reviewed.load_reviewed_config()
    expected = {
        'interior_train': (10000, 500),
        'wide_valid_train': (5000, 250),
        'boundary_challenge': (2000, 250),
        'ood_test': (2000, 250),
    }
    assert {
        distribution: (config[distribution]['candidate_count'], config[distribution]['price_cap'])
        for distribution in expected
    } == expected


def test_persisted_reviewed_audit_artifact_and_failure_gate_contract() -> None:
    expected_artifacts = {
        'boundary_challenge_candidates.csv',
        'distribution_overlap.csv',
        'interior_accepted.csv',
        'interior_candidates.csv',
        'interior_rejected.csv',
        'ood_candidates.csv',
        'priced_surface_metrics.csv',
        'proximity_metrics.csv',
        'rejection_reasons.csv',
        'reviewed_sampling_decision.json',
        'reviewed_sampling_recommendations.md',
        'reviewed_sampling_summary.json',
        'wide_valid_candidates.csv',
    }
    assert reviewed.OUTPUT.is_dir(), 'reviewed sampling audit output is missing; run python -m src.audit_reviewed_sampling'
    assert {path.name for path in reviewed.OUTPUT.iterdir() if path.is_file()} == expected_artifacts
    summary = json.loads((reviewed.OUTPUT / 'reviewed_sampling_summary.json').read_text(encoding='utf-8'))
    decision = json.loads((reviewed.OUTPUT / 'reviewed_sampling_decision.json').read_text(encoding='utf-8'))
    assert summary['priced_surface_failures'] == 4
    assert summary['audit_pass'] is False
    assert decision['audit_pass'] is False
    assert decision['status'] == 'NEEDS_SAMPLER_CORRECTION'
    assert summary['challenge_label_counts'] == {
        'near_correlation_disk': 500,
        'near_feller': 500,
        'near_hard_bound': 500,
        'weak_separation': 500,
    }


def test_every_distribution_declares_containing_transform_support_and_shared_union() -> None:
    config = reviewed.load_reviewed_config()
    frames = {'interior_train': reviewed.sample_distribution('interior_train', 256, config=config), 'wide_valid_train': reviewed.sample_distribution('wide_valid_train', 256, config=config), 'boundary_challenge': reviewed.sample_challenge(256, config=config), 'ood_test': reviewed.sample_distribution('ood_test', 256, config=config)}
    containment = reviewed.validate_declared_range_containment(frames, config)
    flags = ['accepted_hard_bound_near', 'accepted_feller_near', 'accepted_correlation_disk_near', 'accepted_weak_slow_fast_separation']
    for distribution, frame in frames.items():
        assert all((entry['all_generated_samples_contained'] for entry in containment[distribution].values()))
        assert (frame['accepted_any_boundary_near'] == frame['accepted'] & frame[flags].any(axis=1)).all()

def test_noise_evidence_covers_exact_configured_levels_without_ready_gate() -> None:
    config = reviewed.load_reviewed_config()
    clean = reviewed._priced(reviewed.sample_distribution('interior_train', 8, config=config).query('accepted'), cap=2)
    detail, summary = reviewed._noise_evidence(clean, config)
    assert detail['noise_level'].unique().tolist() == [0.0, 0.005, 0.01, 0.02]
    assert len(detail) == len(clean) * 4
    assert not summary['participates_in_ready_gate'].any()
    assert summary['no_clipping_projection_or_drop'].all()

def test_freeze_preserves_prior_bounds_result_and_never_self_attests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reviewed, 'FREEZE', tmp_path)
    (tmp_path / 'decision.json').write_text(json.dumps({'decisive_evidence': {'prior_bounds_audit_pass': True}}), encoding='utf-8')
    decision = {'status': 'NEEDS_SAMPLER_CORRECTION', 'audit_pass': False, 'limitations': {}}
    reviewed._freeze({'audit_pass': False}, decision)
    reviewed._freeze({'audit_pass': False}, decision)
    evidence = json.loads((tmp_path / 'decision.json').read_text(encoding='utf-8'))['decisive_evidence']
    checksums = json.loads((tmp_path / 'source_checksums.json').read_text(encoding='utf-8'))
    assert evidence['prior_bounds_audit_pass'] is True
    assert evidence['current_full_validation_chain_passed'] is False
    assert evidence['current_full_validation_chain_status'] == 'PENDING_PRIMARY_RERUN'
    assert 'src/synthetic_dataset.py' in checksums
