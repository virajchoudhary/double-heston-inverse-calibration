from __future__ import annotations
from pathlib import Path
import pytest
from src.audit_reviewed_sampling import canonical_feller_margin, load_reviewed_config, sample_challenge, sample_distribution, validate_ood_support
from src.synthetic_dataset import assign_reviewed_distribution_splits

def test_reviewed_config_contract() -> None:
    c = load_reviewed_config()
    assert c['noise_tests']['levels'] == [0, 0.005, 0.01, 0.02]
    assert set(('hard_constraints', 'interior_train', 'wide_valid_train', 'boundary_challenge', 'ood_test')) <= set(c)

def test_feller_transform_and_conditional_separation() -> None:
    c = load_reviewed_config()
    f = sample_distribution('interior_train', 100, config=c)
    assert (f.kappa_fast - f.kappa_slow >= 0.8).all()
    assert (f.slow_feller_margin > 0).all()
    assert (f.fast_feller_margin > 0).all()
    assert canonical_feller_margin(1, 0.1, 0.3) > 0

def test_polar_hard_bounds_and_disk() -> None:
    c = load_reviewed_config()
    f = sample_distribution('wide_valid_train', 100, config=c)
    assert f.hard_bounds_valid.all() and (f.correlation_disk_value < 1).all()

def test_challenge_labels_and_ood_isolation() -> None:
    c = load_reviewed_config()
    challenge = sample_challenge(40, config=c)
    ood = sample_distribution('ood_test', 40, config=c)
    assert challenge.accepted.all()
    assert set(challenge.regime) == {'near_feller', 'weak_separation', 'near_hard_bound', 'near_correlation_disk'}
    validate_ood_support(ood, c)
    assert not ood.split.isin(['train', 'validation']).any()

def test_reviewed_split_isolation() -> None:
    ids = ['a', 'b', 'c', 'd']
    assert set(assign_reviewed_distribution_splits(ids, 'boundary_challenge').values()) == {'challenge_excluded'}
    assert set(assign_reviewed_distribution_splits(ids, 'ood_test').values()) == {'ood_test'}
    with pytest.raises(ValueError):
        assign_reviewed_distribution_splits(ids, 'other')

def test_malformed_reviewed_config_fails(tmp_path: Path) -> None:
    p = tmp_path / 'bad.yaml'
    p.write_text('hard_constraints: {}\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_reviewed_config(p)
