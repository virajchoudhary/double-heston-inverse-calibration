"""Deterministic audit for the reviewed Double Heston sampling designs."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from .constants import CALL_OPTION, PARAMETER_NAMES, PUT_OPTION
from .constraints import validate_parameters
from .double_heston import price_double_heston_surface
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/parameter_sampling_REVIEWED.yaml'
OUTPUT = ROOT / 'outputs/reviewed_sampling_audit'
FREEZE = ROOT / 'outputs/engine_freeze'
STRIKES = np.asarray([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
MATURITIES = np.asarray([0.05, 0.25, 0.5, 1.0, 2.0, 5.0])

def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n', encoding='utf-8')
    return p

def _range(x: Any, label: str) -> None:
    if not isinstance(x, dict):
        raise ValueError(f'Missing range: {label}')
    for k in ('lower', 'upper', 'source', 'rationale', 'status', 'provisional', 'reviewed'):
        if k not in x:
            raise ValueError(f'{label} missing {k}')
    if not np.isfinite([float(x['lower']), float(x['upper'])]).all() or float(x['lower']) >= float(x['upper']):
        raise ValueError(f'Invalid range: {label}')

def load_reviewed_config(path: str | Path=CONFIG_PATH) -> dict[str, Any]:
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    needed = {'hard_constraints', 'interior_train', 'wide_valid_train', 'boundary_challenge', 'ood_test', 'noise_tests'}
    if not isinstance(c, dict) or needed - set(c):
        raise ValueError('Reviewed config missing required top-level sections')
    if list(c['noise_tests'].get('levels', [])) != [0, 0.005, 0.01, 0.02]:
        raise ValueError('noise_tests.levels must be exactly [0, 0.005, 0.01, 0.02]')
    for n in PARAMETER_NAMES:
        _range(c['hard_constraints'].get(n), f'hard_constraints.{n}')
    for d in ('interior_train', 'wide_valid_train', 'boundary_challenge', 'ood_test'):
        if c[d].get('intended_training_role') != d:
            raise ValueError(f'{d} training role mismatch')
        policy = c[d].get('acceptance_margin_policy', {})
        if not np.isfinite([float(policy.get('near_threshold', np.nan)), float(policy.get('weak_separation_threshold', np.nan))]).all():
            raise ValueError(f'{d} missing boundary proximity thresholds')
        for n in PARAMETER_NAMES:
            _range(c[d].get('parameter_ranges', {}).get(n), f'{d}.{n}')
    if c['boundary_challenge'].get('train_validation_eligible') is not False or not c['boundary_challenge'].get('future_generation_policy', {}).get('challenge_requires_explicit_opt_in'):
        raise ValueError('challenge isolation contract invalid')
    if not c['ood_test'].get('evaluation_only') or c['ood_test'].get('train_validation_eligible') is not False:
        raise ValueError('OOD isolation contract invalid')
    return c

def canonical_feller_margin(k: float, t: float, s: float) -> float:
    return float((2 * k * t - s * s) / (2 * k * t + s * s))

def _scale(u: float, x: Any) -> float:
    return float(x[0]) + u * (float(x[1]) - float(x[0]))

def _lhs(n: int, seed: int, attempt: int=0) -> np.ndarray:
    return qmc.LatinHypercube(d=10, seed=seed + 104729 * attempt).random(n)

def _split(i: int, d: str) -> str:
    if d == 'boundary_challenge':
        return 'challenge_excluded'
    if d == 'ood_test':
        return 'ood_test'
    return 'train' if i % 20 < 14 else 'validation' if i % 20 < 17 else 'test'

def _boundary_predicates(hard_distance: float, slow_feller: float, fast_feller: float, correlation_margin: float, ordering_margin: float, policy: dict[str, Any]) -> dict[str, bool]:
    """Return the one config-derived union used by flags and audit acceptance."""
    near = float(policy['near_threshold'])
    weak = float(policy['weak_separation_threshold'])
    return {'hard_bound': hard_distance <= near, 'feller': min(slow_feller, fast_feller) <= near, 'correlation_disk': correlation_margin <= near, 'weak_separation': ordering_margin <= weak}

def _polar(ur: float, ua: float, section: dict[str, Any], hard: dict[str, Any], radius: Any | None=None) -> tuple[float, float]:
    r = _scale(ur, radius or section['correlation_radius'])
    env = section['correlation_component_envelope']
    a = np.linspace(0, 2 * np.pi, 8193)
    rs = r * np.cos(a)
    rf = r * np.sin(a)
    ok = (rs >= max(float(env['rho_slow'][0]), float(hard['rho_slow']['lower']))) & (rs <= min(float(env['rho_slow'][1]), float(hard['rho_slow']['upper']))) & (rf >= max(float(env['rho_fast'][0]), float(hard['rho_fast']['lower']))) & (rf <= min(float(env['rho_fast'][1]), float(hard['rho_fast']['upper'])))
    choices = a[ok]
    if not len(choices):
        raise ValueError('no valid polar angle for declared component envelope')
    angle = choices[min(int(ua * len(choices)), len(choices) - 1)]
    return (float(r * np.cos(angle)), float(r * np.sin(angle)))

def _row(i: int, u: np.ndarray, d: str, c: dict[str, Any], regime: str='normal', over: dict[str, Any] | None=None, attempt: int=0) -> dict[str, Any]:
    over = over or {}
    sec = c[d]
    h = c['hard_constraints']
    r = sec['parameter_ranges']
    interval = lambda n: [r[n]['lower'], r[n]['upper']]
    ks = _scale(u[0], over.get('kappa_slow', interval('kappa_slow')))
    ts = _scale(u[1], over.get('theta_slow', interval('theta_slow')))
    vs = _scale(u[2], over.get('v0_slow', interval('v0_slow')))
    tf = _scale(u[3], over.get('theta_fast', interval('theta_fast')))
    vf = _scale(u[4], over.get('v0_fast', interval('v0_fast')))
    gap = over.get('gap', sec.get('separation_gap', 0.25))
    gap = _scale(u[5], gap) if isinstance(gap, (list, tuple)) else float(gap)
    lo = max(float(r['kappa_fast']['lower']), ks + gap)
    hi = float(over.get('kappa_fast_upper', r['kappa_fast']['upper']))
    if lo >= hi:
        raise ValueError('conditional kappa range empty')
    kf = ks + gap if over.get('force_gap') else lo + u[6] * (hi - lo)
    f = over.get('feller_fraction', sec.get('feller_fraction', [0.15, 0.8]))
    fs = _scale(u[7], f)
    ff = _scale(u[8], f)
    fs = min(fs, 0.999 * float(h['sigma_slow']['upper']) / np.sqrt(2 * ks * ts))
    ff = min(ff, 0.999 * float(h['sigma_fast']['upper']) / np.sqrt(2 * kf * tf))
    ss = fs * np.sqrt(2 * ks * ts)
    sf = ff * np.sqrt(2 * kf * tf)
    rs, rf = _polar(u[5], u[9], sec, h, over.get('correlation_radius'))
    v = {'kappa_slow': ks, 'theta_slow': ts, 'sigma_slow': ss, 'rho_slow': rs, 'v0_slow': vs, 'kappa_fast': kf, 'theta_fast': tf, 'sigma_fast': sf, 'rho_fast': rf, 'v0_fast': vf}
    canonical = validate_parameters(v)
    hd = [min((v[n] - float(h[n]['lower'])) / (float(h[n]['upper']) - float(h[n]['lower'])), (float(h[n]['upper']) - v[n]) / (float(h[n]['upper']) - float(h[n]['lower']))) for n in PARAMETER_NAMES]
    hard_ok = all((x >= 0 for x in hd))
    reasons = list(canonical['violations']) + ([] if hard_ok else ['hard_numerical_safety_bounds'])
    slow = canonical_feller_margin(ks, ts, ss)
    fast = canonical_feller_margin(kf, tf, sf)
    disk = rs * rs + rf * rf
    order = (kf - ks) / (float(h['kappa_fast']['upper']) - float(h['kappa_slow']['lower']))
    policy = sec['acceptance_margin_policy']
    predicates = _boundary_predicates(min(hd), slow, fast, 1 - disk, order, policy)
    reasons += ['ordinary_training_margin_policy'] if d in ('interior_train', 'wide_valid_train') and policy.get('exclude_any_boundary_near') and any(predicates.values()) else []
    accepted = not reasons
    return {'candidate_id': i, 'distribution': d, 'regime': regime, 'intended_training_role': d, 'split': _split(i, d), 'attempt': attempt, **v, 'hard_bounds_valid': hard_ok, 'canonical_valid': bool(canonical['is_valid']), 'slow_feller_margin': slow, 'fast_feller_margin': fast, 'correlation_disk_value': disk, 'correlation_margin': 1 - disk, 'ordering_margin': order, 'minimum_hard_bound_distance': min(hd), 'accepted_hard_bound_near': bool(accepted and predicates['hard_bound']), 'accepted_feller_near': bool(accepted and predicates['feller']), 'accepted_correlation_disk_near': bool(accepted and predicates['correlation_disk']), 'accepted_weak_slow_fast_separation': bool(accepted and predicates['weak_separation']), 'accepted_any_boundary_near': bool(accepted and any(predicates.values())), 'accepted': accepted, 'primary_rejection_reason': reasons[0] if reasons else '', 'rejection_reasons': ';'.join(reasons)}

def sample_distribution(distribution: str, count: int | None=None, seed: int | None=None, config: dict[str, Any] | None=None) -> pd.DataFrame:
    c = config or load_reviewed_config()
    n = int(count or c[distribution]['candidate_count'])
    s = int(seed if seed is not None else c['sampler']['seed'])
    rows = []
    for i, u in enumerate(_lhs(n, s)):
        try:
            rows.append(_row(i, u, distribution, c))
        except ValueError as exc:
            rows.append({'candidate_id': i, 'distribution': distribution, 'regime': 'normal', 'intended_training_role': distribution, 'split': _split(i, distribution), 'attempt': 0, 'accepted': False, 'primary_rejection_reason': str(exc), 'rejection_reasons': str(exc)})
    return pd.DataFrame(rows)

def validate_challenge_labels(f: pd.DataFrame) -> None:
    if not f.accepted.all():
        raise ValueError('challenge contains invalid rows')
    cols = {'near_feller': 'accepted_feller_near', 'weak_separation': 'accepted_weak_slow_fast_separation', 'near_hard_bound': 'accepted_hard_bound_near', 'near_correlation_disk': 'accepted_correlation_disk_near'}
    for label, col in cols.items():
        if not f.loc[f.regime == label, col].all():
            raise ValueError(f'challenge label mismatch: {label}')

def sample_challenge(count: int | None=None, seed: int | None=None, config: dict[str, Any] | None=None) -> pd.DataFrame:
    c = config or load_reviewed_config()
    sec = c['boundary_challenge']
    n = int(count or sec['candidate_count'])
    s = int(seed if seed is not None else c['sampler']['seed'])
    labels = list(sec['regimes'])
    sizes = [n // 4] * 4
    for j in range(n % 4):
        sizes[j] += 1
    rows = []
    for j, (label, size) in enumerate(zip(labels, sizes, strict=True)):
        for x, u in enumerate(_lhs(size, s + 7919 * (j + 1))):
            over = {}
            if label == 'near_feller':
                over = {'feller_fraction': sec['regimes'][label]['feller_fraction'], 'kappa_fast_upper': 6.0}
            elif label == 'weak_separation':
                over = {'gap': sec['regimes'][label]['separation_gap'], 'force_gap': True}
            elif label == 'near_hard_bound':
                over = {'v0_slow': [0.0065, 0.018], 'v0_fast': [0.00325, 0.012]}
            elif label == 'near_correlation_disk':
                over = {'correlation_radius': sec['regimes'][label]['correlation_radius']}
            rows.append(_row(len(rows), u, 'boundary_challenge', c, label, over))
    f = pd.DataFrame(rows)
    validate_challenge_labels(f)
    return f

def validate_ood_support(f: pd.DataFrame, c: dict[str, Any]) -> None:
    normal = float(c['wide_valid_train']['parameter_ranges']['kappa_fast']['upper'])
    if not (f.kappa_fast > normal).all() or not f.accepted.all() or f.split.isin(['train', 'validation']).any():
        raise ValueError('OOD support/isolation proof failed')

def validate_declared_range_containment(frames: dict[str, pd.DataFrame], c: dict[str, Any]) -> dict[str, Any]:
    """Prove every emitted value is within its config-declared transform support."""
    result = {}
    for d, f in frames.items():
        result[d] = {}
        for n in PARAMETER_NAMES:
            values = f[n].to_numpy(float)
            declared = c[d]['parameter_ranges'][n]
            lo = float(declared['lower'])
            hi = float(declared['upper'])
            ok = bool(np.isfinite(values).all() and (values >= lo - 1e-12).all() and (values <= hi + 1e-12).all())
            if not ok:
                raise ValueError(f'{d}.{n} emitted values outside declared support')
            result[d][n] = {'declared_lower': lo, 'declared_upper': hi, 'observed_min': float(values.min()), 'observed_max': float(values.max()), 'all_generated_samples_contained': ok}
    return result

def _priced(f: pd.DataFrame, cap: int) -> list[dict[str, Any]]:
    out = []
    ids = np.linspace(0, len(f) - 1, min(cap, len(f)), dtype=int) if len(f) else []
    strikes = np.tile(STRIKES, len(MATURITIES) * 2)
    mats = np.repeat(MATURITIES, len(STRIKES) * 2)
    types = np.tile(np.repeat([CALL_OPTION, PUT_OPTION], len(STRIKES)), len(MATURITIES))
    for _, r in f.iloc[ids].iterrows():
        started = time.perf_counter()
        try:
            p = price_double_heston_surface(100.0, strikes, mats, 0.02, 0.01, types, r[PARAMETER_NAMES].to_numpy(float), node_count=64)
            calls = p[types == CALL_OPTION].reshape(len(MATURITIES), len(STRIKES))
            puts = p[types == PUT_OPTION].reshape(len(MATURITIES), len(STRIKES))
            loS = 100 * np.exp(-0.01 * mats)
            loK = strikes * np.exp(-0.02 * mats)
            low = np.where(types == CALL_OPTION, np.maximum(loS - loK, 0), np.maximum(loK - loS, 0))
            high = np.where(types == CALL_OPTION, loS, loK)
            atm = calls[:, 3] / 100
            out.append({'candidate_id': int(r.candidate_id), 'distribution': r.distribution, 'finite': bool(np.isfinite(p).all()), 'no_arbitrage_valid': bool(np.all(p >= low - 1e-08) & np.all(p <= high + 1e-08)), 'call_strike_monotonicity': bool(np.all(np.diff(calls, axis=1) <= 1e-09)), 'put_strike_monotonicity': bool(np.all(np.diff(puts, axis=1) >= -1e-09)), 'strike_convexity': bool(np.all(np.diff(calls, n=2, axis=1) >= -1e-08) & np.all(np.diff(puts, n=2, axis=1) >= -1e-08)), 'normalized_price_min': float(p.min() / 100), 'normalized_price_max': float(p.max() / 100), 'atm_price_min': float(atm.min()), 'atm_price_max': float(atm.max()), 'skew_proxy': float(calls[2, 2] - calls[2, 4]), 'smile_curvature': float(calls[2, 2] - 2 * calls[2, 3] + calls[2, 4]), 'term_structure': float(atm[-1] - atm[0]), 'slow_variance_contribution': float(r.v0_slow / (r.v0_slow + r.v0_fast)), 'fast_variance_contribution': float(r.v0_fast / (r.v0_slow + r.v0_fast)), 'runtime_seconds': time.perf_counter() - started, 'pricing_error': '', '_raw_prices': p})
        except Exception as e:
            out.append({'candidate_id': int(r.candidate_id), 'distribution': r.distribution, 'finite': False, 'no_arbitrage_valid': False, 'call_strike_monotonicity': False, 'put_strike_monotonicity': False, 'strike_convexity': False, 'runtime_seconds': time.perf_counter() - started, 'pricing_error': repr(e), '_raw_prices': None})
    return out

def _noise_evidence(clean: list[dict[str, Any]], c: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit raw multiplicative noise without clipping, projection, or row dropping."""
    strikes = np.tile(STRIKES, len(MATURITIES) * 2)
    mats = np.repeat(MATURITIES, len(STRIKES) * 2)
    types = np.tile(np.repeat([CALL_OPTION, PUT_OPTION], len(STRIKES)), len(MATURITIES))
    loS = 100 * np.exp(-0.01 * mats)
    loK = strikes * np.exp(-0.02 * mats)
    low = np.where(types == CALL_OPTION, np.maximum(loS - loK, 0), np.maximum(loK - loS, 0))
    high = np.where(types == CALL_OPTION, loS, loK)
    records = []
    for level_index, level in enumerate(c['noise_tests']['levels']):
        rng = np.random.default_rng(int(c['sampler']['seed']) + 5003 * level_index)
        for item in clean:
            raw = item['_raw_prices']
            record = {'candidate_id': item['candidate_id'], 'distribution': item['distribution'], 'noise_level': level, 'noise_model': 'raw_multiplicative_gaussian_no_clipping_projection_or_drop', 'retained_from_clean_pricing_failure': raw is None}
            if raw is None:
                record.update({key: False for key in ('finite', 'no_arbitrage_valid', 'call_strike_monotonicity', 'put_strike_monotonicity', 'strike_convexity')})
            else:
                noisy = raw * (1 + rng.normal(0, level, size=raw.shape))
                calls = noisy[types == CALL_OPTION].reshape(len(MATURITIES), len(STRIKES))
                puts = noisy[types == PUT_OPTION].reshape(len(MATURITIES), len(STRIKES))
                record.update({'finite': bool(np.isfinite(noisy).all()), 'no_arbitrage_valid': bool(np.all(noisy >= low - 1e-08) & np.all(noisy <= high + 1e-08)), 'call_strike_monotonicity': bool(np.all(np.diff(calls, axis=1) <= 1e-09)), 'put_strike_monotonicity': bool(np.all(np.diff(puts, axis=1) >= -1e-09)), 'strike_convexity': bool(np.all(np.diff(calls, n=2, axis=1) >= -1e-08) & np.all(np.diff(puts, n=2, axis=1) >= -1e-08))})
            records.append(record)
    detail = pd.DataFrame(records)
    checks = ['finite', 'no_arbitrage_valid', 'call_strike_monotonicity', 'put_strike_monotonicity', 'strike_convexity']
    summary = pd.DataFrame([{'noise_level': level, 'priced_surface_count': int(len(part)), **{f'{key}_failure_count': int((~part[key].astype(bool)).sum()) for key in checks}, 'all_surface_checks_pass': bool(part[checks].all(axis=1).all()), 'raw_noise_diagnostics_transfer_clean_validity': False, 'participates_in_ready_gate': False, 'no_clipping_projection_or_drop': True} for level, part in detail.groupby('noise_level', sort=True)])
    return (detail, summary)

def _diversity_by_distribution(priced: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Aggregate synthetic-only diversity diagnostics separately by distribution."""
    metrics = ('normalized_price_min', 'normalized_price_max', 'atm_price_min', 'atm_price_max', 'skew_proxy', 'smile_curvature', 'term_structure', 'slow_variance_contribution', 'fast_variance_contribution')
    result = {}
    for name, frame in frames.items():
        priced_part = priced.loc[priced['distribution'] == name]
        ranges = {metric: {'min': float(priced_part[metric].min()), 'max': float(priced_part[metric].max())} for metric in metrics if metric in priced_part and len(priced_part)}
        result[name] = {'synthetic_diagnostic_only': True, 'priced_surface_count': int(len(priced_part)), 'maturity_coverage_years': MATURITIES.tolist(), 'priced_metric_ranges': ranges, 'parameter_correlations': frame[PARAMETER_NAMES].corr().round(6).to_dict()}
    return result

def _freeze(summary: dict[str, Any], decision: dict[str, Any]) -> None:
    FREEZE.mkdir(parents=True, exist_ok=True)
    path = FREEZE / 'source_checksums.json'
    h = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for relative in ('src/double_heston.py', 'src/synthetic_dataset.py', 'configs/parameter_sampling_REVIEWED.yaml', 'src/audit_reviewed_sampling.py'):
        h[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    write_json(path, h)
    write_json(FREEZE / 'reviewed_sampling_summary.json', summary)
    write_json(FREEZE / 'reviewed_sampling_decision.json', decision)
    write_json(FREEZE / 'parameter_sampling_summary.json', summary)
    prior = json.loads((FREEZE / 'decision.json').read_text(encoding='utf-8')) if (FREEZE / 'decision.json').exists() else {}
    benchmark = prior.get('decisive_evidence', {})
    prior_bounds = benchmark.get('prior_bounds_audit_pass')
    if not isinstance(prior_bounds, bool):
        source = ROOT / 'outputs/parameter_bounds_audit/bounds_audit_summary.json'
        prior_bounds = json.loads(source.read_text(encoding='utf-8')).get('audit_pass') if source.exists() else None
    if not isinstance(prior_bounds, bool):
        raise ValueError('Prior bounds-audit result is unavailable; freeze evidence must not replace it with null')
    canonical = {'status': decision['status'], 'timestamp_utc': datetime.now(UTC).isoformat(), 'decisive_evidence': {'benchmark_present': benchmark.get('benchmark_present'), 'benchmark_pass': benchmark.get('benchmark_pass'), 'benchmark_tolerance_breaches': benchmark.get('benchmark_tolerance_breaches'), 'reference_unreliable_count': benchmark.get('reference_unreliable_count'), 'prior_bounds_audit_pass': prior_bounds, 'reviewed_sampling_audit_pass': decision['audit_pass'], 'current_full_validation_chain_passed': False, 'current_full_validation_chain_status': 'PENDING_PRIMARY_RERUN', 'trusted_external_finalization_recorded': False}, 'reviewed_sampling': decision, 'remaining_limitations': decision['limitations']}
    write_json(FREEZE / 'decision.json', canonical)
    (FREEZE / 'validation_commands.txt').write_text('python -m compileall .\npython -m pytest tests -q\npython -m src.run_independent_pricing_benchmark\npython -m src.audit_reviewed_sampling\npython -m src.run_double_heston_validation\npython -m src.run_smoke_test\npython -m src.evaluate_repricing\n\nThe reviewed audit never self-attests this chain; preserve PENDING_PRIMARY_RERUN until trusted external finalization records independently rerun results.\n', encoding='utf-8')

def run_audit() -> dict[str, Any]:
    c = load_reviewed_config()
    s = int(c['sampler']['seed'])
    fs = {'interior_train': sample_distribution('interior_train', seed=s, config=c), 'wide_valid_train': sample_distribution('wide_valid_train', seed=s + 1, config=c), 'boundary_challenge': sample_challenge(seed=s + 2, config=c), 'ood_test': sample_distribution('ood_test', seed=s + 3, config=c)}
    validate_ood_support(fs['ood_test'], c)
    containment = validate_declared_range_containment(fs, c)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    [(OUTPUT / name).unlink(missing_ok=True) for name in ('noise_surface_metrics.csv', 'noise_level_diagnostics.csv')]
    names = {'interior_train': 'interior', 'wide_valid_train': 'wide_valid', 'boundary_challenge': 'boundary_challenge', 'ood_test': 'ood'}
    for d, f in fs.items():
        f.to_csv(OUTPUT / f'{names[d]}_candidates.csv', index=False)
    fs['interior_train'].loc[fs['interior_train'].accepted].to_csv(OUTPUT / 'interior_accepted.csv', index=False)
    fs['interior_train'].loc[~fs['interior_train'].accepted].to_csv(OUTPUT / 'interior_rejected.csv', index=False)
    allf = pd.concat(fs.values(), ignore_index=True)
    rej = allf.loc[~allf.accepted]
    (rej.assign(reason=rej.rejection_reasons.str.split(';')).explode('reason').groupby('reason').size().rename('count').reset_index() if len(rej) else pd.DataFrame({'reason': [], 'count': []})).to_csv(OUTPUT / 'rejection_reasons.csv', index=False)
    prox = pd.DataFrame([{'distribution': d, 'denominator': 'accepted candidates', 'accepted_count': int(f.accepted.sum()), 'any_boundary_rate': float(f.loc[f.accepted, 'accepted_any_boundary_near'].mean()), 'hard_bound_rate': float(f.loc[f.accepted, 'accepted_hard_bound_near'].mean()), 'feller_rate': float(f.loc[f.accepted, 'accepted_feller_near'].mean()), 'weak_separation_rate': float(f.loc[f.accepted, 'accepted_weak_slow_fast_separation'].mean()), 'disk_rate': float(f.loc[f.accepted, 'accepted_correlation_disk_near'].mean())} for d, f in fs.items()])
    prox.to_csv(OUTPUT / 'proximity_metrics.csv', index=False)
    clean = sum((_priced(f.loc[f.accepted].sort_values('candidate_id'), int(c[d]['price_cap'])) for d, f in fs.items()), [])
    priced = pd.DataFrame([{key: value for key, value in item.items() if key != '_raw_prices'} for item in clean])
    priced['evidence_kind'] = 'clean_pricing'
    priced['noise_level'] = np.nan
    noise_detail, noise_summary = _noise_evidence(clean, c)
    noise_detail['evidence_kind'] = 'raw_noise_diagnostic'
    pd.concat([priced, noise_detail], ignore_index=True, sort=False).to_csv(OUTPUT / 'priced_surface_metrics.csv', index=False)
    overlap = []
    for a in fs:
        for b in fs:
            if a < b:
                overlap.append({'left_distribution': a, 'right_distribution': b, 'support_overlap_width_sum': float(sum((max(0, min(fs[a][n].max(), fs[b][n].max()) - max(fs[a][n].min(), fs[b][n].min())) for n in PARAMETER_NAMES))), 'ood_disjoint_by_declared_dimension': bool('ood_test' in (a, b) and (fs[a].kappa_fast.min() > fs[b].kappa_fast.max() or fs[b].kappa_fast.min() > fs[a].kappa_fast.max()))})
    pd.DataFrame(overlap).to_csv(OUTPUT / 'distribution_overlap.csv', index=False)
    acc = {d: {'candidate_count': len(f), 'accepted_count': int(f.accepted.sum()), 'acceptance_rate': float(f.accepted.mean())} for d, f in fs.items()}
    checks = ['finite', 'no_arbitrage_valid', 'call_strike_monotonicity', 'put_strike_monotonicity', 'strike_convexity']
    bad = int((~priced[checks].all(axis=1)).sum())
    p = prox.set_index('distribution')
    normal = p.loc[['interior_train', 'wide_valid_train']]
    margins = bool((normal['any_boundary_rate'] <= 0.1).all() and (normal[['hard_bound_rate', 'feller_rate', 'weak_separation_rate', 'disk_rate']] <= 0.05).all().all())
    passed = bool(acc['interior_train']['acceptance_rate'] >= 0.8 and acc['wide_valid_train']['acceptance_rate'] >= 0.65 and margins and (bad == 0) and fs['boundary_challenge'].accepted.all() and fs['ood_test'].accepted.all())
    summary = {'schema_version': '2.0', 'sampler': c['sampler'], 'candidate_population_policy': 'fixed requested populations; rejected and margin-excluded rows are retained; no accepted-row refill occurs', 'acceptance': acc, 'proximity_denominator': 'accepted candidates per distribution', 'boundary_union_definition': {'source': 'per-distribution acceptance_margin_policy', 'predicates': ['hard_bound', 'feller', 'correlation_disk', 'weak_separation'], 'accepted_any_boundary_near': 'accepted AND (hard_bound OR feller OR correlation_disk OR weak_separation)'}, 'declared_range_containment': containment, 'priced_surface_count': len(priced), 'priced_surface_failures': bad, 'clean_pricing_ready_gate_participates': True, 'noise_diagnostics': {'levels': noise_summary.to_dict(orient='records'), 'detail_artifact': 'priced_surface_metrics.csv where evidence_kind=raw_noise_diagnostic', 'raw_noise_surface_count': int(len(noise_detail)), 'raw_noise_diagnostics_transfer_clean_validity': False, 'participates_in_ready_gate': False, 'no_clipping_projection_or_drop': True}, 'challenge_label_counts': fs['boundary_challenge'].regime.value_counts().sort_index().to_dict(), 'ood_support_proof': {'dimension': 'kappa_fast', 'normal_support_upper': 10.0, 'ood_observed_min': float(fs['ood_test'].kappa_fast.min()), 'every_row_outside_normal_support': True, 'train_validation_assignments': int(fs['ood_test'].split.isin(['train', 'validation']).sum())}, 'diversity': {'synthetic_diagnostic_only': True, 'per_distribution': _diversity_by_distribution(priced, fs), 'row_level_metrics': 'priced_surface_metrics.csv where evidence_kind=clean_pricing'}, 'acceptance_criteria': {'interior_minimum': 0.8, 'wide_minimum': 0.65, 'any_boundary_max': 0.1, 'each_boundary_max': 0.05, 'priced_subset_requires_zero_failures': True}, 'audit_pass': passed, 'limitations': c['limitations']}
    decision = {'status': 'READY_FOR_SYNTHETIC_GENERATION' if passed else 'NEEDS_SAMPLER_CORRECTION', 'audit_pass': passed, 'no_ann_training_occurred': True, 'no_final_large_dataset_generated': True, 'limitations': c['limitations']}
    write_json(OUTPUT / 'reviewed_sampling_summary.json', summary)
    write_json(OUTPUT / 'reviewed_sampling_decision.json', decision)
    (OUTPUT / 'reviewed_sampling_recommendations.md').write_text('# Reviewed sampling audit\n\n- Deterministic latent-coordinate LHS is not physical-space LHS after conditional transforms.\n- Fixed candidate populations retain every rejection; no accepted-row refill occurs.\n- Challenge is excluded from ordinary training unless explicitly opted in.\n- OOD is evaluation-only and uses disjoint high-tail kappa_fast support.\n- Every declared parameter envelope is checked against every generated candidate.\n- Raw noise diagnostics are retained in priced_surface_metrics.csv with evidence_kind=raw_noise_diagnostic; they use no clipping, projection, or dropped rows, do not transfer clean-price validity, and do not participate in the READY gate.\n', encoding='utf-8')
    _freeze(summary, decision)
    return summary

def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
