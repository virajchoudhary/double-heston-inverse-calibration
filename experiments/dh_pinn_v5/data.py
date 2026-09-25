"""New evaluation sets for v5, priced by the frozen exact Fourier teacher.

The sampler is the v4 sampler verbatim (5D Latin hypercube over x, log tau, log v_slow,
log v_fast and the level scale), with new seeds declared in config.json.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent; V4 = HERE.parent / 'nifty_multifactor_v4'; ROOT = HERE.parents[1]
sys.path.insert(0, str(V4)); sys.path.insert(0, str(ROOT))
from literature_exact import batch_teacher, iv as exact_iv

C4 = json.loads((V4 / 'config.json').read_text()); PC = C4['pinn']; P0 = np.array(C4['published_double_slow_first'])


def sample(n, seed, tau_range=(7 / 365, 2.), x_half=.36):
    u = qmc.LatinHypercube(5, seed=seed).random(n)
    x = -x_half + 2 * x_half * u[:, 0]
    tau = np.exp(np.log(tau_range[0]) + np.log(tau_range[1] / tau_range[0]) * u[:, 1])
    vs = np.exp(np.log(PC['slow_state_domain'][0]) + np.log(PC['slow_state_domain'][1] / PC['slow_state_domain'][0]) * u[:, 2])
    vf = np.exp(np.log(PC['fast_state_domain'][0]) + np.log(PC['fast_state_domain'][1] / PC['fast_state_domain'][0]) * u[:, 3])
    scale = PC['scale_domain'][0] + np.ptp(PC['scale_domain']) * u[:, 4]
    p = np.tile(P0, (n, 1)).astype(float)
    p[:, [1, 6]] *= scale[:, None]; p[:, [2, 7]] *= np.sqrt(scale[:, None]); p[:, 4] = vs; p[:, 9] = vf
    return np.column_stack([x, vs, vf, tau]), p


def build(dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True); spec = json.loads((HERE / 'config.json').read_text())['new_datasets']; audit = {}
    for name, s in spec.items():
        f = dest / f'{name}.npz'
        if f.exists(): continue
        kw = {'tau_range': (7 / 365, 30 / 365), 'x_half': .12} if name == 'short_atm_diagnostic' else {}
        coords, p = sample(s['points'], s['seed'], **kw)
        if name == 'rad_pool':
            np.savez_compressed(f, coords=coords, params=p); audit[name] = {'points': len(coords)}; continue
        price, diag = batch_teacher(p, coords); vol = exact_iv(price, coords[:, 0], coords[:, -1]); valid = np.isfinite(vol)
        np.savez_compressed(f, coords=coords, params=p, price=price, iv=vol, iv_valid=valid)
        audit[name] = {'points': len(price), 'invalid_iv': int((~valid).sum()), **diag}
        print(name, audit[name], flush=True)
    return audit


if __name__ == '__main__':
    a = build(HERE / 'artifacts' / 'data')
    (HERE / 'artifacts' / 'data' / 'audit.json').write_text(json.dumps(a, indent=2, default=float) + '\n')
