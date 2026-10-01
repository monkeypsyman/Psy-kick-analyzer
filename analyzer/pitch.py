"""f0 consensus + settled fundamental.

Depends on: numpy, analyzer.pitch_methods, analyzer.envelope
Used by:    analyzer.pipeline
"""

import numpy as np

from .config import F0_LO_HZ, F0_HI_HZ
from .envelope import _find_settled_body
from .pitch_methods import F0_METHODS


def measure_f0_multi(body, sr, lo_hz=F0_LO_HZ, hi_hz=F0_HI_HZ):
    """Run every registered method, MAD-exclude outliers, return median.
    Returns dict: {f0, methods, agreement_hz, window_ms, flag,
                   n_excluded, excluded}."""
    out = {'f0': None, 'methods': {}, 'agreement_hz': None,
           'window_ms': None, 'flag': None,
           'n_excluded': 0, 'excluded': []}
    if body is None or len(body) < 2048:
        out['flag'] = 'too_short'
        return out

    for name, fn in F0_METHODS.items():
        try:
            v = fn(body, sr, lo_hz, hi_hz)
        except Exception:
            v = None
        if v is not None:
            out['methods'][name] = float(v)

    vals = {k: v for k, v in out['methods'].items() if v is not None}
    if len(vals) < 2:
        if len(vals) == 1:
            out['f0'] = list(vals.values())[0]
            out['flag'] = 'single_method'
        else:
            out['flag'] = 'no_method_succeeded'
        return out

    arr = np.array(list(vals.values()))
    med = float(np.median(arr))
    devs = np.abs(arr - med)
    mad = float(np.median(devs))
    if mad < 0.5:
        mad = 1.0
    keep_mask = devs <= 3.0 * mad
    if keep_mask.sum() < 3 and len(arr) >= 3:
        order = np.argsort(devs)
        keep_mask = np.zeros(len(arr), dtype=bool)
        keep_mask[order[:3]] = True

    names = list(vals.keys())
    inliers = [names[i] for i in range(len(names)) if keep_mask[i]]
    excluded = [names[i] for i in range(len(names)) if not keep_mask[i]]
    out['excluded'] = excluded
    out['n_excluded'] = len(excluded)

    inlier_vals = [vals[k] for k in inliers]
    out['f0'] = float(np.median(inlier_vals))
    if len(inlier_vals) >= 2:
        out['agreement_hz'] = float(max(inlier_vals) - min(inlier_vals))
    if len(inlier_vals) < 3:
        out['flag'] = 'weak_consensus'
    return out


def settled_fundamental(t, f, env_t=None, env_v=None, seg_ms=None):
    """Median f0 over the settled body window (see _find_settled_body).
    Returns None if no settled body is found."""
    if len(t) < 4:
        return None
    if env_t is None or env_v is None or len(env_v) == 0:
        t_lo, t_hi = float(t[0]), float(t[-1])
    else:
        if seg_ms is None:
            seg_ms = float(env_t[-1]) if len(env_t) else float(t[-1])
        win = _find_settled_body(env_t, env_v, seg_ms)
        if win is None:
            return None
        t_lo, t_hi = win
    t_lo = max(t_lo, float(t[0]))
    t_hi = min(t_hi, float(t[-1]))
    if t_hi - t_lo < 20.0:
        return None
    grid = np.linspace(t_lo, t_hi, 50)
    f_grid = np.interp(grid, t, f)
    f_grid = f_grid[f_grid > 20.0]
    if len(f_grid) < 10:
        return None
    return float(np.median(f_grid))