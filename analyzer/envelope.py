"""Envelope extraction, landmarks, and settled-body detection.

S5 BUG B: find_landmarks valley prominence threshold is relative to the
attack level, not the local envelope height. Saturated/rising kicks do
not dip 5% below attack. Do not fix without user sign-off;
see PROJECT_STATE §5.3 (Bug B). Touches: find_landmarks.

S5 BUG A: _find_settled_body returns None on late-peak kicks. Do not fix
without user sign-off; see PROJECT_STATE §5.3 (Bug A).
Touches: _find_settled_body. Called from analyzer.pitch.

Depends on: numpy, scipy.signal.find_peaks
Used by:    analyzer.pipeline, analyzer.pitch
"""

import numpy as np
from scipy.signal import find_peaks


def cycle_peak_envelope(x, sr):
    """Return (t_ms, |peak|) per zero-crossing-bounded half-cycle."""
    if len(x) < 4:
        return np.array([]), np.array([])
    s = np.sign(x)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0] + 1
    if len(zc) < 2:
        return np.array([]), np.array([])
    t_o, v_o = [], []
    for k in range(len(zc) - 1):
        a, b = zc[k], zc[k + 1]
        if b - a < 2:
            continue
        sub = x[a:b]
        idx = int(np.argmax(np.abs(sub)))
        t_o.append((a + idx) / sr * 1000)
        v_o.append(abs(float(sub[idx])))
    return np.array(t_o), np.array(v_o)


def sliding_window_envelope(x, sr, win_ms=2.0, hop_ms=0.25):
    w = max(1, int(win_ms * sr / 1000))
    hop = max(1, int(hop_ms * sr / 1000))
    a = np.abs(x)
    n = max(1, 1 + (len(a) - w) // hop)
    env = np.array([float(a[i * hop:i * hop + w].max())
                    for i in range(n)])
    return np.arange(n) * hop / sr * 1000, env


def find_landmarks(t, v, seg_peak):
    """BUG B: valley prominence is relative to attack, not local height.
    Do not fix without user sign-off. Returns the envelope landmark dict."""
    if len(t) < 4 or seg_peak <= 0:
        return {}
    within = t <= 5.0
    if within.any():
        idx_w = np.where(within)[0]
        i_a = int(idx_w[0] + np.argmax(v[idx_w]))
    else:
        i_a = int(np.argmax(v))
    atk_t = float(t[i_a])
    atk_lvl = float(v[i_a])
    if atk_lvl <= 0:
        return {}
    o = {'attack_t_ms': atk_t, 'attack_level': atk_lvl,
         'seg_peak': float(seg_peak)}
    after = t > atk_t + 1.0
    if not after.any():
        return o
    t_a = t[after]
    v_a = v[after]
    prom = 0.05 * atk_lvl
    valleys, _ = find_peaks(-v_a, prominence=prom)
    peaks, _ = find_peaks(v_a, prominence=prom)
    if len(valleys) > 0:
        iv = valleys[0]
        o['valley_t_ms'] = float(t_a[iv])
        o['valley_level_pct'] = float(100 * v_a[iv] / atk_lvl)
        rp = peaks[peaks > iv]
        if len(rp) > 0:
            ir = rp[0]
            o['reswell_t_ms'] = float(t_a[ir])
            o['reswell_level_pct'] = float(100 * v_a[ir] / atk_lvl)
    o['n_peaks'] = int(len(peaks))
    if len(peaks) >= 1:
        o['peak2_t_ms'] = float(t_a[peaks[0]])
        o['peak2_level_pct'] = float(100 * v_a[peaks[0]] / atk_lvl)
    if len(peaks) >= 2:
        o['peak3_t_ms'] = float(t_a[peaks[1]])
        o['peak3_level_pct'] = float(100 * v_a[peaks[1]] / atk_lvl)
    for q, k in [(100.0, 'body_100ms_pct'), (200.0, 'lvl_200ms_pct')]:
        o[k] = float(100 * np.interp(q, t, v) / atk_lvl)
    ab = np.where(v / atk_lvl > 0.01)[0]
    if len(ab):
        o['duration_ms'] = float(t[ab[-1]])
    return o


def _find_settled_body(t_env, v_env, seg_ms,
                       settle_offset_ms=20.0,
                       max_after_peak_ms=90.0,
                       min_body_ms=40.0,
                       floor_frac=0.3):
    """BUG A: returns None on late-peak kicks (Roundhouse, UnderGroover,
    FL Synth, Thin). Do not fix without user sign-off.
    Called from analyzer.pitch.settled_fundamental and measure_f0_multi
    via _resolve_body_and_f0."""
    if len(t_env) == 0 or float(np.max(v_env)) <= 0:
        return None
    vmax = float(np.max(v_env))
    after = t_env > 10.0
    if after.sum() < 3:
        return None
    idxs = np.where(after)[0]
    i_pk = int(idxs[np.argmax(v_env[idxs])])
    t_peak = float(t_env[i_pk])
    floor = floor_frac * vmax
    t_end_limit = t_peak + max_after_peak_ms
    i = i_pk
    while (i < len(v_env) - 1
           and t_env[i + 1] < t_end_limit
           and v_env[i + 1] >= floor):
        i += 1
    t_end = float(t_env[i])
    t_start_pref = t_peak + settle_offset_ms
    if t_end - t_start_pref >= min_body_ms:
        return (t_start_pref, t_end)
    if t_end - t_peak >= min_body_ms:
        return (t_peak, t_end)
    return None