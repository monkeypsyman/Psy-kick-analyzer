"""Onset detection and loop grouping.

S5 BUG C: onset detector locks onto the swell peak on swell-class kicks.
Do not fix without user sign-off; see PROJECT_STATE §5.3 (Bug C).
Touches: detect_onsets.

Depends on: numpy, scipy.signal.find_peaks
Used by:    analyzer.pipeline
"""

import numpy as np
from scipy.signal import find_peaks


def detect_onsets(x, sr, min_distance_ms=200):
    """BUG C: onset detector locks onto the swell peak on swell-class
    kicks. Mid's dual-class reading is the confirmed case.
    Do not fix without user sign-off."""
    win = max(4, int(8 * sr / 1000))
    hop = max(1, int(2 * sr / 1000))
    n = max(1, 1 + (len(x) - win) // hop)
    if n < 2:
        return np.array([0], dtype=int)
    e = np.array([float(np.sum(x[i * hop:i * hop + win] ** 2))
                  for i in range(n)])
    if len(e) > 3:
        e = np.convolve(e, np.ones(3) / 3, mode='same')
    d = np.maximum(np.diff(e, prepend=0.0), 0.0)
    if d.max() <= 0:
        return np.array([0], dtype=int)
    md = max(1, int(min_distance_ms * sr / 1000 / hop))
    pk, _ = find_peaks(d, distance=md, prominence=d.max() * 0.15)
    onsets = pk * hop if len(pk) else np.array([0], dtype=int)
    near_start = int(20 * sr / 1000)
    if len(onsets) > 0 and onsets[0] < near_start:
        onsets = np.concatenate([[0], onsets[1:]])
    elif len(onsets) > 0 and onsets[0] > int(10 * sr / 1000):
        head = x[:int(10 * sr / 1000)]
        if len(head) > 0 and np.max(np.abs(head)) > 0.3 * np.max(np.abs(x)):
            min_sep = int(min_distance_ms * sr / 1000)
            keep = onsets > min_sep
            onsets = np.concatenate([[0], onsets[keep]])
    return onsets


def group_onsets(x, sr, onsets, window_ms=400, threshold=0.85,
                 max_lag_ms=8):
    """Group onsets into classes by lag-tolerant correlation."""
    n = len(onsets)
    if n == 0:
        return np.array([]), {}
    if n == 1:
        return np.array([0]), {0: [0]}
    wl = int(min(window_ms, len(x) / sr * 1000) * sr / 1000)
    segs, valids = [], []
    for o in onsets:
        end = min(o + wl, len(x))
        s = x[o:end].astype(np.float32).copy()
        real_len = len(s)
        idx = _attack_align_idx(s, sr, head_ms=30)
        if idx > 0:
            s = s[idx:]
            real_len -= idx
        segs.append(s)
        valids.append(real_len)
    for i, s in enumerate(segs):
        v = valids[i]
        if v > 0:
            segs[i] = s - float(np.mean(s[:v]))
    max_lag = int(max_lag_ms * sr / 1000)
    labels = np.full(n, -1, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cid
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            c = _lag_corr(segs[i], segs[j], max_lag, valids[i], valids[j])
            if c > threshold:
                labels[j] = cid
        cid += 1
    cls = {}
    for i, l in enumerate(labels):
        cls.setdefault(int(l), []).append(i)
    return labels, cls


def _attack_align_idx(seg, sr, head_ms=30):
    n = min(int(head_ms * sr / 1000), len(seg))
    if n < 4:
        return 0
    return int(np.argmax(np.abs(seg[:n])))


def _lag_corr(a, b, max_lag, va, vb):
    best = -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a_off, b_off = -lag, 0
        elif lag > 0:
            a_off, b_off = 0, lag
        else:
            a_off, b_off = 0, 0
        m = min(va - a_off, vb - b_off)
        if m < 64:
            continue
        aa = a[a_off:a_off + m]
        bb = b[b_off:b_off + m]
        na = np.sqrt(np.sum(aa * aa)) + 1e-10
        nb = np.sqrt(np.sum(bb * bb)) + 1e-10
        c = float(np.sum(aa * bb) / (na * nb))
        if c > best:
            best = c
    return best