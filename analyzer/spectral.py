"""STFT, spectral centroid, click detection, harmonic ratio.

Depends on: numpy, scipy.signal.get_window, scipy.fft.rfft/rfftfreq
Used by:    analyzer.pipeline
"""

import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


def compute_stft(x, sr, n_fft=1024, hop=None, pad_start=True):
    if hop is None:
        hop = n_fft // 4
    pad = n_fft // 2 if pad_start else 0
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    if pad > 0:
        x = np.pad(x, (pad, 0))
    win = get_window('hann', n_fft)
    nf = max(1, 1 + (len(x) - n_fft) // hop)
    S = np.zeros((n_fft // 2 + 1, nf), dtype=np.complex128)
    for i in range(nf):
        S[:, i] = rfft(x[i * hop:i * hop + n_fft] * win)
    times = (np.arange(nf) * hop + n_fft / 2 - pad) / sr * 1000
    return np.abs(S), rfftfreq(n_fft, 1 / sr), times


def spectral_centroid(S, freqs, times, envelope_t=None, envelope_v=None,
                      floor_pct=0.02):
    num = (freqs[:, None] * S).sum(axis=0)
    den = S.sum(axis=0)
    safe = np.where(den > 0, den, 1e-10)
    cen = num / safe
    if (envelope_t is not None and envelope_v is not None
            and len(envelope_v) > 0):
        peak = float(np.max(envelope_v))
        if peak > 0:
            floor = floor_pct * peak
            env_at_frames = np.interp(times, envelope_t, envelope_v)
            cen = np.where(env_at_frames > floor, cen, np.nan)
    return cen


def _body_frames(S, quantile=0.5):
    total = S.sum(axis=0)
    if not np.any(total > 0):
        return np.zeros(S.shape[1], dtype=bool)
    thr = np.quantile(total, quantile)
    return total >= thr


def detect_click(S, freqs, times, f0):
    """True + ratio(dB) if the first 3 frames have >2*f0 energy >6 dB
    above the body frames' >2*f0 energy. Else False, None."""
    if not f0 or f0 <= 0:
        return False, None
    hf = freqs > 2 * f0
    if not hf.any() or S.shape[1] < 5:
        return False, None
    click_frames = np.zeros(S.shape[1], dtype=bool)
    click_frames[:3] = True
    body_frames = _body_frames(S, 0.5) & ~click_frames
    if not body_frames.any() or not click_frames.any():
        return False, None
    hf_click = float(S[np.ix_(hf, click_frames)].mean()) + 1e-10
    hf_body = float(S[np.ix_(hf, body_frames)].mean()) + 1e-10
    r = float(20 * np.log10(hf_click / hf_body))
    return r > 6.0, r


def harmonic_ratio_db(S, freqs, times, f0, seg=None, sr=44100):
    """2*f0 vs f0 energy, from a dedicated 8192-pt FFT of the 150-300 ms
    body slice. Not from the STFT."""
    if not f0 or f0 <= 0 or seg is None or len(seg) < 4096:
        return None
    sr_f = float(sr)
    t_lo, t_hi = 150.0, min(300.0, len(seg) / sr_f * 1000.0)
    if t_hi - t_lo < 50.0:
        t_lo, t_hi = 100.0, min(250.0, len(seg) / sr_f * 1000.0)
        if t_hi - t_lo < 50.0:
            return None
    s0 = max(0, int(t_lo * 1e-3 * sr_f))
    s1 = min(len(seg), int(t_hi * 1e-3 * sr_f))
    if s1 - s0 < 2048:
        return None
    body = seg[s0:s1]
    n_fft = 8192
    if len(body) < n_fft:
        body = np.pad(body, (0, n_fft - len(body)))
    else:
        body = body[:n_fft]
    win = get_window('hann', n_fft)
    spec = np.abs(rfft(body * win))
    fr = rfftfreq(n_fft, 1 / sr_f)
    bw = max(2.0, (sr_f / n_fft) * 1.5)

    def band(fc):
        if fc <= 0:
            return 1e-10
        m = (fr >= fc - bw) & (fr <= fc + bw)
        if not m.any():
            return 1e-10
        return float(spec[m].mean()) + 1e-10

    return float(20 * np.log10(band(2 * f0) / band(f0)))