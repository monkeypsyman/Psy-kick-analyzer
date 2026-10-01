"""f0 estimators. Registry pattern.

Adding a method:
    @f0_method('name')
    def _f0_name(body, sr, lo_hz, hi_hz):
        ...
        return float_or_None

That's the whole extension mechanism. `measure_f0_multi` in pitch.py
loops over the registry. No other file needs to change.

Depends on: numpy, scipy.signal.find_peaks, scipy.signal.get_window,
            scipy.fft.rfft/rfftfreq
Optional:   librosa (for pYIN; unavailable in Pyodide)
Used by:    analyzer.pitch
"""

import numpy as np
from scipy.signal import find_peaks, get_window
from scipy.fft import rfft, rfftfreq

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


F0_METHODS = {}


def f0_method(name):
    """Register an f0 method. Signature: fn(body, sr, lo, hi) -> float|None."""
    def deco(fn):
        F0_METHODS[name] = fn
        return fn
    return deco


def pitch_cycle_length(x, sr, min_c=3):
    """Cycle-by-cycle pitch from zero-crossing intervals."""
    s = np.sign(x)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0] + 1
    if len(zc) < 2:
        return np.array([]), np.array([])
    t_o, f_o = [], []
    for k in range(len(zc) - 1):
        n = zc[k + 1] - zc[k]
        if n < min_c:
            continue
        t_o.append((zc[k] + zc[k + 1]) / 2 / sr * 1000)
        f_o.append(0.5 * sr / n)
    return np.array(t_o), np.array(f_o)


def pitch_pyin(x, sr, fmin=30.0, fmax=None):
    """pYIN via librosa. Returns (t_ms, f0) or (None, None)."""
    if not HAS_LIBROSA or len(x) < 512:
        return None, None
    if fmax is None:
        fmax = min(2000.0, sr / 2 * 0.95)
    need = int(2.2 * sr / fmin)
    fl = max(2048, need)
    fl = min(fl, len(x) if len(x) % 2 == 0 else len(x) - 1)
    if fl % 2:
        fl -= 1
    if fl < 512:
        return None, None
    try:
        f0, _, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=fl)
        hop = 512
        t = np.arange(len(f0)) * hop / sr * 1000
        return t, np.where(np.isnan(f0), 0, f0)
    except Exception:
        return None, None


# -------- registered estimators --------

@f0_method('spectral')
def _f0_spectral(body, sr, lo_hz, hi_hz):
    """First prominent spectral peak in band (bin resolution)."""
    N = 8192
    b = body[:N] if len(body) >= N else np.pad(body, (0, N - len(body)))
    spec = np.abs(rfft(b * get_window('hann', N)))
    fr = rfftfreq(N, 1 / sr)
    band = (fr >= lo_hz) & (fr <= hi_hz)
    if not band.any() or spec[band].max() <= 0:
        return None
    pk, _ = find_peaks(spec[band], height=spec[band].max() * 0.1)
    if not len(pk):
        return None
    return float(fr[band][pk[0]])


@f0_method('zc')
def _f0_zc(body, sr, lo_hz, hi_hz):
    """Median zero-crossing-interval frequency. Fragile on chirps."""
    z = body - body.mean()
    s = np.sign(z)
    s[s == 0] = 1
    zc = np.where(np.diff(s) != 0)[0]
    if len(zc) <= 4:
        return None
    fz = 0.5 / (np.diff(zc) / sr)
    fz = fz[(fz > lo_hz) & (fz < 400.0)]
    if len(fz) < 3:
        return None
    return float(np.median(fz))


@f0_method('autocorr')
def _f0_autocorr(body, sr, lo_hz, hi_hz):
    """First AC peak after first zero-crossing (S4 fix - no shoulder bias)."""
    a = body - body.mean()
    if len(a) < 64:
        return None
    ac = np.correlate(a, a, mode='full')[len(a) // 2:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    sgn = np.sign(ac)
    sgn[sgn == 0] = 1
    dz = np.diff(sgn)
    zc_ac = np.where(dz < 0)[0] + 1
    if len(zc_ac) == 0:
        return None
    first_zc = int(zc_ac[0])
    pk_ac, _ = find_peaks(ac, height=0.0)
    pk_ac = pk_ac[pk_ac > first_zc]
    if len(pk_ac) == 0:
        return None
    lag = int(pk_ac[0])
    if lag <= 0:
        return None
    return float(sr / lag)


@f0_method('pyin')
def _f0_pyin(body, sr, lo_hz, hi_hz):
    """pYIN median. Skipped when librosa is unavailable (Pyodide)."""
    if not HAS_LIBROSA:
        return None
    fmin_py = max(lo_hz, 2.2 * sr / len(body))
    if fmin_py >= hi_hz:
        return None
    _, py = pitch_pyin(body, sr, fmin=fmin_py, fmax=hi_hz)
    if py is None:
        return None
    v = py[py > 0]
    if len(v) < 3:
        return None
    return float(np.median(v))