"""Audio loading. WAV and CSV in, (samples, sr, meta) out.

Depends on: numpy, scipy.io.wavfile
Used by:    analyzer.pipeline
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile


def load_audio(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(path))
    ext = p.suffix.lower()
    if ext == '.wav':
        return _load_wav(p)
    if ext == '.csv':
        return _load_csv(p)
    raise ValueError("Unsupported: " + ext)


def _load_wav(p):
    sr, data = wavfile.read(str(p))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, float(sr), {'format': 'wav', 'sr_nominal': int(sr),
                          'path': str(p)}


def _load_csv(p):
    import pandas as pd
    df = pd.read_csv(str(p), sep=None, engine='python')
    df = df.dropna(axis=1, how='all')
    cols = df.columns.tolist()
    tname = cols[0]
    for c in cols:
        if 'time' in str(c).lower():
            tname = c
            break
    t = df[tname].values.astype(float)
    sig = [c for c in cols if c != tname]
    x = df[sig].mean(axis=1).values.astype(np.float32)
    sr = 1.0 / float(np.median(np.diff(t)))
    return x, sr, {'format': 'csv', 'sr_effective': sr, 'path': str(p)}
