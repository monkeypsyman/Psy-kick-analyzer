"""Query helpers over fingerprint rows. Pure Python; add SQLite later
if the corpus grows past ~10^4 and you want indexed lookups.

Depends on: numpy, cluster.load
Used by:    you
"""

import numpy as np


def find(rows, **predicates):
    """Filter rows by top-level key = value, or key = (min, max)."""
    out = []
    for r in rows:
        ok = True
        for k, v in predicates.items():
            rv = r.get(k)
            if isinstance(v, tuple) and len(v) == 2:
                if rv is None or not (v[0] <= rv <= v[1]):
                    ok = False
                    break
            else:
                if rv != v:
                    ok = False
                    break
        if ok:
            out.append(r)
    return out


def by_cluster(rows, k, cluster_id):
    """All rows whose cluster_k<k> == cluster_id."""
    key = 'cluster_k%d' % k
    return [r for r in rows if r.get(key) == cluster_id]


def nearest_to(rows, target, features=None, top=5):
    """Euclidean distance from target (a dict or flat row) to each row
    in the given feature space. Returns [(row, distance), ...] sorted."""
    from .run import FEATURES
    feats = features or FEATURES
    mu = np.array([_safe(target.get(f)) for f in feats])
    out = []
    for r in rows:
        v = np.array([_safe(r.get(f)) for f in feats])
        if np.any(np.isnan(mu)) or np.any(np.isnan(v)):
            continue
        d = float(np.linalg.norm(mu - v))
        out.append((r, d))
    out.sort(key=lambda x: x[1])
    return out[:top]


def _safe(v):
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan