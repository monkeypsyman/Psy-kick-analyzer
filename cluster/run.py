"""PCA + Ward cluster. Writes cluster artifact to data/cluster/.

Depends on: numpy, scipy.cluster.hierarchy
Used by:    you, from the command line or a notebook
"""

import csv
import json
from pathlib import Path

import numpy as np
from numpy.linalg import svd
from scipy.cluster.hierarchy import linkage, fcluster


FEATURES = ['duration_ms', 'attack_t_ms', 'raw_peak_ms',
            'valley_level_pct', 'reswell_level_pct',
            'body_100ms_pct', 'lvl_200ms_pct',
            'f_settled_hz', 'click_ratio_db', 'harmonic_ratio_2f0_db']


def run_cluster(rows, kind='kick', out_dir='data/cluster', verbose=True):
    """rows: list of flat fingerprint rows (from cluster.load).
    Returns a dict with matrices and assignments."""
    if len(rows) < 4:
        if verbose:
            print("need at least 4 samples")
        return None

    M, ids, names, flags = [], [], [], []
    for r in rows:
        vec = []
        agree = r.get('f_settled_agreement_hz')
        n_excl = r.get('f_settled_n_excluded') or 0
        f_amb = (agree is not None and agree > 5.0) or n_excl >= 2
        flags.append(f_amb)
        ids.append(r.get('_source_id', ''))
        names.append(r.get('_source_file', ''))
        for f in FEATURES:
            v = r.get(f)
            if f == 'f_settled_hz' and f_amb:
                v = np.nan
            vec.append(np.nan if v is None else float(v))
        M.append(vec)
    M = np.array(M)

    col_med = np.nanmedian(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_med, inds[1])

    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1
    Z = (M - mu) / sd

    U, S, Vt = svd(Z - Z.mean(0), full_matrices=False)
    ev = (S ** 2) / np.sum(S ** 2)
    L = linkage(Z, method='ward')

    assignments = {ids[i]: {} for i in range(len(ids))}
    for k in (2, 3, 4):
        lab = fcluster(L, k, criterion='maxclust')
        for i, l in enumerate(lab):
            assignments[ids[i]]['cluster_k%d' % k] = int(l)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        'kind': kind,
        'n_rows': len(rows),
        'features': FEATURES,
        'explained_variance': ev.tolist(),
        'pc1_loadings': dict(zip(FEATURES, Vt[0].tolist())),
        'pc2_loadings': dict(zip(FEATURES, Vt[1].tolist())),
        'ward_merge_costs': L[:, 2].tolist(),
        'assignments': assignments,
        'manifest': [{'id': ids[i], 'source': names[i],
                      'f_ambiguous': bool(flags[i])}
                     for i in range(len(ids))],
    }
    with open(out_dir / ('%s_clusters.json' % kind), 'w') as f:
        json.dump(artifact, f, indent=2)

    with open(out_dir / ('%s_assignments.csv' % kind), 'w',
              newline='') as f:
        cols = ['id', 'source', 'f_ambiguous'] + FEATURES \
             + ['cluster_k2', 'cluster_k3', 'cluster_k4']
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows):
            row = [ids[i], names[i], bool(flags[i])]
            row += [r.get(feat) for feat in FEATURES]
            row += [assignments[ids[i]].get('cluster_k%d' % k)
                    for k in (2, 3, 4)]
            w.writerow(row)

    if verbose:
        print("=" * 74)
        print("  BASELINE CLUSTER - %d rows (kind=%s)" % (len(rows), kind))
        print("=" * 74)
        print("  explained variance (PC1..PC5): %s" % np.round(ev[:5], 3))
        print("\n  PC1 loadings (top axes):")
        for j in np.argsort(-np.abs(Vt[0]))[:6]:
            print("    %14s  %+.3f" % (FEATURES[j], Vt[0, j]))
        for k in (2, 3, 4):
            print("\n  k=%d:" % k)
            for i in range(len(ids)):
                tag = "  *f-ambig" if flags[i] else ""
                print("    c%d  %s%s"
                      % (assignments[ids[i]]['cluster_k%d' % k],
                         Path(names[i]).stem, tag))
        print("\n  saved: %s_clusters.json, %s_assignments.csv"
              % (kind, kind))

    return {'ids': ids, 'names': names, 'flags': flags,
            'M': M, 'Z': Z, 'L': L, 'explained': ev,
            'assignments': assignments, 'artifact': artifact}