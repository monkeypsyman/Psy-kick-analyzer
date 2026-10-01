"""Batch analyzer. Folder-walking wrapper around analyzer.pipeline.

Depends on: analyzer.pipeline
Used by:    you, from the command line or a notebook
"""

import time
import traceback
from pathlib import Path

from .pipeline import analyze_file


def batch_analyze(folder=None, paths=None, save_json=True,
                  max_files=None, progress_cb=None, verbose=True):
    if paths is None:
        folder = Path(folder) if folder else Path('.')
        paths = sorted(folder.glob('*.wav')) + sorted(folder.glob('*.csv'))
    else:
        paths = [Path(p) for p in paths]
    if max_files:
        paths = paths[:max_files]

    if verbose:
        print("Found %d file(s)\n" % len(paths))

    results = []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        if progress_cb:
            progress_cb(i, len(paths), p.name)
        if verbose:
            print("[%d/%d] %s ... " % (i, len(paths), p.name),
                  end='', flush=True)
        t1 = time.time()
        try:
            res = analyze_file(str(p), save_json=save_json, verbose=False)
            results.append(res)
            if verbose:
                print("%.1fs" % (time.time() - t1))
        except Exception as e:
            if verbose:
                print("FAILED: %s: %s" % (type(e).__name__, e))
                traceback.print_exc()
    dt = time.time() - t0
    if verbose:
        print("\nTotal: %.1fs  (%.1fs/file)\n"
              % (dt, dt / max(len(paths), 1)))
        print_summary(results)
    return results


def print_summary(results):
    if not results:
        print("(no results)")
        return
    print("=" * 94)
    print("%-42s %7s %6s %6s %6s %7s %6s %6s"
          % ('name', 'dur', 'atk', 'v%', 's%', 'f0', 'agree', 'click'))
    print("-" * 94)
    for r in results:
        if not r.get('kicks'):
            print("%-42s  (no kicks)" % Path(r['source']).stem)
            continue
        k = max(r['kicks'], key=lambda x: x['n_instances'])
        print("%-42s %7.1f %6.2f %6.1f %6.1f %7.2f %6.2f %6.1f"
              % (Path(r['source']).stem,
                 k.get('duration_ms', 0) or 0,
                 k.get('attack_t_ms', 0) or 0,
                 k.get('valley_level_pct', 0) or 0,
                 k.get('reswell_level_pct', 0) or 0,
                 k.get('f_settled_hz', 0) or 0,
                 k.get('f_settled_agreement_hz', -1)
                 if k.get('f_settled_agreement_hz') is not None else -1,
                 k.get('click_ratio_db', 0) or 0))
    print("=" * 94)


def time_one(path):
    t = time.time()
    analyze_file(str(path), save_json=False, verbose=False)
    dt = time.time() - t
    print("%s: %.2fs" % (Path(path).name, dt))
    return dt