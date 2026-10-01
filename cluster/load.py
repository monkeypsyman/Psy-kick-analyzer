"""Load .fingerprint.json files into flat rows.

Depends on: json, numpy
Used by:    cluster.run, cluster.query
"""

import json
from pathlib import Path


def load_fingerprint(fp, kind='kick'):
    """Read one .fingerprint.json and return a list of flat rows
    (one per kick class)."""
    with open(fp) as f:
        d = json.load(f)
    if d.get('kind') != kind:
        return []
    rows = []
    for k in d.get('kicks', []):
        row = dict(k)
        row['_source_id'] = d.get('id')
        row['_source_file'] = d.get('source')
        row['_schema_version'] = d.get('schema_version')
        rows.append(row)
    return rows


def collect_rows(folder, kind='kick', verbose=True):
    """Walk a folder of .fingerprint.json, return a list of flat rows."""
    folder = Path(folder)
    rows = []
    for fp in sorted(folder.glob('*.fingerprint.json')):
        try:
            rows.extend(load_fingerprint(fp, kind=kind))
        except Exception as e:
            if verbose:
                print("  ! %s: %s" % (fp.name, e))
    if verbose:
        print("loaded %d row(s) from %s" % (len(rows), folder))
    return rows