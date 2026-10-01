"""Fingerprint clustering.

Pipeline shape: analyzer produces fingerprints; this package consumes
them. Two faces:
  - batch: load a folder of fingerprints, produce clusters, write an
    artifact to data/cluster/
  - query: given one fingerprint, find the nearest cluster

Public surface:
    from cluster import collect_rows, run_cluster
    from cluster.query import find, nearest_to, by_cluster
"""

from .load import load_fingerprint, collect_rows
from .run import run_cluster

__all__ = ['load_fingerprint', 'collect_rows', 'run_cluster']