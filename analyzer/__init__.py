"""Psytrance kick analyzer.

Public surface:
    from analyzer import analyze_file, analyze_array, to_json, print_report

Everything else imports from its specific submodule:
    from analyzer.pitch import measure_f0_multi
    from analyzer.envelope import find_landmarks

Pipeline shape: WAV -> fingerprint -> cluster.
This package produces fingerprints. See `cluster/` for the consumer.
"""

from .pipeline import (
    analyze_file,
    analyze_array,
    analyze_kick,
    to_json,
    print_report,
)

__all__ = [
    'analyze_file',
    'analyze_array',
    'analyze_kick',
    'to_json',
    'print_report',
]