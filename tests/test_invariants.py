"""Invariant tests. Runs on whatever WAVs are present in tests/fixtures/.

These assert STRUCTURAL properties that must hold for any healthy kick
regardless of which file it came from. They survive retiring any file.

Optional pins are read from tests/expected.json. If absent, skipped.

Run:  pytest tests/test_invariants.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.pipeline import analyze_file

FIXTURE_DIR = Path(os.environ.get('PSY_KICK_FIXTURES',
                                  Path(__file__).parent / 'fixtures'))
EXPECTED_FILE = Path(__file__).parent / 'expected.json'


def _fixtures():
    if not FIXTURE_DIR.exists():
        return []
    return sorted(list(FIXTURE_DIR.glob('*.wav'))
                  + list(FIXTURE_DIR.glob('*.csv')))


def _pins():
    if not EXPECTED_FILE.exists():
        return {}
    try:
        return json.loads(EXPECTED_FILE.read_text())
    except Exception:
        return {}


@pytest.mark.parametrize('path', _fixtures() or [None])
def test_structural_invariants(path):
    if path is None:
        pytest.skip("no fixtures in tests/fixtures/")
    res = analyze_file(str(path), verbose=False)
    assert res['kicks'], "no kicks found in %s" % path.name
    for k in res['kicks']:
        dur = k.get('duration_ms')
        atk = k.get('attack_t_ms')
        val = k.get('valley_t_ms')
        res_t = k.get('reswell_t_ms')
        f0 = k.get('f_settled_hz')
        agree = k.get('f_settled_agreement_hz')
        n_excl = k.get('f_settled_n_excluded')

        if dur is not None:
            assert 100.0 < dur < 600.0, "%s dur=%s" % (path.name, dur)

        if atk is not None and val is not None:
            assert atk <= val + 5.0, \
                "%s attack=%s valley=%s" % (path.name, atk, val)
        if val is not None and res_t is not None:
            assert val <= res_t + 5.0, \
                "%s valley=%s reswell=%s" % (path.name, val, res_t)

        if f0 is not None:
            assert 20.0 < f0 < 120.0, "%s f0=%s" % (path.name, f0)

        if agree is not None and agree < 5.0 and n_excl is not None:
            assert n_excl <= 1, \
                "%s agree=%s n_excl=%s" % (path.name, agree, n_excl)

        methods = (k.get('detail', {}).get('pitch', {})
                   .get('f_settled_methods', {}))
        assert len(methods) >= 2, \
            "%s only %d methods: %s" % (path.name, len(methods), methods)


@pytest.mark.parametrize('path', _fixtures() or [None])
def test_optional_pins(path):
    pins = _pins()
    if not pins:
        pytest.skip("no expected.json")
    key = path.name if path else None
    if not key or key not in pins:
        pytest.skip("no pin for this file")
    spec = pins[key]
    res = analyze_file(str(path), verbose=False)
    k = res['kicks'][0]
    for field, rng in spec.items():
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            v = k.get(field)
            assert v is not None, "%s: %s is None" % (path.name, field)
            assert rng[0] <= v <= rng[1], \
                "%s: %s=%s not in %s" % (path.name, field, v, rng)