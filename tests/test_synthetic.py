"""Synthetic tests. No external files. Always run.

These test the LOGIC of the analyzer, not a specific data file.
They must survive any change to the corpus.

Run:  pytest tests/test_synthetic.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.pitch import settled_fundamental
from analyzer.pitch_methods import pitch_cycle_length
from analyzer.envelope import cycle_peak_envelope
from analyzer.pipeline import _classify, _content_id
from analyzer.pitch import measure_f0_multi


def _synth_kick(f0, sr=44100, dur_ms=220.0, peak_ms=40.0, attack_ms=3.0):
    n = int(dur_ms * 1e-3 * sr)
    t = np.arange(n) / sr
    tms = t * 1000.0
    env = np.exp(-((tms - peak_ms) ** 2) / (2.0 * 40.0 ** 2))
    a = int(attack_ms * 1e-3 * sr)
    env[:a] *= np.linspace(0.0, 1.0, a)
    return (env * np.sin(2 * np.pi * f0 * t)).astype(np.float32), sr


@pytest.mark.parametrize('f0', [45.0, 55.0, 65.0])
def test_synthetic_f0_recovery(f0):
    x, sr = _synth_kick(f0)
    t_p, f_p = pitch_cycle_length(x, sr)
    t_env, v_env = cycle_peak_envelope(x, sr)
    got = settled_fundamental(t_p, f_p, t_env, v_env)
    assert got is not None, "%s Hz -> None" % f0
    assert abs(got - f0) < 2.0, "%s Hz -> %.2f" % (f0, got)


def test_no_body_returns_none():
    sr = 44100
    t = np.arange(int(0.30 * sr)) / sr
    x = (np.exp(-t * 40.0) * np.sin(2 * np.pi * 50.0 * t)
         ).astype(np.float32)
    t_p, f_p = pitch_cycle_length(x, sr)
    t_env, v_env = cycle_peak_envelope(x, sr)
    got = settled_fundamental(t_p, f_p, t_env, v_env)
    assert got is None or abs(got - 50.0) > 5.0


@pytest.mark.parametrize('peak', [1.0, 5.0, 8.0])
def test_classify_fishtail_boundary(peak):
    assert _classify(peak) == 'fishtail'


@pytest.mark.parametrize('peak', [9.0, 20.0, 80.0])
def test_classify_swell_boundary(peak):
    assert _classify(peak) == 'swell'


def test_multi_method_agreement_on_synth():
    x, sr = _synth_kick(55.0)
    body = x[int(0.060 * sr):int(0.130 * sr)]
    r = measure_f0_multi(body, sr)
    assert r['f0'] is not None, "f0 None on clean synth"
    assert abs(r['f0'] - 55.0) < 3.0, "f0=%.2f" % r['f0']
    assert len(r['methods']) >= 2, "only %d methods ran" % len(r['methods'])


def test_content_id_stable():
    x1, sr = _synth_kick(55.0)
    x2, sr = _synth_kick(55.0)
    x3, sr = _synth_kick(60.0)
    assert _content_id(x1, sr) == _content_id(x2, sr)
    assert _content_id(x1, sr) != _content_id(x3, sr)