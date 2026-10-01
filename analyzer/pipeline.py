"""Top-level pipeline. This file reads top to bottom as the analyzer's
story. Delegate to the specific module for any change to a measurement.

    analyze_array / analyze_file          entry points
      -> normalize, detect_onsets, group_onsets
      -> for each class: _extract_segment -> analyze_kick
          -> envelope, landmarks, pitch, spectral
      -> _assemble_result

The fingerprint shape (flat, per PROJECT_STATE S7) is built in
analyze_kick.

Depends on: everything in this package
Used by:    index.html (via analyzer_bundle.py), analyzer.batch,
            cluster.load (indirectly, via data/fingerprints/)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import (
    SCHEMA_VERSION, KIND, ID_SECONDS, ID_HASH_LEN,
    FISHTAIL_PEAK_MAX_MS,
)
from .envelope import (
    cycle_peak_envelope, sliding_window_envelope, find_landmarks,
)
from .io import load_audio
from .onsets import detect_onsets, group_onsets
from .pitch import measure_f0_multi, settled_fundamental
from .pitch_methods import pitch_cycle_length, pitch_pyin
from .spectral import (
    compute_stft, spectral_centroid, detect_click, harmonic_ratio_db,
)


# =====================================================================
# public entry points
# =====================================================================
def analyze_file(path, window_ms=500, headroom_ms=40,
                 save_json=False, verbose=False):
    x, sr, meta = load_audio(path)
    res = _run(x, sr, meta, source=str(path),
               window_ms=window_ms, headroom_ms=headroom_ms)
    if verbose:
        print_report(res)
    if save_json:
        out = Path(path).with_suffix('.fingerprint.json')
        with open(str(out), 'w') as f:
            json.dump(to_json(res), f, indent=2)
        if verbose:
            print("Saved: " + str(out))
    return res


def analyze_array(x, sr, source='<array>', window_ms=500, headroom_ms=40,
                  verbose=False):
    meta = {'format': 'array', 'sr_nominal': int(sr), 'path': source}
    res = _run(np.asarray(x, dtype=np.float32), float(sr), meta,
               source=source, window_ms=window_ms,
               headroom_ms=headroom_ms)
    if verbose:
        print_report(res)
    return res


# =====================================================================
# the pipeline
# =====================================================================
def _run(x, sr, meta, source, window_ms, headroom_ms):
    """Ordered stages. Read this to see what the analyzer does."""
    meta = _fill_meta(meta, x, sr)
    xn = _normalize(x)
    onsets = detect_onsets(xn, sr)
    labels, classes = group_onsets(xn, sr, onsets, window_ms=window_ms)

    kicks = []
    for cid in sorted(classes.keys()):
        row = _analyze_class(xn, sr, onsets, classes[cid], cid,
                             window_ms, headroom_ms)
        if row is not None:
            kicks.append(row)

    return {
        'schema_version': SCHEMA_VERSION,
        'kind': KIND,
        'id': _content_id(x, sr),
        'source': str(source),
        'meta': meta,
        'loop': _loop_summary(onsets, labels, classes),
        'kicks': kicks,
    }


def _analyze_class(xn, sr, onsets, idxs, class_id, window_ms, headroom_ms):
    seg, onset = _extract_segment(xn, sr, onsets, idxs,
                                  window_ms, headroom_ms)
    if seg is None:
        return None
    return analyze_kick(seg, sr, class_id=class_id,
                        n_instances=len(idxs),
                        onset_samples=[int(onsets[i]) for i in idxs],
                        first_onset_ms=onset / sr * 1000)


def analyze_kick(seg, sr, class_id, n_instances,
                 onset_samples, first_onset_ms):
    """Everything done to one kick segment. Returns a flat fingerprint
    row with a nested `detail` for the fields also promoted to the
    top level. `_arrays` (transient) is stripped by to_json()."""
    seg_peak = float(np.max(np.abs(seg))) if len(seg) else 1.0
    raw_peak_idx = int(np.argmax(np.abs(seg))) if len(seg) else 0
    raw_peak_ms = raw_peak_idx / sr * 1000

    t_env, v_env = cycle_peak_envelope(seg, sr)
    t_sw, v_sw = sliding_window_envelope(seg, sr)
    landmarks = find_landmarks(t_env, v_env, seg_peak)
    f0, f0_zc, f0_methods, f0_agr, f0_win, f0_flag, f0_nex, f0_excl = \
        _resolve_f0(seg, sr, t_env, v_env)
    t_p, f_p = pitch_cycle_length(seg, sr)
    t_py, f_py = pitch_pyin(seg, sr)
    S, freqs, times = compute_stft(seg, sr)
    cen = spectral_centroid(S, freqs, times, t_sw, v_sw, floor_pct=0.02)
    click, click_db = detect_click(S, freqs, times, f0)
    harm_db = harmonic_ratio_db(S, freqs, times, f0, seg=seg, sr=int(sr))

    pitch_detail = {
        'f_settled_hz': f0,
        'f_settled_zc': f0_zc,
        'f_settled_methods': f0_methods,
        'f_settled_agreement_hz': f0_agr,
        'f_settled_window_ms': f0_win,
        'f_settled_flag': f0_flag,
        'f_settled_n_excluded': f0_nex,
        'f_settled_excluded': f0_excl,
        'has_pyin': t_py is not None,
    }
    content_detail = {
        'click_present': click,
        'click_ratio_db': click_db,
        'harmonic_ratio_2f0_db': harm_db,
    }

    row = {
        'class_id': int(class_id),
        'n_instances': int(n_instances),
        'onset_samples': list(onset_samples),
        'first_onset_ms': float(first_onset_ms),
        'seg_peak': float(seg_peak),
        'raw_peak_ms': float(raw_peak_ms),
        'class': _classify(raw_peak_ms),

        # flat query fields
        'f_settled_hz': None,
        'f_settled_agreement_hz': f0_agr,
        'duration_ms': _g(landmarks, 'duration_ms'),
        'attack_t_ms': _g(landmarks, 'attack_t_ms'),
        'valley_t_ms': _g(landmarks, 'valley_t_ms'),
        'valley_level_pct': _g(landmarks, 'valley_level_pct'),
        'reswell_t_ms': _g(landmarks, 'reswell_t_ms'),
        'reswell_level_pct': _g(landmarks, 'reswell_level_pct'),
        'body_100ms_pct': _g(landmarks, 'body_100ms_pct'),
        'lvl_200ms_pct': _g(landmarks, 'lvl_200ms_pct'),
        'click_present': click,
        'click_ratio_db': click_db,
        'harmonic_ratio_2f0_db': harm_db,

        # nested detail (source of truth for the flat fields above)
        'detail': {
            'envelope': landmarks,
            'pitch': pitch_detail,
            'content': content_detail,
        },

        # transient arrays (stripped by to_json, kept for browser render)
        '_arrays': {
            't_env': t_env, 'v_env': v_env,
            't_sw': t_sw, 'v_sw': v_sw,
            't_pitch': t_p, 'f_pitch': f_p,
            't_pyin': t_py, 'f_pyin': f_py,
            'S': S, 'freqs': freqs, 'times': times,
            'centroid': cen, 'seg': seg,
        },
    }
    row['f_settled_hz'] = f0
    row['detail']['pitch']['f_settled_hz'] = f0
    return row


# =====================================================================
# helpers
# =====================================================================
def _fill_meta(meta, x, sr):
    meta = dict(meta)
    meta['n_samples'] = int(len(x))
    meta['duration_ms'] = float(len(x) / sr * 1000)
    meta['nyquist_hz'] = float(sr / 2)
    meta['peak'] = float(np.abs(x).max())
    meta['dc_offset'] = float(x.mean())
    return meta


def _normalize(x):
    pk = np.abs(x).max()
    return x / pk if pk > 0 else x.copy()


def _extract_segment(xn, sr, onsets, idxs, window_ms, headroom_ms):
    onset = int(onsets[idxs[0]])
    after = onsets[onsets > onset]
    next_onset = int(after.min()) if len(after) else len(xn)
    headroom = int(headroom_ms * sr / 1000)
    end = min(onset + int(window_ms * sr / 1000),
              next_onset - headroom,
              len(xn))
    if end <= onset:
        return None, onset
    return xn[onset:end], onset


def _loop_summary(onsets, labels, classes):
    return {
        'n_onsets': int(len(onsets)),
        'n_classes': int(len(classes)),
        'class_sizes': [len(v) for v in classes.values()],
        'class_assignments': labels.tolist(),
        'onset_samples': onsets.tolist(),
    }


def _resolve_f0(seg, sr, t_env, v_env):
    """Wrapper around measure_f0_multi + settled_fundamental.
    BUG A: _find_settled_body (called from measure_f0_multi's
    _resolve_body_and_f0 path) fails on late-peak kicks. Do not fix
    without user sign-off."""
    seg_ms = len(seg) / sr * 1000.0
    from .envelope import _find_settled_body
    win = _find_settled_body(t_env, v_env, seg_ms) if len(v_env) else None
    if win is None:
        return None, None, {}, None, None, 'no_settled_body', 0, []
    b0 = max(0, int(win[0] * 1e-3 * sr))
    b1 = min(len(seg), int(win[1] * 1e-3 * sr))
    if b1 - b0 < 2048:
        return None, None, {}, None, None, 'body_too_short', 0, []
    body = seg[b0:b1]
    mm = measure_f0_multi(body, sr)
    t_p, f_p = pitch_cycle_length(seg, sr)
    zc_f0 = settled_fundamental(t_p, f_p, t_env, v_env)
    return (mm.get('f0'), zc_f0, mm.get('methods', {}),
            mm.get('agreement_hz'), win[1] - win[0], mm.get('flag'),
            mm.get('n_excluded', 0), mm.get('excluded', []))


def _classify(raw_peak_ms):
    """Fishtail if raw_peak_ms <= threshold, else swell."""
    if raw_peak_ms is None:
        return None
    return ('fishtail' if raw_peak_ms <= FISHTAIL_PEAK_MAX_MS
            else 'swell')


def _content_id(x, sr):
    """Content-hash ID over the head of the decoded PCM."""
    n = min(len(x), int(ID_SECONDS * sr))
    q = np.clip(x[:n] * 32767, -32768, 32767).astype(np.int16)
    h = hashlib.sha256(q.tobytes()).hexdigest()[:ID_HASH_LEN]
    return 'kick-' + h


def _g(d, k):
    return d.get(k) if isinstance(d, dict) else None


def to_json(result):
    """JSON-safe copy: strips the transient `_arrays` on each kick."""
    r = dict(result)
    r['kicks'] = []
    for k in result.get('kicks', []):
        kk = {kk: vv for kk, vv in k.items() if kk != '_arrays'}
        r['kicks'].append(kk)
    return r


# =====================================================================
# human-readable report
# =====================================================================
def print_report(r):
    print("=" * 66)
    print("  " + r['source'])
    print("=" * 66)
    m = r['meta']
    sr_val = m.get('sr_nominal') or m.get('sr_effective', 0)
    print("  Format:      " + str(m['format']))
    print("  SR:          %.1f Hz" % sr_val)
    print("  Nyquist:     %.1f Hz" % m['nyquist_hz'])
    print("  Duration:    %.1f ms" % m['duration_ms'])
    print("  Peak:        %.4f" % m['peak'])
    print("")
    print("  ID:          " + str(r.get('id')))
    print("  Onsets: %d   Classes: %d   Sizes: %s" %
          (r['loop']['n_onsets'], r['loop']['n_classes'],
           str(r['loop']['class_sizes'])))
    print("")
    env_keys = ['attack_t_ms', 'attack_level', 'valley_t_ms',
                'valley_level_pct', 'reswell_t_ms', 'reswell_level_pct',
                'peak2_t_ms', 'peak2_level_pct', 'peak3_t_ms',
                'peak3_level_pct', 'body_100ms_pct', 'lvl_200ms_pct',
                'duration_ms', 'n_peaks']
    for k in r['kicks']:
        print("  -- class %d  (%d occurrences)  [%s] --" %
              (k['class_id'], k['n_instances'], k.get('class', '?')))
        print("    segment peak: %.4f   raw peak at: %.2f ms" %
              (k['seg_peak'], k.get('raw_peak_ms', 0)))
        det = k.get('detail', {})
        env = det.get('envelope', {})
        for key in env_keys:
            if key in env:
                v = env[key]
                fmt = "%-22s %10.3f" if isinstance(v, float) \
                    else "%-22s %10d"
                print("    " + fmt % (key, v))
        p = det.get('pitch', {})
        if p.get('f_settled_hz') is not None:
            print("    %-22s %10.2f" % ('f_settled_hz', p['f_settled_hz']))
        if p.get('f_settled_agreement_hz') is not None:
            print("    %-22s %10.2f" % ('f_settled_agreement_hz',
                                        p['f_settled_agreement_hz']))
        if p.get('f_settled_methods'):
            print("    f_settled_methods:    " +
                  ", ".join("%s=%.1f" % (mm, vv)
                            for mm, vv in p['f_settled_methods'].items()))
        c = det.get('content', {})
        print("    %-22s %10s" %
              ('click_present', str(c.get('click_present'))))
        for kk in ['click_ratio_db', 'harmonic_ratio_2f0_db']:
            if c.get(kk) is not None:
                print("    %-22s %10.2f" % (kk, c[kk]))
        print("")