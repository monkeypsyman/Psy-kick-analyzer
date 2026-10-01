# Psy-kick-analyzer

Psytrance kick analyzer. Python in the browser via Pyodide/WASM,
hosted on GitHub Pages. Produces flat JSON fingerprints from WAV
files. A separate `cluster/` package consumes those fingerprints.

**Live:** https://monkeypsyman.github.io/Psy-kick-analyzer/

## Pipeline

```
WAV  ->  fingerprint  ->  cluster
        (analyzer/)     (cluster/)
```

Information flows one direction. The fingerprint is a compressed
*description* of the audio, not a compressed audio file.

## Repository layout

```
index.html               browser shell (Pyodide + canvas)
analyzer_bundle.py       generated - do not edit
build.py                 concatenates analyzer/*.py -> bundle

analyzer/
  __init__.py            public surface
  config.py              cross-cutting constants
  io.py                  audio loading
  onsets.py              onset detection + loop grouping
  envelope.py            cycle-peak envelope, landmarks, settled body
  pitch_methods.py       f0 estimator registry
  pitch.py               f0 consensus + settled fundamental
  spectral.py            STFT, centroid, click, harmonic ratio
  pipeline.py            top-level: analyze_array / analyze_file
  batch.py               folder-walking wrapper

cluster/
  __init__.py
  load.py                .fingerprint.json -> flat rows
  run.py                 PCA + Ward, writes data/cluster/
  query.py               find / nearest_to / by_cluster

tests/
  test_synthetic.py      no external files needed
  test_invariants.py     runs on tests/fixtures/*.wav
  expected.json          optional manual pins (empty by default)

data/
  fingerprints/          .fingerprint.json files (committed)
  wav/                   source WAVs (gitignored)
  cluster/               cluster artifact (committed)
```

## Environment notes

- Pyodide runs CPython 3.12 on WebAssembly. numpy + scipy are
  prebuilt. **librosa is not available** - the three-method consensus
  (spectral + zc + autocorr) carries the f0 estimate.
- `AudioContext({ sampleRate: 44100 })` is mandatory. `decodeAudioData`
  silently resamples otherwise; every f0/harmonic measurement is wrong.
- `AnalyserNode` is not used. The 8192-pt FFT is computed in Python.

## Editing the code

The **package** (`analyzer/`, `cluster/`) is the source of truth for
editing and for tests. The **bundle** (`analyzer_bundle.py`) is what
the browser loads. Edit the package; regenerate the bundle:

```
python build.py
```

A GitHub Action runs `build.py` on every push to `main` that touches
`analyzer/**` and commits the rebuilt bundle automatically.

## Running the tests

```
pip install numpy scipy pytest
pytest tests/
```

`test_synthetic.py` always runs. `test_invariants.py` skips cleanly
when `tests/fixtures/` is empty.

## Clustering

```
python -c "
from cluster import collect_rows, run_cluster
rows = collect_rows('data/fingerprints')
run_cluster(rows, kind='kick', out_dir='data/cluster')
"
```

Writes `data/cluster/kick_clusters.json` and `kick_assignments.csv`.

## Fingerprint shape

Flat, JSON-safe, one file per source WAV. Top-level keys are the
query surface; `detail` holds the nested form; `_arrays` is transient
(stripped on save).

```json
{
  "schema_version": 1,
  "kind": "kick",
  "id": "kick-a1b2c3d4e5f6",
  "source": "Kick_1_Gs.wav",
  "meta": { },
  "loop": { },
  "kicks": [
    {
      "class_id": 0, "class": "swell", "raw_peak_ms": 0.25,
      "f_settled_hz": 42.0, "duration_ms": 277.8,
      "detail": { "envelope": { }, "pitch": { }, "content": { } }
    }
  ]
}
```

`id` is a content hash of the decoded PCM. Same WAV -> same ID ->
overwrite on re-analyze. Git history is the version store.

## Known bugs (do not fix without sign-off)

- **Bug A** `analyzer/envelope.py::_find_settled_body`
- **Bug B** `analyzer/envelope.py::find_landmarks`
- **Bug C** `analyzer/onsets.py::detect_onsets`

See PROJECT_STATE section 5.3.