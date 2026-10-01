const RAW = 'https://raw.githubusercontent.com/monkeypsyman/Psy-kick-analyzer/main/';
let pyodide = null;
let fingerprints = [];

const $ = id => document.getElementById(id);
const status = m => { $('status').textContent = m; };
const err = m => { $('err').textContent = m; };

async function boot() {
  status('Loading Pyodide...');
  pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/'
  });
  status('Loading numpy + scipy...');
  await pyodide.loadPackage(['numpy', 'scipy']);

  status('Fetching cluster package...');
  await loadClusterPackage();
  status('Ready. Drop .fingerprint.json files.');
}

async function fetchText(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error('fetch ' + url + ' -> ' + r.status);
  const text = await r.text();
  if (text.trim().startsWith('<')) throw new Error('Got HTML for ' + url);
  return text;
}

async function loadClusterPackage() {
  await pyodide.runPythonAsync(
    'import os\nos.makedirs("/cluster", exist_ok=True)\n'
  );
  const files = ['__init__.py', 'load.py', 'run.py', 'query.py'];
  for (const f of files) {
    const text = await fetchText(RAW + 'cluster/' + f);
    pyodide.FS.writeFile('/cluster/' + f, text);
  }
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/")');
  await pyodide.runPythonAsync('import cluster');
}

async function handleFiles(fileList) {
  err('');
  const files = Array.from(fileList);
  status('Reading ' + files.length + ' file(s)...');
  fingerprints = [];
  for (const f of files) {
    try {
      const text = await f.text();
      const data = JSON.parse(text);
      if (data.kind !== 'kick') {
        console.warn('skipping non-kick: ' + f.name);
        continue;
      }
      fingerprints.push(data);
    } catch (e) {
      console.warn('bad file ' + f.name + ': ' + e.message);
    }
  }
  status(fingerprints.length + ' fingerprint(s) loaded. Clustering...');
  await new Promise(r => setTimeout(r, 20));
  await runCluster();
}

async function runCluster() {
  try {
    pyodide.globals.set('_fps_json', JSON.stringify(fingerprints));
    const py = [
      'import cluster, json',
      '_fps = json.loads(_fps_json)',
      '_rows = []',
      'for _d in _fps:',
      "    if _d.get('kind') != 'kick':",
      '        continue',
      "    for _k in _d.get('kicks', []):",
      '        _row = dict(_k)',
      "        _row['_source_id'] = _d.get('id')",
      "        _row['_source_file'] = _d.get('source')",
      "        _row['_schema_version'] = _d.get('schema_version')",
      '        _rows.append(_row)',
      "try:",
      "    _res = cluster.run_cluster(_rows, kind='kick', out_dir='/tmp', verbose=False)",
      'except Exception as _e:',
      "    _out = {'error': str(type(_e).__name__) + ': ' + str(_e)}",
      'else:',
      '    if _res is None:',
      "        _out = {'error': 'need at least 4 rows', 'n_rows': len(_rows)}",
      '    else:',
      "        ev = _res['explained']",
      "        ev = ev.tolist() if hasattr(ev, 'tolist') else list(ev)",
      "        _assign = _res['assignments']",
      "        _ids = _res['ids']",
      "        _names = _res['names']",
      "        _flags = _res['flags']",
      "        _labels = {k: [] for k in (2, 3, 4)}",
      "        for _i in _ids:",
      "            _a = _assign.get(_i, {})",
      "            for _k in (2, 3, 4):",
      "                _labels[_k].append(_a.get('cluster_k' + str(_k), -1))",
      "        _out = {",
      "            'n_rows': len(_rows),",
      "            'explained': ev[:5],",
      "            'names': _names,",
      "            'flags': _flags,",
      "            'labels': _labels,",
      '        }',
      'json.dumps(_out)'
    ].join('\n');
    const res = await pyodide.runPythonAsync(py);
    const parsed = JSON.parse(res);
    if (parsed.error) {
      err(parsed.error + (parsed.n_rows ? ' (' + parsed.n_rows + ' rows)' : ''));
      status('');
      return;
    }
    renderResults(parsed);
    status('Clustered ' + parsed.n_rows + ' row(s).');
  } catch (e) {
    console.error(e);
    err('Cluster error: ' + e.message);
    status('');
  }
}

function renderResults(parsed) {
  $('results').style.display = 'block';

  let files = '';
  for (let i = 0; i < parsed.names.length; i++) {
    const name = (parsed.names[i] || '').split('/').pop();
    const tag = parsed.flags[i] ? '  <span class="amb">*f-ambig</span>' : '';
    files += '<span class="val">' + name + '</span>' + tag + '\n';
  }
  $('filesList').innerHTML = files;

  const ev = parsed.explained.map(x => x.toFixed(3)).join(', ');
  $('pcaOut').innerHTML =
    'explained variance (PC1..PC5):\n<span class="val">' + ev + '</span>';

  let out = '';
  for (let i = 0; i < parsed.names.length; i++) {
    const name = (parsed.names[i] || '').split('/').pop().slice(0, 22);
    const k2 = parsed.labels['2'][i];
    const k3 = parsed.labels['3'][i];
    const k4 = parsed.labels['4'][i];
    out += 'k2=<span class="val">' + k2 + '</span>  ' +
           'k3=<span class="val">' + k3 + '</span>  ' +
           'k4=<span class="val">' + k4 + '</span>  ' +
           name + '\n';
  }
  $('assignOut').innerHTML = out;
}

const drop = $('drop');
drop.onclick = () => $('file').click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = () => drop.classList.remove('hover');
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
};
$('file').onchange = e => { if (e.target.files.length) handleFiles(e.target.files); };

boot().catch(e => { console.error(e); err('Boot failed: ' + e.message); });