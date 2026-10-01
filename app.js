const SR = 44100;
const RAW = 'https://raw.githubusercontent.com/monkeypsyman/Psy-kick-analyzer/main/';
let pyodide = null;
let lastX = null;
let lastFingerprint = null;
let audioCtx = null, currentSrc = null;

function $(id) { return document.getElementById(id); }
function status(m) { $('status').textContent = m; }
function err(m) { $('err').textContent = m; }

async function boot() {
  status('Loading Pyodide...');
  pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/'
  });
  status('Loading numpy + scipy...');
  await pyodide.loadPackage(['numpy', 'scipy']);

  status('Fetching analyzer code...');
  const code = await loadAnalyzerCode();
  pyodide.FS.writeFile('/analyzer.py', code);
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/")');
  await pyodide.runPythonAsync('import analyzer');
  status('Ready. Drop a WAV.');
}

async function fetchText(url) {
  const bust = (url.indexOf('?') >= 0 ? '&' : '?') + 't=' + Date.now();
  const r = await fetch(url + bust, { cache: 'no-store' });
  if (!r.ok) throw new Error('fetch ' + url + ' -> ' + r.status);
  const text = await r.text();
  if (text.trim().startsWith('<')) {
    throw new Error('Got HTML for ' + url);
  }
  return text;
}

async function loadAnalyzerCode() {
  try {
    const text = await fetchText(RAW + 'analyzer_bundle.py');
    if (text.trim().length > 100) {
      status('Loaded analyzer_bundle.py');
      return text;
    }
  } catch (e) { }

  status('Bundle missing - fetching sources...');
  const files = [
    'analyzer/config.py', 'analyzer/io.py', 'analyzer/onsets.py',
    'analyzer/envelope.py', 'analyzer/pitch_methods.py',
    'analyzer/pitch.py', 'analyzer/spectral.py', 'analyzer/pipeline.py'
  ];
  const imports = [];
  const seen = new Set();
  const parts = [];
  for (const f of files) {
    const text = await fetchText(RAW + f);
    const lines = text.split('\n');
    const body = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const s = line.trim();
      const top = line.length === 0 || !/^\s/.test(line);
      if (top && (s.startsWith('from ') || s.startsWith('import '))) {
        let depth = (line.split('(').length - 1)
                  - (line.split(')').length - 1);
        const block = [line];
        let j = i + 1;
        while (depth > 0 && j < lines.length) {
          block.push(lines[j]);
          depth += (lines[j].split('(').length - 1)
                 - (lines[j].split(')').length - 1);
          j++;
        }
        const isRel = s.startsWith('from .') || s.startsWith('from ..');
        if (!isRel) {
          const key = block.join('\n').trim();
          if (!seen.has(key)) {
            seen.add(key);
            imports.push(block.join('\n'));
          }
        }
        i = j;
        continue;
      }
      body.push(line);
      i++;
    }
    parts.push('# ===== from ' + f + ' =====\n' + body.join('\n'));
  }
  status('Assembled from ' + files.length + ' source files');
  return imports.join('\n') + '\n\n' + parts.join('\n\n');
}

async function handleFile(file) {
  err('');
  status('Reading ' + file.name + '...');
  try {
    const buf = await file.arrayBuffer();
    lastX = await decodeWav(buf);
    status('Analyzing ' + (lastX.length / SR * 1000).toFixed(1) + ' ms...');
    await new Promise(function (r) { setTimeout(r, 20); });
    await analyze();
    status('Done.');
  } catch (e) {
    console.error(e);
    err('Error: ' + e.message);
    status('');
  }
}

function decodeWav(arrayBuffer) {
  return new Promise(function (resolve, reject) {
    const c = new OfflineAudioContext(1, SR, SR);
    c.decodeAudioData(arrayBuffer.slice(0), function (ab) {
      let mono;
      const ch0 = ab.getChannelData(0);
      if (ab.numberOfChannels > 1) {
        mono = new Float32Array(ch0.length);
        for (let i = 0; i < ch0.length; i++) {
          let s = 0;
          for (let ch = 0; ch < ab.numberOfChannels; ch++)
            s += ab.getChannelData(ch)[i];
          mono[i] = s / ab.numberOfChannels;
        }
      } else {
        mono = new Float32Array(ch0);
      }
      if (ab.sampleRate !== SR) {
        console.warn('File rate ' + ab.sampleRate + ' != ' + SR);
      }
      resolve(mono);
    }, reject);
  });
}

async function analyze() {
  pyodide.globals.set('_x', lastX);
  pyodide.globals.set('_sr', SR);
  const py = [
    'import analyzer, json, numpy as np',
    "_r = analyzer.analyze_array(_x, _sr, source='<browser>', verbose=False)",
    '_fp = analyzer.to_json(_r)',
    "_k = _r['kicks'][0] if _r['kicks'] else None",
    '_plot = None',
    'if _k is not None:',
    "    a = _k['_arrays']",
    "    S = a['S']",
    '    nf_t, nt_t = 128, 96',
    '    S_d = S',
    '    if S_d.shape[0] > nf_t:',
    '        bs = S_d.shape[0] // nf_t',
    '        S_d = S_d[:nf_t*bs].reshape(nf_t, bs, -1).mean(axis=1)',
    '    if S_d.shape[1] > nt_t:',
    '        bs = S_d.shape[1] // nt_t',
    '        S_d = S_d[:, :nt_t*bs].reshape(-1, nt_t, bs).mean(axis=2)',
    '    S_db = 20 * np.log10(np.maximum(S_d, 1e-10))',
    '    S_db = np.maximum(S_db, S_db.max() - 60)',
    '    denom = (S_db.max() - S_db.min() + 1e-10)',
    '    S_norm = np.round((S_db - S_db.min()) / denom * 255).astype(int)',
    "    cen_list = [None if np.isnan(v) else float(v) for v in a['centroid']]",
    '    _plot = {',
    "        't_env': a['t_env'].tolist(),",
    "        'v_env': a['v_env'].tolist(),",
    "        't_sw': a['t_sw'].tolist(),",
    "        'v_sw': a['v_sw'].tolist(),",
    "        't_pitch': a['t_pitch'].tolist(),",
    "        'f_pitch': a['f_pitch'].tolist(),",
    "        't_pyin': (a['t_pyin'].tolist() if a['t_pyin'] is not None else None),",
    "        'f_pyin': (a['f_pyin'].tolist() if a['f_pyin'] is not None else None),",
    "        'times': a['times'].tolist(),",
    "        'freqs': a['freqs'].tolist(),",
    "        'centroid': cen_list,",
    "        'seg': a['seg'].tolist(),",
    "        'S': S_norm.tolist(),",
    "        'env': _k['detail']['envelope'],",
    '    }',
    "json.dumps({'fp': _fp, 'plot': _plot})"
  ].join('\n');
  const res = await pyodide.runPythonAsync(py);
  const parsed = JSON.parse(res);
  lastFingerprint = parsed.fp;
  renderFingerprint(parsed.fp);
  if (parsed.plot) renderPanels(parsed.plot);
  $('panels').style.display = 'block';
  $('playRow').style.display = 'block';
}

function renderFingerprint(fp) {
  const k = fp.kicks && fp.kicks[0];
  if (!k) { $('fp').innerHTML = '<i>no kicks found</i>'; return; }
  function f(v)  { return v == null ? '-' : (typeof v === 'number' ? v.toFixed(3) : v); }
  function f2(v) { return v == null ? '-' : (typeof v === 'number' ? v.toFixed(2) : v); }
  $('fp').innerHTML =
    '<b>id</b> <span class="val">' + fp.id + '</span> &middot; ' +
    '<b>class</b> <span class="val">' + (k.class || '?') + '</span> ' +
    '(raw peak <span class="val">' + f2(k.raw_peak_ms) + ' ms</span>)<br>' +
    '<b>duration</b> <span class="val">' + f(k.duration_ms) + ' ms</span> &middot; ' +
    '<b>attack</b> <span class="val">' + f(k.attack_t_ms) + ' ms</span> &middot; ' +
    '<b>valley</b> <span class="val">' + f2(k.valley_t_ms) + ' ms @ ' + f(k.valley_level_pct) + '%</span> &middot; ' +
    '<b>reswell</b> <span class="val">' + f2(k.reswell_t_ms) + ' ms @ ' + f(k.reswell_level_pct) + '%</span><br>' +
    '<b>f0</b> <span class="val">' + f2(k.f_settled_hz) + ' Hz</span> ' +
    '(agree <span class="val">' + f2(k.f_settled_agreement_hz) + ' Hz</span>)<br>' +
    '<b>click</b> <span class="val">' + (k.click_present ? 'present' : 'none') + ' ' + f2(k.click_ratio_db) + ' dB</span> &middot; ' +
    '<b>2nd harm</b> <span class="val">' + f2(k.harmonic_ratio_2f0_db) + ' dB</span>';
}

function panel(id) {
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const W = 1200, H = 180;
  c.width = W * dpr; c.height = H * dpr;
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx: ctx, W: W, H: H, padL: 44, padR: 12, padT: 10, padB: 24 };
}

function axes(ctx, p, xmax, ylabels) {
  const pw = p.W - p.padL - p.padR, ph = p.H - p.padT - p.padB;
  ctx.strokeStyle = '#1a2028';
  for (let i = 0; i <= 6; i++) {
    const px = p.padL + (i / 6) * pw;
    ctx.beginPath(); ctx.moveTo(px, p.padT); ctx.lineTo(px, p.padT + ph);
    ctx.stroke();
  }
  for (let i = 0; i <= 4; i++) {
    const py = p.padT + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(p.padL, py); ctx.lineTo(p.padL + pw, py);
    ctx.stroke();
  }
  ctx.fillStyle = '#556';
  ctx.font = '10px ui-monospace, monospace';
  for (let i = 0; i <= 3; i++) {
    const t = Math.round((i / 3) * xmax);
    ctx.fillText(t + 'ms', p.padL + (i / 3) * pw + 2, p.H - 6);
  }
  if (ylabels) {
    for (let i = 0; i < ylabels.length; i++) {
      ctx.fillText(ylabels[i], p.padL - 40,
                   p.padT + (i / (ylabels.length - 1 || 1)) * ph + 3);
    }
  }
}

function renderPanels(p) {
  const env = p.env || {};
  const segN = p.seg.length;
  const tmax = Math.min(300, segN / SR * 1000);

  (function drawWave() {
    const pan = panel('p_wave');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    axes(pan.ctx, pan, tmax, ['+1', '0', '-1']);
    const nMax = Math.min(segN, Math.floor(tmax / 1000 * SR));
    pan.ctx.strokeStyle = '#4fd6e0'; pan.ctx.lineWidth = 1;
    pan.ctx.beginPath();
    for (let i = 0; i < nMax; i++) {
      const px = pan.padL + (i / nMax) * pw;
      const py = pan.padT + ph / 2 - p.seg[i] * ph / 2 * 0.9;
      if (i === 0) pan.ctx.moveTo(px, py); else pan.ctx.lineTo(px, py);
    }
    pan.ctx.stroke();
    drawMarks(pan, env, tmax);
  })();

  (function drawEnv() {
    const pan = panel('p_env');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    let maxv = 1e-9;
    for (let i = 0; i < p.v_env.length; i++) if (p.v_env[i] > maxv) maxv = p.v_env[i];
    for (let i = 0; i < p.v_sw.length; i++) if (p.v_sw[i] > maxv) maxv = p.v_sw[i];
    axes(pan.ctx, pan, tmax, [maxv.toFixed(2), (maxv / 2).toFixed(2), '0']);
    pan.ctx.strokeStyle = '#667'; pan.ctx.lineWidth = 1;
    pan.ctx.beginPath();
    for (let i = 0; i < p.t_sw.length; i++) {
      if (p.t_sw[i] > tmax) break;
      const px = pan.padL + (p.t_sw[i] / tmax) * pw;
      const py = pan.padT + ph - (p.v_sw[i] / maxv) * ph;
      if (i === 0) pan.ctx.moveTo(px, py); else pan.ctx.lineTo(px, py);
    }
    pan.ctx.stroke();
    pan.ctx.strokeStyle = '#ffb347'; pan.ctx.lineWidth = 1.4;
    pan.ctx.beginPath();
    for (let i = 0; i < p.t_env.length; i++) {
      if (p.t_env[i] > tmax) break;
      const px = pan.padL + (p.t_env[i] / tmax) * pw;
      const py = pan.padT + ph - (p.v_env[i] / maxv) * ph;
      if (i === 0) pan.ctx.moveTo(px, py); else pan.ctx.lineTo(px, py);
    }
    pan.ctx.stroke();
    drawMarks(pan, env, tmax);
  })();

  (function drawSpec() {
    const pan = panel('p_spec');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    axes(pan.ctx, pan, tmax, ['nyq', 'mid', '0 Hz']);
    const S = p.S;
    const nf = S.length, nt = S[0].length;
    const tmp = document.createElement('canvas');
    tmp.width = nt; tmp.height = nf;
    const ictx = tmp.getContext('2d');
    const img = ictx.createImageData(nt, nf);
    for (let fr = 0; fr < nf; fr++) {
      for (let t = 0; t < nt; t++) {
        const v = S[nf - 1 - fr][t];
        const idx = (fr * nt + t) * 4;
        img.data[idx]     = v;
        img.data[idx + 1] = Math.floor(v * v / 255);
        img.data[idx + 2] = Math.floor(v * 0.6);
        img.data[idx + 3] = 255;
      }
    }
    ictx.putImageData(img, 0, 0);
    pan.ctx.drawImage(tmp, pan.padL, pan.padT, pw, ph);
    drawMarks(pan, env, tmax);
  })();

  (function drawPitch() {
    const pan = panel('p_pitch');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    const vals = [];
    for (let i = 0; i < p.f_pitch.length; i++) if (p.f_pitch[i] > 20) vals.push(p.f_pitch[i]);
    if (p.f_pyin) for (let i = 0; i < p.f_pyin.length; i++) if (p.f_pyin[i] > 20) vals.push(p.f_pyin[i]);
    if (!vals.length) vals.push(80);
    let ymin = Math.min.apply(null, vals) * 0.85;
    let ymax = Math.max.apply(null, vals) * 1.15;
    if (ymax / ymin < 2) {
      const m = Math.sqrt(ymin * ymax); ymin = m / 1.4; ymax = m * 1.4;
    }
    axes(pan.ctx, pan, tmax, [ymax.toFixed(0), 'Hz']);
    function yOf(v) { return pan.padT + ph - (v - ymin) / (ymax - ymin) * ph; }
    pan.ctx.fillStyle = '#ffb347';
    for (let i = 0; i < p.t_pitch.length; i++) {
      if (p.t_pitch[i] > tmax) break;
      const v = p.f_pitch[i]; if (v <= 20) continue;
      const px = pan.padL + (p.t_pitch[i] / tmax) * pw;
      pan.ctx.beginPath(); pan.ctx.arc(px, yOf(v), 1.4, 0, Math.PI * 2); pan.ctx.fill();
    }
    if (p.t_pyin) {
      pan.ctx.fillStyle = '#ff5f6d';
      for (let i = 0; i < p.t_pyin.length; i++) {
        if (p.t_pyin[i] > tmax) break;
        const v = p.f_pyin[i]; if (v <= 20) continue;
        const px = pan.padL + (p.t_pyin[i] / tmax) * pw;
        pan.ctx.beginPath(); pan.ctx.arc(px, yOf(v), 1.4, 0, Math.PI * 2); pan.ctx.fill();
      }
    }
    const f0 = (lastFingerprint.kicks[0] || {}).f_settled_hz;
    if (f0 && f0 >= ymin && f0 <= ymax) {
      pan.ctx.strokeStyle = '#5ee08a'; pan.ctx.setLineDash([4, 3]); pan.ctx.lineWidth = 1;
      pan.ctx.beginPath();
      pan.ctx.moveTo(pan.padL, yOf(f0));
      pan.ctx.lineTo(pan.padL + pw, yOf(f0));
      pan.ctx.stroke();
      pan.ctx.setLineDash([]);
    }
  })();

  (function drawCent() {
    const pan = panel('p_cent');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    const vals = p.centroid.filter(function (v) { return v != null && v > 0; });
    const ymin = vals.length ? Math.min.apply(null, vals) * 0.8 : 100;
    const ymax = vals.length ? Math.max.apply(null, vals) * 1.2 : 20000;
    axes(pan.ctx, pan, tmax,
         [ymax.toFixed(0), ((ymin + ymax) / 2).toFixed(0), ymin.toFixed(0)]);
    function yOf(v) { return pan.padT + ph - (v - ymin) / (ymax - ymin) * ph; }
    pan.ctx.strokeStyle = '#a78bfa'; pan.ctx.lineWidth = 1.4;
    pan.ctx.beginPath();
    let started = false;
    for (let i = 0; i < p.times.length; i++) {
      if (p.times[i] > tmax) break;
      const v = p.centroid[i];
      if (v == null) { started = false; continue; }
      const px = pan.padL + (p.times[i] / tmax) * pw;
      const py = yOf(v);
      if (!started) { pan.ctx.moveTo(px, py); started = true; }
      else pan.ctx.lineTo(px, py);
    }
    pan.ctx.stroke();
  })();

  (function drawZoom() {
    const pan = panel('p_zoom');
    const pw = pan.W - pan.padL - pan.padR;
    const ph = pan.H - pan.padT - pan.padB;
    axes(pan.ctx, pan, 50, ['+1', '0', '-1']);
    const nMax = Math.min(segN, Math.floor(0.050 * SR));
    pan.ctx.strokeStyle = '#4fd6e0'; pan.ctx.lineWidth = 1;
    pan.ctx.beginPath();
    for (let i = 0; i < nMax; i++) {
      const px = pan.padL + (i / nMax) * pw;
      const py = pan.padT + ph / 2 - p.seg[i] * ph / 2 * 0.9;
      if (i === 0) pan.ctx.moveTo(px, py); else pan.ctx.lineTo(px, py);
    }
    pan.ctx.stroke();
    pan.ctx.fillStyle = 'rgba(255,179,71,0.95)';
    let maxv = 1e-9;
    for (let i = 0; i < p.v_env.length; i++) if (p.v_env[i] > maxv) maxv = p.v_env[i];
    for (let i = 0; i < p.t_env.length; i++) {
      if (p.t_env[i] > 50) break;
      const px = pan.padL + (p.t_env[i] / 50) * pw;
      const py = pan.padT + ph / 2 - (p.v_env[i] / maxv) * ph / 2 * 0.9;
      pan.ctx.beginPath(); pan.ctx.arc(px, py, 2, 0, Math.PI * 2); pan.ctx.fill();
    }
    drawMarks(pan, env, 50);
  })();
}

function drawMarks(pan, env, xmax) {
  const pw = pan.W - pan.padL - pan.padR;
  const ph = pan.H - pan.padT - pan.padB;
  const marks = [
    [env.attack_t_ms, '#ff5f6d'],
    [env.valley_t_ms, '#ffb347'],
    [env.reswell_t_ms, '#5ee08a']
  ];
  for (let m = 0; m < marks.length; m++) {
    const t = marks[m][0], c = marks[m][1];
    if (t == null || t < 0 || t > xmax) continue;
    const px = pan.padL + (t / xmax) * pw;
    pan.ctx.strokeStyle = c; pan.ctx.lineWidth = 1.2;
    pan.ctx.beginPath();
    pan.ctx.moveTo(px, pan.padT);
    pan.ctx.lineTo(px, pan.padT + ph);
    pan.ctx.stroke();
  }
}

function getCtx() {
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate: SR });
  return audioCtx;
}
function play() {
  if (!lastX) return;
  const c = getCtx();
  if (c.state === 'suspended') c.resume();
  stop();
  const buf = c.createBuffer(1, lastX.length, SR);
  buf.copyToChannel(lastX, 0);
  const src = c.createBufferSource();
  src.buffer = buf; src.connect(c.destination); src.start();
  currentSrc = src;
}
function stop() {
  if (currentSrc) { try { currentSrc.stop(); } catch (e) {} currentSrc = null; }
}
function downloadFp() {
  if (!lastFingerprint) return;
  const blob = new Blob([JSON.stringify(lastFingerprint, null, 2)],
                        { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (lastFingerprint.id || 'kick') + '.fingerprint.json';
  a.click();
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
}

const drop = $('drop');
drop.onclick = function () { $('file').click(); };
drop.ondragover = function (e) { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = function () { drop.classList.remove('hover'); };
drop.ondrop = function (e) {
  e.preventDefault(); drop.classList.remove('hover');
  const f = e.dataTransfer.files[0]; if (f) handleFile(f);
};
$('file').onchange = function (e) { const f = e.target.files[0]; if (f) handleFile(f); };
$('play').onclick = play;
$('stop').onclick = stop;
$('dl').onclick = downloadFp;

boot().catch(function (e) { console.error(e); err('Boot failed: ' + e.message); });