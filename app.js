// Detector-in-the-Loop Synthetic Data Augmentation — showcase page.
// Client-side re-matching:
// Nothing in the exported data is pre-matched or pre-thresholded; every TP/FP/FN
// verdict on the page is computed here from raw boxes + scores.

const DATASETS = ['cs', 'wood', 'ai', 'amf'];
const DATA_DIR = 'showcase_export';
const MODELS = ['baseline', 'finetuned'];        // scored panels
const PANELS = ['gt', 'baseline', 'finetuned'];  // rendered panels
const MODEL_LABEL = { baseline: 'Baseline — real data only', finetuned: 'After progressive curation' };
const TAG_LABEL = {
  finetune_win: 'finetune win',
  loc_miss: 'localization miss',
  loose_annotation: 'loose annotation',
  shared_failure: 'shared failure',
  baseline_win: 'baseline win',
  negative_image: 'negative image',
};

// ---------- box geometry ----------
const w = b => b[2] - b[0], h = b => b[3] - b[1];
const area = b => Math.max(0, w(b)) * Math.max(0, h(b));
const cx = b => (b[0] + b[2]) / 2, cy = b => (b[1] + b[3]) / 2;
function inter(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return (iw > 0 && ih > 0) ? iw * ih : 0;
}

// A confidence win must clear this margin. Most margins are hair-thin (median ~0.001), and
// without a floor the category fills up with ties that bury the cases worth looking at.
const MIN_CONF_MARGIN = 0.01;

// The one matching criterion: intersection over union, the COCO rule.
const iou = (a, b) => { const i = inter(a, b); return i ? i / (area(a) + area(b) - i) : 0; };

// ---------- state ----------
const S = {
  data: {},
  ds: 'cs',
  thr: 0.5,          // IoU threshold
  // Both models are scored at the BASELINE's F1-max operating point (conservative: the
  // baseline gets its own optimum, the curated model does not). Not adjustable.
  op: { baseline: 0.5, finetuned: 0.5 },
  imgId: null,
  win: 'all',       // 'all' | 'more' | 'conf' | 'fp'
  showConf: true,   // what the box labels carry
  showIou: true,
  showHud: true,    // the info pill in the corner of each image
  view: { zoom: 1, px: 0, py: 0 },
};
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const el = (t, a = {}) => Object.assign(document.createElement(t), a);

// ---------- matching ----------
// Greedy one-to-one assignment in descending score order, class-agnostic.
function match(gt, preds, conf, thr) {
  const kept = preds.filter(p => p.score >= conf).slice().sort((x, y) => y.score - x.score);
  const taken = new Array(gt.length).fill(false);
  const boxes = [];
  let tp = 0, fp = 0;
  for (const p of kept) {
    let best = -1, bestV = 0;
    for (let g = 0; g < gt.length; g++) {
      if (taken[g]) continue;
      const v = iou(p.bbox, gt[g].bbox);
      if (v > bestV) { bestV = v; best = g; }
    }
    if (best >= 0 && bestV >= thr) {
      taken[best] = true; tp++;
      boxes.push({ ...p, kind: 'tp', v: bestV });
    } else {
      // "overlaps a GT but fails the criterion" is judged on raw pixel overlap, so a
      // localization miss stays orange even when the GT it overlaps was already consumed.
      fp++;
      boxes.push({ ...p, kind: gt.some(g => inter(p.bbox, g.bbox) > 0) ? 'loc' : 'fp', v: bestV });
    }
  }
  const misses = gt.filter((g, i) => !taken[i]);
  // `loc` and `spur` partition `fp`: orange boxes overlap a real defect, red ones do not.
  // P/R/F1 use the total, but the readout shows them apart so the four numbers match the
  // four overlay colours exactly.
  const loc = boxes.filter(b => b.kind === 'loc').length;
  return { tp, fp, loc, spur: fp - loc, fn: misses.length, boxes, misses };
}
const f1 = s => (2 * s.tp + s.fp + s.fn) ? (2 * s.tp) / (2 * s.tp + s.fp + s.fn) : 0;
// Precision, recall and F1 are reported as percentages throughout. A difference between two
// percentages is in percentage points, so deltas carry "pp" rather than "%".
const pct = (v, dp = 1) => (v * 100).toFixed(dp) + '%';
const pp = (v, dp = 2) => (v >= 0 ? '+' : '') + (v * 100).toFixed(dp);
const prec = s => (s.tp + s.fp) ? s.tp / (s.tp + s.fp) : 0;
const rec = s => (s.tp + s.fn) ? s.tp / (s.tp + s.fn) : 0;


// ---------- win analysis ----------
// A GT box counts as "detected" when a prediction above the operating point overlaps it at all —
// deliberately looser than the matching criterion, so a detection that is visually right but
// fails IoU still counts as found. The assignment is still greedy ONE-TO-ONE, ranked by
// containment: without that, one sloppy box spanning three defects would be credited with
// finding all three. Returns the assigned prediction's score per GT box, -1 where none.
function bestScorePerGT(im, model) {
  const kept = im[model].filter(p => p.score >= S.op[model]).slice().sort((a, b) => b.score - a.score);
  const out = im.gt.map(() => -1);
  for (const p of kept) {
    let best = -1, bestV = 0;
    for (let g = 0; g < im.gt.length; g++) {
      if (out[g] >= 0) continue;
      const i = inter(p.bbox, im.gt[g].bbox);
      if (!i) continue;
      const v = i / Math.min(area(p.bbox), area(im.gt[g].bbox));   // tightest containment wins
      if (v > bestV) { bestV = v; best = g; }
    }
    if (best >= 0) out[best] = p.score;
  }
  return out;
}

// A prediction above the operating point that overlaps NO ground-truth box at all — the boxes
// drawn red. Localization misses (orange) are excluded on purpose: those overlap a real defect,
// so removing one is not the same kind of win.
function spuriousCount(im, model) {
  return im[model].filter(p => p.score >= S.op[model] && !im.gt.some(g => inter(p.bbox, g.bbox) > 0)).length;
}

// Three ways the curated model wins, per the showcase brief:
//   'more' — it found a defect the baseline missed entirely
//   'fp'   — it fires on fewer things that are not defects
//   'conf' — both found the same defect, but the curated model is more confident
// Images where the curated model found fewer defects are excluded outright. An image qualifying for
// more than one category is filed under the strongest, so the lists stay disjoint.
const WIN_CACHE = new Map();
function winInfo(im) {
  const key = S.ds + '|' + im.id + '|' + S.op.baseline + '|' + S.op.finetuned;
  if (WIN_CACHE.has(key)) return WIN_CACHE.get(key);
  const b = bestScorePerGT(im, 'baseline');
  const f = bestScorePerGT(im, 'finetuned');
  const foundB = b.filter(x => x >= 0).length;
  const foundF = f.filter(x => x >= 0).length;
  let dConf = 0;
  for (let i = 0; i < b.length; i++) if (b[i] >= 0 && f[i] > b[i]) dConf = Math.max(dConf, f[i] - b[i]);
  const spB = spuriousCount(im, 'baseline');
  const spF = spuriousCount(im, 'finetuned');
  let info = { cat: null, foundB, foundF, dFound: foundF - foundB, dConf, spB, spF, dSpur: spB - spF };
  if (foundF > foundB) {
    info.cat = 'more';
    info.margin = info.dFound;
    info.reason = 'curated model found ' + info.dFound + ' defect' + (info.dFound === 1 ? '' : 's')
      + ' the baseline missed (' + foundF + ' of ' + im.gt.length + ' vs ' + foundB + ')';
  } else if (foundF === foundB && spF < spB) {
    info.cat = 'fp';
    info.margin = info.dSpur;
    info.reason = 'same defects found — curated model drops ' + info.dSpur + ' false positive'
      + (info.dSpur === 1 ? '' : 's') + ' (' + spF + ' vs ' + spB + ')';
  } else if (foundF === foundB && dConf >= MIN_CONF_MARGIN) {
    info.cat = 'conf';
    info.margin = dConf;
    info.reason = 'same defect found by both — curated model is +' + dConf.toFixed(2) + ' more confident';
  }
  WIN_CACHE.set(key, info);
  return info;
}

// ---------- derived ----------
const cur = () => S.data[S.ds];
const activeImages = () => cur().images;
const WIN_ORDER = ['more', 'conf', 'fp'];
const visibleImages = () => activeImages()
  .map(im => ({ im, win: winInfo(im) }))
  .filter(x => x.win.cat && (S.win === 'all' || S.win === x.win.cat))
  .sort((p, q) => (p.win.cat === q.win.cat)
    ? q.win.margin - p.win.margin                                        // strongest win first
    : WIN_ORDER.indexOf(p.win.cat) - WIN_ORDER.indexOf(q.win.cat))       // then by category
  .map(x => x.im);
const imgById = id => cur().images.find(im => im.id === id);
const imgSrc = im => DATA_DIR + '/images/' + S.ds + '/' + im.file;
// 240 px copies for the example strip (make_thumbs.py); full images only in the comparison.
const thumbSrc = im => DATA_DIR + '/thumbs/' + S.ds + '/' + im.file;
// Fetch the neighbours' full images in the background so prev/next feels instant.
const _warm = new Set();
function warmNeighbours() {
  const list = visibleImages();
  const at = list.findIndex(x => x.id === S.imgId);
  if (at < 0) return;
  for (const k of [at + 1, at - 1, at + 2]) {
    const im = list[(k + list.length) % list.length];
    if (!im) continue;
    const src = imgSrc(im);
    if (_warm.has(src)) continue;
    _warm.add(src);
    new Image().src = src;
  }
}
const stats = (im, model) => match(im.gt, im[model], S.op[model], S.thr);

// ---------- boot ----------
// Called from the very bottom of this file: it touches consts declared further down, and a
// synchronous call from here would hit their temporal dead zone.
function boot() {
  const bundle = window.SHOWCASE || {};
  const missing = DATASETS.filter(ds => !bundle[ds]);
  if (missing.length) {
    $('#loading').textContent = 'Missing dataset file' + (missing.length > 1 ? 's' : '') + ': '
      + missing.map(ds => DATA_DIR + '/data/' + ds + '.js').join(', ')
      + '. Run build_data.py to regenerate them.';
    return;
  }
  DATASETS.forEach(ds => { S.data[ds] = bundle[ds]; });

  const tabs = $('#tabs');
  DATASETS.forEach(ds => {
    const d = S.data[ds];
    const b = el('button', { type: 'button', className: 'tab-btn' });
    b.setAttribute('role', 'tab');
    b.dataset.ds = ds;
    b.innerHTML = d.dataset + '<small>' + d.title + '</small>';
    b.onclick = () => loadDataset(ds);
    tabs.append(b);
  });

  wire();
  $('#loading').hidden = true;   // stages must be laid out before they can be measured
  loadDataset('cs');
}

function wire() {

  $('#showconf').onchange = e => { S.showConf = e.target.checked; renderAll(); };
  $('#showiou').onchange = e => { S.showIou = e.target.checked; renderAll(); };
  $('#showhud').onchange = e => { S.showHud = e.target.checked; renderAll(); };
  $('#shuffle').onclick = () => renderGen();

  $('#thr').oninput = e => {
    S.thr = +e.target.value;
    $('#thrval').textContent = S.thr.toFixed(2);
    renderAll();
  };
  $('#prev').onclick = () => step(-1);
  $('#next').onclick = () => step(1);

  document.addEventListener('keydown', e => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'j') { step(1); e.preventDefault(); }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'k') { step(-1); e.preventDefault(); }
    if (e.key === '0') { S.view = { zoom: 1, px: 0, py: 0 }; renderAll(); }
  });

  PANELS.forEach(m => wireStage(document.querySelector('.compare-tile[data-model="' + m + '"] .stage')));
  addEventListener('resize', renderAll);
}

function loadDataset(ds) {
  S.ds = ds;
  const d = cur();
  S.win = 'all';
  WIN_CACHE.clear();
  S.view = { zoom: 1, px: 0, py: 0 };
  syncOperatingPoint();

  $$('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.ds === ds));
  $('#dssub').textContent =
    d.dataset + ' — ' + d.title + ' · '
    + d.classes.length + ' class' + (d.classes.length > 1 ? 'es' : '') + ' (' + d.classes.join(', ') + ')'
    + ' · ' + d.images.length + ' validation images'
    + ' · predictions exported down to score ' + d.export.score_floor
    + ' · ' + d.export.nms;
  document.querySelector('.compare-tile[data-model="gt"] [data-r=meta]').textContent =
    'annotation · ' + d.classes.length + ' class' + (d.classes.length > 1 ? 'es' : '');

  renderGen();
  renderTagFilter();
  const list = visibleImages();
  S.imgId = list.length ? list[0].id : null;
  renderStrip();
  renderAll();
}

// Both panels are scored at the baseline's F1-max operating point: a conservative comparison,
// since only the baseline is at its own optimum.
function syncOperatingPoint() {
  const d = cur();
  S.op = { baseline: d.f1max.baseline, finetuned: d.f1max.baseline };
  $('#opval').textContent = S.op.baseline.toFixed(4);
  $('#ophint').innerHTML =
    'Fixed score threshold, applied to <em>both</em> models: the F1-max operating point of the '
    + '<em>baseline</em>, so the comparison does not favour the curated model.';
  $('#opnote').innerHTML =
    'Both models are shown at score threshold <b>' + S.op.baseline.toFixed(4) + '</b>, the F1-max '
    + 'operating point of the <b>baseline</b> (real data only).';
}

const WIN_LABEL = {
  all: 'All wins',
  more: 'Curated model found more defects',
  conf: 'Same defect, higher confidence',
  fp: 'Fewer false positives',
};
const WIN_KEYS = ['all', 'more', 'conf', 'fp'];

function renderTagFilter() {
  const box = $('#tagfilter');
  box.innerHTML = '';
  const counts = { all: 0, more: 0, conf: 0, fp: 0 };
  activeImages().forEach(im => {
    const c = winInfo(im).cat;
    if (c) { counts[c]++; counts.all++; }
  });
  for (const k of WIN_KEYS) {
    const b = el('button', { type: 'button', className: 'tag' + (S.win === k ? ' on' : '') });
    b.innerHTML = WIN_LABEL[k] + '<span class="n">' + counts[k] + '</span>';
    b.onclick = () => {
      S.win = k;
      $$('#tagfilter .tag').forEach((n, i) => n.classList.toggle('on', WIN_KEYS[i] === k));
      const list = visibleImages();
      if (!list.some(im => im.id === S.imgId)) S.imgId = list.length ? list[0].id : null;
      renderStrip();
      renderAll();
    };
    box.append(b);
  }
}

function winBadge(win) {
  if (win.cat === 'more') return '+' + win.dFound + '▲';        // defects found
  if (win.cat === 'fp') return '−' + win.dSpur + '✕';     // false positives dropped
  return '+' + win.dConf.toFixed(2);                                 // confidence
}

function renderStrip() {
  const strip = $('#strip');
  strip.innerHTML = '';
  for (const im of visibleImages()) {
    const win = winInfo(im);
    const fig = el('figure', {
      className: 'thumb' + (im.id === S.imgId ? ' on' : ''),
      title: win.reason + (im.tags.length ? ' · ' + im.tags.map(t => TAG_LABEL[t]).join(', ') : ''),
    });
    fig.dataset.id = im.id;
    const img = el('img', { alt: im.id, loading: 'lazy', decoding: 'async' });
    img.src = thumbSrc(im);
    fig.append(img);
    const cap = el('figcaption');
    cap.append(el('span', { textContent: im.id.replace(S.ds + '_', '') }));
    cap.append(el('span', { className: 'd up', textContent: winBadge(win) }));
    fig.append(cap);
    fig.onclick = () => select(im.id);
    strip.append(fig);
  }
}

// Scroll the strip itself, never the document — scrollIntoView would jump the whole page
// and move the arrows out from under the pointer.
function keepThumbVisible() {
  const strip = $('#strip');
  const active = strip.querySelector('.thumb.on');
  if (!active) return;
  const top = active.offsetTop - strip.offsetTop;
  const bottom = top + active.offsetHeight;
  if (top < strip.scrollTop) strip.scrollTop = top;
  else if (bottom > strip.scrollTop + strip.clientHeight) strip.scrollTop = bottom - strip.clientHeight;
}

// Step through the listed wins, wrapping at both ends.
function step(d) {
  const list = visibleImages();
  if (!list.length) return;
  const i = list.findIndex(im => im.id === S.imgId);
  select(list[(i + d + list.length) % list.length].id);
}

function select(id) {
  S.imgId = id;
  S.view = { zoom: 1, px: 0, py: 0 };
  $$('#strip .thumb').forEach(n => n.classList.toggle('on', n.dataset.id === id));
  keepThumbVisible();
  renderAll();
}

// ---------- generated vs template wipe ----------
// Four random kept samples per dataset. Each card wipes between the generated defect and the
// clean template it was painted onto; the pairing comes from build_gen.py.
const GEN_COUNT = 4;

// Annotated boxes go down to ~1% of the image, which is a handful of pixels on a card this
// size. Anything smaller than this is grown for display — symmetrically, so the marker stays
// centred on the defect. The stored coordinates are left untouched.
const MIN_SPOT = 0.04;

// Cards share one shape per dataset so the row stays even, but generated images do not all
// have that shape (AI has four, AMF three). object-fit: contain letterboxes the odd ones, so a
// box in image space has to be mapped into container space before it can be positioned as a
// percentage — otherwise the marker drifts off the defect by up to a fifth of the card.
function cardShape(pool) {
  const count = new Map();
  for (const r of pool) {
    const k = r.w + '/' + r.h;
    count.set(k, (count.get(k) || 0) + 1);
  }
  let best = '1/1', n = -1;
  for (const [k, v] of count) if (v > n) { n = v; best = k; }
  const [w, h] = best.split('/').map(Number);
  return { w, h };
}

function toContainer(b, imgW, imgH, cardW, cardH) {
  const rc = cardW / cardH, ri = imgW / imgH;
  let sx = 1, sy = 1, ox = 0, oy = 0;
  if (ri > rc) { sy = rc / ri; oy = (1 - sy) / 2; }        // wider than the card: bars top/bottom
  else if (ri < rc) { sx = ri / rc; ox = (1 - sx) / 2; }   // taller: bars left/right
  return [ox + b[0] * sx, oy + b[1] * sy, b[2] * sx, b[3] * sy];
}

function displayBox(b) {
  let [x, y, w, h] = b;
  if (w < MIN_SPOT) { x = Math.max(0, Math.min(1 - MIN_SPOT, x + w / 2 - MIN_SPOT / 2)); w = MIN_SPOT; }
  if (h < MIN_SPOT) { y = Math.max(0, Math.min(1 - MIN_SPOT, y + h / 2 - MIN_SPOT / 2)); h = MIN_SPOT; }
  return [x, y, w, h];
}

// Every kept sample carries at least one annotated defect box, so there is nothing to screen
// out — a card can no longer land on an image with nothing to point at.
const isVisible = r => Array.isArray(r.b) && r.b.length > 0;

function sampleGen(ds) {
  const all = (window.GENSAMPLES || {})[ds] || [];
  const pool = all.filter(isVisible);
  if (pool.length <= GEN_COUNT) return pool.slice();
  const picked = [], used = new Set();
  while (picked.length < GEN_COUNT) {
    const i = Math.floor(Math.random() * pool.length);
    if (used.has(i)) continue;
    used.add(i);
    picked.push(pool[i]);
  }
  return picked;
}

function renderGen() {
  const grid = $('#gengrid');
  const section = $('#gensection');
  const rows = sampleGen(S.ds);
  section.hidden = rows.length === 0;
  grid.innerHTML = '';
  const shape = cardShape((window.GENSAMPLES || {})[S.ds] || rows);
  for (const r of rows) {
    const card = el('div', { className: 'gen-card' });
    const cmp = el('div', { className: 'compare' });
    cmp.style.aspectRatio = shape.w + ' / ' + shape.h;
    // r.b holds every annotated defect box for this image, [x, y, w, h] as fractions,
    // straight from the generator's own COCO labels via build_gen.py.
    const drawn = (r.b || [])
      .map(b => toContainer(b, r.w, r.h, shape.w, shape.h))
      .map(displayBox);
    const spot = drawn.length
      ? '<div class="spotlayer">' + drawn.map((bx, i) =>
          '<div class="spot" title="' + ((r.k && r.k[i]) || '') + '" style="left:' + (bx[0] * 100)
          + '%;top:' + (bx[1] * 100) + '%;width:' + (bx[2] * 100)
          + '%;height:' + (bx[3] * 100) + '%"></div>').join('') + '</div>'
      : '';
    cmp.innerHTML =
      '<img class="clean" alt="clean template" src="' + r.t + '">'
      + '<img class="gen" alt="generated defect" src="' + r.g + '">'
      + spot
      + '<div class="handle"></div>'
      + '<span class="lbl gen">GEN</span><span class="lbl clean">CLEAN</span>';
    wireWipe(cmp, drawn);
    card.append(cmp);

    const meta = el('div', { className: 'gen-meta' });
    meta.append(el('span', { className: 'cat', textContent: r.c }));
    card.append(meta);
    grid.append(card);
  }
}

// Drag anywhere across the card to move the divider; it also follows a plain click.
function wireWipe(cmp, box) {
  const gen = cmp.querySelector('.gen');
  const handle = cmp.querySelector('.handle');
  const layer = cmp.querySelector('.spotlayer');
  const set = pct => {
    const p = Math.max(0, Math.min(100, pct));
    const clip = 'inset(0 ' + (100 - p) + '% 0 0)';
    gen.style.clipPath = clip;
    // the marker rides with the generated half, so it never floats over the clean side
    if (layer) layer.style.clipPath = clip;
    handle.style.left = p + '%';
  };
  // Open with the divider just past the furthest defect, so every card shows what was painted
  // without being touched; dragging left then reveals the clean plate underneath.
  const rightmost = (box && box.length)
    ? Math.max(...box.map(b => (b[0] + b[2]) * 100))
    : null;
  set(rightmost === null ? 50 : Math.max(35, Math.min(100, rightmost + 6)));
  const at = e => {
    const r = cmp.getBoundingClientRect();
    set(((e.clientX - r.left) / r.width) * 100);
  };
  // The marker is in the way while comparing, so it fades out for the duration of a drag.
  let dragging = false;
  cmp.addEventListener('pointerdown', e => {
    dragging = true;
    cmp.classList.add('wiping');
    cmp.setPointerCapture(e.pointerId);
    at(e);
    e.preventDefault();
  });
  cmp.addEventListener('pointermove', e => { if (dragging) at(e); });
  const stop = () => { dragging = false; cmp.classList.remove('wiping'); };
  cmp.addEventListener('pointerup', stop);
  cmp.addEventListener('pointercancel', stop);
}

// ---------- rendering ----------
const COLOR = { tp: '#22c55e', loc: '#f59e0b', fp: '#ef4444', fn: '#94a3b8' };

function renderAll() {
  const im = S.imgId ? imgById(S.imgId) : null;
  $('#imgid').textContent = im ? im.id : '—';
  const list = visibleImages();
  const at = list.findIndex(x => x.id === S.imgId);
  $('#imgpos').textContent = (at < 0 ? 0 : at + 1) + ' / ' + list.length;
  $('#prev').disabled = $('#next').disabled = list.length < 2;
  $('#imgtags').innerHTML = im ? '<span class="chip win">' + winInfo(im).reason + '</span>' : '';
  PANELS.forEach(m => renderTile(m, im));
  renderAgg();
  setTimeout(warmNeighbours, 300);   // after the current image has started loading
}

function renderTile(model, im) {
  const tile = document.querySelector('.compare-tile[data-model="' + model + '"]');
  const img = tile.querySelector('[data-r=img]');
  const svg = tile.querySelector('[data-r=svg]');
  const stage = tile.querySelector('[data-r=stage]');
  const zoom = tile.querySelector('[data-r=zoom]');
  tile.querySelector('[data-r=hud]').hidden = !S.showHud;
  if (!im) {
    svg.innerHTML = '';
    img.removeAttribute('src');
    tile.querySelector('[data-r=stats]').textContent = 'no image selected';
    tile.querySelector('[data-r=hud]').textContent = '';
    return;
  }

  const src = imgSrc(im);
  if (img.getAttribute('src') !== src) img.setAttribute('src', src);
  img.width = im.width;
  img.height = im.height;

  // The GT panel is the annotation reference: no matching, no verdict, no metrics.
  const isGT = model === 'gt';
  const s = isGT ? null : stats(im, model);
  tile.querySelector('[data-r=stats]').innerHTML = isGT
    ? '<span>' + im.gt.length + ' annotated box' + (im.gt.length === 1 ? '' : 'es') + '</span>'
      + (im.gt.length ? '<span>' + classBreakdown(im.gt) + '</span>'
                      : '<span style="color:var(--fn)">negative image</span>')
    : '<span>TP <b style="color:var(--tp)">' + s.tp + '</b></span>'
      + '<span>loc <b style="color:var(--loc)">' + s.loc + '</b></span>'
      + '<span>FP <b style="color:var(--fp)">' + s.spur + '</b></span>'
      + '<span>FN <b style="color:var(--fn)">' + s.fn + '</b></span>';

  // fit + zoom (the view object is shared, so all three tiles stay in lockstep)
  const sw = stage.clientWidth, sh = stage.clientHeight;
  if (!sw || !sh) return;
  const base = Math.min(sw / im.width, sh / im.height);
  const sc = base * S.view.zoom;
  const tx = (sw - im.width * sc) / 2 + S.view.px;
  const ty = (sh - im.height * sc) / 2 + S.view.py;
  zoom.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + sc + ')';
  tile._fit = { base: base, sw: sw, sh: sh, W: im.width, H: im.height };

  svg.setAttribute('viewBox', '0 0 ' + im.width + ' ' + im.height);
  svg.style.width = im.width + 'px';
  svg.style.height = im.height + 'px';

  const fs = 11 / sc, pad = 2 / sc;
  let out = '';
  if (isGT) {
    for (const g of im.gt) {
      out += rect(g.bbox, '#ffffff', 2 / sc, null, 1);
      if (cur().classes.length > 1) {
        out += tag(cur().classes[g.class] || ('class ' + g.class), g.bbox, '#ffffff', fs, pad, im);
      }
    }
    svg.innerHTML = out;
    tile.querySelector('[data-r=hud]').textContent =
      im.width + '×' + im.height + ' · ' + Math.round(S.view.zoom * 100) + '% · annotation reference';
    return;
  }
  for (const g of im.gt) out += rect(g.bbox, '#ffffff', 1.5 / sc, (4 / sc) + ' ' + (3 / sc), 0.85);
  for (const m of s.misses) out += rect(m.bbox, COLOR.fn, 2 / sc, (5 / sc) + ' ' + (4 / sc), 1);
  for (const b of s.boxes) {
    out += rect(b.bbox, COLOR[b.kind], 2 / sc, null, 1);
    const parts = [];
    if (S.showConf) parts.push(b.score.toFixed(2));
    if (S.showIou) parts.push('iou ' + b.v.toFixed(3));
    if (parts.length) out += tag(parts.join(' · '), b.bbox, COLOR[b.kind], fs, pad, im);
  }
  svg.innerHTML = out;

  tile.querySelector('[data-r=hud]').textContent =
    im.width + '×' + im.height + ' · ' + Math.round(S.view.zoom * 100) + '% · '
    + im[model].length + ' raw boxes ≥ ' + cur().export.score_floor;
}

// "2 scratch, 1 stain" for a multi-class dataset; just the count for a single-class one.
function classBreakdown(gt) {
  const names = cur().classes;
  if (names.length < 2) return names[0] || 'defect';
  const n = {};
  for (const g of gt) n[g.class] = (n[g.class] || 0) + 1;
  return Object.keys(n).map(k => n[k] + ' ' + (names[k] || ('class ' + k))).join(', ');
}

// A label sits above its box by default, but boxes touching the top or right edge would push it
// off the image, so it is clamped into the image rect — dropping just inside the box top when
// there is no room above. Coordinates are image space; fs/pad already account for zoom.
function tag(text, box, color, fs, pad, im) {
  const lw = text.length * fs * 0.58 + 2 * pad;
  const lh = fs + 2 * pad;
  let x = box[0];
  let y = box[1] - lh;
  if (y < 0) y = box[1];                                  // no room above: sit inside the box
  if (x + lw > im.width) x = im.width - lw;
  if (x < 0) x = 0;
  if (y + lh > im.height) y = im.height - lh;
  if (y < 0) y = 0;
  return '<rect x="' + x + '" y="' + y + '" width="' + lw + '" height="' + lh
    + '" fill="' + color + '" opacity="0.92"/>'
    + '<text x="' + (x + pad) + '" y="' + (y + fs) + '" font-size="' + fs
    + '" font-family="ui-monospace,monospace" fill="#0a0c11">' + text + '</text>';
}

// Stroke widths arrive already divided by the current scale, so outlines keep a
// constant on-screen thickness at any zoom level.
function rect(b, color, sw, dash, op) {
  return '<rect x="' + b[0] + '" y="' + b[1] + '" width="' + w(b) + '" height="' + h(b)
    + '" fill="none" stroke="' + color + '" stroke-width="' + sw + '" opacity="' + op
    + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/>';
}

function renderAgg() {
  // Deliberately NOT visibleImages(): the strip above is filtered to finetuning wins, and
  // aggregating over a win-filtered subset would be a meaningless headline. This sums over
  // every image in the active split.
  const items = activeImages();
  const agg = {};
  for (const m of MODELS) {
    const t = { tp: 0, fp: 0, loc: 0, spur: 0, fn: 0 };
    for (const im of items) {
      const s = stats(im, m);
      t.tp += s.tp; t.fp += s.fp; t.loc += s.loc; t.spur += s.spur; t.fn += s.fn;
    }
    agg[m] = t;
  }
  // Headline = the paper's reported F1-max (full evaluation, each model at its own optimum, IoU 0.5).
  // The counts underneath are this page's own recount at the baseline's threshold.
  const paper = m => cur().models[m].paper_f1max;
  const d = paper('finetuned') - paper('baseline');
  const crit = 'IoU ≥ ' + S.thr.toFixed(2);

  $('#agghint').textContent =
    'Large numbers: validation F1-max reported in the paper. Counts below: recounted on this page over all '
    + items.length + ' image' + (items.length === 1 ? '' : 's')
    + ' in the split — not only the examples listed above — at ' + crit
    + ', greedy one-to-one and class-agnostic, with both models at the baseline’s threshold ('
    + S.op.baseline.toFixed(4) + ').';

  $('#agg').innerHTML = MODELS.map(m =>
    '<div class="stat-tile"><h4>' + MODEL_LABEL[m] + '</h4>'
    + '<div class="big">' + pct(paper(m), 2) + '<small>F1-max (paper)</small></div>'
    + '<div class="cells">'
    + '<span>TP <b style="color:var(--tp)">' + agg[m].tp + '</b></span>'
    + '<span>loc <b style="color:var(--loc)">' + agg[m].loc + '</b></span>'
    + '<span>FP <b style="color:var(--fp)">' + agg[m].spur + '</b></span>'
    + '<span>FN <b style="color:var(--fn)">' + agg[m].fn + '</b></span>'
    + '<span>P <b>' + pct(prec(agg[m])) + '</b></span>'
    + '<span>R <b>' + pct(rec(agg[m])) + '</b></span>'
    + '</div></div>').join('')
    + '<div class="stat-tile delta"><h4>Curated − baseline</h4>'
    + '<div class="big" style="color:' + (d >= 0 ? 'var(--tp)' : 'var(--fp)') + '">'
    + pp(d) + '<small>pp ΔF1-max (paper)</small></div>'
    + '<div class="cells">'
    + '<span>ΔTP <b>' + sgn(agg.finetuned.tp - agg.baseline.tp) + '</b></span>'
    + '<span>Δloc <b>' + sgn(agg.finetuned.loc - agg.baseline.loc) + '</b></span>'
    + '<span>ΔFP <b>' + sgn(agg.finetuned.spur - agg.baseline.spur) + '</b></span>'
    + '<span>ΔFN <b>' + sgn(agg.finetuned.fn - agg.baseline.fn) + '</b></span>'
    + '</div></div>';
}
const sgn = n => (n > 0 ? '+' : '') + n;

// ---------- zoom / pan (shared across both tiles) ----------
function wireStage(stage) {
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const f = stage.closest('.compare-tile')._fit;
    if (!f) return;
    const r = stage.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const old = S.view.zoom;
    const next = Math.min(40, Math.max(1, old * Math.exp(-e.deltaY * 0.0015)));
    if (next === old) return;
    const sc0 = f.base * old, sc1 = f.base * next;
    const ix = (mx - ((f.sw - f.W * sc0) / 2 + S.view.px)) / sc0;
    const iy = (my - ((f.sh - f.H * sc0) / 2 + S.view.py)) / sc0;
    S.view.zoom = next;
    if (next === 1) {
      S.view.px = 0; S.view.py = 0;
    } else {
      S.view.px = (mx - ix * sc1) - (f.sw - f.W * sc1) / 2;
      S.view.py = (my - iy * sc1) - (f.sh - f.H * sc1) / 2;
    }
    renderAll();
  }, { passive: false });

  let drag = null;
  stage.addEventListener('pointerdown', e => {
    drag = { x: e.clientX, y: e.clientY, px: S.view.px, py: S.view.py };
    stage.setPointerCapture(e.pointerId);
    stage.classList.add('drag');
  });
  stage.addEventListener('pointermove', e => {
    if (!drag) return;
    S.view.px = drag.px + (e.clientX - drag.x);
    S.view.py = drag.py + (e.clientY - drag.y);
    renderAll();
  });
  const up = () => { drag = null; stage.classList.remove('drag'); };
  stage.addEventListener('pointerup', up);
  stage.addEventListener('pointercancel', up);
  stage.addEventListener('dblclick', () => { S.view = { zoom: 1, px: 0, py: 0 }; renderAll(); });
}

boot();
