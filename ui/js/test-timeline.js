// 통합 테스트 타임라인 (Phase 5 – Part 2 · D3/D4):
// tl-lanes.js 레인 등록부 · test_status_delta 증분 DOM 덧붙이기/패치 · 바이트 상한 LRU · 공유 시간 커서 연동.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt, storageGet, storageSet } from './util.js';
import { setStatus, toast } from './toast.js';
import { submitCommand } from './command-bar.js';
import { EVENT_KO, FOCUS_PARTS, PART_KO, drawFocus, focusPanel, resolvePart } from './focus-view.js';
import { isOpen } from './overlays.js';
import { register } from './poller.js';
import { setCursor } from './cursor.js';
import {
  ByteLruCache, KIND_KO, LANE_SPECS, SDS_KEYS, SDS_KO,
  buildProgressChip, buildSdsCell, decideCaption, mergeStatusDelta, rNum, renderGateLane, tickT,
} from './tl-lanes.js';

const VIS_KO = { pattern: '패턴 보기', pre_act: '행위 직전 보기', glance: '기억이 흐려 응시', blind: '보지 않고 기억으로', place: '목적지 보고 위치 보정', search: '탐색', locate: '찾기', approach: '접근 보기', retouch: '다시 보기' };
const PHASE_KO = { game: '1/3 게임 촬영', perceive: '2/3 ML 인식', decide: '3/3 ML 결정', done: '완료' };
const ACT_W = 70;
const TILE_W = 96;

let focusPart = storageGet('tl-focus', 'frame');
let status = null;
let lastRev = 0;
let lastBrief = '';
let fetching = false;
let dirty = false;
let tabActive = false;
let pendingRender = false;
let finishedId = null;
let pps = 100;
let sel = null; // { pass: 'game'|'ml', index }
let observer = null;
let lanesBuiltId = null;
let laneEls = {};
let prevActX = { game: -1e9, ml: -1e9 };
let prevDecX = -1e9;
let collapsedLanes = new Set();
const cache = new ByteLruCache(24 * 1024 * 1024);
const pending = new Map();

function zClass(z) {
  if (!Number.isFinite(z)) return '';
  const a = Math.abs(z);
  return a < 1 ? 'z0' : a < 2 ? 'z1' : a < 3 ? 'z2' : 'z3';
}

function stepColor(step) {
  const s = String(step);
  if (s.includes('탐색')) return 'var(--text-muted)';
  if (s.includes('접근')) return 'var(--signal)';
  if (s.includes('채취')) return 'var(--warning)';
  if (s.includes('줍기')) return 'var(--violet)';
  if (s.includes('운반')) return 'var(--accent-strong)';
  return 'var(--success)';
}

function build() {
  const pane = $('bpane-test');
  clear(pane);
  const zoom = el('select', { id: 'tl-zoom', aria: { label: '확대' }, on: { change: (e) => { pps = Number(e.target.value); rebuildLanes(); } } },
    ...[[50, '×0.5'], [100, '×1'], [160, '×1.6']].map(([v, t]) => el('option', { value: String(v), text: t, selected: v === pps })));
  const view = el('select', { id: 'tl-view', aria: { label: '보기' }, on: { change: (e) => setView(e.target.value) } },
    el('option', { value: 'timeline', text: '타임라인' }), el('option', { value: 'stats', text: '성공률 · N:N' }));
  const focus = el('select', { id: 'tl-focus', aria: { label: '키프레임 칸에 보일 부위' }, on: { change: (e) => { focusPart = e.target.value; storageSet('tl-focus', focusPart); rebuildLanes(); } } },
    ...FOCUS_PARTS.map((p) => el('option', { value: p.id, text: p.ko, selected: p.id === focusPart })));
  const mirror = el('button', { type: 'button', class: 'vt-btn', id: 'tl-mirror', text: 'DB(Lance) 기록', on: { click: () => mirrorDb(false) } });

  const labels = el('div', { class: 'tl-labels', id: 'tl-labels' },
    el('div', { class: 'tl-l-ruler' }, el('small', { text: '초 (시뮬 시각)' })),
    ...LANE_SPECS.map((l) => el('div', { class: `tl-l-${l.id}`, style: `height:${l.height}px`, title: l.sub },
      el('button', {
        type: 'button', class: 'tl-l-btn', text: collapsedLanes.has(l.id) ? '▸' : '▾', title: '레인 접기/펼치기',
        on: {
          click: (e) => {
            if (collapsedLanes.has(l.id)) collapsedLanes.delete(l.id);
            else collapsedLanes.add(l.id);
            e.target.textContent = collapsedLanes.has(l.id) ? '▸' : '▾';
            laneEls[l.id]?.classList.toggle('tl-lane-collapsed', collapsedLanes.has(l.id));
          },
        },
      }),
      l.label, el('small', { text: l.sub }))));

  pane.append(
    el('div', { class: 'tl-root' },
      el('div', { class: 'tl-toolbar' },
        el('span', { class: 'tl-phase', id: 'tl-phase', text: '대기' }),
        el('span', { class: 'tl-modes', id: 'tl-modes' }),
        el('span', { style: 'margin-left:auto;display:flex;gap:6px;align-items:center' }, el('span', { class: 'vt-label', text: '칸' }), focus, zoom, view, mirror)),
      el('div', { class: 'tl-body', id: 'tl-body' },
        el('div', { class: 'tl-main' }, labels,
          el('div', { class: 'tl-scroll', id: 'tl-scroll', on: { click: onScrollClick } },
            el('div', { class: 'tl-canvas', id: 'tl-canvas' }))),
        el('aside', { class: 'tl-inspector', id: 'tl-inspector', aria: { label: '키프레임 검사기' } },
          el('div', { class: 'tl-empty', text: '키프레임을 누르면 검사기가 열리고 시간 커서가 모든 패널을 그 틱으로 맞춥니다.' })),
        el('div', { class: 'tl-stats', id: 'tl-stats' }))));
  rebuildLanes();
}

function setView(v) {
  $('tl-body').classList.toggle('stats-mode', v === 'stats');
  if (v === 'stats') refreshStats();
}

function rowsOf(st) {
  return { game: st?.game?.keyframes ?? [], decide: st?.decide?.keyframes ?? [] };
}

function updateRulerAndWidth() {
  const canvas = $('tl-canvas');
  if (!canvas) return;
  const { game, decide } = rowsOf(status);
  const tMax = Math.max(10, ...game.map(tickT), ...decide.map(tickT)) + 2;
  canvas.style.width = `${Math.ceil(tMax * pps) + TILE_W + 20}px`;
  const ruler = canvas.querySelector('.tl-ruler');
  if (ruler) {
    clear(ruler);
    const step = pps >= 100 ? 5 : 10;
    for (let t = 0; t <= tMax; t += step) ruler.append(el('span', { class: 'tick', style: `left:${t * pps}px`, text: `${t}s` }));
  }
}

function makeTile(pass, k, x, kind, caption, flagged = false) {
  const t = el('button', {
    type: 'button', class: 'tl-tile', style: `left:${x}px`,
    dataset: { pass, index: String(k.i), kind, kfKey: `${kind}:${pass}:${k.i}` },
    title: caption,
    on: { click: (e) => { e.stopPropagation(); select(pass, k.i); } },
  }, el('div', { class: 'ph', text: '…' }), el('span', { class: 'cap', text: caption }));
  if (flagged) t.append(el('i', { class: 'flag', title: '혼합 특징 이상 표시' }));
  const c = cache.get(`${pass}:${k.i}`);
  if (c && (kind !== 'ml' || c.ml_webp)) fill(t, c);
  else observer?.observe(t);
  return t;
}

function makeActTile(pass, k, x, row) {
  const e = k.event;
  const t = el('button', {
    type: 'button', class: `tl-tile tl-act${e.ok ? '' : ' fail'}`,
    style: `left:${x}px;top:${4 + row * 48}px`,
    dataset: { pass, index: String(k.i), kind: 'act', kfKey: `act:${pass}:${k.i}` },
    title: `${pass === 'ml' ? 'ML 결정' : '게임'} ${rNum(tickT(k), 1)} s · ${e.label}`,
    on: { click: (ev) => { ev.stopPropagation(); select(pass, k.i); } },
  }, el('div', { class: 'ph', text: '…' }), el('span', { class: 'cap', text: `${EVENT_KO[e.kind] ?? e.kind}` }));
  const z = k.sds?.act;
  if (Number.isFinite(z)) t.append(el('i', { class: `zdot ${zClass(z)}`, title: `행위 SDS z ${rNum(z, 1)}` }));
  const c = cache.get(`${pass}:${k.i}`);
  if (c) fill(t, c);
  else observer?.observe(t);
  return t;
}

function mlCaption(k) {
  const ml = k.ml;
  return ml?.target
    ? `${ml.target.label} d${rNum(ml.target.depth_range, 1)}${ml.target.tri_range != null ? ` △${rNum(ml.target.tri_range, 1)}` : ''}`
    : (ml ? '대상 없음' : '분석 대기');
}

function appendGameKf(k) {
  if (k.event) {
    const t0 = tickT(k) * pps;
    const x = Math.max(t0, prevActX.game + ACT_W + 2);
    prevActX.game = x;
    laneEls.act?.append(el('i', { class: `tl-act-tick${k.event.ok ? '' : ' fail'}`, style: `left:${t0}px;top:0` }), makeActTile('game', k, x, 0));
    return;
  }
  const x = tickT(k) * pps;
  const chip = buildProgressChip(k, x);
  if (chip) laneEls.progress?.append(chip);
  if (k.step) laneEls.game?.append(el('div', { class: 'band', style: `left:${x}px;width:${TILE_W}px;background:${stepColor(k.step)}`, title: k.step }));
  laneEls.game?.append(makeTile('game', k, x, 'frame', `${rNum(tickT(k), 1)}s ${k.step ?? ''}`));
  laneEls.perceive?.append(makeTile('game', k, x, 'ml', mlCaption(k), (k.ml?.flags ?? []).length > 0));
  laneEls.sds?.append(buildSdsCell('game', k, x, 0, (p, item) => select(p, item.i)));
}

function appendDecideKf(k) {
  if (k.event) {
    const t0 = tickT(k) * pps;
    const x = Math.max(t0, prevActX.ml + ACT_W + 2);
    prevActX.ml = x;
    laneEls.act?.append(el('i', { class: `tl-act-tick${k.event.ok ? '' : ' fail'}`, style: `left:${t0}px;top:48px` }), makeActTile('ml', k, x, 1));
    return;
  }
  const t0 = tickT(k) * pps;
  const x = Math.max(t0, prevDecX + TILE_W * 0.78);
  prevDecX = x;
  if (x > t0 + 1) laneEls.decide?.append(el('i', { class: 'tl-act-tick', style: `left:${t0}px;top:0` }));
  const t = makeTile('ml', k, x, 'frame', decideCaption(k.decision), false);
  if (k.decision?.kind) t.classList.add(`k-${k.decision.kind}`);
  if (k.decision && !k.decision.ok) t.classList.add('fail');
  laneEls.decide?.append(t);
  laneEls.sds?.append(buildSdsCell('ml', k, x, 1, (p, item) => select(p, item.i)));
}

/** 인식/실행으로 바뀐 키프레임만 기존 DOM 에서 제자리 갱신한다(노드 재생성 없음). */
function patchKf(pass, k) {
  cache.map.delete(`${pass}:${k.i}`);
  if (pass === 'game' && !k.event) {
    const mlTile = laneEls.perceive?.querySelector(`[data-kf-key="ml:game:${k.i}"]`);
    if (mlTile) {
      const cap = mlTile.querySelector('.cap');
      if (cap) cap.textContent = mlCaption(k);
      mlTile.title = mlCaption(k);
      if ((k.ml?.flags ?? []).length > 0 && !mlTile.querySelector('.flag')) mlTile.append(el('i', { class: 'flag', title: '혼합 특징 이상 표시' }));
      observer?.observe(mlTile);
    }
    const oldSds = laneEls.sds?.querySelector(`[data-kf-key="sds:game:${k.i}"]`);
    if (oldSds) oldSds.replaceWith(buildSdsCell('game', k, parseFloat(oldSds.style.left) || tickT(k) * pps, 0, (p, item) => select(p, item.i)));
  } else if (pass === 'ml' && !k.event) {
    const decTile = laneEls.decide?.querySelector(`[data-kf-key="frame:ml:${k.i}"]`);
    if (decTile) {
      const cap = decTile.querySelector('.cap');
      if (cap) cap.textContent = decideCaption(k.decision);
      decTile.classList.toggle('fail', !!(k.decision && !k.decision.ok));
    }
  }
}

function renderVisionLane() {
  const lane = laneEls.vision;
  if (!lane) return;
  clear(lane);
  for (const [rowData, row] of [[status?.game, 0], [status?.decide, 1]]) {
    const top = 4 + row * 22;
    for (const m of rowData?.vision ?? []) {
      const x = (m.tick / 10) * pps;
      lane.append(el('i', { class: `tl-vis ${m.kind}`, style: `left:${x}px;top:${top}px`, title: `${rNum(m.tick / 10, 1)}s · ${VIS_KO[m.kind] ?? m.kind}` }));
    }
    const pts = rowData?.belief_err ?? [];
    if (pts.length > 1) {
      const maxE = Math.max(0.05, ...pts.map((p) => p[1]));
      const h = 18;
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${((p[0] / 10) * pps).toFixed(1)},${(h - (p[1] / maxE) * h).toFixed(1)}`).join(' ');
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'tl-vis-err');
      svg.setAttribute('width', String(Math.ceil((pts[pts.length - 1][0] / 10) * pps) + 4));
      svg.setAttribute('height', String(h));
      svg.style.top = `${top}px`;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      svg.append(path);
      lane.append(svg);
    }
  }
}

function rebuildLanes() {
  const canvas = $('tl-canvas');
  if (!canvas) return;
  clear(canvas);
  observer?.disconnect();
  laneEls = {};
  prevActX = { game: -1e9, ml: -1e9 };
  prevDecX = -1e9;
  canvas.append(el('div', { class: 'tl-ruler' }));
  for (const l of LANE_SPECS) {
    const node = el('div', { class: `tl-lane ${l.h}${collapsedLanes.has(l.id) ? ' tl-lane-collapsed' : ''}`, dataset: { lane: l.id } });
    laneEls[l.id] = node;
    canvas.append(node);
  }
  updateRulerAndWidth();
  if (!status?.active) {
    lanesBuiltId = null;
    laneEls.game?.append(el('div', { class: 'tl-empty', text: state.mock ? '통합 테스트는 개발 서버 또는 Tauri 앱에서 동작합니다.' : '하단 입력줄에 명령을 내고 실행하면 1 게임 → 2 ML 인식 → 3 ML 결정이 여기에 쌓입니다.' }));
    return;
  }
  lanesBuiltId = status.id;
  observer = new IntersectionObserver(onVisible, { root: $('tl-scroll'), rootMargin: '0px 200px' });
  const { game, decide } = rowsOf(status);
  game.forEach(appendGameKf);
  decide.forEach(appendDecideKf);
  renderVisionLane();
  if (sel) highlight();
}

function fill(t, data) {
  const kind = t.dataset.kind;
  const ph = t.querySelector('.ph, img, canvas.fcv');
  const mode = kind === 'act' && focusPart === 'frame' ? 'auto' : focusPart;
  if (kind !== 'ml' && mode !== 'frame') {
    const f = data.focus;
    if (!f) { ph?.replaceWith(el('div', { class: 'ph', text: '포커스 없음' })); return; }
    const part = resolvePart(mode, f);
    const cv = el('canvas', { class: 'fcv', title: PART_KO[part] ?? part });
    ph?.replaceWith(cv);
    requestAnimationFrame(() => drawFocus(cv, f, part, { compact: true }));
    return;
  }
  const src = kind === 'ml' ? data.ml_webp : data.frame_webp;
  if (!src) { ph?.replaceWith(el('div', { class: 'ph', text: '없음' })); return; }
  ph?.replaceWith(el('img', { alt: '', src: `data:image/webp;base64,${src}` }));
}

async function fetchKf(pass, index, needMl = false, full = false) {
  const key = `${pass}:${index}`;
  const enough = (v) => v && (!needMl || v.ml_webp) && (!full || v.focus_webp);
  const have = cache.get(key);
  if (enough(have)) return have;
  if (pending.has(key)) {
    const v = await pending.get(key);
    if (enough(v)) return v;
  }
  const p = call('test_keyframe', { pass, index, close_ups: full }, { quiet: true })
    .then((v) => { cache.set(key, v); return v; })
    .catch(() => null)
    .finally(() => { if (pending.get(key) === p) pending.delete(key); });
  pending.set(key, p);
  return p;
}

function onVisible(entries) {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const t = e.target;
    observer.unobserve(t);
    fetchKf(t.dataset.pass, Number(t.dataset.index), t.dataset.kind === 'ml').then((v) => v && fill(t, v));
  }
}

function onScrollClick(e) {
  const rect = $('tl-canvas').getBoundingClientRect();
  const t = (e.clientX - rect.left) / pps;
  const { game } = rowsOf(status);
  if (!game.length) return;
  const k = game.reduce((a, b) => (Math.abs(tickT(b) - t) < Math.abs(tickT(a) - t) ? b : a));
  select('game', k.i);
}

function movePlayhead(tick) {
  const canvas = $('tl-canvas');
  if (!canvas || tick == null) return;
  let ph = canvas.querySelector('.tl-playhead');
  if (!ph) { ph = el('div', { class: 'tl-playhead' }); canvas.append(ph); }
  ph.style.left = `${(tick / 10) * pps}px`;
}

function highlight() {
  for (const t of document.querySelectorAll('#tl-canvas .tl-tile.sel')) t.classList.remove('sel');
  if (!sel) return;
  for (const t of document.querySelectorAll(`#tl-canvas .tl-tile[data-pass="${sel.pass}"][data-index="${sel.index}"]`)) t.classList.add('sel');
  const rows = rowsOf(status)[sel.pass === 'ml' ? 'decide' : 'game'];
  const k = rows.find((x) => x.i === sel.index);
  if (k) movePlayhead(k.tick);
}

function kv(rows) {
  return el('div', { class: 'tl-kv' }, ...rows.flatMap(([k, v]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: v })]));
}

function table(head, rows) {
  return el('table', { class: 'tl-mini' },
    el('thead', {}, el('tr', {}, ...head.map((h) => el('th', { text: h })))),
    el('tbody', {}, ...rows.map((row) => el('tr', {}, ...row.map((c) => el('td', {}, c))))));
}

async function select(pass, index) {
  sel = { pass, index };
  highlight();
  const rows = rowsOf(status)[pass === 'ml' ? 'decide' : 'game'];
  const hit = rows.find((x) => x.i === index);
  if (hit) setCursor(hit.tick, pass);
  const insp = $('tl-inspector');
  const v = await fetchKf(pass, index, true, true) ?? cache.get(`${pass}:${index}`);
  if (!v || sel.pass !== pass || sel.index !== index) return;
  clear(insp);
  const b = v.brief;
  const pos = rows.findIndex((x) => x.i === index);
  const nav = (d) => { const k = rows[pos + d]; if (k) select(pass, k.i); };
  insp.append(el('h5', {},
    el('span', { class: 'grow', text: `${pass === 'ml' ? 'ML 결정' : '게임'} #${index} · ${rNum(b.t, 1)}s · 틱 ${b.tick}` }),
    el('button', { type: 'button', class: 'vt-btn', text: '◀', on: { click: () => nav(-1) } }),
    el('button', { type: 'button', class: 'vt-btn', text: '▶', on: { click: () => nav(1) } })));
  insp.append(el('div', { class: 'tl-pics' },
    el('figure', {}, el('img', { alt: '원본', src: `data:image/webp;base64,${v.frame_webp}` }), el('figcaption', { text: `원본 ${v.width}×${v.height}` })),
    el('figure', {}, v.ml_webp ? el('img', { alt: '깊이/마스크', src: `data:image/webp;base64,${v.ml_webp}` }) : el('div', { class: 'tl-empty', text: '분석 전' }), el('figcaption', { text: '깊이 · 추적 마스크 · 의미 위치' }))));
  if (v.focus || v.event) insp.append(el('div', { class: 'ct-sec', text: '행위 · 부위 포커스' }), focusPanel(v.focus, v.focus_webp, v.event));
  if (v.decision) {
    const d = v.decision;
    const bandText = Array.isArray(d.band) && d.band.length >= 4 ? `x[${rNum(d.band[0])},${rNum(d.band[2])}] z[${rNum(d.band[1])},${rNum(d.band[3])}]` : '—';
    insp.append(el('div', { class: 'ct-sec', text: '결정 (예측 띠 · 삼각 측량)' }), kv([
      ['결정', `${KIND_KO[d.kind] ?? d.kind} · ${d.decision}`], ['메모', d.note || '—'],
      ['예측기', d.forecast_method || '—'], ['예측 띠', bandText], ['실행', `${d.executed || '—'}${d.ok ? '' : ' · 실패'}`],
    ]));
  }
  insp.append(el('div', { class: 'ct-sec', text: 'SDS 칸 (스코프 · n)' }),
    el('div', { class: 'tl-flags' }, ...SDS_KEYS.map((s) => {
      const cell = b.sds_cells?.[s];
      const val = cell ? cell.value : b.sds?.[s];
      const tag = cell ? `${SDS_KO[s]} ${rNum(val, 1)} (n=${cell.n})` : `${SDS_KO[s]} ${rNum(val, 1)}`;
      return el('span', { class: 'tl-chip', text: tag, title: cell?.scope ?? '' });
    })));
}

async function refreshStats() {
  const box = $('tl-stats');
  if (!box) return;
  try {
    const s = await call('test_stats', {}, { quiet: true });
    clear(box);
    const stats = (s.stats ?? []).slice().sort((a, b) => a.key.localeCompare(b.key));
    box.append(el('div', { class: 'ct-sec', text: '퀘스트 · 단계별 성공률' }),
      table(['키', '단계', '성공/시도', '성공률'], stats.map((r0) => [r0.key, r0.step, `${r0.successes}/${r0.attempts}`, fmt.pct(r0.rate)])));
  } catch (e) { clear(box).append(el('div', { class: 'tl-empty', text: e.message })); }
}

function renderToolbar() {
  const st = status;
  const ph = $('tl-phase');
  if (!ph) return;
  if (!st?.active) ph.textContent = '대기';
  else {
    const g = st.game?.keyframes?.length ?? 0;
    const d = st.decide?.keyframes?.length ?? 0;
    const p = st.perceive ?? {};
    ph.textContent = `${PHASE_KO[st.phase] ?? st.phase} · "${st.command ?? ''}" · 틱 ${st.tick} · 게임 ${g} (분석 ${p.done ?? 0}/${p.total ?? g}) · 결정 ${d}`;
  }
  const modes = $('tl-modes');
  if (modes) {
    clear(modes);
    if (st?.error) modes.append(el('span', { class: 'badge badge-warn', text: `오류: ${st.error}` }));
    if (st?.notes?.length) modes.append(el('span', { class: 'tl-chip stand', text: st.notes[st.notes.length - 1] }));
  }
  renderLive(null);
}

function renderLive(brief) {
  const box = $('tl-live');
  if (!box) return;
  const b = brief ?? (status?.active ? { active: true, live: status.live, phase: status.phase, game: status.game?.keyframes?.length ?? 0, decide: status.decide?.keyframes?.length ?? 0, perceived: status.perceive?.done ?? 0 } : null);
  box.classList.toggle('on', !!b?.active && b.live && b.phase !== 'done');
  box.textContent = !b?.active ? (b?.queued ? `대기 명령 ${b.queued}` : '') : `${PHASE_KO[b.phase] ?? b.phase} · 게임 ${b.game} · 결정 ${b.decide}`;
}

async function refreshDelta(forceFull = false) {
  if (state.mock) return;
  if (fetching) { dirty = true; return; }
  fetching = true;
  dirty = false;
  try {
    if (forceFull || !status?.active) {
      const s = await call('test_status', {}, { quiet: true });
      if (s?.id !== status?.id) { cache.clear(); sel = null; }
      status = s;
      lastRev = s?.test_rev ?? 0;
      if (tabActive && !document.hidden) { pendingRender = false; rebuildLanes(); }
      else pendingRender = true;
    } else {
      const d = await call('test_status_delta', {
        id: status.id, since_rev: lastRev, if_rev: lastRev,
        vision_from_game: status.game?.vision?.length ?? 0,
        vision_from_decide: status.decide?.vision?.length ?? 0,
        belief_from_game: status.game?.belief_err?.length ?? 0,
        belief_from_decide: status.decide?.belief_err?.length ?? 0,
      }, { quiet: true });
      if (d && !d.unchanged) {
        const merged = mergeStatusDelta(status, d);
        status = merged.status;
        lastRev = d.test_rev ?? lastRev;
        if (tabActive && !document.hidden) {
          if (merged.reset || lanesBuiltId !== status.id) rebuildLanes();
          else {
            updateRulerAndWidth();
            merged.appended.game.forEach(appendGameKf);
            merged.appended.decide.forEach(appendDecideKf);
            merged.updated.game.forEach((k) => patchKf('game', k));
            merged.updated.decide.forEach((k) => patchKf('ml', k));
            if (d.game?.vision_tail?.length || d.decide?.vision_tail?.length || d.game?.belief_tail?.length || d.decide?.belief_tail?.length) renderVisionLane();
          }
          pendingRender = false;
        } else pendingRender = true;
      }
    }
    renderToolbar();
    if (status?.active && status.phase === 'done' && finishedId !== status.id) {
      finishedId = status.id;
      await mirrorDb(true);
    }
  } catch { /* 조용히 */ } finally { fetching = false; }
  if (dirty) refreshDelta();
}

async function mirrorDb(quiet) {
  if (state.mock) return;
  try {
    const r0 = await call('test_db_mirror', {}, { quiet });
    if (!quiet) setStatus(`LanceDB 기록 완료 → ${r0.store_dir}`);
  } catch (e) { if (!quiet) toast('DB 기록 실패', e.message, 'err'); }
}

export function initTestTimeline() {
  build();
  $('tl-stop')?.addEventListener('click', async () => {
    await submitCommand('멈춰', { quietHistory: true }).catch(() => {});
    await call('test_cancel', {}, { quiet: true }).catch(() => {});
    refreshDelta(true);
  });
  bus.on('sim-test', (brief) => {
    renderLive(brief);
    if (!brief) return;
    const key = JSON.stringify([brief.id, brief.phase, brief.game, brief.perceived, brief.decide]);
    if (key === lastBrief) return;
    lastBrief = key;
    if (tabActive || brief.phase === 'done') refreshDelta();
    else pendingRender = true;
  });
  bus.on('cursor', (c) => {
    if (c.mode === 'past') {
      movePlayhead(c.tick);
      if (c.ctx?.gate_marks) renderGateLane(laneEls.gate, c.ctx.gate_marks, pps, (tk) => setCursor(tk, c.pass));
    }
  });
  bus.on('cursor-step', (dir) => {
    if (!tabActive || !status?.active) return;
    const pass = sel?.pass ?? 'game';
    const rows = rowsOf(status)[pass === 'ml' ? 'decide' : 'game'];
    if (!rows.length) return;
    const idx = sel ? rows.findIndex((x) => x.i === sel.index) : 0;
    const next = rows[Math.max(0, Math.min(rows.length - 1, idx + dir))];
    if (next) select(pass, next.i);
  });
  bus.on('command', (c) => { if (c?.recording || c?.queued) refreshDelta(true); });
  bus.on('world-new', () => { status = null; lastRev = 0; lanesBuiltId = null; cache.clear(); sel = null; rebuildLanes(); renderToolbar(); });
  bus.on('tab', ({ group, name }) => {
    if (group !== 'analysis') return;
    tabActive = name === 'test';
    if (tabActive && (pendingRender || !status)) refreshDelta(!status);
  });
  register('test-timeline', {
    rev: 'test',
    minGap: 300,
    visible: () => isOpen('analysis-modal') && tabActive,
    run: () => refreshDelta(),
  });
  renderToolbar();
}
