// SDS 스코프 탐색기 (Phase 5 – Part 2 · D1):
// 트리(sds_tree) · 서버 필터 표(sds_query) · 드리프트 감시 스파크라인(sds_watch) · "시계열로 열기".
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { isOpen } from './overlays.js';
import { register, invalidate } from './poller.js';

let activeTab = null;
let selectedPrefix = '';
let selectedTrack = '';
let textFilter = '';
let kindFilter = '';
let fbOnly = false;
let currentPhaseOnly = false;
let sortBy = 'n_desc';
let expanded = new Map(); // nodeId -> children[]
let rootNodes = [];
let queryRes = null;
let watchRes = null;
let lastTreeRev = null;
let lastQueryRev = null;
let cursorTick = null;

function openInTimeseries(scope, axis) {
  call('sds_watch', { pin: { scope, axis } }, { quiet: true }).catch(() => {});
  bus.emit('open-ts', { series: `sds:${scope}:${axis}` });
}

function drawSparkline(pts, w = 96, h = 24) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', String(w));
  svg.setAttribute('height', String(h));
  svg.style.display = 'block';
  const yOf = (z) => h / 2 - (Math.max(-4, Math.min(4, z)) / 4) * (h / 2 - 2);
  // ±3 드리프트 기준선
  for (const refZ of [3, -3]) {
    const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    ln.setAttribute('x1', '0');
    ln.setAttribute('x2', String(w));
    ln.setAttribute('y1', yOf(refZ).toFixed(1));
    ln.setAttribute('y2', yOf(refZ).toFixed(1));
    ln.setAttribute('stroke', 'rgba(224,108,108,0.45)');
    ln.setAttribute('stroke-dasharray', '2,2');
    svg.append(ln);
  }
  if (pts.length > 1) {
    const t0 = pts[0].tick;
    const span = Math.max(1, pts[pts.length - 1].tick - t0);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${(((p.tick - t0) / span) * w).toFixed(1)},${yOf(p.drift_z).toFixed(1)}`).join(' ');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#a08fd6');
    path.setAttribute('stroke-width', '1.4');
    svg.append(path);
    if (cursorTick != null && cursorTick >= t0 && cursorTick <= pts[pts.length - 1].tick) {
      const cx = (((cursorTick - t0) / span) * w).toFixed(1);
      const cl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      cl.setAttribute('x1', cx); cl.setAttribute('x2', cx);
      cl.setAttribute('y1', '0'); cl.setAttribute('y2', String(h));
      cl.setAttribute('stroke', '#d6b65e');
      svg.append(cl);
    }
  }
  return svg;
}

function renderTree(box) {
  clear(box);
  box.append(el('button', {
    type: 'button', class: `sds-node${selectedPrefix === '' && selectedTrack === '' ? ' active' : ''}`,
    text: '전체 트랙',
    on: { click: () => { selectedTrack = ''; selectedPrefix = ''; lastQueryRev = null; refreshQuery(); } },
  }));
  const addNodes = (nodes, depth) => {
    for (const n of nodes) {
      const isExp = expanded.has(n.id);
      const active = selectedPrefix === n.id || (depth === 0 && selectedTrack === n.id && !selectedPrefix);
      const row = el('button', {
        type: 'button', class: `sds-node depth-${depth}${active ? ' active' : ''}`,
        on: {
          click: async () => {
            if (n.kind === 'track') { selectedTrack = n.id; selectedPrefix = ''; }
            else { selectedTrack = ''; selectedPrefix = n.id; }
            if (n.has_children) {
              if (isExp) expanded.delete(n.id);
              else {
                const res = await call('sds_tree', { parent: n.id }, { quiet: true }).catch(() => null);
                if (res?.nodes) expanded.set(n.id, res.nodes);
              }
            }
            lastQueryRev = null;
            renderTree(box);
            refreshQuery();
          },
        },
      },
        el('span', { text: n.has_children ? (isExp ? '▾' : '▸') : '·' }),
        el('span', { text: n.label, style: 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }));
      if (n.need && n.n_max != null) {
        const pct = Math.min(100, Math.round((n.n_max / Math.max(1, n.need)) * 100));
        row.append(el('span', { class: `sds-mat${n.n_max < n.need ? ' immature' : ''}`, title: `n=${n.n_max}/${n.need}` }, el('i', { style: `width:${pct}%` })));
      }
      for (const f of (n.feedback ?? []).slice(0, 3)) row.append(el('span', { class: 'sds-fb-chip', text: f }));
      if ((n.engine_revs ?? []).length > 1) row.append(el('span', { class: 'badge badge-warn', text: '혼재', title: n.engine_revs.join(', ') }));
      box.append(row);
      if (isExp) addNodes(expanded.get(n.id) ?? [], Math.min(2, depth + 1));
    }
  };
  addNodes(rootNodes, 0);
}

function renderRows(box) {
  clear(box);
  // 드리프트 감시 스파크라인 띠
  const ws = watchRes?.series ?? [];
  if (ws.length) {
    const strip = el('div', { class: 'sds-watch-strip' });
    for (const s of ws) {
      const last = s.points?.[s.points.length - 1];
      strip.append(el('div', { class: 'sds-watch-card' },
        el('div', { class: 'mono', text: `${s.scope} · ${s.axis}` }),
        drawSparkline(s.points ?? []),
        el('div', { class: 'hint', style: 'margin:0', text: last ? `z=${fmt.n(last.drift_z, 2)} · n=${last.n}` : '대기' })));
    }
    box.append(strip);
  }
  const rows = queryRes?.rows ?? [];
  if (!rows.length) {
    box.append(el('p', { class: 'empty-note', text: '조건에 맞는 SDS 축이 없습니다.' }));
    return;
  }
  box.append(
    el('div', { class: 'hint', style: 'margin:0 0 4px', text: `총 ${queryRes.total}행 (병합 스코프 ${queryRes.merged_scopes} · rev ${queryRes.rev})` }),
    el('table', { class: 'pv-table' },
      el('thead', {}, el('tr', {}, ['스코프 / 축', '종류', '성숙도', '주요 값', '드리프트 z', '되먹임', ''].map((h) => el('th', { text: h })))),
      el('tbody', {}, rows.map((r) => {
        const matPct = Math.min(100, Math.round((r.maturity ?? 0) * 100));
        let valTxt = '—';
        if (r.kind === 'gate') valTxt = `차단 ${fmt.pct(r.blocked_rate, 0)} · 단독 ${fmt.pct(r.sole_rate, 0)}`;
        else if (r.kind === 'reference') valTxt = `${fmt.n(r.mean, 2)} (기준 ${fmt.n(r.reference, 2)}, 띠 ${fmt.pct(r.in_band, 0)})`;
        else if (r.kind === 'agreement') valTxt = `비율 ${fmt.pct(r.rate, 1)}`;
        else if (r.kind === 'reliability') valTxt = `구간 ${(r.bins ?? []).length}개`;
        else valTxt = `${fmt.n(r.mean, 3)} ± ${fmt.n(r.sd, 3)} (최근 ${fmt.n(r.recent, 3)})`;
        const fbCell = el('td', {});
        for (const p of r.feedback ?? []) fbCell.append(el('span', { class: 'sds-fb-chip', text: p }));
        return el('tr', {},
          el('td', { class: 'mono', title: r.resolves_to ? `실제 적용 키: ${r.resolves_to}` : '' }, `${r.scope}`, el('div', { class: 'hint', style: 'margin:0', text: r.axis })),
          el('td', { text: r.kind }),
          el('td', { title: `${r.n}/${r.need}` }, el('span', { class: `sds-mat${r.n < r.need ? ' immature' : ''}` }, el('i', { style: `width:${matPct}%` })), ` ${r.n}/${r.need}`),
          el('td', { class: 'mono', text: valTxt }),
          el('td', { class: `num ${Math.abs(r.drift_z ?? 0) >= 3 ? 'warn' : ''}`, text: fmt.n(r.drift_z ?? 0, 2) }),
          fbCell,
          el('td', {}, el('button', { type: 'button', class: 'pb-btn', text: '시계열', on: { click: () => openInTimeseries(r.scope, r.axis) } })));
      }))));
}

async function refreshTree() {
  const res = await call('sds_tree', { parent: '', if_rev: lastTreeRev }, { quiet: true }).catch(() => null);
  if (!res || res.unchanged) return;
  lastTreeRev = res.rev;
  rootNodes = res.nodes ?? [];
  const box = $('sds-exp-tree');
  if (box) renderTree(box);
}

async function refreshQuery() {
  let prefix = selectedPrefix;
  if (currentPhaseOnly && state.progress?.nn?.now?.length) {
    const a = state.progress.nn.now[0];
    prefix = `balance|nn:${a.scenario}`;
  }
  const args = {
    track: selectedTrack || undefined,
    prefix: prefix || undefined,
    text: textFilter || undefined,
    kinds: kindFilter ? [kindFilter] : undefined,
    feedback_only: fbOnly || undefined,
    sort: sortBy,
    limit: 120,
    if_rev: lastQueryRev ?? undefined,
  };
  const [qr, wr] = await Promise.all([
    call('sds_query', args, { quiet: true }).catch(() => null),
    call('sds_watch', {}, { quiet: true }).catch(() => null),
  ]);
  if (qr && !qr.unchanged) {
    lastQueryRev = qr.rev;
    queryRes = qr;
  }
  if (wr && !wr.unchanged) watchRes = wr;
  const box = $('sds-exp-rows');
  if (box) renderRows(box);
}

function buildShell() {
  const host = $('sds-explorer');
  if (!host) return;
  clear(host);
  const filters = el('div', { class: 'sds-filters' },
    el('input', { type: 'text', placeholder: '스코프·축 검색', on: { input: (e) => { textFilter = e.target.value.trim(); lastQueryRev = null; refreshQuery(); } } }),
    el('select', { on: { change: (e) => { kindFilter = e.target.value; lastQueryRev = null; refreshQuery(); } } },
      ...[['', '모든 종류'], ['baseline', '기준선'], ['gate', '게이트'], ['reference', '사람 기준 대비'], ['agreement', '일치율']].map(([v, t]) => el('option', { value: v, text: t }))),
    el('select', { on: { change: (e) => { sortBy = e.target.value; lastQueryRev = null; refreshQuery(); } } },
      ...[['n_desc', '관측순'], ['drift_desc', '드리프트순'], ['maturity_asc', '미성숙순']].map(([v, t]) => el('option', { value: v, text: t }))),
    el('label', { class: 'ct-check' }, el('input', { type: 'checkbox', on: { change: (e) => { fbOnly = e.target.checked; lastQueryRev = null; refreshQuery(); } } }), '되먹임만'),
    el('label', { class: 'ct-check' }, el('input', { type: 'checkbox', on: { change: (e) => { currentPhaseOnly = e.target.checked; lastQueryRev = null; refreshQuery(); } } }), '지금 위상만'));
  const tree = el('div', { class: 'sds-tree', id: 'sds-exp-tree' });
  const right = el('div', { style: 'min-width:0;overflow:auto;display:flex;flex-direction:column' },
    filters,
    el('div', { id: 'sds-exp-rows', style: 'flex:1;overflow:auto' }));
  host.append(el('div', { class: 'sds-exp' }, tree, right));
}

export function initSdsExplorer() {
  buildShell();
  bus.on('tab', ({ group, name }) => {
    if (group !== 'analysis') return;
    activeTab = name;
    if (name === 'ts') {
      lastTreeRev = null;
      lastQueryRev = null;
      invalidate('sds-explorer');
    }
  });
  bus.on('cursor', (c) => {
    cursorTick = c.mode === 'past' ? c.tick : null;
    const box = $('sds-exp-rows');
    if (box && activeTab === 'ts') renderRows(box);
  });
  register('sds-explorer', {
    rev: 'sds',
    minGap: 800,
    visible: () => isOpen('analysis-modal') && activeTab === 'ts',
    run: async () => { await Promise.all([refreshTree(), refreshQuery()]); },
  });
}
