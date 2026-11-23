// N:N 상태 Drop-up 판 (Phase 5 – Part 1 · D2/D3/D4):
// B1 시나리오 목록(5상태 + S4 사다리) · B2 위상 상세(비게이트 점수 + 7부위 막대 + 감점 분해표 + 게이트) · B3 7×7 지도 · 보류/기권 탭.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { isOpen } from './overlays.js';
import { register, invalidate } from './poller.js';

const STATE_KO = { active: '진행', seen: '관측', unseen: '미관측', skipped_form: '형태 생략', no_record: '미기록' };
const SEG_KO = ['머리', '가슴', '배', '팔', '손', '다리', '발'];
const DOMAIN_KO = { traffic: '통행', precision: '정밀도', decide: '결정', nn: 'N:N' };

let activeTab = 'scenarios'; // 'scenarios' | 'matrix' | 'holds'
let selScenario = 'S4';
let selPhase = null;
let data = null;

function rungClass(p) {
  if (p.excused) return 'excused';
  if (p.item === 2) return 'item-2';
  if (p.item === 1) return 'item-1';
  if (p.item === 0) return 'item-0';
  if (p.score != null) return p.score >= 80 ? 'item-2' : p.score >= 60 ? 'item-1' : 'item-0';
  return '';
}

function renderScenarioList(box) {
  clear(box);
  const list = data?.scenario_states ?? [];
  if (!list.length) {
    box.append(el('p', { class: 'empty-note', text: '시나리오 데이터가 아직 없습니다.' }));
    return;
  }
  const catMap = new Map((data?.catalog ?? []).map((c) => [c.id, c]));
  for (const s of list) {
    const sid = s.scenario;
    const cat = catMap.get(sid);
    const scScore = (s.phases ?? []).map((p) => p.score).filter((x) => x != null);
    const minSc = scScore.length ? Math.min(...scScore) : null;
    const stCls = s.state === 'active' ? 'now' : s.state === 'seen' ? 'observed' : s.state === 'skipped_form' ? 'skipped' : 'unrecorded';
    const card = el('button', {
      type: 'button', class: `nn-scn-card${sid === selScenario ? ' selected' : ''}`,
      on: { click: () => { selScenario = sid; selPhase = s.phases?.[0]?.phase ?? null; fetchNn(); } },
    },
      el('div', { class: 'nn-scn-head' },
        el('strong', { text: `${sid} ${s.name_ko ?? ''}` }),
        el('span', { class: `nn-state ${stCls}`, text: STATE_KO[s.state] ?? s.state })),
      el('div', { class: 'hint', style: 'margin:0', text: minSc != null ? `점수 ${fmt.n(minSc, 1)}` : (cat?.skipped_reason ?? '—') }));
    if (s.phases?.length) {
      const lad = el('div', { class: 'nn-ladder' });
      for (const ph of s.phases) {
        lad.append(el('span', {
          class: `nn-rung ${rungClass(ph)}`,
          title: `${ph.phase}: ${STATE_KO[ph.state] ?? ph.state}${ph.score != null ? ` · ${fmt.n(ph.score, 1)}` : ''}`,
          on: { click: (e) => { e.stopPropagation(); selScenario = sid; selPhase = ph.phase; fetchNn(); } },
        }));
      }
      card.append(lad);
    }
    box.append(card);
  }
}

function renderPhaseDetail(box) {
  clear(box);
  const d = data?.detail;
  if (!d) {
    box.append(el('p', { class: 'empty-note', text: '왼쪽에서 시나리오·위상을 고르면 감점 분해와 게이트 판정이 보입니다.' }));
    return;
  }
  const ung = d.ungated ?? d.score;
  const gatedNote = ung != null && Math.abs((ung ?? 0) - (d.score ?? 0)) > 0.05
    ? ` (게이트 전 ${fmt.n(ung, 1)} → 게이트 후 ${fmt.n(d.score, 1)})`
    : '';
  box.append(
    el('div', { class: 'ct-sec', text: `${d.scenario} / ${d.phase} · 점수 ${fmt.n(d.score, 1)}${gatedNote}` }),
  );
  // 7부위 막대
  const segs = d.seg_scores ?? [];
  if (segs.length) {
    const row = el('div', { class: 'nn-segs' });
    SEG_KO.forEach((nm, i) => {
      const v = segs[i];
      const h = v == null ? 0 : Math.max(2, Math.min(100, v));
      const cls = v == null ? '' : v >= 75 ? '' : v >= 50 ? 'warn' : 'bad';
      row.append(el('div', { class: 'nn-seg', title: `${nm}: ${v == null ? '—' : fmt.n(v, 1)}` },
        el('div', { class: 'nn-seg-bar' }, el('div', { class: `nn-seg-fill ${cls}`, style: `height:${h}%` })),
        el('span', { text: nm })));
    });
    box.append(row);
  }
  // 게이트 판정 (면제 포함)
  for (const g of d.gate_hits ?? []) {
    const label = g.excused ? `[면제] ${g.code}` : `[게이트] ${g.code}`;
    box.append(el('div', { class: `nn-gate-row${g.excused ? ' excused' : ''}`, text: `${label} · 측정 ${fmt.n(g.measured, 3)} / 한계 ${fmt.n(g.limit, 3)}` }));
  }
  // 감점 분해표 (합 = 100 - ungated)
  const bd = (d.cells ?? []).filter((c) => (c.loss ?? 0) > 0 || c.z != null);
  if (bd.length) {
    const sumLoss = bd.reduce((a, x) => a + (x.loss ?? 0), 0);
    box.append(
      el('div', { class: 'hint', text: `감점 분해 합계 ${fmt.n(sumLoss, 1)} (= 100 − 비게이트 ${fmt.n(ung ?? 100 - sumLoss, 1)})` }),
      el('table', { class: 'pv-table' },
        el('thead', {}, el('tr', {}, ['부위', '축', '값', '기준', 'z', '감점'].map((h, i) => el('th', { text: h, class: i >= 2 ? 'num' : null })))),
        el('tbody', {}, bd.map((r) => el('tr', {},
          el('td', { text: r.target }),
          el('td', { class: 'mono', text: r.axis }),
          el('td', { class: 'num', text: fmt.n(r.value, 2) }),
          el('td', { class: 'num', text: r.reference != null ? fmt.n(r.reference, 2) : '—' }),
          el('td', { class: 'num', text: r.z != null ? fmt.n(r.z, 2) : '—' }),
          el('td', { class: 'num', text: `-${fmt.n(r.loss, 1)}` }))))));
  }
}

function renderMatrix(box) {
  clear(box);
  const grid = data?.matrix;
  if (!Array.isArray(grid) || !grid.length) {
    box.append(el('p', { class: 'empty-note', text: '7×7 균형 지도가 비어 있습니다.' }));
    return;
  }
  const tbl = el('table', { class: 'nn-7x7' },
    el('thead', {}, el('tr', {}, el('th', { text: '원인\\부위' }), ...SEG_KO.map((s) => el('th', { text: s })))),
    el('tbody', {}, grid.map((row) => el('tr', {},
      el('th', { text: row[0]?.source ?? '' }),
      ...row.map((c) => {
        const cls = c.item == null ? 'c-empty' : c.item === 2 ? 'c-ok' : c.item === 1 ? 'c-warn' : 'c-bad';
        const txt = c.worst_z == null ? '·' : `${fmt.n(c.worst_z, 1)}z`;
        return el('td', { class: cls, title: `${c.source} → ${c.target} (n=${c.count ?? 0}, 감점 ${fmt.n(c.loss_sum, 1)})` }, txt);
      })))));
  box.append(el('div', { class: 'ct-sec', text: '7×7 원장 최악 |z| 지도' }), tbl);
}

function renderHolds(box) {
  clear(box);
  const h = data?.holds ?? state.progress?.holds;
  const items = h?.items ?? [];
  box.append(el('div', { class: 'ct-sec', text: `통합 보류(${h?.hold_count ?? 0}) · 기권(${h?.abstain_count ?? 0})` }));
  if (!items.length) {
    box.append(el('p', { class: 'empty-note', text: '최근 보류나 기권 기록이 없습니다.' }));
    return;
  }
  box.append(el('table', { class: 'pv-table' },
    el('thead', {}, el('tr', {}, ['틱', '영역', '판정', '코드', '대상', '설명'].map((t) => el('th', { text: t })))),
    el('tbody', {}, items.slice().reverse().map((x) => el('tr', {},
      el('td', { class: 'mono', text: String(x.tick) }),
      el('td', { text: DOMAIN_KO[x.domain] ?? x.domain }),
      el('td', {}, el('span', { class: `pb-chip ${x.verdict === 'abstained' ? 'abstain' : 'hold'}`, text: x.verdict === 'abstained' ? '기권' : '보류' })),
      el('td', { class: 'mono', text: x.code }),
      el('td', { text: x.subject }),
      el('td', { text: x.detail_ko }))))));
}

function render() {
  const body = $('nn-dropup-body');
  if (!body) return;
  clear(body);
  $('nn-meta').textContent = data ? `형태 ${data.form_id ?? '—'} · rev ${data.rev ?? 0}` : '';
  document.querySelectorAll('#nn-tabs .nn-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === activeTab));
  if (activeTab === 'holds') {
    renderHolds(body);
    return;
  }
  if (activeTab === 'matrix') {
    renderMatrix(body);
    return;
  }
  const grid = el('div', { class: 'nn-grid' });
  const c1 = el('div', { class: 'nn-col' });
  const c2 = el('div', { class: 'nn-col' });
  const c3 = el('div', { class: 'nn-col' });
  renderScenarioList(c1);
  renderPhaseDetail(c2);
  renderMatrix(c3);
  grid.append(c1, c2, c3);
  body.append(grid);
}

async function fetchNn() {
  try {
    data = await call('nn_status', {
      worker: state.progressWorker ?? 1,
      scenario: selScenario,
      phase: selPhase,
    }, { quiet: true });
    render();
  } catch { /* 조용히 */ }
}

export function initNnPanel() {
  const tabs = $('nn-tabs');
  if (tabs) {
    tabs.addEventListener('click', (e) => {
      const btn = e.target.closest?.('.nn-tab');
      if (!btn) return;
      activeTab = btn.dataset.tab;
      render();
    });
  }
  bus.on('nn-focus', ({ scenario, phase, tab }) => {
    if (scenario) selScenario = scenario;
    if (phase !== undefined) selPhase = phase;
    if (tab) activeTab = tab;
    render();
    fetchNn();
  });
  bus.on('panel', ({ id, open }) => {
    if (id === 'nn-dropup' && open) invalidate('nn-panel');
  });
  // 열려 있고 nn rev 가 바뀌었을 때만, 최소 1초 간격으로 갱신 (Phase 5 – Part 1 · D2)
  register('nn-panel', {
    rev: 'nn',
    minGap: 1000,
    visible: () => isOpen('nn-dropup'),
    run: () => fetchNn(),
  });
}
