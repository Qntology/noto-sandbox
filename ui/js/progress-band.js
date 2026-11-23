// 실시간 진행 띠 (Phase 5 – Part 1 · D1): 퀘스트 · 절차/구간 · N:N 현재 위상 · 최근 60초 자취 · 작업자 · 보류/기권.
import { bus, state } from './state.js';
import { $, clear, el, fmt, storageGet, storageSet } from './util.js';
import { openPanel } from './overlays.js';

let lastSig = '';
let collapsed = storageGet('pb-collapsed', '0') === '1';

function itemClass(a) {
  if (a.excused) return 'excused';
  if (a.item === 2) return 'item-2';
  if (a.item === 1) return 'item-1';
  return 'item-0';
}

function openNnFor(scenario = null, phase = null, tab = 'scenarios') {
  bus.emit('nn-focus', { scenario, phase, tab });
  openPanel('nn-dropup');
}

function render(p) {
  const root = $('progress-band');
  if (!root || !p) return;
  const sig = JSON.stringify([
    collapsed, p.focus, p.quest?.label_ko, p.quest?.index, p.quest?.count,
    p.procedure?.stage_ko, p.procedure?.segment, p.nn?.status, p.nn?.rev,
    (p.nn?.now ?? []).map((x) => `${x.scenario}:${x.phase}:${Math.round(x.score)}:${x.item}:${x.excused}`),
    p.holds?.hold_count, p.holds?.abstain_count,
    (p.workers ?? []).map((w) => `${w.id}:${w.stage_ko}:${w.holds_n}`),
  ]);
  if (sig === lastSig) return;
  lastSig = sig;
  root.classList.toggle('collapsed', collapsed);
  clear(root);

  // 1행: 퀘스트 + 절차/구간 + 작업자 칩 + 접기
  const row1 = el('div', { class: 'pb-row' });
  const qTxt = p.quest
    ? `${p.quest.index}/${p.quest.count} ${p.quest.label_ko} (${Math.round((p.quest.progress ?? 0) * 100)}%)`
    : '대기';
  row1.append(el('span', { class: 'pb-tag', text: '퀘스트' }), el('span', { class: 'pb-val', text: qTxt }));
  if (p.procedure) {
    const pTxt = p.procedure.stage_ko
      ? `${p.procedure.index ? `${p.procedure.index}/${p.procedure.count} ` : ''}${p.procedure.stage_ko}`
      : (p.procedure.segment ? `구간 · ${p.procedure.segment}` : '대기');
    row1.append(el('span', { class: 'pb-tag', text: p.procedure.stage_ko ? '절차' : '구간' }), el('span', { class: 'pb-val', text: pTxt }));
  }

  const end = el('span', { class: 'pb-end' });
  const workers = p.workers ?? [];
  if (workers.length > 1) {
    for (const w of workers) {
      const cls = `pb-chip worker${w.id === p.focus ? ' active' : ''}${w.nn === 'not_computed' ? ' not-comp' : ''}`;
      const lbl = `W${w.id}${w.nn === 'not_computed' ? ' 미계산' : ''}${w.holds_n ? ` ⚠${w.holds_n}` : ''}`;
      end.append(el('button', {
        type: 'button', class: cls, text: lbl,
        title: `작업자 ${w.id} (${w.form}) · ${w.stage_ko}${w.nn === 'not_computed' ? ' · N:N 미계산' : ''}`,
        on: { click: () => { state.progressWorker = w.id; bus.emit('progress-worker', w.id); } },
      }));
    }
  }
  end.append(el('button', {
    type: 'button', class: 'pb-btn', id: 'nn-open-btn',
    'aria-expanded': 'false', 'aria-controls': 'nn-dropup', text: 'N:N 판 ▲',
    on: { click: () => openNnFor(null, null, 'scenarios') },
  }));
  end.append(el('button', {
    type: 'button', class: 'pb-btn', text: collapsed ? '펼치기 ▾' : '접기 ▴',
    on: { click: () => { collapsed = !collapsed; storageSet('pb-collapsed', collapsed ? '1' : '0'); lastSig = ''; render(p); } },
  }));
  row1.append(end);
  root.append(row1);

  // 2행: N:N 현재 위상 + 60초 자취 + 보류/기권 칩
  const row2 = el('div', { class: 'pb-row pb-row-extra' });
  row2.append(el('span', { class: 'pb-tag', text: 'N:N' }));
  const chips = el('span', { class: 'pb-chips' });
  if (p.nn?.status === 'not_computed') {
    chips.append(el('span', { class: 'pb-chip not-comp', text: `미계산 (${p.nn.reason ?? '서브 작업자'})` }));
  } else if (p.nn?.now?.length) {
    for (const a of p.nn.now) {
      const badge = a.excused ? '면제' : `항목${a.item}`;
      chips.append(el('button', {
        type: 'button', class: `pb-chip ${itemClass(a)}`,
        text: `${a.scenario}/${a.phase} ${fmt.n(a.score, 0)} · ${badge}`,
        title: `비게이트 ${fmt.n(a.ungated, 1)} · 누르면 상세`,
        on: { click: () => openNnFor(a.scenario, a.phase, 'scenarios') },
      }));
    }
  } else {
    chips.append(el('span', { class: 'pb-chip', text: '정적/해당 위상 없음', on: { click: () => openNnFor(null, null, 'scenarios') } }));
  }
  row2.append(chips);

  const trail = p.nn?.trail ?? [];
  if (trail.length) {
    const trBox = el('span', { class: 'pb-trail', title: '최근 60초 N:N 위상 자취' });
    for (const t of trail.slice(-12)) {
      trBox.append(el('i', { class: `pb-dot ${itemClass(t)}`, title: `${t.scenario}/${t.phase} ${fmt.n(t.score, 0)}` }));
    }
    row2.append(trBox);
  }

  const hEnd = el('span', { class: 'pb-end' });
  const hCnt = p.holds?.hold_count ?? 0;
  const aCnt = p.holds?.abstain_count ?? 0;
  hEnd.append(el('button', {
    type: 'button', class: `pb-chip${hCnt ? ' hold' : ''}`, text: `보류 ${hCnt}`,
    title: '최근 보류(Hold) 기록 보기',
    on: { click: () => openNnFor(null, null, 'holds') },
  }));
  hEnd.append(el('button', {
    type: 'button', class: `pb-chip${aCnt ? ' abstain' : ''}`, text: `기권 ${aCnt}`,
    title: '최근 기권(이력 부족) 기록 — 보류 수에 포함되지 않음',
    on: { click: () => openNnFor(null, null, 'holds') },
  }));
  row2.append(hEnd);
  root.append(row2);
}

export function initProgressBand() {
  state.progressWorker = state.progressWorker ?? 1;
  bus.on('progress', (p) => {
    state.progress = p;
    render(p);
  });
}
