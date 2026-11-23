// 공유 시간 커서 (Phase 5 – Part 2 · D4): 키프레임을 고르면 월드·로그·SDS·시계열이 같은 틱으로 맞춘다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';

const cursor = { mode: 'live', tick: null, pass: 'game', dual: false, ctx: null };
let reqSeq = 0;

export function getCursor() {
  return cursor;
}

function renderBanner() {
  const box = $('cursor-banner');
  if (!box) return;
  if (cursor.mode !== 'past' || cursor.tick == null) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  clear(box);
  const tSec = fmt.n(cursor.tick / 10, 1);
  const passKo = cursor.pass === 'ml' ? 'ML 결정 통과' : '게임 통과';
  const span = cursor.ctx?.phase_spans?.[0];
  const phTxt = span ? ` · ${span.scenario}/${span.phase}` : '';
  box.append(
    el('span', { class: 'cursor-chip', text: `과거 ${tSec}초 (틱 ${cursor.tick})` }),
    el('span', { text: `${passKo}${phTxt}` }),
    el('button', {
      type: 'button', class: 'pb-btn', text: cursor.dual ? '두 통과 끄기' : '두 통과 보기',
      on: { click: () => { cursor.dual = !cursor.dual; renderBanner(); bus.emit('cursor', { ...cursor }); } },
    }),
    el('button', { type: 'button', class: 'pb-btn', text: '실시간으로 (Esc)', on: { click: () => toLive() } }),
  );
}

/** 지정 틱으로 커서를 옮기고 서버 test_context 를 받아 모든 패널에 알린다. */
export async function setCursor(tick, pass = 'game') {
  if (tick == null) return toLive();
  cursor.mode = 'past';
  cursor.tick = Math.max(0, Math.round(tick));
  cursor.pass = pass === 'ml' ? 'ml' : 'game';
  state.cursor = { ...cursor };
  renderBanner();
  const seq = ++reqSeq;
  try {
    const ctx = await call('test_context', { pass: cursor.pass, tick: cursor.tick }, { quiet: true });
    if (seq !== reqSeq || cursor.mode !== 'past') return;
    cursor.ctx = ctx;
    state.cursor = { ...cursor };
    renderBanner();
    scrollLogTo(ctx?.log_window?.center_seq);
    bus.emit('cursor', { ...cursor });
  } catch {
    bus.emit('cursor', { ...cursor });
  }
}

export function toLive() {
  if (cursor.mode === 'live') return;
  reqSeq++;
  cursor.mode = 'live';
  cursor.tick = null;
  cursor.ctx = null;
  cursor.dual = false;
  state.cursor = { ...cursor };
  renderBanner();
  bus.emit('cursor', { ...cursor });
}

function scrollLogTo(seq) {
  if (!seq) return;
  const view = $('log-view');
  if (!view) return;
  const row = view.querySelector(`[data-seq="${seq}"]`);
  if (row) row.scrollIntoView({ block: 'center' });
}

export function initCursor() {
  state.cursor = { ...cursor };
  renderBanner();
  document.addEventListener('keydown', (e) => {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.key === 'Escape' && cursor.mode === 'past') {
      toLive();
      e.preventDefault();
    } else if (e.key === ',' || e.key === '.') {
      bus.emit('cursor-step', e.key === '.' ? 1 : -1);
    }
  });
  bus.on('world-new', () => toLive());
}
