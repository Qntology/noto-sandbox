// 도킹 분석 작업면 (Phase 5 – Part 2 · D5): 오른쪽(40%, 최소 420px) · 아래(45vh) · 접힘/반/전체 3크기.
// 전체 크기일 때만 월드 폴링을 멈추고, 반/접힘에서는 월드와 분석을 나란히 본다.
import { bus } from './state.js';
import { $, storageGet, storageSet } from './util.js';
import { isOpen } from './overlays.js';
import { getRevs, getTransport } from './poller.js';

let side = storageGet('dock-side', 'right'); // 'right' | 'bottom'
let size = storageGet('dock-size', 'half'); // 'collapsed' | 'half' | 'full'

export function isWorldCovered() {
  return isOpen('analysis-modal') && size === 'full';
}

export function getDockState() {
  return { side, size, open: isOpen('analysis-modal') };
}

function applyClasses() {
  const modal = $('analysis-modal');
  if (!modal) return;
  const open = !modal.hidden;
  modal.classList.toggle('docked', size !== 'full');
  modal.classList.toggle('dock-right', side === 'right' && size !== 'full');
  modal.classList.toggle('dock-bottom', side === 'bottom' && size !== 'full');
  modal.classList.toggle('dock-collapsed', size === 'collapsed');
  modal.classList.toggle('dock-half', size === 'half');
  modal.classList.toggle('dock-full', size === 'full');
  document.body.classList.toggle('dock-right-half', open && side === 'right' && size === 'half');
  document.body.classList.toggle('dock-bottom-half', open && side === 'bottom' && size === 'half');
  const posBtn = $('dock-side');
  if (posBtn) posBtn.textContent = side === 'right' ? '⬇ 아래로' : '➡ 오른쪽';
  const szBtn = $('dock-size');
  if (szBtn) szBtn.textContent = size === 'collapsed' ? '펼치기' : size === 'half' ? '전체' : '반 크기';
  const colBtn = $('dock-collapse');
  if (colBtn) colBtn.textContent = size === 'collapsed' ? '◀' : '▶';
  bus.emit('dock', { side, size, open });
}

function updateMeta(revs) {
  const m = $('dock-meta');
  if (m && revs) m.textContent = `틱 ${revs.tick ?? 0} · sds:${revs.sds ?? 0} · ts:${revs.ts ?? 0} · test:${revs.test ?? 0}`;
  const tr = $('dock-transport');
  if (tr) {
    const mode = getTransport();
    tr.textContent = mode === 'tauri' ? '푸시(Tauri)' : mode === 'sse' ? '푸시(SSE)' : '폴링';
  }
}

export function setDockSize(next) {
  size = next;
  storageSet('dock-size', size);
  applyClasses();
}

export function initDock() {
  $('dock-side')?.addEventListener('click', () => {
    side = side === 'right' ? 'bottom' : 'right';
    storageSet('dock-side', side);
    applyClasses();
  });
  $('dock-size')?.addEventListener('click', () => {
    size = size === 'full' ? 'half' : 'full';
    storageSet('dock-size', size);
    applyClasses();
  });
  $('dock-collapse')?.addEventListener('click', () => {
    size = size === 'collapsed' ? 'half' : 'collapsed';
    storageSet('dock-size', size);
    applyClasses();
  });
  bus.on('panel', ({ id }) => { if (id === 'analysis-modal') applyClasses(); });
  bus.on('revs', updateMeta);
  bus.on('poller-transport', () => updateMeta(getRevs()));
  applyClasses();
  updateMeta(getRevs());
}
