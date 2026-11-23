// 통합 테스트 타임라인 레인 등록부 · 증분 DOM 패치 · 바이트 상한 LRU 캐시 (Phase 5 – Part 2 · D3).
import { el } from './util.js';

/** 확장 가능한 레인 등록부: 새 레인은 여기에 한 줄만 더한다. */
export const LANE_SPECS = [
  { id: 'progress', kind: 'progress', label: '진행 띠', sub: 'N:N 위상 구간', h: 'h-prog', height: 24 },
  { id: 'game', kind: 'keyframes', pass: 'game', label: '1 게임', sub: '게임 로직 촬영', h: 'h-img', height: 76 },
  { id: 'act', kind: 'events', label: '행위', sub: '물리 순간 · 게임 / ML', h: 'h-act', height: 96 },
  { id: 'vision', kind: 'vision', label: '시각', sub: '봄 · 기억 · 오차', h: 'h-vis', height: 48 },
  { id: 'perceive', kind: 'ml', pass: 'game', label: '2 ML 인식', sub: '깊이 · 위치 · 추적', h: 'h-img', height: 76 },
  { id: 'decide', kind: 'decide', pass: 'ml', label: '3 ML 결정', sub: '예측 띠 · 이동 · 촬영', h: 'h-img', height: 76 },
  { id: 'gate', kind: 'gate', label: '게이트', sub: '시계열 판정 |z|', h: 'h-gate', height: 34 },
  { id: 'sds', kind: 'sds', label: '4 SDS', sub: 'z-점수 / 여유 부족', h: 'h-sds', height: 46 },
];

export const SDS_KEYS = ['depth', 'locate', 'tracker', 'decide', 'fusion', 'balance', 'act'];
export const SDS_KO = { depth: '깊이', locate: '위치', tracker: '추적', decide: '결정', fusion: '혼합', balance: '밸런스', act: '행위' };
export const KIND_KO = { search: '탐색 회전', triangulate: '삼각 측량', approach: '접근', arrive: '도달', hold: '보류', explore: '시점 이동', look_down: '아래 응시' };

/** 바이트 상한 LRU 키프레임 캐시 (기본 24 MiB ≒ 보이는 칸 4화면분). */
export class ByteLruCache {
  constructor(maxBytes = 24 * 1024 * 1024) {
    this.maxBytes = maxBytes;
    this.bytes = 0;
    this.map = new Map(); // key -> { value, size }
  }

  estimateBytes(v) {
    if (!v) return 0;
    let b = 512;
    if (typeof v.frame_webp === 'string') b += v.frame_webp.length;
    if (typeof v.ml_webp === 'string') b += v.ml_webp.length;
    if (v.focus_webp) for (const s of Object.values(v.focus_webp)) if (typeof s === 'string') b += s.length;
    return b;
  }

  has(k) { return this.map.has(k); }

  get(k) {
    const e = this.map.get(k);
    if (!e) return undefined;
    this.map.delete(k);
    this.map.set(k, e);
    return e.value;
  }

  set(k, value) {
    const size = this.estimateBytes(value);
    if (this.map.has(k)) {
      this.bytes -= this.map.get(k).size;
      this.map.delete(k);
    }
    this.map.set(k, { value, size });
    this.bytes += size;
    while (this.bytes > this.maxBytes && this.map.size > 1) {
      const oldest = this.map.keys().next().value;
      this.bytes -= this.map.get(oldest).size;
      this.map.delete(oldest);
    }
  }

  clear() {
    this.map.clear();
    this.bytes = 0;
  }
}

export function tickT(k) {
  return Number.isFinite(k.t) ? k.t : (k.tick ?? 0) / 10;
}

export function rNum(x, d = 2) {
  return Number.isFinite(x) ? Number(x).toFixed(d) : '—';
}

function zClass(z) {
  if (!Number.isFinite(z)) return '';
  const a = Math.abs(z);
  return a < 1 ? 'z0' : a < 2 ? 'z1' : a < 3 ? 'z2' : 'z3';
}

/** F9 수정: SDS 칸을 키보드 접근 가능한 <button> 으로 만들고, 스코프·n 과 밸런스 여유 부족 척도를 구분한다. */
export function buildSdsCell(pass, k, x, row, onPick) {
  const cells = k.sds_cells ?? {};
  const raw = k.sds ?? {};
  const dots = [];
  const tips = [];
  for (const key of SDS_KEYS) {
    const det = cells[key];
    const v = det ? det.value : raw[key];
    if (v == null) continue;
    const isShortfall = det?.kind === 'shortfall' || key === 'balance';
    const cls = isShortfall ? (v <= 0.1 ? 'z0' : v < 1 ? 'z1' : v < 2 ? 'z2' : 'z3') : zClass(v);
    dots.push(el('i', { class: cls, title: `${SDS_KO[key] ?? key}: ${rNum(v, 2)}` }));
    const meta = det ? ` (n=${det.n}, ${det.scope})` : '';
    tips.push(`${SDS_KO[key] ?? key} ${isShortfall ? '여유부족' : 'z'}=${rNum(v, 2)}${meta}`);
  }
  const btn = el('button', {
    type: 'button',
    class: 'tl-sds-cell tl-sds-btn',
    dataset: { kfKey: `sds:${pass}:${k.i}`, pass, index: String(k.i) },
    style: `left:${x}px;top:${4 + row * 21}px`,
    title: tips.length ? `${rNum(tickT(k), 1)}s · ${tips.join(' | ')}` : `${rNum(tickT(k), 1)}s · SDS 없음`,
    on: { click: (e) => { e.stopPropagation(); onPick?.(pass, k); } },
  }, ...dots);
  return btn;
}

/** F12 수정: ML 결정 칸에 예측 띠(band) 폭 요약을 함께 표시한다. */
export function decideCaption(d) {
  if (!d) return '—';
  const base = KIND_KO[d.kind] ?? d.label ?? '—';
  const res = d.residual != null ? ` ${rNum(d.residual, 1)}°` : '';
  let bandTxt = '';
  if (Array.isArray(d.band) && d.band.length === 4) {
    const wx = Math.abs(d.band[2] - d.band[0]);
    const wz = Math.abs(d.band[3] - d.band[1]);
    bandTxt = ` 띠±${rNum(0.5 * Math.hypot(wx, wz), 2)}m`;
  }
  return `${base}${res}${bandTxt}`;
}

/** 진행 띠 레인에 키프레임 위상 참조(ctx.phase_ref) 색 구간을 덧붙인다. */
export function buildProgressChip(k, x, w = 44) {
  const ref = k.ctx?.phase_ref;
  if (!ref) return null;
  return el('span', {
    class: 'pb-chip',
    dataset: { kfKey: `prog:${k.i}` },
    style: `position:absolute;left:${x}px;top:2px;max-width:${Math.max(44, w)}px;overflow:hidden;text-overflow:ellipsis;padding:0 5px;font-size:0.56rem`,
    title: `${rNum(tickT(k), 1)}s · ${ref}`,
    text: ref,
  });
}

/** 게이트 레인에 게이트 판정 막대(|z| 높이, 결과 색)를 그린다. */
export function renderGateLane(lane, marks, pps, onPickTick) {
  if (!lane) return;
  while (lane.firstChild) lane.removeChild(lane.firstChild);
  for (const m of marks ?? []) {
    const x = (m.tick / 10) * pps;
    const h = Math.max(6, Math.min(30, Math.round(Math.abs(m.z ?? 0.5) * 7)));
    lane.append(el('button', {
      type: 'button',
      class: `tl-gate-bar ${m.outcome ?? 'accepted'}`,
      style: `left:${x}px;height:${h}px;border:none;padding:0`,
      title: `틱 ${m.tick} · ${m.label ?? ''} · ${m.outcome}${m.rescue ? ` (${m.rescue})` : ''} · z=${rNum(m.z, 2)}`,
      on: { click: () => onPickTick?.(m.tick) },
    }));
  }
}

/** delta 응답을 기존 status 객체에 병합해 변경된(appended / updated) 키프레임 목록을 돌려준다. */
export function mergeStatusDelta(prev, delta) {
  if (!delta?.active) return { status: { active: false }, reset: true, appended: { game: [], decide: [] }, updated: { game: [], decide: [] } };
  const reset = !prev?.active || prev.id !== delta.id;
  const base = reset
    ? {
      active: true, id: delta.id, live: delta.live, ml_needed: delta.ml_needed, command: delta.command,
      phase: delta.phase, tick: delta.tick, test_rev: delta.test_rev, perceive: delta.perceive,
      game: { keyframes: [], vision: [], belief_err: [] },
      decide: { keyframes: [], vision: [], belief_err: [] },
      notes: delta.notes ?? [], error: delta.error ?? null,
    }
    : { ...prev, phase: delta.phase, tick: delta.tick, test_rev: delta.test_rev, perceive: delta.perceive, game_quest: delta.game_quest, ml_quest: delta.ml_quest, notes: delta.notes ?? [], error: delta.error ?? null };

  const appended = { game: [], decide: [] };
  const updated = { game: [], decide: [] };
  for (const key of ['game', 'decide']) {
    const d = delta[key];
    if (!d) continue;
    const cur = base[key] ?? { keyframes: [], vision: [], belief_err: [] };
    const kfs = cur.keyframes ? [...cur.keyframes] : [];
    const byIdx = new Map(kfs.map((k, pos) => [k.i, pos]));
    for (const k of d.appended ?? []) {
      if (byIdx.has(k.i)) {
        kfs[byIdx.get(k.i)] = k;
        updated[key].push(k);
      } else {
        byIdx.set(k.i, kfs.length);
        kfs.push(k);
        appended[key].push(k);
      }
    }
    for (const k of d.updated ?? []) {
      if (byIdx.has(k.i)) {
        kfs[byIdx.get(k.i)] = k;
        updated[key].push(k);
      } else {
        byIdx.set(k.i, kfs.length);
        kfs.push(k);
        appended[key].push(k);
      }
    }
    base[key] = {
      ...cur,
      run_id: d.run_id ?? cur.run_id,
      start_tick: d.start_tick ?? cur.start_tick,
      end_tick: d.end_tick ?? cur.end_tick,
      video: d.video ?? cur.video,
      dropped: d.dropped ?? cur.dropped,
      keyframes: kfs,
      vision: [...(cur.vision ?? []), ...(d.vision_tail ?? [])],
      belief_err: [...(cur.belief_err ?? []), ...(d.belief_tail ?? [])],
    };
  }
  return { status: base, reset, appended, updated };
}
