// 시계열 · 예측 화면 (Phase 5 – Part 2 · D2 & D4):
// ts_series 등록부(lm:* · label:* · dec:* · sds:* · ts:*) + ts_view 게이트 판정 추적(틱 x축, 계단 띠, 구제/보류 점, 가중치 막대, 실현 오차, 커서 until_tick).
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, cssVar, el, fitCanvas, fmt, observeResize } from './util.js';
import { isOpen } from './overlays.js';
import { register, invalidate } from './poller.js';

let tab = null;
let busy = false;
let hoverIdx = null;
let catalog = [];
let viewData = null;
let selectedSeries = '';
let lastCatalogRev = null;
let cursorTick = null;

function alpha(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function syncLegacySelectors() {
  const scopeSel = $('ts-scope');
  const axisSel = $('ts-axis');
  if (!scopeSel || !axisSel) return;
  const map = new Map();
  for (const f of state.sds?.forecastable ?? state.sds?.axes ?? []) {
    if (!map.has(f.scope)) map.set(f.scope, new Set());
    map.get(f.scope).add(f.axis);
  }
  const scopes = [...map.keys()];
  const prevS = scopeSel.value;
  clear(scopeSel);
  for (const s of scopes) scopeSel.append(el('option', { value: s, text: s }));
  scopeSel.value = scopes.includes(prevS) ? prevS : (scopes[0] ?? '');
  const axes = [...(map.get(scopeSel.value) ?? [])];
  const prevA = axisSel.value;
  clear(axisSel);
  for (const a of axes) axisSel.append(el('option', { value: a, text: a }));
  axisSel.value = axes.includes(prevA) ? prevA : (axes[0] ?? '');
}

function syncSeriesSelector() {
  const sel = $('ts-series-sel');
  if (!sel) return;
  const keep = selectedSeries || sel.value;
  clear(sel);
  if (!catalog.length) {
    sel.append(el('option', { value: '', text: '(계열 없음)' }));
    return;
  }
  for (const item of catalog) {
    const badge = item.gate_use ? '★ ' : '';
    sel.append(el('option', { value: item.id, text: `${badge}${item.label} (${item.n})` }));
  }
  if (catalog.some((x) => x.id === keep)) sel.value = keep;
  else {
    sel.value = catalog[0].id;
    selectedSeries = sel.value;
  }
}

export async function refreshCatalog() {
  const res = await call('ts_series', { if_rev: lastCatalogRev ?? undefined }, { quiet: true }).catch(() => null);
  if (!res || res.unchanged) return;
  lastCatalogRev = res.rev;
  catalog = res.series ?? [];
  state.tsCatalog = res;
  // F13: TTM 상태 출처 일원화 (ts_series.ttm)
  if (res.ttm && $('sb-ttm')) {
    $('sb-ttm').textContent = res.ttm.attached ? `TTM ${res.ttm.fp8 ?? 'ON'}` : '미연결';
  }
  syncSeriesSelector();
}

export async function refreshForecast() {
  if (busy) return;
  busy = true;
  try {
    await refreshCatalog();
    const series = selectedSeries || $('ts-series-sel')?.value || '';
    if (series) {
      const args = { series, horizon: 12 };
      if (cursorTick != null) args.until_tick = cursorTick;
      viewData = await call('ts_view', args, { quiet: true });
      state.forecast = viewData;
      bus.emit('forecast', viewData);
      renderSide();
      draw();
      return;
    }
    const scope = $('ts-scope')?.value;
    const axis = $('ts-axis')?.value;
    if (!scope || !axis) {
      viewData = null;
      $('ts-info').textContent = '예측할 계열이 아직 없습니다. 인식이 쌓이면 선택지가 나타납니다.';
      renderSide();
      draw();
      return;
    }
    const f = await call('ts_forecast', { scope, axis }, { quiet: true });
    viewData = {
      series: `${scope}:${axis}`, gate_use: false,
      points: (f.history ?? []).map((v, i) => ({ tick: i * 10, value: v, outcome: 'accepted' })),
      band_1step: [],
      forecast: { point: f.point ?? [], p10: f.p10 ?? [], p90: f.p90 ?? [], step_ticks: 10, method: f.method },
      bands: f.bands ?? null,
      weights: f.weights ?? [],
      rmse_fused: f.rmse_fused, holdout: f.holdout, realized: { n: 0, nominal: 0.8 }, notes: f.notes ?? [],
    };
    state.forecast = viewData;
    renderSide();
    draw();
  } catch (e) {
    $('ts-info').textContent = `예측 조회 실패: ${e.message}`;
  } finally {
    busy = false;
  }
}

function renderSide() {
  const side = $('ts-side');
  if (!side) return;
  clear(side);
  const v = viewData;
  if (!v) {
    side.append(el('p', { class: 'empty-note', text: '계열을 고르면 방법별 가중치 · 실현 오차 · 최근 12판정이 보입니다.' }));
    return;
  }
  // 실현 오차 요약 + 스파크라인
  const rl = v.realized ?? {};
  const inBandTxt = rl.in_band != null ? fmt.pct(rl.in_band, 0) : '—';
  side.append(
    el('div', { class: 'ct-sec', text: '실현 1스텝 오차' }),
    el('div', { class: 'ts-realized' },
      el('div', { text: `n=${rl.n ?? 0} · MAE ${rl.mae != null ? fmt.n(rl.mae, 3) : '—'}` }),
      el('div', { text: `띠 안 ${inBandTxt} (명목 80%)` }),
      el('div', { class: 'hint', style: 'margin:2px 0 0', text: 'SDS σ로 넓힌 게이트 띠는 명목 80%보다 높게 나오는 것이 정상입니다.' })));

  // 방법별 가중치 막대
  const ws = v.weights ?? [];
  if (ws.length) {
    side.append(el('div', { class: 'ct-sec', text: `예측기 가중치 (융합 RMSE ${v.rmse_fused != null ? fmt.n(v.rmse_fused, 3) : '—'})` }));
    for (const w of ws) {
      const pct = Math.round((w.weight ?? 0) * 100);
      side.append(el('div', { class: 'ts-w-bar', title: `RMSE ${fmt.n(w.rmse, 3)} · 과거 ${w.history_n ? `${fmt.n(w.history_rmse, 3)} (n=${w.history_n})` : '없음'}` },
        el('span', { class: 'mono', text: w.method }),
        el('div', { class: 'ts-w-track' }, el('div', { class: 'ts-w-fill', style: `width:${pct}%` })),
        el('span', { class: 'mono', text: `${pct}%` })));
    }
  }

  // 최근 12판정 표
  const pts = (v.points ?? []).slice(-12).reverse();
  if (pts.length) {
    side.append(
      el('div', { class: 'ct-sec', text: '최근 12판정' }),
      el('table', { class: 'pv-table' },
        el('thead', {}, el('tr', {}, ['틱', '값', '결과', 'z'].map((h) => el('th', { text: h })))),
        el('tbody', {}, pts.map((p) => el('tr', {},
          el('td', { class: 'mono', text: String(p.tick) }),
          el('td', { class: 'num', text: fmt.n(p.value, 2) }),
          el('td', { text: p.rescue ? `구제(${p.rescue})` : p.outcome }),
          el('td', { class: 'num', text: p.z != null ? fmt.n(p.z, 1) : '—' }))))));
  }
}

function draw() {
  const fit = fitCanvas($('ts-canvas'));
  if (!fit) return;
  const { ctx, W, H } = fit;
  ctx.clearRect(0, 0, W, H);
  const C = {
    grid: cssVar('--plot-grid-strong'), axis: cssVar('--text-muted'), text: cssVar('--text-dim'),
    hist: cssVar('--accent'), fc: cssVar('--violet'), warn: cssVar('--warning'),
    ok: cssVar('--success'), hold: cssVar('--hold'), alert: cssVar('--alert'),
  };
  ctx.font = `10px ${cssVar('--font-mono')}`;
  const v = viewData;
  const pts = v?.points ?? [];
  if (!v || !pts.length) {
    ctx.fillStyle = C.axis;
    ctx.textAlign = 'center';
    ctx.fillText('계열 관측이 없습니다 — 실행 후 새로고침하세요', W / 2, H / 2);
    return;
  }
  const fc = v.forecast;
  const stepT = fc?.step_ticks ?? 10;
  const lastTk = pts[pts.length - 1].tick;
  const fcPts = (fc?.point ?? []).map((val, i) => ({ tick: lastTk + (i + 1) * stepT, value: val, p10: fc.p10?.[i] ?? val, p90: fc.p90?.[i] ?? val }));
  const gateBand = v.bands?.gate;
  const calBand = v.bands?.cal80;
  const infoBits = [
    `${v.series}${v.gate_use ? ' [게이트 입력]' : ''}`,
    `점 ${pts.length}`,
    fc?.method ? `방법 ${fc.method}` : null,
    v.holdout ? `홀드아웃 ${v.holdout}` : null,
    v.until_tick != null ? `커서 ≤ 틱 ${v.until_tick}` : null,
    ...(v.notes ?? []),
  ].filter(Boolean);
  $('ts-info').textContent = infoBits.join(' · ');

  const tMin = pts[0].tick;
  const tMax = Math.max(tMin + 10, fcPts.length ? fcPts[fcPts.length - 1].tick : lastTk + 10);
  const yAll = [
    ...pts.map((p) => p.value),
    ...fcPts.flatMap((p) => [p.value, p.p10, p.p90]),
    ...(v.band_1step ?? []).flatMap((b) => [b.lo, b.hi]).filter(Number.isFinite),
    ...(gateBand ? [...(gateBand.lo ?? []), ...(gateBand.hi ?? [])] : []),
  ].filter(Number.isFinite);
  let lo = Math.min(...yAll);
  let hi = Math.max(...yAll);
  if (hi - lo < 1e-6) { lo -= 0.25; hi += 0.25; }
  const span = hi - lo;
  lo -= span * 0.08;
  hi += span * 0.08;

  const padL = 48, padR = 14, padT = 24, padB = 24;
  const X = (tk) => padL + ((tk - tMin) / Math.max(1, tMax - tMin)) * (W - padL - padR);
  const Y = (val) => padT + (1 - (val - lo) / (hi - lo)) * (H - padT - padB);

  // 격자와 y 눈금
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = C.axis;
  ctx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const val = lo + ((hi - lo) * k) / 4;
    const y = Math.round(Y(val)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(fmt.n(val, Math.abs(hi - lo) < 2 ? 3 : 1), padL - 5, y + 3);
  }

  // 판정 당시 1스텝 게이트 띠 (계단 음영)
  for (let i = 0; i < (v.band_1step ?? []).length; i++) {
    const b = v.band_1step[i];
    if (b.lo == null || b.hi == null) continue;
    const x0 = X(b.tick - stepT * 0.45);
    const x1 = X(b.tick + stepT * 0.45);
    ctx.fillStyle = alpha(C.ok, 0.14);
    ctx.fillRect(x0, Y(b.hi), Math.max(3, x1 - x0), Math.max(1, Y(b.lo) - Y(b.hi)));
  }

  // 미래 예측 띠 (gate / cal80 / model)
  const fillFan = (loArr, hiArr, col) => {
    if (!fcPts.length || !loArr?.length || !hiArr?.length) return;
    ctx.beginPath();
    ctx.moveTo(X(lastTk), Y(pts[pts.length - 1].value));
    fcPts.forEach((p, i) => ctx.lineTo(X(p.tick), Y(hiArr[i] ?? p.p90)));
    for (let i = fcPts.length - 1; i >= 0; i--) ctx.lineTo(X(fcPts[i].tick), Y(loArr[i] ?? fcPts[i].p10));
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();
  };
  if (gateBand) fillFan(gateBand.lo, gateBand.hi, alpha(C.warn, 0.12));
  if (calBand) fillFan(calBand.lo, calBand.hi, alpha(C.fc, 0.18));
  else if (fcPts.length) fillFan(fcPts.map((p) => p.p10), fcPts.map((p) => p.p90), alpha(C.fc, 0.20));

  // 이력 선
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = C.hist;
  ctx.beginPath();
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p.tick), Y(p.value)) : ctx.lineTo(X(p.tick), Y(p.value))));
  ctx.stroke();

  // 미래 P50 점선 (중앙 표본 간격 가정)
  if (fcPts.length) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = C.fc;
    ctx.beginPath();
    ctx.moveTo(X(lastTk), Y(pts[pts.length - 1].value));
    fcPts.forEach((p) => ctx.lineTo(X(p.tick), Y(p.value)));
    ctx.stroke();
    ctx.restore();
  }

  // 관측 점 (수락=채움, 보류=빈 빨강, 구제=빈 주황, 기권=회색)
  for (const p of pts) {
    const cx = X(p.tick);
    const cy = Y(p.value);
    ctx.beginPath();
    ctx.arc(cx, cy, 3.2, 0, Math.PI * 2);
    if (p.rescue) {
      ctx.strokeStyle = C.alert; ctx.lineWidth = 1.6; ctx.stroke();
    } else if (p.outcome === 'held') {
      ctx.strokeStyle = C.hold; ctx.lineWidth = 1.6; ctx.stroke();
    } else if (p.outcome === 'warn') {
      ctx.fillStyle = C.warn; ctx.fill();
    } else if (p.outcome === 'abstained') {
      ctx.fillStyle = C.axis; ctx.fill();
    } else {
      ctx.fillStyle = C.ok; ctx.fill();
    }
  }

  // 커서 세로선 + x축 틱 눈금
  if (cursorTick != null) {
    const cx = Math.round(X(cursorTick)) + 0.5;
    ctx.strokeStyle = C.warn;
    ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, H - padB); ctx.stroke();
  }
  ctx.fillStyle = C.axis;
  ctx.textAlign = 'left';
  ctx.fillText(`틱 ${tMin}`, padL, H - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`틱 ${tMax} (간격≈${stepT}틱)`, W - padR, H - 8);

  // 범례
  const leg = [[C.ok, '수락'], [C.hold, '보류(빈원)'], [C.alert, '구제'], [C.fc, 'P50/보정띠']];
  let lx = padL;
  ctx.textAlign = 'left';
  for (const [col, nm] of leg) {
    ctx.fillStyle = col;
    ctx.fillRect(lx, 8, 8, 8);
    ctx.fillStyle = C.text;
    ctx.fillText(nm, lx + 11, 16);
    lx += 18 + ctx.measureText(nm).width;
  }
  if (hoverIdx != null && pts[hoverIdx]) {
    const p = pts[hoverIdx];
    ctx.fillStyle = C.text;
    ctx.textAlign = 'right';
    ctx.fillText(`틱 ${p.tick} · 값 ${fmt.n(p.value, 3)} · ${p.rescue ? `구제(${p.rescue})` : p.outcome}`, W - padR, 16);
  }
}

export function initTimeseries() {
  const canvas = $('ts-canvas');
  observeResize(canvas, draw);
  canvas.addEventListener('mousemove', (e) => {
    const pts = viewData?.points ?? [];
    if (!pts.length) return;
    const r = canvas.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left - 48) / Math.max(1, r.width - 62)));
    hoverIdx = Math.min(pts.length - 1, Math.round(frac * (pts.length - 1)));
    draw();
  });
  canvas.addEventListener('mouseleave', () => { hoverIdx = null; draw(); });
  $('ts-series-sel')?.addEventListener('change', (e) => { selectedSeries = e.target.value; refreshForecast(); });
  $('ts-scope')?.addEventListener('change', () => { syncLegacySelectors(); selectedSeries = ''; refreshForecast(); });
  $('ts-axis')?.addEventListener('change', () => { selectedSeries = ''; refreshForecast(); });
  $('ts-refresh')?.addEventListener('click', () => { lastCatalogRev = null; refreshForecast(); });

  bus.on('sds', () => syncLegacySelectors());
  bus.on('open-ts', ({ series }) => {
    if (series) {
      selectedSeries = series;
      lastCatalogRev = null;
      refreshForecast();
    }
  });
  bus.on('cursor', (c) => {
    cursorTick = c.mode === 'past' ? c.tick : null;
    if (isOpen('analysis-modal') && tab === 'ts') refreshForecast();
  });
  bus.on('tab', ({ group, name }) => {
    if (group !== 'analysis') return;
    tab = name;
    if (name === 'ts') {
      syncLegacySelectors();
      invalidate('ts-view');
    }
  });
  register('ts-view', {
    rev: 'ts',
    minGap: 600,
    visible: () => isOpen('analysis-modal') && tab === 'ts',
    run: () => refreshForecast(),
  });
  syncLegacySelectors();
  draw();
}
