// 모의 백엔드용 진행 띠 · N:N · SDS 탐색기 · 시계열 · 커서 응답 생성기 (Phase 5 · D5).
// 실제 Rust 응답(progress.rs, sdsview.rs, api.rs, testrun.rs)과 키 구조를 맞춘다.

const SEG_NAMES = ['head', 'chest', 'belly', 'arm', 'hand', 'leg', 'foot'];
const MATRIX_SRCS = ['arm_raise', 'arm_lift', 'foot_lift', 'step_up', 'slope', 'carry_front', 'accel'];

const S4_LADDER = [
  { id: 'carry', score: 86.2, ungated: 86.2, item: 2, gated: false, excused: false },
  { id: 'brake_0.5', score: 82.5, ungated: 82.5, item: 2, gated: false, excused: false },
  { id: 'brake_1.5', score: 74.1, ungated: 74.1, item: 1, gated: false, excused: false },
  { id: 'brake_2.5', score: 69.0, ungated: 76.4, item: 1, gated: true, excused: true },
];

export function mockRevs(tick = 0) {
  return {
    tick,
    sds: Math.floor(tick / 4) + 12,
    ts: Math.floor(tick / 6) + 5,
    test: Math.floor(tick / 10) + 1,
    nn: Math.floor(tick / 15) + 3,
    log_seq: Math.floor(tick / 3) + 8,
  };
}

export function mockProgress(tick = 0, focus = 1) {
  const idx = Math.floor(tick / 40) % S4_LADDER.length;
  const cur = S4_LADDER[idx];
  const holds = [
    { tick: Math.max(0, tick - 30), worker: 1, domain: 'precision', code: 'temporal', verdict: 'abstained', subject: 'tree', measured: 2, limit: 6, band: null, unit: '점', detail_ko: '이력 2점 (판정에 6점 필요)' },
    { tick: Math.max(0, tick - 20), worker: 1, domain: 'precision', code: 'temporal', verdict: 'abstained', subject: 'tree', measured: 3, limit: 6, band: null, unit: '점', detail_ko: '이력 3점 (판정에 6점 필요)' },
    { tick: Math.max(0, tick - 12), worker: 1, domain: 'precision', code: 'temporal', verdict: 'abstained', subject: 'tree', measured: 4, limit: 6, band: null, unit: '점', detail_ko: '이력 4점 (판정에 6점 필요)' },
    { tick: Math.max(0, tick - 4), worker: 1, domain: 'precision', code: 'temporal', verdict: 'held', subject: 'log', measured: 0.42, limit: 0.78, band: [0.68, 0.88], unit: '', detail_ko: '점수 급락 보류 (z=-3.1)' },
  ];
  const now = focus === 1 ? [{
    scenario: 'S4', phase: cur.id, ladder: 'brake', since_tick: Math.max(0, tick - 12),
    state: 'now', score: cur.score, ungated: cur.ungated, item: cur.item,
    gated: cur.gated, excused: cur.excused, top_loss: ['foot', 11.4],
  }] : [];
  const trail = S4_LADDER.slice(0, idx + 1).map((r, i) => ({
    scenario: 'S4', phase: r.id, ladder: 'brake', since_tick: Math.max(0, tick - (idx - i) * 30),
    state: i === idx ? 'now' : 'observed', score: r.score, ungated: r.ungated, item: r.item,
    gated: r.gated, excused: r.excused, top_loss: ['foot', 10.0],
  }));
  return {
    tick,
    focus,
    quest: { kind: 'carry', index: 4, count: 6, progress: 0.62, label_ko: '원목 운반 (모의)', title: '원목 1개 가져오기' },
    procedure: focus === 1 ? { mechanism: 'hand', stage: 'lift', stage_ko: '들어올림', index: 6, count: 8, segment: null } : null,
    nn: focus === 1
      ? { status: 'computed', reason: null, form_id: 'mid:humanfoot:f5', rev: mockRevs(tick).nn, now, trail }
      : { status: 'not_computed', reason: '서브 작업자 N:N 미계산', form_id: 'crew', rev: 0, now: [], trail: [] },
    holds: { hold_count: 1, abstain_count: 3, items: holds },
    workers: [{ id: 1, first: true, form: 'humanoid', quest_ko: '원목 운반', stage_ko: '들어올림', nn: 'computed', holds_n: 1 }],
  };
}

export function mockNnStatus(args = {}, tick = 0) {
  const revs = mockRevs(tick);
  const selScn = args?.scenario || 'S4';
  const selPh = args?.phase || 'brake_2.5';
  const cells = [
    { cell_key: 'accel>foot:cop_shift_frac', source: 'accel', target: 'foot', axis: 'cop_shift_frac', value: 0.44, reference: 0.30, tolerance: 0.12, z: 1.15, score: 86.2, loss: 11.8, dynamic: true },
    { cell_key: 'accel>belly:lean_deg', source: 'accel', target: 'belly', axis: 'lean_deg', value: 11.4, reference: 8.0, tolerance: 4.0, z: 0.88, score: 89.4, loss: 7.2, dynamic: true },
    { cell_key: 'accel>head:tilt_deg', source: 'accel', target: 'head', axis: 'tilt_deg', value: 4.2, reference: 2.0, tolerance: 5.0, z: 0.45, score: 94.6, loss: 4.6, dynamic: true },
  ];
  const scenario_states = [
    { scenario: 'S1', name_ko: '경사 위 들기', state: 'seen', phases: [{ phase: 'slope', state: 'seen', ladder: null, score: 88.4, events: 4, sds_n: 4 }, { phase: 'lift', state: 'no_record', ladder: null, score: null, events: 0, sds_n: 0 }] },
    { scenario: 'S2', name_ko: '앞 짐 들고 턱 오르기', state: 'seen', phases: [{ phase: 'carry_walk', state: 'seen', ladder: null, score: 84.1, events: 8, sds_n: 8 }] },
    { scenario: 'S3', name_ko: '한 발로 서서 앞으로 뻗기', state: 'skipped_form', phases: [{ phase: 'weight_shift', state: 'skipped_form', ladder: null, score: null, events: 0, sds_n: 0 }] },
    {
      scenario: 'S4', name_ko: '짐 들고 제동 (강도 사다리)', state: 'active',
      phases: S4_LADDER.map((r, i) => ({
        phase: r.id, ladder: 'brake', state: i === 3 ? 'active' : 'seen',
        score: r.score, item: r.item, excused: r.excused, events: i + 2, sds_n: i + 2,
      })),
    },
    { scenario: 'S6', name_ko: '경사 위에서 팔 들어 치고 출발', state: 'no_record', phases: [{ phase: 'slope', state: 'no_record', ladder: null, score: null, events: 0, sds_n: 0 }] },
  ];
  const matrix = MATRIX_SRCS.map((src, si) => SEG_NAMES.map((seg, gi) => {
    const has = (si + gi) % 3 !== 0;
    const wz = has ? Number((0.4 + ((si * 7 + gi * 3) % 5) * 0.45).toFixed(2)) : null;
    return {
      source: src, target: seg, worst_z: wz,
      item: wz == null ? null : wz <= 1 ? 2 : wz <= 2 ? 1 : 0,
      loss_sum: has ? Number((wz * 3.1).toFixed(1)) : 0,
      count: has ? 6 + si : 0, has_ref: has, last_tick: has ? tick : null, in_now: si === 6,
    };
  }));
  return {
    ref_rev: 1,
    rev: revs.nn,
    epoch: 1,
    form_id: 'mid:humanfoot:f5',
    engine_rev: 'mock',
    catalog: scenario_states.map((s) => ({ id: s.scenario, name_ko: s.name_ko, needs_legs: s.scenario === 'S2' || s.scenario === 'S3', skipped_reason: s.state === 'skipped_form' ? '다리 전용 시나리오' : null })),
    scenario_states,
    detail: {
      scenario: selScn, name_ko: '짐 들고 제동', phase: selPh, score: 69.0, ungated: 76.4, unscored_w: 0.35, item: 1,
      seg_scores: [92, 86, 78, 84, 88, 72, 61],
      cells,
      top_loss: ['accel>foot:cop_shift_frac', 11.8],
      gates: ['dyn_margin'],
      gate_hits: [{ code: 'dyn_margin', measured: -0.02, limit: 0.0, excused: true }],
      history: Array.from({ length: 8 }, (_, i) => ({ tick: Math.max(0, tick - (7 - i) * 10), score: 78 - i * 1.1, ungated: 80 - i * 0.4, item: i < 5 ? 2 : 1, gated: i === 7 })),
    },
    matrix,
    holds: mockProgress(tick, 1).holds,
    scenarios: [{ scenario: 'S4', name_ko: '짐 들고 제동 (강도 사다리)', phases_seen: ['carry', 'brake_0.5', 'brake_1.5', 'brake_2.5'], score: 69.0, events: 6, gated: 1, min_margin: 0.04, peak_ratio: 0.72 }],
    phases: S4_LADDER.map((r) => ({ scenario: 'S4', phase: r.id, count: 3, score_mean: r.score, gated: r.gated ? 1 : 0 })),
  };
}

export function mockSdsTree(args = {}, tick = 0) {
  const rev = mockRevs(tick).sds;
  if (args?.if_rev === rev) return { unchanged: true, rev };
  const parent = args?.parent || '';
  if (!parent) {
    return {
      rev, parent: '',
      nodes: [
        { id: 'vision', label: 'vision', kind: 'track', scopes: 3, n_max: 24, immature: 1, max_drift_z: 1.4, feedback: ['A', 'B'], engine_revs: ['mock'], has_children: true },
        { id: 'forecast', label: 'forecast', kind: 'track', scopes: 2, n_max: 18, immature: 0, max_drift_z: 0.8, feedback: ['C'], engine_revs: ['mock'], has_children: true },
        { id: 'balance', label: 'balance', kind: 'track', scopes: 4, n_max: 32, immature: 0, max_drift_z: 1.1, feedback: ['F'], engine_revs: ['mock'], has_children: true },
        { id: 'nn_scenarios', label: '시나리오 (N:N)', kind: 'virtual', scopes: 2, n_max: 2, immature: 0, max_drift_z: 0.0, feedback: [], engine_revs: ['mock'], has_children: true },
      ],
    };
  }
  if (parent === 'nn_scenarios') {
    return { rev, parent, nodes: [{ id: 'balance|nn:S4', label: 'nn:S4', kind: 'primary', scopes: 4, has_children: false }] };
  }
  return { rev, parent, nodes: [{ id: `${parent}|log`, label: 'log', kind: 'primary', scopes: 1, n_max: 14, max_drift_z: 1.2, has_children: false }] };
}

export function mockSdsQuery(args = {}, tick = 0) {
  const rev = mockRevs(tick).sds;
  if (args?.if_rev === rev) return { unchanged: true, rev };
  const all = [
    { scope: 'vision|log|embedding', axis: 'score', kind: 'baseline', level: 'secondary', n: 14, need: 20, maturity: 0.7, resolves_to: 'vision|log|', mean: 0.61, sd: 0.07, recent: 0.58, drift_z: -1.4, feedback: ['A'], engine_revs: ['mock'] },
    { scope: 'vision|tree|', axis: 'gate:temporal', kind: 'gate', level: 'primary', n: 18, need: 8, maturity: 2.25, evaluated: 18, blocked_rate: 0.11, sole_rate: 0.11, drift_z: 0.0, feedback: [], engine_revs: ['mock'] },
    { scope: 'balance|nn:S4|brake_2.5', axis: 'cop_shift_frac', kind: 'reference', level: 'secondary', n: 12, need: 10, maturity: 1.2, resolves_to: 'balance|nn:S4|brake_2.5', mean: 0.44, sd: 0.05, recent: 0.46, reference: 0.30, tolerance: 0.12, dev: 1.15, in_band: 0.58, drift_z: 0.9, feedback: [], engine_revs: ['mock'] },
  ];
  const prefix = args?.prefix || '';
  const text = (args?.text || '').toLowerCase();
  const rows = all.filter((r) => (!prefix || prefix === 'nn_scenarios' ? (!prefix || r.scope.startsWith('balance|nn:')) : r.scope.startsWith(prefix)) && (!text || r.scope.toLowerCase().includes(text) || r.axis.toLowerCase().includes(text)));
  return { rev, total: rows.length, merged_scopes: rows.length, cursor: 0, next_cursor: null, rows };
}

export function mockSdsWatch(args = {}, tick = 0) {
  const rev = mockRevs(tick).sds;
  const pts = Array.from({ length: 12 }, (_, i) => ({ tick: i * 10, n: i + 5, mean: 0.62, recent: 0.60 + 0.02 * Math.sin(i), drift_z: Number((1.8 * Math.sin(i / 3)).toFixed(2)) }));
  return { rev, every_ticks: 10, pinned: [{ scope: 'vision|log|embedding', axis: 'score' }], series: [{ scope: 'vision|log|embedding', axis: 'score', points: pts }] };
}

export function mockTsSeries(args = {}, tick = 0) {
  const rev = mockRevs(tick).ts;
  if (args?.if_rev === rev) return { unchanged: true, rev };
  return {
    rev,
    ttm: { attached: false, fp8: null, verified: false, device: null, stats: null },
    series: [
      { id: 'lm:1:score', group: 'landmark', label: 'L1 log', n: 12, last_tick: tick, gate_use: true },
      { id: 'label:log:score', group: 'label', label: 'log · score', n: 16, last_tick: tick, gate_use: false },
      { id: 'sds:vision|log|embedding:score', group: 'sds', label: 'vision|log|embedding · score', n: 12, last_tick: tick, gate_use: false },
    ],
  };
}

export function mockTsView(args = {}, tick = 0) {
  const rev = mockRevs(tick).ts;
  const until = args?.until_tick ?? null;
  const pts = [];
  const band = [];
  for (let i = 0; i < 12; i++) {
    const tk = i * 10;
    if (until != null && tk > until) continue;
    const held = i === 9;
    const val = held ? 0.42 : Number((0.78 + 0.02 * Math.sin(i)).toFixed(3));
    const outcome = i < 6 ? 'abstained' : held ? 'held' : 'accepted';
    pts.push({ tick: tk, value: val, outcome, rescue: null, z: i < 6 ? null : held ? -3.2 : 0.4, method: i < 6 ? '' : 'fused' });
    if (i >= 6) band.push({ tick: tk, expected: 0.78, lo: 0.70, hi: 0.86, z: held ? -3.2 : 0.4, sds_sd: 0.03, observed: val });
  }
  return {
    rev, series: args?.series || 'lm:1:score', gate_use: true, until_tick: until,
    points: pts, band_1step: band,
    forecast: { point: [0.78, 0.77, 0.77, 0.76], p10: [0.72, 0.70, 0.69, 0.68], p90: [0.84, 0.84, 0.85, 0.84], step_ticks: 10, method: 'fused' },
    bands: { model: { p10: [0.72, 0.70, 0.69, 0.68], p90: [0.84, 0.84, 0.85, 0.84] }, cal80: { lo: [0.71, 0.69, 0.68, 0.67], hi: [0.85, 0.85, 0.86, 0.85], k: 1.05 }, gate: { lo: [0.66, 0.64, 0.63, 0.62], hi: [0.90, 0.90, 0.91, 0.90], z_g: 2.33 } },
    weights: [{ method: 'holt', weight: 0.48, rmse: 0.024, history_rmse: 0.028, history_n: 8 }, { method: 'ewma', weight: 0.34, rmse: 0.029, history_rmse: 0.031, history_n: 8 }, { method: 'last', weight: 0.18, rmse: 0.041, history_rmse: 0.044, history_n: 8 }],
    rmse_fused: 0.022, holdout: 4,
    realized: { n: band.length, mae: 0.031, in_band: 0.83, nominal: 0.8, series: band.map((b) => ({ tick: b.tick, err: Number((b.observed - b.expected).toFixed(3)), abs_err: Number(Math.abs(b.observed - b.expected).toFixed(3)), in_band: b.observed >= b.lo && b.observed <= b.hi })) },
    gate_cfg: { min_history: 6, temporal_z: 2.33, enabled: true },
    notes: [],
  };
}
