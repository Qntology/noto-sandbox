// 중앙 폴링·푸시 스케줄러 (Phase 5 – Part 2 · D5).
// 타이머는 여기 하나만 두고, 작업마다 보임 여부·판 번호(rev)·주기 조건을 확인해 불필요한 조회를 건너뛴다.
import { call } from './api.js';
import { bus, state } from './state.js';

const jobs = new Map();
let latestRevs = { tick: 0, sds: 0, ts: 0, test: 0, nn: 0, log_seq: 0 };
let transport = 'poll'; // 'tauri' | 'sse' | 'poll'
let masterTimer = null;
let lastRevPoll = 0;
let sse = null;

/**
 * 작업을 등록한다.
 * @param {string} id
 * @param {{ every?: number, rev?: string, visible?: () => boolean, run: (revs: object) => any }} job
 */
export function register(id, job) {
  jobs.set(id, { last: 0, seen: null, running: false, every: 2000, visible: () => true, ...job });
}

export function unregister(id) {
  jobs.delete(id);
}

export function getRevs() {
  return latestRevs;
}

export function getTransport() {
  return transport;
}

function runJob(j, now) {
  if (j.running) return;
  j.running = true;
  j.last = now;
  Promise.resolve(j.run(latestRevs))
    .catch(() => { /* 개별 패널 오류는 조용히 */ })
    .finally(() => { j.running = false; });
}

/** sim_step 응답 · SSE · Tauri 이벤트에서 받은 판 번호를 전파한다. */
export function onRevs(revs) {
  if (!revs || typeof revs !== 'object') return;
  const changed = Object.keys(revs).some((k) => revs[k] !== latestRevs[k]);
  latestRevs = { ...latestRevs, ...revs };
  if (changed) bus.emit('revs', latestRevs);
  if (document.hidden) return;
  const now = Date.now();
  for (const j of jobs.values()) {
    if (!j.rev) continue;
    if (!j.visible()) continue;
    const cur = latestRevs[j.rev];
    if (cur == null || cur === j.seen) continue;
    if (j.minGap && now - j.last < j.minGap) continue;
    j.seen = cur;
    runJob(j, now);
  }
}

/** 특정 작업의 판 번호 기억을 초기화해 즉시 한 번 실행되게 한다. */
export function invalidate(id) {
  const j = jobs.get(id);
  if (!j) return;
  j.seen = null;
  if (!document.hidden && j.visible()) {
    j.seen = j.rev ? latestRevs[j.rev] : null;
    runJob(j, Date.now());
  }
}

function connectPush() {
  const listen = window.__TAURI__?.event?.listen;
  if (typeof listen === 'function') {
    transport = 'tauri';
    listen('rt-revs', (ev) => onRevs(ev?.payload)).catch(() => { transport = 'poll'; });
    bus.emit('poller-transport', transport);
    return;
  }
  if (window.__notoEventsUrl && typeof EventSource !== 'undefined') {
    try {
      sse = new EventSource(window.__notoEventsUrl);
      sse.addEventListener('rt-revs', (ev) => {
        try { onRevs(JSON.parse(ev.data)); } catch { /* 무시 */ }
      });
      sse.onopen = () => { transport = 'sse'; bus.emit('poller-transport', transport); };
      sse.onerror = () => { transport = 'poll'; bus.emit('poller-transport', transport); };
      return;
    } catch { /* EventSource 실패 시 폴링 폴백 */ }
  }
  transport = 'poll';
  bus.emit('poller-transport', transport);
}

export function initPoller() {
  if (masterTimer) return;
  connectPush();
  masterTimer = setInterval(() => {
    if (document.hidden) return;
    const now = Date.now();
    // 정지 중이며 푸시 연결이 없을 때는 2초 주기로 rt_revs 만 가볍게 확인한다.
    if (!state.playing && transport === 'poll' && now - lastRevPoll >= 2000) {
      lastRevPoll = now;
      call('rt_revs', {}, { quiet: true }).then((r) => onRevs(r)).catch(() => {});
    }
    for (const j of jobs.values()) {
      if (!j.visible()) continue;
      if (j.rev) {
        const cur = latestRevs[j.rev];
        if (cur != null && cur !== j.seen && (!j.minGap || now - j.last >= j.minGap)) {
          j.seen = cur;
          runJob(j, now);
        }
      } else if (now - j.last >= j.every) {
        runJob(j, now);
      }
    }
  }, 100);
}
