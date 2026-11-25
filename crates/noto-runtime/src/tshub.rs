//! 3-tier time-series hub (`TsHub`): L0 fixed rings, L1 binary run logs (`keys.json` + `samples.bin`),
//! and L2 span summaries (Phase 3 – Part 1 · D2–D5).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::ts::series::{align_span, q, regularize, ChannelKind, ChannelSpec, SpanRef, TsSample};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const CONTRACT: &str = "noto-ts-v1";
pub const RECORD_BYTES: usize = 16;
pub const RING_CAP: usize = 512;
pub const MAX_RETAINED_RUNS: usize = 20;
pub const ROLLOUT_SAMPLE_CAP: usize = 64;

/// Static registry of built-in channels (Phase 3 – Part 1 · D3 & D5).
pub const CHANNELS: &[ChannelSpec] = &[
    // N:N evaluation channels (display/eval only — never readable by DECISION_FILES).
    ChannelSpec { id: "nn.score", kind: ChannelKind::Score, unit: "", period: 0, hold: 0, cross_phase: false, decision_ok: false, crew: false },
    ChannelSpec { id: "nn.min_margin", kind: ChannelKind::Trough, unit: "m", period: 0, hold: 0, cross_phase: false, decision_ok: false, crew: false },
    ChannelSpec { id: "nn.peak_ratio", kind: ChannelKind::Peak, unit: "ratio", period: 0, hold: 0, cross_phase: false, decision_ok: false, crew: false },
    ChannelSpec { id: "nn.sustained_effort", kind: ChannelKind::Gauge, unit: "ratio", period: 0, hold: 0, cross_phase: false, decision_ok: false, crew: false },
    // Whole-body balance channels (2-tick cadence).
    ChannelSpec { id: "bal.margin", kind: ChannelKind::Gauge, unit: "m", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "bal.margin_x", kind: ChannelKind::Gauge, unit: "m", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "bal.margin_y", kind: ChannelKind::Gauge, unit: "m", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "bal.head_pitch", kind: ChannelKind::Gauge, unit: "deg", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "bal.trunk_lean", kind: ChannelKind::Gauge, unit: "deg", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "bal.effort_sum", kind: ChannelKind::Gauge, unit: "ratio", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: false },
    // Per-joint mechanical & thermal channels (10-tick cadence).
    ChannelSpec { id: "joint.effort", kind: ChannelKind::Gauge, unit: "ratio", period: 10, hold: 30, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "joint.peak_ratio", kind: ChannelKind::Peak, unit: "ratio", period: 10, hold: 30, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "joint.heat", kind: ChannelKind::Gauge, unit: "ratio", period: 10, hold: 30, cross_phase: true, decision_ok: true, crew: false },
    // Cumulative wear counter (50-tick cadence, monotone across phases).
    ChannelSpec { id: "wear.life_used", kind: ChannelKind::Counter, unit: "life", period: 50, hold: 150, cross_phase: true, decision_ok: true, crew: false },
    // Cargo ride dynamics (2-tick while carrying & moving, crew-enabled).
    ChannelSpec { id: "cargo.use_ratio", kind: ChannelKind::Gauge, unit: "ratio", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: true },
    ChannelSpec { id: "cargo.shift_m", kind: ChannelKind::Gauge, unit: "m", period: 2, hold: 6, cross_phase: false, decision_ok: true, crew: true },
    // Odometry belief error vs truth (5-tick, always-on, eval-only).
    ChannelSpec { id: "nav.belief_err_m", kind: ChannelKind::Gauge, unit: "m", period: 5, hold: 15, cross_phase: true, decision_ok: false, crew: true },
    // Sub-tick dynamics rollout (20 ms, capped per rollout).
    ChannelSpec { id: "roll.margin", kind: ChannelKind::Gauge, unit: "m", period: 0, hold: 0, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "roll.ratio_max", kind: ChannelKind::Gauge, unit: "ratio", period: 0, hold: 0, cross_phase: false, decision_ok: true, crew: false },
    // Drive wheel/track slip (10-tick while driving).
    ChannelSpec { id: "drive.slip", kind: ChannelKind::Gauge, unit: "ratio", period: 10, hold: 30, cross_phase: false, decision_ok: true, crew: false },
    // Event-indexed vision & decision channels.
    ChannelSpec { id: "vision.score", kind: ChannelKind::Score, unit: "score", period: 0, hold: 0, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "decide.fix_x", kind: ChannelKind::Score, unit: "m", period: 0, hold: 0, cross_phase: false, decision_ok: true, crew: false },
    ChannelSpec { id: "decide.fix_z", kind: ChannelKind::Score, unit: "m", period: 0, hold: 0, cross_phase: false, decision_ok: true, crew: false },
    // External environment input (e.g. moving floor S5-A).
    ChannelSpec { id: "env.floor_acc", kind: ChannelKind::Gauge, unit: "m/s2", period: 1, hold: 3, cross_phase: false, decision_ok: true, crew: false },
];

const EVENT_SPEC: ChannelSpec = ChannelSpec {
    id: "ev.*",
    kind: ChannelKind::Event,
    unit: "count",
    period: 0,
    hold: 0,
    cross_phase: true,
    decision_ok: true,
    crew: true,
};

const FALLBACK_SPEC: ChannelSpec = ChannelSpec {
    id: "custom",
    kind: ChannelKind::Gauge,
    unit: "",
    period: 0,
    hold: 0,
    cross_phase: false,
    decision_ok: true,
    crew: false,
};

pub fn channel_spec(channel: &str) -> ChannelSpec {
    if let Some(sp) = CHANNELS.iter().find(|c| c.id == channel) {
        return *sp;
    }
    if channel.starts_with("ev.") {
        return EVENT_SPEC;
    }
    FALLBACK_SPEC
}

/// Escape `:` and `%` inside subject strings so `ts:<worker>:<channel>:<subject>` is unambiguous.
pub fn escape_subject(s: &str) -> String {
    s.replace('%', "%25").replace(':', "%3A")
}

pub fn unescape_subject(s: &str) -> String {
    s.replace("%3A", ":").replace("%25", "%")
}

/// Canonical series key string: `ts:<worker>:<channel>:<escaped_subject>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SeriesKey {
    pub worker: u32,
    pub channel: String,
    pub subject: String,
}

impl SeriesKey {
    pub fn new(worker: u32, channel: impl Into<String>, subject: impl Into<String>) -> Self {
        Self { worker, channel: channel.into(), subject: subject.into() }
    }

    pub fn encode(&self) -> String {
        format!("ts:{}:{}:{}", self.worker, self.channel, escape_subject(&self.subject))
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let mut it = raw.splitn(4, ':');
        if it.next()? != "ts" {
            return None;
        }
        let worker = it.next()?.parse::<u32>().ok()?;
        let channel = it.next()?.to_string();
        let subject = unescape_subject(it.next().unwrap_or(""));
        Some(Self { worker, channel, subject })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerMeta {
    pub form: String,
    pub tier: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyMeta {
    pub id: u32,
    pub key: String,
    pub worker: u32,
    pub channel: String,
    pub subject: String,
    pub kind: ChannelKind,
    pub unit: String,
    pub period: u32,
    pub hold: u32,
    pub cross_phase: bool,
    pub decision_ok: bool,
}

/// L2 span-level summary computed over samples inside `[start, end)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpanSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub slope: f64,
    pub delta_jump: Option<f64>,
    pub counter_inc: Option<f64>,
    pub n: usize,
}

/// Row shape suitable for mirroring L1 time-series samples to Lance (`ts_samples`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TsSampleRow {
    pub run_id: String,
    pub worker: u32,
    pub channel: String,
    pub subject: String,
    pub dtick: u32,
    pub sub_ms: u16,
    pub value: f64,
    pub q: u16,
}

struct RingEntry {
    meta: KeyMeta,
    ring: VecDeque<TsSample>,
    /// Running count when merging duplicate Score samples on the exact same tick.
    last_dup_count: u32,
    /// Pending quality bits to attach to the next observed sample on this key.
    pending_q: u16,
}

pub struct TsHub {
    dir: Option<PathBuf>,
    run_id: String,
    start_tick: u64,
    tick_s: f64,
    workers: BTreeMap<u32, WorkerMeta>,
    key_to_id: HashMap<SeriesKey, u32>,
    str_to_id: HashMap<String, u32>,
    entries: Vec<RingEntry>,
    l1_pending: Vec<(u32, TsSample)>,
    l1_dirty_keys: bool,
    phase_breaks: Vec<u32>,
    /// Per-joint max peak_ratio tracker across the 10-tick window so `joint.peak_ratio` captures tick maxima.
    joint_peak_win: HashMap<String, f32>,
    /// Last motion start_s already emitted into `roll.*` so each rollout is emitted once.
    last_rollout_start_s: f64,
    late: u64,
    rev: u64,
    ring_cap: usize,
}

impl TsHub {
    pub fn new(dir: Option<PathBuf>, run_id: &str, start_tick: u64, tick_s: f64) -> Self {
        Self {
            dir,
            run_id: run_id.to_string(),
            start_tick,
            tick_s,
            workers: BTreeMap::new(),
            key_to_id: HashMap::new(),
            str_to_id: HashMap::new(),
            entries: Vec::new(),
            l1_pending: Vec::new(),
            l1_dirty_keys: false,
            phase_breaks: Vec::new(),
            joint_peak_win: HashMap::new(),
            last_rollout_start_s: -1.0,
            late: 0,
            rev: 0,
            ring_cap: RING_CAP,
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn start_tick(&self) -> u64 {
        self.start_tick
    }

    pub fn rev(&self) -> u64 {
        self.rev
    }

    pub fn late_count(&self) -> u64 {
        self.late
    }

    pub fn channel_count(&self) -> usize {
        self.entries.len()
    }

    /// Exact L0 ring memory footprint in bytes (16 bytes per stored sample slot, matching D2 budget).
    pub fn memory_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.ring.len() * RECORD_BYTES).sum()
    }

    pub fn register_worker(&mut self, worker: u32, form: &str, tier: &str) {
        let changed = self.workers.get(&worker).map_or(true, |w| w.form != form || w.tier != tier);
        if changed {
            self.workers.insert(worker, WorkerMeta { form: form.to_string(), tier: tier.to_string() });
            self.l1_dirty_keys = true;
        }
    }

    /// Switch active run id (e.g. on test pass start), flushing prior L1 data and pruning old run dirs.
    pub fn set_run(&mut self, run_id: &str, start_tick: u64) {
        if self.run_id == run_id && start_tick >= self.start_tick {
            return;
        }
        self.flush_l1();
        self.run_id = run_id.to_string();
        self.start_tick = start_tick;
        self.entries.clear();
        self.key_to_id.clear();
        self.str_to_id.clear();
        self.l1_pending.clear();
        self.l1_dirty_keys = false;
        self.phase_breaks.clear();
        self.joint_peak_win.clear();
        self.last_rollout_start_s = -1.0;
        self.late = 0;
        self.rev += 1;
        self.prune_old_runs();
    }

    /// Mark a world reset: if ticks rewound, restart the L0 epoch cleanly so `dtick` stays monotone;
    /// always flag the next sample on every active channel with `q::RESET`.
    pub fn mark_reset(&mut self, world_tick: u64) {
        self.flush_l1();
        let rewound = world_tick < self.start_tick
            || self.entries.iter().any(|e| {
                e.ring.back().is_some_and(|s| self.start_tick.saturating_add(s.dtick as u64) > world_tick)
            });
        if rewound {
            self.start_tick = world_tick;
            for e in &mut self.entries {
                e.ring.clear();
                e.last_dup_count = 0;
                e.pending_q |= q::RESET;
            }
            self.phase_breaks.clear();
            self.l1_pending.clear();
        } else {
            for e in &mut self.entries {
                e.pending_q |= q::RESET;
            }
        }
        self.joint_peak_win.clear();
        self.last_rollout_start_s = -1.0;
        self.rev += 1;
    }

    /// Mark a phase or environment boundary tick so non-`cross_phase` channels do not interpolate across it.
    pub fn mark_phase_break(&mut self, world_tick: u64) {
        let dtick = world_tick.saturating_sub(self.start_tick) as u32;
        if self.phase_breaks.last().copied() != Some(dtick) {
            self.phase_breaks.push(dtick);
            if self.phase_breaks.len() > 2048 {
                self.phase_breaks.remove(0);
            }
        }
        for e in &mut self.entries {
            if !e.meta.cross_phase {
                e.pending_q |= q::PHASE_BREAK;
            }
        }
    }

    /// Track per-tick joint peak ratios so 10-tick `joint.peak_ratio` emission records the exact window max.
    pub fn track_joint_peak(&mut self, joint: &str, peak_ratio: f32) {
        let cur = self.joint_peak_win.entry(joint.to_string()).or_insert(0.0);
        if peak_ratio > *cur {
            *cur = peak_ratio;
        }
    }

    pub fn take_joint_peak(&mut self, joint: &str, fallback: f32) -> f32 {
        self.joint_peak_win.remove(joint).map_or(fallback, |v| v.max(fallback))
    }

    pub fn should_emit_rollout(&mut self, start_s: f64) -> bool {
        if (start_s - self.last_rollout_start_s).abs() > 1e-6 {
            self.last_rollout_start_s = start_s;
            true
        } else {
            false
        }
    }

    fn intern(&mut self, worker: u32, channel: &str, subject: &str) -> u32 {
        let sk = SeriesKey::new(worker, channel, subject);
        if let Some(&id) = self.key_to_id.get(&sk) {
            return id;
        }
        let id = self.entries.len() as u32;
        let encoded = sk.encode();
        let spec = channel_spec(channel);
        let meta = KeyMeta {
            id,
            key: encoded.clone(),
            worker,
            channel: channel.to_string(),
            subject: subject.to_string(),
            kind: spec.kind,
            unit: spec.unit.to_string(),
            period: spec.period,
            hold: spec.hold,
            cross_phase: spec.cross_phase,
            decision_ok: spec.decision_ok,
        };
        self.entries.push(RingEntry {
            meta,
            ring: VecDeque::with_capacity(self.ring_cap.min(64)),
            last_dup_count: 0,
            pending_q: 0,
        });
        self.key_to_id.insert(sk, id);
        self.str_to_id.insert(encoded, id);
        self.l1_dirty_keys = true;
        id
    }

    pub fn record(&mut self, worker: u32, channel: &str, subject: &str, world_tick: u64, value: f32) {
        self.record_sub(worker, channel, subject, world_tick, 0, value, 0);
    }

    pub fn record_sub(
        &mut self,
        worker: u32,
        channel: &str,
        subject: &str,
        world_tick: u64,
        sub_ms: u16,
        value: f32,
        extra_q: u16,
    ) {
        if !value.is_finite() {
            return;
        }
        if world_tick < self.start_tick {
            self.late += 1;
            return;
        }
        let dtick = (world_tick - self.start_tick).min(u32::MAX as u64) as u32;
        let id = self.intern(worker, channel, subject);
        let entry = &mut self.entries[id as usize];
        let q_bits = q::OBSERVED | extra_q | entry.pending_q;

        if let Some(last) = entry.ring.back_mut() {
            if dtick < last.dtick || (dtick == last.dtick && sub_ms < last.sub_ms) {
                self.late += 1;
                return;
            }
            if dtick == last.dtick && sub_ms == last.sub_ms {
                entry.pending_q = 0;
                match entry.meta.kind {
                    ChannelKind::Gauge => last.value = value,
                    ChannelKind::Peak | ChannelKind::Counter => last.value = last.value.max(value),
                    ChannelKind::Trough => last.value = last.value.min(value),
                    ChannelKind::Score => {
                        let c = entry.last_dup_count.max(1);
                        last.value = ((last.value as f64 * c as f64 + value as f64) / (c + 1) as f64) as f32;
                        entry.last_dup_count = c + 1;
                    }
                    ChannelKind::Event => last.value += value,
                }
                last.q |= q_bits;
                let updated = *last;
                if let Some((pid, ps)) = self.l1_pending.last_mut() {
                    if *pid == id && ps.dtick == dtick && ps.sub_ms == sub_ms {
                        *ps = updated;
                    } else {
                        self.l1_pending.push((id, updated));
                    }
                } else {
                    self.l1_pending.push((id, updated));
                }
                self.rev += 1;
                return;
            }
        }

        entry.pending_q = 0;
        let sample = TsSample { dtick, value, sub_ms, q: q_bits };
        entry.ring.push_back(sample);
        entry.last_dup_count = 1;
        while entry.ring.len() > self.ring_cap {
            entry.ring.pop_front();
        }
        self.l1_pending.push((id, sample));
        self.rev += 1;
    }

    pub fn samples_vec(&self, worker: u32, channel: &str, subject: &str) -> Vec<TsSample> {
        let sk = SeriesKey::new(worker, channel, subject);
        self.key_to_id
            .get(&sk)
            .map(|&id| self.entries[id as usize].ring.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn samples_by_key(&self, key_str: &str) -> Option<(&KeyMeta, Vec<TsSample>)> {
        let &id = self.str_to_id.get(key_str)?;
        let e = &self.entries[id as usize];
        Some((&e.meta, e.ring.iter().copied().collect()))
    }

    /// Return all stored samples for any key matching `(worker, channel)` regardless of subject.
    pub fn matching_keys(&self, worker: Option<u32>, channel: Option<&str>) -> Vec<&KeyMeta> {
        self.entries
            .iter()
            .filter(|e| {
                worker.map_or(true, |w| e.meta.worker == w)
                    && channel.map_or(true, |c| e.meta.channel == c)
            })
            .map(|e| &e.meta)
            .collect()
    }

    /// Compute L2 span summary over samples in `[start_tick, end_tick)`.
    pub fn span_summary(&self, key_str: &str, start_tick: u64, end_tick: u64) -> Option<SpanSummary> {
        let (meta, all) = self.samples_by_key(key_str)?;
        let s0 = start_tick.saturating_sub(self.start_tick) as u32;
        let s1 = end_tick.saturating_sub(self.start_tick) as u32;
        let slice: Vec<TsSample> = all.iter().copied().filter(|s| s.dtick >= s0 && s.dtick < s1).collect();
        if slice.is_empty() {
            return None;
        }
        let n = slice.len();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut t_sum = 0.0;
        for s in &slice {
            let v = s.value as f64;
            min = min.min(v);
            max = max.max(v);
            sum += v;
            t_sum += (s.dtick - s0) as f64;
        }
        let mean = sum / n as f64;
        let t_mean = t_sum / n as f64;
        let (mut num, mut den) = (0.0, 0.0);
        for s in &slice {
            let dt = (s.dtick - s0) as f64 - t_mean;
            num += dt * (s.value as f64 - mean);
            den += dt * dt;
        }
        let slope = if den > 1e-9 { num / den } else { 0.0 };
        let counter_inc = if meta.kind == ChannelKind::Counter {
            Some(slice.last().unwrap().value as f64 - slice.first().unwrap().value as f64)
        } else {
            None
        };
        // Boundary jump Δ = mean(after W) - mean(before W) around span start s0.
        let p = meta.period.max(1);
        let span_len = s1.saturating_sub(s0).max(1);
        let w = (span_len / 2).min(10 * p).max(1);
        let before: Vec<f64> = all
            .iter()
            .filter(|s| s.dtick < s0 && s0 - s.dtick <= w)
            .map(|s| s.value as f64)
            .collect();
        let after: Vec<f64> = slice
            .iter()
            .filter(|s| s.dtick - s0 <= w)
            .map(|s| s.value as f64)
            .collect();
        let delta_jump = if !before.is_empty() && !after.is_empty() {
            let bm = before.iter().sum::<f64>() / before.len() as f64;
            let am = after.iter().sum::<f64>() / after.len() as f64;
            Some(am - bm)
        } else {
            None
        };
        Some(SpanSummary { min, max, mean, slope, delta_jump, counter_inc, n })
    }

    pub fn keys_json(&self) -> Value {
        let workers: BTreeMap<String, &WorkerMeta> = self.workers.iter().map(|(k, v)| (k.to_string(), v)).collect();
        let keys: Vec<Value> = self
            .entries
            .iter()
            .map(|e| {
                let last = e.ring.back();
                json!({
                    "id": e.meta.id,
                    "key": e.meta.key,
                    "worker": e.meta.worker,
                    "channel": e.meta.channel,
                    "subject": e.meta.subject,
                    "kind": e.meta.kind,
                    "unit": e.meta.unit,
                    "period": e.meta.period,
                    "hold": e.meta.hold,
                    "cross_phase": e.meta.cross_phase,
                    "decision_ok": e.meta.decision_ok,
                    "count": e.ring.len(),
                    "last_dtick": last.map(|s| s.dtick),
                    "last_value": last.map(|s| s.value),
                })
            })
            .collect();
        json!({
            "contract": CONTRACT,
            "run_id": self.run_id,
            "start_tick": self.start_tick,
            "tick_s": self.tick_s,
            "record_bytes": RECORD_BYTES,
            "workers": workers,
            "keys": keys,
            "late": self.late,
            "memory_bytes": self.memory_bytes(),
            "rev": self.rev,
        })
    }

    pub fn series_json(
        &self,
        key_str: &str,
        t0: Option<u32>,
        period: u32,
        n: Option<usize>,
        spans: &[SpanRef],
    ) -> Result<Value, String> {
        let (meta, raw) = self
            .samples_by_key(key_str)
            .ok_or_else(|| format!("시계열 키를 찾을 수 없습니다: {key_str}"))?;
        let aligned: Vec<Value> = spans
            .iter()
            .map(|sp| {
                let v = align_span(&raw, sp, 20);
                let summ = self.span_summary(
                    key_str,
                    self.start_tick + sp.start as u64,
                    self.start_tick + sp.end.unwrap_or_else(|| raw.last().map_or(sp.start + 1, |s| s.dtick + 1)) as u64,
                );
                json!({ "span_id": sp.span_id, "start": sp.start, "end": sp.end, "view": v, "summary": summ })
            })
            .collect();
        if period == 0 {
            let start = t0.unwrap_or_else(|| raw.first().map_or(0, |s| s.dtick));
            let filtered: Vec<TsSample> = raw.into_iter().filter(|s| s.dtick >= start).collect();
            let capped: Vec<TsSample> = match n {
                Some(lim) if filtered.len() > lim => filtered[filtered.len() - lim..].to_vec(),
                _ => filtered,
            };
            let dticks: Vec<u32> = capped.iter().map(|s| s.dtick).collect();
            let values: Vec<f64> = capped.iter().map(|s| s.value as f64).collect();
            let qs: Vec<u16> = capped.iter().map(|s| s.q).collect();
            let samples: Vec<Value> = capped
                .iter()
                .map(|s| json!({"dtick": s.dtick, "value": s.value, "sub_ms": s.sub_ms, "q": s.q}))
                .collect();
            return Ok(json!({
                "key": meta.key,
                "channel": meta.channel,
                "subject": meta.subject,
                "kind": meta.kind,
                "unit": meta.unit,
                "start_tick": self.start_tick,
                "tick_s": self.tick_s,
                "t0": start,
                "period": 0,
                "samples": samples,
                "dticks": dticks,
                "values": values,
                "q": qs,
                "mask": vec![true; values.len()],
                "segments": if values.is_empty() { Vec::<(usize, usize)>::new() } else { vec![(0, values.len())] },
                "imputed_share": 0.0,
                "spans": aligned,
            }));
        }
        let first_t = raw.first().map_or(0, |s| s.dtick);
        let last_t = raw.last().map_or(first_t, |s| s.dtick);
        let grid_t0 = t0.unwrap_or(first_t);
        let default_n = (((last_t.saturating_sub(grid_t0)) / period) as usize + 1).clamp(1, self.ring_cap);
        let grid_n = n.unwrap_or(default_n).clamp(1, 4096);
        let hold = if meta.hold > 0 { meta.hold } else { 3 * period };
        let breaks: &[u32] = if meta.cross_phase { &[] } else { &self.phase_breaks };
        let reg = regularize(&raw, meta.kind, grid_t0, period, grid_n, hold, breaks);
        // Serialize NaN as JSON null for browser safety.
        let values_json: Vec<Value> = reg
            .values
            .iter()
            .map(|&v| if v.is_finite() { json!(v) } else { Value::Null })
            .collect();
        Ok(json!({
            "key": meta.key,
            "channel": meta.channel,
            "subject": meta.subject,
            "kind": meta.kind,
            "unit": meta.unit,
            "start_tick": self.start_tick,
            "tick_s": self.tick_s,
            "t0": reg.t0,
            "period": reg.period,
            "values": values_json,
            "mask": reg.mask,
            "segments": reg.segments,
            "imputed_share": reg.imputed_share,
            "spans": aligned,
        }))
    }

    fn run_dir(&self) -> Option<PathBuf> {
        let base = self.dir.as_ref()?;
        let safe: String = self
            .run_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        Some(base.join("ts").join(if safe.is_empty() { "live".into() } else { safe }))
    }

    /// Flush buffered L1 samples (`samples.bin`) and updated key dictionary (`keys.json`).
    pub fn flush_l1(&mut self) {
        let Some(rdir) = self.run_dir() else {
            self.l1_pending.clear();
            self.l1_dirty_keys = false;
            return;
        };
        if self.l1_pending.is_empty() && !self.l1_dirty_keys {
            return;
        }
        if fs::create_dir_all(&rdir).is_err() {
            return;
        }
        if self.l1_dirty_keys {
            let workers: BTreeMap<String, &WorkerMeta> = self.workers.iter().map(|(k, v)| (k.to_string(), v)).collect();
            let keys: Vec<Value> = self
                .entries
                .iter()
                .map(|e| {
                    json!({
                        "id": e.meta.id,
                        "worker": e.meta.worker,
                        "channel": e.meta.channel,
                        "subject": e.meta.subject,
                        "kind": e.meta.kind,
                        "period": e.meta.period,
                        "hold": e.meta.hold,
                    })
                })
                .collect();
            let doc = json!({
                "contract": CONTRACT,
                "run_id": self.run_id,
                "start_tick": self.start_tick,
                "tick_s": self.tick_s,
                "record_bytes": RECORD_BYTES,
                "workers": workers,
                "keys": keys,
            });
            let tmp = rdir.join("keys.json.tmp");
            let dst = rdir.join("keys.json");
            if let Ok(bytes) = serde_json::to_vec_pretty(&doc) {
                if fs::write(&tmp, bytes).is_ok() {
                    let _ = fs::rename(&tmp, &dst);
                    self.l1_dirty_keys = false;
                }
            }
        }
        if !self.l1_pending.is_empty() {
            let bin = rdir.join("samples.bin");
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&bin) {
                let mut buf = Vec::with_capacity(self.l1_pending.len() * RECORD_BYTES);
                for (id, s) in self.l1_pending.drain(..) {
                    buf.extend_from_slice(&id.to_le_bytes());
                    buf.extend_from_slice(&s.dtick.to_le_bytes());
                    buf.extend_from_slice(&s.value.to_le_bytes());
                    buf.extend_from_slice(&s.sub_ms.to_le_bytes());
                    buf.extend_from_slice(&s.q.to_le_bytes());
                }
                let _ = f.write_all(&buf);
            }
        }
    }

    /// Read back an L1 run directory, verifying `contract == "noto-ts-v1"`.
    /// On contract mismatch, renames `keys.json` aside and returns an error.
    pub fn read_l1(run_dir: &Path) -> Result<(Value, Vec<(u32, TsSample)>), String> {
        let keys_path = run_dir.join("keys.json");
        let raw = fs::read_to_string(&keys_path).map_err(|e| format!("keys.json 읽기 실패: {e}"))?;
        let meta: Value = serde_json::from_str(&raw).map_err(|e| format!("keys.json 파싱 실패: {e}"))?;
        let contract = meta.get("contract").and_then(Value::as_str).unwrap_or("");
        if contract != CONTRACT {
            let archived = run_dir.join(format!("keys.incompatible.{}.json", if contract.is_empty() { "unknown" } else { contract }));
            let _ = fs::rename(&keys_path, &archived);
            return Err(format!("지원하지 않는 ts 계약 `{contract}` (기대: `{CONTRACT}`)"));
        }
        let bin_path = run_dir.join("samples.bin");
        let mut samples = Vec::new();
        if let Ok(mut f) = fs::File::open(&bin_path) {
            let mut bytes = Vec::new();
            f.read_to_end(&mut bytes).map_err(|e| format!("samples.bin 읽기 실패: {e}"))?;
            for chunk in bytes.chunks_exact(RECORD_BYTES) {
                let id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let dtick = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let value = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let sub_ms = u16::from_le_bytes(chunk[12..14].try_into().unwrap());
                let q_bits = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
                samples.push((id, TsSample { dtick, value, sub_ms, q: q_bits }));
            }
        }
        Ok((meta, samples))
    }

    /// Export all currently buffered L0 samples as Lance rows (`ts_samples`).
    pub fn export_lance_rows(&self) -> Vec<TsSampleRow> {
        let mut rows = Vec::new();
        for e in &self.entries {
            for s in &e.ring {
                rows.push(TsSampleRow {
                    run_id: self.run_id.clone(),
                    worker: e.meta.worker,
                    channel: e.meta.channel.clone(),
                    subject: e.meta.subject.clone(),
                    dtick: s.dtick,
                    sub_ms: s.sub_ms,
                    value: s.value as f64,
                    q: s.q,
                });
            }
        }
        rows
    }

    fn prune_old_runs(&self) {
        let Some(base) = self.dir.as_ref() else { return };
        let ts_dir = base.join("ts");
        let Ok(rd) = fs::read_dir(&ts_dir) else { return };
        let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.is_dir() {
                    let m = e.metadata().ok()?.modified().ok().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    Some((m, p))
                } else {
                    None
                }
            })
            .collect();
        if dirs.len() <= MAX_RETAINED_RUNS {
            return;
        }
        dirs.sort_by_key(|(m, _)| *m);
        let drop_n = dirs.len() - MAX_RETAINED_RUNS;
        for (_, p) in dirs.into_iter().take(drop_n) {
            let _ = fs::remove_dir_all(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_escaping_round_trips_colons() {
        let k = SeriesKey::new(1, "nn.score", "s4:brake_1.5");
        let enc = k.encode();
        assert_eq!(enc, "ts:1:nn.score:s4%3Abrake_1.5");
        assert_eq!(SeriesKey::parse(&enc), Some(k));
    }

    #[test]
    fn tshub_memory_stays_under_ring_budget() {
        let mut hub = TsHub::new(None, "budget", 0, 0.1);
        for t in 0..30_000u64 {
            if t % 2 == 0 {
                hub.record(1, "bal.margin", "", t, 0.12);
                hub.record(1, "bal.effort_sum", "", t, 1.5);
            }
            if t % 10 == 0 {
                for j in ["ankle", "knee", "waist", "shoulder"] {
                    hub.record(1, "joint.effort", j, t, 0.4);
                    hub.record(1, "joint.heat", j, t, 0.2);
                }
            }
            if t % 50 == 0 {
                hub.record(1, "wear.life_used", "knee", t, t as f32 * 1e-5);
            }
        }
        let budget = hub.channel_count() * RING_CAP * RECORD_BYTES;
        assert!(hub.memory_bytes() <= budget, "{} > {budget}", hub.memory_bytes());
        assert_eq!(hub.late_count(), 0);
    }

    #[test]
    fn l1_binary_roundtrip_and_contract_guard() {
        let dir = std::env::temp_dir().join(format!("noto_ts_l1_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut hub = TsHub::new(Some(dir.clone()), "r_test", 100, 0.1);
        hub.register_worker(1, "pro:humanfoot:f5", "primary");
        hub.record(1, "bal.margin", "", 102, 0.085);
        hub.record(1, "nn.score", "s4:brake_1.5", 102, 88.5);
        hub.flush_l1();
        let rdir = dir.join("ts").join("r_test");
        let (meta, samples) = TsHub::read_l1(&rdir).expect("read L1");
        assert_eq!(meta["contract"], CONTRACT);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].1.dtick, 2);
        assert!((samples[0].1.value - 0.085).abs() < 1e-5);
        let _ = fs::remove_dir_all(&dir);
    }
}
