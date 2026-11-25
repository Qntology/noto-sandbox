//! Live N:N progression tracker: run/worker/form partitions, monotonic clock,
//! hysteresis phase spans, ordered scenario instances, and world-event marks
//! (Phase 1 – Part 1 · D1–D5).

use std::collections::{BTreeMap, VecDeque};

use noto_core::balance::{Body, Posture, Source};
use noto_core::nn::{self, causes_of, matching_phases, MatchClass, PhaseResult, PhaseSpec, ScenarioSpec};
use noto_core::{BaseKind, BodyPlan, FingerCount};
use serde::Serialize;
use serde_json::{json, Value};
use world_sim::{BodySense, TimedEvent, WorldEvent};

const MAX_PARTITIONS: usize = 20;
const MAX_CLOSED_SPANS: usize = 2_000;
const MAX_CLOSED_INSTANCES: usize = 512;

/// Unique key partitioning live N:N state by run, pass, worker, and body form.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RunKey {
    pub run_id: String,
    pub pass: String,
    pub worker: u32,
    pub form: String,
}

/// Global monotonic sequence paired with the world's physical tick and time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Clock {
    pub seq: u64,
    pub tick: u64,
    pub time_s: f64,
}

/// Reason a [`PhaseSpan`] closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CloseWhy {
    Left,
    Superseded,
    ActionDone,
    Event(&'static str),
    Reset,
    RunEnd,
}

impl CloseWhy {
    pub fn as_str(self) -> String {
        match self {
            CloseWhy::Left => "left".into(),
            CloseWhy::Superseded => "superseded".into(),
            CloseWhy::ActionDone => "action_done".into(),
            CloseWhy::Event(e) => format!("event:{e}"),
            CloseWhy::Reset => "reset".into(),
            CloseWhy::RunEnd => "run_end".into(),
        }
    }
}

/// World event mark attached to an open span (or worker partition).
#[derive(Clone, Debug, Serialize)]
pub struct Mark {
    pub seq: u64,
    pub kind: &'static str,
    pub value: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Continuous interval spent in one N:N phase.
#[derive(Clone, Debug, Serialize)]
pub struct PhaseSpan {
    pub id: u64,
    pub instance: u64,
    pub scenario: &'static str,
    pub phase: String,
    pub start: Clock,
    pub end: Option<Clock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enter_from: Option<String>,
    pub interruptions: u16,
    pub held: bool,
    pub score_first: Option<f32>,
    pub score_min: Option<f32>,
    pub gated: bool,
    pub static_margin_min: f32,
    pub peak_ratio_max: f32,
    pub plan_gap_max: f32,
    pub use_ratio_max: f32,
    pub marks: Vec<Mark>,
    pub close: Option<CloseWhy>,
}

impl PhaseSpan {
    pub fn dwell_s(&self, now_s: f64) -> f64 {
        let end_s = self.end.map_or(now_s, |c| c.time_s);
        (end_s - self.start.time_s).max(0.0)
    }
}

/// One ordered occurrence of a scenario's phases.
#[derive(Clone, Debug, Serialize)]
pub struct Instance {
    pub id: u64,
    pub scenario: &'static str,
    pub opened: Clock,
    pub closed: Option<Clock>,
    pub last_index: usize,
    pub segments: Vec<u64>,
    pub phases_seen: Vec<String>,
    pub complete: bool,
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_why: Option<String>,
}

/// Hysteresis thresholds for live phase span detection (Phase 1 – Part 1 · D2).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Hysteresis {
    pub enter_ticks: u8,
    pub exit_ticks: u8,
    pub slope_on_deg: f32,
    pub slope_off_deg: f32,
    pub accel_on: f32,
    pub accel_off: f32,
    pub rung_margin: f32,
    pub instance_gap_s: f32,
}

impl Default for Hysteresis {
    fn default() -> Self {
        Self {
            enter_ticks: 2,
            exit_ticks: 3,
            slope_on_deg: 3.0,
            slope_off_deg: 2.5,
            accel_on: 0.15,
            accel_off: 0.075,
            rung_margin: 0.15,
            instance_gap_s: 10.0,
        }
    }
}

/// Contract payload emitted when a phase span closes (Phase 2 SDS consumer, D5).
#[derive(Clone, Debug, Serialize)]
pub struct SpanClosed {
    pub run: RunKey,
    pub scenario: &'static str,
    pub phase: String,
    pub span_id: u64,
    pub instance_id: u64,
    pub start: Clock,
    pub end: Clock,
    pub dwell_s: f64,
    pub score_first: Option<f32>,
    pub score_min: Option<f32>,
    pub gated: bool,
    pub static_margin_min: f32,
    pub peak_ratio_max: f32,
    pub plan_gap_max: f32,
    pub use_ratio_max: f32,
    pub marks: Vec<Mark>,
    pub enter_from: Option<String>,
    pub close: CloseWhy,
}

/// Contract payload emitted on every world tick (Phase 3 TS consumer, D5).
#[derive(Clone, Debug, Serialize)]
pub struct NnTick {
    pub seq: u64,
    pub run: RunKey,
    pub tick: u64,
    pub time_s: f64,
    pub open: Vec<u64>,
    pub static_margin: f32,
    pub peak_ratio: f32,
    pub accel_meas: f32,
    pub accel_plan: f32,
    pub pitch_deg: f32,
    pub use_ratio: f32,
}

/// Legacy per-(scenario, phase) cumulative stats (extended with dwell time).
#[derive(Clone, Debug, Default, Serialize)]
pub struct PhaseStat {
    pub scenario: String,
    pub name_ko: String,
    pub phase: String,
    pub subsystem: String,
    pub weight: f32,
    pub ladder: Option<String>,
    pub count: u32,
    pub score_mean: f32,
    pub gated: u32,
    pub item_ok: u32,
    pub min_margin: f32,
    pub peak_ratio: f32,
    pub last: Option<PhaseResult>,
    pub dwell_s_total: f64,
    pub score_time_mean: f32,
}

#[derive(Clone, Debug, Default)]
struct PendingEntry {
    phase: String,
    count: u8,
    start: Clock,
    static_margin_min: f32,
    peak_ratio_max: f32,
    plan_gap_max: f32,
    use_ratio_max: f32,
    marks: Vec<Mark>,
    scores: Vec<PhaseResult>,
    held: bool,
}

#[derive(Clone, Debug, Default)]
struct ScenarioLane {
    open: Option<PhaseSpan>,
    last_present: Clock,
    last_accel_meas: f32,
    last_plan_gap: f32,
    last_closed_phase: Option<String>,
    miss_count: u8,
    pending: Option<PendingEntry>,
    open_instance: Option<Instance>,
    last_span_end_s: Option<f64>,
    /// Tracks latest attached score per phase inside the open instance for combine().
    inst_phase_scores: BTreeMap<String, (f32, Option<&'static str>, f32, bool)>,
}

/// Compact history point kept per `(scenario, phase)` in a 32-entry ring (Phase 5 – Part 1 · D1).
#[derive(Clone, Debug, Serialize)]
pub struct PhaseHistoryPoint {
    pub tick: u64,
    pub score: f32,
    pub ungated: f32,
    pub item: u8,
    pub gated: bool,
    pub min_margin: f32,
    pub peak_ratio: f32,
}

/// Active phase chip snapshot for the live progress band and trail (Phase 5 – Part 1 · D1).
#[derive(Clone, Debug, Serialize)]
pub struct ActiveNow {
    pub scenario: String,
    pub phase: String,
    pub ladder: Option<String>,
    pub since_tick: u64,
    pub state: String,
    pub score: f32,
    pub ungated: f32,
    pub item: u8,
    pub gated: bool,
    pub excused: bool,
    pub top_loss: Option<(String, f32)>,
}

const PHASE_HISTORY_CAP: usize = 32;
const TRAIL_CAP: usize = 32;
/// Score threshold below which a phase enters a low-score episode (Phase 6 – Part 1 · D3).
pub const LOW_SCORE_THRESH: f32 = 60.0;
/// Minimum elapsed shift duration (s) before showing extrapolated live SLI (Phase 6 – Part 1 · D4).
pub const SHIFT_MIN_S: f32 = 30.0;

/// Time-weighted, dwell-aware per-(scenario, phase) series with low-score episode & recovery tracking
/// (Phase 6 – Part 1 · D3). Display/audit only — banned in `DECISION_FILES`.
#[derive(Clone, Debug, Serialize)]
pub struct PhaseSeries {
    pub n: u32,
    pub dwell_s: f32,
    pub score_tw_num: f32,
    pub score_min: f32,
    pub low_s: f32,
    pub episodes: u32,
    pub low_since: Option<f32>,
    pub recovery_s: Vec<f32>,
    pub unrecovered: bool,
    /// Match-class counts: `[dev, between, holdout_like]`.
    pub match_counts: [u32; 3],
    #[serde(skip)]
    pub score_samples: Vec<(f32, f32)>,
    #[serde(skip)]
    pub last: Option<(f32, PhaseResult)>,
}

impl Default for PhaseSeries {
    fn default() -> Self {
        Self {
            n: 0,
            dwell_s: 0.0,
            score_tw_num: 0.0,
            score_min: f32::INFINITY,
            low_s: 0.0,
            episodes: 0,
            low_since: None,
            recovery_s: Vec::new(),
            unrecovered: false,
            match_counts: [0; 3],
            score_samples: Vec::new(),
            last: None,
        }
    }
}

impl PhaseSeries {
    pub fn push(&mut self, time_s: f32, pr: PhaseResult, mc: MatchClass, is_transition: bool) {
        if let Some((prev_t, prev_pr)) = &self.last {
            let dt = (time_s - *prev_t).max(0.0);
            if dt > 0.0 {
                self.dwell_s += dt;
                self.score_tw_num += prev_pr.score * dt;
                if prev_pr.score < LOW_SCORE_THRESH {
                    self.low_s += dt;
                }
                self.score_samples.push((prev_pr.score, dt));
            }
        }
        if is_transition {
            self.n += 1;
            let idx = match mc {
                MatchClass::Dev => 0,
                MatchClass::Between => 1,
                MatchClass::HoldoutLike => 2,
            };
            self.match_counts[idx] += 1;
        }
        self.score_min = self.score_min.min(pr.score);
        if pr.score < LOW_SCORE_THRESH {
            if self.low_since.is_none() {
                self.low_since = Some(time_s);
                self.episodes += 1;
            }
            self.unrecovered = true;
        } else if let Some(t0) = self.low_since.take() {
            self.recovery_s.push((time_s - t0).max(0.0));
            self.unrecovered = false;
        }
        self.last = Some((time_s, pr));
    }

    /// Finalize any open dwell interval up to `now_s`.
    pub fn finish(&mut self, now_s: f32) {
        if let Some((prev_t, prev_pr)) = self.last.take() {
            let dt = (now_s - prev_t).max(0.0);
            if dt > 0.0 {
                self.dwell_s += dt;
                self.score_tw_num += prev_pr.score * dt;
                if prev_pr.score < LOW_SCORE_THRESH {
                    self.low_s += dt;
                }
                self.score_samples.push((prev_pr.score, dt));
            }
        }
        self.unrecovered = self.low_since.is_some();
    }

    pub fn score_tw(&self) -> Option<f32> {
        (self.dwell_s > 1e-6).then(|| self.score_tw_num / self.dwell_s)
    }

    /// Time-weighted 10th percentile score (available once `n >= 10`).
    pub fn score_p10(&self) -> Option<f32> {
        if self.n < 10 || self.score_samples.is_empty() {
            return None;
        }
        let total_w: f32 = self.score_samples.iter().map(|(_, w)| *w).sum();
        if total_w <= 1e-6 {
            return None;
        }
        let mut sorted = self.score_samples.clone();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut acc = 0.0f32;
        let target = 0.10 * total_w;
        for (s, w) in sorted {
            acc += w;
            if acc >= target {
                return Some(s);
            }
        }
        None
    }
}

/// One task bucket inside [`ShiftLedger`] (Phase 6 – Part 1 · D4).
#[derive(Clone, Debug, Default, Serialize)]
pub struct ShiftTask {
    pub dwell_s: f32,
    pub li: f32,
    pub lmax: f32,
}

/// Live shift-level composite load index ledger (`S7-L`, Phase 6 – Part 1 · D4).
/// Strictly display/audit only — banned in `DECISION_FILES`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ShiftLedger {
    pub tasks: BTreeMap<String, ShiftTask>,
    pub since_s: f32,
    pub now_s: f32,
    #[serde(skip)]
    started: bool,
    #[serde(skip)]
    last_active: Option<(String, f32, f32, f32)>,
}

impl ShiftLedger {
    /// Record an observed task state `(key, li, lmax)` at `time_s`, attributing elapsed time since
    /// the previous sample to the previously active task.
    pub fn observe(&mut self, key: &str, time_s: f32, li: f32, lmax: f32) {
        if !self.started {
            self.since_s = time_s;
            self.started = true;
        }
        if let Some((prev_k, prev_t, prev_li, prev_lmax)) = self.last_active.take() {
            let dt = (time_s - prev_t).max(0.0);
            if dt > 0.0 {
                self.add_dwell(&prev_k, dt, prev_li, prev_lmax);
            }
        }
        let e = self.tasks.entry(key.to_string()).or_default();
        e.li = e.li.max(li);
        e.lmax = e.lmax.max(lmax.max(li));
        self.now_s = self.now_s.max(time_s);
        self.last_active = Some((key.to_string(), time_s, li, lmax.max(li)));
    }

    /// Directly accumulate a known dwell interval for a task bucket.
    pub fn add_dwell(&mut self, key: &str, dwell_s: f32, li: f32, lmax: f32) {
        if dwell_s <= 0.0 {
            return;
        }
        if !self.started {
            self.since_s = 0.0;
            self.started = true;
        }
        let e = self.tasks.entry(key.to_string()).or_default();
        e.dwell_s += dwell_s;
        e.li = e.li.max(li);
        e.lmax = e.lmax.max(lmax.max(li));
        self.now_s = (self.now_s).max(self.since_s + self.tasks.values().map(|t| t.dwell_s).sum::<f32>());
    }

    pub fn elapsed_s(&self) -> f32 {
        let sum: f32 = self.tasks.values().map(|t| t.dwell_s).sum();
        (self.now_s - self.since_s).max(sum)
    }

    /// Compute NIOSH SLI using the exact shared formula [`nn::sli_from_tasks`].
    pub fn sli(&self) -> Option<f32> {
        let total_t = self.elapsed_s();
        if total_t <= 1e-6 || self.tasks.is_empty() {
            return None;
        }
        let v: Vec<(f32, f32, f32)> = self
            .tasks
            .values()
            .filter(|k| k.dwell_s > 0.0)
            .map(|k| (k.li, k.lmax, k.dwell_s / total_t))
            .collect();
        nn::sli_from_tasks(&v)
    }

    pub fn to_json(&self, heat_max: f32) -> Value {
        let t_s = self.elapsed_s();
        let shown = t_s >= SHIFT_MIN_S && !self.tasks.is_empty();
        let sli = self.sli();
        let tasks: Vec<Value> = self
            .tasks
            .iter()
            .map(|(k, v)| {
                json!({
                    "key": k,
                    "dwell_s": (v.dwell_s as f64 * 10.0).round() / 10.0,
                    "frac": if t_s > 1e-6 { ((v.dwell_s / t_s) as f64 * 1000.0).round() / 1000.0 } else { 0.0 },
                    "li": (v.li as f64 * 1000.0).round() / 1000.0,
                    "lmax": (v.lmax as f64 * 1000.0).round() / 1000.0,
                })
            })
            .collect();
        json!({
            "sli": if shown { sli.map(|x| (x as f64 * 1000.0).round() / 1000.0) } else { None },
            "sli_raw": sli.map(|x| (x as f64 * 1000.0).round() / 1000.0),
            "t_s": (t_s as f64 * 10.0).round() / 10.0,
            "shown": shown,
            "assumption": "관측 구성이 교대 내내 반복",
            "tasks": tasks,
            "heat_max": (heat_max as f64 * 1000.0).round() / 1000.0,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Partition {
    pub key: RunKey,
    pub phases: BTreeMap<(String, String), PhaseStat>,
    pub series: BTreeMap<(String, String), PhaseSeries>,
    pub shift: ShiftLedger,
    pub phase_history: BTreeMap<(String, String), VecDeque<PhaseHistoryPoint>>,
    pub last_ticks: BTreeMap<(String, String), u64>,
    pub now: Vec<ActiveNow>,
    pub trail: VecDeque<ActiveNow>,
    pub rev: u64,
    lanes: BTreeMap<&'static str, ScenarioLane>,
    pub closed_spans: VecDeque<PhaseSpan>,
    pub closed_instances: VecDeque<Instance>,
    pub cause_ticks: BTreeMap<String, u64>,
    pub worker_marks: VecDeque<Mark>,
    slope_latched: bool,
    accel_latched: i8,
    last_clock: Clock,
}

impl Partition {
    fn new(key: RunKey) -> Self {
        Self { key, ..Self::default() }
    }

    fn record_phase_stat(&mut self, scenario: &str, name_ko: &str, r: &PhaseResult) {
        let e = self.phases.entry((scenario.to_string(), r.id.clone())).or_insert_with(|| PhaseStat {
            scenario: scenario.into(),
            name_ko: name_ko.into(),
            phase: r.id.clone(),
            subsystem: r.subsystem.into(),
            weight: r.weight,
            ladder: r.ladder.map(Into::into),
            min_margin: f32::INFINITY,
            ..PhaseStat::default()
        });
        e.count += 1;
        e.score_mean += (r.score - e.score_mean) / e.count as f32;
        e.gated += (!r.gates.is_empty()) as u32;
        e.item_ok += (r.item >= 1) as u32;
        e.min_margin = e.min_margin.min(r.min_margin);
        e.peak_ratio = e.peak_ratio.max(r.peak_ratio);
        e.last = Some(r.clone());
    }

    fn add_dwell_to_stat(&mut self, scenario: &str, phase: &str, dwell_s: f64, span_score: Option<f32>) {
        if let Some(e) = self.phases.get_mut(&(scenario.to_string(), phase.to_string())) {
            let prev_t = e.dwell_s_total;
            e.dwell_s_total += dwell_s;
            if let Some(s) = span_score {
                if e.dwell_s_total > 1e-6 {
                    e.score_time_mean = ((e.score_time_mean as f64 * prev_t + s as f64 * dwell_s) / e.dwell_s_total) as f32;
                }
            }
        }
    }

    fn summary(&self) -> Vec<Value> {
        let mut by: BTreeMap<&str, Vec<&PhaseStat>> = BTreeMap::new();
        for p in self.phases.values() {
            by.entry(p.scenario.as_str()).or_default().push(p);
        }
        by.into_iter()
            .map(|(id, ps)| {
                let mut parts: Vec<(f32, f32)> = Vec::new();
                let mut ladders: BTreeMap<&str, (f32, f32)> = BTreeMap::new();
                for p in &ps {
                    match p.ladder.as_deref() {
                        Some(l) => {
                            let e = ladders.entry(l).or_insert((p.weight, f32::INFINITY));
                            e.1 = e.1.min(p.score_mean);
                        }
                        None => parts.push((p.weight, p.score_mean)),
                    }
                }
                parts.extend(ladders.values().copied());
                let w: f32 = parts.iter().map(|x| x.0).sum();
                let score = parts.iter().map(|(w, s)| w * s).sum::<f32>() / w.max(1e-6);
                json!({
                    "scenario": id,
                    "name_ko": ps[0].name_ko,
                    "phases_seen": ps.iter().map(|p| p.phase.clone()).collect::<Vec<_>>(),
                    "score": (score as f64 * 10.0).round() / 10.0,
                    "events": ps.iter().map(|p| p.count).sum::<u32>(),
                    "gated": ps.iter().map(|p| p.gated).sum::<u32>(),
                    "min_margin": ps.iter().map(|p| p.min_margin).fold(f32::INFINITY, f32::min),
                    "peak_ratio": ps.iter().map(|p| p.peak_ratio).fold(0.0, f32::max),
                })
            })
            .collect()
    }

    fn open_span_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.lanes.values().filter_map(|l| l.open.as_ref().map(|s| s.id)).collect();
        ids.sort_unstable();
        ids
    }
}

#[derive(Clone, Debug)]
pub struct NnLive {
    pub hyst: Hysteresis,
    partitions: BTreeMap<RunKey, Partition>,
    order: VecDeque<RunKey>,
    active_key: RunKey,
    next_span_id: u64,
    next_instance_id: u64,
    cached_lib: Option<(String, Vec<ScenarioSpec>)>,
    last_seq: u64,
    epoch: u32,
    closed_queue: VecDeque<SpanClosed>,
    closed_inst_queue: VecDeque<(RunKey, Instance)>,
}

impl Default for NnLive {
    fn default() -> Self {
        let key = RunKey { run_id: "default".into(), pass: "game".into(), worker: 1, form: String::new() };
        let mut partitions = BTreeMap::new();
        partitions.insert(key.clone(), Partition::new(key.clone()));
        let mut order = VecDeque::new();
        order.push_back(key.clone());
        Self {
            hyst: Hysteresis::default(),
            partitions,
            order,
            active_key: key,
            next_span_id: 1,
            next_instance_id: 1,
            cached_lib: None,
            last_seq: 0,
            epoch: 1,
            closed_queue: VecDeque::new(),
            closed_inst_queue: VecDeque::new(),
        }
    }
}

fn form_id_for(plan: &BodyPlan, base: BaseKind, fingers: FingerCount) -> String {
    format!("{}:{:?}:f{}", plan.id(), base, u8::from(fingers)).to_lowercase()
}

fn cause_combo_key(post: &Posture) -> String {
    let causes = causes_of(post);
    let mut parts: Vec<&'static str> = causes
        .iter()
        .map(|c| match c {
            nn::Cause::ArmRaise => "arm_raise",
            nn::Cause::ArmLift => "arm_lift",
            nn::Cause::FootLift => "foot_lift",
            nn::Cause::StepUp => "step_up",
            nn::Cause::Slope => "slope",
            nn::Cause::CarryFront => "carry_front",
            nn::Cause::Start => "start",
            nn::Cause::Brake => "brake",
        })
        .collect();
    if parts.is_empty() {
        parts.push("quiet");
    }
    let mut key = parts.join("+");
    if post.moving {
        key.push_str("+moving");
    }
    key
}

fn phase_dist(p: &PhaseSpec, post: &Posture) -> f32 {
    (p.posture.accel_mps2 - post.accel_mps2).abs()
        + (p.posture.slope_deg - post.slope_deg).abs() / 10.0
        + (p.posture.step_m - post.step_m).abs() * 10.0
}

fn finalize_instance(inst: &mut Instance, spec: Option<&ScenarioSpec>, scores: &BTreeMap<String, (f32, Option<&'static str>, f32, bool)>, legged: bool, clock: Clock) {
    inst.closed = Some(clock);
    if let Some(spec) = spec {
        let mut missing = Vec::new();
        let mut has_ladder = false;
        let mut saw_ladder = false;
        for ph in &spec.phases {
            if ph.ladder.is_some() {
                has_ladder = true;
                if inst.phases_seen.iter().any(|p| p == &ph.id) {
                    saw_ladder = true;
                }
            } else if !inst.phases_seen.iter().any(|p| p == &ph.id) {
                missing.push(ph.id.clone());
            }
        }
        if has_ladder && !saw_ladder {
            missing.push("ladder".into());
        }
        inst.complete = missing.is_empty();
        if !inst.complete {
            let reason = if spec.id == "S3" && missing.iter().any(|m| m == "reach") {
                "reach 실시간 불가".to_string()
            } else if spec.id == "S4" && legged && missing.iter().any(|m| m == "carry") {
                "carry 소스 불일치".to_string()
            } else {
                format!("미관측 위상: {}", missing.join(", "))
            };
            inst.partial_why = Some(reason);
        }
    }
    if !scores.is_empty() {
        let mut parts: Vec<(f32, f32)> = Vec::new();
        let mut ladders: BTreeMap<&'static str, (f32, f32)> = BTreeMap::new();
        let mut any_gated = false;
        for &(w, lad, s, gated) in scores.values() {
            any_gated |= gated;
            match lad {
                Some(l) => {
                    let e = ladders.entry(l).or_insert((w, f32::INFINITY));
                    e.1 = e.1.min(s);
                }
                None => parts.push((w, s)),
            }
        }
        parts.extend(ladders.values().copied());
        let wsum: f32 = parts.iter().map(|x| x.0).sum();
        let combined = if any_gated { 0.0 } else { parts.iter().map(|(w, s)| w * s).sum::<f32>() / wsum.max(1e-6) };
        inst.score = Some((combined * 10.0).round() / 10.0);
    }
}

impl NnLive {
    /// Return the cached N:N scenario library for the given body form, rebuilding only when the form changes.
    pub fn library_for(&mut self, form_id: &str, body: &Body) -> &[ScenarioSpec] {
        let needs = self.cached_lib.as_ref().map_or(true, |(f, _)| f != form_id);
        if needs {
            self.cached_lib = Some((form_id.to_string(), nn::library(body)));
        }
        &self.cached_lib.as_ref().unwrap().1
    }

    pub fn active_key(&self) -> &RunKey {
        &self.active_key
    }

    pub fn partitions(&self) -> &BTreeMap<RunKey, Partition> {
        &self.partitions
    }

    pub fn active_partition(&self) -> &Partition {
        self.partitions.get(&self.active_key).expect("active partition exists")
    }

    fn ensure_partition(&mut self, key: RunKey) {
        if !self.partitions.contains_key(&key) {
            self.partitions.insert(key.clone(), Partition::new(key.clone()));
            self.order.push_back(key.clone());
            while self.order.len() > MAX_PARTITIONS {
                if let Some(old) = self.order.pop_front() {
                    if old != key && old != self.active_key {
                        self.partitions.remove(&old);
                    } else {
                        self.order.push_back(old);
                        break;
                    }
                }
            }
        }
    }

    /// Switch active partition to a new run/pass/worker/form, closing open spans in the previous partition.
    pub fn begin_run(&mut self, run_id: &str, pass: &str, worker: u32, form: &str, clock: Clock) {
        let next = RunKey {
            run_id: run_id.into(),
            pass: pass.into(),
            worker,
            form: form.into(),
        };
        if next == self.active_key {
            return;
        }
        self.close_all(CloseWhy::RunEnd, clock);
        self.epoch = self.epoch.saturating_add(1);
        self.ensure_partition(next.clone());
        self.active_key = next;
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn rev(&self) -> u64 {
        self.active_partition().rev
    }

    pub fn form_id(&self) -> &str {
        &self.active_key.form
    }

    pub fn active_now(&self) -> &[ActiveNow] {
        &self.active_partition().now
    }

    pub fn trail(&self) -> Vec<ActiveNow> {
        self.active_partition().trail.iter().cloned().collect()
    }

    pub fn phase_history(&self, scenario: &str, phase: &str) -> Vec<PhaseHistoryPoint> {
        self.active_partition()
            .phase_history
            .get(&(scenario.to_string(), phase.to_string()))
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn set_now(&mut self, items: Vec<ActiveNow>) {
        let key = self.active_key.clone();
        let Some(part) = self.partitions.get_mut(&key) else { return };
        for it in &items {
            let dup = part.trail.back().is_some_and(|b| {
                b.scenario == it.scenario && b.phase == it.phase && b.since_tick == it.since_tick
            });
            if !dup {
                part.trail.push_back(it.clone());
                while part.trail.len() > TRAIL_CAP {
                    part.trail.pop_front();
                }
            }
        }
        part.now = items;
        part.rev = part.rev.saturating_add(1);
    }

    pub fn clear_now(&mut self) {
        let key = self.active_key.clone();
        let Some(part) = self.partitions.get_mut(&key) else { return };
        if !part.now.is_empty() {
            part.now.clear();
            part.rev = part.rev.saturating_add(1);
        }
    }

    /// Record a dwell-weighted sample into `PhaseSeries` (Phase 6 – Part 1 · D3).
    pub fn record_series_sample(&mut self, scenario: &str, phase: &str, time_s: f32, pr: PhaseResult, mc: MatchClass, is_transition: bool) {
        let key = self.active_key.clone();
        let Some(part) = self.partitions.get_mut(&key) else { return };
        part.series
            .entry((scenario.to_string(), phase.to_string()))
            .or_default()
            .push(time_s, pr, mc, is_transition);
    }

    /// Record an observed task into the active partition's `ShiftLedger` (Phase 6 – Part 1 · D4).
    pub fn record_shift_task(&mut self, task_key: &str, time_s: f32, li: f32, lmax: f32) {
        let key = self.active_key.clone();
        let Some(part) = self.partitions.get_mut(&key) else { return };
        part.shift.observe(task_key, time_s, li, lmax);
    }

    /// Ensure the active partition's key matches the runtime's current `(run_id, pass, worker, form)`.
    pub fn sync_run(&mut self, run_id: &str, pass: &str, worker: u32, plan: &BodyPlan, base: BaseKind, fingers: FingerCount, clock: Clock) {
        let form = form_id_for(plan, base, fingers);
        if self.active_key.run_id == "default" && self.active_partition().phases.is_empty() && self.active_partition().closed_spans.is_empty() && self.active_partition().lanes.is_empty() {
            let old_key = self.active_key.clone();
            let next = RunKey { run_id: run_id.into(), pass: pass.into(), worker, form };
            self.partitions.remove(&old_key);
            self.order.retain(|k| k != &old_key);
            self.ensure_partition(next.clone());
            self.active_key = next;
        } else if self.active_key.run_id != run_id || self.active_key.pass != pass || self.active_key.worker != worker || self.active_key.form != form {
            self.begin_run(run_id, pass, worker, &form, clock);
        }
    }

    /// Close all open spans and open instances in the active partition.
    pub fn close_all(&mut self, why: CloseWhy, clock: Clock) {
        let key = self.active_key.clone();
        let lib_snapshot = self.cached_lib.as_ref().map(|(_, l)| l.clone());
        let legged = lib_snapshot.as_ref().map_or(false, |l| l.iter().any(|s| s.id == "S6" && s.phases.iter().any(|p| p.id == "start" && p.posture.single_support)));
        let Some(part) = self.partitions.get_mut(&key) else { return };
        let mut newly_closed: Vec<(SpanClosed, Option<f32>)> = Vec::new();
        for (&scn_id, lane) in part.lanes.iter_mut() {
            if let Some(mut span) = lane.open.take() {
                span.end = Some(clock);
                span.close = Some(why);
                let dwell_s = span.dwell_s(clock.time_s);
                let closed = SpanClosed {
                    run: key.clone(),
                    scenario: span.scenario,
                    phase: span.phase.clone(),
                    span_id: span.id,
                    instance_id: span.instance,
                    start: span.start,
                    end: clock,
                    dwell_s,
                    score_first: span.score_first,
                    score_min: span.score_min,
                    gated: span.gated,
                    static_margin_min: span.static_margin_min,
                    peak_ratio_max: span.peak_ratio_max,
                    plan_gap_max: span.plan_gap_max,
                    use_ratio_max: span.use_ratio_max,
                    marks: span.marks.clone(),
                    enter_from: span.enter_from.clone(),
                    close: why,
                };
                lane.last_closed_phase = Some(span.phase.clone());
                lane.last_span_end_s = Some(clock.time_s);
                newly_closed.push((closed, span.score_min.or(span.score_first)));
                part.closed_spans.push_back(span);
                while part.closed_spans.len() > MAX_CLOSED_SPANS {
                    part.closed_spans.pop_front();
                }
            }
            lane.pending = None;
            lane.miss_count = 0;
            if let Some(mut inst) = lane.open_instance.take() {
                let spec = lib_snapshot.as_ref().and_then(|l| l.iter().find(|s| s.id == scn_id));
                finalize_instance(&mut inst, spec, &lane.inst_phase_scores, legged, clock);
                lane.inst_phase_scores.clear();
                self.closed_inst_queue.push_back((key.clone(), inst.clone()));
                while self.closed_inst_queue.len() > MAX_CLOSED_INSTANCES {
                    self.closed_inst_queue.pop_front();
                }
                part.closed_instances.push_back(inst);
                while part.closed_instances.len() > MAX_CLOSED_INSTANCES {
                    part.closed_instances.pop_front();
                }
            }
        }
        for (c, sc) in newly_closed {
            part.add_dwell_to_stat(c.scenario, &c.phase, c.dwell_s, sc);
            self.closed_queue.push_back(c);
            while self.closed_queue.len() > MAX_CLOSED_SPANS {
                self.closed_queue.pop_front();
            }
        }
    }

    /// Drain closed span events produced since the last call (Phase 2 SDS consumer hook).
    pub fn drain_closed_spans(&mut self) -> Vec<SpanClosed> {
        self.closed_queue.drain(..).collect()
    }

    /// Drain closed scenario instances produced since the last call (Phase 2 – Part 1 · D3).
    pub fn drain_closed_instances(&mut self) -> Vec<(RunKey, Instance)> {
        self.closed_inst_queue.drain(..).collect()
    }

    /// Return `(instance_id, span_id)` for an open or pending span matching `(scenario, phase)`.
    pub fn active_span_info(&self, scenario: &str, phase: &str) -> (Option<u64>, Option<u64>) {
        let part = self.active_partition();
        if let Some(lane) = part.lanes.get(scenario) {
            if let Some(sp) = lane.open.as_ref().filter(|s| s.phase == phase) {
                return (Some(sp.instance), Some(sp.id));
            }
            if let Some(inst) = lane.open_instance.as_ref() {
                return (Some(inst.id), None);
            }
        }
        (None, None)
    }

    /// Observe one world tick: applies hysteresis, updates open/closed spans and instances,
    /// attaches world-event marks, and emits the [`NnTick`] summary.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        clock: Clock,
        run_id: &str,
        pass: &str,
        plan: &BodyPlan,
        base: BaseKind,
        fingers: FingerCount,
        body: &Body,
        posture: &Posture,
        sense: &BodySense,
        static_margin: f32,
        peak_ratio: f32,
        events: &[TimedEvent],
    ) -> NnTick {
        self.last_seq = clock.seq;
        self.sync_run(run_id, pass, 1, plan, base, fingers, clock);
        let form = self.active_key.form.clone();
        let lib: Vec<ScenarioSpec> = self.library_for(&form, body).to_vec();
        let legged = body.legged();
        let hyst = self.hyst;

        // Route sub-worker marks into their own partitions before borrowing active partition.
        for te in events {
            if let Some((w, mark)) = event_to_mark(clock.seq, &te.event) {
                if w > 1 {
                    let wk_key = RunKey {
                        run_id: run_id.into(),
                        pass: pass.into(),
                        worker: w,
                        form: format!("crew:{w}"),
                    };
                    self.ensure_partition(wk_key.clone());
                    if let Some(p) = self.partitions.get_mut(&wk_key) {
                        p.worker_marks.push_back(mark);
                        while p.worker_marks.len() > 256 {
                            p.worker_marks.pop_front();
                        }
                    }
                }
            }
        }

        // Collect primary-worker marks & event-driven closes for this tick.
        let mut tick_marks: Vec<Mark> = Vec::new();
        let mut close_step = false;
        let mut close_cargo = false;
        let mut close_work = false;
        let mut had_hold = false;
        for te in events {
            match &te.event {
                WorldEvent::ActionDone { .. } => close_work = true,
                WorldEvent::StepTaken { worker: 1, .. } => close_step = true,
                WorldEvent::CargoLost { worker: 1, .. } => close_cargo = true,
                WorldEvent::Hold { worker: 1, .. } => had_hold = true,
                _ => {}
            }
            if let Some((1, mark)) = event_to_mark(clock.seq, &te.event) {
                tick_marks.push(mark);
            }
        }

        let plan_gap = (sense.accel_meas_mps2 - sense.accel_plan_mps2).abs();
        let key = self.active_key.clone();
        let mut closed_out: Vec<(SpanClosed, Option<f32>)> = Vec::new();
        let mut closed_insts_out: Vec<Instance> = Vec::new();
        let mut next_span_id = self.next_span_id;
        let mut next_inst_id = self.next_instance_id;

        {
            let part = self.partitions.get_mut(&key).expect("active partition");
            part.last_clock = clock;

            // Schmitt triggers on slope and acceleration.
            let mut eff_post = *posture;
            if part.slope_latched {
                if eff_post.slope_deg.abs() < hyst.slope_off_deg {
                    part.slope_latched = false;
                } else if eff_post.slope_deg.abs() < hyst.slope_on_deg {
                    let sign = if eff_post.slope_deg < 0.0 { -1.0 } else { 1.0 };
                    eff_post.slope_deg = sign * hyst.slope_on_deg;
                }
            } else if eff_post.slope_deg.abs() >= hyst.slope_on_deg {
                part.slope_latched = true;
            }

            if part.accel_latched != 0 {
                if eff_post.accel_mps2.abs() < hyst.accel_off {
                    part.accel_latched = 0;
                } else if eff_post.accel_mps2.abs() < hyst.accel_on && eff_post.source == Source::Accel {
                    eff_post.accel_mps2 = (part.accel_latched as f32) * hyst.accel_on;
                }
            } else if eff_post.accel_mps2 >= hyst.accel_on {
                part.accel_latched = 1;
            } else if eff_post.accel_mps2 <= -hyst.accel_on {
                part.accel_latched = -1;
            }

            // Cause lane tracking.
            let ckey = cause_combo_key(&eff_post);
            *part.cause_ticks.entry(ckey).or_insert(0) += 1;

            // Match candidate phases per scenario.
            let matched = matching_phases(&lib, &eff_post);
            let mut by_scn: BTreeMap<&'static str, &PhaseSpec> = BTreeMap::new();
            for (s, p) in matched {
                by_scn.insert(s.id, p);
            }

            for spec in &lib {
                let scn_id = spec.id;
                let lane = part.lanes.entry(scn_id).or_default();

                // Apply ladder rung hysteresis when staying on the same ladder.
                let mut cand: Option<&PhaseSpec> = by_scn.get(scn_id).copied();
                if let (Some(open), Some(c)) = (lane.open.as_ref(), cand) {
                    if open.phase != c.id && c.ladder.is_some() {
                        if let Some(cur_spec) = spec.phases.iter().find(|p| p.id == open.phase) {
                            if cur_spec.ladder == c.ladder {
                                let d_cur = phase_dist(cur_spec, &eff_post);
                                let d_new = phase_dist(c, &eff_post);
                                if d_cur - d_new <= hyst.rung_margin {
                                    cand = Some(cur_spec);
                                }
                            }
                        }
                    }
                }

                // Helper closure to open a span from a PendingEntry.
                let open_from_pending = |lane: &mut ScenarioLane,
                                         pe: PendingEntry,
                                         next_span_id: &mut u64,
                                         next_inst_id: &mut u64,
                                         closed_instances: &mut VecDeque<Instance>,
                                         closed_insts_out: &mut Vec<Instance>| {
                    let ph_idx = spec.phases.iter().position(|p| p.id == pe.phase).unwrap_or(0);
                    let need_new_inst = match &lane.open_instance {
                        None => true,
                        Some(inst) => {
                            let gap_expired = lane.last_span_end_s.is_some_and(|t| pe.start.time_s - t > hyst.instance_gap_s as f64);
                            ph_idx < inst.last_index || gap_expired
                        }
                    };
                    if need_new_inst {
                        if let Some(mut old_inst) = lane.open_instance.take() {
                            finalize_instance(&mut old_inst, Some(spec), &lane.inst_phase_scores, legged, pe.start);
                            lane.inst_phase_scores.clear();
                            closed_insts_out.push(old_inst.clone());
                            closed_instances.push_back(old_inst);
                            while closed_instances.len() > MAX_CLOSED_INSTANCES {
                                closed_instances.pop_front();
                            }
                        }
                        let iid = *next_inst_id;
                        *next_inst_id += 1;
                        lane.open_instance = Some(Instance {
                            id: iid,
                            scenario: scn_id,
                            opened: pe.start,
                            closed: None,
                            last_index: ph_idx,
                            segments: Vec::new(),
                            phases_seen: Vec::new(),
                            complete: false,
                            score: None,
                            partial_why: None,
                        });
                    }
                    let sid = *next_span_id;
                    *next_span_id += 1;
                    let inst = lane.open_instance.as_mut().unwrap();
                    inst.last_index = ph_idx;
                    inst.segments.push(sid);
                    if !inst.phases_seen.iter().any(|p| p == &pe.phase) {
                        inst.phases_seen.push(pe.phase.clone());
                    }
                    let mut score_first = None;
                    let mut score_min = None;
                    let mut gated = false;
                    for pr in &pe.scores {
                        if score_first.is_none() {
                            score_first = Some(pr.score);
                        }
                        score_min = Some(score_min.map_or(pr.score, |m: f32| m.min(pr.score)));
                        gated |= !pr.gates.is_empty();
                        lane.inst_phase_scores.insert(pr.id.clone(), (pr.weight, pr.ladder, pr.score, !pr.gates.is_empty()));
                    }
                    lane.open = Some(PhaseSpan {
                        id: sid,
                        instance: inst.id,
                        scenario: scn_id,
                        phase: pe.phase,
                        start: pe.start,
                        end: None,
                        enter_from: lane.last_closed_phase.clone(),
                        interruptions: 0,
                        held: pe.held,
                        score_first,
                        score_min,
                        gated,
                        static_margin_min: pe.static_margin_min,
                        peak_ratio_max: pe.peak_ratio_max,
                        plan_gap_max: pe.plan_gap_max,
                        use_ratio_max: pe.use_ratio_max,
                        marks: pe.marks,
                        close: None,
                    });
                    lane.last_present = clock;
                    lane.miss_count = 0;
                };

                // Helper to close current open span.
                let close_open = |lane: &mut ScenarioLane,
                                  end_clock: Clock,
                                  why: CloseWhy,
                                  closed_spans: &mut VecDeque<PhaseSpan>,
                                  closed_out: &mut Vec<(SpanClosed, Option<f32>)>| {
                    if let Some(mut span) = lane.open.take() {
                        span.end = Some(end_clock);
                        span.close = Some(why);
                        let dwell_s = span.dwell_s(end_clock.time_s);
                        let sc = span.score_min.or(span.score_first);
                        closed_out.push((
                            SpanClosed {
                                run: key.clone(),
                                scenario: span.scenario,
                                phase: span.phase.clone(),
                                span_id: span.id,
                                instance_id: span.instance,
                                start: span.start,
                                end: end_clock,
                                dwell_s,
                                score_first: span.score_first,
                                score_min: span.score_min,
                                gated: span.gated,
                                static_margin_min: span.static_margin_min,
                                peak_ratio_max: span.peak_ratio_max,
                                plan_gap_max: span.plan_gap_max,
                                use_ratio_max: span.use_ratio_max,
                                marks: span.marks.clone(),
                                enter_from: span.enter_from.clone(),
                                close: why,
                            },
                            sc,
                        ));
                        lane.last_closed_phase = Some(span.phase.clone());
                        lane.last_span_end_s = Some(end_clock.time_s);
                        closed_spans.push_back(span);
                        while closed_spans.len() > MAX_CLOSED_SPANS {
                            closed_spans.pop_front();
                        }
                    }
                    lane.miss_count = 0;
                };

                // Process candidate against open span / pending entry.
                let same_as_open = matches!((lane.open.as_ref(), cand), (Some(o), Some(c)) if o.phase == c.id);
                if same_as_open {
                    let o = lane.open.as_mut().unwrap();
                    o.static_margin_min = o.static_margin_min.min(static_margin);
                    o.peak_ratio_max = o.peak_ratio_max.max(peak_ratio);
                    o.plan_gap_max = o.plan_gap_max.max(plan_gap);
                    o.use_ratio_max = o.use_ratio_max.max(sense.cargo_use_ratio);
                    if had_hold {
                        o.held = true;
                    }
                    o.marks.extend(tick_marks.iter().cloned());
                    lane.last_present = clock;
                    lane.last_accel_meas = sense.accel_meas_mps2;
                    lane.last_plan_gap = plan_gap;
                    lane.miss_count = 0;
                    lane.pending = None;
                } else if lane.open.is_some() {
                    // Candidate absent or rival phase while a span is open.
                    let o = lane.open.as_mut().unwrap();
                    if lane.miss_count == 0 {
                        o.interruptions = o.interruptions.saturating_add(1);
                    }
                    if had_hold {
                        o.held = true;
                    }
                    o.marks.extend(tick_marks.iter().cloned());
                    lane.miss_count = lane.miss_count.saturating_add(1);

                    // Track rival candidate if present.
                    if let Some(c) = cand {
                        match lane.pending.as_mut() {
                            Some(pe) if pe.phase == c.id => {
                                pe.count = pe.count.saturating_add(1);
                                pe.static_margin_min = pe.static_margin_min.min(static_margin);
                                pe.peak_ratio_max = pe.peak_ratio_max.max(peak_ratio);
                                pe.plan_gap_max = pe.plan_gap_max.max(plan_gap);
                                pe.use_ratio_max = pe.use_ratio_max.max(sense.cargo_use_ratio);
                                pe.held |= had_hold;
                                pe.marks.extend(tick_marks.iter().cloned());
                            }
                            _ => {
                                lane.pending = Some(PendingEntry {
                                    phase: c.id.clone(),
                                    count: 1,
                                    start: clock,
                                    static_margin_min: static_margin,
                                    peak_ratio_max: peak_ratio,
                                    plan_gap_max: plan_gap,
                                    use_ratio_max: sense.cargo_use_ratio,
                                    marks: tick_marks.clone(),
                                    scores: Vec::new(),
                                    held: had_hold,
                                });
                            }
                        }
                    } else {
                        lane.pending = None;
                    }

                    if lane.miss_count >= hyst.exit_ticks.max(1) {
                        let why = if cand.is_some() { CloseWhy::Superseded } else { CloseWhy::Left };
                        let end_clk = lane.last_present;
                        close_open(lane, end_clk, why, &mut part.closed_spans, &mut closed_out);
                        if let Some(pe) = lane.pending.take() {
                            if pe.count >= hyst.enter_ticks.max(1) {
                                open_from_pending(lane, pe, &mut next_span_id, &mut next_inst_id, &mut part.closed_instances, &mut closed_insts_out);
                                lane.last_accel_meas = sense.accel_meas_mps2;
                                lane.last_plan_gap = plan_gap;
                            } else {
                                lane.pending = Some(pe);
                            }
                        }
                    }
                } else if let Some(c) = cand {
                    // No open span: build pending entry until `enter_ticks`.
                    match lane.pending.as_mut() {
                        Some(pe) if pe.phase == c.id => {
                            pe.count = pe.count.saturating_add(1);
                            pe.static_margin_min = pe.static_margin_min.min(static_margin);
                            pe.peak_ratio_max = pe.peak_ratio_max.max(peak_ratio);
                            pe.plan_gap_max = pe.plan_gap_max.max(plan_gap);
                            pe.use_ratio_max = pe.use_ratio_max.max(sense.cargo_use_ratio);
                            pe.held |= had_hold;
                            pe.marks.extend(tick_marks.iter().cloned());
                        }
                        _ => {
                            lane.pending = Some(PendingEntry {
                                phase: c.id.clone(),
                                count: 1,
                                start: clock,
                                static_margin_min: static_margin,
                                peak_ratio_max: peak_ratio,
                                plan_gap_max: plan_gap,
                                use_ratio_max: sense.cargo_use_ratio,
                                marks: tick_marks.clone(),
                                scores: Vec::new(),
                                held: had_hold,
                            });
                        }
                    }
                    if lane.pending.as_ref().is_some_and(|pe| pe.count >= hyst.enter_ticks.max(1)) {
                        let pe = lane.pending.take().unwrap();
                        open_from_pending(lane, pe, &mut next_span_id, &mut next_inst_id, &mut part.closed_instances, &mut closed_insts_out);
                        lane.last_accel_meas = sense.accel_meas_mps2;
                        lane.last_plan_gap = plan_gap;
                    }
                } else {
                    lane.pending = None;
                    // Close idle open instance if gap exceeded.
                    if let (Some(t_end), Some(_)) = (lane.last_span_end_s, lane.open_instance.as_ref()) {
                        if clock.time_s - t_end > hyst.instance_gap_s as f64 {
                            let mut inst = lane.open_instance.take().unwrap();
                            finalize_instance(&mut inst, Some(spec), &lane.inst_phase_scores, legged, clock);
                            lane.inst_phase_scores.clear();
                            closed_insts_out.push(inst.clone());
                            part.closed_instances.push_back(inst);
                            while part.closed_instances.len() > MAX_CLOSED_INSTANCES {
                                part.closed_instances.pop_front();
                            }
                        }
                    }
                }

                // Event-driven closes on open spans.
                if let Some(open) = lane.open.as_ref() {
                    if close_step && open.phase == "step" {
                        close_open(lane, clock, CloseWhy::Event("step_taken"), &mut part.closed_spans, &mut closed_out);
                    } else if close_cargo && (scn_id == "S4" || open.phase.contains("carry")) {
                        close_open(lane, clock, CloseWhy::Event("cargo_lost"), &mut part.closed_spans, &mut closed_out);
                        if let Some(mut inst) = lane.open_instance.take() {
                            finalize_instance(&mut inst, Some(spec), &lane.inst_phase_scores, legged, clock);
                            lane.inst_phase_scores.clear();
                            closed_insts_out.push(inst.clone());
                            part.closed_instances.push_back(inst);
                        }
                    } else if close_work && matches!(open.phase.as_str(), "lift" | "strike") {
                        close_open(lane, clock, CloseWhy::ActionDone, &mut part.closed_spans, &mut closed_out);
                    }
                }
            }
        }

        self.next_span_id = next_span_id;
        self.next_instance_id = next_inst_id;
        for inst in closed_insts_out {
            self.closed_inst_queue.push_back((key.clone(), inst));
            while self.closed_inst_queue.len() > MAX_CLOSED_INSTANCES {
                self.closed_inst_queue.pop_front();
            }
        }
        if let Some(part) = self.partitions.get_mut(&key) {
            for (c, sc) in closed_out {
                part.add_dwell_to_stat(c.scenario, &c.phase, c.dwell_s, sc);
                self.closed_queue.push_back(c);
                while self.closed_queue.len() > MAX_CLOSED_SPANS {
                    self.closed_queue.pop_front();
                }
            }
        }

        let open = self.open_span_ids();
        NnTick {
            seq: clock.seq,
            run: key,
            tick: clock.tick,
            time_s: clock.time_s,
            open,
            static_margin,
            peak_ratio,
            accel_meas: sense.accel_meas_mps2,
            accel_plan: sense.accel_plan_mps2,
            pitch_deg: sense.pitch_deg,
            use_ratio: sense.cargo_use_ratio,
        }
    }

    /// Attach a scored [`PhaseResult`] at `tick` to the active partition's stats and matching span
    /// (open, pending during enter hysteresis, or recently closed covering `tick`).
    pub fn attach_scores(&mut self, tick: u64, scenario: &str, name_ko: &str, r: &PhaseResult) {
        let key = self.active_key.clone();
        let Some(part) = self.partitions.get_mut(&key) else { return };
        part.record_phase_stat(scenario, name_ko, r);
        let pkey = (scenario.to_string(), r.id.clone());
        part.last_ticks.insert(pkey.clone(), tick);
        let hist = part.phase_history.entry(pkey).or_default();
        hist.push_back(PhaseHistoryPoint {
            tick,
            score: r.score,
            ungated: r.ungated_score,
            item: r.item,
            gated: !r.gates.is_empty(),
            min_margin: r.min_margin,
            peak_ratio: r.peak_ratio,
        });
        while hist.len() > PHASE_HISTORY_CAP {
            hist.pop_front();
        }
        part.rev = part.rev.saturating_add(1);
        if let Some(lane) = part.lanes.values_mut().find(|l| {
            l.open.as_ref().is_some_and(|s| s.scenario == scenario && s.phase == r.id)
                || l.pending.as_ref().is_some_and(|p| p.phase == r.id)
        }) {
            if let Some(span) = lane.open.as_mut().filter(|s| s.scenario == scenario && s.phase == r.id) {
                if span.score_first.is_none() {
                    span.score_first = Some(r.score);
                }
                span.score_min = Some(span.score_min.map_or(r.score, |m| m.min(r.score)));
                span.gated |= !r.gates.is_empty();
                lane.inst_phase_scores.insert(r.id.clone(), (r.weight, r.ladder, r.score, !r.gates.is_empty()));
                return;
            }
            if let Some(pe) = lane.pending.as_mut().filter(|p| p.phase == r.id) {
                pe.scores.push(r.clone());
                return;
            }
        }
        // Fallback: attach to a closed span whose [start.tick, end.tick] covers `tick`.
        if let Some(span) = part.closed_spans.iter_mut().rev().find(|s| {
            s.scenario == scenario && s.phase == r.id && s.start.tick <= tick && s.end.map_or(true, |e| tick <= e.tick)
        }) {
            if span.score_first.is_none() {
                span.score_first = Some(r.score);
            }
            span.score_min = Some(span.score_min.map_or(r.score, |m| m.min(r.score)));
            span.gated |= !r.gates.is_empty();
        }
    }

    /// Legacy compatibility recorder (writes into the active partition's phase stats).
    pub fn record(&mut self, scenario: &str, name_ko: &str, r: &PhaseResult) {
        let key = self.active_key.clone();
        if let Some(part) = self.partitions.get_mut(&key) {
            part.record_phase_stat(scenario, name_ko, r);
        }
    }

    pub fn phases(&self) -> impl Iterator<Item = &PhaseStat> {
        self.active_partition().phases.values()
    }

    pub fn is_empty(&self) -> bool {
        let p = self.active_partition();
        p.phases.is_empty() && p.closed_spans.is_empty() && p.lanes.values().all(|l| l.open.is_none())
    }

    pub fn open_span_ids(&self) -> Vec<u64> {
        self.active_partition().open_span_ids()
    }

    pub fn spans_for_worker(&self, worker: u32) -> Vec<&PhaseSpan> {
        if worker != 1 {
            return Vec::new();
        }
        let p = self.active_partition();
        let mut out: Vec<&PhaseSpan> = p.closed_spans.iter().collect();
        for l in p.lanes.values() {
            if let Some(sp) = &l.open {
                out.push(sp);
            }
        }
        out
    }

    /// Legacy summary format for the active partition only (keeps existing `nn_status.scenarios` contract).
    pub fn summary(&self) -> Vec<Value> {
        self.active_partition().summary()
    }

    /// Incremental status view since `since_seq` (Phase 1 – Part 1 · D3 & D5).
    pub fn status_since(&self, since_seq: u64) -> Value {
        let p = self.active_partition();
        let now_s = p.last_clock.time_s;
        let open: Vec<Value> = p
            .lanes
            .values()
            .filter_map(|l| {
                let s = l.open.as_ref()?;
                Some(json!({
                    "segment": s.id,
                    "instance": s.instance,
                    "scenario": s.scenario,
                    "phase": s.phase,
                    "start_tick": s.start.tick,
                    "start_seq": s.start.seq,
                    "dwell_s": (s.dwell_s(now_s) * 100.0).round() / 100.0,
                    "static_margin_min": s.static_margin_min,
                    "peak_ratio_max": s.peak_ratio_max,
                    "accel_meas": l.last_accel_meas,
                    "plan_gap": l.last_plan_gap,
                    "plan_gap_max": s.plan_gap_max,
                    "use_ratio_max": s.use_ratio_max,
                    "interruptions": s.interruptions,
                    "held": s.held,
                }))
            })
            .collect();
        let segments: Vec<Value> = p
            .closed_spans
            .iter()
            .filter(|s| s.end.map_or(s.start.seq, |e| e.seq) > since_seq || since_seq == 0)
            .map(|s| {
                json!({
                    "id": s.id,
                    "instance": s.instance,
                    "scenario": s.scenario,
                    "phase": s.phase,
                    "start_tick": s.start.tick,
                    "end_tick": s.end.map(|e| e.tick),
                    "dwell_s": (s.dwell_s(now_s) * 100.0).round() / 100.0,
                    "enter_from": s.enter_from,
                    "interruptions": s.interruptions,
                    "held": s.held,
                    "score_first": s.score_first,
                    "score_min": s.score_min,
                    "gated": s.gated,
                    "static_margin_min": s.static_margin_min,
                    "peak_ratio_max": s.peak_ratio_max,
                    "plan_gap_max": s.plan_gap_max,
                    "use_ratio_max": s.use_ratio_max,
                    "close": s.close.map(CloseWhy::as_str),
                    "marks": s.marks,
                })
            })
            .collect();
        let mut instances: Vec<Value> = p
            .closed_instances
            .iter()
            .filter(|i| i.closed.map_or(i.opened.seq, |c| c.seq) > since_seq || since_seq == 0)
            .map(|i| {
                json!({
                    "id": i.id,
                    "scenario": i.scenario,
                    "open": false,
                    "start_tick": i.opened.tick,
                    "end_tick": i.closed.map(|c| c.tick),
                    "segments": i.segments,
                    "phases_seen": i.phases_seen,
                    "complete": i.complete,
                    "score": i.score,
                    "partial_why": i.partial_why,
                })
            })
            .collect();
        for l in p.lanes.values() {
            if let Some(i) = &l.open_instance {
                instances.push(json!({
                    "id": i.id,
                    "scenario": i.scenario,
                    "open": true,
                    "start_tick": i.opened.tick,
                    "segments": i.segments,
                    "phases_seen": i.phases_seen,
                    "complete": i.complete,
                    "score": i.score,
                    "partial_why": i.partial_why,
                }));
            }
        }
        let mut coverage = serde_json::Map::new();
        for def in nn::defs().iter().filter(|d| d.core && d.kind == nn::ScenarioKind::Phase) {
            let seen: Vec<String> = def
                .phases
                .iter()
                .filter(|ph| {
                    p.phases.contains_key(&(def.id.clone(), ph.id.clone()))
                        || p.closed_spans.iter().any(|s| s.scenario == def.id && s.phase == ph.id)
                        || p.lanes.get(def.id.as_str()).and_then(|l| l.open.as_ref()).is_some_and(|s| s.phase == ph.id)
                })
                .map(|ph| ph.id.clone())
                .collect();
            let live_unreachable: Vec<String> = def.phases.iter().filter(|ph| !ph.live).map(|ph| ph.id.clone()).collect();
            coverage.insert(def.id.clone(), json!({ "seen": seen, "live_unreachable": live_unreachable }));
        }
        json!({
            "run": p.key,
            "seq": self.last_seq,
            "open": open,
            "segments": segments,
            "instances": instances,
            "coverage": Value::Object(coverage),
            "cause_ticks": p.cause_ticks,
            "partitions": self.partitions.len(),
        })
    }

    /// Deterministic signature of closed + open spans in the active partition (Phase 1 – Part 1 · §5.3).
    pub fn span_signature(&self) -> Vec<(String, String, u64, Option<u64>, Option<String>)> {
        let p = self.active_partition();
        p.closed_spans
            .iter()
            .map(|s| (s.scenario.to_string(), s.phase.clone(), s.start.tick, s.end.map(|e| e.tick), s.close.map(CloseWhy::as_str)))
            .collect()
    }
}

fn event_to_mark(seq: u64, ev: &WorldEvent) -> Option<(u32, Mark)> {
    match ev {
        WorldEvent::Bump { worker, peak_n, a_g, fell, .. } => Some((
            *worker,
            Mark { seq, kind: "bump", value: *a_g, detail: Some(format!("peak={peak_n:.0}N fell={fell}")) },
        )),
        WorldEvent::CargoSlip { worker, shift_m, use_ratio, .. } => Some((
            *worker,
            Mark { seq, kind: "cargo_slip", value: *use_ratio, detail: Some(format!("shift={:.1}cm", shift_m * 100.0)) },
        )),
        WorldEvent::CargoLost { worker, cause, .. } => Some((
            *worker,
            Mark { seq, kind: "cargo_lost", value: 1.0, detail: Some(cause.clone()) },
        )),
        WorldEvent::StepSeen { worker, dist_m, crouch_m, .. } => Some((
            *worker,
            Mark { seq, kind: "step_seen", value: *dist_m, detail: Some(format!("crouch={crouch_m:.2}m")) },
        )),
        WorldEvent::StepTaken { worker, a_z, .. } => Some((
            *worker,
            Mark { seq, kind: "step_taken", value: *a_z, detail: None },
        )),
        WorldEvent::Hold { worker, gap_m, .. } => Some((
            *worker,
            Mark { seq, kind: "hold", value: *gap_m, detail: None },
        )),
        _ => None,
    }
}

/// Persistent SDS-backed view of N:N scenario/phase results (Phase 2 – Part 1 · D3).
/// Supports `"run"` (current session run buffer), `"all"` (base + run merged), and `"instance"` (live).
pub struct NnView;

impl NnView {
    pub fn from_store(sds: &crate::sds::SdsStore, window: &str, live: &NnLive) -> Value {
        if window == "instance" {
            return json!({
                "window": "instance",
                "scenarios": live.summary(),
                "phases": live.phases().collect::<Vec<_>>(),
            });
        }
        let defs = nn::defs();
        let ledger = crate::sds::Track::Balance.ledger();
        let keys = sds.scope_keys_for_track(crate::sds::Track::Balance);

        // Extract per-(scenario_upper, phase) stats from persisted SDS keys (v2 `+ph` rung preferred, fallback v1 `balance|nn:<scn>|<ph>`).
        let mut by_phase: BTreeMap<(String, String), (crate::sds::ScopeStat, String, BTreeMap<String, String>)> = BTreeMap::new();
        for key in &keys {
            let st_opt = if window == "run" {
                sds.run_scope_by_key(ledger, key)
            } else {
                sds.merged_scope_by_key(ledger, key)
            };
            let Some(st) = st_opt else { continue };
            if !st.baseline.contains_key("score") {
                continue;
            }
            if let Ok(sk) = crate::sds::ScopeKey::parse(key) {
                if let Some(scn_raw) = sk.subject.strip_prefix("nn:") {
                    // Use the `+ph` rung (exactly one facet: Ph) for scenario-phase rollups.
                    if sk.facets.len() == 1 && sk.facets[0].0 == crate::sds::Facet::Ph {
                        let scn = scn_raw.to_ascii_uppercase();
                        let ph = sk.facets[0].1.clone();
                        let facets: BTreeMap<String, String> = sk.facets.iter().map(|(f, v)| (f.as_str().into(), v.clone())).collect();
                        by_phase.insert((scn, ph), (st, key.clone(), facets));
                    }
                }
            } else if let Some(rest) = key.strip_prefix("balance|nn:") {
                let mut parts = rest.split('|');
                let scn_raw = parts.next().unwrap_or("");
                let ph = parts.next().unwrap_or("");
                if !scn_raw.is_empty() && !ph.is_empty() {
                    let scn = scn_raw.to_ascii_uppercase();
                    by_phase.entry((scn, ph.to_string())).or_insert_with(|| {
                        let mut facets = BTreeMap::new();
                        facets.insert("ph".into(), ph.to_string());
                        (st, key.clone(), facets)
                    });
                }
            }
        }

        let mut phases_out: Vec<Value> = Vec::new();
        let mut by_scn: BTreeMap<String, Vec<(f32, Option<String>, f64, u64, u64, u64)>> = BTreeMap::new();

        for ((scn, ph), (st, key, facets)) in &by_phase {
            let Some(score_w) = st.baseline.get("score") else { continue };
            if score_w.n == 0 {
                continue;
            }
            let def = defs.iter().find(|d| d.id.eq_ignore_ascii_case(scn));
            let pdef = def.and_then(|d| d.phases.iter().find(|p| p.id == *ph));
            let name_ko = def.map(|d| d.name_ko.as_str()).unwrap_or(scn.as_str());
            let weight = pdef.map_or(1.0f32, |p| p.weight);
            let ladder = pdef.and_then(|p| p.ladder.clone());
            let n_eff = score_w.t.as_ref().map(|t| t.long.n_eff()).unwrap_or(score_w.n as f64);
            let item_ok = st.agreement.get("item_ok").map_or(0, |a| (a.mean * a.n as f64).round() as u64);
            let gated_n: u64 = st.gates.values().map(|g| g.blocked).max().unwrap_or(0);
            let min_margin = st.extrema.get("min_margin").map(|e| e.min);
            let peak_ratio_max = st.extrema.get("peak_ratio").map(|e| e.max);

            phases_out.push(json!({
                "scenario": scn,
                "name_ko": name_ko,
                "phase": ph,
                "weight": weight,
                "ladder": ladder,
                "n": score_w.n,
                "n_eff": (n_eff * 10.0).round() / 10.0,
                "mean_score": (score_w.mean * 10.0).round() / 10.0,
                "gated": gated_n,
                "item_ok": item_ok,
                "min_margin": min_margin,
                "peak_ratio_max": peak_ratio_max,
                "key": key,
                "facets": facets,
            }));
            by_scn.entry(scn.clone()).or_default().push((weight, ladder, score_w.mean, score_w.n, gated_n, item_ok));
        }

        let mut scenarios_out: Vec<Value> = Vec::new();
        for (scn, rows) in &by_scn {
            let def = defs.iter().find(|d| d.id.eq_ignore_ascii_case(scn));
            let name_ko = def.map(|d| d.name_ko.as_str()).unwrap_or(scn.as_str());
            let mut parts: Vec<(f64, f64)> = Vec::new();
            let mut ladders: BTreeMap<String, (f64, f64)> = BTreeMap::new();
            let mut total_n = 0u64;
            let mut total_gated = 0u64;
            for (w, lad, mean, n, g, _) in rows {
                total_n += *n;
                total_gated += *g;
                match lad {
                    Some(l) => {
                        let e = ladders.entry(l.clone()).or_insert((*w as f64, f64::INFINITY));
                        e.1 = e.1.min(*mean);
                    }
                    None => parts.push((*w as f64, *mean)),
                }
            }
            parts.extend(ladders.values().copied());
            let wsum: f64 = parts.iter().map(|x| x.0).sum();
            let score = if wsum > 0.0 {
                parts.iter().map(|(w, s)| w * s).sum::<f64>() / wsum
            } else {
                0.0
            };
            scenarios_out.push(json!({
                "id": scn,
                "name_ko": name_ko,
                "score": (score * 10.0).round() / 10.0,
                "n": total_n,
                "gated": total_gated,
                "phases": rows.len(),
                "window": window,
            }));
        }

        json!({
            "window": window,
            "scenarios": scenarios_out,
            "phases": phases_out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::parts::Tier;

    fn phase(id: &str, weight: f32, ladder: Option<&'static str>, score: f32) -> PhaseResult {
        PhaseResult {
            id: id.into(),
            subsystem: "reactive",
            weight,
            ladder,
            cells: vec![],
            seg_scores: [None; 7],
            score,
            ungated_score: score,
            coverage_w: 1.0,
            score_bounds: [score, score],
            partial: false,
            item: 2,
            gates: vec![],
            gate_codes: vec![],
            gate_hits: vec![],
            beyond_human: vec![],
            static_margin: 0.5,
            min_margin: 0.4,
            peak_ratio: 0.3,
            worst_joint: None,
            sustained_effort: 0.2,
        }
    }

    #[test]
    fn a_ladder_counts_once_at_its_worst_rung() {
        let mut live = NnLive::default();
        live.record("S4", "짐 들고 제동", &phase("carry", 0.625, None, 90.0));
        live.record("S4", "짐 들고 제동", &phase("brake_0.5", 0.375, Some("brake"), 80.0));
        live.record("S4", "짐 들고 제동", &phase("brake_2.5", 0.375, Some("brake"), 40.0));
        let s = live.summary();
        let score = s[0]["score"].as_f64().unwrap();
        assert!((score - 71.3).abs() < 0.051, "{score}");
        let mut one = NnLive::default();
        one.record("S4", "짐 들고 제동", &phase("carry", 0.625, None, 90.0));
        one.record("S4", "짐 들고 제동", &phase("brake_0.5", 0.375, Some("brake"), 80.0));
        assert!((one.summary()[0]["score"].as_f64().unwrap() - 86.3).abs() < 0.051);
        assert_eq!(live.phases().filter(|p| p.ladder.as_deref() == Some("brake")).count(), 2);
    }

    #[test]
    fn segment_hysteresis_merges_flicker() {
        let plan = BodyPlan::uniform(Tier::Mid);
        let body = Body::new(&plan, BaseKind::Wheel, FingerCount::Five);
        let mut live = NnLive::default();
        let sense = BodySense::default();
        // Enter S1/slope at 3.1°, then oscillate 2.9° / 3.1° (held by Schmitt trigger), then drop to 0° for 2 ticks (< exit_ticks=3) and return to 3.1°.
        let slopes = [3.1f32, 3.1, 2.9, 3.1, 2.9, 3.1, 0.0, 0.0, 3.1, 3.1];
        for (i, &deg) in slopes.iter().enumerate() {
            let clk = Clock { seq: i as u64 + 1, tick: i as u64, time_s: i as f64 * 0.1 };
            let post = if deg > 0.0 {
                Posture { source: Source::Slope, slope_deg: deg, ..Posture::quiet(0.0) }
            } else {
                Posture::quiet(0.0)
            };
            live.observe(clk, "r1", "game", &plan, BaseKind::Wheel, FingerCount::Five, &body, &post, &sense, 0.5, 0.2, &[]);
        }
        live.close_all(CloseWhy::RunEnd, Clock { seq: 11, tick: 10, time_s: 1.0 });
        let s1_spans: Vec<&PhaseSpan> = live.active_partition().closed_spans.iter().filter(|s| s.scenario == "S1" && s.phase == "slope").collect();
        assert_eq!(s1_spans.len(), 1, "flicker must stay merged into 1 span: {s1_spans:?}");
        assert_eq!(s1_spans[0].start.tick, 0, "start tick backdated");
        assert_eq!(s1_spans[0].interruptions, 1, "one 2-tick dropout counted as 1 interruption");
    }

    #[test]
    fn segment_start_is_backdated() {
        let plan = BodyPlan::uniform(Tier::Mid);
        let body = Body::new(&plan, BaseKind::Wheel, FingerCount::Five);
        let mut live = NnLive::default();
        let sense = BodySense::default();
        let post = Posture { source: Source::Slope, slope_deg: 8.0, ..Posture::quiet(0.0) };
        // Tick 10: first candidate tick (still pending since enter_ticks=2).
        live.observe(Clock { seq: 1, tick: 10, time_s: 1.0 }, "r1", "game", &plan, BaseKind::Wheel, FingerCount::Five, &body, &post, &sense, 0.45, 0.2, &[]);
        assert!(live.open_span_ids().is_empty());
        // Balance score arrives during the pending wait at tick 10!
        live.attach_scores(10, "S1", "경사 위 들기", &phase("slope", 0.5, None, 83.5));
        // Tick 11: second candidate tick -> opens span backdated to tick 10 with score_first = 83.5.
        live.observe(Clock { seq: 2, tick: 11, time_s: 1.1 }, "r1", "game", &plan, BaseKind::Wheel, FingerCount::Five, &body, &post, &sense, 0.40, 0.25, &[]);
        let open = live.active_partition().lanes.get("S1").unwrap().open.as_ref().expect("S1 span opened");
        assert_eq!(open.start.tick, 10, "start must be backdated to first candidate tick");
        assert_eq!(open.score_first, Some(83.5), "score arriving during entry wait must become score_first");
    }

    #[test]
    fn dwell_weighting_follows_world_time() {
        let mut s = PhaseSeries::default();
        s.push(0.0, phase("carry", 0.625, None, 90.0), MatchClass::Dev, true);
        s.push(30.0, phase("carry", 0.625, None, 50.0), MatchClass::Dev, true);
        s.finish(33.0);
        let want = (90.0 * 30.0 + 50.0 * 3.0) / 33.0;
        assert!((s.dwell_s - 33.0).abs() < 1e-4, "dwell={}", s.dwell_s);
        assert!((s.score_tw().unwrap() - want).abs() < 1e-4, "{} vs {want}", s.score_tw().unwrap());
        assert_eq!(s.n, 2);
        assert_eq!(s.match_counts, [2, 0, 0]);
    }

    #[test]
    fn recovery_runs_from_the_first_low_to_the_next_ok() {
        let mut s = PhaseSeries::default();
        s.push(0.0, phase("brake", 0.375, Some("brake"), 85.0), MatchClass::Dev, true);
        s.push(10.0, phase("brake", 0.375, Some("brake"), 45.0), MatchClass::Between, true);
        s.push(12.0, phase("brake", 0.375, Some("brake"), 50.0), MatchClass::Between, true);
        s.push(16.0, phase("brake", 0.375, Some("brake"), 80.0), MatchClass::Dev, true);
        s.push(20.0, phase("brake", 0.375, Some("brake"), 40.0), MatchClass::HoldoutLike, true);
        s.finish(25.0);
        assert_eq!(s.episodes, 2);
        assert_eq!(s.recovery_s.len(), 1);
        assert!((s.recovery_s[0] - 6.0).abs() < 1e-4, "{:?}", s.recovery_s);
        assert!(s.unrecovered);
        assert!((s.low_s - 11.0).abs() < 1e-4, "low_s={}", s.low_s);
        assert_eq!(s.match_counts, [2, 2, 1]);
    }

    #[test]
    fn live_sli_equals_offline_for_the_same_mix() {
        let body = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let ff = noto_core::balance::FeedForward::default();
        let lib = nn::library(&body);
        let res: Vec<nn::ScenarioResult> = lib.iter().map(|s| nn::evaluate(&body, s, &ff)).collect();
        let by = |id: &str| res.iter().find(|r| r.id == id).unwrap();
        let offline = nn::shift_index(&[(by("S1"), 120.0), (by("S2"), 60.0), (by("S4"), 60.0)]).unwrap();
        let mut ledger = ShiftLedger::default();
        for (id, mins) in [("S1", 120.0f32), ("S2", 60.0), ("S4", 60.0)] {
            let r = by(id);
            let li = r.phases.iter().map(|p| p.peak_ratio).fold(0.0, f32::max);
            let lmax = r.phases.iter().map(|p| p.sustained_effort).fold(0.0, f32::max).max(li);
            ledger.add_dwell(id, mins * 60.0, li, lmax);
        }
        let live = ledger.sli().unwrap();
        assert!((live - offline).abs() < 1e-4, "live={live} vs offline={offline}");
    }
}
