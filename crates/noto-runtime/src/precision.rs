use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use noto_sds::{DriftState, Scope, SdsStore, Track, Window};
use noto_ts::ttm::TtmForecaster;
use noto_ts::{
    forecast, forecast_with_prior, temporal_verdict, Cusum, ErrBook, Forecast, GateBand, HistoryMse, PendingRing,
    Predictor, PredictorRegistry, TemporalOutcome, TemporalVerdict, TtmPredictor,
};
use serde::Serialize;

use crate::config::GateCfg;
use crate::feedback::{FeedbackHub, FeedbackPath, Mode};
use crate::perception::{Candidate, EntKind};
use crate::vision::{DriftRate, Memory};

const CONFLICT_RADIUS_M: f32 = 0.4;
const STALE_TICKS: u64 = 400;
const MAX_LANDMARKS: usize = 256;
const SERIES_CAP: usize = 64;
const HISTORY_CAP: usize = 32;
const SHADOW_CAP: usize = 32;
const METHODS: [&str; 4] = ["last", "ewma", "holt", "ttm"];
const SHIFT_RUN: usize = 3;
const SHIFT_SPREAD: f64 = 0.08;
const PRIOR_NU0: f64 = 4.0;
const DEPTH_LOG_MIN: f32 = 0.1;
const FORECAST_H: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LmState {
    Active,
    Rejected,
}

/// Display-only record of every temporal gate evaluation on a landmark (Phase 5 – Part 2 · D2).
/// Never read by any control decision.
#[derive(Clone, Debug, Serialize)]
pub struct GateMark {
    pub tick: u64,
    pub observed: f64,
    pub expected: Option<f64>,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub z: Option<f64>,
    pub outcome: &'static str,
    pub rescue: Option<&'static str>,
    pub method: String,
    pub sds_sd: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Landmark {
    pub id: u32,
    pub label: String,
    pub kind: EntKind,
    pub link: String,
    pub x: f32,
    pub z: f32,
    pub score: f32,
    pub margin: f32,
    pub good: u32,
    pub seen: u32,
    pub first_tick: u64,
    pub last_tick: u64,
    pub confirmed: bool,
    pub state: LmState,
    pub unreachable: u32,
    pub history: Vec<f64>,
    pub depths: Vec<f32>,
    pub held_run: Vec<f64>,
    pub shadow: Vec<f64>,
    pub cusum: Cusum,
    pub seg_offset: f64,
    #[serde(skip)]
    pub pending: PendingRing,
    #[serde(skip)]
    pub eval_seq: u64,
    pub trace: VecDeque<GateMark>,
    pub sigma_m: f32,
    pub odo_m: f32,
}

impl Landmark {
    pub fn distance_to(&self, x: f32, z: f32) -> f32 {
        ((self.x - x).powi(2) + (self.z - z).powi(2)).sqrt()
    }

    pub fn memory(&self) -> Memory {
        Memory::seen(self.x, self.z, self.sigma_m, self.odo_m, self.last_tick)
    }
}

/// Two sightings of one item label in one scan are the same item when their centres lie this close (m) [추정: under
/// the 0.15 m log; a felled tree's logs lie 0.15–0.26 m apart].
pub const DROP_SAME_M: f32 = 0.12;

#[derive(Clone, Debug)]
pub struct MemCtx {
    pub odo_m: f32,
    pub rate: DriftRate,
    pub base: String,
    pub camera: (f32, f32, f32),
    pub obs_floor_m: f32,
    /// The scope the observation scatter (`obs_floor_m`) is learned under: the base, and the eye's rig when it is a
    /// stereo pair — what one rig's looks scatter by says nothing about the other's.
    pub obs_key: String,
}

impl MemCtx {
    pub fn still() -> MemCtx {
        MemCtx { odo_m: 0.0, rate: DriftRate { per_m: 0.0, walk_m: 0.0, source: "spec" }, base: String::new(), camera: (424.0, 0.050, 0.08), obs_floor_m: 0.0, obs_key: String::new() }
    }

    pub fn sigma_obs(&self, depth_m: f32) -> f32 {
        self.sigma_law(depth_m).max(self.obs_floor_m).max(0.005)
    }

    /// The camera's own law at `depth_m`, without the scatter learned on top: `stereo_depth_sigma` along the ray and
    /// a pixel's bearing across it.
    pub fn sigma_law(&self, depth_m: f32) -> f32 {
        let (f, b, sub) = self.camera;
        let along = noto_core::act::material::stereo_depth_sigma(depth_m, f, b, sub);
        let across = depth_m / f.max(1.0);
        (along * along + across * across).sqrt()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GateResult {
    pub gate: &'static str,
    pub passed: bool,
    pub applied: bool,
    pub detail: String,
    pub severity: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub cand: Candidate,
    pub accepted: bool,
    pub gates: Vec<GateResult>,
    pub landmark: Option<u32>,
    pub floor: f32,
    pub floor_source: String,
    pub temporal: Option<TemporalVerdict>,
}

impl Judged {
    pub fn reason_ko(&self) -> String {
        match self.gates.iter().find(|g| g.applied && !g.passed) {
            Some(g) => format!("보류 · {} 게이트: {}", gate_ko(g.gate), g.detail),
            None => format!("수락 · 점수 {:.2} ≥ 바닥 {:.2} ({})", self.cand.score, self.floor, self.floor_source),
        }
    }

    pub fn blocked_by(&self) -> Option<&'static str> {
        self.gates.iter().find(|g| g.applied && !g.passed).map(|g| g.gate)
    }

    pub fn sound(&self) -> bool {
        self.gates.iter().filter(|g| g.applied && !g.passed).all(|g| matches!(g.gate, "confirm" | "temporal" | "conflict"))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GateCounters {
    pub evaluated: u64,
    pub accepted: u64,
    pub blocked_by: HashMap<String, u64>,
}

pub struct PrecisionGates {
    cfg: GateCfg,
    next_id: u32,
    landmarks: Vec<Landmark>,
    series: HashMap<String, Vec<f64>>,
    margins: HashMap<String, Vec<f64>>,
    books: HashMap<String, ErrBook>,
    registry: PredictorRegistry,
    ttm: Option<Arc<TtmForecaster>>,
    pub counters: GateCounters,
    scope_mode: String,
    ts_rev: u64,
}

fn with_extra<T>(ttm: &Option<Arc<TtmForecaster>>, f: impl FnOnce(&[&dyn Predictor]) -> T) -> T {
    match ttm {
        Some(t) => {
            let p = TtmPredictor(t.as_ref());
            f(&[&p])
        }
        None => f(&[]),
    }
}

pub fn gate_ko(gate: &str) -> &str {
    match gate {
        "score" => "점수",
        "margin" => "여유폭",
        "temporal" => "시계열",
        "conflict" => "라벨 충돌",
        "touch" => "접촉 확인",
        "confirm" => "반복 관측",
        other => other,
    }
}

fn severity(shortfall: f64, scale: f64) -> u8 {
    ((shortfall / scale.max(1e-6)).clamp(0.0, 1.0) * 100.0).round() as u8
}

pub fn past_errors(sds: &SdsStore, label: &str) -> Vec<HistoryMse> {
    past_errors_with(sds, label, None, &FeedbackHub::live(), 0)
}

pub fn past_errors_with(sds: &SdsStore, label: &str, ttm_fp8: Option<&str>, fb: &FeedbackHub, tick: u64) -> Vec<HistoryMse> {
    past_errors_for_series(sds, label, "score", "oracle", ttm_fp8, None, fb, tick)
}

/// D2 hierarchical realized-error lookup: live ErrBook -> secondary `<series>@<mode>` -> primary -> global relative `R_m / R_last`.
pub fn past_errors_for_series(
    sds: &SdsStore,
    label: &str,
    series: &str,
    mode: &str,
    ttm_fp8: Option<&str>,
    live_book: Option<&ErrBook>,
    fb: &FeedbackHub,
    tick: u64,
) -> Vec<HistoryMse> {
    let eff_mode = fb.effective_mode(FeedbackPath::C);
    if eff_mode == Mode::Off {
        return Vec::new();
    }
    let fscope = Scope::new(Track::Forecast, label, &format!("{series}@{mode}"));
    let base_scope = Scope::new(Track::Forecast, label, "");
    let global_scope = Scope::new(Track::Forecast, "", "");
    let mut out = Vec::new();

    // First look up local baseline R_last for global relative transfer when a new method joins.
    let local_last = live_book
        .filter(|_| eff_mode == Mode::Live)
        .and_then(|b| b.r.get("last").copied().filter(|v| v.is_finite() && *v > 0.0))
        .or_else(|| {
            fb.estimate_mean(FeedbackPath::C, sds, &fscope, &format!("{series}:res2:last"), Window::Auto, tick)
                .map(|e| e.value)
        })
        .or_else(|| fb.estimate_fusion_mse(FeedbackPath::C, sds, &base_scope, "last", tick).map(|e| e.value));

    for m in METHODS {
        let m_key = if m == "ttm" {
            ttm_fp8.map(|fp| format!("ttm@{fp}")).unwrap_or_else(|| "ttm".to_string())
        } else {
            m.to_string()
        };
        // 1) In-memory discounted realized ErrBook (live mode only).
        if eff_mode == Mode::Live {
            if let Some(book) = live_book {
                if let (Some(&mse), Some(&ne)) = (book.r.get(m), book.n_eff.get(m)) {
                    if mse.is_finite() && ne >= 0.5 {
                        out.push(HistoryMse { method: m.to_string(), mse, n: ne.round().max(1.0) as u64 });
                        continue;
                    }
                }
            }
        }
        // 2) SDS realized 1-step error on `<series>@<mode>` -> primary fallback.
        let res2_axis = format!("{series}:res2:{m_key}");
        if let Some(est) = fb.estimate_mean(FeedbackPath::C, sds, &fscope, &res2_axis, Window::Auto, tick) {
            if est.value.is_finite() && est.value >= 0.0 {
                out.push(HistoryMse { method: m.to_string(), mse: est.value, n: est.n_eff.round().max(1.0) as u64 });
                continue;
            }
        }
        // 3) Legacy `fusion_mse:<m>` compatibility lookup.
        let legacy_scope = if m == "ttm" {
            ttm_fp8
                .map(|fp| Scope::new(Track::Forecast, label, &format!("ttm@{fp}")))
                .unwrap_or_else(|| base_scope.clone())
        } else {
            base_scope.clone()
        };
        if let Some(est) = fb.estimate_fusion_mse(FeedbackPath::C, sds, &legacy_scope, &m_key, tick) {
            out.push(HistoryMse { method: m.to_string(), mse: est.value, n: est.n_eff.round().max(1.0) as u64 });
            continue;
        }
        // 4) Global relative fallback: borrow `R_m / R_last` scaled by local `R_last` (weak prior n=1).
        if m != "last" {
            if let Some(loc_r_last) = local_last.filter(|&v| v > 1e-12) {
                let g_m = fb.estimate_mean(FeedbackPath::C, sds, &global_scope, &res2_axis, Window::Auto, tick);
                let g_last = fb.estimate_mean(FeedbackPath::C, sds, &global_scope, &format!("{series}:res2:last"), Window::Auto, tick);
                if let (Some(gm), Some(gl)) = (g_m, g_last) {
                    if gl.value > 1e-12 && gm.value.is_finite() && gm.value >= 0.0 {
                        let ratio = (gm.value / gl.value).clamp(0.1, 10.0);
                        out.push(HistoryMse { method: m.to_string(), mse: ratio * loc_r_last, n: 1 });
                    }
                }
            }
        }
    }
    out
}

fn record_memory(sds: &mut SdsStore, mem: &MemCtx, label: &str, travelled: f32, innov: f32, predicted: f32, sigma0: f32, sigma_obs: f32) {
    if mem.base.is_empty() {
        return;
    }
    let sq = innov * innov;
    let scatter = 2.0 * (sigma0 * sigma0 + sigma_obs * sigma_obs);
    let walk_var = mem.rate.walk_m * mem.rate.walk_m * travelled.max(0.0);
    if travelled >= 2.0 {
        sds.record_baseline(&Scope::new(Track::Memory, &mem.base, "drift"), "drift_var_per_m2", ((sq - scatter - 2.0 * walk_var) / (travelled * travelled)) as f64);
    } else if travelled < 0.2 {
        let key = if mem.obs_key.is_empty() { &mem.base } else { &mem.obs_key };
        sds.record_baseline(&Scope::new(Track::Memory, key, "obs"), "scatter_var_m2", (0.5 * sq - sigma0 * sigma0) as f64);
    }
    let spread = (sigma0 * sigma0 + predicted * predicted + 2.0 * sigma_obs * sigma_obs + walk_var).sqrt().max(1e-4);
    sds.record_baseline(&Scope::new(Track::Memory, label, &mem.base), "innovation_z", (innov / spread) as f64);
}

impl PrecisionGates {
    pub fn new(cfg: GateCfg) -> PrecisionGates {
        PrecisionGates {
            cfg,
            next_id: 1,
            landmarks: Vec::new(),
            series: HashMap::new(),
            margins: HashMap::new(),
            books: HashMap::new(),
            registry: PredictorRegistry::with_builtins(),
            ttm: None,
            counters: GateCounters::default(),
            scope_mode: String::new(),
            ts_rev: 0,
        }
    }

    pub fn ts_rev(&self) -> u64 {
        self.ts_rev
    }

    pub fn set_recognizer_mode(&mut self, mode: &str) {
        self.scope_mode = if mode == "oracle" { String::new() } else { mode.to_string() };
    }

    pub fn recognizer_mode_tag(&self) -> &str {
        if self.scope_mode.is_empty() { "oracle" } else { &self.scope_mode }
    }

    pub fn vision_scope(&self, label: &str) -> Scope {
        Scope::new(Track::Vision, label, &self.scope_mode)
    }

    pub fn forecast_scope(&self, label: &str, series: &str) -> Scope {
        Scope::new(Track::Forecast, label, &format!("{series}@{}", self.recognizer_mode_tag()))
    }

    pub fn book_for(&self, label: &str, series: &str) -> Option<&ErrBook> {
        let key = self.forecast_scope(label, series).key_secondary();
        self.books.get(&key)
    }

    pub fn registry(&self) -> &PredictorRegistry {
        &self.registry
    }

    pub fn cfg(&self) -> &GateCfg {
        &self.cfg
    }

    pub fn set_cfg(&mut self, cfg: GateCfg) {
        self.cfg = cfg;
    }

    pub fn set_ttm(&mut self, ttm: Option<Arc<TtmForecaster>>) {
        self.registry.remove_by_id("ttm");
        if let Some(t) = &ttm {
            let p = TtmPredictor(t.as_ref());
            let _ = self.registry.register(noto_ts::PredictorSpec {
                id: "ttm".into(),
                rev: t.fp8().to_string(),
                caps: p.caps(),
            });
        }
        self.ttm = ttm;
        self.ts_rev = self.ts_rev.saturating_add(1);
    }

    pub fn ttm(&self) -> Option<Arc<TtmForecaster>> {
        self.ttm.clone()
    }

    pub fn has_ttm(&self) -> bool {
        self.ttm.is_some()
    }

    pub fn landmarks(&self) -> &[Landmark] {
        &self.landmarks
    }

    pub fn shift_landmarks(&mut self, f: impl Fn(f32, f32, f32) -> (f32, f32)) {
        for l in self.landmarks.iter_mut() {
            (l.x, l.z) = f(l.x, l.z, l.odo_m);
        }
    }

    pub fn landmark(&self, id: u32) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }

    pub fn clear(&mut self) {
        self.landmarks.clear();
        self.series.clear();
        self.margins.clear();
        self.books.clear();
        self.counters = GateCounters::default();
        self.ts_rev = self.ts_rev.saturating_add(1);
    }

    pub fn consume(&mut self, id: u32) {
        self.landmarks.retain(|l| l.id != id);
    }

    pub fn reject(&mut self, id: u32) {
        if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
            l.state = LmState::Rejected;
        }
    }

    pub fn note_unreachable(&mut self, id: u32) {
        if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
            l.unreachable += 1;
            if l.unreachable >= 2 {
                l.state = LmState::Rejected;
            }
        }
    }

    pub fn series(&self, label: &str) -> &[f64] {
        self.series.get(label).map_or(&[], Vec::as_slice)
    }

    pub fn margin_series(&self, label: &str) -> &[f64] {
        self.margins.get(label).map_or(&[], Vec::as_slice)
    }

    pub fn forecast_margin(&self, label: &str, horizon: usize) -> Option<Forecast> {
        let m = self.margins.get(label)?;
        if m.is_empty() {
            return None;
        }
        Some(with_extra(&self.ttm, |extra| forecast(m, horizon, extra, &[])))
    }

    pub fn forecast_margin_with(&self, label: &str, horizon: usize, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> Option<Forecast> {
        let m = self.margins.get(label)?;
        if m.is_empty() {
            return None;
        }
        let fp8 = self.ttm.as_deref().map(|t| t.fp8());
        let mode = self.recognizer_mode_tag();
        let book = self.book_for(label, "margin");
        let past = past_errors_for_series(sds, label, "margin", mode, fp8, book, fb, tick);
        let prior_s0 = self.prior_s0(label, "margin", sds, fb, tick);
        Some(with_extra(&self.ttm, |extra| forecast_with_prior(m, horizon, extra, &past, prior_s0)))
    }

    pub fn series_labels(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.series.iter().map(|(k, s)| (k.clone(), s.len())).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    pub fn prior_s0(&self, label: &str, series: &str, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> Option<f64> {
        let fscope = self.forecast_scope(label, series);
        fb.estimate_mean(FeedbackPath::B, sds, &fscope, &format!("{series}:res2:fused"), Window::Auto, tick)
            .map(|e| e.value.max(0.0).sqrt())
            .filter(|&s| s.is_finite() && s > 1e-5)
    }

    pub fn forecast_for(&self, label: &str, horizon: usize, sds: &SdsStore) -> Option<Forecast> {
        self.forecast_for_with(label, horizon, sds, &FeedbackHub::live(), 0)
    }

    pub fn forecast_for_with(&self, label: &str, horizon: usize, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> Option<Forecast> {
        let s = self.series.get(label)?;
        if s.is_empty() {
            return None;
        }
        let fp8 = self.ttm.as_deref().map(|t| t.fp8());
        let mode = self.recognizer_mode_tag();
        let book = self.book_for(label, "score");
        let past = past_errors_for_series(sds, label, "score", mode, fp8, book, fb, tick);
        let prior_s0 = self.prior_s0(label, "score", sds, fb, tick);
        Some(with_extra(&self.ttm, |extra| forecast_with_prior(s, horizon, extra, &past, prior_s0)))
    }

    pub fn ttm_history_n(&self, label: &str, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> u64 {
        let Some(fp8) = self.ttm.as_deref().map(|t| t.fp8()) else { return 0 };
        let fscope = self.forecast_scope(label, "score");
        if let Some(est) = fb.estimate_mean(FeedbackPath::C, sds, &fscope, &format!("score:res2:ttm@{fp8}"), Window::Auto, tick) {
            return est.n;
        }
        let scope = Scope::new(Track::Forecast, label, &format!("ttm@{fp8}"));
        fb.fusion_mse(FeedbackPath::C, sds, &scope, &format!("ttm@{fp8}"), tick).map_or(0, |(_, n)| n)
    }

    fn seed_book_from_sds(&self, label: &str, series: &str, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> ErrBook {
        let mut book = ErrBook::default();
        let fscope = self.forecast_scope(label, series);
        if fb.effective_mode(FeedbackPath::C) == Mode::Off {
            return book;
        }
        if let Some((k_mean, _, _)) = sds.adaptive_recent(&fscope, &format!("{series}:band_k")) {
            book.seed_k(k_mean);
        }
        if let Some(est) = fb.estimate_mean(FeedbackPath::B, sds, &fscope, &format!("{series}:res2:fused"), Window::Auto, tick) {
            if est.value.is_finite() && est.value >= 0.0 {
                book.fused = est.value;
                book.fused_n = est.n_eff.min(8.0);
                book.h_fused[0] = est.value;
                book.h_fused_n[0] = book.fused_n;
            }
        }
        let fp8 = self.ttm.as_deref().map(|t| t.fp8());
        for m in METHODS {
            let m_key = if m == "ttm" {
                fp8.map(|fp| format!("ttm@{fp}")).unwrap_or_else(|| "ttm".to_string())
            } else {
                m.to_string()
            };
            if let Some(est) = fb.estimate_mean(FeedbackPath::C, sds, &fscope, &format!("{series}:res2:{m_key}"), Window::Auto, tick) {
                book.seed_method(m, est.value, est.n_eff.round().max(1.0) as u64);
            }
        }
        book
    }

    pub fn floor_for(&self, label: &str, sds: &SdsStore) -> (f32, String) {
        self.floor_for_with(label, sds, &FeedbackHub::live(), 0)
    }

    pub fn floor_for_with(&self, label: &str, sds: &SdsStore, fb: &FeedbackHub, tick: u64) -> (f32, String) {
        let fixed = self.cfg.min_score as f64;
        let mut floor = fixed;
        let mut source = format!("고정 {fixed:.2}");
        if self.cfg.adaptive {
            let scope = self.vision_scope(label);
            if let Some(est) = fb
                .estimate_mean(FeedbackPath::A, sds, &scope, "score", Window::Auto, tick)
                .filter(|e| self.scope_mode.is_empty() || e.key == scope.key_secondary())
            {
                let shifted_thin = matches!(est.state, DriftState::Shifted { .. })
                    && (est.window != Window::Span || est.n_eff < 4.0);
                if shifted_thin {
                    source = format!("고정 {fixed:.2} (드리프트 구간 재수집 중)");
                } else {
                    let sd = est.se * est.n_eff.max(1.0).sqrt();
                    let pred = if (self.cfg.adaptive_k - 2.0).abs() < 0.05 {
                        noto_sds::estimate::pred_lower(est.value, sd, est.n_eff, 0.0228)
                    } else {
                        est.value - self.cfg.adaptive_k * sd * (1.0 + 1.0 / est.n_eff.max(1.0)).sqrt()
                    };
                    let learned = pred.min(fixed + 0.25);
                    if learned > floor {
                        floor = learned;
                        source = format!(
                            "SDS 학습 {learned:.2} (접촉 확인 {}건·n_e={:.1}, 평균 {:.2}±{sd:.2})",
                            est.n, est.n_eff, est.value
                        );
                    }
                }
            }
            if let Some(est) = fb
                .estimate_rate(FeedbackPath::A, sds, &scope, "touch", Window::Auto, tick)
                .filter(|e| self.scope_mode.is_empty() || e.key == scope.key_secondary())
            {
                let rate = est.value;
                if rate < 0.8 {
                    let hw = (0.5 * (est.hi - est.lo)).max(1e-6);
                    let w = ((0.8 - rate) / hw).clamp(0.0, 1.0);
                    let extra = (0.8 - rate) * 0.5 * w;
                    if extra > 1e-4 {
                        floor += extra;
                        source = format!("{source} + 접촉 일치율 {:.0}% ({}건)로 +{extra:.2}", rate * 100.0, est.n);
                    }
                }
            }
        }
        (floor.min(0.95) as f32, source)
    }

    fn dedup_scan_candidates(&self, mut cands: Vec<Candidate>) -> Vec<Candidate> {
        cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Candidate> = Vec::with_capacity(cands.len());
        for c in cands {
            let mut merged = false;
            for k in &mut kept {
                if k.label != c.label || k.kind != c.kind {
                    continue;
                }
                let (iou, iomin) = crate::perception::box_iou_and_iomin(&k.bbox, &c.bbox);
                let apart = match (k.world, c.world) {
                    (Some((x1, z1)), Some((x2, z2))) => Some((x1 - x2).hypot(z1 - z2)),
                    _ => None,
                };
                let overlap = iou >= 0.4 || iomin >= 0.65;
                // Items on the ground are a hand across and lie a hand apart (a felled tree's logs), and up close one
                // hides part of the next, so their boxes overlap: where both say where they lie, two sightings farther
                // apart than an item are two items, not one seen twice.
                let same = if c.kind == EntKind::Drop {
                    apart.map_or(overlap, |d| d <= DROP_SAME_M)
                } else {
                    overlap || apart.is_some_and(|d| d <= (0.45 * self.cfg.track_radius_m).max(0.35))
                };
                if same {
                    let area_k = (k.bbox[2] - k.bbox[0]) * (k.bbox[3] - k.bbox[1]);
                    let area_c = (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]);
                    if area_c < area_k && c.score >= k.score * 0.85 && c.world.is_some() {
                        k.bbox = c.bbox;
                        k.world = c.world;
                        k.depth_m = c.depth_m;
                        k.pixels = c.pixels;
                    }
                    merged = true;
                    break;
                }
            }
            if !merged {
                kept.push(c);
            }
        }
        kept.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.label.cmp(&b.label)));
        kept
    }

    fn associate(&mut self, c: &Candidate, x: f32, z: f32, tick: u64) -> u32 {
        let r = self.cfg.track_radius_m;
        let best = self
            .landmarks
            .iter()
            .filter(|l| l.label == c.label && l.kind == c.kind)
            .map(|l| (l.id, l.distance_to(x, z)))
            .filter(|(_, d)| *d <= r)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((id, _)) = best {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.landmarks.push(Landmark {
            id,
            label: c.label.clone(),
            kind: c.kind,
            link: c.link.clone(),
            x,
            z,
            score: c.score,
            margin: c.margin,
            good: 0,
            seen: 0,
            first_tick: tick,
            last_tick: tick,
            confirmed: false,
            state: LmState::Active,
            unreachable: 0,
            history: Vec::new(),
            depths: Vec::new(),
            held_run: Vec::new(),
            shadow: Vec::new(),
            cusum: Cusum::default(),
            seg_offset: 0.0,
            pending: PendingRing::default(),
            eval_seq: 1,
            trace: VecDeque::new(),
            sigma_m: 0.0,
            odo_m: 0.0,
        });
        id
    }

    fn prune(&mut self, tick: u64) {
        self.landmarks.retain(|l| l.confirmed || l.state == LmState::Rejected || tick.saturating_sub(l.last_tick) <= STALE_TICKS);
        if self.landmarks.len() > MAX_LANDMARKS {
            self.landmarks.sort_by(|a, b| b.last_tick.cmp(&a.last_tick));
            self.landmarks.truncate(MAX_LANDMARKS);
        }
    }

    pub fn judge(&mut self, cands: Vec<Candidate>, tick: u64, sds: &mut SdsStore) -> Vec<Judged> {
        self.judge_with(cands, tick, sds, &MemCtx::still())
    }

    pub fn judge_with(&mut self, cands: Vec<Candidate>, tick: u64, sds: &mut SdsStore, mem: &MemCtx) -> Vec<Judged> {
        self.judge_with_fb(cands, tick, sds, mem, &FeedbackHub::live())
    }

    pub fn judge_with_fb(&mut self, cands: Vec<Candidate>, tick: u64, sds: &mut SdsStore, mem: &MemCtx, fb: &FeedbackHub) -> Vec<Judged> {
        self.prune(tick);
        let deduped = self.dedup_scan_candidates(cands);
        deduped.into_iter().map(|c| self.judge_one(c, tick, sds, mem, fb)).collect()
    }

    /// Evaluate candidates reading feedback from `sds_read` while writing all new observations exclusively
    /// into `sds_write` (used by `holdout`/`eval` passes so evaluation never mutates control ledgers).
    pub fn judge_with_split(&mut self, cands: Vec<Candidate>, tick: u64, sds_read: &SdsStore, sds_write: &mut SdsStore, mem: &MemCtx, fb: &FeedbackHub) -> Vec<Judged> {
        if fb.mode().default == crate::feedback::Mode::Off || fb.mode().uses_frozen() {
            return self.judge_with_fb(cands, tick, sds_write, mem, fb);
        }
        let mut frozen_fb = FeedbackHub::live();
        let mut mode = fb.mode().clone();
        mode.default = crate::feedback::Mode::Frozen;
        frozen_fb.set_mode(mode, sds_read);
        self.judge_with_fb(cands, tick, sds_write, mem, &frozen_fb)
    }

    fn judge_one(&mut self, c: Candidate, tick: u64, sds: &mut SdsStore, mem: &MemCtx, fb: &FeedbackHub) -> Judged {
        let vscope = self.vision_scope(&c.label);
        sds.record_baseline(&vscope, "score_all", c.score as f64);
        sds.record_baseline(&vscope, "margin", c.margin as f64);

        let mut gates: Vec<GateResult> = Vec::new();

        let (floor, floor_source) = self.floor_for_with(&c.label, sds, fb, tick);
        let score_ok = c.score >= floor;
        gates.push(GateResult {
            gate: "score",
            passed: score_ok,
            applied: true,
            severity: if score_ok { 0 } else { severity((floor - c.score) as f64, floor as f64) },
            detail: format!("점수 {:.2} {} 하한 {floor:.2} ({floor_source})", c.score, if score_ok { "≥" } else { "<" }),
        });

        let margin_ok = c.margin >= self.cfg.min_margin;
        gates.push(GateResult {
            gate: "margin",
            passed: margin_ok,
            applied: true,
            severity: if margin_ok { 0 } else { severity((self.cfg.min_margin - c.margin) as f64, self.cfg.min_margin as f64) },
            detail: format!(
                "차점{} 대비 여유 {:.2} {} {:.2}",
                c.runner_up.as_ref().map(|r| format!("({r})")).unwrap_or_default(),
                c.margin,
                if margin_ok { "≥" } else { "<" },
                self.cfg.min_margin
            ),
        });
        let plausible = score_ok && margin_ok;

        let tid = c.world.map(|(x, z)| self.associate(&c, x, z, tick));
        let state = tid.and_then(|id| self.landmark(id)).map(|l| l.state);

        let mut temporal = None;
        match tid {
            Some(id) => {
                let (hist, last_depth) = self
                    .landmark(id)
                    .map(|l| (l.history.clone(), l.depths.last().copied()))
                    .unwrap_or_default();
                let need = self.cfg.temporal_min_history;
                let fscope = self.forecast_scope(&c.label, "score");
                // Learn online depth slope `b = Δscore / Δln(depth)` on plausible consecutive sightings with |Δln d| >= 0.1.
                if plausible {
                    if let (Some(&prev_s), Some(pd)) = (hist.last(), last_depth) {
                        if pd > 0.05 && c.depth_m > 0.05 {
                            let dlogd = (c.depth_m / pd).ln();
                            if dlogd.abs() >= DEPTH_LOG_MIN {
                                let raw_b = ((c.score as f64 - prev_s) / dlogd as f64).clamp(-2.0, 2.0);
                                sds.record_baseline(&fscope, "score:slope_dlogd", raw_b);
                            }
                        }
                    }
                }
                if plausible && hist.len() >= need {
                    let fp8 = self.ttm.as_deref().map(|t| t.fp8());
                    let mode = self.recognizer_mode_tag().to_string();
                    let fkey = fscope.key_secondary();
                    let live_fb = fb.effective_mode(FeedbackPath::C) == Mode::Live;

                    // Seed ErrBook once per scope from SDS (or use an ephemeral frozen copy in Frozen/Off modes).
                    if !self.books.contains_key(&fkey) {
                        let seeded = self.seed_book_from_sds(&c.label, "score", sds, fb, tick);
                        self.books.insert(fkey.clone(), seeded);
                    }
                    let mut work_book = if live_fb {
                        self.books.get(&fkey).cloned().unwrap_or_default()
                    } else {
                        self.seed_book_from_sds(&c.label, "score", sds, fb, tick)
                    };

                    // Depth covariate adjustment: subtract b * ln(d_now / d_prev) once slope has >= nu0 samples.
                    let learned_slope = fb
                        .estimate_mean(FeedbackPath::B, sds, &fscope, "score:slope_dlogd", Window::Auto, tick)
                        .filter(|e| e.n as f64 >= PRIOR_NU0 && e.value.is_finite())
                        .map(|e| e.value.clamp(-1.5, 0.5));
                    let dlogd = match last_depth {
                        Some(pd) if pd > 0.05 && c.depth_m > 0.05 => (c.depth_m / pd).ln(),
                        _ => 0.0,
                    };
                    let depth_adj = learned_slope.map_or(0.0, |b| b * dlogd as f64);
                    let y_adj = c.score as f64 - depth_adj;

                    let past = past_errors_for_series(sds, &c.label, "score", &mode, fp8, Some(&work_book), fb, tick);
                    let f = with_extra(&self.ttm, |extra| forecast(&hist, FORECAST_H, extra, &past));

                    let s_model = ((f.p90[0] - f.p10[0]) * 0.5 / 1.2816).max(1e-4);
                    let s0 = self.prior_s0(&c.label, "score", sds, fb, tick);
                    let s_g = work_book.s_gate(s0, PRIOR_NU0, s_model);
                    let band = GateBand { k: work_book.k, z_g: self.cfg.temporal_z, s_g };
                    let v = temporal_verdict(y_adj, &f, &band);

                    // Realized 1-step errors & coverage on ALL plausible observations (NO censorship on Held/Warn!).
                    let e_f = y_adj - f.point[0];
                    let inside_model80 = y_adj >= f.p10[0] && y_adj <= f.p90[0];
                    let base_fscope = Scope::new(Track::Forecast, &c.label, "");
                    for w in &f.weights {
                        let e_m = y_adj - w.point1;
                        let e2 = e_m * e_m;
                        let m_key = if w.method == "ttm" {
                            fp8.map(|fp| format!("ttm@{fp}")).unwrap_or_else(|| "ttm".to_string())
                        } else {
                            w.method.clone()
                        };
                        sds.record_baseline(&fscope, &format!("score:res2:{m_key}"), e2);
                        sds.record_baseline(&fscope, &format!("score:res:{m_key}"), e_m);
                        // Keep legacy `fusion_mse:<m>` populated for backward compatibility.
                        if w.method == "ttm" {
                            if let Some(fp) = fp8 {
                                let tscope = Scope::new(Track::Forecast, &c.label, &format!("ttm@{fp}"));
                                sds.record_baseline(&tscope, &format!("fusion_mse:ttm@{fp}"), e2);
                                continue;
                            }
                        }
                        sds.record_baseline(&base_fscope, &format!("fusion_mse:{}", w.method), e2);
                    }
                    sds.record_baseline(&fscope, "score:res2:fused", e_f * e_f);
                    sds.record_agreement(&fscope, "score:cov80", inside_model80);
                    sds.record_agreement(&fscope, "score:cov80c", v.inside80);
                    sds.record_agreement(&fscope, "score:covg", v.outcome != TemporalOutcome::Held);

                    let (pit, is80) = work_book.update(&f.weights, y_adj, e_f, s_model);
                    work_book.record_coverage(inside_model80, v.inside80, v.outcome != TemporalOutcome::Held);
                    work_book.aci_step(!v.inside80);
                    sds.record_baseline(&fscope, "score:pit", pit);
                    sds.record_baseline(&fscope, "score:is80", is80);
                    sds.record_baseline(&fscope, "score:band_k", work_book.k);

                    // Score multi-horizon pending forecasts (h = 2, 4, 8, 16) and issue the new 16-step forecast.
                    if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                        let cur_seq = l.eval_seq;
                        l.eval_seq += 1;
                        let scored_h = l.pending.score_ready_from(cur_seq, tick, y_adj, 2, &mut work_book);
                        for sc in scored_h {
                            sds.record_agreement(&fscope, &format!("h{}:cov80c", sc.h), sc.inside80);
                            sds.record_baseline(&fscope, &format!("h{}:dt_ticks", sc.h), sc.dt_ticks as f64);
                            for (m, sq) in sc.method_sq {
                                let mk = if m == "ttm" {
                                    fp8.map(|fp| format!("ttm@{fp}")).unwrap_or_else(|| "ttm".to_string())
                                } else {
                                    m
                                };
                                sds.record_baseline(&fscope, &format!("h{}:res2:{mk}", sc.h), sq);
                            }
                        }
                        l.pending.issue(cur_seq.saturating_sub(1), tick, &f.point, &f.p10, &f.p90, &f.weights);
                    }

                    let mut held = v.outcome == TemporalOutcome::Held;
                    let mut rescue: Option<&'static str> = None;
                    let mut detail = format!("{} · {}", v.method, v.reason);
                    if depth_adj.abs() >= 0.005 {
                        detail = format!("{detail} (깊이 보정 {:+.2})", -depth_adj);
                    }
                    // Cold-start depth-closer rescue (only when learned_slope < nu0 AND target genuinely got closer).
                    let depth_closer_rescue = held
                        && learned_slope.is_none()
                        && dlogd <= -DEPTH_LOG_MIN
                        && v.expected.is_some_and(|exp| (c.score as f64) > exp && c.score >= floor + 0.05);
                    if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                        let z_clamped = v.z.unwrap_or(0.0).clamp(-2.0, 2.0);
                        if depth_closer_rescue {
                            l.held_run.clear();
                            l.cusum.reset();
                            held = false;
                            rescue = Some("depth_closer");
                            detail = format!("{detail} → 근접 확인(깊이 감소)으로 점수 상승({:.2}) 수락", c.score);
                        } else if held {
                            l.shadow.push(c.score as f64);
                            if l.shadow.len() > SHADOW_CAP {
                                l.shadow.remove(0);
                            }
                            l.held_run.push(c.score as f64);
                            let cusum_hit = l.cusum.push(z_clamped).is_some();
                            let run = &l.held_run;
                            let spread = run.iter().cloned().fold(f64::MIN, f64::max) - run.iter().cloned().fold(f64::MAX, f64::min);
                            let agreeing_run = run.len() >= SHIFT_RUN && spread <= SHIFT_SPREAD;
                            if cusum_hit || agreeing_run {
                                let new_mean = run.iter().sum::<f64>() / run.len().max(1) as f64;
                                let tail_n = l.history.len().min(4).max(1);
                                let old_tail = l.history[l.history.len() - tail_n..].iter().sum::<f64>() / tail_n as f64;
                                let delta = new_mean - old_tail;
                                // Shift prior history by delta so its length stays >= need and the gate stays ON!
                                for x in &mut l.history {
                                    *x += delta;
                                }
                                l.seg_offset += delta;
                                l.held_run.clear();
                                l.cusum.reset();
                                work_book.halve_n_eff();
                                held = false;
                                rescue = Some("level_shift");
                                detail = format!("{detail} → 수준 변화({delta:+.2})로 받아들임 (이력 유지)");
                            }
                        } else {
                            let _ = l.cusum.push(z_clamped);
                            l.held_run.clear();
                        }
                        if l.trace.len() >= HISTORY_CAP {
                            l.trace.pop_front();
                        }
                        let trace_outcome = if held {
                            "held"
                        } else if v.outcome == TemporalOutcome::Warn && rescue.is_none() {
                            "warn"
                        } else {
                            "accepted"
                        };
                        l.trace.push_back(GateMark {
                            tick,
                            observed: c.score as f64,
                            expected: v.expected,
                            lo: v.lo,
                            hi: v.hi,
                            z: v.z,
                            outcome: trace_outcome,
                            rescue,
                            method: v.method.clone(),
                            sds_sd: s0,
                        });
                    }
                    self.ts_rev = self.ts_rev.saturating_add(1);

                    if live_fb {
                        self.books.insert(fkey, work_book);
                    }

                    gates.push(GateResult {
                        gate: "temporal",
                        passed: !held,
                        applied: v.outcome != TemporalOutcome::Abstained,
                        severity: if held { v.severity.max(1) } else if v.outcome == TemporalOutcome::Warn { v.severity } else { 0 },
                        detail,
                    });
                    temporal = Some(v);
                } else {
                    if plausible {
                        if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                            if l.trace.len() >= HISTORY_CAP {
                                l.trace.pop_front();
                            }
                            l.trace.push_back(GateMark {
                                tick,
                                observed: c.score as f64,
                                expected: None,
                                lo: None,
                                hi: None,
                                z: None,
                                outcome: "abstained",
                                rescue: None,
                                method: String::new(),
                                sds_sd: None,
                            });
                        }
                        self.ts_rev = self.ts_rev.saturating_add(1);
                    }
                    gates.push(GateResult {
                        gate: "temporal",
                        passed: true,
                        applied: false,
                        severity: 0,
                        detail: format!("이력 {}점 (판정에 {need}점 필요)", hist.len()),
                    });
                }
            }
            None => gates.push(GateResult { gate: "temporal", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 건너뜀".into() }),
        }

        if let Some((x, z)) = c.world {
            let close_inspection = c.depth_m <= 1.6 && c.margin >= self.cfg.min_margin * 1.5;
            let rival = self
                .landmarks
                .iter()
                .filter(|l| l.state == LmState::Active && l.kind == c.kind && l.label != c.label && l.good >= 1 && l.distance_to(x, z) <= CONFLICT_RADIUS_M)
                .filter(|l| !(close_inspection && !l.confirmed && l.last_tick < tick))
                .max_by(|a, b| a.score.total_cmp(&b.score));
            match rival {
                Some(r) if r.score >= c.score => gates.push(GateResult {
                    gate: "conflict",
                    passed: false,
                    applied: true,
                    severity: severity((r.score - c.score) as f64, 0.5),
                    detail: format!("같은 자리에 '{}'(점수 {:.2})가 더 확실함", r.label, r.score),
                }),
                _ => gates.push(GateResult { gate: "conflict", passed: true, applied: true, severity: 0, detail: "겹치는 다른 라벨 없음".into() }),
            }
        } else {
            gates.push(GateResult { gate: "conflict", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 건너뜀".into() });
        }

        let ruled_out = state == Some(LmState::Rejected);
        gates.push(GateResult {
            gate: "touch",
            passed: !ruled_out,
            applied: tid.is_some(),
            severity: if ruled_out { 100 } else { 0 },
            detail: if ruled_out { "이 자리는 접촉으로 아니라고 확인됨".into() } else { "접촉으로 배제된 적 없음".into() },
        });

        let base_ok = gates.iter().all(|g| g.passed);
        let temporal_ok = gates.iter().find(|g| g.gate == "temporal").map_or(true, |g| g.passed);
        let record_ok = plausible && temporal_ok;

        if let Some(id) = tid {
            let (x, z) = c.world.unwrap_or((0.0, 0.0));
            let sigma_obs = mem.sigma_obs(c.depth_m);
            if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                let first = l.seen == 0;
                l.seen += 1;
                if record_ok {
                    l.history.push(c.score as f64);
                    l.depths.push(c.depth_m);
                    if l.history.len() > HISTORY_CAP {
                        l.history.remove(0);
                    }
                    if l.depths.len() > HISTORY_CAP {
                        l.depths.remove(0);
                    }
                    l.score = if first { c.score } else { 0.5 * l.score + 0.5 * c.score };
                    l.margin = if first { c.margin } else { 0.5 * l.margin + 0.5 * c.margin };
                    if first || l.sigma_m <= 0.0 {
                        l.x = x;
                        l.z = z;
                        l.sigma_m = sigma_obs;
                    } else {
                        let before = l.memory();
                        let travelled = mem.odo_m - before.odo_m;
                        let predicted = before.sigma_m(mem.odo_m, mem.rate);
                        let (fused, innov) = before.fuse(x, z, sigma_obs, mem.odo_m, tick, mem.rate);
                        record_memory(sds, mem, &c.label, travelled, innov, predicted, before.sigma0_m, sigma_obs);
                        l.x = fused.x;
                        l.z = fused.z;
                        l.sigma_m = fused.sigma0_m.max(0.5 * sigma_obs);
                    }
                    l.odo_m = mem.odo_m;
                }
                l.last_tick = tick;
                if base_ok {
                    l.good += 1;
                }
            }
            if record_ok {
                let s = self.series.entry(c.label.clone()).or_default();
                s.push(c.score as f64);
                if s.len() > SERIES_CAP {
                    s.remove(0);
                }
                let m = self.margins.entry(c.label.clone()).or_default();
                m.push(c.margin as f64);
                if m.len() > SERIES_CAP {
                    m.remove(0);
                }
                self.ts_rev = self.ts_rev.saturating_add(1);
            }
        }

        let need = self.cfg.confirm_sightings;
        match tid.and_then(|id| self.landmark(id)) {
            Some(l) => {
                let ok = l.good >= need;
                gates.push(GateResult {
                    gate: "confirm",
                    passed: ok,
                    applied: true,
                    severity: if ok { 0 } else { severity((need - l.good.min(need)) as f64, need as f64) },
                    detail: format!("통과한 관측 {}/{need}", l.good),
                });
            }
            None => gates.push(GateResult { gate: "confirm", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 확인 불가".into() }),
        }

        let by_gates = gates.iter().all(|g| g.passed) && tid.is_some();
        let accepted = if self.cfg.enabled { by_gates } else { tid.is_some() };
        if accepted {
            if let Some(l) = tid.and_then(|id| self.landmarks.iter_mut().find(|l| l.id == id)) {
                l.confirmed = true;
            }
        }

        self.counters.evaluated += 1;
        if accepted {
            self.counters.accepted += 1;
        }
        let failed = gates.iter().filter(|g| g.applied && !g.passed).count();
        if let Some(first) = gates.iter().find(|g| g.applied && !g.passed) {
            *self.counters.blocked_by.entry(first.gate.to_string()).or_insert(0) += 1;
        }
        let items: Vec<(String, u8, bool, bool)> =
            gates.iter().filter(|g| g.applied).map(|g| (g.gate.to_string(), g.severity, !g.passed, !g.passed && failed == 1)).collect();
        sds.record_gates(&vscope, &items);

        Judged { cand: c, accepted, gates, landmark: tid, floor, floor_source, temporal }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn cand(label: &str, score: f32, margin: f32, x: f32, z: f32) -> Candidate {
        let (kind, link) = match label {
            "tree" => (EntKind::Source, "log"),
            "boulder" => (EntKind::Source, "stone"),
            "coal_ore" => (EntKind::Source, "coal"),
            "log" => (EntKind::Drop, "log"),
            "warehouse" => (EntKind::Waypoint, "warehouse"),
            _ => (EntKind::Source, "x"),
        };
        Candidate {
            label: label.into(),
            kind,
            link: link.into(),
            score,
            margin,
            runner_up: None,
            bbox: [0.4, 0.4, 0.6, 0.6],
            pixels: 100,
            depth_m: 3.0,
            world: Some((x, z)),
            truth: None,
            truth_label: None,
        }
    }

    fn judge1(g: &mut PrecisionGates, sds: &mut SdsStore, c: Candidate, tick: u64) -> Judged {
        g.judge(vec![c], tick, sds).pop().unwrap()
    }

    #[test]
    fn a_landmark_needs_two_good_sightings() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let a = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1);
        assert!(!a.accepted);
        assert_eq!(a.blocked_by(), Some("confirm"));
        let b = judge1(&mut g, &mut sds, cand("tree", 0.82, 0.2, 5.1, 5.0), 3);
        assert!(b.accepted, "{:?}", b.gates);
        assert_eq!(g.landmarks().len(), 1, "both sightings are one landmark");
        let lm = &g.landmarks()[0];
        assert!(lm.confirmed && lm.good == 2);
        assert!((lm.x - 5.05).abs() < 0.01, "position is averaged: {}", lm.x);
    }

    #[test]
    fn low_score_and_low_margin_are_blocked_and_do_not_confirm() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..4 {
            let low = judge1(&mut g, &mut sds, cand("tree", 0.2, 0.2, 5.0, 5.0), t);
            assert_eq!(low.blocked_by(), Some("score"));
            assert!(!low.accepted);
        }
        for t in 4..8 {
            let thin = judge1(&mut g, &mut sds, cand("boulder", 0.7, 0.01, 8.0, 5.0), t);
            assert_eq!(thin.blocked_by(), Some("margin"));
        }
        assert!(g.landmarks().iter().all(|l| !l.confirmed && l.good == 0));
        assert!(g.series("tree").is_empty(), "implausible scores do not enter the forecast series");
    }

    #[test]
    fn the_sds_can_raise_the_floor_but_never_lower_it() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        assert_eq!(g.floor_for("tree", &sds).0, 0.30);
        let scope = Scope::new(Track::Vision, "tree", "");
        for i in 0..40 {
            sds.record_baseline(&scope, "score", 0.85 + 0.01 * ((i % 3) as f64 - 1.0));
        }
        let (floor, src) = g.floor_for("tree", &sds);
        assert!((floor - 0.55).abs() < 1e-6, "{floor}");
        assert!(src.contains("SDS"), "{src}");
        let weak = judge1(&mut g, &mut sds, cand("tree", 0.50, 0.2, 5.0, 5.0), 1);
        assert_eq!(weak.blocked_by(), Some("score"), "0.50 passes the fixed floor but not the learned one");
        let fine = judge1(&mut g, &mut sds, cand("tree", 0.70, 0.2, 5.0, 5.0), 2);
        assert!(fine.gates.iter().find(|x| x.gate == "score").unwrap().passed);
        let mut sds2 = SdsStore::memory();
        for i in 0..40 {
            sds2.record_baseline(&scope, "score", 0.20 + 0.02 * ((i % 3) as f64 - 1.0));
        }
        assert_eq!(g.floor_for("tree", &sds2).0, 0.30);
        g.set_cfg(GateCfg { adaptive: false, ..GateCfg::default() });
        assert_eq!(g.floor_for("tree", &sds).0, 0.30);
    }

    #[test]
    fn a_low_touch_agreement_rate_tightens_the_floor() {
        let g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let scope = Scope::new(Track::Vision, "boulder", "");
        for i in 0..20 {
            sds.record_agreement(&scope, "touch", i % 2 == 0);
        }
        let (floor, src) = g.floor_for("boulder", &sds);
        assert!(floor > 0.30 + 0.1, "agreement 50% should add (0.8-0.5)*0.5=0.15, got {floor}");
        assert!(src.contains("접촉 일치율"), "{src}");
    }

    #[test]
    fn the_temporal_gate_holds_a_jump_once_there_is_history() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            let s = 0.80 + 0.02 * ((t % 3) as f32 - 1.0);
            let j = judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
            let temporal = j.gates.iter().find(|x| x.gate == "temporal").unwrap();
            if t < 6 {
                assert!(!temporal.applied, "no history yet at {t}");
            }
        }
        let jump = judge1(&mut g, &mut sds, cand("tree", 0.40, 0.2, 5.0, 5.0), 11);
        let t = jump.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(t.applied && !t.passed, "{:?}", jump.gates);
        assert!(!jump.accepted);
        assert!(jump.temporal.as_ref().unwrap().expected.unwrap() > 0.7);
        let steady = judge1(&mut g, &mut sds, cand("tree", 0.81, 0.2, 5.0, 5.0), 12);
        assert!(steady.accepted, "{:?}", steady.gates);
        assert!(g.landmark(steady.landmark.unwrap()).unwrap().history.iter().all(|v| *v > 0.7), "the held score did not enter the history");
        assert!(g.series("tree").iter().all(|v| *v > 0.7), "nor the label series");
    }

    #[test]
    fn three_agreeing_held_scores_count_as_a_level_change() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            judge1(&mut g, &mut sds, cand("tree", 0.80 + 0.02 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0), t);
        }
        let a = judge1(&mut g, &mut sds, cand("tree", 0.45, 0.2, 5.0, 5.0), 20);
        let b = judge1(&mut g, &mut sds, cand("tree", 0.46, 0.2, 5.0, 5.0), 21);
        assert!(!a.accepted && !b.accepted, "held at first");
        let c = judge1(&mut g, &mut sds, cand("tree", 0.44, 0.2, 5.0, 5.0), 22);
        let tg = c.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(tg.passed && tg.detail.contains("수준 변화"), "{:?}", tg);
        let lm = g.landmark(c.landmark.unwrap()).unwrap();
        assert!(
            lm.history.len() >= g.cfg().temporal_min_history,
            "level change shifts prior history in place so the gate stays ON: len={}",
            lm.history.len()
        );
    }

    #[test]
    fn realized_errors_are_scored_on_the_same_call() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..6u64 {
            judge1(&mut g, &mut sds, cand("tree", 0.80 + 0.01 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0), t);
        }
        let fscope = Scope::new(Track::Forecast, "tree", "score@oracle");
        assert!(sds.exact_baseline(&fscope, "score:res2:last").is_none(), "no gate call yet before 6 points");
        let j = judge1(&mut g, &mut sds, cand("tree", 0.82, 0.2, 5.0, 5.0), 6);
        assert!(j.temporal.is_some());
        let (_, _, n_last) = sds.exact_baseline(&fscope, "score:res2:last").expect("realized res2 recorded on same call");
        assert_eq!(n_last, 1);
        let (mse_fused, _, n_fused) = sds.exact_baseline(&fscope, "score:res2:fused").expect("fused res2 recorded");
        assert_eq!(n_fused, 1);
        let exp = j.temporal.as_ref().unwrap().expected.unwrap();
        assert!((mse_fused - (0.82 - exp).powi(2)).abs() < 1e-6, "got {mse_fused} vs {}", (0.82 - exp).powi(2));
    }

    #[test]
    fn held_scores_still_enter_the_error_ledger() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            let s = 0.80 + 0.02 * ((t % 3) as f32 - 1.0);
            judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
        }
        let fscope = Scope::new(Track::Forecast, "tree", "score@oracle");
        let before_n = sds.exact_baseline(&fscope, "score:res2:fused").map_or(0, |(_, _, n)| n);
        let jump = judge1(&mut g, &mut sds, cand("tree", 0.40, 0.2, 5.0, 5.0), 11);
        assert!(!jump.accepted && jump.blocked_by() == Some("temporal"));
        let after_n = sds.exact_baseline(&fscope, "score:res2:fused").map_or(0, |(_, _, n)| n);
        assert_eq!(after_n, before_n + 1, "held score must still be recorded in score:res2:fused without censorship");
        assert!(!g.landmark(jump.landmark.unwrap()).unwrap().shadow.is_empty());
    }

    #[test]
    fn a_level_change_keeps_the_gate_on() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            judge1(&mut g, &mut sds, cand("tree", 0.80 + 0.015 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0), t);
        }
        let _ = judge1(&mut g, &mut sds, cand("tree", 0.45, 0.2, 5.0, 5.0), 20);
        let _ = judge1(&mut g, &mut sds, cand("tree", 0.46, 0.2, 5.0, 5.0), 21);
        let shift = judge1(&mut g, &mut sds, cand("tree", 0.44, 0.2, 5.0, 5.0), 22);
        assert!(shift.gates.iter().find(|x| x.gate == "temporal").unwrap().passed);
        let next = judge1(&mut g, &mut sds, cand("tree", 0.45, 0.2, 5.0, 5.0), 23);
        let tg = next.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(tg.applied && tg.passed, "gate must remain ON immediately after level change: {tg:?}");
    }

    #[test]
    fn a_jump_needs_a_depth_reason() {
        // Case 1: depth unchanged -> unexplained positive jump is HELD.
        let mut g1 = PrecisionGates::new(GateCfg::default());
        let mut sds1 = SdsStore::memory();
        for t in 0..10u64 {
            let mut c = cand("tree", 0.65 + 0.01 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0);
            c.depth_m = 4.0;
            judge1(&mut g1, &mut sds1, c, t);
        }
        let mut same_d = cand("tree", 0.88, 0.2, 5.0, 5.0);
        same_d.depth_m = 4.0;
        let held = judge1(&mut g1, &mut sds1, same_d, 11);
        let tg1 = held.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(tg1.applied && !tg1.passed, "positive jump with no depth change must be held: {tg1:?}");

        // Case 2: depth decreases (approaching) -> positive jump is ACCEPTED (both cold start & learned slope).
        let mut g2 = PrecisionGates::new(GateCfg::default());
        let mut sds2 = SdsStore::memory();
        for t in 0..10u64 {
            let mut c = cand("tree", 0.65 + 0.01 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0);
            c.depth_m = 4.0;
            judge1(&mut g2, &mut sds2, c, t);
        }
        let mut closer = cand("tree", 0.88, 0.2, 5.0, 5.0);
        closer.depth_m = 2.0;
        let ok = judge1(&mut g2, &mut sds2, closer, 11);
        let tg2 = ok.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(tg2.applied && tg2.passed, "positive jump accompanied by closer depth must be accepted: {tg2:?}");
    }

    #[test]
    fn forecast_errors_flow_into_the_sds() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..40 {
            let s = 0.80 + 0.02 * (((t * 7) % 5) as f32 - 2.0) / 2.0;
            judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
        }
        let past = past_errors(&sds, "tree");
        assert!(!past.is_empty(), "after many judged sightings the SDS remembers method errors");
        assert!(past.iter().all(|h| h.mse.is_finite() && h.mse >= 0.0 && h.n > 0));
        let f = g.forecast_for("tree", 3, &sds).unwrap();
        assert_eq!(f.point.len(), 3);
        assert!(f.point[0] > 0.7 && f.point[0] < 0.9, "{:?}", f.point);
        assert!(f.p10[0] <= f.point[0] && f.point[0] <= f.p90[0]);
    }

    #[test]
    fn a_weaker_label_on_the_same_spot_is_blocked_as_a_conflict() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("boulder", 0.85, 0.2, 6.0, 6.0), 1);
        judge1(&mut g, &mut sds, cand("boulder", 0.86, 0.2, 6.0, 6.0), 2);
        let alt = judge1(&mut g, &mut sds, cand("coal_ore", 0.60, 0.2, 6.1, 6.0), 3);
        assert_eq!(alt.blocked_by(), Some("conflict"), "{:?}", alt.gates);
        let far = judge1(&mut g, &mut sds, cand("coal_ore", 0.60, 0.2, 9.0, 6.0), 4);
        assert!(far.gates.iter().find(|x| x.gate == "conflict").unwrap().passed);
        let strong = judge1(&mut g, &mut sds, cand("coal_ore", 0.95, 0.3, 6.1, 6.0), 5);
        assert!(strong.gates.iter().find(|x| x.gate == "conflict").unwrap().passed);
    }

    #[test]
    fn a_rejected_spot_blocks_the_same_label_but_not_other_labels() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1);
        let id = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 2).landmark.unwrap();
        g.reject(id);
        let again = judge1(&mut g, &mut sds, cand("tree", 0.9, 0.3, 5.1, 5.0), 3);
        assert_eq!(again.blocked_by(), Some("touch"));
        assert!(!again.accepted);
        let other = judge1(&mut g, &mut sds, cand("boulder", 0.9, 0.3, 5.1, 5.0), 4);
        assert!(other.gates.iter().find(|x| x.gate == "touch").unwrap().passed);
        g.consume(id);
        assert!(g.landmark(id).is_none());
    }

    #[test]
    fn two_failed_approaches_rule_a_landmark_out() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let id = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1).landmark.unwrap();
        g.note_unreachable(id);
        assert_eq!(g.landmark(id).unwrap().state, LmState::Active);
        g.note_unreachable(id);
        assert_eq!(g.landmark(id).unwrap().state, LmState::Rejected);
    }

    #[test]
    fn with_the_gates_off_everything_with_a_position_is_accepted() {
        let mut g = PrecisionGates::new(GateCfg { enabled: false, ..GateCfg::default() });
        let mut sds = SdsStore::memory();
        let j = judge1(&mut g, &mut sds, cand("tree", 0.05, 0.0, 5.0, 5.0), 1);
        assert!(j.accepted);
        assert!(g.landmarks()[0].confirmed);
        let mut c = cand("tree", 0.9, 0.3, 0.0, 0.0);
        c.world = None;
        assert!(!judge1(&mut g, &mut sds, c, 2).accepted, "without a position there is nothing to act on");
    }

    #[test]
    fn unconfirmed_phantoms_expire_but_confirmed_landmarks_stay() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1);
        judge1(&mut g, &mut sds, cand("boulder", 0.8, 0.2, 9.0, 5.0), 1);
        judge1(&mut g, &mut sds, cand("boulder", 0.8, 0.2, 9.0, 5.0), 2);
        assert_eq!(g.landmarks().len(), 2);
        judge1(&mut g, &mut sds, cand("log", 0.8, 0.2, 2.0, 2.0), 2 + STALE_TICKS + 5);
        let labels: Vec<&str> = g.landmarks().iter().map(|l| l.label.as_str()).collect();
        assert!(labels.contains(&"boulder") && !labels.contains(&"tree"), "{labels:?}");
    }

    #[test]
    fn every_applied_gate_is_recorded_in_the_sds() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..6 {
            judge1(&mut g, &mut sds, cand("tree", if t % 2 == 0 { 0.8 } else { 0.1 }, 0.2, 5.0, 5.0), t);
        }
        let rates = sds.gate_rates(&Scope::new(Track::Vision, "tree", ""));
        let score = rates.iter().find(|r| r.gate == "score").expect("score gate recorded");
        assert_eq!(score.evaluated, 6);
        assert!((score.blocked_rate - 0.5).abs() < 1e-9, "{}", score.blocked_rate);
        assert!(g.counters.evaluated == 6 && g.counters.blocked_by.contains_key("score"));
    }

    #[test]
    fn gate_trace_equals_gate_input() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let mut lm_id = 0;
        for t in 0..10u64 {
            let s = 0.80 + 0.02 * ((t % 3) as f32 - 1.0);
            let j = judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
            lm_id = j.landmark.unwrap();
        }
        // Single isolated jump is held and NOT in history.
        let held1 = judge1(&mut g, &mut sds, cand("tree", 0.40, 0.2, 5.0, 5.0), 11);
        assert_eq!(held1.blocked_by(), Some("temporal"));
        let lm_before = g.landmark(lm_id).unwrap();
        let entered_before: Vec<f64> = lm_before
            .trace
            .iter()
            .filter(|m| m.outcome != "held")
            .map(|m| m.observed)
            .collect();
        assert_eq!(entered_before, lm_before.history, "accepted+abstained trace points must match landmark history exactly");
        assert_eq!(lm_before.trace.iter().filter(|m| m.outcome == "abstained").count(), g.cfg().temporal_min_history);
        assert_eq!(lm_before.trace.back().unwrap().outcome, "held");

        // Now verify level-shift rescue aligns with three_agreeing_held_scores_count_as_a_level_change.
        let mut g2 = PrecisionGates::new(GateCfg::default());
        let mut sds2 = SdsStore::memory();
        for t in 0..10u64 {
            judge1(&mut g2, &mut sds2, cand("tree", 0.80 + 0.02 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0), t);
        }
        let _ = judge1(&mut g2, &mut sds2, cand("tree", 0.45, 0.2, 5.0, 5.0), 20);
        let _ = judge1(&mut g2, &mut sds2, cand("tree", 0.46, 0.2, 5.0, 5.0), 21);
        let c = judge1(&mut g2, &mut sds2, cand("tree", 0.44, 0.2, 5.0, 5.0), 22);
        let lm2 = g2.landmark(c.landmark.unwrap()).unwrap();
        let last = lm2.trace.back().unwrap();
        assert_eq!(last.outcome, "accepted");
        assert_eq!(last.rescue, Some("level_shift"));
        assert_eq!(lm2.trace.iter().filter(|m| m.outcome != "held").count(), lm2.history.len());
        assert!(g2.ts_rev() > 0);
    }
}
