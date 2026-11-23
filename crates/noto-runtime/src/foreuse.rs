//! Forecast-use registry, Group-A-only limit types, trust state machine, monotonic cap tightening,
//! and post-hoc warning Brier audit (Phase 3 – Part 3 · D1 & D6).
//!
//! This module is listed in `DECISION_FILES` and `ENGINE_FILES`: it strictly uses Group A robot
//! geometry/ratings and never reads human reference benchmarks or N:N evaluation scopes.

use std::collections::{BTreeMap, VecDeque};

use noto_core::balance::REACTION_DELAY_S;
use noto_sds::{Scope, SdsStore, Track};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// How a forecast use interacts with robot behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grade {
    /// Visualization and logging only.
    Display,
    /// Surfaces recommendations in quest log / HUD without modifying control.
    Advise,
    /// Modifies control caps strictly under [`Trust::Calibrated`] and monotonic tightening.
    Control,
}

/// Group-A-only physical/geometric limits. Human references cannot be represented here by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    /// Support polygon boundary (`MARGIN_KEEP`).
    SupportEdge,
    /// Static friction utilization limit (`SAFE_SHARE` = 0.85 for control, 1.0 physical slip).
    StaticFriction,
    /// Actuator peak torque ratio (1.0).
    JointPeak,
    /// Actuator continuous thermal rating (1.0).
    ThermalCont,
    /// Cumulative Miner wear life (1.0).
    WearLife,
    /// Half of target item's short dimension (memory pickup tolerance).
    ItemTolerance,
    /// Passage / step clearance margin along route.
    PassageClearance,
    /// Detector temporal consistency gate (calibrated score/position band).
    TemporalBand,
}

impl Limit {
    pub fn as_str(self) -> &'static str {
        match self {
            Limit::SupportEdge => "support_edge",
            Limit::StaticFriction => "static_friction",
            Limit::JointPeak => "joint_peak",
            Limit::ThermalCont => "thermal_cont",
            Limit::WearLife => "wear_life",
            Limit::ItemTolerance => "item_tolerance",
            Limit::PassageClearance => "passage_clearance",
            Limit::TemporalBand => "temporal_band",
        }
    }

    pub fn is_group_a(self) -> bool {
        true
    }
}

/// Calibration trust state per `(use_id, scope)`. Control uses act ONLY in [`Trust::Calibrated`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    #[default]
    Cold,
    Warm,
    Calibrated,
    Suspended,
}

/// Physical context needed to compute `h_act` (lead time needed for a corrective action to take effect).
#[derive(Clone, Copy, Debug, Default)]
pub struct UseCtx {
    pub speed_mps: f32,
    pub accel_limit: f32,
    pub crouch_rem_m: f32,
    pub crouch_rate_mps: f32,
    pub look_s: f32,
    pub remaining_s: f32,
}

pub struct ForecastUse {
    pub id: &'static str,
    pub stage: &'static str,
    pub series: &'static str,
    pub rising: bool,
    pub limit: Limit,
    pub grade: Grade,
    pub act_s: fn(&UseCtx) -> f32,
    pub fallback: &'static str,
}

fn act_immediate(_: &UseCtx) -> f32 {
    REACTION_DELAY_S
}

fn act_look(c: &UseCtx) -> f32 {
    c.look_s.max(0.25) + REACTION_DELAY_S
}

fn act_decel(c: &UseCtx) -> f32 {
    (c.speed_mps / c.accel_limit.max(0.3)) + REACTION_DELAY_S
}

fn act_crouch(c: &UseCtx) -> f32 {
    let rate = c.crouch_rate_mps.max(0.05);
    (c.crouch_rem_m.max(0.0) / rate) + REACTION_DELAY_S
}

fn act_remaining(c: &UseCtx) -> f32 {
    c.remaining_s.max(1.0)
}

/// Static registry of all forecast uses across task stages (Phase 3 – Part 3 · D1).
pub const USES: &[ForecastUse] = &[
    ForecastUse {
        id: "vision_score_gate",
        stage: "approach",
        series: "vision.score",
        rising: false,
        limit: Limit::TemporalBand,
        grade: Grade::Control,
        act_s: act_immediate,
        fallback: "이력 6점 미만 기권, 기존 점수·여유폭 게이트 유지",
    },
    ForecastUse {
        id: "decide_pos_gate",
        stage: "approach",
        series: "decide.fix_x",
        rising: true,
        limit: Limit::TemporalBand,
        grade: Grade::Control,
        act_s: act_immediate,
        fallback: "측량 2점 미만이면 삼각 측량 값 그대로 사용",
    },
    ForecastUse {
        id: "memory_relook_ahead",
        stage: "sight",
        series: "nav.odo_sigma",
        rising: true,
        limit: Limit::ItemTolerance,
        grade: Grade::Control,
        act_s: act_look,
        fallback: "기존 memory_ok + 15틱 주기 재응시 유지",
    },
    ForecastUse {
        id: "support_margin_ahead",
        stage: "grasp",
        series: "bal.margin",
        rising: false,
        limit: Limit::SupportEdge,
        grade: Grade::Advise,
        act_s: act_immediate,
        fallback: "표시·권고 전용 (제어 불변)",
    },
    ForecastUse {
        id: "grip_slip_ahead",
        stage: "grasp",
        series: "ev.cargo_slip",
        rising: true,
        limit: Limit::StaticFriction,
        grade: Grade::Advise,
        act_s: act_immediate,
        fallback: "권고 전용 (쥠 재확인 알림)",
    },
    ForecastUse {
        id: "pre_carry_reacquire",
        stage: "reacquire",
        series: "nav.odo_sigma",
        rising: true,
        limit: Limit::PassageClearance,
        grade: Grade::Control,
        act_s: act_look,
        fallback: "출발 전 기본 스캔 규칙만 유지 (추가만 가능)",
    },
    ForecastUse {
        id: "cargo_slip_ahead",
        stage: "move",
        series: "cargo.use_ratio",
        rising: true,
        limit: Limit::StaticFriction,
        grade: Grade::Control,
        act_s: act_decel,
        fallback: "기존 화물 안전 가속 상한 그대로",
    },
    ForecastUse {
        id: "step_crouch_ahead",
        stage: "step",
        series: "bal.margin",
        rising: false,
        limit: Limit::PassageClearance,
        grade: Grade::Control,
        act_s: act_crouch,
        fallback: "기존 한 보폭 앞 STEP_SPEED_MPS 제한 유지",
    },
    ForecastUse {
        id: "brake_stop_ahead",
        stage: "brake",
        series: "cargo.use_ratio",
        rising: true,
        limit: Limit::StaticFriction,
        grade: Grade::Control,
        act_s: act_decel,
        fallback: "기존 계획 감속·화물 상한 제동 유지",
    },
    ForecastUse {
        id: "thermal_wear_rest",
        stage: "all",
        series: "joint.heat",
        rising: true,
        limit: Limit::ThermalCont,
        grade: Grade::Advise,
        act_s: act_remaining,
        fallback: "기존 열 채굴 감속 유지, 휴식 권고만 출력",
    },
];

pub fn find_use(id: &str) -> Option<&'static ForecastUse> {
    USES.iter().find(|u| u.id == id)
}

/// Monotonic safety cap tightening: forecasts may ONLY lower speed/accel caps, and ONLY when Calibrated.
pub fn tighten_cap(base: f32, from_forecast: Option<f32>, trust: Trust) -> f32 {
    match (trust, from_forecast) {
        (Trust::Calibrated, Some(c)) if c.is_finite() && c > 0.0 => base.min(c),
        _ => base,
    }
}

/// Monotonic earlier look trigger: forecasts may ONLY make a re-look happen earlier, never later.
pub fn tighten_look_dist(base_dist: f32, forecast_dist: Option<f32>, trust: Trust) -> f32 {
    match (trust, forecast_dist) {
        (Trust::Calibrated, Some(d)) if d.is_finite() && d >= 0.0 => base_dist.min(d),
        _ => base_dist,
    }
}

/// Monotonic boolean scan guard: forecasts may ONLY OR-in an additional safety scan, never suppress one.
pub fn tighten_scan_needed(base_needed: bool, forecast_needed: bool, trust: Trust) -> bool {
    match trust {
        Trust::Calibrated => base_needed || forecast_needed,
        _ => base_needed,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WarnResolution {
    pub crossed: bool,
    pub at_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForecastWarn {
    pub kind: &'static str,
    #[serde(rename = "use")]
    pub use_id: String,
    pub tick: u64,
    pub time_s: f64,
    pub stage: String,
    pub worker: u32,
    pub grade: Grade,
    pub trust: Trust,
    pub series: String,
    pub limit_kind: String,
    pub limit_value: f64,
    pub t_cross_s: Option<f64>,
    pub act_s: f32,
    pub p: f64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<Value>,
    pub resolved: Option<WarnResolution>,
    #[serde(skip)]
    pub expire_tick: u64,
}

#[derive(Clone, Debug, Default)]
struct UseStats {
    trust: Trust,
    issued: u64,
    resolved: u64,
    hits: u64,
    brier_sum: f64,
    base_pos: u64,
    consecutive_misses: u32,
    recent: VecDeque<ForecastWarn>,
}

impl UseStats {
    const RECENT_CAP: usize = 10;

    fn brier(&self) -> Option<f64> {
        (self.resolved > 0).then(|| self.brier_sum / self.resolved as f64)
    }

    fn base_rate_brier(&self) -> Option<f64> {
        if self.resolved == 0 {
            return None;
        }
        let p_bar = self.base_pos as f64 / self.resolved as f64;
        Some(p_bar * (1.0 - p_bar))
    }

    fn hit_rate(&self) -> Option<f64> {
        (self.resolved > 0).then(|| self.hits as f64 / self.resolved as f64)
    }
}

/// Runtime tracker of forecast uses, trust states, active/resolved warnings, and Brier audit.
#[derive(Clone, Debug, Default)]
pub struct ForeUseHub {
    stats: BTreeMap<String, UseStats>,
    pending: Vec<ForecastWarn>,
}

impl ForeUseHub {
    pub fn trust(&self, use_id: &str) -> Trust {
        self.stats.get(use_id).map_or(Trust::Cold, |s| s.trust)
    }

    pub fn set_trust(&mut self, use_id: &str, trust: Trust) {
        self.stats.entry(use_id.to_string()).or_default().trust = trust;
    }

    pub fn issue(&mut self, mut warn: ForecastWarn, horizon_ticks: u64) {
        // Deduplicate active unresolved warning for the same use + worker.
        if self.pending.iter().any(|w| w.use_id == warn.use_id && w.worker == warn.worker && w.resolved.is_none()) {
            return;
        }
        warn.expire_tick = warn.tick.saturating_add(horizon_ticks.max(1));
        let st = self.stats.entry(warn.use_id.clone()).or_default();
        st.issued += 1;
        if st.trust == Trust::Cold && st.issued >= 3 {
            st.trust = Trust::Warm;
        }
        st.recent.push_back(warn.clone());
        while st.recent.len() > UseStats::RECENT_CAP {
            st.recent.pop_front();
        }
        self.pending.push(warn);
    }

    /// Record an observed limit crossing (or horizon expiry) for Brier audit & trust update.
    pub fn observe_outcome(
        &mut self,
        use_id: &str,
        worker: u32,
        crossed: bool,
        time_s: f64,
        sds: &mut SdsStore,
    ) {
        let mut matched = false;
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].use_id == use_id && self.pending[idx].worker == worker {
                let mut w = self.pending.remove(idx);
                w.resolved = Some(WarnResolution { crossed, at_s: time_s });
                self.score_resolved(&w, crossed, sds);
                matched = true;
            } else {
                idx += 1;
            }
        }
        // Unwarned crossing (false negative): score as p = 0.0 forecast for recall & Brier honesty.
        if crossed && !matched {
            let st = self.stats.entry(use_id.to_string()).or_default();
            st.resolved += 1;
            st.base_pos += 1;
            st.brier_sum += 1.0; // (0.0 - 1.0)^2
            st.consecutive_misses += 1;
            if st.consecutive_misses >= 4 && st.trust == Trust::Calibrated {
                st.trust = Trust::Suspended;
            }
            let scope = Scope::new(Track::Forecast, use_id, "warn");
            sds.record_baseline(&scope, "brier", 1.0);
            sds.record_agreement(&scope, "hit", false);
        }
    }

    /// Expire warnings whose horizon passed without crossing (`crossed = false`).
    pub fn expire(&mut self, tick: u64, time_s: f64, sds: &mut SdsStore) {
        let mut idx = 0;
        while idx < self.pending.len() {
            if tick >= self.pending[idx].expire_tick {
                let mut w = self.pending.remove(idx);
                w.resolved = Some(WarnResolution { crossed: false, at_s: time_s });
                self.score_resolved(&w, false, sds);
            } else {
                idx += 1;
            }
        }
    }

    fn score_resolved(&mut self, w: &ForecastWarn, crossed: bool, sds: &mut SdsStore) {
        let y = if crossed { 1.0 } else { 0.0 };
        let p = w.p.clamp(0.0, 1.0);
        let brier = (p - y).powi(2);
        let st = self.stats.entry(w.use_id.clone()).or_default();
        st.resolved += 1;
        if crossed {
            st.hits += 1;
            st.base_pos += 1;
            st.consecutive_misses = 0;
        } else {
            st.consecutive_misses += 1;
        }
        st.brier_sum += brier;
        if let Some(last) = st.recent.iter_mut().rev().find(|r| r.tick == w.tick && r.resolved.is_none()) {
            last.resolved = w.resolved.clone();
        }
        // Promote Warm -> Calibrated once >= 10 resolved warnings beat base-rate Brier; suspend on severe drift.
        if st.resolved >= 10 {
            if let (Some(b), Some(bb)) = (st.brier(), st.base_rate_brier()) {
                if b <= bb + 1e-6 && st.consecutive_misses < 5 {
                    st.trust = Trust::Calibrated;
                } else if b > bb + 0.10 {
                    st.trust = Trust::Suspended;
                }
            }
        }
        let scope = Scope::new(Track::Forecast, &w.use_id, "warn");
        sds.record_baseline(&scope, "brier", brier);
        sds.record_agreement(&scope, "hit", crossed);
    }

    pub fn active_warnings(&self) -> &[ForecastWarn] {
        &self.pending
    }

    pub fn status_json(&self) -> Value {
        let items: Vec<Value> = USES
            .iter()
            .map(|u| {
                let st = self.stats.get(u.id);
                json!({
                    "id": u.id,
                    "stage": u.stage,
                    "series": u.series,
                    "rising": u.rising,
                    "limit": u.limit.as_str(),
                    "grade": u.grade,
                    "fallback": u.fallback,
                    "trust": st.map_or(Trust::Cold, |s| s.trust),
                    "issued": st.map_or(0, |s| s.issued),
                    "resolved": st.map_or(0, |s| s.resolved),
                    "hit_rate": st.and_then(|s| s.hit_rate()),
                    "brier": st.and_then(|s| s.brier()),
                    "base_brier": st.and_then(|s| s.base_rate_brier()),
                    "recent": st.map(|s| s.recent.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
                })
            })
            .collect();
        json!({
            "uses": items,
            "active": self.pending,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_registry_has_no_reference_limits() {
        assert!(USES.len() >= 10);
        for u in USES {
            assert!(u.limit.is_group_a(), "use {} must have a Group A limit", u.id);
            assert!(!u.series.starts_with("nn."), "use {} must not read display-only nn.* channel", u.id);
        }
    }

    #[test]
    fn control_uses_only_tighten() {
        let mut seed = 0x5EED_1234u64;
        for _ in 0..500 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = 0.2 + ((seed >> 32) as f32 / u32::MAX as f32) * 3.0;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let cand = ((seed >> 32) as f32 / u32::MAX as f32) * 5.0;
            for trust in [Trust::Cold, Trust::Warm, Trust::Calibrated, Trust::Suspended] {
                let cap = tighten_cap(base, Some(cand), trust);
                assert!(cap <= base + 1e-6, "cap {cap} exceeded base {base} under {trust:?}");
                if trust != Trust::Calibrated {
                    assert_eq!(cap, base, "non-Calibrated trust {trust:?} must equal base");
                }
                let look = tighten_look_dist(base, Some(cand), trust);
                assert!(look <= base + 1e-6, "look {look} exceeded base {base} under {trust:?}");
                if trust != Trust::Calibrated {
                    assert_eq!(look, base);
                }
            }
        }
    }

    #[test]
    fn warn_brier_beats_base_rate_and_promotes_calibrated() {
        let mut hub = ForeUseHub::default();
        let mut sds = SdsStore::memory();
        for i in 0..20u64 {
            let will_cross = i % 2 == 0;
            let p = if will_cross { 0.85 } else { 0.15 };
            hub.issue(
                ForecastWarn {
                    kind: "forecast_warn",
                    use_id: "cargo_slip_ahead".into(),
                    tick: i * 10,
                    time_s: i as f64,
                    stage: "move".into(),
                    worker: 1,
                    grade: Grade::Control,
                    trust: hub.trust("cargo_slip_ahead"),
                    series: "cargo.use_ratio".into(),
                    limit_kind: "static_friction".into(),
                    limit_value: 0.85,
                    t_cross_s: Some(0.8),
                    act_s: 0.6,
                    p,
                    method: "fusion".into(),
                    effect: None,
                    resolved: None,
                    expire_tick: i * 10 + 10,
                },
                10,
            );
            hub.observe_outcome("cargo_slip_ahead", 1, will_cross, i as f64 + 0.5, &mut sds);
        }
        let st = hub.status_json();
        let row = st["uses"].as_array().unwrap().iter().find(|r| r["id"] == "cargo_slip_ahead").unwrap();
        assert_eq!(row["trust"], "calibrated");
        assert!(row["brier"].as_f64().unwrap() < row["base_brier"].as_f64().unwrap());
    }
}
