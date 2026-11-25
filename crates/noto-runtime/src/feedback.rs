use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::sds::{classify, DriftState, Provenance, Scope, SdsEstimate, SdsStore, Track, Window};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The eight decision-time SDS feedback paths (Phase 2 – Part 3 · D1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FeedbackPath {
    /// Vision score floor & touch agreement adjustment (`precision::floor_for`).
    A,
    /// Temporal gate band spread (`precision::judge_one` `score_all` sd).
    B,
    /// Forecast fusion weights (`precision::past_errors` `fusion_mse:*`).
    C,
    /// ML locate score floor (`mlsds::ml_analyse_with_rgb`).
    D,
    /// Depth scale calibration (`mlsds::depth_scale` → `mldrive`).
    E,
    /// Balance feed-forward gain (`mlsds::record_balance_events` → `world.ff_gains`).
    F,
    /// Odometry drift variance rate (`vision::DriftRate::learned` → `runtime::drift_rate`).
    G,
    /// Observation scatter floor (`runtime::mem_ctx`).
    H,
}

impl FeedbackPath {
    pub const ALL: [FeedbackPath; 8] = [
        FeedbackPath::A,
        FeedbackPath::B,
        FeedbackPath::C,
        FeedbackPath::D,
        FeedbackPath::E,
        FeedbackPath::F,
        FeedbackPath::G,
        FeedbackPath::H,
    ];

    pub fn id(self) -> &'static str {
        match self {
            FeedbackPath::A => "A",
            FeedbackPath::B => "B",
            FeedbackPath::C => "C",
            FeedbackPath::D => "D",
            FeedbackPath::E => "E",
            FeedbackPath::F => "F",
            FeedbackPath::G => "G",
            FeedbackPath::H => "H",
        }
    }

    pub fn from_id(s: &str) -> Option<FeedbackPath> {
        match s.trim().to_ascii_uppercase().as_str() {
            "A" => Some(FeedbackPath::A),
            "B" => Some(FeedbackPath::B),
            "C" => Some(FeedbackPath::C),
            "D" => Some(FeedbackPath::D),
            "E" => Some(FeedbackPath::E),
            "F" => Some(FeedbackPath::F),
            "G" => Some(FeedbackPath::G),
            "H" => Some(FeedbackPath::H),
            _ => None,
        }
    }

    pub fn spec(self) -> &'static PathSpec {
        &FEEDBACK_PATHS[self as usize]
    }
}

/// Generation inheritance policy when engine fingerprint changes (Phase 2 – Part 3 · D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Inherit {
    Never,
    Prior,
}

#[derive(Clone, Copy, Debug)]
pub struct PathSpec {
    pub id: FeedbackPath,
    pub track: Track,
    pub axis: &'static str,
    pub consumer: &'static str,
    pub cold_default: &'static str,
    pub inherit: Inherit,
}

pub const FEEDBACK_PATHS: [PathSpec; 8] = [
    PathSpec { id: FeedbackPath::A, track: Track::Vision, axis: "score,touch", consumer: "precision::floor_for", cold_default: "cfg.min_score", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::B, track: Track::Vision, axis: "score_all", consumer: "precision::judge_one", cold_default: "none (model band only)", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::C, track: Track::Forecast, axis: "fusion_mse:*", consumer: "precision::past_errors", cold_default: "holdout mse only", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::D, track: Track::Locate, axis: "score", consumer: "mlsds::ml_analyse_with_rgb", cold_default: "default_floor(mode)", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::E, track: Track::Depth, axis: "scale", consumer: "mlsds::depth_scale", cold_default: "none (1.0)", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::F, track: Track::Balance, axis: "ff", consumer: "mlsds::record_balance_events", cold_default: "1.0", inherit: Inherit::Never },
    PathSpec { id: FeedbackPath::G, track: Track::Memory, axis: "drift_var_per_m2", consumer: "vision::DriftRate::learned", cold_default: "DriftRate::spec", inherit: Inherit::Prior },
    PathSpec { id: FeedbackPath::H, track: Track::Memory, axis: "scatter_var_m2", consumer: "runtime::mem_ctx", cold_default: "0.0", inherit: Inherit::Prior },
];

/// Reproducibility mode for SDS decision reads (Phase 2 – Part 3 · D1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Live,
    Frozen,
    Off,
}

impl Mode {
    pub fn from_str(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "live" => Some(Mode::Live),
            "frozen" => Some(Mode::Frozen),
            "off" => Some(Mode::Off),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdsMode {
    #[serde(default)]
    pub default: Mode,
    #[serde(default)]
    pub overrides: BTreeMap<FeedbackPath, Mode>,
    #[serde(default)]
    pub snapshot: Option<String>,
}

impl SdsMode {
    pub fn effective(&self, path: FeedbackPath) -> Mode {
        self.overrides.get(&path).copied().unwrap_or(self.default)
    }

    pub fn uses_frozen(&self) -> bool {
        self.default == Mode::Frozen || self.overrides.values().any(|m| *m == Mode::Frozen)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FeedbackRecord {
    pub path: FeedbackPath,
    pub mode: Mode,
    pub value: Option<f64>,
    pub key: Option<String>,
    pub level: Option<String>,
    pub n: u64,
    pub last_tick: u64,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeedbackChange {
    pub path: FeedbackPath,
    pub mode: Mode,
    pub key: Option<String>,
    pub n: u64,
    pub value: Option<f64>,
    pub tick: u64,
}

fn scope_level(key: &str) -> &'static str {
    let mut parts = key.split('|');
    let _track = parts.next();
    let p = parts.next().unwrap_or("");
    let s = parts.next().unwrap_or("");
    if p.is_empty() {
        "global"
    } else if s.is_empty() {
        "primary"
    } else {
        "secondary"
    }
}

/// Central gatekeeper for all decision-time SDS reads (A–H).
pub struct FeedbackHub {
    mode: SdsMode,
    frozen: Option<SdsStore>,
    records: Mutex<BTreeMap<FeedbackPath, FeedbackRecord>>,
    last_seen: Mutex<BTreeMap<(FeedbackPath, String), (Mode, Option<f64>)>>,
    pending_changes: Mutex<Vec<FeedbackChange>>,
}

impl Default for FeedbackHub {
    fn default() -> Self {
        Self::live()
    }
}

impl FeedbackHub {
    pub fn live() -> Self {
        FeedbackHub {
            mode: SdsMode::default(),
            frozen: None,
            records: Mutex::new(BTreeMap::new()),
            last_seen: Mutex::new(BTreeMap::new()),
            pending_changes: Mutex::new(Vec::new()),
        }
    }

    pub fn mode(&self) -> &SdsMode {
        &self.mode
    }

    pub fn effective_mode(&self, path: FeedbackPath) -> Mode {
        self.mode.effective(path)
    }

    pub fn set_mode(&mut self, mut mode: SdsMode, live: &SdsStore) {
        if mode.uses_frozen() {
            let snap = live.clone_frozen();
            mode.snapshot = Some(snap.combined_digest());
            self.frozen = Some(snap);
        } else {
            mode.snapshot = None;
            self.frozen = None;
        }
        self.mode = mode;
    }

    /// Freeze against a pre-built snapshot store (used by replay).
    pub fn set_mode_with_snapshot(&mut self, mut mode: SdsMode, snap: SdsStore) {
        mode.snapshot = Some(snap.combined_digest());
        self.frozen = Some(snap);
        self.mode = mode;
    }

    pub fn drain_changes(&self) -> Vec<FeedbackChange> {
        self.pending_changes.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default()
    }

    fn store_for<'a>(&'a self, path: FeedbackPath, live: &'a SdsStore) -> Option<&'a SdsStore> {
        match self.effective_mode(path) {
            Mode::Off => None,
            Mode::Frozen => self.frozen.as_ref(),
            Mode::Live => Some(live),
        }
    }

    fn guard_provenance(scope: &Scope, axis: &str) -> Option<Provenance> {
        let prov = classify(scope.track, &scope.primary, axis);
        if !prov.control_allowed() {
            #[cfg(debug_assertions)]
            panic!(
                "SDS provenance guard blocked control read of {:?} axis {}|{}::{}",
                prov,
                scope.track.as_str(),
                scope.primary,
                axis
            );
            #[cfg(not(debug_assertions))]
            return None;
        }
        Some(prov)
    }

    fn query_tag(scope: &Scope, axis: &str) -> String {
        format!("{}|{}|{}::{axis}", scope.track.as_str(), scope.primary, scope.secondary)
    }

    fn record_use(&self, path: FeedbackPath, query: &str, prov: Provenance, val: Option<f64>, key: Option<String>, n: u64, tick: u64) {
        let mode = self.effective_mode(path);
        let level = key.as_deref().map(|k| scope_level(k).to_string());
        let mut changed = false;
        if let Ok(mut seen) = self.last_seen.lock() {
            let slot = (path, query.to_string());
            if let Some((prev_mode, prev_val)) = seen.get(&slot) {
                let v_diff = match (*prev_val, val) {
                    (Some(a), Some(b)) => (a - b).abs() > 1e-6,
                    (None, None) => false,
                    _ => true,
                };
                changed = v_diff || *prev_mode != mode;
            } else if val.is_some() {
                changed = true;
            }
            if seen.len() < 4096 || seen.contains_key(&slot) {
                seen.insert(slot, (mode, val));
            }
        }
        if let Ok(mut map) = self.records.lock() {
            let update = val.is_some()
                || map.get(&path).is_none_or(|p| p.mode != mode || p.value.is_none());
            if update {
                map.insert(
                    path,
                    FeedbackRecord { path, mode, value: val, key: key.clone(), level, n, last_tick: tick, provenance: prov },
                );
            }
        }
        if changed {
            if let Ok(mut q) = self.pending_changes.lock() {
                if q.len() < 256 {
                    q.push(FeedbackChange { path, mode, key, n, value: val, tick });
                }
            }
        }
    }

    pub fn baseline_scoped(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, tick: u64) -> Option<((f64, f64, u64), String)> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.adaptive_baseline_scoped(scope, axis);
        match &got {
            Some(((m, _, n), k)) => self.record_use(path, &qtag, prov, Some(*m), Some(k.clone()), *n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn baseline(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, tick: u64) -> Option<(f64, f64, u64)> {
        self.baseline_scoped(path, live, scope, axis, tick).map(|(v, _)| v)
    }

    pub fn agreement_scoped(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, tick: u64) -> Option<((f64, u64), String)> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.agreement_rate_scoped(scope, axis);
        match &got {
            Some(((r, n), k)) => self.record_use(path, &qtag, prov, Some(*r), Some(k.clone()), *n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn calibration(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, tick: u64) -> Option<(f64, u64, f64, String)> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.calibration_factor(scope, axis);
        match &got {
            Some((f, n, _, k)) => self.record_use(path, &qtag, prov, Some(*f), Some(k.clone()), *n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn fusion_mse(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, method_key: &str, tick: u64) -> Option<(f64, u64)> {
        let axis = format!("fusion_mse:{method_key}");
        self.baseline_scoped(path, live, scope, &axis, tick).map(|((m, _, n), _)| (m, n))
    }

    pub fn estimate_mean(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, window: Window, tick: u64) -> Option<SdsEstimate> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.estimate_mean(scope, axis, window, 0.95);
        match &got {
            Some(est) => self.record_use(path, &qtag, prov, Some(est.value), Some(est.key.clone()), est.n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn estimate_rate(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, window: Window, tick: u64) -> Option<SdsEstimate> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.estimate_rate(scope, axis, window, 0.95);
        match &got {
            Some(est) => self.record_use(path, &qtag, prov, Some(est.value), Some(est.key.clone()), est.n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn estimate_spread(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, window: Window, tick: u64) -> Option<SdsEstimate> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.estimate_spread(scope, axis, window, 0.95);
        match &got {
            Some(est) => self.record_use(path, &qtag, prov, Some(est.value), Some(est.key.clone()), est.n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn estimate_factor(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str, window: Window, tick: u64) -> Option<SdsEstimate> {
        let prov = Self::guard_provenance(scope, axis)?;
        let qtag = Self::query_tag(scope, axis);
        let Some(store) = self.store_for(path, live) else {
            self.record_use(path, &qtag, prov, None, None, 0, tick);
            return None;
        };
        let got = store.estimate_factor(scope, axis, window, 0.95);
        match &got {
            Some(est) => self.record_use(path, &qtag, prov, Some(est.value), Some(est.key.clone()), est.n, tick),
            None => self.record_use(path, &qtag, prov, None, None, 0, tick),
        }
        got
    }

    pub fn estimate_fusion_mse(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, method_key: &str, tick: u64) -> Option<SdsEstimate> {
        let axis = format!("fusion_mse:{method_key}");
        self.estimate_mean(path, live, scope, &axis, Window::Auto, tick)
    }

    /// Query CUSUM drift state strictly on control-allowed axes (Phase 2 – Part 2 · D2).
    pub fn drift_state(&self, path: FeedbackPath, live: &SdsStore, scope: &Scope, axis: &str) -> DriftState {
        if Self::guard_provenance(scope, axis).is_none() {
            return DriftState::Stable;
        }
        let Some(store) = self.store_for(path, live) else {
            return DriftState::Stable;
        };
        store.drift_state(scope, axis)
    }

    /// Snapshot of last observed decision values for `ReplayManifest.feedback_at_start`.
    pub fn snapshot_values(&self) -> BTreeMap<String, Option<f64>> {
        let recs = self.records.lock().map(|g| g.clone()).unwrap_or_default();
        FEEDBACK_PATHS
            .iter()
            .map(|spec| (spec.id.id().to_string(), recs.get(&spec.id).and_then(|r| r.value)))
            .collect()
    }

    pub fn status_json(&self) -> Value {
        let recs = self.records.lock().map(|g| g.clone()).unwrap_or_default();
        let paths: Vec<Value> = FEEDBACK_PATHS
            .iter()
            .map(|spec| {
                let rec = recs.get(&spec.id);
                json!({
                    "id": spec.id.id(),
                    "mode": self.effective_mode(spec.id),
                    "track": spec.track.as_str(),
                    "axis": spec.axis,
                    "consumer": spec.consumer,
                    "cold_default": spec.cold_default,
                    "inherit": spec.inherit,
                    "value": rec.and_then(|r| r.value),
                    "key": rec.and_then(|r| r.key.clone()),
                    "level": rec.and_then(|r| r.level.clone()),
                    "n": rec.map_or(0, |r| r.n),
                    "last_tick": rec.map_or(0, |r| r.last_tick),
                    "provenance": rec.map(|r| r.provenance).unwrap_or(Provenance::Experience),
                })
            })
            .collect();
        json!({
            "mode": self.mode,
            "paths": paths,
        })
    }

    /// Snapshot of current numeric feedback values keyed by path id (for ReplayManifest).
    pub fn values_at_start(&self) -> BTreeMap<String, Option<f64>> {
        let recs = self.records.lock().map(|g| g.clone()).unwrap_or_default();
        FeedbackPath::ALL.iter().map(|p| (p.id().to_string(), recs.get(p).and_then(|r| r.value))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_returns_none_even_when_ledger_is_populated() {
        let mut sds = SdsStore::memory();
        let scope = Scope::new(Track::Memory, "wheel", "drift");
        for _ in 0..10 {
            sds.record_baseline(&scope, "drift_var_per_m2", 0.004);
        }
        let mut fb = FeedbackHub::live();
        assert!(fb.baseline(FeedbackPath::G, &sds, &scope, "drift_var_per_m2", 1).is_some());
        fb.set_mode(SdsMode { default: Mode::Off, overrides: BTreeMap::new(), snapshot: None }, &sds);
        assert!(fb.baseline(FeedbackPath::G, &sds, &scope, "drift_var_per_m2", 2).is_none());
    }

    #[test]
    fn frozen_mode_ignores_subsequent_writes() {
        let mut sds = SdsStore::memory();
        let scope = Scope::new(Track::Memory, "wheel", "drift");
        for _ in 0..10 {
            sds.record_baseline(&scope, "drift_var_per_m2", 0.002);
        }
        let mut fb = FeedbackHub::live();
        fb.set_mode(SdsMode { default: Mode::Frozen, overrides: BTreeMap::new(), snapshot: None }, &sds);
        assert!(fb.mode().snapshot.is_some());
        let before = fb.baseline(FeedbackPath::G, &sds, &scope, "drift_var_per_m2", 1).unwrap().0;
        for _ in 0..50 {
            sds.record_baseline(&scope, "drift_var_per_m2", 0.09);
        }
        let after = fb.baseline(FeedbackPath::G, &sds, &scope, "drift_var_per_m2", 2).unwrap().0;
        assert!((before - after).abs() < 1e-12, "frozen read drifted: {before} vs {after}");
        let live_val = sds.adaptive_baseline(&scope, "drift_var_per_m2").unwrap().0;
        assert!(live_val > 0.05, "live ledger still received writes: {live_val}");
    }

    #[test]
    fn per_path_override_isolates_a_single_path() {
        let mut sds = SdsStore::memory();
        let s_g = Scope::new(Track::Memory, "wheel", "drift");
        let s_h = Scope::new(Track::Memory, "wheel", "obs");
        for _ in 0..10 {
            sds.record_baseline(&s_g, "drift_var_per_m2", 0.003);
            sds.record_baseline(&s_h, "scatter_var_m2", 0.001);
        }
        let mut overrides = BTreeMap::new();
        overrides.insert(FeedbackPath::G, Mode::Off);
        let mut fb = FeedbackHub::live();
        fb.set_mode(SdsMode { default: Mode::Live, overrides, snapshot: None }, &sds);
        assert!(fb.baseline(FeedbackPath::G, &sds, &s_g, "drift_var_per_m2", 1).is_none());
        assert!(fb.baseline(FeedbackPath::H, &sds, &s_h, "scatter_var_m2", 1).is_some());
    }

    #[test]
    #[should_panic(expected = "provenance guard")]
    fn truth_axis_never_reaches_control() {
        let sds = SdsStore::memory();
        let fb = FeedbackHub::live();
        let scope = Scope::new(Track::Depth, "tips", "tree");
        let _ = fb.baseline(FeedbackPath::E, &sds, &scope, "absrel", 1);
    }

    #[test]
    #[should_panic(expected = "provenance guard")]
    fn nn_scope_never_reaches_control() {
        let sds = SdsStore::memory();
        let fb = FeedbackHub::live();
        let scope = Scope::new(Track::Balance, concat!("nn", ":s4"), "brake_1.0");
        let _ = fb.baseline(FeedbackPath::F, &sds, &scope, "score", 1);
    }

    #[test]
    fn polling_multiple_scopes_on_one_path_does_not_spam_changes() {
        let mut sds = SdsStore::memory();
        let scope = Scope::new(Track::Forecast, "world", "");
        for _ in 0..10 {
            sds.record_baseline(&scope, "fusion_mse:holt", 0.02);
        }
        let fb = FeedbackHub::live();
        for tick in 1..=10u64 {
            assert!(fb.fusion_mse(FeedbackPath::C, &sds, &scope, "holt", tick).is_some());
            assert!(fb.fusion_mse(FeedbackPath::C, &sds, &scope, "ttm", tick).is_none());
        }
        assert_eq!(fb.drain_changes().len(), 1);
        assert!(fb.values_at_start()["C"].is_some());
    }
}
