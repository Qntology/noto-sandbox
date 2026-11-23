use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::catalog::ContactLog;
use crate::estimate::{effective_n, factor_ci, mean_ci, spread_ci, wilson_ci};
use crate::schema::{chain_keys, validate_key};
use crate::scope::{key_depth, ring_for_key, track_of_key, AxisKind, Ledger, Scope, ScopeKey, Track, Window};
use crate::stats::{decay_shape, ConfusionStat, DriftState, GateStat, ScopeStat, Temporal, Welford};

pub const SDS_RECIPE: &str = "noto-sds-v1:welford+ring/hier-write/ledger-split/calib-shrink/atomic";
const SDS_RECIPE_LEGACY: &str = "hand-sds-v1:welford+ring/hier-write/ledger-split/calib-shrink/atomic";

const AUTO_FLUSH_EVERY: u32 = 8;
const AUTO_FLUSH_MIN_GAP_MS: u128 = 5_000;
const DEFAULT_CAP_BYTES: usize = 2 * 1024 * 1024;
const CALIBRATION_SHRINK_K: f64 = 5.0;
/// The ring's `drift_z` past which a calibration centres on the ring alone. At 2 a steady scatter recentred one look in
/// twenty on a handful of values; at 3 it does not, and a 2 sd shift over a ring of 6 still passes 94 % of the time.
const RECENTRE_Z: f64 = 3.0;

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(format!(".tmp.{}.{seq}", std::process::id()));
    let tmp = PathBuf::from(tmp_os);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(_) if path.exists() => {
            let mut bak_os = path.as_os_str().to_owned();
            bak_os.push(format!(".bak.{}.{seq}", std::process::id()));
            let bak = PathBuf::from(bak_os);
            let _ = std::fs::remove_file(&bak);
            if let Err(e) = std::fs::rename(path, &bak) {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
            match std::fs::rename(&tmp, path) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&bak);
                    Ok(())
                }
                Err(e) => {
                    let _ = std::fs::rename(&bak, path);
                    let _ = std::fs::remove_file(&tmp);
                    Err(e)
                }
            }
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SdsFile {
    pub recipe: String,
    pub updated_at: i64,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
    #[serde(default)]
    pub engine_revs: Vec<String>,
}

pub const UNSTAMPED: &str = "unstamped";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtherRevs {
    #[default]
    Keep,
    SetAside,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenOptions {
    pub engine_rev: String,
    pub other_revs: OtherRevs,
}

impl OpenOptions {
    pub fn stamped(rev: &str) -> OpenOptions {
        OpenOptions { engine_rev: rev.into(), other_revs: OtherRevs::Keep }
    }

    pub fn only(rev: &str) -> OpenOptions {
        OpenOptions { engine_rev: rev.into(), other_revs: OtherRevs::SetAside }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopePolicy {
    pub pool_secondaries: bool,
    pub pool_primaries: bool,
}

impl Default for ScopePolicy {
    fn default() -> Self {
        ScopePolicy::POOLED
    }
}

impl ScopePolicy {
    pub const POOLED: ScopePolicy = ScopePolicy { pool_secondaries: true, pool_primaries: true };
    pub const KEEP_SECONDARIES: ScopePolicy = ScopePolicy { pool_secondaries: false, pool_primaries: true };
    pub const EXACT: ScopePolicy = ScopePolicy { pool_secondaries: false, pool_primaries: false };
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunFile {
    recipe: String,
    ledger: String,
    label: String,
    started_at: i64,
    updated_at: i64,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
    #[serde(default)]
    engine_rev: String,
}

#[derive(Debug, Clone, Default)]
struct RunBuf {
    label: String,
    file: String,
    started_ms: i64,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerStatus {
    pub ledger: String,
    pub scopes: usize,
    pub observations: u64,
    pub run_observations: u64,
    #[serde(default)]
    pub engine_revs: Vec<String>,
}

/// How an [`SdsEstimate`] was resolved along the hierarchy (Phase 2 – Part 2 · D3/D4/D5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Level {
    #[default]
    Exact,
    Pooled {
        borrowed: f64,
    },
    Mixed,
}

/// Uncertainty-aware SDS read result (Phase 2 – Part 2 · D3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SdsEstimate {
    pub value: f64,
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
    pub n: u64,
    pub n_eff: f64,
    pub rho: f64,
    pub window: Window,
    pub key: String,
    pub level: Level,
    pub state: DriftState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thin: Option<String>,
}

/// Generation metadata persisted in `sds/index.json` (Phase 2 – Part 3 · D2).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationEntry {
    pub id: String,
    pub parent: Option<String>,
    pub engine_rev: String,
    pub engine_fp: String,
    pub inherit: String,
    pub created_ms: i64,
    pub recipe: String,
    pub observations: u64,
    pub state: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenIndex {
    pub current: Option<String>,
    #[serde(default)]
    pub generations: Vec<GenerationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenStatus {
    pub current: Option<String>,
    pub parent: Option<String>,
    pub inherit: String,
    pub generations: Vec<GenerationEntry>,
    pub archive_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisStatus {
    pub scope: String,
    pub axis: String,
    pub kind: String,
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
    pub recent: f64,
    pub drift_z: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<Window>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_eff: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lo: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hi: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<DriftState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seeded: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_kind: Option<AxisKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRate {
    pub scope: String,
    pub gate: String,
    pub evaluated: u64,
    pub blocked_rate: f64,
    pub sole_rate: f64,
    pub mean_severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionStatus {
    pub scope: String,
    pub pair: String,
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub mean_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceReport {
    pub scope: String,
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
    pub mean_dev: f64,
    pub recent_dev: f64,
    pub in_band_rate: f64,
    #[serde(default)]
    pub mean_ref: f64,
    #[serde(default)]
    pub mean_tol: f64,
    #[serde(default)]
    pub mean_abs_dev: Option<f64>,
    #[serde(default)]
    pub rms_dev: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaAxis {
    pub scope: String,
    pub axis: String,
    pub n: u64,
    pub mean_dev: f64,
    pub mean_abs_dev: Option<f64>,
    pub rms_dev: Option<f64>,
    pub in_band_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaReport {
    pub track: String,
    pub prefix: String,
    pub axes: Vec<AreaAxis>,
    pub n: u64,
    pub mean_abs_dev: Option<f64>,
    pub rms_dev: Option<f64>,
    pub in_band_rate: Option<f64>,
    pub legacy_axes: usize,
    pub worst: Option<AreaAxis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdsStatus {
    pub dir: String,
    pub recipe: String,
    pub ledgers: Vec<LedgerStatus>,
    pub axes: Vec<AxisStatus>,
    pub confusions: Vec<ConfusionStatus>,
    pub log: Vec<String>,
}

fn sanitize_label(raw: &str) -> String {
    let s: String = raw.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let s = s.trim_matches('_').to_string();
    if s.is_empty() {
        "run".into()
    } else {
        s.chars().take(40).collect()
    }
}

pub struct SdsStore {
    dir: Option<PathBuf>,
    root_dir: Option<PathBuf>,
    generation_id: Option<String>,
    parent_generation: Option<String>,
    inherit_mode: String,
    bases: HashMap<Ledger, SdsFile>,
    runs: HashMap<Ledger, RunBuf>,
    dirty: HashSet<Ledger>,
    write_tick: u32,
    rev: u64,
    last_flush: Option<Instant>,
    log: Vec<String>,
    cap_bytes: usize,
    engine_rev: String,
    policies: HashMap<Track, ScopePolicy>,
    contacts: Option<Mutex<ContactLog>>,
    drift_transitions: Vec<(String, String, DriftState, DriftState)>,
}

/// Event-scoped batch writer deduplicating upstream scope keys so multi-match events
/// (e.g. S1/slope + S6/slope) count once at shared ancestor scopes (Phase 2 – Part 1 · D2).
pub struct Batch<'a> {
    store: &'a mut SdsStore,
    seen_baseline: HashSet<(String, String)>,
    seen_agreement: HashSet<(String, String)>,
    seen_extrema: HashSet<(String, String)>,
}

impl<'a> Batch<'a> {
    pub fn record_baseline(&mut self, scope: &Scope, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.store.note_write(scope, axis);
        let keys: Vec<String> = scope
            .write_keys()
            .into_iter()
            .filter(|k| self.seen_baseline.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_baseline(scope.track, &keys, axis, value);
    }

    pub fn record_agreement(&mut self, scope: &Scope, axis: &str, agreed: bool) {
        self.store.note_write(scope, axis);
        let keys: Vec<String> = scope
            .write_keys()
            .into_iter()
            .filter(|k| self.seen_agreement.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_agreement(scope.track, &keys, axis, agreed);
    }

    pub fn record_extrema(&mut self, scope: &Scope, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        let keys: Vec<String> = scope
            .write_keys()
            .into_iter()
            .filter(|k| self.seen_extrema.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_extrema(scope.track, &keys, axis, value);
    }

    pub fn record_v2_baseline(&mut self, key: &ScopeKey, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let keys: Vec<String> = chain_keys(key)
            .into_iter()
            .filter(|k| self.seen_baseline.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_baseline(key.track, &keys, axis, value);
    }

    pub fn record_v2_agreement(&mut self, key: &ScopeKey, axis: &str, agreed: bool) {
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let keys: Vec<String> = chain_keys(key)
            .into_iter()
            .filter(|k| self.seen_agreement.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_agreement(key.track, &keys, axis, agreed);
    }

    pub fn record_v2_extrema(&mut self, key: &ScopeKey, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let keys: Vec<String> = chain_keys(key)
            .into_iter()
            .filter(|k| self.seen_extrema.insert((k.clone(), axis.to_string())))
            .collect();
        self.store.write_keys_extrema(key.track, &keys, axis, value);
    }

    pub fn record_v2_gates(&mut self, key: &ScopeKey, items: &[(String, u8, bool, bool)]) {
        if items.is_empty() {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let keys: Vec<String> = chain_keys(key)
            .into_iter()
            .filter(|k| self.seen_baseline.insert((k.clone(), "__gates__".to_string())))
            .collect();
        let ledger = key.track.ledger();
        let ring = key.track.ring_len();
        let run = self.store.open_run(ledger, key.track.as_str());
        for k in keys {
            let s = run.scopes.entry(k).or_default();
            for (id, sev, blocked, sole) in items {
                let g: &mut GateStat = s.gates.entry(id.clone()).or_default();
                g.evaluated += 1;
                if *blocked {
                    g.blocked += 1;
                }
                if *sole {
                    g.sole += 1;
                }
                g.severity.push(*sev as f64, ring);
            }
            s.updated_at = now_ms();
        }
        self.store.finish_write(ledger);
    }

    pub fn record_v2_transition(&mut self, key: &ScopeKey, from: &str, to: &str) {
        if from.is_empty() || to.is_empty() {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let tkey = format!("{from}>{to}");
        let keys: Vec<String> = chain_keys(key)
            .into_iter()
            .filter(|k| self.seen_baseline.insert((k.clone(), format!("__tr:{tkey}"))))
            .collect();
        let ledger = key.track.ledger();
        let run = self.store.open_run(ledger, key.track.as_str());
        for k in keys {
            let s = run.scopes.entry(k).or_default();
            *s.transition.entry(tkey.clone()).or_insert(0) += 1;
            s.updated_at = now_ms();
        }
        self.store.finish_write(ledger);
    }

    /// Record reference cell detail only on the schema's `detail_rungs` (e.g. `+ph` and `+form`),
    /// avoiding 7× write amplification at global/root rungs (Phase 2 – Part 1 · D2).
    pub fn record_v2_reference_detail(&mut self, key: &ScopeKey, axis: &str, value: f64, reference: f64, tolerance: f64) {
        if !value.is_finite() || !reference.is_finite() || !(tolerance.is_finite() && tolerance > 0.0) {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let all = chain_keys(key);
        let detail_indices = crate::schema::schema_for(key.track, &key.subject).detail_rungs;
        // `chain_keys` prepends `[track/, track/subject, ...rungs]`; rung `r` is at index `r + 1`.
        let mut keys = Vec::new();
        for &r in detail_indices {
            if let Some(k) = all.get(r + 1) {
                if self.seen_baseline.insert((k.clone(), format!("__ref:{axis}"))) {
                    keys.push(k.clone());
                }
            }
        }
        if keys.is_empty() {
            return;
        }
        let dev = (value - reference) / tolerance;
        let abs = dev.abs();
        self.store.write_keys_baseline(key.track, &keys, axis, value);
        self.store.write_keys_baseline(key.track, &keys, &format!("{axis}:dev"), dev);
        self.store.write_keys_baseline(key.track, &keys, &format!("{axis}:absdev"), abs);
        self.store.write_keys_baseline(key.track, &keys, &format!("{axis}:dev2"), dev * dev);
        self.store.write_keys_baseline(key.track, &keys, &format!("{axis}:ref"), reference);
        self.store.write_keys_baseline(key.track, &keys, &format!("{axis}:tol"), tolerance);
        self.store.write_keys_agreement(key.track, &keys, &format!("{axis}:in_band"), abs <= 1.0);
    }

    pub fn record_v2_exact_baseline(&mut self, key: &ScopeKey, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        if validate_key(key).is_err() {
            self.store.note_rejected(key.track);
            return;
        }
        let k = key.to_string();
        if self.seen_baseline.insert((k.clone(), axis.to_string())) {
            self.store.write_keys_baseline(key.track, &[k], axis, value);
        }
    }
}

fn should_track_temporal(track: Track, axis: &str) -> bool {
    matches!(
        (track, axis),
        (Track::Vision, "score" | "score_all" | "touch")
            | (Track::Locate, "score")
            | (Track::Depth, "scale")
            | (Track::Balance, "ff" | "score" | "min_margin" | "peak_ratio")
            | (Track::Memory, "drift_var_per_m2" | "scatter_var_m2")
            | (Track::Tracker, "iou_prev")
            | (Track::Act, "value" | "ok")
    ) || axis.starts_with("fusion_mse:")
}

fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size_bytes(&p);
            } else if let Ok(m) = e.metadata() {
                total += m.len();
            }
        }
    }
    total
}

fn archive_file(base_dir: &Path, tag: &str, src: &Path) -> Option<PathBuf> {
    if !src.exists() {
        return None;
    }
    let arch = base_dir.join("archive").join(format!("{}-{tag}", now_ms()));
    let _ = std::fs::create_dir_all(&arch);
    let name = src.file_name()?;
    let dst = arch.join(name);
    std::fs::rename(src, &dst).ok().map(|_| dst)
}

impl SdsStore {
    pub fn memory() -> Self {
        let mut s = SdsStore {
            dir: None,
            root_dir: None,
            generation_id: None,
            parent_generation: None,
            inherit_mode: "fresh".into(),
            bases: HashMap::new(),
            runs: HashMap::new(),
            dirty: HashSet::new(),
            write_tick: 0,
            rev: 0,
            last_flush: None,
            log: Vec::new(),
            cap_bytes: DEFAULT_CAP_BYTES,
            engine_rev: String::new(),
            policies: HashMap::new(),
            contacts: None,
            drift_transitions: Vec::new(),
        };
        for l in Ledger::ALL {
            s.bases.insert(l, fresh_file());
        }
        s.log.push("[SDS] 메모리 전용 저장소 (디스크에 쓰지 않음)".into());
        s
    }

    pub fn open(dir: &Path) -> Self {
        SdsStore::open_with(dir, OpenOptions::default())
    }

    pub fn open_with(dir: &Path, opts: OpenOptions) -> Self {
        let mut store = SdsStore::memory();
        store.dir = Some(dir.to_path_buf());
        store.root_dir = Some(dir.to_path_buf());
        store.engine_rev = opts.engine_rev.clone();
        store.log.clear();
        let _ = std::fs::create_dir_all(dir);
        let mut loaded = 0usize;
        for l in Ledger::ALL {
            let path = dir.join(l.file_name());
            let mut file = fresh_file();
            if path.exists() {
                match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|t| serde_json::from_str::<SdsFile>(&t).map_err(|e| e.to_string()))
                {
                    Ok(mut f) if f.recipe == SDS_RECIPE || f.recipe == SDS_RECIPE_LEGACY => {
                        if f.engine_revs.is_empty() && !f.scopes.is_empty() {
                            f.engine_revs.push(UNSTAMPED.into());
                        }
                        let own = store.stamp();
                        let foreign = f.engine_revs.iter().any(|r| *r != own);
                        if foreign && opts.other_revs == OtherRevs::SetAside {
                            let aside = dir.join(format!("{}.rev-{}.json", l.as_str(), now_ms()));
                            match std::fs::rename(&path, &aside) {
                                Ok(()) => store.log.push(format!(
                                    "[SDS] [{}] 다른 엔진 개정({})의 관측을 {}로 옮겨 두고 새로 시작합니다",
                                    l.as_str(),
                                    f.engine_revs.join(", "),
                                    aside.display()
                                )),
                                Err(e) => store.log.push(format!("[SDS] [{}] 다른 엔진 개정의 원장을 옮기지 못해({e}) 읽지 않고 새로 시작합니다", l.as_str())),
                            }
                        } else {
                            loaded += 1;
                            file = f;
                        }
                    }
                    Ok(f) => {
                        let moved = archive_file(dir, "recipe", &path);
                        store.log.push(format!(
                            "[SDS] [{}] 레시피 세대 불일치 ('{}' vs '{}') — 원장을 보관({})하고 새로 시작합니다",
                            l.as_str(),
                            f.recipe,
                            SDS_RECIPE,
                            moved.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "실패".into())
                        ));
                    }
                    Err(e) => {
                        let moved = archive_file(dir, "recipe", &path);
                        store.log.push(format!(
                            "[SDS] [{}] 원장 읽기/파싱 실패({}) — 보관({})하고 새로 시작합니다",
                            l.as_str(),
                            e,
                            moved.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "실패".into())
                        ));
                    }
                }
            }
            store.bases.insert(l, file);
        }
        store.log.push(if loaded == 0 {
            "[SDS] 저장된 원장이 없어 냉간 시작합니다 (보정 미개입, 고정 기본값 사용)".into()
        } else {
            format!("[SDS] 원장 {}종을 불러왔습니다", loaded)
        });
        store
    }

    /// Open a generational ledger under `<root_dir>/gen/<engine_rev>+<engine_fp>/`
    /// tracking parent lineage in `<root_dir>/index.json` (Phase 2 – Part 3 · D2).
    /// - When `engine_rev` changes: opens a `Fresh` generation.
    /// - When only `engine_fp` changes: opens an `Inherit` generation, copying `Prior`-eligible
    ///   scopes with counts shrunk to `min(n_parent, track.min_obs().1)` and skipping `Never`
    ///   axes (`Balance::ff`).
    pub fn open_generation(root_dir: &Path, engine_rev: &str, engine_fp: &str) -> Self {
        let _ = std::fs::create_dir_all(root_dir);
        let gen_id = if engine_fp.is_empty() {
            engine_rev.to_string()
        } else {
            format!("{engine_rev}+{engine_fp}")
        };
        let index_path = root_dir.join("index.json");
        let mut index: GenIndex = std::fs::read_to_string(&index_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let gen_dir = root_dir.join("gen").join(&gen_id);
        let existed = gen_dir.exists();
        let prev_entry = index.current.as_deref().and_then(|cur| {
            if cur == gen_id {
                None
            } else {
                index.generations.iter().find(|g| g.id == cur).cloned()
            }
        });

        let mut store = SdsStore::open_with(&gen_dir, OpenOptions::stamped(engine_rev));
        store.root_dir = Some(root_dir.to_path_buf());
        store.generation_id = Some(gen_id.clone());

        let (parent_id, inherit_mode) = match prev_entry {
            Some(prev) if !existed => {
                let same_rev = prev.engine_rev == engine_rev;
                let mode = if same_rev { "inherit" } else { "fresh" };
                if same_rev {
                    let parent_dir = root_dir.join("gen").join(&prev.id);
                    if parent_dir.exists() {
                        let parent_store = SdsStore::open_with(&parent_dir, OpenOptions::stamped(&prev.engine_rev));
                        store.inherit_prior_from(&parent_store);
                    }
                }
                (Some(prev.id), mode.to_string())
            }
            Some(prev) => (Some(prev.id), "existing".to_string()),
            None => (None, if existed { "existing".into() } else { "fresh".into() }),
        };
        store.parent_generation = parent_id.clone();
        store.inherit_mode = inherit_mode.clone();

        if !index.generations.iter().any(|g| g.id == gen_id) {
            index.generations.push(GenerationEntry {
                id: gen_id.clone(),
                parent: parent_id,
                engine_rev: engine_rev.into(),
                engine_fp: engine_fp.into(),
                inherit: inherit_mode,
                created_ms: now_ms(),
                recipe: SDS_RECIPE.into(),
                observations: store.total_observations(),
                state: "active".into(),
            });
        }
        for g in &mut index.generations {
            g.state = if g.id == gen_id { "active".into() } else { "archived".into() };
        }
        index.current = Some(gen_id);
        if let Ok(txt) = serde_json::to_string_pretty(&index) {
            let _ = write_atomic(&index_path, txt.as_bytes());
        }
        store
    }

    fn inherit_prior_from(&mut self, parent: &SdsStore) {
        for l in Ledger::ALL {
            let Some(pf) = parent.bases.get(&l) else { continue };
            let target = self.bases.entry(l).or_insert_with(fresh_file);
            for (k, st) in &pf.scopes {
                let Some(tr) = track_of_key(k) else { continue };
                let cap_n = tr.min_obs().1.max(1);
                let shrink = |w: &Welford| -> Welford {
                    if w.n == 0 {
                        return Welford::default();
                    }
                    let n_new = w.n.min(cap_n);
                    let m2_new = if n_new > 1 { w.variance() * (n_new - 1) as f64 } else { 0.0 };
                    let pol = tr.temporal_policy();
                    let mut out = Welford { n: n_new, mean: w.mean, m2: m2_new, ring: w.ring.clone(), t: None };
                    if should_track_temporal(tr, "") || w.t.is_some() {
                        out.t = Some(Temporal::seed_from(&out, pol.m_short, pol.m_long));
                    }
                    out
                };
                let entry = target.scopes.entry(k.clone()).or_default();
                for (a, w) in &st.baseline {
                    if tr == Track::Balance && a == "ff" {
                        continue; // Path F is Inherit::Never
                    }
                    entry.baseline.insert(a.clone(), shrink(w));
                }
                for (a, w) in &st.calibration {
                    if tr == Track::Balance && a == "ff" {
                        continue; // Path F is Inherit::Never
                    }
                    entry.calibration.insert(a.clone(), shrink(w));
                }
                for (a, w) in &st.agreement {
                    entry.agreement.insert(a.clone(), shrink(w));
                }
                entry.updated_at = now_ms();
            }
            self.dirty.insert(l);
        }
    }

    pub fn generation_id(&self) -> Option<&str> {
        self.generation_id.as_deref()
    }

    pub fn set_generation_id(&mut self, gen: &str) {
        self.generation_id = Some(gen.to_string());
    }

    pub fn prune_archive(&self, keep: usize) -> Vec<String> {
        let Some(root) = self.root_dir.as_deref().or(self.dir.as_deref()) else { return Vec::new() };
        let arch = root.join("archive");
        let Ok(rd) = std::fs::read_dir(&arch) else { return Vec::new() };
        let mut entries: Vec<(String, PathBuf)> = rd
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut removed = Vec::new();
        while entries.len() > keep {
            let (name, path) = entries.remove(0);
            let ok = if path.is_dir() {
                std::fs::remove_dir_all(&path).is_ok()
            } else {
                std::fs::remove_file(&path).is_ok()
            };
            if ok {
                removed.push(name);
            }
        }
        removed
    }

    pub fn list_archive(&self) -> Vec<(String, u64)> {
        let Some(root) = self.root_dir.as_deref().or(self.dir.as_deref()) else { return Vec::new() };
        let arch = root.join("archive");
        let Ok(rd) = std::fs::read_dir(&arch) else { return Vec::new() };
        let mut out: Vec<(String, u64)> = rd
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), dir_size_bytes(&e.path())))
            .collect();
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out
    }

    pub fn total_observations(&self) -> u64 {
        Ledger::ALL
            .iter()
            .map(|l| {
                let b: u64 = self.bases.get(l).map(|f| f.scopes.values().map(ScopeStat::observations).sum()).unwrap_or(0);
                let r: u64 = self.runs.get(l).map(|rb| rb.observations).unwrap_or(0);
                b + r
            })
            .sum()
    }

    pub fn archive_bytes(&self) -> u64 {
        let Some(root) = self.root_dir.as_deref().or(self.dir.as_deref()) else { return 0 };
        dir_size_bytes(&root.join("archive"))
    }

    pub fn generations_status(&self) -> GenStatus {
        let Some(root) = self.root_dir.as_deref().or(self.dir.as_deref()) else {
            return GenStatus {
                current: self.generation_id.clone(),
                parent: self.parent_generation.clone(),
                inherit: self.inherit_mode.clone(),
                generations: Vec::new(),
                archive_bytes: 0,
            };
        };
        let index: GenIndex = std::fs::read_to_string(root.join("index.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        GenStatus {
            current: self.generation_id.clone().or(index.current),
            parent: self.parent_generation.clone(),
            inherit: self.inherit_mode.clone(),
            generations: index.generations,
            archive_bytes: self.archive_bytes(),
        }
    }

    /// Save a content-addressed snapshot under `<snap_root>/<digest>/` (Phase 2 – Part 3 · D6).
    pub fn save_snapshot(&self, snap_root: &Path) -> std::io::Result<String> {
        let frozen = self.clone_frozen();
        let digest = frozen.combined_digest();
        let dir = snap_root.join(&digest);
        if dir.exists() {
            return Ok(digest);
        }
        std::fs::create_dir_all(&dir)?;
        for l in Ledger::ALL {
            if let Some(f) = frozen.bases.get(&l) {
                let txt = serde_json::to_string_pretty(f).unwrap_or_default();
                write_atomic(&dir.join(l.file_name()), txt.as_bytes())?;
            }
        }
        Ok(digest)
    }

    /// Load a previously saved content-addressed snapshot for reproducible replay.
    pub fn load_snapshot(snap_root: &Path, digest: &str) -> std::io::Result<SdsStore> {
        let dir = snap_root.join(digest);
        if !dir.exists() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("snapshot {digest} not found")));
        }
        let store = SdsStore::open_with(&dir, OpenOptions::default());
        Ok(store.clone_frozen())
    }

    /// Migrate legacy v1 pipe-delimited keys in place to v2 faceted `ScopeKey` strings,
    /// preserving a `.v1-<ms>.json` backup when disk-backed (Phase 2 – Part 1 · §5).
    /// Returns `(migrated_keys, legacy_kept)`.
    pub fn migrate_v1_to_v2(&mut self) -> (usize, usize) {
        let mut migrated = 0usize;
        let mut legacy_kept = 0usize;
        let stamp = now_ms();
        for l in Ledger::ALL {
            if let Some(dir) = &self.dir {
                let src = dir.join(l.file_name());
                if src.exists() {
                    let bak = dir.join(format!("{}.v1-{stamp}.json", l.as_str()));
                    let _ = std::fs::copy(&src, &bak);
                }
            }
            let Some(base) = self.bases.get_mut(&l) else { continue };
            let old = std::mem::take(&mut base.scopes);
            for (k, st) in old {
                let ring = ring_for_key(&k);
                match ScopeKey::from_v1(&k) {
                    Ok(sk) => {
                        let new_k = sk.to_key_string();
                        if new_k != k {
                            migrated += 1;
                        }
                        base.scopes.entry(new_k).or_default().absorb(st, ring, stamp);
                    }
                    Err(_) => {
                        legacy_kept += 1;
                        let leg_k = format!("legacy/{k}");
                        base.scopes.entry(leg_k).or_default().absorb(st, ring, stamp);
                    }
                }
            }
            if migrated > 0 || legacy_kept > 0 {
                self.dirty.insert(l);
            }
        }
        if migrated > 0 || legacy_kept > 0 {
            self.rev = self.rev.saturating_add(1);
        }
        (migrated, legacy_kept)
    }

    pub fn drain_drift_transitions(&mut self) -> Vec<(String, String, DriftState, DriftState)> {
        std::mem::take(&mut self.drift_transitions)
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Monotonic write counter incremented on every write/purge/close_runs.
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// Enable in-memory read/write contact logging (for test symmetry verification).
    pub fn enable_contact_log(&mut self) {
        if self.contacts.is_none() {
            self.contacts = Some(Mutex::new(ContactLog::default()));
        }
    }

    pub fn contact_snapshot(&self) -> Option<ContactLog> {
        self.contacts.as_ref().and_then(|m| m.lock().ok().map(|g| g.clone()))
    }

    fn note_write(&self, scope: &Scope, axis: &str) {
        if let Some(m) = &self.contacts {
            if let Ok(mut g) = m.lock() {
                g.record_write(scope.track, &scope.primary, axis);
            }
        }
    }

    fn note_read(&self, scope: &Scope, axis: &str) {
        if let Some(m) = &self.contacts {
            if let Ok(mut g) = m.lock() {
                g.record_read(scope.track, &scope.primary, axis);
            }
        }
    }

    /// Freeze the current merged ledger state (base + open run buffer) into an isolated memory store
    /// for `Mode::Frozen` reproducible reads (Phase 2 – Part 3 · D1/D6).
    pub fn clone_frozen(&self) -> SdsStore {
        let mut frozen = SdsStore::memory();
        frozen.engine_rev = self.engine_rev.clone();
        frozen.policies = self.policies.clone();
        frozen.rev = self.rev;
        for l in Ledger::ALL {
            let mut file = self.bases.get(&l).cloned().unwrap_or_else(fresh_file);
            if let Some(run) = self.runs.get(&l) {
                for (k, v) in &run.scopes {
                    file.scopes.entry(k.clone()).or_default().absorb(v.clone(), ring_for_key(k), file.updated_at);
                }
            }
            file.engine_revs = self.engine_revs(l);
            frozen.bases.insert(l, file);
        }
        frozen
    }

    /// Canonical SHA-256 hex digest of each ledger's merged scopes (order-independent of HashMap seeds).
    pub fn ledger_digests(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for l in Ledger::ALL {
            let mut keys: Vec<String> = self.bases.get(&l).map(|b| b.scopes.keys().cloned().collect()).unwrap_or_default();
            if let Some(r) = self.runs.get(&l) {
                keys.extend(r.scopes.keys().cloned());
            }
            keys.sort();
            keys.dedup();
            let mut map = serde_json::Map::new();
            for k in keys {
                if let Some(st) = self.merged_scope(l, &k) {
                    if let Ok(v) = serde_json::to_value(&st) {
                        map.insert(k, canonical_json_no_ts(v));
                    }
                }
            }
            let bytes = serde_json::to_vec(&serde_json::Value::Object(map)).unwrap_or_default();
            let digest = format!("{:x}", Sha256::digest(&bytes));
            out.insert(l.as_str().to_string(), digest);
        }
        out
    }

    /// Combined digest across all ledgers in canonical order.
    pub fn combined_digest(&self) -> String {
        let map = self.ledger_digests();
        let mut h = Sha256::new();
        for (k, v) in map {
            h.update(k.as_bytes());
            h.update(b"=");
            h.update(v.as_bytes());
            h.update(b"\n");
        }
        format!("{:x}", h.finalize())
    }

    pub fn set_cap_bytes(&mut self, cap: usize) {
        self.cap_bytes = cap;
    }

    pub fn engine_rev(&self) -> &str {
        &self.engine_rev
    }

    pub fn set_engine_rev(&mut self, rev: &str) {
        self.engine_rev = rev.into();
    }

    pub fn engine_revs(&self, l: Ledger) -> Vec<String> {
        let mut v = self.bases.get(&l).map(|b| b.engine_revs.clone()).unwrap_or_default();
        if self.runs.get(&l).is_some_and(|r| r.observations > 0) {
            v.push(self.stamp());
        }
        v.sort();
        v.dedup();
        v
    }

    fn stamp(&self) -> String {
        if self.engine_rev.is_empty() {
            UNSTAMPED.into()
        } else {
            self.engine_rev.clone()
        }
    }

    pub fn set_policy(&mut self, track: Track, policy: ScopePolicy) {
        self.policies.insert(track, policy);
    }

    pub fn policy(&self, track: Track) -> ScopePolicy {
        self.policies.get(&track).copied().unwrap_or_default()
    }

    fn levels(&self, scope: &Scope) -> Vec<(String, u64)> {
        let (m2, m1, mg) = scope.track.min_obs();
        let mut v = vec![(scope.key_secondary(), m2), (scope.key_primary(), m1), (scope.key_global(), mg)];
        let p = self.policy(scope.track);
        if !p.pool_secondaries && !scope.secondary.is_empty() {
            v.truncate(1);
        } else if !p.pool_primaries && !scope.primary.is_empty() {
            v.truncate(2);
        }
        v
    }

    pub fn log_lines(&self) -> &[String] {
        &self.log
    }

    fn push_log(&mut self, line: String) {
        if self.log.len() >= 200 {
            self.log.drain(0..100);
        }
        self.log.push(line);
    }

    fn open_run(&mut self, l: Ledger, label: &str) -> &mut RunBuf {
        let dir = self.dir.clone();
        self.runs.entry(l).or_insert_with(|| {
            let stamp = now_ms();
            let lab = sanitize_label(label);
            let mut file = format!("{}_{}.json", stamp, lab);
            if let Some(dir) = &dir {
                let mut i = 2;
                while dir.join("runs").join(l.as_str()).join(&file).exists() {
                    file = format!("{}_{}_{}.json", stamp, lab, i);
                    i += 1;
                }
            }
            RunBuf { label: lab, file, started_ms: stamp, observations: 0, scopes: HashMap::new() }
        })
    }

    fn with_scopes<F: Fn(&mut ScopeStat, usize)>(&mut self, scope: &Scope, f: F) {
        let ledger = scope.track.ledger();
        let ring = scope.track.ring_len();
        let keys = scope.write_keys();
        let run = self.open_run(ledger, scope.track.as_str());
        run.observations = run.observations.saturating_add(1);
        for k in keys {
            let e = run.scopes.entry(k).or_default();
            f(e, ring);
            e.updated_at = now_ms();
        }
        self.dirty.insert(ledger);
        self.rev = self.rev.saturating_add(1);
        self.write_tick = self.write_tick.saturating_add(1);
        let gap_ok = self.last_flush.map(|t| t.elapsed().as_millis() >= AUTO_FLUSH_MIN_GAP_MS).unwrap_or(true);
        if self.dir.is_some() && self.write_tick >= AUTO_FLUSH_EVERY && gap_ok {
            self.flush();
        }
    }

    fn note_rejected(&mut self, track: Track) {
        let meta_scope = Scope::new(Track::Quest, "_meta", "");
        let axis = format!("rejected:{}", track.as_str());
        self.record_baseline(&meta_scope, &axis, 1.0);
        self.push_log(format!("[SDS] 스코프 문법 위반 버림 ({})", track.as_str()));
    }

    fn seed_run_temporal(&mut self, ledger: Ledger, key: &str, axis: &str, kind: u8) {
        let base_t = self.bases.get(&ledger).and_then(|b| b.scopes.get(key)).and_then(|st| {
            let map = match kind {
                0 => &st.baseline,
                1 => &st.calibration,
                _ => &st.agreement,
            };
            map.get(axis).map(|w| {
                let pol = track_of_key(key).unwrap_or(Track::Vision).temporal_policy();
                w.temporal_or_seeded(pol.m_short, pol.m_long)
            })
        });
        if let Some(bt) = base_t {
            if let Some(run) = self.runs.get_mut(&ledger) {
                if let Some(st) = run.scopes.get_mut(key) {
                    let map = match kind {
                        0 => &mut st.baseline,
                        1 => &mut st.calibration,
                        _ => &mut st.agreement,
                    };
                    let entry = map.entry(axis.to_string()).or_default();
                    if entry.t.is_none() {
                        entry.t = Some(bt);
                    }
                }
            }
        }
    }

    fn write_keys_baseline(&mut self, track: Track, keys: &[String], axis: &str, value: f64) {
        if !value.is_finite() || keys.is_empty() {
            return;
        }
        let ledger = track.ledger();
        let ring = track.ring_len();
        let track_t = should_track_temporal(track, axis);
        let pol = track.temporal_policy();
        self.open_run(ledger, track.as_str());
        if track_t {
            for k in keys {
                self.seed_run_temporal(ledger, k, axis, 0);
            }
        }
        let mut transitions = Vec::new();
        if let Some(run) = self.runs.get_mut(&ledger) {
            run.observations = run.observations.saturating_add(1);
            for k in keys {
                let e = run.scopes.entry(k.clone()).or_default();
                let w = e.baseline.entry(axis.to_string()).or_default();
                if track_t {
                    if let Some((prev, next)) = w.push_temporal(value, ring, pol.m_short, pol.m_long, pol.cusum_k, pol.cusum_h) {
                        transitions.push((k.clone(), axis.to_string(), prev, next));
                    }
                } else {
                    w.push(value, ring);
                }
                e.updated_at = now_ms();
            }
        }
        self.drift_transitions.extend(transitions);
        self.finish_write(ledger);
    }

    fn write_keys_calibration(&mut self, track: Track, keys: &[String], axis: &str, ratio: f64) {
        if !ratio.is_finite() || keys.is_empty() {
            return;
        }
        let ledger = track.ledger();
        let ring = track.ring_len();
        let track_t = should_track_temporal(track, axis);
        let pol = track.temporal_policy();
        self.open_run(ledger, track.as_str());
        if track_t {
            for k in keys {
                self.seed_run_temporal(ledger, k, axis, 1);
            }
        }
        let mut transitions = Vec::new();
        if let Some(run) = self.runs.get_mut(&ledger) {
            run.observations = run.observations.saturating_add(1);
            for k in keys {
                let e = run.scopes.entry(k.clone()).or_default();
                let w = e.calibration.entry(axis.to_string()).or_default();
                if track_t {
                    if let Some((prev, next)) = w.push_temporal(ratio, ring, pol.m_short, pol.m_long, pol.cusum_k, pol.cusum_h) {
                        transitions.push((k.clone(), axis.to_string(), prev, next));
                    }
                } else {
                    w.push(ratio, ring);
                }
                e.updated_at = now_ms();
            }
        }
        self.drift_transitions.extend(transitions);
        self.finish_write(ledger);
    }

    fn write_keys_agreement(&mut self, track: Track, keys: &[String], axis: &str, agreed: bool) {
        if keys.is_empty() {
            return;
        }
        let ledger = track.ledger();
        let ring = track.ring_len();
        let val = if agreed { 1.0 } else { 0.0 };
        let track_t = should_track_temporal(track, axis);
        let pol = track.temporal_policy();
        self.open_run(ledger, track.as_str());
        if track_t {
            for k in keys {
                self.seed_run_temporal(ledger, k, axis, 2);
            }
        }
        let mut transitions = Vec::new();
        if let Some(run) = self.runs.get_mut(&ledger) {
            run.observations = run.observations.saturating_add(1);
            for k in keys {
                let e = run.scopes.entry(k.clone()).or_default();
                let w = e.agreement.entry(axis.to_string()).or_default();
                if track_t {
                    if let Some((prev, next)) = w.push_temporal(val, ring, pol.m_short, pol.m_long, pol.cusum_k, pol.cusum_h) {
                        transitions.push((k.clone(), axis.to_string(), prev, next));
                    }
                } else {
                    w.push(val, ring);
                }
                e.updated_at = now_ms();
            }
        }
        self.drift_transitions.extend(transitions);
        self.finish_write(ledger);
    }

    fn write_keys_extrema(&mut self, track: Track, keys: &[String], axis: &str, value: f64) {
        if !value.is_finite() || keys.is_empty() {
            return;
        }
        let ledger = track.ledger();
        let run = self.open_run(ledger, track.as_str());
        for k in keys {
            let e = run.scopes.entry(k.clone()).or_default();
            e.extrema.entry(axis.to_string()).or_default().push(value);
            e.updated_at = now_ms();
        }
        self.finish_write(ledger);
    }

    fn finish_write(&mut self, ledger: Ledger) {
        self.dirty.insert(ledger);
        self.rev = self.rev.saturating_add(1);
        self.write_tick = self.write_tick.saturating_add(1);
        let gap_ok = self.last_flush.map(|t| t.elapsed().as_millis() >= AUTO_FLUSH_MIN_GAP_MS).unwrap_or(true);
        if self.dir.is_some() && self.write_tick >= AUTO_FLUSH_EVERY && gap_ok {
            self.flush();
        }
    }

    /// Execute an event-scoped write batch deduplicating upstream keys (Phase 2 – Part 1 · D2).
    pub fn record_batch(&mut self, _event_id: &str, f: impl FnOnce(&mut Batch<'_>)) {
        let mut batch = Batch {
            store: self,
            seen_baseline: HashSet::new(),
            seen_agreement: HashSet::new(),
            seen_extrema: HashSet::new(),
        };
        f(&mut batch);
    }

    pub fn record_extrema(&mut self, scope: &Scope, axis: &str, value: f64) {
        let keys = scope.write_keys();
        self.write_keys_extrema(scope.track, &keys, axis, value);
    }

    pub fn record_v2_baseline(&mut self, key: &ScopeKey, axis: &str, value: f64) {
        if validate_key(key).is_err() {
            self.note_rejected(key.track);
            return;
        }
        let keys = chain_keys(key);
        self.write_keys_baseline(key.track, &keys, axis, value);
    }

    pub fn record_v2_agreement(&mut self, key: &ScopeKey, axis: &str, agreed: bool) {
        if validate_key(key).is_err() {
            self.note_rejected(key.track);
            return;
        }
        let keys = chain_keys(key);
        self.write_keys_agreement(key.track, &keys, axis, agreed);
    }

    pub fn record_v2_extrema(&mut self, key: &ScopeKey, axis: &str, value: f64) {
        if validate_key(key).is_err() {
            self.note_rejected(key.track);
            return;
        }
        let keys = chain_keys(key);
        self.write_keys_extrema(key.track, &keys, axis, value);
    }

    pub fn record_baseline(&mut self, scope: &Scope, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.note_write(scope, axis);
        let keys = scope.write_keys();
        self.write_keys_baseline(scope.track, &keys, axis, value);
    }

    pub fn record_calibration(&mut self, scope: &Scope, axis: &str, measured: f64, predicted: f64) {
        if !measured.is_finite() || !predicted.is_finite() || predicted.abs() < 1e-9 {
            return;
        }
        self.note_write(scope, axis);
        let ratio = (measured / predicted).clamp(0.05, 20.0);
        let keys = scope.write_keys();
        self.write_keys_calibration(scope.track, &keys, axis, ratio);
    }

    pub fn record_decay(&mut self, scope: &Scope, axis: &str, scores: &[f32]) {
        let Some(shape) = decay_shape(scores) else { return };
        self.note_write(scope, axis);
        let a = axis.to_string();
        self.with_scopes(scope, move |s, ring| {
            let d = s.decay.entry(a.clone()).or_default();
            d.margin.push(shape.margin, ring);
            d.top_gap_ratio.push(shape.top_gap_ratio, ring);
            d.tail_flatness.push(shape.tail_flatness, ring);
            d.entropy_norm.push(shape.entropy_norm, ring);
        });
    }

    pub fn record_confusion(&mut self, scope: &Scope, winner: &str, loser: &str, margin: f64) {
        if winner.is_empty() || loser.is_empty() || winner == loser {
            return;
        }
        self.note_write(scope, "confusion");
        let (a, b, a_won) = if winner <= loser {
            (winner.to_string(), loser.to_string(), true)
        } else {
            (loser.to_string(), winner.to_string(), false)
        };
        let key = format!("{}|{}", a, b);
        self.with_scopes(scope, move |s, ring| {
            let e: &mut ConfusionStat = s.confusion.entry(key.clone()).or_default();
            e.ties += 1;
            if a_won {
                e.a_wins += 1;
            } else {
                e.b_wins += 1;
            }
            if margin.is_finite() {
                e.margin.push(margin, ring);
            }
        });
    }

    pub fn record_agreement(&mut self, scope: &Scope, axis: &str, agreed: bool) {
        self.note_write(scope, axis);
        let keys = scope.write_keys();
        self.write_keys_agreement(scope.track, &keys, axis, agreed);
    }

    pub fn record_reference(&mut self, scope: &Scope, axis: &str, value: f64, reference: f64, tolerance: f64) {
        if !value.is_finite() || !reference.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
            return;
        }
        self.note_write(scope, axis);
        let dev = (value - reference) / tolerance;
        let (a, ad, ab, ar, at) = (axis.to_string(), format!("{axis}:dev"), format!("{axis}:in_band"), format!("{axis}:ref"), format!("{axis}:tol"));
        let (aa, a2) = (format!("{axis}:absdev"), format!("{axis}:dev2"));
        self.with_scopes(scope, move |s, ring| {
            s.baseline.entry(a.clone()).or_default().push(value, ring);
            s.baseline.entry(ad.clone()).or_default().push(dev, ring);
            s.baseline.entry(aa.clone()).or_default().push(dev.abs(), ring);
            s.baseline.entry(a2.clone()).or_default().push(dev * dev, ring);
            s.baseline.entry(ar.clone()).or_default().push(reference, ring);
            s.baseline.entry(at.clone()).or_default().push(tolerance, ring);
            s.agreement.entry(ab.clone()).or_default().push(if dev.abs() <= 1.0 { 1.0 } else { 0.0 }, ring);
        });
    }

    pub fn record_transition(&mut self, scope: &Scope, from: &str, to: &str) {
        if from.is_empty() || to.is_empty() {
            return;
        }
        self.note_write(scope, "transition");
        let key = format!("{}>{}", from, to);
        self.with_scopes(scope, move |s, _| {
            *s.transition.entry(key.clone()).or_insert(0) += 1;
        });
    }

    pub fn record_gates(&mut self, scope: &Scope, items: &[(String, u8, bool, bool)]) {
        if items.is_empty() {
            return;
        }
        self.note_write(scope, "gates");
        let list = items.to_vec();
        self.with_scopes(scope, move |s, ring| {
            for (id, sev, blocked, sole) in list.iter() {
                let g: &mut GateStat = s.gates.entry(id.clone()).or_default();
                g.evaluated += 1;
                if *blocked {
                    g.blocked += 1;
                }
                if *sole {
                    g.sole += 1;
                }
                g.severity.push(*sev as f64, ring);
            }
        });
    }

    fn merged_scope(&self, l: Ledger, key: &str) -> Option<ScopeStat> {
        let base = self.bases.get(&l).and_then(|f| f.scopes.get(key).cloned());
        let run = self.runs.get(&l).and_then(|r| r.scopes.get(key).cloned());
        match (base, run) {
            (Some(mut b), Some(r)) => {
                b.absorb(r, ring_for_key(key), now_ms());
                Some(b)
            }
            (Some(b), None) => Some(b),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    pub fn merged_scope_by_key(&self, l: Ledger, key: &str) -> Option<ScopeStat> {
        self.merged_scope(l, key)
    }

    pub fn run_scope_by_key(&self, l: Ledger, key: &str) -> Option<ScopeStat> {
        self.runs.get(&l).and_then(|r| r.scopes.get(key).cloned())
    }

    /// Return all scope keys (both v1 and v2) present in base or run buffer for `track`.
    pub fn scope_keys_for_track(&self, track: Track) -> Vec<String> {
        let l = track.ledger();
        let p1 = format!("{}|", track.as_str());
        let p2 = format!("{}/", track.as_str());
        let mut keys: Vec<String> = self
            .bases
            .get(&l)
            .map(|b| b.scopes.keys().filter(|k| k.starts_with(&p1) || k.starts_with(&p2)).cloned().collect())
            .unwrap_or_default();
        if let Some(r) = self.runs.get(&l) {
            keys.extend(r.scopes.keys().filter(|k| k.starts_with(&p1) || k.starts_with(&p2)).cloned());
        }
        keys.sort();
        keys.dedup();
        keys
    }

    /// Collect `(mean, n, variance)` across sibling scopes directly below `parent_key` at `rung_depth`
    /// for empirical Bayes shrinkage (`pool.rs`).
    pub(crate) fn sibling_stats(&self, track: Track, parent_key: &str, rung_depth: usize, axis: &str, agreement: bool) -> Vec<(f64, f64, f64)> {
        let l = track.ledger();
        let mut out = Vec::new();
        for k in self.scope_keys_for_track(track) {
            if k == parent_key || (!parent_key.is_empty() && !k.starts_with(parent_key)) {
                continue;
            }
            if key_depth(&k) != rung_depth {
                continue;
            }
            if let Some(st) = self.merged_scope(l, &k) {
                let w = if agreement { st.agreement.get(axis) } else { st.baseline.get(axis) };
                if let Some(w) = w.filter(|w| w.n >= 2) {
                    out.push((w.mean, w.n as f64, w.variance()));
                }
            }
        }
        out
    }

    /// Extract `(mean, sd, n_eff, rho, used_window, state)` from a [`Welford`] for a requested [`Window`].
    fn extract_window(track: Track, w: &Welford, req: Window) -> (f64, f64, f64, f64, Window, DriftState) {
        let pol = track.temporal_policy();
        let t = w.t.as_ref();
        let state = t.map_or(DriftState::Stable, |x| x.state);
        let pick = match req {
            Window::Auto => {
                if t.and_then(|x| x.seg.as_ref()).is_some_and(|sp| sp.n >= 2) {
                    Window::Span
                } else if t.is_some() {
                    Window::Long
                } else {
                    Window::Lifetime
                }
            }
            other => other,
        };
        match pick {
            Window::Lifetime => (w.mean, w.sd(), w.n as f64, t.map_or(0.0, |x| x.long.rho()), Window::Lifetime, state),
            Window::Long => {
                let ts = t.cloned().unwrap_or_else(|| w.temporal_or_seeded(pol.m_short, pol.m_long));
                (ts.long.mean, ts.long.sd(), ts.long.n_eff(), ts.long.rho(), Window::Long, state)
            }
            Window::Short => {
                let ts = t.cloned().unwrap_or_else(|| w.temporal_or_seeded(pol.m_short, pol.m_long));
                (ts.short.mean, ts.short.sd(), ts.short.n_eff(), ts.short.rho(), Window::Short, state)
            }
            Window::Span => {
                if let Some(sp) = t.and_then(|x| x.seg.as_ref()).filter(|s| s.n >= 1) {
                    (sp.mean, sp.sd(), sp.n as f64, 0.0, Window::Span, state)
                } else {
                    (w.mean, w.sd(), w.n as f64, t.map_or(0.0, |x| x.long.rho()), Window::Lifetime, state)
                }
            }
            Window::Auto => unreachable!(),
        }
    }

    /// Uncertainty-aware mean estimate (Phase 2 – Part 2 · D3).
    pub fn estimate_mean(&self, scope: &Scope, axis: &str, window: Window, conf: f64) -> Option<SdsEstimate> {
        self.note_read(scope, axis);
        let pol = scope.track.temporal_policy();
        let ((w, need), key) = self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.clone(), 0u64), w.n)))?;
        let _ = need;
        let (mean, sd, raw_neff, rho, used, state) = Self::extract_window(scope.track, &w, window);
        let n_e = effective_n(raw_neff, rho);
        let (se, lo, hi) = mean_ci(mean, sd, n_e, conf);
        let hw = 0.5 * (hi - lo);
        let thin = if n_e < pol.min_obs.floor_n as f64 {
            Some(format!("n_eff {:.1} < {}", n_e, pol.min_obs.floor_n))
        } else if pol.min_obs.target_hw.is_some_and(|t| hw > t) {
            Some(format!("반폭 {hw:.3} > {:.3}", pol.min_obs.target_hw.unwrap()))
        } else {
            None
        };
        Some(SdsEstimate { value: mean, se, lo, hi, n: w.n, n_eff: n_e, rho, window: used, key, level: Level::Exact, state, thin })
    }

    /// Uncertainty-aware Bernoulli rate estimate with Wilson CI (Phase 2 – Part 2 · D3).
    pub fn estimate_rate(&self, scope: &Scope, axis: &str, window: Window, conf: f64) -> Option<SdsEstimate> {
        self.note_read(scope, axis);
        let pol = scope.track.temporal_policy();
        let (w, key) = self.resolve(scope, |st| st.agreement.get(axis).map(|w| (w.clone(), w.n)))?;
        let (mean, _, raw_neff, rho, used, state) = Self::extract_window(scope.track, &w, window);
        let n_e = effective_n(raw_neff, rho);
        let (se, lo, hi) = wilson_ci(mean, n_e, conf);
        let thin = if n_e < pol.min_obs.floor_n as f64 {
            Some(format!("n_eff {:.1} < {}", n_e, pol.min_obs.floor_n))
        } else {
            None
        };
        Some(SdsEstimate { value: mean.clamp(0.0, 1.0), se, lo, hi, n: w.n, n_eff: n_e, rho, window: used, key, level: Level::Exact, state, thin })
    }

    /// Uncertainty-aware spread (SD) estimate with chi-square CI (Phase 2 – Part 2 · D3).
    pub fn estimate_spread(&self, scope: &Scope, axis: &str, window: Window, conf: f64) -> Option<SdsEstimate> {
        self.note_read(scope, axis);
        let pol = scope.track.temporal_policy();
        let (w, key) = self.resolve(scope, |st| st.baseline.get(axis).map(|w| (w.clone(), w.n)))?;
        let (_, sd, raw_neff, rho, used, state) = Self::extract_window(scope.track, &w, window);
        let n_e = effective_n(raw_neff, rho);
        let (se, lo, hi) = spread_ci(sd, n_e, conf);
        let thin = if n_e < pol.min_obs.floor_n as f64 {
            Some(format!("n_eff {:.1} < {}", n_e, pol.min_obs.floor_n))
        } else {
            None
        };
        Some(SdsEstimate { value: sd, se, lo, hi, n: w.n, n_eff: n_e, rho, window: used, key, level: Level::Exact, state, thin })
    }

    /// Uncertainty-aware multiplicative calibration factor estimate with `n_e` shrinkage and log-scale CI.
    pub fn estimate_factor(&self, scope: &Scope, axis: &str, window: Window, conf: f64) -> Option<SdsEstimate> {
        self.note_read(scope, axis);
        let pol = scope.track.temporal_policy();
        let (w, key) = self.resolve(scope, |st| st.calibration.get(axis).map(|w| (w.clone(), w.n)))?;
        let (mean, sd, raw_neff, rho, used, state) = Self::extract_window(scope.track, &w, window);
        let n_e = effective_n(raw_neff, rho);
        let cv = if mean > 1e-6 { sd / mean } else { 0.0 };
        let geo = if matches!(used, Window::Lifetime) {
            let drift = w.drift_z();
            if drift.abs() > RECENTRE_Z { w.recent_geo_mean() } else { w.geo_mean() }
        } else if mean > 0.0 {
            mean / (1.0 + cv * cv).sqrt()
        } else {
            mean
        };
        let shrunk = (1.0 + (geo - 1.0) * (n_e / (n_e + CALIBRATION_SHRINK_K))).clamp(0.25, 4.0);
        let (se, lo, hi) = factor_ci(shrunk, cv, n_e, conf);
        let thin = if n_e < pol.min_obs.floor_n as f64 {
            Some(format!("n_eff {:.1} < {}", n_e, pol.min_obs.floor_n))
        } else {
            None
        };
        Some(SdsEstimate { value: shrunk, se, lo, hi, n: w.n, n_eff: n_e, rho, window: used, key, level: Level::Exact, state, thin })
    }

    /// Current CUSUM drift state for a scope/axis (checks baseline, calibration, agreement).
    pub fn drift_state(&self, scope: &Scope, axis: &str) -> DriftState {
        let l = scope.track.ledger();
        for (key, _) in self.levels(scope) {
            if let Some(st) = self.merged_scope(l, &key) {
                if let Some(w) = st.baseline.get(axis).or_else(|| st.calibration.get(axis)).or_else(|| st.agreement.get(axis)) {
                    if let Some(t) = &w.t {
                        return t.state;
                    }
                }
            }
        }
        DriftState::Stable
    }

    fn resolve<T, F: Fn(&ScopeStat) -> Option<(T, u64)>>(&self, scope: &Scope, pick: F) -> Option<(T, String)> {
        let l = scope.track.ledger();
        for (key, need) in self.levels(scope) {
            if need == 0 {
                continue;
            }
            if let Some(st) = self.merged_scope(l, &key) {
                if let Some((v, n)) = pick(&st) {
                    if n >= need {
                        return Some((v, key));
                    }
                }
            }
        }
        None
    }

    pub fn adaptive_baseline(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.mean, w.sd(), w.n), w.n))).map(|v| v.0)
    }

    pub fn adaptive_baseline_scoped(&self, scope: &Scope, axis: &str) -> Option<((f64, f64, u64), String)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.mean, w.sd(), w.n), w.n)))
    }

    pub fn adaptive_recent(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, f64)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.recent_mean(), w.mean, w.drift_z()), w.n))).map(|v| v.0)
    }

    pub fn calibration_factor(&self, scope: &Scope, axis: &str) -> Option<(f64, u64, f64, String)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| {
            st.calibration.get(axis).map(|w| {
                let drift = w.drift_z();
                let center = if drift.abs() > RECENTRE_Z { w.recent_geo_mean() } else { w.geo_mean() };
                let shrunk = 1.0 + (center - 1.0) * (w.n as f64 / (w.n as f64 + CALIBRATION_SHRINK_K));
                ((shrunk.clamp(0.25, 4.0), w.n, drift), w.n)
            })
        })
        .map(|((f, n, d), key)| (f, n, d, key))
    }

    pub fn transition_prior(&self, scope: &Scope, from: &str, to: &str) -> Option<f64> {
        let l = scope.track.ledger();
        for (key, need) in self.levels(scope).into_iter().take(2) {
            let Some(st) = self.merged_scope(l, &key) else { continue };
            let prefix = format!("{}>", from);
            let total: u64 = st.transition.iter().filter(|(k, _)| k.starts_with(&prefix)).map(|(_, v)| *v).sum();
            if total < need {
                continue;
            }
            let outs = st.transition.keys().filter(|k| k.starts_with(&prefix)).count().max(1) as f64;
            let hit = *st.transition.get(&format!("{}>{}", from, to)).unwrap_or(&0) as f64;
            return Some((hit + 1.0) / (total as f64 + outs + 1.0));
        }
        None
    }

    pub fn agreement_rate(&self, scope: &Scope, axis: &str) -> Option<(f64, u64)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| st.agreement.get(axis).map(|w| ((w.mean, w.n), w.n))).map(|v| v.0)
    }

    pub fn agreement_rate_scoped(&self, scope: &Scope, axis: &str) -> Option<((f64, u64), String)> {
        self.note_read(scope, axis);
        self.resolve(scope, |st| st.agreement.get(axis).map(|w| ((w.mean, w.n), w.n)))
    }

    pub fn adaptive_decay(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        self.resolve(scope, |st| st.decay.get(axis).map(|d| ((d.margin.mean, d.margin.sd(), d.margin.n), d.margin.n))).map(|v| v.0)
    }

    pub fn confusion_share(&self, scope: &Scope, a: &str, b: &str) -> Option<(f64, u64)> {
        let (x, y) = if a <= b { (a, b) } else { (b, a) };
        let key = format!("{}|{}", x, y);
        self.resolve(scope, |st| {
            let total: u64 = st.confusion.values().map(|c| c.ties).sum();
            st.confusion.get(&key).map(|c| ((c.ties as f64 / total.max(1) as f64, c.ties), total))
        })
        .map(|v| v.0)
    }

    pub fn reference_report(&self, scope: &Scope, axis: &str) -> Option<ReferenceReport> {
        self.resolve(scope, |st| reference_of(st, axis).map(|r| {
            let n = r.n;
            (r, n)
        }))
        .map(|(r, scope_key)| ReferenceReport { scope: scope_key, ..r })
    }

    pub fn exact_baseline(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        let key = if scope.secondary.is_empty() { scope.key_primary() } else { scope.key_secondary() };
        let st = self.merged_scope(scope.track.ledger(), &key)?;
        st.baseline.get(axis).filter(|w| w.n > 0).map(|w| (w.mean, w.sd(), w.n))
    }

    pub fn exact_reference(&self, scope: &Scope, axis: &str) -> Option<ReferenceReport> {
        let key = if scope.secondary.is_empty() { scope.key_primary() } else { scope.key_secondary() };
        let st = self.merged_scope(scope.track.ledger(), &key)?;
        st.baseline.get(axis).filter(|w| w.n > 0)?;
        reference_of(&st, axis).map(|r| ReferenceReport { scope: key, ..r })
    }

    pub fn area_report(&self, track: Track, prefix: &str) -> AreaReport {
        let l = track.ledger();
        let head = format!("{}|", track.as_str());
        let prefix = prefix.trim().to_lowercase();
        let mut keys: Vec<String> = self.bases.get(&l).map(|b| b.scopes.keys().cloned().collect()).unwrap_or_default();
        if let Some(r) = self.runs.get(&l) {
            keys.extend(r.scopes.keys().cloned());
        }
        keys.retain(|k| {
            let Some(rest) = k.strip_prefix(&head) else { return false };
            match rest.split_once('|') {
                Some((p, s)) => !p.is_empty() && s.is_empty() && p.starts_with(&prefix),
                None => false,
            }
        });
        keys.sort();
        keys.dedup();
        let mut axes = Vec::new();
        for k in &keys {
            let Some(st) = self.merged_scope(l, k) else { continue };
            let mut names: Vec<&String> = st.baseline.keys().filter(|a| a.ends_with(":dev")).collect();
            names.sort();
            for a in names {
                let axis = a.trim_end_matches(":dev");
                let d = &st.baseline[a];
                if d.n == 0 {
                    continue;
                }
                axes.push(AreaAxis {
                    scope: k.clone(),
                    axis: axis.to_string(),
                    n: d.n,
                    mean_dev: d.mean,
                    mean_abs_dev: st.baseline.get(&format!("{axis}:absdev")).filter(|w| w.n > 0).map(|w| w.mean),
                    rms_dev: st.baseline.get(&format!("{axis}:dev2")).filter(|w| w.n > 0).map(|w| w.mean.max(0.0).sqrt()),
                    in_band_rate: st.agreement.get(&format!("{axis}:in_band")).filter(|w| w.n > 0).map(|w| w.mean),
                });
            }
        }
        let weighted = |f: &dyn Fn(&AreaAxis) -> Option<f64>| {
            let (num, den) = axes.iter().filter_map(|a| f(a).map(|v| (v * a.n as f64, a.n as f64))).fold((0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1));
            (den > 0.0).then(|| num / den)
        };
        let mean_abs_dev = weighted(&|a| a.mean_abs_dev);
        let rms_dev = weighted(&|a| a.rms_dev.map(|r| r * r)).map(f64::sqrt);
        let in_band_rate = weighted(&|a| a.in_band_rate);
        let far = |a: &AreaAxis| a.mean_abs_dev.unwrap_or(a.mean_dev.abs());
        let worst = axes.iter().max_by(|a, b| far(a).total_cmp(&far(b))).cloned();
        AreaReport {
            track: track.as_str().into(),
            prefix,
            n: axes.iter().map(|a| a.n).sum(),
            legacy_axes: axes.iter().filter(|a| a.mean_abs_dev.is_none()).count(),
            mean_abs_dev,
            rms_dev,
            in_band_rate,
            worst,
            axes,
        }
    }

    pub fn fusion_mse(&self, scope: &Scope, method: &str) -> Option<(f64, u64)> {
        self.adaptive_baseline(scope, &format!("fusion_mse:{}", method)).map(|(m, _, n)| (m, n))
    }

    pub fn gate_rates(&self, scope: &Scope) -> Vec<GateRate> {
        let l = scope.track.ledger();
        let mut out = Vec::new();
        for (key, _) in self.levels(scope) {
            if let Some(st) = self.merged_scope(l, &key) {
                if st.gates.is_empty() {
                    continue;
                }
                for (id, g) in st.gates.iter() {
                    let n = g.evaluated.max(1) as f64;
                    out.push(GateRate {
                        scope: key.clone(),
                        gate: id.clone(),
                        evaluated: g.evaluated,
                        blocked_rate: g.blocked as f64 / n,
                        sole_rate: g.sole as f64 / n,
                        mean_severity: g.severity.mean,
                    });
                }
                break;
            }
        }
        out.sort_by(|a, b| b.blocked_rate.partial_cmp(&a.blocked_rate).unwrap_or(std::cmp::Ordering::Equal).then(a.gate.cmp(&b.gate)));
        out
    }

    pub fn axes_of(&self, scope: &Scope, prefix: &str) -> Vec<AxisStatus> {
        let l = scope.track.ledger();
        for (key, _) in self.levels(scope) {
            if let Some(st) = self.merged_scope(l, &key) {
                let mut v: Vec<AxisStatus> = st
                    .baseline
                    .iter()
                    .filter(|(a, _)| a.starts_with(prefix))
                    .map(|(a, w)| axis_status(&key, a, "baseline", w))
                    .collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_by(|a, b| a.axis.cmp(&b.axis));
                return v;
            }
        }
        Vec::new()
    }

    pub fn flush(&mut self) -> Vec<String> {
        let Some(dir) = self.dir.clone() else { return Vec::new() };
        let mut lines = Vec::new();
        let open: Vec<Ledger> = self.runs.keys().cloned().collect();
        for l in Ledger::ALL {
            if !self.dirty.contains(&l) && !open.contains(&l) {
                continue;
            }
            if let Some(run) = self.runs.get(&l) {
                let rf = RunFile {
                    recipe: SDS_RECIPE.into(),
                    ledger: l.as_str().into(),
                    label: run.label.clone(),
                    started_at: run.started_ms,
                    updated_at: now_ms(),
                    observations: run.observations,
                    scopes: run.scopes.clone(),
                    engine_rev: self.engine_rev.clone(),
                };
                let rdir = dir.join("runs").join(l.as_str());
                let _ = std::fs::create_dir_all(&rdir);
                if let Ok(txt) = serde_json::to_string_pretty(&rf) {
                    match write_atomic(&rdir.join(&run.file), txt.as_bytes()) {
                        Ok(()) => lines.push(format!("{}={}건→{}", l.as_str(), run.observations, run.file)),
                        Err(e) => lines.push(format!("{} 실행 파일 저장 실패: {}", l.as_str(), e)),
                    }
                }
            }
            let mut snapshot = self.bases.get(&l).cloned().unwrap_or_default();
            snapshot.recipe = SDS_RECIPE.into();
            snapshot.engine_revs = self.engine_revs(l);
            if let Some(run) = self.runs.get(&l) {
                for (k, v) in run.scopes.iter() {
                    snapshot.scopes.entry(k.clone()).or_default().absorb(v.clone(), ring_for_key(k), now_ms());
                }
            }
            snapshot.updated_at = now_ms();
            let mut txt = serde_json::to_string_pretty(&snapshot).unwrap_or_default();
            if txt.len() > self.cap_bytes {
                // Depth-first deterministic truncation (Phase 2 – Part 1 · D5, Part 3 · D6):
                // drop deepest scopes first, then fewest observations, oldest, then key name;
                // never drop depth-0 global scopes.
                let mut keys: Vec<(String, usize, u64, i64)> = snapshot
                    .scopes
                    .iter()
                    .filter(|(k, _)| key_depth(k) > 0)
                    .map(|(k, v)| (k.clone(), key_depth(k), v.observations(), v.updated_at))
                    .collect();
                keys.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)).then(a.0.cmp(&b.0)));
                for (k, _, _, _) in keys {
                    if txt.len() <= self.cap_bytes {
                        break;
                    }
                    snapshot.scopes.remove(&k);
                    txt = serde_json::to_string_pretty(&snapshot).unwrap_or(txt);
                }
                lines.push(format!("{} 상한 초과로 저관측 스코프 절삭 (잔존 {})", l.as_str(), snapshot.scopes.len()));
            }
            match write_atomic(&dir.join(l.file_name()), txt.as_bytes()) {
                Ok(()) => {
                    self.dirty.remove(&l);
                    lines.push(format!("{}.json={}B", l.as_str(), txt.len()));
                }
                Err(e) => lines.push(format!("{} 원장 저장 실패: {}", l.as_str(), e)),
            }
        }
        self.write_tick = 0;
        self.last_flush = Some(Instant::now());
        if !lines.is_empty() {
            self.push_log(format!("[SDS] 저장: {}", lines.join(" | ")));
        }
        lines
    }

    pub fn close_runs(&mut self) {
        self.flush();
        let revs: Vec<(Ledger, Vec<String>)> = Ledger::ALL.iter().map(|l| (*l, self.engine_revs(*l))).collect();
        let runs: Vec<(Ledger, RunBuf)> = self.runs.drain().collect();
        for (l, revs) in revs {
            if let Some(b) = self.bases.get_mut(&l) {
                b.engine_revs = revs;
            }
        }
        for (l, run) in runs {
            let base = self.bases.entry(l).or_default();
            for (k, v) in run.scopes.into_iter() {
                let ring = ring_for_key(&k);
                base.scopes.entry(k).or_default().absorb(v, ring, now_ms());
            }
            base.updated_at = now_ms();
            self.dirty.insert(l);
        }
        self.flush();
    }

    /// Archive current ledger files into `archive/<ms>-purge/` and reset to cold start
    /// (Phase 2 – Part 3 · D2: archive instead of irreversible deletion).
    pub fn purge(&mut self) -> Option<PathBuf> {
        let mut archived_dir = None;
        self.bases.clear();
        self.runs.clear();
        self.dirty.clear();
        self.rev = self.rev.saturating_add(1);
        if let Some(dir) = self.dir.clone() {
            let arch_root = self.root_dir.clone().unwrap_or_else(|| dir.clone());
            let arch = arch_root.join("archive").join(format!("{}-purge", now_ms()));
            let mut moved_any = false;
            for l in Ledger::ALL {
                let p = dir.join(l.file_name());
                if p.exists() {
                    let _ = std::fs::create_dir_all(&arch);
                    if std::fs::rename(&p, arch.join(l.file_name())).is_ok() {
                        moved_any = true;
                    }
                }
            }
            let runs_dir = dir.join("runs");
            if runs_dir.exists() {
                let _ = std::fs::create_dir_all(&arch);
                if std::fs::rename(&runs_dir, arch.join("runs")).is_ok() {
                    moved_any = true;
                }
            }
            if moved_any {
                archived_dir = Some(arch);
            }
            let _ = std::fs::create_dir_all(&dir);
        }
        for l in Ledger::ALL {
            self.bases.insert(l, fresh_file());
        }
        self.push_log(match &archived_dir {
            Some(p) => format!("[SDS] 원장·실행 파일을 {}로 보관하고 냉간 상태로 복귀했습니다", p.display()),
            None => "[SDS] 원장·실행 파일을 초기화했습니다. 판정은 고정 기본값으로 복귀합니다".into(),
        });
        archived_dir
    }

    pub fn status(&self, max_axes: usize) -> SdsStatus {
        let mut ledgers = Vec::new();
        let mut axes = Vec::new();
        let mut confusions = Vec::new();
        for l in Ledger::ALL {
            let run = self.runs.get(&l);
            let mut keys: Vec<String> = self.bases.get(&l).map(|b| b.scopes.keys().cloned().collect()).unwrap_or_default();
            if let Some(r) = run {
                for k in r.scopes.keys() {
                    if !keys.contains(k) {
                        keys.push(k.clone());
                    }
                }
            }
            keys.sort();
            let mut obs = 0u64;
            for k in keys.iter() {
                let Some(st) = self.merged_scope(l, k) else { continue };
                obs += st.observations();
                for (a, w) in st.baseline.iter() {
                    axes.push(axis_status(k, a, "baseline", w));
                }
                for (a, w) in st.calibration.iter() {
                    axes.push(axis_status(k, a, "calibration", w));
                }
                for (a, d) in st.decay.iter() {
                    axes.push(axis_status(k, a, "decay_margin", &d.margin));
                }
                for (a, w) in st.agreement.iter() {
                    axes.push(axis_status(k, a, "agreement", w));
                }
                for (g, gs) in st.gates.iter() {
                    let n = gs.evaluated.max(1) as f64;
                    axes.push(AxisStatus {
                        scope: k.clone(),
                        axis: g.clone(),
                        kind: "gate_block".into(),
                        n: gs.evaluated,
                        mean: gs.blocked as f64 / n,
                        sd: gs.severity.sd(),
                        recent: gs.sole as f64 / n,
                        drift_z: gs.severity.drift_z(),
                        window: Some(Window::Lifetime),
                        n_eff: Some(n),
                        rho: None,
                        lo: None,
                        hi: None,
                        state: Some(DriftState::Stable),
                        thin: None,
                        seeded: None,
                        axis_kind: Some(AxisKind::Rate),
                    });
                }
                for (pair, c) in st.confusion.iter() {
                    confusions.push(ConfusionStatus {
                        scope: k.clone(),
                        pair: pair.clone(),
                        ties: c.ties,
                        a_wins: c.a_wins,
                        b_wins: c.b_wins,
                        mean_margin: c.margin.mean,
                    });
                }
            }
            ledgers.push(LedgerStatus {
                ledger: l.as_str().into(),
                scopes: keys.len(),
                observations: obs,
                run_observations: run.map(|r| r.observations).unwrap_or(0),
                engine_revs: self.engine_revs(l),
            });
        }
        // Sort Shifted axes first, then by observation count descending.
        axes.sort_by(|a, b| {
            let sa = matches!(a.state, Some(DriftState::Shifted { .. }));
            let sb = matches!(b.state, Some(DriftState::Shifted { .. }));
            sb.cmp(&sa).then(b.n.cmp(&a.n)).then(a.scope.cmp(&b.scope)).then(a.axis.cmp(&b.axis))
        });
        axes.truncate(max_axes);
        confusions.sort_by(|a, b| b.ties.cmp(&a.ties).then(a.pair.cmp(&b.pair)));
        confusions.truncate(max_axes);
        SdsStatus {
            dir: self.dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "(memory)".into()),
            recipe: SDS_RECIPE.into(),
            ledgers,
            axes,
            confusions,
            log: self.log.iter().rev().take(30).cloned().collect(),
        }
    }

    pub fn scope_count(&self, track: Track) -> usize {
        let prefix = format!("{}|", track.as_str());
        let l = track.ledger();
        let mut keys: HashSet<&String> = HashSet::new();
        if let Some(b) = self.bases.get(&l) {
            keys.extend(b.scopes.keys().filter(|k| k.starts_with(&prefix)));
        }
        if let Some(r) = self.runs.get(&l) {
            keys.extend(r.scopes.keys().filter(|k| k.starts_with(&prefix)));
        }
        keys.len()
    }
}

fn fresh_file() -> SdsFile {
    SdsFile { recipe: SDS_RECIPE.into(), updated_at: now_ms(), scopes: HashMap::new(), engine_revs: Vec::new() }
}

fn canonical_json_no_ts(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut sorted: BTreeMap<String, serde_json::Value> = BTreeMap::new();
            for (k, val) in m {
                if k == "updated_at" {
                    continue;
                }
                sorted.insert(k, canonical_json_no_ts(val));
            }
            let mut out = serde_json::Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.into_iter().map(canonical_json_no_ts).collect()),
        other => other,
    }
}

fn reference_of(st: &ScopeStat, axis: &str) -> Option<ReferenceReport> {
    let v = st.baseline.get(axis)?;
    let d = st.baseline.get(&format!("{axis}:dev"))?;
    let b = st.agreement.get(&format!("{axis}:in_band"))?;
    let mean_ref = st.baseline.get(&format!("{axis}:ref")).map_or(f64::NAN, |w| w.mean);
    let mean_tol = st.baseline.get(&format!("{axis}:tol")).map_or(f64::NAN, |w| w.mean);
    let mean_abs_dev = st.baseline.get(&format!("{axis}:absdev")).filter(|w| w.n > 0).map(|w| w.mean);
    let rms_dev = st.baseline.get(&format!("{axis}:dev2")).filter(|w| w.n > 0).map(|w| w.mean.max(0.0).sqrt());
    Some(ReferenceReport { scope: String::new(), n: v.n, mean: v.mean, sd: v.sd(), mean_dev: d.mean, recent_dev: d.recent_mean(), in_band_rate: b.mean, mean_ref, mean_tol, mean_abs_dev, rms_dev })
}

fn axis_status(scope: &str, axis: &str, kind: &str, w: &Welford) -> AxisStatus {
    let tr = track_of_key(scope);
    let ak = if kind == "agreement" {
        AxisKind::Rate
    } else if axis.ends_with(":life_used") {
        AxisKind::Cumulative
    } else {
        AxisKind::Level
    };
    let (window, n_eff, rho, lo, hi, state, thin, seeded) = if let Some(t) = &w.t {
        let used = if t.seg.as_ref().is_some_and(|s| s.n >= 2) { Window::Span } else { Window::Long };
        let (m, s, raw_n, r) = match used {
            Window::Span => {
                let sp = t.seg.as_ref().unwrap();
                (sp.mean, sp.sd(), sp.n as f64, 0.0)
            }
            _ => (t.long.mean, t.long.sd(), t.long.n_eff(), t.long.rho()),
        };
        let n_e = effective_n(raw_n, r);
        let (_, ci_lo, ci_hi) = if ak == AxisKind::Rate { wilson_ci(m, n_e, 0.95) } else { mean_ci(m, s, n_e, 0.95) };
        let floor_n = tr.map_or(4, |x| x.temporal_policy().min_obs.floor_n);
        let thin = (n_e < floor_n as f64).then(|| format!("n_eff {n_e:.1} < {floor_n}"));
        (Some(used), Some(n_e), Some(r), Some(ci_lo), Some(ci_hi), Some(t.state), thin, Some(t.seeded))
    } else {
        (Some(Window::Lifetime), Some(w.n as f64), None, None, None, Some(DriftState::Stable), None, Some(false))
    };
    AxisStatus {
        scope: scope.to_string(),
        axis: axis.to_string(),
        kind: kind.into(),
        n: w.n,
        mean: w.mean,
        sd: w.sd(),
        recent: w.recent_mean(),
        drift_z: w.drift_z(),
        window,
        n_eff,
        rho,
        lo,
        hi,
        state,
        thin,
        seeded,
        axis_kind: Some(ak),
    }
}

impl Drop for SdsStore {
    fn drop(&mut self) {
        if self.dir.is_some() && (!self.dirty.is_empty() || !self.runs.is_empty()) {
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("noto_sds_{}_{}_{}", tag, std::process::id(), now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn feed(store: &mut SdsStore, scope: &Scope, axis: &str, n: usize, base: f64) {
        for i in 0..n {
            store.record_baseline(scope, axis, base + (i % 3) as f64 * 0.01);
        }
    }

    #[test]
    fn cold_start_answers_none_until_enough_observations() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "tree", "near");
        assert!(s.adaptive_baseline(&sc, "score").is_none());
        feed(&mut s, &sc, "score", 7, 0.6);
        assert!(s.adaptive_baseline(&sc, "score").is_none(), "primary needs 8, secondary 20, global 30");
        feed(&mut s, &sc, "score", 1, 0.6);
        let (_, key) = s.adaptive_baseline_scoped(&sc, "score").unwrap();
        assert_eq!(key, "vision|tree|", "falls back to the primary scope");
        feed(&mut s, &sc, "score", 12, 0.6);
        let (_, key) = s.adaptive_baseline_scoped(&sc, "score").unwrap();
        assert_eq!(key, "vision|tree|near", "secondary now has 20");
    }

    #[test]
    fn a_thin_secondary_scope_backs_off_to_a_thicker_primary() {
        let mut s = SdsStore::memory();
        let near = Scope::new(Track::Vision, "tree", "near");
        let far = Scope::new(Track::Vision, "tree", "far");
        feed(&mut s, &near, "score", 25, 0.70);
        feed(&mut s, &far, "score", 3, 0.30);
        let (m_far, _, _) = s.adaptive_baseline(&far, "score").unwrap();
        assert!(m_far > 0.6, "pooled mean {m_far}");
        let (m_near, _, n) = s.adaptive_baseline(&near, "score").unwrap();
        assert_eq!(n, 25);
        assert!((m_near - 0.71).abs() < 0.03);
    }

    #[test]
    fn calibration_is_shrunk_toward_one_when_data_is_thin() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "track", "f5");
        for _ in 0..4 {
            s.record_calibration(&sc, "mine_time", 1.5, 1.0);
        }
        let (f4, n4, _, _) = s.calibration_factor(&sc, "mine_time").unwrap();
        assert_eq!(n4, 4);
        let expected4 = 1.0 + 0.5 * (4.0 / 9.0);
        assert!((f4 - expected4).abs() < 1e-9, "{f4} vs {expected4}");
        for _ in 0..96 {
            s.record_calibration(&sc, "mine_time", 1.5, 1.0);
        }
        let (f100, _, _, _) = s.calibration_factor(&sc, "mine_time").unwrap();
        assert!(f100 > f4 && f100 < 1.5 && f100 > 1.45, "{f100}");
        let before = s.calibration_factor(&sc, "mine_time").unwrap().1;
        s.record_calibration(&sc, "mine_time", 1.0, 0.0);
        s.record_calibration(&sc, "mine_time", f64::NAN, 1.0);
        assert_eq!(s.calibration_factor(&sc, "mine_time").unwrap().1, before);
    }

    #[test]
    fn calibration_recentres_when_the_recent_values_drifted() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "wheel", "f5");
        for i in 0..40 {
            s.record_calibration(&sc, "drive_time", 1.0 + 0.01 * (i % 2) as f64, 1.0);
        }
        let (steady, _, d0, _) = s.calibration_factor(&sc, "drive_time").unwrap();
        assert!(d0.abs() < 2.0);
        for _ in 0..8 {
            s.record_calibration(&sc, "drive_time", 2.0, 1.0);
        }
        let (shifted, _, d1, _) = s.calibration_factor(&sc, "drive_time").unwrap();
        assert!(d1.abs() > 2.0, "drift z {d1}");
        assert!(shifted > steady + 0.3, "{shifted} vs {steady}");
    }

    #[test]
    fn confusion_pairs_are_symmetric_and_counted() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "", "");
        for _ in 0..6 {
            s.record_confusion(&sc, "coal_ore", "iron_ore_node", 0.02);
        }
        for _ in 0..4 {
            s.record_confusion(&sc, "iron_ore_node", "coal_ore", 0.03);
        }
        s.record_confusion(&sc, "tree", "tree", 0.1);
        s.record_confusion(&sc, "", "tree", 0.1);
        for _ in 0..20 {
            s.record_confusion(&sc, "tree", "boulder", 0.05);
        }
        let (share, ties) = s.confusion_share(&sc, "iron_ore_node", "coal_ore").unwrap();
        assert_eq!(ties, 10);
        assert!((share - 10.0 / 30.0).abs() < 1e-9);
        let st = s.status(10);
        let pair = st.confusions.iter().find(|c| c.pair == "coal_ore|iron_ore_node").unwrap();
        assert_eq!((pair.a_wins, pair.b_wins), (6, 4));
    }

    #[test]
    fn agreement_and_transition_priors() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "track", "");
        for i in 0..10 {
            s.record_agreement(&sc, "locate_ok", i % 5 != 0);
        }
        let (rate, n) = s.agreement_rate(&sc, "locate_ok").unwrap();
        assert_eq!(n, 10);
        assert!((rate - 0.8).abs() < 1e-9);
        for _ in 0..3 {
            s.record_transition(&sc, "locate", "approach");
        }
        s.record_transition(&sc, "locate", "locate");
        let p = s.transition_prior(&sc, "locate", "approach").unwrap();
        assert!((p - 4.0 / 7.0).abs() < 1e-9, "{p}");
        assert!(s.transition_prior(&sc, "mine", "collect").is_none());
        s.record_transition(&sc, "", "x");
        s.record_transition(&sc, "a", "");
    }

    #[test]
    fn gates_report_block_and_sole_rates() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "tree", "");
        for i in 0..10 {
            s.record_gates(&sc, &[("floor".into(), 2, i < 3, i == 0), ("temporal".into(), 1, i < 1, false)]);
        }
        let rates = s.gate_rates(&sc);
        assert_eq!(rates[0].gate, "floor");
        assert!((rates[0].blocked_rate - 0.3).abs() < 1e-9);
        assert!((rates[0].sole_rate - 0.1).abs() < 1e-9);
        assert!((rates[1].blocked_rate - 0.1).abs() < 1e-9);
        s.record_gates(&sc, &[]);
    }

    #[test]
    fn decay_margin_baseline_is_learned() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "", "");
        for _ in 0..30 {
            s.record_decay(&sc, "tile_scores", &[0.8, 0.3, 0.25, 0.2, 0.1]);
        }
        let (m, sd, n) = s.adaptive_decay(&sc, "tile_scores").unwrap();
        assert_eq!(n, 30);
        assert!((m - 0.5).abs() < 1e-6 && sd < 1e-6);
    }

    #[test]
    fn ledgers_survive_a_restart_and_leave_no_temp_files() {
        let dir = tmpdir("persist");
        {
            let mut s = SdsStore::open(&dir);
            let sc = Scope::new(Track::Forecast, "score:tree", "");
            for i in 0..5 {
                s.record_baseline(&sc, "fusion_mse:holt", 0.01 + i as f64 * 0.001);
            }
            s.close_runs();
        }
        let leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).collect();
        assert!(leftovers.is_empty());
        assert!(dir.join("task.json").exists());
        let run_files: Vec<_> = std::fs::read_dir(dir.join("runs").join("task")).unwrap().collect();
        assert_eq!(run_files.len(), 1);

        let s = SdsStore::open(&dir);
        let sc = Scope::new(Track::Forecast, "score:tree", "");
        let (mse, n) = s.fusion_mse(&sc, "holt").expect("5 observations are enough for the forecast track");
        assert_eq!(n, 5);
        assert!((mse - 0.012).abs() < 1e-9);
        assert!(s.log_lines().iter().any(|l| l.contains("원장 1종")), "{:?}", s.log_lines());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unflushed_run_data_is_flushed_on_drop() {
        let dir = tmpdir("drop");
        {
            let mut s = SdsStore::open(&dir);
            s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", 1.0);
        }
        assert!(dir.join("task.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_ledger_with_another_recipe_is_discarded_with_a_reason() {
        let dir = tmpdir("recipe");
        std::fs::create_dir_all(&dir).unwrap();
        let mut old = fresh_file();
        old.recipe = "noto-sds-v0:old".into();
        let mut st = ScopeStat::default();
        st.baseline.entry("score".into()).or_default().push(0.5, 4);
        old.scopes.insert("vision||".into(), st);
        std::fs::write(dir.join("perception.json"), serde_json::to_string(&old).unwrap()).unwrap();
        std::fs::write(dir.join("command.json"), "{ not json").unwrap();
        let s = SdsStore::open(&dir);
        let log = s.log_lines().join("\n");
        assert!(log.contains("레시피 세대 불일치"), "{log}");
        assert!(log.contains("읽기/파싱 실패"), "{log}");
        assert!(s.status(5).ledgers.iter().all(|l| l.observations == 0));
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_ledgers_drop_the_least_observed_scopes_first() {
        let dir = tmpdir("cap");
        let mut s = SdsStore::open(&dir);
        s.set_cap_bytes(6_000);
        let busy = Scope::new(Track::Motion, "busy", "");
        for i in 0..60 {
            s.record_baseline(&busy, "a", i as f64);
        }
        for i in 0..40 {
            let sc = Scope::new(Track::Motion, &format!("rare{i}"), "");
            s.record_baseline(&sc, "a", 1.0);
        }
        s.close_runs();
        let txt = std::fs::read_to_string(dir.join("task.json")).unwrap();
        assert!(txt.len() <= 6_000 + 400, "ledger is {} bytes", txt.len());
        let file: SdsFile = serde_json::from_str(&txt).unwrap();
        assert!(file.scopes.contains_key("motion|busy|"), "the well observed scope is kept");
        assert!(file.scopes.len() < 41);
        assert!(s.log_lines().iter().any(|l| l.contains("절삭")));
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_returns_to_cold_start() {
        let dir = tmpdir("purge");
        let mut s = SdsStore::open(&dir);
        let sc = Scope::new(Track::Forecast, "x", "");
        for _ in 0..5 {
            s.record_baseline(&sc, "a", 1.0);
        }
        s.close_runs();
        assert!(s.adaptive_baseline(&sc, "a").is_some());
        s.purge();
        assert!(s.adaptive_baseline(&sc, "a").is_none());
        assert!(!dir.join("task.json").exists());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomic_replaces_the_file_without_leaving_a_temp() {
        let dir = tmpdir("atomic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.json");
        write_atomic(&path, b"{\"a\":1}").unwrap();
        write_atomic(&path, b"{\"a\":2}").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":2}");
        assert!(!dir.join("a.json.tmp").exists());
        assert!(write_atomic(&dir.join("missing").join("b.json"), b"x").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_lists_axes_by_observation_count() {
        let mut s = SdsStore::memory();
        feed(&mut s, &Scope::new(Track::Vision, "tree", ""), "score", 12, 0.6);
        feed(&mut s, &Scope::new(Track::Motion, "track", ""), "mine_time", 3, 3.0);
        let st = s.status(3);
        assert_eq!(st.axes.len(), 3);
        assert!(st.axes[0].n >= st.axes[1].n && st.axes[1].n >= st.axes[2].n);
        assert!(st.ledgers.iter().any(|l| l.ledger == "perception" && l.observations > 0));
        assert_eq!(s.scope_count(Track::Vision), 2, "global and primary");
        let axes = s.axes_of(&Scope::new(Track::Vision, "tree", ""), "sc");
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0].kind, "baseline");
    }

    #[test]
    fn reference_axes_report_deviation_and_band_share() {
        let mut s = SdsStore::memory();
        let scope = Scope::new(Track::Balance, "arm_raise>belly", "flat");
        for _ in 0..4 {
            for v in [2.0, 2.25, 3.0] {
                s.record_reference(&scope, "comp_deg", v, 2.0, 0.5);
            }
        }
        let r = s.reference_report(&scope, "comp_deg").expect("12 observations reach the secondary minimum of 8");
        assert_eq!(r.scope, "balance|arm_raise>belly|flat");
        assert_eq!(r.n, 12);
        assert!((r.mean_dev - (0.0 + 0.5 + 2.0) / 3.0).abs() < 1e-9, "{r:?}");
        assert!((r.in_band_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.mean_ref - 2.0).abs() < 1e-12 && (r.mean_tol - 0.5).abs() < 1e-12);
        s.record_reference(&scope, "comp_deg", 9.0, 2.0, 0.0);
        assert_eq!(s.reference_report(&scope, "comp_deg").unwrap().n, 12);
        assert!(s.status(10).ledgers.iter().any(|l| l.ledger == "body" && l.observations > 0));
        assert_eq!(Track::Depth.ledger(), Ledger::Ml);
    }

    #[test]
    fn reference_reports_say_how_far_and_not_only_which_side() {
        let mut s = SdsStore::memory();
        let scope = Scope::new(Track::Act, "grasp:ts", "log");
        for _ in 0..4 {
            s.record_reference(&scope, "preload_s", 2.0, 1.0, 0.5);
            s.record_reference(&scope, "preload_s", 0.0, 1.0, 0.5);
        }
        let r = s.reference_report(&scope, "preload_s").unwrap();
        assert!(r.mean_dev.abs() < 1e-12, "{r:?}");
        assert!((r.mean_abs_dev.unwrap() - 2.0).abs() < 1e-12 && (r.rms_dev.unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(r.in_band_rate, 0.0);
        let e = s.exact_reference(&scope, "preload_s").unwrap();
        assert_eq!((e.mean_abs_dev, e.rms_dev, e.scope.as_str()), (r.mean_abs_dev, r.rms_dev, "act|grasp:ts|log"));
    }

    #[test]
    fn the_area_report_pools_every_reference_axis_of_a_prefix_once() {
        let mut s = SdsStore::memory();
        for item in ["log", "stone"] {
            for v in [1.0, 3.0] {
                s.record_reference(&Scope::new(Track::Act, "grasp:ts", item), "preload_s", v, 1.0, 1.0);
            }
        }
        for _ in 0..3 {
            s.record_reference(&Scope::new(Track::Act, "grasp>part", "reach"), "trunk_deg", 5.0, 1.0, 1.0);
        }
        s.record_reference(&Scope::new(Track::Act, "nn:grasp", "foot"), "margin", 9.0, 0.0, 1.0);
        for _ in 0..2 {
            s.record_baseline(&Scope::new(Track::Act, "grasp:old", ""), "x:dev", -0.5);
        }
        let r = s.area_report(Track::Act, "grasp");
        assert_eq!(r.axes.len(), 3, "{:?}", r.axes);
        assert_eq!(r.n, 4 + 3 + 2);
        assert_eq!(r.legacy_axes, 1);
        assert!((r.mean_abs_dev.unwrap() - 16.0 / 7.0).abs() < 1e-9, "{r:?}");
        assert!((r.rms_dev.unwrap() - (56.0f64 / 7.0).sqrt()).abs() < 1e-9);
        assert_eq!(r.worst.as_ref().map(|a| a.axis.as_str()), Some("trunk_deg"));
        assert_eq!(s.area_report(Track::Act, "").axes.len(), 4, "the whole track");
        assert!(s.area_report(Track::Vision, "").axes.is_empty());
    }

    #[test]
    fn a_scope_policy_keeps_model_modes_apart() {
        let mut s = SdsStore::memory();
        let real = Scope::new(Track::Locate, "parcel", "siglip2");
        let stand_in = Scope::new(Track::Locate, "parcel", "stand_in");
        feed(&mut s, &stand_in, "peak", 30, 0.9);
        feed(&mut s, &real, "peak", 3, 0.1);
        let (m, _, _) = s.adaptive_baseline(&real, "peak").unwrap();
        assert!(m > 0.7, "{m}");
        s.set_policy(Track::Locate, ScopePolicy::KEEP_SECONDARIES);
        assert!(s.adaptive_baseline(&real, "peak").is_none(), "the real model waits for its own observations");
        assert_eq!(s.adaptive_baseline_scoped(&stand_in, "peak").unwrap().1, "locate|parcel|stand_in");
        assert!(s.adaptive_baseline(&Scope::new(Track::Locate, "parcel", ""), "peak").is_some());
        let tips = Scope::new(Track::Depth, "tips", "");
        feed(&mut s, &Scope::new(Track::Depth, "stand_in", ""), "scale", 25, 1.25);
        feed(&mut s, &tips, "scale", 2, 0.9);
        assert!(s.adaptive_baseline(&tips, "scale").is_some(), "pooled through the global scope");
        s.set_policy(Track::Depth, ScopePolicy::EXACT);
        assert!(s.adaptive_baseline(&tips, "scale").is_none());
        assert_eq!(s.policy(Track::Vision), ScopePolicy::POOLED);
    }

    #[test]
    fn runs_and_ledgers_carry_the_engine_revision() {
        let dir = tmpdir("rev");
        {
            let mut s = SdsStore::open_with(&dir, OpenOptions::stamped("r1"));
            for i in 0..5 {
                s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", i as f64);
            }
            s.close_runs();
        }
        let ledger: SdsFile = serde_json::from_str(&std::fs::read_to_string(dir.join("task.json")).unwrap()).unwrap();
        assert_eq!(ledger.engine_revs, vec!["r1".to_string()]);
        let run = std::fs::read_dir(dir.join("runs").join("task")).unwrap().next().unwrap().unwrap().path();
        assert!(std::fs::read_to_string(run).unwrap().contains("\"engine_rev\": \"r1\""));
        {
            let mut s = SdsStore::open_with(&dir, OpenOptions::stamped("r2"));
            assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "x", ""), "a").is_some());
            s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", 9.0);
            assert_eq!(s.engine_revs(Ledger::Task), vec!["r1".to_string(), "r2".to_string()]);
            assert!(s.status(1).ledgers.iter().any(|l| l.ledger == "task" && l.engine_revs.len() == 2));
            s.close_runs();
        }
        let s = SdsStore::open_with(&dir, OpenOptions::only("r2"));
        assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "x", ""), "a").is_none());
        assert!(s.log_lines().iter().any(|l| l.contains("다른 엔진 개정")), "{:?}", s.log_lines());
        let aside: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("task.rev-")).collect();
        assert_eq!(aside.len(), 1);
        drop(s);
        let legacy = tmpdir("rev_legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        let mut old = fresh_file();
        let mut st = ScopeStat::default();
        st.baseline.entry("a".into()).or_default().push(1.0, 4);
        old.scopes.insert("forecast||".into(), st);
        let txt = serde_json::to_string(&old).unwrap().replace(",\"engine_revs\":[]", "");
        std::fs::write(legacy.join("task.json"), txt).unwrap();
        let s = SdsStore::open_with(&legacy, OpenOptions::stamped("r2"));
        assert_eq!(s.engine_revs(Ledger::Task), vec![UNSTAMPED.to_string()]);
        drop(s);
        let s = SdsStore::open_with(&legacy, OpenOptions::only(""));
        assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "", ""), "a").is_none(), "one observation is below the minimum");
        assert_eq!(s.engine_revs(Ledger::Task), vec![UNSTAMPED.to_string()]);
        assert!(!s.log_lines().iter().any(|l| l.contains("다른 엔진 개정")), "{:?}", s.log_lines());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&legacy);
    }

    #[test]
    fn legacy_v1_ledger_migrates() {
        let dir = tmpdir("migrate_v1");
        let mut s = SdsStore::open(&dir);
        for i in 0..10 {
            s.record_baseline(&Scope::new(Track::Balance, "nn:s4", "brake_1.0"), "score", 70.0 + i as f64);
            s.record_baseline(&Scope::new(Track::Depth, "tips", "tree"), "scale", 1.05 + i as f64 * 0.01);
        }
        s.close_runs();
        let old_mean = s.merged_scope(Ledger::Body, "balance|nn:s4|brake_1.0").unwrap().baseline["score"].mean;
        let old_n = s.merged_scope(Ledger::Body, "balance|nn:s4|brake_1.0").unwrap().baseline["score"].n;
        let (migrated, legacy_kept) = s.migrate_v1_to_v2();
        assert!(migrated >= 2 && legacy_kept == 0);
        let new_st = s.merged_scope(Ledger::Body, "balance/nn:s4;ph=brake_1.0").expect("migrated nn:s4 scope");
        assert_eq!(new_st.baseline["score"].n, old_n);
        assert!((new_st.baseline["score"].mean - old_mean).abs() < 1e-9);
        assert!(s.merged_scope(Ledger::Ml, "depth/tree;mode=tips").is_some());
        let has_backup = std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().contains(".v1-"));
        assert!(has_backup, "v1 backup must be preserved on disk");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_match_event_is_counted_once_upstream() {
        let mut s = SdsStore::memory();
        let s1 = Scope::new(Track::Balance, "nn:s1", "slope");
        let s6 = Scope::new(Track::Balance, "nn:s6", "slope");
        s.record_batch("ev#1", |b| {
            b.record_baseline(&s1, "score", 85.0);
            b.record_baseline(&s6, "score", 85.0);
        });
        assert_eq!(s.merged_scope(Ledger::Body, "balance|nn:s1|slope").unwrap().baseline["score"].n, 1);
        assert_eq!(s.merged_scope(Ledger::Body, "balance|nn:s6|slope").unwrap().baseline["score"].n, 1);
        assert_eq!(s.merged_scope(Ledger::Body, "balance||").unwrap().baseline["score"].n, 1, "shared upstream balance|| must count once");
    }

    #[test]
    fn old_ledger_loads_and_seeds_temporal() {
        let dir = tmpdir("seed_t");
        std::fs::create_dir_all(&dir).unwrap();
        let raw = r#"{"recipe":"noto-sds-v1:welford+ring/hier-write/ledger-split/calib-shrink/atomic","updated_at":1,"engine_revs":["r1"],"scopes":{"vision|tree|":{"baseline":{"score":{"n":40,"mean":0.72,"m2":0.39,"ring":[0.7,0.74]}},"updated_at":1}}}"#;
        std::fs::write(dir.join("perception.json"), raw).unwrap();
        let s = SdsStore::open(&dir);
        let est = s.estimate_mean(&Scope::new(Track::Vision, "tree", ""), "score", Window::Long, 0.95).unwrap();
        assert!((est.value - 0.72).abs() < 1e-6);
        assert!(est.n == 40 && est.lo < est.value && est.value < est.hi);
        let w = s.merged_scope(Ledger::Perception, "vision|tree|").unwrap().baseline["score"].clone();
        assert!(w.temporal_or_seeded(16.0, 128.0).seeded);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fingerprint_change_opens_a_generation_and_keeps_the_parent() {
        let dir = tmpdir("gen");
        {
            let mut g1 = SdsStore::open_generation(&dir, "r1", "aaaaaaaaaaaa");
            for _ in 0..30 {
                g1.record_baseline(&Scope::new(Track::Vision, "tree", ""), "score", 0.78);
                g1.record_calibration(&Scope::new(Track::Balance, "accel>foot", ""), "ff", 1.3, 1.0);
            }
            g1.close_runs();
        }
        // Fingerprint change under same engine_rev -> Inherit generation
        let g2 = SdsStore::open_generation(&dir, "r1", "bbbbbbbbbbbb");
        assert_eq!(g2.generation_id(), Some("r1+bbbbbbbbbbbb"));
        let st = g2.generations_status();
        assert_eq!(st.parent.as_deref(), Some("r1+aaaaaaaaaaaa"));
        assert_eq!(st.inherit, "inherit");
        assert!(dir.join("gen").join("r1+aaaaaaaaaaaa").join("perception.json").exists(), "parent files kept");
        // Path A (Vision::score) inherits prior shrunk to primary min_obs (8)
        let (_, _, n_a) = g2.adaptive_baseline(&Scope::new(Track::Vision, "tree", ""), "score").expect("path A inherited prior");
        assert_eq!(n_a, Track::Vision.min_obs().1);
        // Path F (Balance::ff) is Inherit::Never -> cold start
        assert!(g2.calibration_factor(&Scope::new(Track::Balance, "accel>foot", ""), "ff").is_none(), "path F must never inherit across generations");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recipe_mismatch_and_purge_archive_instead_of_deleting() {
        let dir = tmpdir("archive");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("perception.json"), r#"{"recipe":"old-v0","updated_at":1,"scopes":{}}"#).unwrap();
        let mut s = SdsStore::open(&dir);
        let arch_recipe = std::fs::read_dir(dir.join("archive")).unwrap().flatten().any(|e| e.file_name().to_string_lossy().ends_with("-recipe"));
        assert!(arch_recipe, "recipe mismatch must archive instead of deleting");
        for _ in 0..5 {
            s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", 1.0);
        }
        s.close_runs();
        let moved = s.purge();
        assert!(moved.is_some());
        assert!(s.archive_bytes() > 0);
        let arch_purge = std::fs::read_dir(dir.join("archive")).unwrap().flatten().any(|e| e.file_name().to_string_lossy().ends_with("-purge"));
        assert!(arch_purge, "purge must archive instead of deleting");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn phase_switching_alone_does_not_alarm() {
        // Alternating phase means (90 vs 55) with phase-expected profile residual CUSUM should stay Stable,
        // while a simultaneous +1.5σ shift across all phases triggers Shifted.
        let mut t = Temporal::default();
        let m_s = 12.0;
        let m_l = 96.0;
        for i in 0..120 {
            let phase_mu = if i % 2 == 0 { 90.0 } else { 55.0 };
            let noise = if (i / 2) % 2 == 0 { 2.0 } else { -2.0 };
            t.push_with_expected(phase_mu + noise, Some(phase_mu), m_s, m_l, 0.5, 5.0);
        }
        assert_eq!(t.state, DriftState::Stable, "pure phase switching with expected profile must not alarm");
        let mut shifted = false;
        for i in 0..25 {
            let phase_mu = if i % 2 == 0 { 90.0 } else { 55.0 };
            t.push_with_expected(phase_mu + 12.0, Some(phase_mu), m_s, m_l, 0.5, 5.0);
            if matches!(t.state, DriftState::Shifted { .. }) {
                shifted = true;
                break;
            }
        }
        assert!(shifted, "common shift across all phases must trigger Shifted");
    }
}
