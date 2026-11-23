use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::act::cocarry::{self, Carrier, CarryOpts, Load};
use crate::balance::{reference, BalanceReport, Body, FeedForward, Posture, Source, G};
use crate::base::BaseKind;
use crate::dynamics::{simulate, DynCfg, DynDigest};
use crate::hand::FingerCount;
use crate::morph::{Cargo, FORMS};
use crate::parts::{BodyPlan, Segment};
use crate::surface::Surface;
use crate::wear;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    ArmRaise,
    ArmLift,
    FootLift,
    StepUp,
    Slope,
    CarryFront,
    Start,
    Brake,
}

impl Cause {
    pub const ALL: [Cause; 8] = [
        Cause::ArmRaise,
        Cause::ArmLift,
        Cause::FootLift,
        Cause::StepUp,
        Cause::Slope,
        Cause::CarryFront,
        Cause::Start,
        Cause::Brake,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Cause::ArmRaise => "arm_raise",
            Cause::ArmLift => "arm_lift",
            Cause::FootLift => "foot_lift",
            Cause::StepUp => "step_up",
            Cause::Slope => "slope",
            Cause::CarryFront => "carry_front",
            Cause::Start => "start",
            Cause::Brake => "brake",
        }
    }
}

/// Motion requirement for live phase matching (Phase 1 – Part 1 · D4).
/// Offline `evaluate` ignores this field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Motion {
    #[default]
    Any,
    Still,
    Moving,
}

/// Scenario definition family (Phase 6 – Part 1 · D5, Part 2 · D1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioKind {
    #[default]
    Phase,
    Sequence,
    Aggregate,
    Crew,
    RecordOnly,
    Reserved,
}

/// Live reachability classification of a phase in `world-sim` (Phase 6 – Part 1 · D2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LiveReach {
    #[default]
    Expected,
    NeedsAdapter { fields: &'static [&'static str] },
    OfflineOnly { reason: &'static str },
}

impl LiveReach {
    pub fn label(&self) -> String {
        match self {
            LiveReach::Expected => "expected".into(),
            LiveReach::NeedsAdapter { fields } => format!("needs_adapter:{}", fields.join(",")),
            LiveReach::OfflineOnly { .. } => "offline_only".into(),
        }
    }
}

/// Distance/regime classification of a live matched phase relative to development conditions (Phase 6 – Part 1 · D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchClass {
    Dev,
    Between,
    HoldoutLike,
}

/// Form applicability classification for a scenario on a concrete [`Body`] (Phase 6 – Part 2 · D4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormCoverage {
    Evaluated,
    Partial,
    Substituted,
    NotApplicable,
}

/// Body capability token required by a scenario or phase (Phase 6 – Part 2 · D4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cap {
    Legs,
    Arms,
    Arms2,
    Neck,
    Waist,
    Cargo,
    Bay,
    Tote,
    Bed,
}

/// Compute the physical capability set of `body` directly from its morphology & actuators.
pub fn body_caps(body: &Body) -> BTreeSet<Cap> {
    let mut caps = BTreeSet::new();
    if body.legged() {
        caps.insert(Cap::Legs);
    }
    if body.arm_count() >= 1 {
        caps.insert(Cap::Arms);
    }
    if body.arm_count() >= 2 {
        caps.insert(Cap::Arms2);
    }
    if body.has_neck() {
        caps.insert(Cap::Neck);
    }
    if body.has_torso() && body.belly.pitch.is_some() {
        caps.insert(Cap::Waist);
    }
    match body.morph.cargo {
        Cargo::Bay => {
            caps.insert(Cap::Cargo);
            caps.insert(Cap::Bay);
        }
        Cargo::Tote => {
            caps.insert(Cap::Cargo);
            caps.insert(Cap::Tote);
        }
        Cargo::Bed { .. } => {
            caps.insert(Cap::Cargo);
            caps.insert(Cap::Bed);
        }
        Cargo::None => {}
    }
    caps
}

pub fn causes_of(p: &Posture) -> BTreeSet<Cause> {
    let mut c = BTreeSet::new();
    if p.source == Source::ArmRaise || p.arm_elev_deg[0] >= 60.0 {
        c.insert(Cause::ArmRaise);
    }
    if p.source == Source::ArmLift || p.hand_load_kg.iter().any(|l| *l > 0.0) {
        c.insert(Cause::ArmLift);
    }
    if p.single_support {
        c.insert(Cause::FootLift);
    }
    if p.step_m > 0.0 {
        c.insert(Cause::StepUp);
    }
    if p.slope_deg.abs() >= 3.0 {
        c.insert(Cause::Slope);
    }
    if p.cargo_kg > 0.0 {
        c.insert(Cause::CarryFront);
    }
    if p.accel_mps2 > 0.0 {
        c.insert(Cause::Start);
    }
    if p.accel_mps2 < 0.0 {
        c.insert(Cause::Brake);
    }
    c
}

pub const SEGMENTS: [Segment; 7] = Segment::ALL;

/// Reference table revision: increment when adding or modifying human reference rules.
pub const REF_REV: u32 = 1;

/// Why a cell is or is not scored against a human reference (Phase 6 – Part 1 · D1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefState {
    Human,
    RobotLimit,
    #[default]
    Raw,
    Mismatch,
    Absent,
}

/// Effective scored weight and sensitivity bounds when unscored segments take 0 or 100.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Coverage {
    pub coverage_w: f32,
    pub score_lo: f32,
    pub score_hi: f32,
    pub partial: bool,
}

pub fn coverage(seg_w: &[f32; 7], seg_scores: &[Option<f32>; 7]) -> Coverage {
    let (mut num, mut den, mut miss) = (0.0f32, 0.0f32, 0.0f32);
    let max_w = seg_w.iter().copied().fold(0.0f32, f32::max);
    let mut max_missing = false;
    for i in 0..7 {
        match seg_scores[i] {
            Some(s) => {
                num += seg_w[i] * s;
                den += seg_w[i];
            }
            None => {
                miss += seg_w[i];
                if max_w > 0.0 && (seg_w[i] - max_w).abs() < 1e-5 {
                    max_missing = true;
                }
            }
        }
    }
    let all = (den + miss).max(1e-6);
    let coverage_w = den / all;
    Coverage {
        coverage_w,
        score_lo: num / all,
        score_hi: (num + 100.0 * miss) / all,
        partial: coverage_w < 0.5 || max_missing,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhaseSpec {
    pub id: String,
    pub subsystem: &'static str,
    pub weight: f32,
    pub posture: Posture,
    pub causes: Vec<Cause>,
    pub ladder: Option<&'static str>,
    pub imposed: bool,
    #[serde(default)]
    pub motion: Motion,
    #[serde(default)]
    pub ground: Option<Surface>,
    #[serde(default)]
    pub live_reach: LiveReach,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioSpec {
    pub id: &'static str,
    pub name_ko: &'static str,
    #[serde(default)]
    pub kind: ScenarioKind,
    pub seg_w: [f32; 7],
    pub weights_from: &'static str,
    pub needs_legs: bool,
    pub holdout: bool,
    pub phases: Vec<PhaseSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cell {
    pub source: Source,
    pub target: Segment,
    pub axis: String,
    pub value: f32,
    pub reference: Option<f32>,
    pub tolerance: Option<f32>,
    pub cite: Option<&'static str>,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub dynamic: bool,
    pub ref_state: RefState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_note: Option<&'static str>,
}

/// Structured gate trigger on a phase evaluation (Phase 5 – Part 1 · D3).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GateHit {
    pub code: &'static str,
    pub measured: f32,
    pub limit: f32,
    pub excused: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhaseResult {
    pub id: String,
    pub subsystem: &'static str,
    pub weight: f32,
    pub ladder: Option<&'static str>,
    pub cells: Vec<Cell>,
    pub seg_scores: [Option<f32>; 7],
    pub score: f32,
    pub ungated_score: f32,
    pub coverage_w: f32,
    pub score_bounds: [f32; 2],
    pub partial: bool,
    pub item: u8,
    pub gates: Vec<String>,
    pub gate_codes: Vec<&'static str>,
    pub gate_hits: Vec<GateHit>,
    pub beyond_human: Vec<String>,
    pub static_margin: f32,
    pub min_margin: f32,
    pub peak_ratio: f32,
    pub worst_joint: Option<String>,
    pub sustained_effort: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    #[serde(default)]
    pub kind: ScenarioKind,
    pub holdout: bool,
    pub skipped: Option<String>,
    pub phases: Vec<PhaseResult>,
    pub score: f32,
    pub gated: bool,
    pub gates: Vec<String>,
    pub passed: bool,
    pub weights_from: &'static str,
}

fn cause_set(c: &[Cause]) -> Vec<Cause> {
    c.to_vec()
}

/// Co-carry load template for `crew` scenarios (`C1`–`C3`, Phase 6 – Part 2 · D3).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CoCarryTpl {
    #[serde(default = "default_cocarry_kind")]
    pub kind: String,
    #[serde(default = "default_cocarry_len")]
    pub length_m: f32,
    #[serde(default = "default_cocarry_diam")]
    pub diameter_m: f32,
    #[serde(default)]
    pub slope_deg: f32,
    #[serde(default = "default_load_factor")]
    pub load_factor: f32,
    #[serde(default)]
    pub let_go: Option<usize>,
    #[serde(default)]
    pub even_axial: bool,
}

fn default_cocarry_kind() -> String {
    "log".into()
}
fn default_cocarry_len() -> f32 {
    2.0
}
fn default_cocarry_diam() -> f32 {
    0.18
}
fn default_load_factor() -> f32 {
    1.4
}

/// Posture template in `data/nn_scenarios.json` resolved against a concrete [`Body`] (Phase 6 – Part 2 · D1).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PostureTpl {
    #[serde(default = "default_base")]
    pub base: String,
    #[serde(default)]
    pub source: Option<Source>,
    #[serde(default)]
    pub moving: Option<bool>,
    #[serde(default)]
    pub load_frac: Option<f32>,
    #[serde(default)]
    pub cargo_frac: Option<f32>,
    #[serde(default)]
    pub slope_deg: Option<f32>,
    #[serde(default)]
    pub step_m: Option<f32>,
    #[serde(default)]
    pub single_support: Option<String>,
    #[serde(default)]
    pub accel_fixed: Option<f32>,
    #[serde(default)]
    pub accel_planner: Option<f32>,
    #[serde(default)]
    pub two_arms: Option<String>,
    #[serde(default)]
    pub arm_elev_deg: Option<[f32; 2]>,
    #[serde(default)]
    pub elbow_deg: Option<[f32; 2]>,
    #[serde(default)]
    pub ground: Option<Surface>,
    #[serde(default)]
    pub floor_amp_mps2: Option<f32>,
    #[serde(default)]
    pub floor_hz: Option<f32>,
}

fn default_base() -> String {
    "quiet".into()
}

/// Phase definition loaded from `data/nn_scenarios.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PhaseDef {
    pub id: String,
    #[serde(default)]
    pub stage: String,
    pub subsystem: String,
    pub weight: f32,
    pub posture: PostureTpl,
    pub causes: Vec<Cause>,
    #[serde(default)]
    pub legged_causes: Option<Vec<Cause>>,
    #[serde(default)]
    pub ladder: Option<String>,
    #[serde(default)]
    pub imposed: bool,
    #[serde(default)]
    pub motion: Motion,
    #[serde(default)]
    pub live: bool,
    #[serde(default)]
    pub offline_reason: Option<String>,
    #[serde(default)]
    pub dwell_s: Option<f32>,
    #[serde(default)]
    pub requires: Vec<Cap>,
    #[serde(default)]
    pub ground: Option<Surface>,
    #[serde(default)]
    pub cocarry: Option<CoCarryTpl>,
    #[serde(default)]
    pub context: Option<serde_json::Value>,
}

/// Scenario definition loaded from `data/nn_scenarios.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScenarioDef {
    pub id: String,
    pub name_ko: String,
    #[serde(default)]
    pub kind: ScenarioKind,
    #[serde(default)]
    pub core: bool,
    pub seg_w: [f32; 7],
    pub weights_from: String,
    #[serde(default)]
    pub needs_legs: bool,
    #[serde(default)]
    pub requires: Vec<Cap>,
    #[serde(default)]
    pub substitute_when_partial: bool,
    #[serde(default)]
    pub live: bool,
    #[serde(default)]
    pub repeat: Option<usize>,
    #[serde(default)]
    pub rest_s: Option<f32>,
    #[serde(default)]
    pub series: Vec<String>,
    #[serde(default)]
    pub of: Vec<(String, f32)>,
    #[serde(default)]
    pub phases: Vec<PhaseDef>,
}

const NN_DEFS_RAW: &str = include_str!("../../../data/nn_scenarios.json");
static DEFS: OnceLock<Vec<ScenarioDef>> = OnceLock::new();
static DEFS_REV: OnceLock<String> = OnceLock::new();

/// Static list of data-driven N:N scenario definitions from `data/nn_scenarios.json`.
pub fn defs() -> &'static [ScenarioDef] {
    DEFS.get_or_init(|| serde_json::from_str(NN_DEFS_RAW).expect("valid data/nn_scenarios.json")).as_slice()
}

/// Short 8-hex revision digest of `data/nn_scenarios.json` for ledger/audit provenance.
pub fn defs_rev() -> &'static str {
    DEFS_REV
        .get_or_init(|| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in NN_DEFS_RAW.as_bytes() {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            format!("{:08x}", (h ^ (h >> 32)) as u32)
        })
        .as_str()
}

fn lift_posture(body: &Body, kg: f32, slope: f32) -> Posture {
    let two = !body.solve(&Posture { slope_deg: slope, ..Posture::arm_lift(kg, false, 0.0) }, &FeedForward::default(), 1e3).feasible;
    Posture { slope_deg: slope, ..Posture::arm_lift(kg, two, 0.0) }
}

fn resolve_posture(tpl: &PostureTpl, body: &Body) -> Posture {
    let mass = body.mass_kg();
    let cargo = tpl.cargo_frac.unwrap_or(0.0) * mass;
    let slope = tpl.slope_deg.unwrap_or(0.0);
    let mut p = match tpl.base.as_str() {
        "arm_lift" => {
            let kg = tpl.load_frac.unwrap_or(0.10) * mass;
            match tpl.two_arms.as_deref() {
                Some("true") => Posture { slope_deg: slope, ..Posture::arm_lift(kg, true, cargo) },
                Some("false") => Posture { slope_deg: slope, ..Posture::arm_lift(kg, false, cargo) },
                _ => {
                    let mut lp = lift_posture(body, kg, slope);
                    lp.cargo_kg = cargo;
                    lp
                }
            }
        }
        "arm_raise" => Posture { slope_deg: slope, ..Posture::arm_raise(cargo) },
        "slope_start" => {
            let s = tpl.slope_deg.unwrap_or(10.0);
            if body.legged() {
                Posture { source: Source::FootLift, single_support: true, slope_deg: s, ..Posture::quiet(cargo) }
            } else {
                let a = body.planned_accel(&body.base.spec(), s, cargo, tpl.ground.unwrap_or(Surface::Firm), 1.0, PLANNER_MIN_RAMP_S);
                Posture { source: Source::Accel, accel_mps2: a, slope_deg: s, ..Posture::quiet(cargo) }
            }
        }
        _ => Posture::quiet(cargo),
    };
    if let Some(src) = tpl.source {
        p.source = src;
    }
    if let Some(m) = tpl.moving {
        p.moving = m;
    }
    if let Some(s) = tpl.slope_deg {
        p.slope_deg = s;
    }
    if let Some(st) = tpl.step_m {
        p.step_m = st;
    }
    if let Some(ss) = tpl.single_support.as_deref() {
        p.single_support = match ss {
            "true" => true,
            "false" => false,
            "legged" => body.legged(),
            _ => p.single_support,
        };
    }
    if let Some(a) = tpl.accel_fixed {
        p.accel_mps2 = a;
    } else if let Some(dir) = tpl.accel_planner {
        p.accel_mps2 = dir.signum() * body.planned_accel(&body.base.spec(), p.slope_deg, p.cargo_kg, tpl.ground.unwrap_or(p.ground), dir.signum(), PLANNER_MIN_RAMP_S);
    }
    if let Some(ae) = tpl.arm_elev_deg {
        p.arm_elev_deg = ae;
    }
    if let Some(el) = tpl.elbow_deg {
        p.elbow_deg = el;
    }
    if let Some(g) = tpl.ground {
        p.ground = g;
    }
    if let Some(fa) = tpl.floor_amp_mps2 {
        p.floor_amp_mps2 = fa;
    }
    if let Some(fh) = tpl.floor_hz {
        p.floor_hz = fh;
    }
    p
}

/// Resolve a static [`ScenarioDef`] against `body` into a concrete [`ScenarioSpec`].
pub fn resolve(def: &'static ScenarioDef, body: &Body) -> ScenarioSpec {
    let walk = body.legged();
    let phases = def
        .phases
        .iter()
        .map(|ph| {
            let mut posture = resolve_posture(&ph.posture, body);
            if let Some(g) = ph.ground {
                posture.ground = g;
            }
            let causes = if walk {
                ph.legged_causes.clone().unwrap_or_else(|| ph.causes.clone())
            } else {
                ph.causes.clone()
            };
            let live_reach = if ph.live {
                LiveReach::Expected
            } else {
                LiveReach::OfflineOnly {
                    reason: ph.offline_reason.as_deref().unwrap_or("오프라인 전용 위상"),
                }
            };
            PhaseSpec {
                id: ph.id.clone(),
                subsystem: ph.subsystem.as_str(),
                weight: ph.weight,
                posture,
                causes,
                ladder: ph.ladder.as_deref(),
                imposed: ph.imposed,
                motion: ph.motion,
                ground: ph.ground,
                live_reach,
            }
        })
        .collect();
    ScenarioSpec {
        id: def.id.as_str(),
        name_ko: def.name_ko.as_str(),
        kind: def.kind,
        seg_w: def.seg_w,
        weights_from: def.weights_from.as_str(),
        needs_legs: def.needs_legs || def.requires.contains(&Cap::Legs),
        holdout: false,
        phases,
    }
}

fn phase(id: &str, subsystem: &'static str, weight: f32, posture: Posture, causes: &[Cause]) -> PhaseSpec {
    PhaseSpec {
        id: id.into(),
        subsystem,
        weight,
        posture,
        causes: cause_set(causes),
        ladder: None,
        imposed: false,
        motion: Motion::Any,
        ground: None,
        live_reach: LiveReach::Expected,
    }
}

pub fn def_by_id(id: &str) -> Option<&'static ScenarioDef> {
    defs().iter().find(|d| d.id == id)
}

pub fn library(body: &Body) -> Vec<ScenarioSpec> {
    defs()
        .iter()
        .filter(|d| d.core && d.kind == ScenarioKind::Phase)
        .map(|d| resolve(d, body))
        .collect()
}

/// Resolve all executable definitions (core + extended phase/sequence/crew/record_only) for `body`.
pub fn all_scenarios(body: &Body) -> Vec<ScenarioSpec> {
    defs()
        .iter()
        .filter(|d| !d.phases.is_empty() && d.kind != ScenarioKind::Reserved && d.kind != ScenarioKind::Aggregate)
        .map(|d| resolve(d, body))
        .collect()
}

/// Resolve the S5-A moving-floor record-only scenario specification if present in `defs()`.
pub fn s5a_spec(body: &Body) -> Option<ScenarioSpec> {
    def_by_id("S5-A").map(|d| resolve(d, body))
}

/// Evaluate capability coverage of `def` on a concrete `body` (Phase 6 – Part 2 · D4).
pub fn check_capabilities(def: &ScenarioDef, body: &Body) -> (FormCoverage, Option<String>) {
    if def.kind == ScenarioKind::Reserved {
        return (FormCoverage::NotApplicable, Some("예약 시나리오".into()));
    }
    let caps = body_caps(body);
    if def.needs_legs && !caps.contains(&Cap::Legs) {
        return (FormCoverage::NotApplicable, Some("필요 능력 부족: legs".into()));
    }
    let missing: Vec<&str> = def
        .requires
        .iter()
        .filter(|c| !caps.contains(c))
        .map(|c| match c {
            Cap::Legs => "legs",
            Cap::Arms => "arms",
            Cap::Arms2 => "arms2",
            Cap::Neck => "neck",
            Cap::Waist => "waist",
            Cap::Cargo => "cargo",
            Cap::Bay => "bay",
            Cap::Tote => "tote",
            Cap::Bed => "bed",
        })
        .collect();
    if !missing.is_empty() {
        return (FormCoverage::NotApplicable, Some(format!("필요 능력 부족: {}", missing.join(", "))));
    }
    if def.phases.is_empty() {
        return (FormCoverage::Evaluated, None);
    }
    let ok_phases = def.phases.iter().filter(|ph| ph.requires.iter().all(|c| caps.contains(c))).count();
    if ok_phases == def.phases.len() {
        (FormCoverage::Evaluated, None)
    } else if ok_phases > 0 {
        if def.substitute_when_partial {
            (FormCoverage::Substituted, Some("일부 위상 대체 적용".into()))
        } else {
            (FormCoverage::Partial, Some(format!("위상 {ok_phases}/{} 적용", def.phases.len())))
        }
    } else {
        (FormCoverage::NotApplicable, Some("적용 가능한 위상 없음".into()))
    }
}

/// Form × scenario coverage matrix across all 7 [`morph::FORMS`] (Phase 6 – Part 2 · D4).
pub fn coverage_matrix() -> BTreeMap<String, BTreeMap<String, FormCoverage>> {
    let mut out = BTreeMap::new();
    for form in FORMS.iter() {
        let base = form.morph.lower.base();
        let plan = BodyPlan { morph: form.morph, ..BodyPlan::default() };
        let body = Body::new(&plan, base, FingerCount::Five);
        let mut row = BTreeMap::new();
        for def in defs() {
            row.insert(def.id.clone(), check_capabilities(def, &body).0);
        }
        out.insert(form.id.to_string(), row);
    }
    out
}

/// Collect all development condition magnitudes from `defs_slice` for automatic holdout disjointness checks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DevConditions {
    pub slopes_deg: Vec<f32>,
    pub steps_m: Vec<f32>,
    pub brakes_mps2: Vec<f32>,
    pub load_fracs: Vec<f32>,
}

pub fn dev_conditions(defs_slice: &[ScenarioDef]) -> DevConditions {
    let mut c = DevConditions::default();
    for d in defs_slice {
        for ph in &d.phases {
            if let Some(s) = ph.posture.slope_deg {
                if s.abs() > 0.0 {
                    c.slopes_deg.push(s.abs());
                }
            }
            if let Some(st) = ph.posture.step_m {
                if st > 0.0 {
                    c.steps_m.push(st);
                }
            }
            if let Some(a) = ph.posture.accel_fixed {
                if a < 0.0 {
                    c.brakes_mps2.push(a.abs());
                }
            }
            if let Some(lf) = ph.posture.load_frac {
                if lf > 0.0 {
                    c.load_fracs.push(lf);
                }
            }
            if let Some(cf) = ph.posture.cargo_frac {
                if cf > 0.0 {
                    c.load_fracs.push(cf);
                }
            }
            if let Some(cc) = &ph.cocarry {
                if cc.slope_deg.abs() > 0.0 {
                    c.slopes_deg.push(cc.slope_deg.abs());
                }
            }
        }
    }
    c
}

pub const PLANNER_MIN_RAMP_S: f32 = 0.5;

pub const HOLDOUT_SLOPE_DEG: (f32, f32) = (12.0, 18.0);
pub const HOLDOUT_STEP_M: (f32, f32) = (0.12, 0.16);
pub const HOLDOUT_BRAKE: [(f32, f32); 2] = [(0.75, 0.95), (1.7, 2.2)];
pub const HOLDOUT_LOAD_FRAC: [(f32, f32); 2] = [(0.05, 0.08), (0.12, 0.15)];

pub fn holdout(body: &Body, seed: u64, count: usize) -> Vec<ScenarioSpec> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let pick = |r: (f32, f32), u: f32| r.0 + (r.1 - r.0) * u;
    let mut out = Vec::new();
    for i in 0..count {
        let slope = pick(HOLDOUT_SLOPE_DEG, next()) * if next() < 0.5 { 1.0 } else { -1.0 };
        let lf = HOLDOUT_LOAD_FRAC[(next() * 2.0) as usize % 2];
        let load = pick(lf, next()) * body.mass_kg();
        let br = HOLDOUT_BRAKE[(next() * 2.0) as usize % 2];
        let brake = pick(br, next());
        let step = pick(HOLDOUT_STEP_M, next());
        let kind = (next() * 6.0) as usize % 6;
        let (name, needs_legs, phases): (&'static str, bool, Vec<PhaseSpec>) = match kind {
            0 => (
                "경사 + 들기",
                false,
                vec![phase("lift", "anticipatory", 1.0, lift_posture(body, load, slope), &[Cause::ArmLift, Cause::Slope])],
            ),
            1 => (
                "짐 + 제동",
                false,
                vec![PhaseSpec {
                    imposed: true,
                    ..phase("brake", "reactive", 1.0, Posture { source: Source::Accel, accel_mps2: -brake, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake])
                }],
            ),
            2 => (
                "짐 + 턱 → 제동",
                true,
                vec![
                    phase("step", "anticipatory", 0.5, Posture { source: Source::StepUp, step_m: step, single_support: body.legged(), ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::StepUp]),
                    PhaseSpec {
                        imposed: true,
                        ..phase("brake", "reactive", 0.5, Posture { source: Source::Accel, accel_mps2: -brake, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake])
                    },
                ],
            ),
            3 => (
                "경사 → 들기",
                false,
                vec![
                    phase("slope", "sensory", 0.5, Posture { source: Source::Slope, slope_deg: slope, ..Posture::quiet(0.0) }, &[Cause::Slope]),
                    phase("lift", "anticipatory", 0.5, lift_posture(body, load, slope), &[Cause::ArmLift, Cause::Slope]),
                ],
            ),
            4 => (
                "경사 → 짐 운반 → 제동",
                false,
                vec![
                    phase("slope", "sensory", 0.333, Posture { source: Source::Slope, slope_deg: slope, ..Posture::quiet(0.0) }, &[Cause::Slope]),
                    phase("carry_slope", "gait", 0.333, Posture { source: Source::Slope, slope_deg: slope, moving: true, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Slope]),
                    PhaseSpec {
                        imposed: true,
                        ..phase("brake", "reactive", 0.334, Posture { source: Source::Accel, accel_mps2: -brake, slope_deg: slope, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake, Cause::Slope])
                    },
                ],
            ),
            _ => (
                "짐 + 턱",
                true,
                vec![phase("step", "anticipatory", 1.0, Posture { source: Source::StepUp, step_m: step, single_support: body.legged(), ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::StepUp])],
            ),
        };
        let id: &'static str = Box::leak(format!("H{}-{i}", seed % 1000).into_boxed_str());
        out.push(ScenarioSpec {
            id,
            name_ko: name,
            kind: ScenarioKind::Phase,
            seg_w: [1.0 / 7.0; 7],
            weights_from: "DARPA DRC: 과제당 통과/실패, 동일 가중",
            needs_legs,
            holdout: true,
            phases,
        });
    }
    out
}

pub fn composite_reference(source: Source, target: Segment, axis: &str, accel: f32) -> Option<(f32, f32, &'static str)> {
    match (source, target, axis) {
        (Source::Accel, Segment::Foot, "los_frac") => {
            let lim = if accel < 0.0 { 6.25f32 } else { 4.45 }.to_radians().tan();
            let v = (accel.abs() / G) / lim;
            Some((v, 0.3 * v.max(0.05), "LOS 6.25°/4.45° (Jbabdi 2008), 내딛기 문턱 1.0/0.7 m/s² (de Kam 2017 via Krašna 2021) — 유도"))
        }
        (Source::FootLift, Segment::Belly, "pelvis_lat_acc") => Some((0.43, 0.18, "보행 시작 골반 좌우 가속 0.044 ± 0.018 g (Schlenstedt 2018)")),
        (_, Segment::Foot, "ff_dip_ratio") => Some((0.47, 0.14, "예측 가능한 교란의 CoP 최대 28/60 mm (Santos 2010) — 유도")),
        (Source::ArmLift, Segment::Belly, "slope_moment_ratio") => Some((1.068, 0.05, "경사 들기 L5/S1 모멘트 +6.8% (Mirka & Shin 2004) — 허용폭 선택")),
        _ => None,
    }
}

fn scored(c: &mut Cell) {
    if let (Some(reference), Some(tolerance)) = (c.reference, c.tolerance) {
        let v = crate::bench::judge(Some(c.value), crate::bench::Rule::Like { reference, tolerance });
        c.z = v.z;
        c.score = v.score;
    }
}

fn classify_ref_state(source: Source, target: Segment, axis: &str, single_support: bool, has_ref: bool) -> (RefState, Option<&'static str>) {
    if has_ref {
        return (RefState::Human, None);
    }
    if single_support && source == Source::ArmRaise && target == Segment::Foot && matches!(axis, "ankle_rise" | "com_shift_frac") {
        return (RefState::Mismatch, Some("사람 기준은 두 발 정지 조건(한 발 지지 불일치)"));
    }
    if target == Segment::Hand && axis == "grip_effort" {
        return (RefState::RobotLimit, Some("로봇 손 파지 정격 대비 비율"));
    }
    (RefState::Raw, None)
}

pub fn cells(body: &Body, post: &Posture, r: &BalanceReport, d: Option<&DynDigest>) -> Vec<Cell> {
    let ctx = r.ref_ctx_for(body.mass_kg(), post.single_support);
    let mut out: Vec<Cell> = r
        .couplings
        .iter()
        .map(|c| {
            let rf = reference(c.source, c.target, &c.axis, &ctx);
            let (ref_state, ref_note) = classify_ref_state(c.source, c.target, &c.axis, post.single_support, rf.is_some());
            let mut cell = Cell {
                source: c.source,
                target: c.target,
                axis: c.axis.clone(),
                value: c.value,
                reference: rf.as_ref().map(|x| x.value),
                tolerance: rf.as_ref().map(|x| x.tolerance),
                cite: rf.as_ref().map(|x| x.cite),
                z: None,
                score: None,
                dynamic: false,
                ref_state,
                ref_note,
            };
            scored(&mut cell);
            cell
        })
        .collect();
    let mut add = |target: Segment, axis: &str, value: f32| {
        let rf = composite_reference(post.source, target, axis, post.accel_mps2).or_else(|| reference(post.source, target, axis, &ctx).map(|x| (x.value, x.tolerance, x.cite)));
        let (ref_state, ref_note) = classify_ref_state(post.source, target, axis, post.single_support, rf.is_some());
        let mut cell = Cell {
            source: post.source,
            target,
            axis: axis.into(),
            value,
            reference: rf.map(|x| x.0),
            tolerance: rf.map(|x| x.1),
            cite: rf.map(|x| x.2),
            z: None,
            score: None,
            dynamic: true,
            ref_state,
            ref_note,
        };
        scored(&mut cell);
        out.push(cell);
    };
    if let Some(d) = d {
        add(Segment::Foot, "margin", d.min_margin);
        add(Segment::Head, "tilt_deg", d.head_pitch_max_deg);
        if post.single_support {
            add(Segment::Belly, "pelvis_lat_acc", d.pelvis_lat_acc_max);
        }
        if let Some(gain) = d.sway_gain {
            add(Segment::Foot, "sway_gain", gain);
        }
        if let Some(phase_deg) = d.sway_phase_deg {
            add(Segment::Foot, "sway_phase_deg", phase_deg);
        }
    }
    if post.accel_mps2 != 0.0 {
        let s = r.support;
        let edge = if post.accel_mps2 < 0.0 { s[1] - r.com[0] } else { r.com[0] - s[0] };
        add(Segment::Foot, "los_frac", r.com[2] * post.accel_mps2.abs() / G / edge.max(1e-3));
    }
    out
}

fn seg_index(s: Segment) -> usize {
    SEGMENTS.iter().position(|x| *x == s).unwrap_or(0)
}

pub fn score_phase(spec: &PhaseSpec, seg_w: &[f32; 7], cells: Vec<Cell>, r: &BalanceReport, d: Option<&DynDigest>) -> PhaseResult {
    let mut sums = [0.0f32; 7];
    let mut counts = [0u32; 7];
    let mut worst_z = 0.0f32;
    for c in &cells {
        if let (Some(s), Some(z)) = (c.score, c.z) {
            let i = seg_index(c.target);
            sums[i] += s;
            counts[i] += 1;
            worst_z = worst_z.max(z.abs());
        }
    }
    let seg_scores: [Option<f32>; 7] = std::array::from_fn(|i| (counts[i] > 0).then(|| sums[i] / counts[i] as f32));
    let (num, den) = (0..7).filter_map(|i| seg_scores[i].map(|s| (seg_w[i] * s, seg_w[i]))).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let mut gates = Vec::new();
    let mut gate_codes: Vec<&'static str> = Vec::new();
    let mut gate_hits: Vec<GateHit> = Vec::new();
    let human_steps = spec.imposed && cells.iter().any(|c| c.axis == "los_frac" && c.reference.map_or(false, |v| v >= 1.0));
    let mut beyond = Vec::new();
    if !r.feasible {
        let why = format!("정적 불가: {}", r.limiting.clone().unwrap_or_default());
        let exc = human_steps && r.margin < 0.0 && r.worst().map_or(true, |w| w.peak_ratio <= 1.0);
        gate_hits.push(GateHit { code: "static_infeasible", measured: r.margin, limit: 0.0, excused: exc });
        if exc {
            beyond.push(why);
        } else {
            gates.push(why);
            gate_codes.push("infeasible");
        }
    }
    if let Some(d) = d {
        if d.min_margin < 0.0 {
            let why = format!("동적 여유 {:.2} < 0", d.min_margin);
            gate_hits.push(GateHit { code: "dyn_margin", measured: d.min_margin, limit: 0.0, excused: human_steps });
            if human_steps {
                beyond.push(why);
            } else {
                gates.push(why);
                gate_codes.push("margin_neg");
            }
        }
        if d.peak_ratio > 1.0 {
            gates.push(format!("{} 피크 {:.0}%", d.worst_joint.clone().unwrap_or_default(), d.peak_ratio * 100.0));
            gate_codes.push("peak_over");
            gate_hits.push(GateHit { code: "joint_peak", measured: d.peak_ratio, limit: 1.0, excused: false });
        }
        if !d.reached {
            gates.push("목표 자세 미도달".into());
            gate_codes.push("unreached");
            gate_hits.push(GateHit { code: "not_reached", measured: 0.0, limit: 1.0, excused: false });
        }
        if d.head_pitch_max_deg > 20.0 {
            gates.push(format!("머리 기울기 {:.0}° > 20° (Pozzo 1990)", d.head_pitch_max_deg));
            gate_codes.push("head_tilt");
            gate_hits.push(GateHit { code: "head_tilt", measured: d.head_pitch_max_deg, limit: 20.0, excused: false });
        }
    }
    let cov = coverage(seg_w, &seg_scores);
    let ungated_score = if den > 0.0 { num / den } else { 100.0 };
    let score = if gates.is_empty() { ungated_score } else { 0.0 };
    let item = if !gates.is_empty() { 0 } else if worst_z <= 1.0 { 2 } else if worst_z <= 2.0 { 1 } else { 0 };
    let sustained = r.joints.iter().map(|j| j.effort).fold(0.0f32, f32::max);
    PhaseResult {
        id: spec.id.clone(),
        subsystem: spec.subsystem,
        weight: spec.weight,
        ladder: spec.ladder,
        cells,
        seg_scores,
        score,
        ungated_score,
        coverage_w: cov.coverage_w,
        score_bounds: [cov.score_lo, cov.score_hi],
        partial: cov.partial,
        item,
        gates,
        gate_codes,
        gate_hits,
        beyond_human: beyond,
        static_margin: r.margin,
        min_margin: d.map_or(r.margin, |d| d.min_margin),
        peak_ratio: d.map_or(r.worst().map_or(0.0, |w| w.peak_ratio), |d| d.peak_ratio),
        worst_joint: d.and_then(|d| d.worst_joint.clone()),
        sustained_effort: sustained,
    }
}

pub fn evaluate(body: &Body, spec: &ScenarioSpec, ff: &FeedForward) -> ScenarioResult {
    let cfg = DynCfg::default();
    if spec.needs_legs && !body.legged() {
        return ScenarioResult { id: spec.id, name_ko: spec.name_ko, kind: spec.kind, holdout: spec.holdout, skipped: Some("다리가 없는 베이스에는 이 동작(17 cm 턱 · 한 발 서기)이 없음".into()), phases: vec![], score: 0.0, gated: false, gates: vec![], passed: false, weights_from: spec.weights_from };
    }
    let mut prev = Posture::quiet(spec.phases.first().map_or(0.0, |p| p.posture.cargo_kg));
    let mut phases = Vec::new();
    for ph in &spec.phases {
        let p = ph.posture;
        let grip = 1e3;
        let report = body.solve(&p, ff, grip);
        let roll = simulate(body, &prev, &p, ff, grip, &cfg);
        let digest = roll.digest();
        let mut cs = cells(body, &p, &report, Some(&digest));
        if ff.enabled {
            let off = simulate(body, &prev, &p, &FeedForward { enabled: false, gain: 1.0 }, grip, &cfg);
            let (dip_ff, dip_off) = (digest.static_margin - digest.min_margin, off.static_margin - off.min_margin);
            if dip_off > 0.01 {
                let rf = composite_reference(p.source, Segment::Foot, "ff_dip_ratio", 0.0);
                let mut c = Cell { source: p.source, target: Segment::Foot, axis: "ff_dip_ratio".into(), value: (dip_ff / dip_off).max(0.0), reference: rf.map(|x| x.0), tolerance: rf.map(|x| x.1), cite: rf.map(|x| x.2), z: None, score: None, dynamic: true, ref_state: if rf.is_some() { RefState::Human } else { RefState::Raw }, ref_note: None };
                scored(&mut c);
                cs.push(c);
            }
        }
        if p.source == Source::ArmLift && p.slope_deg.abs() >= 3.0 {
            let flat = Posture { slope_deg: 0.0, ..p };
            let f = simulate(body, &Posture { slope_deg: 0.0, ..prev }, &flat, ff, grip, &cfg);
            let moment = |r: &crate::dynamics::Rollout| -> f32 {
                let k = r.joints.iter().position(|j| j.name == "waist_pitch").unwrap_or(0);
                r.samples.iter().map(|s| s.tau_nm[k]).fold(0.0, f32::max)
            };
            let (ms, mf) = (moment(&roll), moment(&f));
            if mf > 1e-3 {
                let rf = composite_reference(Source::ArmLift, Segment::Belly, "slope_moment_ratio", 0.0);
                let mut c = Cell { source: p.source, target: Segment::Belly, axis: "slope_moment_ratio".into(), value: ms / mf, reference: rf.map(|x| x.0), tolerance: rf.map(|x| x.1), cite: rf.map(|x| x.2), z: None, score: None, dynamic: true, ref_state: if rf.is_some() { RefState::Human } else { RefState::Raw }, ref_note: None };
                scored(&mut c);
                cs.push(c);
            }
        }
        phases.push(score_phase(ph, &spec.seg_w, cs, &report, Some(&digest)));
        prev = p;
    }
    combine(spec, phases)
}

pub fn combine(spec: &ScenarioSpec, phases: Vec<PhaseResult>) -> ScenarioResult {
    let mut parts: Vec<(f32, f32)> = Vec::new();
    let mut ladders: std::collections::BTreeMap<&'static str, (f32, f32)> = std::collections::BTreeMap::new();
    for p in &phases {
        match p.ladder {
            Some(l) => {
                let e = ladders.entry(l).or_insert((p.weight, f32::INFINITY));
                e.1 = e.1.min(p.score);
            }
            None => parts.push((p.weight, p.score)),
        }
    }
    parts.extend(ladders.values().copied());
    let wsum: f32 = parts.iter().map(|x| x.0).sum();
    let gates: Vec<String> = phases.iter().flat_map(|p| p.gates.iter().map(move |g| format!("{}: {g}", p.id))).collect();
    let gated = !gates.is_empty();
    let score = if gated { 0.0 } else { parts.iter().map(|(w, s)| w * s).sum::<f32>() / wsum.max(1e-6) };
    let passed = !gated && phases.iter().all(|p| p.item >= 1);
    ScenarioResult { id: spec.id, name_ko: spec.name_ko, kind: spec.kind, holdout: spec.holdout, skipped: None, phases, score, gated, gates, passed, weights_from: spec.weights_from }
}

/// Core NIOSH SLI formula shared by offline `shift_index` and live `ShiftLedger::sli` (Phase 6 – Part 1 · D4).
/// Each tuple is `(li, lmax, share)` where `share = task_duration / shift_duration`.
pub fn sli_from_tasks(tasks: &[(f32, f32, f32)]) -> Option<f32> {
    if tasks.is_empty() {
        return None;
    }
    let mut v = tasks.to_vec();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (li1, lmax1, _) = v[0];
    let k = v.iter().map(|t| t.1 * t.2).sum::<f32>() / lmax1.max(1e-6);
    Some(li1 + (lmax1 - li1) * k)
}

pub fn shift_index(results: &[(&ScenarioResult, f32)]) -> Option<f32> {
    let tasks: Vec<(f32, f32, f32)> = results
        .iter()
        .filter(|(r, _)| r.skipped.is_none() && !r.phases.is_empty())
        .map(|(r, minutes)| {
            let li = r.phases.iter().map(|p| p.peak_ratio).fold(0.0, f32::max);
            let lmax = r.phases.iter().map(|p| p.sustained_effort).fold(0.0, f32::max).max(li);
            (li, lmax, minutes / 240.0)
        })
        .collect();
    sli_from_tasks(&tasks)
}

/// Time-step (s) used for offline thermal & wear integration (matches `world-sim` 10 Hz tick).
pub const SEQ_DT_S: f32 = 0.1;

/// Repetition schedule for `kind: "sequence"` scenarios (Phase 6 – Part 2 · D2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeqPlan {
    pub repeat: usize,
    pub dwell_s: Vec<f32>,
    pub rest_s: f32,
}

impl SeqPlan {
    pub fn cycle_s(&self) -> f32 {
        self.dwell_s.iter().sum::<f32>() + self.rest_s.max(0.0)
    }

    pub fn from_def(def: &ScenarioDef) -> Self {
        let dwell_s = def.phases.iter().map(|p| p.dwell_s.unwrap_or(5.0)).collect();
        SeqPlan {
            repeat: def.repeat.unwrap_or(6).max(1),
            dwell_s,
            rest_s: def.rest_s.unwrap_or(0.0),
        }
    }
}

/// Per-repetition time-series sample produced by [`evaluate_sequence`].
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RepSample {
    pub rep: usize,
    pub t_s: f32,
    pub heat_max: f32,
    pub wear_max: f32,
    pub min_margin: f32,
    pub peak_ratio: f32,
    pub gated: bool,
}

/// Output of [`evaluate_sequence`] (Phase 6 – Part 2 · D2).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceResult {
    pub base: ScenarioResult,
    pub reps: Vec<RepSample>,
    pub first_gate_rep: Option<usize>,
    pub heat_max: f32,
    pub wear_max: f32,
    pub sli: Option<f32>,
    pub gates: Vec<String>,
}

/// Evaluate a time-dependent repeating sequence scenario (`T1`–`T3`, `E1`, `E2`),
/// integrating thermal heat via [`wear::heat_step`] and wear via [`wear::damage`].
pub fn evaluate_sequence(body: &Body, spec: &ScenarioSpec, plan: &SeqPlan, ff: &FeedForward) -> SequenceResult {
    let base = evaluate(body, spec, ff);
    if base.skipped.is_some() || spec.phases.is_empty() {
        return SequenceResult {
            base,
            reps: Vec::new(),
            first_gate_rep: None,
            heat_max: 0.0,
            wear_max: 0.0,
            sli: None,
            gates: Vec::new(),
        };
    }
    let reports: Vec<BalanceReport> = spec.phases.iter().map(|ph| body.solve(&ph.posture, ff, 1e3)).collect();
    let base_min_margin = base.phases.iter().map(|p| p.min_margin).fold(f32::INFINITY, f32::min);
    let base_peak = base.phases.iter().map(|p| p.peak_ratio).fold(0.0f32, f32::max);
    let mut heat: BTreeMap<String, f32> = BTreeMap::new();
    let mut wear_acc: BTreeMap<String, f32> = BTreeMap::new();
    let mut reps = Vec::with_capacity(plan.repeat);
    let mut first_gate_rep = None;
    let mut t_s = 0.0f32;
    let mut task_entries: Vec<(f32, f32, f32)> = Vec::new();
    let total_dur_s = (plan.cycle_s() * plan.repeat as f32).max(1e-3);
    for (i, ph) in spec.phases.iter().enumerate() {
        let d = plan.dwell_s.get(i).copied().unwrap_or(5.0);
        if let Some(pr) = base.phases.get(i) {
            let li = pr.peak_ratio;
            let lmax = pr.sustained_effort.max(li);
            let frac = (d * plan.repeat as f32) / total_dur_s;
            task_entries.push((li, lmax, frac));
        }
        let _ = ph;
    }
    for rep in 0..plan.repeat {
        for (i, rep_sol) in reports.iter().enumerate() {
            let dwell = plan.dwell_s.get(i).copied().unwrap_or(5.0).max(0.0);
            let steps = (dwell / SEQ_DT_S).round().max(0.0) as usize;
            for j in &rep_sol.joints {
                let tau = body.thermal_tau_s(&j.joint);
                let h = heat.entry(j.joint.clone()).or_insert(0.0);
                for _ in 0..steps {
                    *h = wear::heat_step(*h, j.effort, SEQ_DT_S, tau);
                }
                if let Some(act) = body.joint_actuator(&j.joint) {
                    *wear_acc.entry(j.joint.clone()).or_insert(0.0) += wear::damage(&act, j.torque_nm, dwell);
                }
            }
            t_s += dwell;
        }
        if plan.rest_s > 0.0 {
            let r_steps = (plan.rest_s / SEQ_DT_S).round().max(0.0) as usize;
            for (joint, h) in &mut heat {
                let tau = body.thermal_tau_s(joint);
                for _ in 0..r_steps {
                    *h = wear::heat_step(*h, 0.0, SEQ_DT_S, tau);
                }
            }
            t_s += plan.rest_s;
        }
        let heat_max = heat.values().copied().fold(0.0f32, f32::max);
        let wear_max = wear_acc.values().copied().fold(0.0f32, f32::max);
        let gated = heat_max > 1.0 || base_peak > 1.0 || base_min_margin < 0.0;
        if gated && first_gate_rep.is_none() {
            first_gate_rep = Some(rep);
        }
        reps.push(RepSample {
            rep,
            t_s,
            heat_max,
            wear_max,
            min_margin: base_min_margin,
            peak_ratio: base_peak,
            gated,
        });
    }
    let heat_max = reps.iter().map(|r| r.heat_max).fold(0.0f32, f32::max);
    let wear_max = reps.last().map_or(0.0, |r| r.wear_max);
    let mut gates = base.gates.clone();
    if heat_max > 1.0 {
        gates.push(format!("열 정격 초과: heat_max {:.2} > 1.0", heat_max));
    }
    SequenceResult {
        base,
        reps,
        first_gate_rep,
        heat_max,
        wear_max,
        sli: sli_from_tasks(&task_entries),
        gates,
    }
}

/// Output of [`evaluate_aggregate`] (`S7` shift index computed from definition `of` table).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AggregateResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub sli: Option<f32>,
    pub tasks: Vec<(String, f32)>,
    pub passed: bool,
}

pub fn evaluate_aggregate(def: &'static ScenarioDef, results: &[ScenarioResult]) -> AggregateResult {
    let pairs: Vec<(&ScenarioResult, f32)> = def
        .of
        .iter()
        .filter_map(|(id, mins)| results.iter().find(|r| r.id == id.as_str()).map(|r| (r, *mins)))
        .collect();
    let sli = shift_index(&pairs);
    AggregateResult {
        id: def.id.as_str(),
        name_ko: def.name_ko.as_str(),
        sli,
        tasks: def.of.clone(),
        passed: sli.is_some_and(|v| v <= 1.0),
    }
}

/// Per-member evaluation inside a 2-worker cooperative carry scenario (`C1`–`C3`, Phase 6 – Part 2 · D3).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CrewMemberEval {
    pub member: usize,
    pub load_kg: f32,
    pub share: f32,
    pub ok: bool,
    pub phase: PhaseResult,
}

/// Output of [`evaluate_crew`] (`C1`–`C3`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CrewResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub load_kg: f32,
    pub team_ok: bool,
    pub share_ratio: f32,
    pub axial_n: f32,
    pub weak_util: f32,
    pub instant_n: Option<f32>,
    pub members: Vec<CrewMemberEval>,
    pub team_cells: Vec<Cell>,
    pub gates: Vec<String>,
    pub skipped: Option<String>,
}

/// Evaluate a `kind: "crew"` cooperative-carry definition (`C1`–`C3`) using `act::cocarry::share` and `let_go`.
pub fn evaluate_crew(def: &'static ScenarioDef, body_a: &Body, body_b: &Body, ff: &FeedForward) -> CrewResult {
    let (cov_a, why_a) = check_capabilities(def, body_a);
    let (cov_b, why_b) = check_capabilities(def, body_b);
    if cov_a == FormCoverage::NotApplicable || cov_b == FormCoverage::NotApplicable {
        return CrewResult {
            id: def.id.as_str(),
            name_ko: def.name_ko.as_str(),
            load_kg: 0.0,
            team_ok: false,
            share_ratio: 1.0,
            axial_n: 0.0,
            weak_util: 0.0,
            instant_n: None,
            members: Vec::new(),
            team_cells: Vec::new(),
            gates: Vec::new(),
            skipped: why_a.or(why_b).or_else(|| Some("협업 능력 부족".into())),
        };
    }
    let Some(ph_def) = def.phases.first() else {
        return CrewResult {
            id: def.id.as_str(),
            name_ko: def.name_ko.as_str(),
            load_kg: 0.0,
            team_ok: false,
            share_ratio: 1.0,
            axial_n: 0.0,
            weak_util: 0.0,
            instant_n: None,
            members: Vec::new(),
            team_cells: Vec::new(),
            gates: Vec::new(),
            skipped: Some("위상 없음".into()),
        };
    };
    let tpl = ph_def.cocarry.clone().unwrap_or_default();
    let cap_a = cocarry::member_capacity_kg(body_a, Surface::Firm, tpl.slope_deg, cocarry::carry_hand_z(body_a));
    let cap_b = cocarry::member_capacity_kg(body_b, Surface::Firm, tpl.slope_deg, cocarry::carry_hand_z(body_b));
    let weak_cap = cap_a.min(cap_b).max(0.5);
    let mass_kg = (weak_cap * tpl.load_factor).max(0.5);
    let load = Load::log(mass_kg, tpl.length_m, tpl.diameter_m, false);
    let span = (tpl.length_m - 0.4).max(0.6);
    let dz = span * tpl.slope_deg.to_radians().tan();
    let mut ca = Carrier::at(body_a, 0.0, 0.0, 0.0, Surface::Firm);
    let mut cb = Carrier::at(body_b, span, 0.0, dz, Surface::Firm);
    ca.slope_deg = tpl.slope_deg;
    cb.slope_deg = tpl.slope_deg;
    let carriers = [ca, cb];
    let opts = CarryOpts { even_axial: tpl.even_axial, ..CarryOpts::default() };
    let cc = cocarry::share(&load, &carriers, &opts);
    let lg = tpl.let_go.map(|idx| cocarry::let_go(&load, &carriers, idx.min(1), &opts));
    let instant_n = lg.as_ref().and_then(|x| x.instant_n.first().copied());
    let team_ok = match &lg {
        Some(x) => !x.dropped && x.end_held,
        None => cc.ok,
    };
    let share_ratio = cc.share_ratio();
    let axial_n = cc.axial_even_n.abs().max(cc.axial_n.abs());
    let weak_util = cc
        .members
        .iter()
        .map(|m| m.load_kg / m.capacity_kg.max(1e-3))
        .fold(0.0f32, f32::max);
    let raw_cell = |axis: &str, value: f32| Cell {
        source: Source::ArmLift,
        target: Segment::Belly,
        axis: axis.into(),
        value,
        reference: None,
        tolerance: None,
        cite: None,
        z: None,
        score: None,
        dynamic: false,
        ref_state: RefState::Raw,
        ref_note: None,
    };
    let mut team_cells = vec![
        raw_cell("share_ratio", share_ratio),
        raw_cell("axial_n", axial_n),
        raw_cell("weak_util", weak_util),
    ];
    if let Some(inst) = instant_n {
        team_cells.push(raw_cell("instant_n", inst));
    }
    let spec = resolve(def, body_a);
    let ph_spec = &spec.phases[0];
    let bodies = [body_a, body_b];
    let mut members = Vec::with_capacity(2);
    let mut gates = Vec::new();
    for (idx, m) in cc.members.iter().enumerate() {
        let b = bodies[idx];
        let post = Posture { slope_deg: tpl.slope_deg, ..Posture::arm_lift(m.load_kg, true, 0.0) };
        let rep = b.solve(&post, ff, 1e3);
        let cs = cells(b, &post, &rep, None);
        let pr = score_phase(ph_spec, &spec.seg_w, cs, &rep, None);
        if !m.ok {
            gates.push(format!("구성원 {}: {}", idx + 1, m.why_ko.clone().unwrap_or_else(|| "능력 초과".into())));
        }
        members.push(CrewMemberEval {
            member: idx,
            load_kg: m.load_kg,
            share: m.share,
            ok: m.ok,
            phase: pr,
        });
    }
    if !team_ok && gates.is_empty() {
        gates.push(if lg.is_some() { "놓침 교란 후 물체 낙하/한계 초과".into() } else { "팀 운반 조건 불충족".into() });
    }
    CrewResult {
        id: def.id.as_str(),
        name_ko: def.name_ko.as_str(),
        load_kg: mass_kg,
        team_ok,
        share_ratio,
        axial_n,
        weak_util,
        instant_n,
        members,
        team_cells,
        gates,
        skipped: None,
    }
}

/// Entry in the 27-pair cause combination map (Phase 6 – Part 2 · D5).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CausePairEntry {
    pub pair: (Cause, Cause),
    pub used_by: Vec<String>,
    pub live_reachable: bool,
    pub refs: usize,
    pub forms: Vec<String>,
}

/// Enumerate representative postures that `world-sim` can generate across all action branches.
pub fn live_world_postures(body: &Body) -> Vec<Posture> {
    let cargo = 0.10 * body.mass_kg();
    let legged = body.legged();
    let mut out = Vec::new();
    for g in [Surface::Firm, Surface::WetSand] {
        for c in [0.0f32, cargo] {
            for slope in [-10.0f32, 0.0, 4.76, 10.0] {
                // Still on slope / flat
                if slope.abs() >= 3.0 {
                    out.push(Posture { source: Source::Slope, slope_deg: slope, ground: g, moving: false, ..Posture::quiet(c) });
                    out.push(Posture { source: Source::Slope, slope_deg: slope, ground: g, moving: true, ..Posture::quiet(c) });
                } else {
                    out.push(Posture { ground: g, moving: false, ..Posture::quiet(c) });
                    if c > 0.0 {
                        out.push(Posture { source: Source::CarryFront, ground: g, moving: true, ..Posture::quiet(c) });
                    }
                }
                // Work: Mine (ArmRaise) & Pick/Deliver (ArmLift) with floor slope
                out.push(Posture { slope_deg: slope, ground: g, ..Posture::arm_raise(c) });
                out.push(Posture { slope_deg: slope, ground: g, ..Posture::arm_lift(cargo.max(0.5), false, c) });
                // Move: Start / Brake
                for a in [0.5f32, -0.5, -1.0, -1.5, -2.5] {
                    out.push(Posture { source: Source::Accel, accel_mps2: a, slope_deg: slope, ground: g, moving: true, ..Posture::quiet(c) });
                }
                // Legged FootLift & StepUp
                if legged {
                    out.push(Posture { source: Source::FootLift, single_support: true, slope_deg: slope, ground: g, moving: true, ..Posture::quiet(c) });
                    out.push(Posture { source: Source::StepUp, step_m: 0.17, single_support: true, slope_deg: slope, ground: g, moving: true, ..Posture::quiet(c) });
                }
            }
        }
    }
    out
}

/// Enumerate all 27 feasible cause pairs (8 choose 2 minus impossible Start+Brake) against `defs_slice`.
pub fn cause_map(defs_slice: &[ScenarioDef]) -> Vec<CausePairEntry> {
    let mut live_pairs: BTreeSet<(Cause, Cause)> = BTreeSet::new();
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        let body = Body::new(&BodyPlan::default(), base, FingerCount::Five);
        for p in live_world_postures(&body) {
            let cs: Vec<Cause> = causes_of(&p).into_iter().collect();
            for i in 0..cs.len() {
                for j in (i + 1)..cs.len() {
                    let pair = if cs[i] < cs[j] { (cs[i], cs[j]) } else { (cs[j], cs[i]) };
                    live_pairs.insert(pair);
                }
            }
        }
    }
    let ctx = crate::balance::RefCtx { load_ratio: 0.10, load_dist_m: 0.25, slope_deg: 10.0, single_support: false };
    let cause_to_source = |c: Cause| match c {
        Cause::ArmRaise => Source::ArmRaise,
        Cause::ArmLift => Source::ArmLift,
        Cause::FootLift => Source::FootLift,
        Cause::StepUp => Source::StepUp,
        Cause::Slope => Source::Slope,
        Cause::CarryFront => Source::CarryFront,
        Cause::Start | Cause::Brake => Source::Accel,
    };
    let ref_axes = ["tilt_deg", "peak_ratio", "com_shift_frac", "ankle_rise", "restore_frac", "shift_frac", "margin", "cop_shift_frac", "los_frac", "pelvis_lat_acc", "slope_moment_ratio"];
    let mut out = Vec::with_capacity(27);
    for i in 0..Cause::ALL.len() {
        for j in (i + 1)..Cause::ALL.len() {
            let (a, b) = (Cause::ALL[i], Cause::ALL[j]);
            if (a == Cause::Start && b == Cause::Brake) || (a == Cause::Brake && b == Cause::Start) {
                continue;
            }
            let used_by: Vec<String> = defs_slice
                .iter()
                .filter(|d| {
                    d.phases.iter().any(|ph| {
                        let has = |cs: &[Cause]| cs.contains(&a) && cs.contains(&b);
                        has(&ph.causes) || ph.legged_causes.as_deref().is_some_and(has)
                    })
                })
                .map(|d| d.id.clone())
                .collect();
            let mut segs_with_ref: BTreeSet<Segment> = BTreeSet::new();
            for c in [a, b] {
                let src = cause_to_source(c);
                let accel = if c == Cause::Brake { -1.0 } else if c == Cause::Start { 1.0 } else { 0.0 };
                for seg in SEGMENTS {
                    for ax in ref_axes {
                        if reference(src, seg, ax, &ctx).is_some() || composite_reference(src, seg, ax, accel).is_some() {
                            segs_with_ref.insert(seg);
                        }
                    }
                }
            }
            let forms: Vec<String> = FORMS
                .iter()
                .filter(|f| {
                    let needs_legs = matches!(a, Cause::FootLift | Cause::StepUp) || matches!(b, Cause::FootLift | Cause::StepUp);
                    let needs_arms = matches!(a, Cause::ArmRaise | Cause::ArmLift) || matches!(b, Cause::ArmRaise | Cause::ArmLift);
                    (!needs_legs || f.morph.legged()) && (!needs_arms || f.morph.arm_count() > 0)
                })
                .map(|f| f.id.to_string())
                .collect();
            let pair = if a < b { (a, b) } else { (b, a) };
            out.push(CausePairEntry {
                pair,
                used_by,
                live_reachable: live_pairs.contains(&pair),
                refs: segs_with_ref.len(),
                forms,
            });
        }
    }
    out
}

/// Collect the union of coupling/dynamic axis names observed across all definitions on `body` (Phase 6 – Part 2 · D6).
pub fn known_axes_from_defs(body: &Body) -> Vec<String> {
    let ff = FeedForward::default();
    let mut axes: BTreeSet<String> = [
        "tilt_deg",
        "lean_deg",
        "restore_frac",
        "shift_frac",
        "peak_ratio",
        "grip_effort",
        "com_shift_frac",
        "ankle_rise",
        "margin",
        "cop_shift_frac",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for spec in all_scenarios(body) {
        if spec.needs_legs && !body.legged() {
            continue;
        }
        for ph in &spec.phases {
            let rep = body.solve(&ph.posture, &ff, 1e3);
            for c in cells(body, &ph.posture, &rep, None) {
                axes.insert(c.axis);
            }
        }
    }
    for def in defs() {
        for s in &def.series {
            axes.insert(s.clone());
        }
    }
    axes.into_iter().collect()
}

/// Match a live posture against `lib`, returning `(scenario, phase, distance, MatchClass)`
/// while honoring `ground`, excluding `OfflineOnly` phases, and preferring more specific
/// cause sets on the same `source` (Phase 6 – Part 1 · D2 & Part 2 · D6).
pub fn matching_phases_classified<'a>(
    lib: &'a [ScenarioSpec],
    post: &Posture,
    body_mass_kg: f32,
) -> Vec<(&'a ScenarioSpec, &'a PhaseSpec, f32, MatchClass)> {
    let have = causes_of(post);
    let is_moving = post.moving || post.accel_mps2.abs() > 0.0;
    let mass = body_mass_kg.max(1.0);
    let dist = |p: &PhaseSpec| {
        (p.posture.accel_mps2 - post.accel_mps2).abs()
            + (p.posture.slope_deg - post.slope_deg).abs() / 10.0
            + (p.posture.step_m - post.step_m).abs() * 10.0
            + (p.posture.cargo_kg - post.cargo_kg).abs() / mass
    };
    let matches_phase = |p: &PhaseSpec| -> bool {
        if matches!(p.live_reach, LiveReach::OfflineOnly { .. }) {
            return false;
        }
        if let Some(g) = p.ground {
            if post.ground != g {
                return false;
            }
        }
        let motion_ok = match p.motion {
            Motion::Any => true,
            Motion::Still => !is_moving,
            Motion::Moving => is_moving,
        };
        if !motion_ok {
            return false;
        }
        // Live slope start triggers only on measured positive acceleration from rest,
        // avoiding false matches during steady legged walking on a slope (Phase 1 – Part 1 · D4).
        if p.id == "start" {
            return post.source == Source::Accel && post.accel_mps2 > 0.0 && have.contains(&Cause::Slope);
        }
        p.posture.source == post.source && p.causes.iter().all(|c| have.contains(c))
    };
    let in_holdout_band = (post.slope_deg.abs() >= HOLDOUT_SLOPE_DEG.0 && post.slope_deg.abs() <= HOLDOUT_SLOPE_DEG.1)
        || (post.step_m >= HOLDOUT_STEP_M.0 && post.step_m <= HOLDOUT_STEP_M.1)
        || (post.accel_mps2 < 0.0 && HOLDOUT_BRAKE.iter().any(|&(lo, hi)| post.accel_mps2.abs() >= lo && post.accel_mps2.abs() <= hi))
        || (post.cargo_kg > 0.0 && HOLDOUT_LOAD_FRAC.iter().any(|&(lo, hi)| {
            let f = post.cargo_kg / mass;
            f >= lo && f <= hi
        }));
    let classify = |d: f32| {
        if d <= 0.35 && !in_holdout_band {
            MatchClass::Dev
        } else if in_holdout_band {
            MatchClass::HoldoutLike
        } else {
            MatchClass::Between
        }
    };
    let raw: Vec<(&'a ScenarioSpec, &'a PhaseSpec, f32, MatchClass)> = lib
        .iter()
        .filter(|s| !s.holdout && s.kind != ScenarioKind::RecordOnly && s.kind != ScenarioKind::Reserved)
        .filter_map(|s| {
            s.phases
                .iter()
                .filter(|p| matches_phase(p))
                .min_by(|a, b| b.causes.len().cmp(&a.causes.len()).then_with(|| dist(a).total_cmp(&dist(b))))
                .map(|p| {
                    let d = dist(p);
                    (s, p, d, classify(d))
                })
        })
        .collect();
    // Prevent loaded Slope postures ({Slope, CarryFront} matching G1) from polluting unloaded Slope phases.
    raw.iter()
        .copied()
        .filter(|&(_, ph, _, _)| {
            if ph.posture.source != Source::Slope {
                return true;
            }
            !raw.iter().any(|&(_, other, _, _)| {
                other.posture.source == Source::Slope
                    && other.causes.len() > ph.causes.len()
                    && ph.causes.iter().all(|c| other.causes.contains(c))
            })
        })
        .collect()
}

pub fn matching_phases<'a>(lib: &'a [ScenarioSpec], post: &Posture) -> Vec<(&'a ScenarioSpec, &'a PhaseSpec)> {
    matching_phases_classified(lib, post, 60.0)
        .into_iter()
        .map(|(s, p, _, _)| (s, p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    #[test]
    fn weights_sum_to_one_and_cells_score_on_the_gdi_scale() {
        let b = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        for s in library(&b) {
            let w: f32 = s.seg_w.iter().sum();
            assert!((w - 1.0).abs() < 2e-3, "{}: {w}", s.id);
            let mut ladders = std::collections::BTreeMap::new();
            let mut total = 0.0f32;
            for p in &s.phases {
                match p.ladder {
                    Some(l) => {
                        ladders.insert(l, p.weight);
                    }
                    None => total += p.weight,
                }
            }
            let total = total + ladders.values().sum::<f32>();
            assert!((total - 1.0).abs() < 2e-3, "{}: phase weights {total}", s.id);
        }
        let mut c = Cell { source: Source::Quiet, target: Segment::Foot, axis: "x".into(), value: 1.3, reference: Some(1.0), tolerance: Some(0.1), cite: None, z: None, score: None, dynamic: false, ref_state: RefState::Human, ref_note: None };
        scored(&mut c);
        assert!((c.z.unwrap() - 3.0).abs() < 1e-4 && (c.score.unwrap() - 70.0).abs() < 1e-3);
    }

    #[test]
    fn coverage_bounds_contain_the_score() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let ff = FeedForward::default();
        for s in library(&b) {
            let res = evaluate(&b, &s, &ff);
            for p in &res.phases {
                assert!(p.coverage_w > 0.0 && p.coverage_w <= 1.0 + 1e-5, "{}/{}: {}", s.id, p.id, p.coverage_w);
                assert!(p.score_bounds[0] <= p.ungated_score + 1e-3 && p.ungated_score <= p.score_bounds[1] + 1e-3, "{}/{}: bounds {:?} vs {}", s.id, p.id, p.score_bounds, p.ungated_score);
                if (p.coverage_w - 1.0).abs() < 1e-4 {
                    assert!((p.score_bounds[0] - p.score_bounds[1]).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn every_cell_says_why_it_is_or_is_not_scored() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let ff = FeedForward::default();
        for s in library(&b) {
            let res = evaluate(&b, &s, &ff);
            for p in &res.phases {
                for c in &p.cells {
                    match c.ref_state {
                        RefState::Human => {
                            assert!(c.reference.is_some() && c.tolerance.is_some() && c.z.is_some() && c.score.is_some(), "{}/{}/{}: human cell missing score", s.id, p.id, c.axis);
                            assert!(c.cite.is_some_and(|x| !x.is_empty()), "{}/{}/{}: human cell missing cite", s.id, p.id, c.axis);
                        }
                        RefState::Mismatch => {
                            assert!(c.z.is_none() && c.score.is_none() && c.ref_note.is_some(), "{}/{}/{}: mismatch cell invalid", s.id, p.id, c.axis);
                        }
                        RefState::RobotLimit | RefState::Raw | RefState::Absent => {
                            assert!(c.z.is_none() && c.score.is_none(), "{}/{}/{}: non-human cell should not be scored", s.id, p.id, c.axis);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ankle_rise_is_not_compared_on_one_foot() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let ff = FeedForward::default();
        let lib = library(&b);
        let s3 = lib.iter().find(|s| s.id == "S3").unwrap();
        let res = evaluate(&b, s3, &ff);
        let reach = res.phases.iter().find(|p| p.id == "reach").unwrap();
        let ankle = reach.cells.iter().find(|c| c.target == Segment::Foot && c.axis == "ankle_rise").unwrap();
        assert_eq!(ankle.ref_state, RefState::Mismatch);
        assert!(ankle.z.is_none() && ankle.score.is_none());
        let shift = reach.cells.iter().find(|c| c.target == Segment::Foot && c.axis == "com_shift_frac").unwrap();
        assert_eq!(shift.ref_state, RefState::Mismatch);
        assert!(shift.z.is_none() && shift.score.is_none());
    }

    #[test]
    fn braking_uses_the_forward_limit_and_starting_the_narrower_backward_one() {
        let (b, s) = (composite_reference(Source::Accel, Segment::Foot, "los_frac", -1.0).unwrap(), composite_reference(Source::Accel, Segment::Foot, "los_frac", 1.0).unwrap());
        assert!((b.0 - 0.931).abs() < 0.005, "{}", b.0);
        assert!(s.0 > b.0, "the same acceleration uses more of the backward limit");
    }

    #[test]
    fn world_postures_match_the_phases_with_all_their_causes() {
        let b = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let lib = library(&b);
        let brake = Posture { source: Source::Accel, accel_mps2: -1.2, ..Posture::quiet(3.0) };
        let m = matching_phases(&lib, &brake);
        assert!(m.iter().any(|(s, p)| s.id == "S4" && p.ladder == Some("brake")), "{:?}", m.iter().map(|(s, p)| (s.id, p.id.clone())).collect::<Vec<_>>());
        let empty_brake = Posture { source: Source::Accel, accel_mps2: -1.2, ..Posture::quiet(0.0) };
        assert!(matching_phases(&lib, &empty_brake).iter().all(|(s, _)| s.id != "S4"), "no cargo, no S4");
    }

    #[test]
    fn holdout_never_touches_any_library_condition() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let conds = dev_conditions(defs());
        assert!(!conds.slopes_deg.is_empty() && !conds.steps_m.is_empty() && !conds.brakes_mps2.is_empty() && !conds.load_fracs.is_empty());
        for &a in &conds.brakes_mps2 {
            assert!(HOLDOUT_BRAKE.iter().all(|r| a < r.0 || a > r.1), "dev brake {a} inside holdout");
        }
        for &sl in &conds.slopes_deg {
            assert!(sl < HOLDOUT_SLOPE_DEG.0 || sl > HOLDOUT_SLOPE_DEG.1, "dev slope {sl} inside holdout");
        }
        for &st in &conds.steps_m {
            assert!(st < HOLDOUT_STEP_M.0 || st > HOLDOUT_STEP_M.1, "dev step {st} inside holdout");
        }
        for &lf in &conds.load_fracs {
            assert!(HOLDOUT_LOAD_FRAC.iter().all(|r| lf < r.0 || lf > r.1), "dev load_frac {lf} inside holdout");
        }
        let seed = crate::bench::holdout_seed("NN");
        let h = holdout(&b, seed, 8);
        assert_eq!(h.len(), 8);
        assert!(h.iter().all(|s| s.holdout && (1..=3).contains(&s.phases.len())));
        assert!(h.iter().any(|s| s.phases.len() >= 2), "holdout must include multi-phase templates");
        assert_eq!(holdout(&b, seed, 8), h, "a seed gives the same scenarios");
    }

    fn legacy_coded_library(body: &Body) -> Vec<ScenarioSpec> {
        let load = 0.10 * body.mass_kg();
        let walk = body.legged();
        let mut out = Vec::new();
        let ph = |id: &str, subsystem: &'static str, weight: f32, posture: Posture, causes: &[Cause], motion: Motion| PhaseSpec {
            id: id.into(),
            subsystem,
            weight,
            posture,
            causes: causes.to_vec(),
            ladder: None,
            imposed: false,
            motion,
            ground: None,
            live_reach: LiveReach::Expected,
        };
        out.push(ScenarioSpec {
            id: "S1",
            name_ko: "경사 위 들기",
            kind: ScenarioKind::Phase,
            seg_w: [0.218, 0.253, 0.052, 0.138, 0.103, 0.069, 0.167],
            weights_from: "Mini-BESTest 감각 6 : 예측 6 (Franchignoni 2010) · REBA 분절 최대점 (Hignett & McAtamney 2000)",
            needs_legs: false,
            holdout: false,
            phases: vec![
                ph("slope", "sensory", 0.5, Posture { source: Source::Slope, slope_deg: 10.0, ..Posture::quiet(0.0) }, &[Cause::Slope], Motion::Still),
                ph("lift", "anticipatory", 0.5, lift_posture(body, load, 10.0), &[Cause::ArmLift, Cause::Slope], Motion::Any),
            ],
        });
        out.push(ScenarioSpec {
            id: "S2",
            name_ko: "앞 짐 들고 턱 오르기",
            kind: ScenarioKind::Phase,
            seg_w: [0.056, 0.093, 0.210, 0.149, 0.111, 0.228, 0.154],
            weights_from: "BESTest APA 18 : 보행 21 항목 점수 (Horak 2009) · REBA 분절 최대점",
            needs_legs: true,
            holdout: false,
            phases: vec![
                ph("carry_walk", "gait", 0.538, Posture { source: Source::FootLift, single_support: walk, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::FootLift], Motion::Moving),
                ph("step", "anticipatory", 0.462, Posture { source: Source::StepUp, step_m: 0.17, single_support: walk, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::StepUp], Motion::Any),
            ],
        });
        let mut s3_reach = ph("reach", "limits", 0.538, Posture { source: Source::ArmRaise, arm_elev_deg: [90.0, 0.0], elbow_deg: [0.0, 10.0], single_support: true, ..Posture::quiet(0.0) }, &[Cause::FootLift, Cause::ArmRaise], Motion::Any);
        s3_reach.live_reach = LiveReach::OfflineOnly { reason: "퀘스트에 한 발 뻗기 동작 없음" };
        out.push(ScenarioSpec {
            id: "S3",
            name_ko: "한 발로 서서 앞으로 뻗기",
            kind: ScenarioKind::Phase,
            seg_w: [0.0, 0.179, 0.154, 0.179, 0.0, 0.154, 0.334],
            weights_from: "BESTest 안정 한계 21 : APA 18 · 하위체계 안 동일 가중 (Rasch)",
            needs_legs: true,
            holdout: false,
            phases: vec![
                ph("weight_shift", "anticipatory", 0.462, Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) }, &[Cause::FootLift], Motion::Any),
                s3_reach,
            ],
        });
        let mut s4 = vec![ph("carry", "gait", 0.625, Posture { source: Source::CarryFront, ..Posture::quiet(load) }, &[Cause::CarryFront], Motion::Moving)];
        for a in [0.5f32, 1.0, 1.5, 2.5] {
            let mut p = ph(&format!("brake_{a:.1}"), "reactive", 0.375, Posture { source: Source::Accel, accel_mps2: -a, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake], Motion::Moving);
            p.ladder = Some("brake");
            p.imposed = true;
            s4.push(p);
        }
        out.push(ScenarioSpec {
            id: "S4",
            name_ko: "짐 들고 제동 (강도 사다리)",
            kind: ScenarioKind::Phase,
            seg_w: [0.190, 0.233, 0.065, 0.172, 0.129, 0.086, 0.125],
            weights_from: "Mini-BESTest 반응 6 : 보행 10 · 사다리는 NIOSH CLI 의 최악 단계 바닥값 (Waters 1994)",
            needs_legs: false,
            holdout: false,
            phases: s4,
        });
        let start = if walk {
            Posture { source: Source::FootLift, single_support: true, slope_deg: 10.0, ..Posture::quiet(0.0) }
        } else {
            let a = body.planned_accel(&body.base.spec(), 10.0, 0.0, Surface::Firm, 1.0, PLANNER_MIN_RAMP_S);
            Posture { source: Source::Accel, accel_mps2: a, slope_deg: 10.0, ..Posture::quiet(0.0) }
        };
        out.push(ScenarioSpec {
            id: "S6",
            name_ko: "경사 위에서 팔 들어 치고 출발",
            kind: ScenarioKind::Phase,
            seg_w: [0.168, 0.168, 0.140, 0.114, 0.057, 0.130, 0.223],
            weights_from: "Brief-BESTest APA 6 : 감각 3 (Padgett 2012) · RULA 분절 최대점 (McAtamney & Corlett 1993)",
            needs_legs: false,
            holdout: false,
            phases: vec![
                ph("slope", "sensory", 0.333, Posture { source: Source::Slope, slope_deg: 10.0, ..Posture::quiet(0.0) }, &[Cause::Slope], Motion::Still),
                ph("strike", "anticipatory", 0.3335, Posture { slope_deg: 10.0, ..Posture::arm_raise(0.0) }, &[Cause::ArmRaise, Cause::Slope], Motion::Any),
                ph("start", "anticipatory", 0.3335, start, if walk { &[Cause::FootLift, Cause::Slope] } else { &[Cause::Start, Cause::Slope] }, Motion::Moving),
            ],
        });
        out
    }

    #[test]
    fn defs_reproduce_the_coded_library() {
        for form in crate::bench::forms() {
            let plan = BodyPlan::named(form.plan).unwrap();
            let body = Body::new(&plan, form.base, form.fingers);
            let from_data = library(&body);
            let from_code = legacy_coded_library(&body);
            assert_eq!(from_data, from_code, "mismatch on form {}", form.id());
        }
    }

    #[test]
    fn nn_bench_lines_unchanged_by_the_data_move() {
        let ff = FeedForward::default();
        for form in crate::bench::forms() {
            let plan = BodyPlan::named(form.plan).unwrap();
            let body = Body::new(&plan, form.base, form.fingers);
            let data_res: Vec<ScenarioResult> = library(&body).iter().map(|s| evaluate(&body, s, &ff)).collect();
            let code_res: Vec<ScenarioResult> = legacy_coded_library(&body).iter().map(|s| evaluate(&body, s, &ff)).collect();
            let a = serde_json::to_string(&data_res).unwrap();
            let b = serde_json::to_string(&code_res).unwrap();
            assert_eq!(a, b, "serialized ScenarioResult differed on {}", form.id());
        }
    }

    #[test]
    fn scenario_defs_are_well_formed() {
        let mut seen_ids = std::collections::BTreeSet::new();
        assert_eq!(defs_rev().len(), 8);
        for d in defs() {
            assert!(!d.id.is_empty() && d.id.len() <= 8, "bad id {}", d.id);
            assert!(d.id.chars().next().unwrap().is_ascii_uppercase(), "id must start uppercase: {}", d.id);
            assert!(seen_ids.insert(d.id.as_str()), "duplicate scenario id {}", d.id);
            let seg_sum: f32 = d.seg_w.iter().sum();
            assert!((seg_sum - 1.0).abs() < 2e-3, "{}: seg_w sum {seg_sum}", d.id);
            let mut phase_ids = std::collections::BTreeSet::new();
            let mut ladders: std::collections::BTreeMap<&str, f32> = std::collections::BTreeMap::new();
            let mut wsum = 0.0f32;
            for p in &d.phases {
                assert!(phase_ids.insert(p.id.as_str()), "{}: duplicate phase {}", d.id, p.id);
                if let Some(l) = p.ladder.as_deref() {
                    if let Some(&prev) = ladders.get(l) {
                        assert!((prev - p.weight).abs() < 1e-5, "{}: ladder {l} weights differ", d.id);
                    } else {
                        ladders.insert(l, p.weight);
                    }
                } else {
                    wsum += p.weight;
                }
            }
            if matches!(d.kind, ScenarioKind::Phase | ScenarioKind::Sequence | ScenarioKind::RecordOnly) && !d.phases.is_empty() {
                let total = wsum + ladders.values().sum::<f32>();
                assert!((total - 1.0).abs() < 2e-3, "{}: phase weights sum {total}", d.id);
            }
        }
    }

    #[test]
    fn cause_map_counts_27_pairs() {
        let core_defs: Vec<ScenarioDef> = defs().iter().filter(|d| d.core).cloned().collect();
        let core_map = cause_map(&core_defs);
        assert_eq!(core_map.len(), 27, "8 choose 2 minus impossible Start+Brake = 27");
        for (a, b) in [
            (Cause::ArmRaise, Cause::CarryFront),
            (Cause::ArmLift, Cause::CarryFront),
            (Cause::Slope, Cause::CarryFront),
        ] {
            let pair = if a < b { (a, b) } else { (b, a) };
            let entry = core_map.iter().find(|e| e.pair == pair).expect("pair present");
            assert!(entry.live_reachable && entry.used_by.is_empty(), "core gap {pair:?}: {entry:?}");
        }
        let full_map = cause_map(defs());
        assert_eq!(full_map.len(), 27);
        for (a, b, expected_id) in [
            (Cause::ArmRaise, Cause::CarryFront, "M1"),
            (Cause::ArmLift, Cause::CarryFront, "L1"),
            (Cause::Slope, Cause::CarryFront, "G1"),
        ] {
            let pair = if a < b { (a, b) } else { (b, a) };
            let entry = full_map.iter().find(|e| e.pair == pair).expect("pair present");
            assert!(entry.used_by.iter().any(|id| id == expected_id), "full map {pair:?} missing {expected_id}: {:?}", entry.used_by);
        }
    }

    #[test]
    fn cocarry_phase_members_carry_the_load() {
        let body = Body::new(&BodyPlan::uniform(Tier::Pro), BaseKind::HumanFoot, FingerCount::Five);
        let ff = FeedForward::default();
        for id in ["C1", "C2", "C3"] {
            let def = def_by_id(id).expect(id);
            let res = evaluate_crew(def, &body, &body, &ff);
            assert!(res.skipped.is_none(), "{id}: {:?}", res.skipped);
            assert_eq!(res.members.len(), 2);
            let sum_kg: f32 = res.members.iter().map(|m| m.load_kg).sum();
            assert!((sum_kg - res.load_kg).abs() <= 0.05 * res.load_kg.max(1.0), "{id}: sum={sum_kg} vs load={}", res.load_kg);
            assert!(res.team_cells.iter().all(|c| c.ref_state == RefState::Raw));
            assert_eq!(res.gates.is_empty(), res.team_ok, "{id}: gates={:?} team_ok={}", res.gates, res.team_ok);
        }
    }

    #[test]
    fn coverage_matrix_snapshot() {
        let mat = coverage_matrix();
        assert_eq!(mat.len(), FORMS.len());
        assert_eq!(mat["legs_humanoid"]["S2"], FormCoverage::Evaluated);
        assert_eq!(mat["wheel_humanoid"]["S2"], FormCoverage::NotApplicable);
        assert_eq!(mat["wheel_humanoid"]["S4"], FormCoverage::Evaluated);
        assert_eq!(mat["truck"]["S1"], FormCoverage::Substituted);
        assert_eq!(mat["truck"]["S4"], FormCoverage::Evaluated);
        assert_eq!(mat["truck"]["S6"], FormCoverage::Partial);
        assert_eq!(mat["truck"]["C1"], FormCoverage::NotApplicable);
        assert_eq!(mat["truck"]["L1"], FormCoverage::NotApplicable);
        assert_eq!(mat["truck"]["M1"], FormCoverage::NotApplicable);
        assert_eq!(mat["mast_gripper"]["C1"], FormCoverage::NotApplicable);
        assert_eq!(mat["mast_gripper"]["L1"], FormCoverage::Evaluated);
        assert_eq!(mat["wheel_humanoid"]["E2"], FormCoverage::Partial);
        assert_eq!(mat["legs_humanoid"]["E2"], FormCoverage::Evaluated);
    }

    #[test]
    fn evaluate_sequence_tracks_heat_and_wear() {
        let body = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::Wheel, FingerCount::Five);
        let ff = FeedForward::default();
        let def = def_by_id("T1").expect("T1");
        let spec = resolve(def, &body);
        let plan = SeqPlan::from_def(def);
        let res = evaluate_sequence(&body, &spec, &plan, &ff);
        assert_eq!(res.reps.len(), plan.repeat);
        assert!(res.heat_max >= 0.0 && res.wear_max >= 0.0);
        assert!(res.reps.windows(2).all(|w| w[1].t_s > w[0].t_s && w[1].wear_max >= w[0].wear_max - 1e-9));
        assert!(res.sli.is_some());
    }

    #[test]
    fn every_live_phase_is_reachable() {
        let legged = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        let wheeled = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let all_l = all_scenarios(&legged);
        let all_w = all_scenarios(&wheeled);
        let posts_l = live_world_postures(&legged);
        let posts_w = live_world_postures(&wheeled);
        let mut matched_keys: BTreeSet<(String, String)> = BTreeSet::new();
        for p in &posts_l {
            for (s, ph, _, _) in matching_phases_classified(&all_l, p, legged.mass_kg()) {
                matched_keys.insert((s.id.to_string(), ph.id.clone()));
            }
        }
        for p in &posts_w {
            for (s, ph, _, _) in matching_phases_classified(&all_w, p, wheeled.mass_kg()) {
                matched_keys.insert((s.id.to_string(), ph.id.clone()));
            }
        }
        assert!(!matched_keys.contains(&("S3".into(), "reach".into())), "OfflineOnly S3/reach must never match live");
        assert!(!matched_keys.iter().any(|(s, _)| s == "S5-A"), "RecordOnly S5-A must never match live");
        for def in defs() {
            for ph in &def.phases {
                if ph.live {
                    assert!(
                        matched_keys.contains(&(def.id.clone(), ph.id.clone())),
                        "live:true phase {}/{} never matched any world posture",
                        def.id,
                        ph.id
                    );
                } else {
                    assert!(
                        !matched_keys.contains(&(def.id.clone(), ph.id.clone())),
                        "live:false phase {}/{} matched live",
                        def.id,
                        ph.id
                    );
                }
            }
        }
    }
}
