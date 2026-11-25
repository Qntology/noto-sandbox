//! Live progression assembly, N:N loss decomposition, 7×7 evidence matrix, and unified hold/abstain log
//! (Phase 5 – Part 1 · D1–D4).
//!
//! Deliberately OUTSIDE `ENGINE_FILES` and `DECISION_FILES` so display code can safely read `nn` types,
//! SDS observation counts, and hold records without ever feeding back into control decisions.

use std::collections::{BTreeMap, VecDeque};

use noto_core::balance::Source;
use noto_core::nn::{self, GateHit, PhaseResult, RefState, SEGMENTS};
use noto_core::parts::Segment;
use crate::sds::{Scope, Track};
use serde::Serialize;
use serde_json::{json, Value};

use crate::nnlive::ActiveNow;
use crate::runtime::Runtime;

const HOLD_RING_CAP: usize = 64;
const HOLD_WINDOW_TICKS: u64 = 100;

/// The 7 active perturbation sources (all `Source` variants except `Quiet`) forming rows of the 7×7 map.
pub const MATRIX_SOURCES: [Source; 7] = [
    Source::ArmRaise,
    Source::ArmLift,
    Source::FootLift,
    Source::StepUp,
    Source::Slope,
    Source::CarryFront,
    Source::Accel,
];

/// Unified hold / abstain / excused event across all 4 runtime domains (Phase 5 – Part 1 · D3).
#[derive(Clone, Debug, Serialize)]
pub struct HoldItem {
    pub tick: u64,
    pub worker: u32,
    /// `"nn_gate" | "precision" | "decide" | "traffic"`
    pub domain: &'static str,
    pub code: String,
    /// `"held" | "abstained" | "excused"`
    pub verdict: &'static str,
    pub subject: String,
    pub measured: Option<f64>,
    pub limit: Option<f64>,
    pub band: Option<[f64; 2]>,
    pub unit: &'static str,
    pub detail_ko: String,
}

/// Bounded ring of recent hold & abstain events (display-only; banned in `DECISION_FILES`).
#[derive(Clone, Debug, Default)]
pub struct HoldLog {
    items: VecDeque<HoldItem>,
}

impl HoldLog {
    pub fn push(&mut self, item: HoldItem) {
        self.items.push_back(item);
        while self.items.len() > HOLD_RING_CAP {
            self.items.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn all(&self) -> Vec<HoldItem> {
        self.items.iter().cloned().collect()
    }

    pub fn recent_window(&self, now_tick: u64, window_ticks: u64) -> Vec<HoldItem> {
        let cutoff = now_tick.saturating_sub(window_ticks);
        self.items.iter().filter(|h| h.tick >= cutoff).cloned().collect()
    }

    /// Count active holds (`verdict == "held"`) for `worker` in `[now_tick - window, now_tick]`.
    /// Abstains and beyond-human excuses are explicitly excluded (`abstain_is_not_a_hold`).
    pub fn hold_count(&self, worker: u32, now_tick: u64, window_ticks: u64) -> usize {
        let cutoff = now_tick.saturating_sub(window_ticks);
        self.items
            .iter()
            .filter(|h| h.worker == worker && h.tick >= cutoff && h.verdict == "held")
            .count()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CellBreakdown {
    pub cell_key: String,
    pub source: &'static str,
    pub target: &'static str,
    pub axis: String,
    pub value: f32,
    pub reference: Option<f32>,
    pub tolerance: Option<f32>,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub loss: f32,
    pub dynamic: bool,
    pub cite: Option<&'static str>,
    pub ref_state: RefState,
}

#[derive(Clone, Debug, Serialize)]
pub struct PhaseBreakdown {
    pub ungated: f32,
    pub unscored_w: f32,
    pub scored_w: f32,
    pub cells: Vec<CellBreakdown>,
    pub top_loss: Option<(String, f32)>,
}

fn seg_idx(s: Segment) -> usize {
    SEGMENTS.iter().position(|x| *x == s).unwrap_or(0)
}

/// Exact score-loss decomposition for one [`PhaseResult`] (Phase 5 – Part 1 · D2).
///
/// For every scored cell `c` in segment `i` with `n_i` scored cells and total scored weight `W`:
/// `loss(c) = seg_w[i] * (100 - s_c) / (n_i * W)`, so `Σ loss(c) == 100 - ungated_score` exactly.
pub fn breakdown(pr: &PhaseResult, seg_w: &[f32; 7]) -> PhaseBreakdown {
    let mut counts = [0u32; 7];
    for c in &pr.cells {
        if c.score.is_some() && c.z.is_some() {
            counts[seg_idx(c.target)] += 1;
        }
    }
    let mut scored_w = 0.0f32;
    let mut unscored_w = 0.0f32;
    let mut num = 0.0f32;
    for i in 0..7 {
        if let Some(s) = pr.seg_scores[i] {
            scored_w += seg_w[i];
            num += seg_w[i] * s;
        } else {
            unscored_w += seg_w[i];
        }
    }
    let ungated = if scored_w > 1e-6 { num / scored_w } else { pr.ungated_score };

    let mut cells: Vec<CellBreakdown> = pr
        .cells
        .iter()
        .map(|c| {
            let i = seg_idx(c.target);
            let loss = match (c.score, c.z) {
                (Some(s), Some(_)) if scored_w > 1e-6 && counts[i] > 0 => {
                    seg_w[i] * (100.0 - s).max(0.0) / (counts[i] as f32 * scored_w)
                }
                _ => 0.0,
            };
            CellBreakdown {
                cell_key: format!("{}>{}:{}", c.source.id(), c.target.id(), c.axis),
                source: c.source.id(),
                target: c.target.id(),
                axis: c.axis.clone(),
                value: c.value,
                reference: c.reference,
                tolerance: c.tolerance,
                z: c.z,
                score: c.score,
                loss,
                dynamic: c.dynamic,
                cite: c.cite,
                ref_state: c.ref_state,
            }
        })
        .collect();
    cells.sort_by(|a, b| b.loss.total_cmp(&a.loss).then_with(|| a.cell_key.cmp(&b.cell_key)));
    let top_loss = cells
        .first()
        .filter(|c| c.loss > 1e-4)
        .map(|c| (c.cell_key.clone(), (c.loss * 10.0).round() / 10.0));
    PhaseBreakdown { ungated, unscored_w, scored_w, cells, top_loss }
}

/// One cell of the 7×7 (Source × Segment) N:N evidence map (Phase 5 – Part 1 · D2).
#[derive(Clone, Debug, Serialize)]
pub struct MatrixCell {
    pub source: &'static str,
    pub target: &'static str,
    pub worst_z: Option<f32>,
    pub item: Option<u8>,
    pub loss_sum: f32,
    pub count: u32,
    pub has_ref: bool,
    pub last_tick: Option<u64>,
    pub in_now: bool,
}

/// Aggregate all latest `PhaseResult` cells in the active instance into a 7×7 matrix.
pub fn matrix_7x7(items: &[(&PhaseResult, &[f32; 7], u64, bool)]) -> Vec<Vec<MatrixCell>> {
    let mut grid: Vec<Vec<MatrixCell>> = MATRIX_SOURCES
        .iter()
        .map(|src| {
            SEGMENTS
                .iter()
                .map(|seg| MatrixCell {
                    source: src.id(),
                    target: seg.id(),
                    worst_z: None,
                    item: None,
                    loss_sum: 0.0,
                    count: 0,
                    has_ref: false,
                    last_tick: None,
                    in_now: false,
                })
                .collect()
        })
        .collect();

    for &(pr, seg_w, tick, in_now) in items {
        let bd = breakdown(pr, seg_w);
        for c in bd.cells {
            let Some(r_idx) = MATRIX_SOURCES.iter().position(|s| s.id() == c.source) else { continue };
            let Some(c_idx) = SEGMENTS.iter().position(|s| s.id() == c.target) else { continue };
            let cell = &mut grid[r_idx][c_idx];
            cell.count += 1;
            cell.loss_sum += c.loss;
            cell.in_now |= in_now;
            cell.last_tick = Some(cell.last_tick.map_or(tick, |prev| prev.max(tick)));
            if let Some(z) = c.z {
                cell.has_ref = true;
                let az = z.abs();
                cell.worst_z = Some(cell.worst_z.map_or(az, |w| w.max(az)));
            }
        }
    }
    for row in &mut grid {
        for cell in row {
            cell.loss_sum = (cell.loss_sum * 10.0).round() / 10.0;
            cell.item = cell.worst_z.map(|z| if z <= 1.0 { 2 } else if z <= 2.0 { 1 } else { 0 });
        }
    }
    grid
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkerLine {
    pub id: u32,
    pub first: bool,
    pub form: String,
    pub quest_ko: String,
    pub stage_ko: String,
    pub nn: &'static str,
    pub holds_n: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuestLane {
    pub kind: String,
    pub index: usize,
    pub count: usize,
    pub progress: f32,
    pub label_ko: String,
    pub title: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProcLane {
    pub mechanism: String,
    pub stage: Option<String>,
    pub stage_ko: Option<String>,
    pub index: Option<usize>,
    pub count: Option<usize>,
    pub segment: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NnLane {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    pub form_id: String,
    pub rev: u64,
    pub now: Vec<ActiveNow>,
    pub trail: Vec<ActiveNow>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProgressView {
    pub tick: u64,
    pub epoch: u32,
    pub focus: u32,
    pub workers: Vec<WorkerLine>,
    pub quest: Option<QuestLane>,
    pub procedure: Option<ProcLane>,
    pub nn: NnLane,
    pub holds: Vec<HoldItem>,
    pub abstains_n: usize,
    pub holds_n: usize,
}

fn segment_from_balance(rt: &Runtime) -> Option<String> {
    let key = rt.world().balance.as_ref().map(|(k, _)| k.as_str())?;
    let seg = key.rsplit('|').next().unwrap_or("idle");
    if seg == "idle" {
        None
    } else {
        Some(seg.to_string())
    }
}

pub fn build_progress(rt: &Runtime, focus: u32) -> ProgressView {
    let tick = rt.world().tick;
    let w = rt.world();
    let active_q = rt
        .quest_book()
        .current_id()
        .and_then(|id| rt.quest_book().get(&id));
    let primary_quest_ko = active_q
        .and_then(|q| q.steps.iter().find(|s| s.state == noto_quest::State::Active).map(|s| s.label_ko.clone()))
        .unwrap_or_else(|| "대기".into());
    let proc_json = rt.procedure_json();
    let primary_stage_ko = proc_json
        .as_ref()
        .and_then(|p| p.get("stage_ko").and_then(Value::as_str).map(str::to_string))
        .or_else(|| segment_from_balance(rt))
        .unwrap_or_else(|| "대기".into());

    let mut workers = vec![WorkerLine {
        id: 1,
        first: true,
        form: world_sim::crew::Form::of_base(rt.cfg.noto.base).id().to_string(),
        quest_ko: primary_quest_ko,
        stage_ko: primary_stage_ko,
        nn: "computed",
        holds_n: rt.holds.hold_count(1, tick, HOLD_WINDOW_TICKS),
    }];
    for cw in &w.crew {
        let stage = rt
            .crew
            .lanes
            .get(&cw.id)
            .map(|l| l.stage.label_ko())
            .unwrap_or_else(|| cw.status_ko());
        workers.push(WorkerLine {
            id: cw.id,
            first: false,
            form: cw.form.id().to_string(),
            quest_ko: cw.status_ko(),
            stage_ko: stage,
            nn: "not_computed",
            holds_n: rt.holds.hold_count(cw.id, tick, HOLD_WINDOW_TICKS),
        });
    }

    let focus_id = if workers.iter().any(|x| x.id == focus) { focus } else { 1 };
    let quest = if focus_id == 1 {
        active_q.map(|q| {
            let count = q.steps.len().max(1);
            let idx = q
                .steps
                .iter()
                .position(|s| s.state == noto_quest::State::Active)
                .unwrap_or_else(|| count.saturating_sub(1));
            let step = q.steps.get(idx);
            let label = step.map(|s| s.label_ko.clone()).unwrap_or_else(|| q.title.clone());
            let kind = step.map(|s| format!("{:?}", s.kind).to_lowercase()).unwrap_or_else(|| "quest".into());
            let done_n = q.steps.iter().filter(|s| s.state == noto_quest::State::Done).count() as f32;
            let active_p = step.map_or(0.0, |s| s.progress.clamp(0.0, 1.0));
            let progress = ((done_n + active_p) / count as f32).clamp(0.0, 1.0);
            QuestLane {
                kind,
                index: idx + 1,
                count,
                progress,
                label_ko: label,
                title: q.title.clone(),
            }
        })
    } else {
        w.crew_worker(focus_id).map(|cw| {
            let st = cw.status_ko();
            QuestLane {
                kind: "crew".into(),
                index: 1,
                count: 1,
                progress: if matches!(cw.job, world_sim::crew::Job::Idle) { 0.0 } else { 0.5 },
                label_ko: st.clone(),
                title: format!("작업자 {} · {st}", cw.id),
            }
        })
    };

    let procedure = if focus_id == 1 {
        if let Some(pj) = proc_json {
            let stages = pj.get("stages").and_then(Value::as_array);
            let count = stages.map(|a| a.len());
            let idx = stages.and_then(|a| a.iter().position(|s| s.get("state").and_then(Value::as_str) == Some("active"))).map(|i| i + 1);
            Some(ProcLane {
                mechanism: pj.get("mechanism").and_then(Value::as_str).unwrap_or("hand").to_string(),
                stage: pj.get("stage").and_then(Value::as_str).map(str::to_string),
                stage_ko: pj.get("stage_ko").and_then(Value::as_str).map(str::to_string),
                index: idx,
                count,
                segment: None,
            })
        } else {
            segment_from_balance(rt).map(|seg| ProcLane {
                mechanism: if noto_core::act::fork::applies(&w.body) { "fork".into() } else { "hand".into() },
                stage: None,
                stage_ko: None,
                index: None,
                count: None,
                segment: Some(seg),
            })
        }
    } else {
        None
    };

    let nn = if focus_id == 1 {
        NnLane {
            status: "computed",
            reason: None,
            form_id: rt.nn_live.form_id().to_string(),
            rev: rt.nn_live.rev(),
            now: rt.nn_live.active_now().to_vec(),
            trail: rt.nn_live.trail(),
        }
    } else {
        let form = w
            .crew_worker(focus_id)
            .map(|c| c.form.id().to_string())
            .unwrap_or_else(|| "crew".into());
        NnLane {
            status: "not_computed",
            reason: Some("서브 작업자는 동역학·N:N 라이브 채점을 계산하지 않습니다"),
            form_id: form,
            rev: 0,
            now: Vec::new(),
            trail: Vec::new(),
        }
    };

    let holds = rt.holds.recent_window(tick, HOLD_WINDOW_TICKS);
    let holds_n = holds.iter().filter(|h| h.verdict == "held").count();
    let abstains_n = holds.iter().filter(|h| h.verdict == "abstained").count();

    ProgressView {
        tick,
        epoch: rt.nn_live.epoch(),
        focus: focus_id,
        workers,
        quest,
        procedure,
        nn,
        holds,
        abstains_n,
        holds_n,
    }
}

/// Build the full `nn_status` response combining backward-compatible live/store fields with Phase 5-P1
/// `catalog`, 5-state `scenario_states`, `detail` breakdown, 7×7 `matrix`, and `holds`.
pub fn build_nn_status_full(
    rt: &Runtime,
    worker: u32,
    sel_scenario: Option<&str>,
    sel_phase: Option<&str>,
    since: u64,
    window: Option<&str>,
) -> Value {
    let mut out = rt.nn_live.status_since(since);
    let lib = nn::library(&rt.world().body);
    let legged = rt.world().body.legged();
    let part = rt.nn_live.active_partition();

    let catalog: Vec<Value> = lib
        .iter()
        .map(|s| {
            let skipped_reason = if s.needs_legs && !legged {
                Some("다리가 없는 베이스에는 이 동작(17 cm 턱 · 한 발 서기)이 없음")
            } else {
                None
            };
            json!({
                "id": s.id,
                "name_ko": s.name_ko,
                "holdout": s.holdout,
                "needs_legs": s.needs_legs,
                "skipped_reason": skipped_reason,
                "seg_w": s.seg_w,
                "phases": s.phases.iter().map(|p| json!({
                    "id": p.id,
                    "subsystem": p.subsystem,
                    "weight": p.weight,
                    "ladder": p.ladder,
                    "imposed": p.imposed,
                    "source": p.posture.source.id(),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    // 5 phase states: active | seen | unseen | skipped_form | no_record.
    let mut scenario_states = Vec::new();
    for s in &lib {
        if s.holdout {
            continue;
        }
        let skipped = s.needs_legs && !legged;
        let mut ph_rows = Vec::new();
        let mut any_active = false;
        let mut any_seen = false;
        let mut any_recorded = false;
        for p in &s.phases {
            let is_active = part.now.iter().any(|n| n.scenario == s.id && n.phase == p.id);
            let live_stat = part.phases.get(&(s.id.to_string(), p.id.clone()));
            let ser = part.series.get(&(s.id.to_string(), p.id.clone()));
            let sds_n = rt
                .sds()
                .exact_baseline(&Scope::new(Track::Balance, &format!("nn:{}", s.id), &p.id), "score")
                .map_or(0, |(_, _, n)| n);
            let state = if skipped {
                "skipped_form"
            } else if is_active {
                any_active = true;
                any_seen = true;
                any_recorded = true;
                "active"
            } else if live_stat.is_some() {
                any_seen = true;
                any_recorded = true;
                "seen"
            } else if sds_n > 0 {
                any_recorded = true;
                "unseen"
            } else {
                "no_record"
            };
            ph_rows.push(json!({
                "phase": p.id,
                "state": state,
                "ladder": p.ladder,
                "live": p.live_reach.label(),
                "score": live_stat.map(|st| (st.score_mean as f64 * 10.0).round() / 10.0),
                "score_tw": ser.and_then(|x| x.score_tw()).map(|v| (v as f64 * 10.0).round() / 10.0),
                "score_min": ser.filter(|x| x.score_min.is_finite()).map(|x| (x.score_min as f64 * 10.0).round() / 10.0),
                "score_p10": ser.and_then(|x| x.score_p10()).map(|v| (v as f64 * 10.0).round() / 10.0),
                "dwell_s": ser.map_or(0.0, |x| (x.dwell_s as f64 * 10.0).round() / 10.0),
                "low_s": ser.map_or(0.0, |x| (x.low_s as f64 * 10.0).round() / 10.0),
                "episodes": ser.map_or(0, |x| x.episodes),
                "recovery_s": ser.map(|x| x.recovery_s.clone()).unwrap_or_default(),
                "unrecovered": ser.is_some_and(|x| x.unrecovered),
                "match": ser.map(|x| json!({"dev": x.match_counts[0], "between": x.match_counts[1], "holdout_like": x.match_counts[2]})),
                "events": live_stat.map_or(0, |st| st.count),
                "sds_n": sds_n,
            }));
        }
        let scn_state = if skipped {
            "skipped_form"
        } else if any_active {
            "active"
        } else if any_seen {
            "seen"
        } else if any_recorded {
            "unseen"
        } else {
            "no_record"
        };
        scenario_states.push(json!({
            "scenario": s.id,
            "name_ko": s.name_ko,
            "state": scn_state,
            "phases": ph_rows,
        }));
    }

    // Build 7x7 matrix from latest PhaseResults in active partition.
    let seg_w_map: BTreeMap<&str, [f32; 7]> = lib.iter().map(|s| (s.id, s.seg_w)).collect();
    let mut matrix_inputs: Vec<(&PhaseResult, &[f32; 7], u64, bool)> = Vec::new();
    for ((scn, ph), st) in &part.phases {
        if let (Some(pr), Some(sw)) = (st.last.as_ref(), seg_w_map.get(scn.as_str())) {
            let t = part.last_ticks.get(&(scn.clone(), ph.clone())).copied().unwrap_or(0);
            let in_now = part.now.iter().any(|n| n.scenario == *scn && n.phase == *ph);
            matrix_inputs.push((pr, sw, t, in_now));
        }
    }
    let matrix = matrix_7x7(&matrix_inputs);

    // Build selected phase detail breakdown.
    let pick_key: Option<(String, String)> = match (sel_scenario, sel_phase) {
        (Some(sc), Some(ph)) => Some((sc.to_string(), ph.to_string())),
        (Some(sc), None) => part
            .now
            .iter()
            .find(|n| n.scenario == sc)
            .map(|n| (n.scenario.clone(), n.phase.clone()))
            .or_else(|| part.phases.keys().find(|(s, _)| s == sc).cloned()),
        _ => part
            .now
            .first()
            .map(|n| (n.scenario.clone(), n.phase.clone()))
            .or_else(|| part.phases.keys().next().cloned()),
    };
    let detail = if worker == 1 {
        pick_key.and_then(|(sc, ph)| {
            let st = part.phases.get(&(sc.clone(), ph.clone()))?;
            let pr = st.last.as_ref()?;
            let spec = lib.iter().find(|s| s.id == sc)?;
            let bd = breakdown(pr, &spec.seg_w);
            let hist = rt.nn_live.phase_history(&sc, &ph);
            Some(json!({
                "scenario": sc,
                "name_ko": spec.name_ko,
                "phase": ph,
                "subsystem": pr.subsystem,
                "weight": pr.weight,
                "ladder": pr.ladder,
                "score": (pr.score as f64 * 10.0).round() / 10.0,
                "ungated": (bd.ungated as f64 * 10.0).round() / 10.0,
                "unscored_w": (bd.unscored_w as f64 * 1000.0).round() / 1000.0,
                "item": pr.item,
                "seg_w": spec.seg_w,
                "seg_scores": pr.seg_scores,
                "cells": bd.cells,
                "top_loss": bd.top_loss,
                "gates": pr.gates,
                "gate_hits": pr.gate_hits,
                "beyond_human": pr.beyond_human,
                "history": hist,
            }))
        })
    } else {
        None
    };

    // Unseen expected live phases (Phase 6 – Part 1 · D2).
    let mut unseen_expected = Vec::new();
    for s in &lib {
        if s.holdout || (s.needs_legs && !legged) {
            continue;
        }
        for p in &s.phases {
            if matches!(p.live_reach, nn::LiveReach::OfflineOnly { .. }) {
                continue;
            }
            if !part.phases.contains_key(&(s.id.to_string(), p.id.clone())) {
                unseen_expected.push(json!({
                    "scenario": s.id,
                    "phase": p.id,
                    "live": p.live_reach.label(),
                }));
            }
        }
    }

    // I^2 cross-phase heterogeneity per scenario (Phase 2 – Part 2 · D5).
    let mut i2_mixed = serde_json::Map::new();
    for s in &lib {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for p in &s.phases {
            if let Some((mean, sd, n)) = rt.sds().exact_baseline(&Scope::new(Track::Balance, &format!("nn:{}", s.id), &p.id), "score") {
                if n >= 2 {
                    let var = (sd * sd / n as f64).max(1e-3);
                    pts.push((mean, 1.0 / var));
                }
            }
        }
        if pts.len() >= 2 {
            let w_sum: f64 = pts.iter().map(|(_, w)| *w).sum();
            let mu = pts.iter().map(|(m, w)| m * w).sum::<f64>() / w_sum.max(1e-9);
            let q: f64 = pts.iter().map(|(m, w)| w * (m - mu).powi(2)).sum();
            let df = (pts.len() - 1) as f64;
            let i2 = if q > 1e-9 { ((q - df) / q).clamp(0.0, 1.0) } else { 0.0 };
            i2_mixed.insert(
                s.id.to_string(),
                json!({ "i2": (i2 * 1000.0).round() / 1000.0, "mixed": i2 > 0.5, "phases_k": pts.len() }),
            );
        }
    }

    // Scenario definition metadata + 7-form coverage matrix (Phase 6 – Part 2 · D6).
    let defs_meta: Vec<Value> = nn::defs()
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                "name_ko": d.name_ko,
                "kind": d.kind,
                "core": d.core,
                "requires": d.requires,
                "live": d.live,
                "series": d.series,
                "phases": d.phases.iter().map(|p| json!({
                    "id": p.id,
                    "stage": p.stage,
                    "live": p.live,
                    "requires": p.requires,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let heat_max = rt.world().heat.values().copied().fold(0.0f32, f32::max);

    if let Some(obj) = out.as_object_mut() {
        obj.insert("ref_rev".into(), json!(nn::REF_REV));
        obj.insert("defs_rev".into(), json!(nn::defs_rev()));
        obj.insert("engine_rev".into(), json!(noto_core::bench::ENGINE_REV));
        obj.insert("worker".into(), json!(worker));
        obj.insert("status".into(), json!(if worker == 1 { "computed" } else { "not_computed" }));
        obj.insert("rev".into(), json!(rt.nn_live.rev()));
        obj.insert("epoch".into(), json!(rt.nn_live.epoch()));
        obj.insert("form_id".into(), json!(rt.nn_live.form_id()));
        obj.insert("catalog".into(), json!(catalog));
        obj.insert("defs".into(), json!(defs_meta));
        obj.insert("coverage".into(), json!(nn::coverage_matrix()));
        obj.insert("scenario_states".into(), json!(scenario_states));
        obj.insert("unseen_expected".into(), json!(unseen_expected));
        obj.insert("shift".into(), part.shift.to_json(heat_max));
        obj.insert("i2_mixed".into(), Value::Object(i2_mixed));
        obj.insert("detail".into(), detail.unwrap_or(Value::Null));
        obj.insert("matrix".into(), json!(matrix));
        obj.insert("holds".into(), json!(rt.holds.all()));
        if let Some(win) = window {
            let view = crate::nnlive::NnView::from_store(rt.sds(), win, &rt.nn_live);
            obj.insert("window".into(), json!(win));
            obj.insert("scenarios".into(), view["scenarios"].clone());
            obj.insert("phases".into(), view["phases"].clone());
        } else {
            obj.insert("window".into(), json!("instance"));
            obj.insert("scenarios".into(), json!(rt.nn_live.summary()));
            obj.insert("phases".into(), json!(rt.nn_live.phases().collect::<Vec<_>>()));
        }
    }
    out
}

/// Convert `GateHit`s from a scored phase into `HoldItem`s for `HoldLog`.
pub fn hold_items_from_nn(tick: u64, worker: u32, scenario: &str, phase: &str, hits: &[GateHit]) -> Vec<HoldItem> {
    hits.iter()
        .map(|h| {
            let (unit, detail_ko) = match h.code {
                "static_infeasible" => ("m", format!("정적 불가 (여유 {:.2} m < 0)", h.measured)),
                "dyn_margin" => ("m", format!("동적 여유 {:.2} m < 0", h.measured)),
                "joint_peak" => ("%", format!("관절 피크 {:.0}% > 100%", h.measured * 100.0)),
                "not_reached" => ("", "목표 자세 미도달".to_string()),
                "head_tilt" => ("°", format!("머리 기울기 {:.0}° > {:.0}°", h.measured, h.limit)),
                other => ("", format!("게이트 {other}")),
            };
            HoldItem {
                tick,
                worker,
                domain: "nn_gate",
                code: h.code.to_string(),
                verdict: if h.excused { "excused" } else { "held" },
                subject: format!("{scenario}/{phase}"),
                measured: Some(h.measured as f64),
                limit: Some(h.limit as f64),
                band: None,
                unit,
                detail_ko,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::balance::{Body, FeedForward};
    use noto_core::{BaseKind, BodyPlan, FingerCount};

    #[test]
    fn breakdown_sums_to_the_score_gap() {
        let b = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        let lib = nn::library(&b);
        let ff = FeedForward::default();
        let mut checked = 0usize;
        for spec in &lib {
            let res = nn::evaluate(&b, spec, &ff);
            if res.skipped.is_some() {
                continue;
            }
            for ph in &res.phases {
                let bd = breakdown(ph, &spec.seg_w);
                let loss_sum: f32 = bd.cells.iter().map(|c| c.loss).sum();
                let expected_gap = 100.0 - bd.ungated;
                assert!(
                    (loss_sum - expected_gap).abs() <= 0.05,
                    "{}/{}: loss_sum={loss_sum:.4} vs gap={expected_gap:.4}",
                    spec.id,
                    ph.id
                );
                if ph.gates.is_empty() {
                    assert!((ph.score - bd.ungated).abs() <= 0.05);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn a_gated_phase_shows_its_ungated_score() {
        let mut pr = PhaseResult {
            id: "brake_2.5".into(),
            subsystem: "reactive",
            weight: 0.375,
            ladder: Some("brake"),
            cells: vec![],
            seg_scores: [Some(80.0), None, Some(60.0), None, None, None, Some(90.0)],
            score: 0.0,
            ungated_score: 0.0,
            coverage_w: 0.7,
            score_bounds: [50.0, 90.0],
            partial: false,
            item: 0,
            gates: vec!["동적 여유 -0.04 < 0".into()],
            gate_codes: vec!["margin_neg"],
            gate_hits: vec![GateHit { code: "dyn_margin", measured: -0.04, limit: 0.0, excused: false }],
            beyond_human: vec![],
            static_margin: 0.05,
            min_margin: -0.04,
            peak_ratio: 0.6,
            worst_joint: None,
            sustained_effort: 0.4,
        };
        let seg_w = [0.20, 0.25, 0.15, 0.0, 0.0, 0.0, 0.40];
        let expected = (0.20 * 80.0 + 0.15 * 60.0 + 0.40 * 90.0) / (0.20 + 0.15 + 0.40);
        pr.ungated_score = expected;
        let bd = breakdown(&pr, &seg_w);
        assert_eq!(pr.score, 0.0);
        assert!((bd.ungated - expected).abs() < 1e-4, "{} vs {expected}", bd.ungated);
        assert!((bd.unscored_w - 0.25).abs() < 1e-4);
    }

    #[test]
    fn abstain_is_not_a_hold() {
        let mut log = HoldLog::default();
        log.push(HoldItem {
            tick: 10,
            worker: 1,
            domain: "precision",
            code: "temporal".into(),
            verdict: "abstained",
            subject: "iron_ore".into(),
            measured: Some(4.0),
            limit: Some(6.0),
            band: None,
            unit: "점",
            detail_ko: "이력 4점 (판정에 6점 필요)".into(),
        });
        log.push(HoldItem {
            tick: 12,
            worker: 1,
            domain: "precision",
            code: "temporal".into(),
            verdict: "held",
            subject: "iron_ore".into(),
            measured: Some(0.35),
            limit: Some(0.60),
            band: Some([0.60, 0.90]),
            unit: "",
            detail_ko: "점수 급락 보류".into(),
        });
        assert_eq!(log.hold_count(1, 20, 100), 1, "abstained must not be counted as a hold");
        assert_eq!(log.recent_window(20, 100).len(), 2);
    }
}
