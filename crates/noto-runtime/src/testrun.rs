use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{anyhow, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use noto_core::balance::{reference, Source};
use noto_core::parts::Segment;
use noto_core::{BaseKind, BodyPlan};
use noto_quest::{Quest, QuestBook};
use crate::sds::{Level, Scope, Track};
use crate::vision_compat::{triangulate, Bearing, Rgb};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use world_sim::World;

use crate::precision::PrecisionGates;
use crate::record::{capture_keyframe, capture_video, Keyframe, KeyframeCtx, RecordCfg, Recording};
use crate::runtime::{Runtime, ScanReport, Submission};

fn default_perceive_budget_ms() -> u64 {
    160
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestCfg {
    pub record: RecordCfg,
    pub max_pass_ticks: u64,
    pub max_ml_decisions: u32,
    pub perceive_per_step: usize,
    #[serde(default = "default_perceive_budget_ms")]
    pub perceive_budget_ms: u64,
}

impl Default for TestCfg {
    fn default() -> Self {
        TestCfg {
            record: RecordCfg::default(),
            max_pass_ticks: 30_000,
            max_ml_decisions: 160,
            perceive_per_step: 6,
            perceive_budget_ms: default_perceive_budget_ms(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TestPhase {
    Game,
    Perceive,
    Decide,
    Done,
}

/// Reproducibility specification captured at test start (Phase 2 – Part 3 · D6).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayManifest {
    pub seed: u64,
    pub preset: String,
    pub body: BodyPlan,
    pub base: BaseKind,
    pub command: String,
    pub generation: String,
    pub sds_mode: crate::feedback::SdsMode,
    pub snapshot_digest: String,
    pub ledger_digest: BTreeMap<String, String>,
    pub feedback_at_start: BTreeMap<String, Option<f64>>,
    pub recognizer_mode: String,
    pub depth_mode: String,
    pub ttm_on: bool,
    pub ttm_fp8: Option<String>,
    #[serde(default)]
    pub traj_digests: Vec<(u64, String)>,
}

pub struct TestSession {
    pub id: String,
    pub command: String,
    pub seed: u64,
    pub body: BodyPlan,
    pub base: BaseKind,
    pub phase: TestPhase,
    pub test_rev: u64,
    pub game: Option<Recording>,
    pub decide: Option<Recording>,
    pub perceive_next: usize,
    pub game_quest: Option<Quest>,
    pub ml_quest: Option<Quest>,
    pub notes: Vec<String>,
    pub error: Option<String>,
    started_tick: u64,
    rays: HashMap<u32, Vec<Bearing>>,
    pub live: bool,
    pub ml_needed: bool,
    start_world: Option<World>,
    stash: Option<LiveStash>,
    pub manifest: Option<ReplayManifest>,
}

struct LiveStash {
    world: World,
    quests: QuestBook,
    gates: PrecisionGates,
    last_scan: Option<ScanReport>,
}

fn b64(v: &[u8]) -> String {
    STANDARD.encode(v)
}

fn r3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}

impl Runtime {
    pub fn test_session(&self) -> Option<&TestSession> {
        self.test.as_ref()
    }

    pub fn test_start(&mut self, command: &str, seed: u64, body: BodyPlan, base: Option<BaseKind>) -> Result<()> {
        self.cancel();
        let mut hand = self.cfg.noto;
        if let Some(b) = base {
            hand.base = b;
        }
        self.cfg.body = body;
        self.reset_with(Some(seed), None, Some(hand))?;
        self.test_seq += 1;
        let id = format!("{}-t{}", self.session_id, self.test_seq);
        self.begin_pass("game", &id);
        let s = self.submit(command);
        if !s.ok {
            self.recording = None;
            self.pass = "game".into();
            return Err(anyhow!("{}", s.error.unwrap_or_else(|| "명령을 이해하지 못했습니다".into())));
        }
        let snapshot_digest = if let Some(dir) = self.data_dir() {
            let snap_dir = dir.join("sds").join("snap");
            self.sds.save_snapshot(&snap_dir).unwrap_or_else(|_| self.sds.combined_digest())
        } else {
            self.sds.combined_digest()
        };
        let manifest = ReplayManifest {
            seed,
            preset: body.id(),
            body,
            base: hand.base,
            command: command.into(),
            generation: self.sds.generation_id().map(String::from).unwrap_or_else(crate::generation_id),
            sds_mode: self.feedback.mode().clone(),
            snapshot_digest,
            ledger_digest: self.sds.ledger_digests(),
            feedback_at_start: self.feedback.snapshot_values(),
            recognizer_mode: self.recognizer_mode().to_string(),
            depth_mode: self.ml.depth_mode(),
            ttm_on: self.has_ttm(),
            ttm_fp8: self.gates.ttm().map(|t| t.fp8().to_string()),
            traj_digests: Vec::new(),
        };
        self.clear_traj_digests();
        self.log("replay", Level::Info, format!("재현 명세 기록: {id} · 지문 {}", &manifest.snapshot_digest[..8.min(manifest.snapshot_digest.len())]), Some(json!({"replay": manifest})));
        self.test = Some(TestSession {
            id: id.clone(),
            command: command.into(),
            seed,
            body,
            base: hand.base,
            phase: TestPhase::Game,
            test_rev: self.recording.as_ref().map_or(1, |r| r.test_rev.max(1)),
            game: None,
            decide: None,
            perceive_next: 0,
            game_quest: None,
            ml_quest: None,
            notes: vec![],
            error: None,
            started_tick: self.world.tick,
            rays: HashMap::new(),
            live: false,
            ml_needed: true,
            start_world: None,
            stash: None,
            manifest: Some(manifest),
        });
        self.log("quest", Level::Info, format!("통합 테스트 {id} 시작: \"{command}\" · 시드 {seed} · 프리셋 {} · 베이스 {:?}", body.id(), hand.base), None);
        Ok(())
    }

    pub fn live_start(&mut self, command: &str) -> Result<Submission> {
        let start = self.world.clone();
        self.test = None;
        self.test_seq += 1;
        let id = format!("{}-t{}", self.session_id, self.test_seq);
        self.begin_pass("game", &id);
        let s = self.submit(command);
        let plan = s.quest_id.as_deref().and_then(|q| self.quests.get(q)).map(|q| q.plan.clone());
        let Some(plan) = plan.filter(|_| s.ok && !s.stopped) else {
            self.recording = None;
            self.pass = "game".into();
            return Ok(s);
        };
        let ml_needed = matches!(plan, noto_search::Plan::Fetch { .. } | noto_search::Plan::Collect { .. });
        self.test = Some(TestSession {
            id: id.clone(),
            command: command.into(),
            seed: self.cfg.seed,
            body: self.cfg.body,
            base: self.cfg.noto.base,
            phase: TestPhase::Game,
            test_rev: self.recording.as_ref().map_or(1, |r| r.test_rev.max(1)),
            game: None,
            decide: None,
            perceive_next: 0,
            game_quest: None,
            ml_quest: None,
            notes: vec![],
            error: None,
            started_tick: self.world.tick,
            rays: HashMap::new(),
            live: true,
            ml_needed,
            start_world: Some(start),
            stash: None,
            manifest: None,
        });
        self.log(
            "quest",
            Level::Info,
            format!("기록 시작 {id}: \"{command}\" — 1 게임(지금 월드) → 2 ML 인식 → 3 ML 결정{}", if ml_needed { "(같은 시작 상태 복제 월드)" } else { " 생략: 찾을 대상이 없는 명령" }),
            None,
        );
        Ok(s)
    }

    pub(crate) fn live_restore(&mut self) {
        if let Some(st) = self.test.as_mut().and_then(|t| t.stash.take()) {
            self.nn_live.close_all(
                crate::nnlive::CloseWhy::RunEnd,
                crate::nnlive::Clock { seq: self.clock_seq, tick: self.world.tick, time_s: self.world.time_s },
            );
            self.ml_task = None;
            self.task = None;
            self.world = st.world;
            self.quests = st.quests;
            self.gates = st.gates;
            self.last_scan = st.last_scan;
            self.log("world", Level::Info, "ML 결정 통과를 멈추고 라이브 월드로 돌아옴", None);
        }
    }

    pub fn test_rev(&self) -> u64 {
        let sess = self.test.as_ref().map_or(0, |t| t.test_rev);
        let rec = self.recording.as_ref().map_or(0, |r| r.test_rev);
        sess.max(rec)
    }

    pub(crate) fn bump_test_rev(&mut self) -> u64 {
        let next = self.test_rev().saturating_add(1);
        if let Some(t) = self.test.as_mut() {
            t.test_rev = next;
        }
        if let Some(r) = self.recording.as_mut() {
            r.test_rev = next;
        }
        next
    }

    pub(crate) fn stamp_keyframe_ctx(&self, kf: &mut Keyframe) {
        let proc_stage = self
            .procedure_json()
            .and_then(|v| v.get("stage").and_then(Value::as_str).map(String::from));
        let phase_ref = self
            .nn_live
            .active_now()
            .first()
            .map(|a| format!("{}/{}", a.scenario, a.phase));
        kf.ctx = KeyframeCtx {
            log_seq: self.audit.last_seq(),
            quest_step: kf.step.clone().or_else(|| self.current_step_label()),
            procedure_stage: proc_stage,
            phase_ref,
        };
        if kf.nn_open.is_empty() {
            kf.nn_open = self.nn_live.open_span_ids();
        }
    }

    pub(crate) fn push_recorded_keyframe(&mut self, mut kf: Keyframe) -> Option<usize> {
        let max_kf = self.test_cfg.record.max_keyframes;
        let can_push = self.recording.as_ref().is_some_and(|r| r.keyframes.len() < max_kf);
        if !can_push {
            if let Some(r) = self.recording.as_mut() {
                r.dropped_keyframes += 1;
            }
            return None;
        }
        self.stamp_keyframe_ctx(&mut kf);
        let rev = self.bump_test_rev();
        kf.created_rev = rev;
        kf.rev = rev;
        if let Some(r) = self.recording.as_mut() {
            kf.index = r.keyframes.len() as u32;
            r.keyframes.push(kf);
            return Some(r.keyframes.len() - 1);
        }
        None
    }

    pub(crate) fn bump_recorded_keyframe_rev(&mut self, index: usize) {
        let rev = self.bump_test_rev();
        if let Some(kf) = self.recording.as_mut().and_then(|r| r.keyframes.get_mut(index)) {
            kf.rev = rev;
        }
    }

    fn begin_pass(&mut self, pass: &str, test_id: &str) {
        self.pass = pass.into();
        self.run_id = format!("{test_id}-{pass}");
        self.ts.set_run(&self.run_id, self.world.tick);
        self.nn_live.sync_run(
            &self.run_id,
            &self.pass,
            1,
            &self.world.cfg.body,
            self.world.cfg.noto.base,
            self.world.cfg.noto.fingers,
            crate::nnlive::Clock { seq: self.clock_seq, tick: self.world.tick, time_s: self.world.time_s },
        );
        self.ml.reset();
        self.ml_task = None;
        let rec = Recording::new(pass, &self.run_id, self.world.tick);
        self.recording = Some(rec);
        if pass == "game" {
            if let Ok(mut kf) = capture_keyframe(&self.world, &self.cam, self.test_cfg.record.quality, 0, self.stereo_baseline()) {
                kf.nn_open = self.nn_live.open_span_ids();
                let _ = self.push_recorded_keyframe(kf);
            }
        }
    }

    pub(crate) fn record_tick(&mut self) {
        let Some((started, is_game, n_video, n_kf)) = self.recording.as_ref().map(|r| (r.started_tick, r.pass == "game", r.video.len(), r.keyframes.len())) else { return };
        let cfg = self.test_cfg.record.clone();
        let since = self.world.tick.saturating_sub(started);
        let video = since % cfg.video_every_ticks.max(1) as u64 == 0 && n_video < cfg.max_video_frames;
        let key = is_game && since % cfg.keyframe_every_ticks.max(1) as u64 == 0;
        if video {
            if let Ok(v) = capture_video(&self.world, &self.cam, &cfg) {
                if let Some(r) = self.recording.as_mut() {
                    r.video.push(v);
                }
            }
        }
        if since % BELIEF_EVERY_TICKS == 0 {
            let e = self.world.belief_error_m();
            if let Some(r) = self.recording.as_mut() {
                if r.belief_err.len() < MAX_BELIEF_POINTS {
                    r.belief_err.push((self.world.tick, e));
                }
            }
        }
        if key {
            let index = n_kf as u32;
            if n_kf >= cfg.max_keyframes {
                if let Some(r) = self.recording.as_mut() {
                    r.dropped_keyframes += 1;
                }
                return;
            }
            if let Ok(mut kf) = capture_keyframe(&self.world, &self.cam, cfg.quality, index, self.stereo_baseline()) {
                kf.step = self.current_step_label();
                kf.nn_open = self.nn_live.open_span_ids();
                let _ = self.push_recorded_keyframe(kf);
            }
        }
    }

    pub(crate) fn current_step_label(&self) -> Option<String> {
        let id = self.quests.current_id()?;
        let q = self.quests.get(&id)?;
        q.steps.iter().find(|s| s.state == noto_quest::State::Active).map(|s| s.label_ko.clone())
    }

    pub fn test_advance(&mut self, budget_ticks: u32) -> Result<TestPhase> {
        let Some(phase) = self.test.as_ref().map(|t| t.phase) else { return Err(anyhow!("진행 중인 통합 테스트가 없습니다")) };
        match phase {
            TestPhase::Game | TestPhase::Decide => {
                let start = self.test.as_ref().map_or(0, |t| t.started_tick);
                for _ in 0..budget_ticks.max(1) {
                    if !self.busy() {
                        break;
                    }
                    self.tick();
                    if self.world.tick.saturating_sub(start) > self.test_cfg.max_pass_ticks {
                        self.log("quest", Level::Warn, format!("통합 테스트: 통과당 틱 한도 {}를 넘어 중지", self.test_cfg.max_pass_ticks), None);
                        self.cancel();
                        break;
                    }
                }
                if !self.busy() {
                    self.end_pass(phase)?;
                }
            }
            TestPhase::Perceive => self.perceive_some()?,
            TestPhase::Done => {}
        }
        Ok(self.test.as_ref().map_or(TestPhase::Done, |t| t.phase))
    }

    fn end_pass(&mut self, phase: TestPhase) -> Result<()> {
        self.nn_live.close_all(
            crate::nnlive::CloseWhy::RunEnd,
            crate::nnlive::Clock { seq: self.clock_seq, tick: self.world.tick, time_s: self.world.time_s },
        );
        let mut rec = self.recording.take();
        if let Some(r) = rec.as_mut() {
            r.ended_tick = Some(self.world.tick);
        }
        let quest = self.quests.all().last().cloned();
        let traj = self.traj_digests.clone();
        let Some(t) = self.test.as_mut() else { return Ok(()) };
        t.test_rev = t.test_rev.saturating_add(1);
        match phase {
            TestPhase::Game => {
                t.game = rec;
                t.game_quest = quest;
                if let Some(m) = t.manifest.as_mut() {
                    m.traj_digests = traj;
                }
                t.phase = TestPhase::Perceive;
                t.perceive_next = 0;
                self.ml.reset();
            }
            TestPhase::Decide => {
                t.decide = rec;
                t.ml_quest = quest;
                t.phase = TestPhase::Done;
                self.pass = "game".into();
                if let Some(st) = t.stash.take() {
                    self.world = st.world;
                    self.quests = st.quests;
                    self.gates = st.gates;
                    self.last_scan = st.last_scan;
                    self.task = None;
                    self.ml_task = None;
                }
                let id = t.id.clone();
                let ok = |q: &Option<Quest>| q.as_ref().map_or("—".to_string(), |q| format!("{:?}", q.state).to_lowercase());
                let msg = format!("통합 테스트 {id} 끝: 게임 {} · ML {}", ok(&t.game_quest), ok(&t.ml_quest));
                self.log("quest", Level::Info, msg, None);
                for l in self.db.flush() {
                    self.audit.info("sds", format!("[DB] {l}"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn quest_labels(&self, q: Option<&Quest>) -> Vec<String> {
        let Some(q) = q else { return vec![] };
        let resources: Vec<String> = match &q.plan {
            noto_search::Plan::Fetch { targets, .. } | noto_search::Plan::Collect { targets } => targets.iter().map(|t| t.resource.clone()).collect(),
            _ => vec![],
        };
        resources
            .iter()
            .flat_map(|r| match world_sim::ItemKind::from_id(r) {
                Some(k) if !k.from_node() => vec![r.clone()],
                _ => self.ml.source_labels_for(r),
            })
            .collect()
    }

    fn perceive_some(&mut self) -> Result<()> {
        let Some(mut t) = self.test.take() else { return Ok(()) };
        let wanted = self.quest_labels(t.game_quest.as_ref());
        let n = t.game.as_ref().map_or(0, |g| g.keyframes.len());
        let cap = (t.perceive_next + self.test_cfg.perceive_per_step.max(1)).min(n);
        let budget_ms = self.test_cfg.perceive_budget_ms;
        let t_start = Instant::now();
        let prev_pass = std::mem::replace(&mut self.pass, "game".into());
        let run = t.game.as_ref().map(|g| g.run_id.clone()).unwrap_or_default();
        let prev_run = std::mem::replace(&mut self.run_id, run);
        let mut decoded: HashMap<usize, Rgb> = HashMap::new();
        if let Some(g) = t.game.as_ref() {
            let mut intrs: Vec<(usize, crate::vision_compat::Intrinsics)> = Vec::new();
            for i in t.perceive_next..cap {
                if let Some(k) = g.keyframes.get(i) {
                    if let Ok(rgb) = k.rgb() {
                        decoded.insert(i, rgb);
                        intrs.push((i, k.intr));
                    }
                }
            }
            let refs: Vec<(&Rgb, Option<&crate::vision_compat::Intrinsics>)> = intrs
                .iter()
                .filter_map(|(i, intr)| decoded.get(i).map(|rgb| (rgb, Some(intr))))
                .collect();
            if let Err(e) = self.ml.prefetch_with_cameras(&refs) {
                t.notes.push(format!("일괄 추론 실패, 키프레임마다 다시 계산: {e}"));
            }
        }
        let mut processed = 0usize;
        let mut end = t.perceive_next;
        for i in t.perceive_next..cap {
            if processed >= 1 && budget_ms > 0 && t_start.elapsed().as_millis() as u64 >= budget_ms {
                break;
            }
            let Some(mut kf) = t.game.as_mut().and_then(|g| g.keyframes.get_mut(i)).map(|k| std::mem::replace(k, placeholder(k))) else { continue };
            let res = match decoded.remove(&i) {
                Some(rgb) => self.ml_analyse_with_rgb(&mut kf, &rgb, &wanted, None),
                None => self.ml_analyse(&mut kf, &wanted, None),
            };
            if let Ok(out) = res {
                if let Some(s) = out.sighting {
                    let rays = t.rays.entry(s.track_id).or_default();
                    if rays.last().map_or(true, |r| (r.x - s.pose.x).hypot(r.z - s.pose.z) > 0.2) {
                        rays.push(Bearing { x: s.pose.x, z: s.pose.z, heading: s.heading, sigma: s.sigma });
                        if rays.len() > 4 {
                            rays.remove(0);
                        }
                    }
                    if let Ok(fix) = triangulate(rays, 6.0) {
                        let label = out.frame.target.as_ref().map(|x| x.label.clone()).unwrap_or_default();
                        if let Some(d) = s.depth_range_m.filter(|_| crate::vision_compat::scale_usable(&fix, &s)) {
                            let dmode = self.ml.depth_mode();
                            let scope = Scope::new(Track::Depth, &dmode, &label);
                            self.active_sds_mut().record_calibration(&scope, "scale", fix.range_m as f64, d as f64);
                        }
                        if let Some(tv) = kf.ml.as_mut().and_then(|m| m.target.as_mut()) {
                            tv.tri_range_m = Some(fix.range_m);
                            tv.tri_parallax_deg = Some(fix.parallax_deg);
                            tv.tri_sigma_m = Some(fix.sigma_m);
                            tv.depth_scale = s.depth_range_m.filter(|_| crate::vision_compat::scale_usable(&fix, &s)).map(|d| fix.range_m / d);
                            tv.range_used_m = Some(fix.range_m);
                        }
                    }
                }
                let target = wanted.first().cloned().unwrap_or_default();
                self.record_keyframe_row("game", &kf, &target);
            } else if let Err(e) = res {
                t.notes.push(format!("키프레임 {i}: {e}"));
            }
            kf.truth.shed_pixels();
            t.test_rev = t.test_rev.saturating_add(1);
            kf.rev = t.test_rev;
            if let Some(g) = t.game.as_mut() {
                g.test_rev = t.test_rev;
                if let Some(slot) = g.keyframes.get_mut(i) {
                    *slot = kf;
                }
            }
            processed += 1;
            end = i + 1;
        }
        t.perceive_next = end;
        self.pass = prev_pass;
        self.run_id = prev_run;
        if end >= n && t.live && !t.ml_needed {
            t.phase = TestPhase::Done;
            self.log("quest", Level::Info, format!("기록 {} 끝: ML 결정 통과는 찾을 대상이 없어 생략", t.id), None);
            for l in self.db.flush() {
                self.audit.info("sds", format!("[DB] {l}"));
            }
            self.test = Some(t);
            return Ok(());
        }
        if end >= n {
            let (seed, body, base, cmd, id) = (t.seed, t.body, t.base, t.command.clone(), t.id.clone());
            if t.live {
                let start = t.start_world.take().unwrap_or_else(|| self.world.clone());
                let world = std::mem::replace(&mut self.world, start);
                let quests = std::mem::replace(&mut self.quests, QuestBook::new());
                let gates = std::mem::replace(&mut self.gates, PrecisionGates::new(self.cfg.gates.clone()));
                t.stash = Some(LiveStash { world, quests, gates, last_scan: self.last_scan.take() });
                self.task = None;
                self.ml_task = None;
                self.focus = None;
            } else {
                let mut hand = self.cfg.noto;
                hand.base = base;
                self.cfg.body = body;
                self.reset_with(Some(seed), None, Some(hand))?;
            }
            t.phase = TestPhase::Decide;
            t.started_tick = self.world.tick;
            self.test = Some(t);
            self.begin_pass("ml", &id);
            let s = self.submit(&cmd);
            if !s.ok {
                if let Some(t) = self.test.as_mut() {
                    t.error = s.error.clone();
                    t.phase = TestPhase::Done;
                }
                self.pass = "game".into();
            }
            return Ok(());
        }
        self.test = Some(t);
        Ok(())
    }

    fn kf_brief(kf: &Keyframe) -> Value {
        let ml = kf.ml.as_ref();
        let sds_cells: serde_json::Map<String, Value> = kf
            .sds_detail
            .iter()
            .map(|(k, (v, n, sc))| {
                let kind = if k == "balance" { "shortfall" } else { "z" };
                (k.clone(), json!({"value": r3(*v), "n": n, "scope": sc, "kind": kind}))
            })
            .collect();
        json!({
            "i": kf.index, "rev": kf.rev, "tick": kf.tick, "t": r3(kf.time_s as f32),
            "step": kf.step, "action": kf.action,
            "ctx": kf.ctx,
            "nn_open": kf.nn_open,
            "pose": [r3(kf.pose.x), r3(kf.pose.z), r3(kf.pose.yaw.to_degrees())],
            "balance": kf.balance.as_ref().map(|b| json!({"source": b.source, "source_ko": b.source_ko, "margin": r3(b.margin), "worst": b.worst_joint, "worst_ratio": r3(b.worst_ratio), "head_tilt_deg": r3(b.head_tilt_deg)})),
            "ml": ml.map(|m| json!({
                "absrel": m.depth_metrics.map(|d| r3(d.absrel)),
                "iou_truth": m.track_iou_truth.map(r3),
                "locate_err_px": m.locate_err_px.map(r3),
                "flags": m.flags,
                "target": m.target.as_ref().map(|t| json!({"track": t.track_id, "label": t.label, "bearing": r3(t.rel_bearing_deg), "depth_range": t.depth_range_m.map(r3), "tri_range": t.tri_range_m.map(r3), "true_range": t.range_true_m.map(r3), "scale": t.depth_scale.map(r3)})),
                "stage_ms": m.stage_ms,
            })),
            "decision": kf.decision.as_ref().map(|d| json!({
                "label": d.decision, "kind": d.kind, "note": d.note,
                "residual": d.residual_deg.map(r3), "forecast_method": d.forecast_method,
                "band": d.band.map(|b| b.map(r3)), "pos_model": d.pos_model,
                "estimate": d.estimate.map(|e| [r3(e.0), r3(e.1), r3(e.2), r3(e.3)]),
                "fix": d.fix.map(|f| [r3(f.0), r3(f.1), r3(f.2), r3(f.3), r3(f.4)]),
                "executed": d.executed, "ok": d.ok,
            })),
            "sds": kf.sds.iter().map(|(k, v)| (k.clone(), json!(r3(*v)))).collect::<serde_json::Map<String, Value>>(),
            "sds_cells": sds_cells,
            "event": kf.event.as_ref().map(|e| json!({"act": e.act, "kind": e.kind, "label": e.label, "ok": e.ok, "parts": e.parts})),
            "focus": kf.focus.as_ref().map(|f| json!({"action": f.action, "action_ko": f.action_ko, "phase": f.phase, "parts": f.parts, "margin": r3(f.body.margin), "trunk_deg": r3(f.body.trunk_deg), "slip_ratio": f.hand.grip.map(|g| r3(g.slip_ratio)), "closure": r3(f.hand.closure)})),
        })
    }

    fn quest_json(q: &Option<Quest>) -> Value {
        match q {
            None => Value::Null,
            Some(q) => json!({
                "id": q.id, "title": q.title, "state": q.state,
                "steps": q.steps.iter().map(|s| json!({"kind": s.kind, "label_ko": s.label_ko, "state": s.state})).collect::<Vec<_>>(),
                "failure": q.failure, "result": q.result,
            }),
        }
    }

    pub fn api_test_status(&self) -> Result<Value, String> {
        let Some(t) = &self.test else { return Ok(json!({"active": false})) };
        let live = self.recording.as_ref();
        let rows = |r: Option<&Recording>| -> Value {
            match r {
                None => json!({"keyframes": [], "video": 0, "start_tick": null, "end_tick": null}),
                Some(r) => json!({
                    "run_id": r.run_id, "start_tick": r.started_tick, "end_tick": r.ended_tick,
                    "video": r.video.len(), "dropped": r.dropped_keyframes,
                    "memory": memory_json(r),
                    "keyframes": r.keyframes.iter().map(Self::kf_brief).collect::<Vec<_>>(),
                    "vision": r.vision.iter().map(|m| json!({"tick": m.tick, "kind": m.kind, "what": m.what, "s0": m.sigma_before_m.map(r3), "s1": m.sigma_after_m.map(r3), "innov": m.innovation_m.map(r3)})).collect::<Vec<_>>(),
                    "belief_err": r.belief_err.iter().map(|(t, e)| json!([t, r3(*e)])).collect::<Vec<_>>(),
                }),
            }
        };
        let game = if t.phase == TestPhase::Game { live } else { t.game.as_ref() };
        let decide = if t.phase == TestPhase::Decide { live } else { t.decide.as_ref() };
        let game_quest = if t.phase == TestPhase::Game { self.quests.all().last().cloned() } else { t.game_quest.clone() };
        let ml_quest = if t.phase == TestPhase::Decide { self.quests.all().last().cloned() } else { t.ml_quest.clone() };
        let manifest_out = t.manifest.as_ref().map(|m| {
            let mut copy = m.clone();
            if t.phase == TestPhase::Game && copy.traj_digests.is_empty() {
                copy.traj_digests = self.traj_digests.clone();
            }
            copy
        });
        Ok(json!({
            "active": true,
            "test_rev": self.test_rev(),
            "live": t.live, "ml_needed": t.ml_needed,
            "id": t.id, "command": t.command, "seed": t.seed, "preset": t.body.id(), "body": t.body, "base": t.base,
            "phase": t.phase, "tick": self.world.tick,
            "perceive": {"done": t.perceive_next, "total": t.game.as_ref().map_or(0, |g| g.keyframes.len())},
            "modes": {
                "depth": self.ml.depth_mode(),
                "locate": self.ml.locate_mode(),
                "track": self.ml.track_mode(),
                "ttm": self.has_ttm(),
                "ttm_fp8": self.gates.ttm().map(|x| x.fp8().to_string()),
                "ttm_verified": self.gates.ttm().map_or(false, |x| x.fingerprint().verified),
            },
            "game": rows(game), "decide": rows(decide),
            "game_quest": Self::quest_json(&game_quest), "ml_quest": Self::quest_json(&ml_quest),
            "notes": t.notes, "error": t.error,
            "manifest": manifest_out,
            "keyframe_every_ticks": self.test_cfg.record.keyframe_every_ticks,
        }))
    }

    /// Incremental timeline delta (`test_status_delta`, Phase 5 – Part 2 · D3).
    pub fn api_test_status_delta(&self, args: &Value) -> Result<Value, String> {
        let cur_rev = self.test_rev();
        if args.get("if_rev").and_then(Value::as_u64) == Some(cur_rev) {
            return Ok(json!({"unchanged": true, "rev": cur_rev}));
        }
        let Some(t) = &self.test else { return Ok(json!({"active": false, "test_rev": cur_rev})) };
        let req_id = args.get("id").and_then(Value::as_str);
        let since_rev = if req_id.is_some_and(|id| id != t.id) {
            0
        } else {
            args.get("since_rev").and_then(Value::as_u64).unwrap_or(0)
        };
        let v_from_g = args.get("vision_from_game").and_then(Value::as_u64).unwrap_or(0) as usize;
        let v_from_d = args.get("vision_from_decide").and_then(Value::as_u64).unwrap_or(0) as usize;
        let b_from_g = args.get("belief_from_game").and_then(Value::as_u64).unwrap_or(0) as usize;
        let b_from_d = args.get("belief_from_decide").and_then(Value::as_u64).unwrap_or(0) as usize;
        let delta_rec = |r: Option<&Recording>, v_from: usize, b_from: usize| -> Value {
            let Some(r) = r else {
                return json!({
                    "total_keyframes": 0, "video": 0, "start_tick": null, "end_tick": null,
                    "appended": [], "updated": [], "vision_tail": [], "vision_total": 0, "belief_tail": [], "belief_total": 0
                });
            };
            let mut appended = Vec::new();
            let mut updated = Vec::new();
            for kf in &r.keyframes {
                if kf.created_rev > since_rev {
                    appended.push(Self::kf_brief(kf));
                } else if kf.rev > since_rev {
                    updated.push(Self::kf_brief(kf));
                }
            }
            let vision_tail: Vec<Value> = r
                .vision
                .iter()
                .skip(v_from)
                .map(|m| json!({"tick": m.tick, "kind": m.kind, "what": m.what, "s0": m.sigma_before_m.map(r3), "s1": m.sigma_after_m.map(r3), "innov": m.innovation_m.map(r3)}))
                .collect();
            let belief_tail: Vec<Value> = r.belief_err.iter().skip(b_from).map(|(tk, e)| json!([tk, r3(*e)])).collect();
            json!({
                "run_id": r.run_id, "start_tick": r.started_tick, "end_tick": r.ended_tick,
                "video": r.video.len(), "dropped": r.dropped_keyframes,
                "total_keyframes": r.keyframes.len(),
                "memory": memory_json(r),
                "appended": appended,
                "updated": updated,
                "vision_tail": vision_tail,
                "vision_total": r.vision.len(),
                "belief_tail": belief_tail,
                "belief_total": r.belief_err.len(),
            })
        };
        let live = self.recording.as_ref();
        let game = if t.phase == TestPhase::Game { live } else { t.game.as_ref() };
        let decide = if t.phase == TestPhase::Decide { live } else { t.decide.as_ref() };
        let game_quest = if t.phase == TestPhase::Game { self.quests.all().last().cloned() } else { t.game_quest.clone() };
        let ml_quest = if t.phase == TestPhase::Decide { self.quests.all().last().cloned() } else { t.ml_quest.clone() };
        Ok(json!({
            "active": true,
            "id": t.id,
            "live": t.live,
            "ml_needed": t.ml_needed,
            "command": t.command,
            "test_rev": cur_rev,
            "since_rev": since_rev,
            "phase": t.phase,
            "tick": self.world.tick,
            "perceive": {"done": t.perceive_next, "total": t.game.as_ref().map_or(0, |g| g.keyframes.len())},
            "game": delta_rec(game, v_from_g, b_from_g),
            "decide": delta_rec(decide, v_from_d, b_from_d),
            "game_quest": Self::quest_json(&game_quest),
            "ml_quest": Self::quest_json(&ml_quest),
            "notes": t.notes,
            "error": t.error,
        }))
    }

    /// Synchronized multi-panel context at a cursor `tick` (`test_context`, Phase 5 – Part 2 · D4).
    pub fn api_test_context(&self, args: &Value) -> Result<Value, String> {
        let pass = args.get("pass").and_then(Value::as_str).unwrap_or("game");
        let tick = args.get("tick").and_then(Value::as_u64).unwrap_or(self.world.tick);
        let rec = self.rec_of(pass);
        let (kf_json, video_idx, video_tick, video_total, center_seq) = match rec {
            Some(r) => {
                let kf = r.keyframes.iter().min_by_key(|k| k.tick.abs_diff(tick));
                let vf = r.video.iter().enumerate().min_by_key(|(_, v)| v.tick.abs_diff(tick));
                let seq = kf.map(|k| k.ctx.log_seq).filter(|&s| s > 0);
                let kf_v = kf.map(|k| {
                    let mut b = Self::kf_brief(k);
                    b["delta_ticks"] = json!(k.tick.abs_diff(tick));
                    b
                });
                (kf_v, vf.map(|(i, _)| i), vf.map(|(_, v)| v.tick), r.video.len(), seq)
            }
            None => (None, None, None, 0, None),
        };
        let (lines, _) = self.audit.tail(0, 4000, None);
        let nearest_seq = center_seq.or_else(|| {
            lines
                .iter()
                .filter_map(|l| l.tick.map(|tk| (tk.abs_diff(tick), l.seq)))
                .min_by_key(|(d, _)| *d)
                .map(|(_, s)| s)
        });
        let log_window = nearest_seq.map(|s| json!({
            "center_seq": s,
            "from_seq": s.saturating_sub(8),
            "to_seq": s.saturating_add(8),
        }));
        let phase_spans: Vec<Value> = self
            .nn_live
            .spans_for_worker(1)
            .into_iter()
            .filter(|sp| sp.start.tick <= tick && sp.end.map_or(true, |e| e.tick >= tick))
            .map(|sp| json!({
                "id": sp.id, "scenario": sp.scenario, "phase": sp.phase,
                "start_tick": sp.start.tick, "end_tick": sp.end.map(|e| e.tick),
                "close": sp.close,
            }))
            .collect();
        let mut gate_marks: Vec<Value> = Vec::new();
        for lm in self.gates.landmarks() {
            for gm in &lm.trace {
                if gm.tick.abs_diff(tick) <= 5 {
                    gate_marks.push(json!({
                        "landmark": lm.id,
                        "label": lm.label,
                        "tick": gm.tick,
                        "observed": gm.observed,
                        "expected": gm.expected,
                        "lo": gm.lo,
                        "hi": gm.hi,
                        "z": gm.z,
                        "outcome": gm.outcome,
                        "rescue": gm.rescue,
                        "method": gm.method,
                        "sds_sd": gm.sds_sd,
                    }));
                }
            }
        }
        gate_marks.sort_by_key(|m| m["tick"].as_u64().unwrap_or(0));
        Ok(json!({
            "pass": pass,
            "tick": tick,
            "keyframe": kf_json,
            "video_index": video_idx,
            "video_tick": video_tick,
            "video_total": video_total,
            "log_window": log_window,
            "phase_spans": phase_spans,
            "sds_watch": self.sds_watch.at_tick(tick),
            "gate_marks": gate_marks,
        }))
    }

    /// Replay a [`ReplayManifest`] in Frozen mode against its saved content-addressed snapshot (if available)
    /// and compare 50-tick trajectory digests (Phase 2 – Part 3 · D6).
    pub fn replay_manifest(&mut self, manifest: &ReplayManifest, max_ticks: u64) -> Result<Value> {
        let prev_mode = self.feedback.mode().clone();
        let prev_run = self.run_id.clone();
        let prev_pass = self.pass.clone();
        let mut frozen_mode = manifest.sds_mode.clone();
        frozen_mode.default = crate::feedback::Mode::Frozen;
        let loaded_snap = self.data_dir().and_then(|d| {
            crate::sds::SdsStore::load_snapshot(&d.join("sds").join("snap"), &manifest.snapshot_digest).ok()
        });
        match loaded_snap {
            Some(snap) => self.feedback.set_mode_with_snapshot(frozen_mode, snap),
            None => self.feedback.set_mode(frozen_mode, &self.sds),
        }
        let mut hand = self.cfg.noto;
        hand.base = manifest.base;
        self.cfg.body = manifest.body;
        self.reset_with(Some(manifest.seed), None, Some(hand))?;
        self.test_seq += 1;
        self.run_id = format!("{}-replay-{}", self.session_id, self.test_seq);
        self.pass = "game".into();
        self.nn_live.sync_run(
            &self.run_id,
            &self.pass,
            1,
            &self.world.cfg.body,
            self.world.cfg.noto.base,
            self.world.cfg.noto.fingers,
            crate::nnlive::Clock { seq: self.clock_seq, tick: self.world.tick, time_s: self.world.time_s },
        );
        self.ml.reset();
        self.clear_traj_digests();
        let sub = self.submit(&manifest.command);
        if !sub.ok {
            self.run_id = prev_run;
            self.pass = prev_pass;
            self.set_sds_mode(prev_mode);
            return Err(anyhow!("{}", sub.error.unwrap_or_else(|| "재현 명령 제출 실패".into())));
        }
        let ticks = self.run_until_idle(max_ticks.max(1));
        self.nn_live.close_all(
            crate::nnlive::CloseWhy::RunEnd,
            crate::nnlive::Clock { seq: self.clock_seq, tick: self.world.tick, time_s: self.world.time_s },
        );
        let replay_digests = self.traj_digests.clone();
        let fb_at_end = self.feedback.snapshot_values();
        self.run_id = prev_run;
        self.pass = prev_pass;
        self.set_sds_mode(prev_mode);

        let mut first_mismatch: Option<u64> = None;
        let orig = &manifest.traj_digests;
        for (i, (t_a, d_a)) in orig.iter().enumerate() {
            match replay_digests.get(i) {
                Some((t_b, d_b)) if t_a == t_b && d_a == d_b => {}
                _ => {
                    first_mismatch = Some(*t_a);
                    break;
                }
            }
        }
        if first_mismatch.is_none() && replay_digests.len() != orig.len() {
            first_mismatch = replay_digests.get(orig.len()).map(|(t, _)| *t).or_else(|| orig.last().map(|(t, _)| *t));
        }
        let matched = first_mismatch.is_none();
        self.log(
            "replay",
            if matched { Level::Info } else { Level::Warn },
            format!("재현 비교: {} ({ticks}틱, 50틱 지문 {}건)", if matched { "일치" } else { "불일치" }, replay_digests.len()),
            Some(json!({"matched": matched, "first_mismatch_tick": first_mismatch, "ticks": ticks})),
        );
        Ok(json!({
            "matched": matched,
            "first_mismatch_tick": first_mismatch,
            "ticks": ticks,
            "original_digests": orig,
            "replay_digests": replay_digests,
            "feedback_at_end": fb_at_end,
            "manifest": manifest,
        }))
    }

    pub(crate) fn rec_of(&self, pass: &str) -> Option<&Recording> {
        let t = self.test.as_ref()?;
        match (pass, t.phase) {
            ("game", TestPhase::Game) | ("ml", TestPhase::Decide) => self.recording.as_ref(),
            ("game", _) => t.game.as_ref(),
            _ => t.decide.as_ref(),
        }
    }

    pub fn api_test_keyframe(&self, pass: &str, index: usize) -> Result<Value, String> {
        self.api_test_keyframe_with(pass, index, true)
    }

    pub fn api_test_keyframe_with(&self, pass: &str, index: usize, close_ups: bool) -> Result<Value, String> {
        let rec = self.rec_of(pass).ok_or("이 통과의 기록이 없습니다")?;
        let kf = rec.keyframes.get(index).ok_or_else(|| format!("키프레임 {index}이(가) 없습니다"))?;
        let mut v = json!({
            "brief": Self::kf_brief(kf),
            "frame_webp": b64(&kf.webp),
            "ml_webp": kf.ml_webp.as_deref().map(b64),
            "width": kf.intr.width, "height": kf.intr.height,
            "ml": kf.ml, "decision": kf.decision, "balance": kf.balance,
            "focus": kf.focus, "event": kf.event,
            "truth": kf.truth.objects.iter().map(|o| json!({"entity": o.entity, "label": o.label, "label_ko": o.label_ko, "bbox": [o.bbox[0] as f32 / kf.truth.src_width as f32, o.bbox[1] as f32 / kf.truth.src_height as f32, o.bbox[2] as f32 / kf.truth.src_width as f32, o.bbox[3] as f32 / kf.truth.src_height as f32], "depth_m": r3(o.depth_m)})).collect::<Vec<_>>(),
        });
        if close_ups {
            v["focus_webp"] = Value::Object(kf.focus_webp.iter().map(|(k, w)| (k.clone(), json!(b64(w)))).collect());
        }
        Ok(v)
    }

    pub fn api_test_video(&self, pass: &str, from: usize, count: usize) -> Result<Value, String> {
        let rec = self.rec_of(pass).ok_or("이 통과의 기록이 없습니다")?;
        let frames: Vec<Value> = rec.video.iter().skip(from).take(count.clamp(1, 120)).map(|v| json!({"tick": v.tick, "t": r3(v.time_s as f32), "webp": b64(&v.webp)})).collect();
        Ok(json!({"from": from, "total": rec.video.len(), "frames": frames}))
    }

    pub fn api_test_stats(&self) -> Result<Value, String> {
        let stats = self.db.stats();
        let mut nn: Vec<Value> = Vec::new();
        let def_axes = noto_core::nn::known_axes_from_defs(&self.world.body);
        for s in Source::ALL.into_iter().filter(|s| *s != Source::Quiet) {
            for g in Segment::ALL {
                let scope = Scope::new(Track::Balance, &format!("{}>{}", s.id(), g.id()), "");
                let mut axes: std::collections::BTreeSet<String> = def_axes.iter().cloned().collect();
                if let Some(st) = self.sds.merged_scope_by_key(Track::Balance.ledger(), &scope.key_primary()) {
                    for k in st.baseline.keys() {
                        if !k.contains(':') {
                            axes.insert(k.clone());
                        }
                    }
                }
                for axis in axes {
                    let rf = reference(s, g, &axis, &noto_core::balance::RefCtx { load_ratio: 0.0, load_dist_m: 0.0, slope_deg: 10.0, single_support: false });
                    let rep = self.sds.exact_reference(&scope, &axis);
                    let base = self.sds.exact_baseline(&scope, &axis);
                    if rep.is_none() && base.is_none() {
                        continue;
                    }
                    nn.push(json!({
                        "source": s.id(), "source_ko": s.label_ko(), "target": g.id(), "target_ko": g.label_ko(), "axis": axis,
                        "mean": rep.as_ref().map(|r| r.mean).or(base.map(|b| b.0)),
                        "n": rep.as_ref().map(|r| r.n).or(base.map(|b| b.2)),
                        "dev": rep.as_ref().map(|r| r.mean_dev), "in_band": rep.as_ref().map(|r| r.in_band_rate),
                        "reference": rep.as_ref().map(|r| r.mean_ref).filter(|x| x.is_finite()).or(rf.as_ref().map(|r| r.value as f64)),
                        "tolerance": rep.as_ref().map(|r| r.mean_tol).filter(|x| x.is_finite()).or(rf.as_ref().map(|r| r.tolerance as f64)),
                        "cite": rf.as_ref().map(|r| r.cite),
                    }));
                }
            }
        }
        let mut gains: BTreeMap<String, f64> = BTreeMap::new();
        for (s, g) in &self.world.ff_gains {
            gains.insert(s.id().into(), *g as f64);
        }
        let runs: Vec<Value> = self.db.runs().iter().rev().take(20).map(|r| json!({"run_id": r.run_id, "pass": r.pass, "kind": r.kind, "target": r.target, "preset": r.preset, "ok": r.ok, "failure": r.failure, "time_s": r.time_s, "steps": r.steps})).collect();
        Ok(json!({"stats": stats, "nn": nn, "axes": def_axes, "ff_gains": gains, "runs": runs, "db_dir": self.db.dir().map(|d| d.display().to_string())}))
    }
}

const BELIEF_EVERY_TICKS: u64 = 5;
const MAX_BELIEF_POINTS: usize = 6000;

fn memory_json(r: &Recording) -> Value {
    let (pixels, images, video) = r.memory();
    json!({"truth_pixels": pixels, "images": images, "video": video})
}

fn placeholder(k: &Keyframe) -> Keyframe {
    Keyframe {
        index: k.index,
        created_rev: k.created_rev,
        rev: k.rev,
        tick: k.tick,
        time_s: k.time_s,
        pose: k.pose,
        intr: k.intr,
        webp: Vec::new(),
        webp_right: None,
        truth: crate::ml::KeyTruth { width: 0, height: 0, ids: vec![], depth: vec![], objects: vec![], positions: vec![], src_width: 0, src_height: 0 },
        step: None,
        action: None,
        balance: None,
        ml: None,
        ml_webp: None,
        decision: None,
        sds: BTreeMap::new(),
        sds_detail: BTreeMap::new(),
        ctx: k.ctx.clone(),
        event: k.event.clone(),
        focus: None,
        focus_webp: Vec::new(),
        nn_open: k.nn_open.clone(),
    }
}
