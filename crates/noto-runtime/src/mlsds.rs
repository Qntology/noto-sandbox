use std::collections::BTreeMap;

use anyhow::Result;
use noto_core::balance::{reference, Source};
use noto_sds::{BalanceRow, DriftState, Facet, KeyframeRow, Level, NnPhaseRow, QuestRun, Scope, ScopeKey, StepRecord, Track, Window};
use noto_vision::{features, flags, FlagCfg, FusionInput, Rgb, Sighting};
use serde_json::json;
use world_sim::encode_webp;

use crate::feedback::FeedbackPath;
use crate::ml::{overlay, MlFrame};
use crate::record::{Keyframe, MlView, PointView, TargetView, TrackView};
use crate::runtime::Runtime;

fn default_floor(mode: &str) -> f32 {
    if mode.starts_with("stand-in") && !mode.contains("measured") {
        0.35
    } else {
        0.0
    }
}

fn is_stand_in(mode: &str) -> bool {
    mode.starts_with("stand-in")
}

/// Predictive z-score `(x - m) / (s * sqrt(1 + 1/n_e))` with resolved observation count, key, and `thin` flag.
fn z_detail(rt: &Runtime, scope: &Scope, axis: &str, x: f64) -> Option<(f64, u64, String, bool)> {
    let est = rt.sds.estimate_mean(scope, axis, Window::Auto, 0.95)?;
    let sd = est.se * est.n_eff.max(1.0).sqrt();
    let pred_sd = sd * (1.0 + 1.0 / est.n_eff.max(1.0)).sqrt();
    let z = (x - est.value) / pred_sd.max(1e-6);
    Some((z, est.n, est.key, est.thin.is_some()))
}

fn slope_bucket(deg: f32) -> &'static str {
    if deg >= 7.0 {
        "up_hi"
    } else if deg >= 2.0 {
        "up_lo"
    } else if deg <= -7.0 {
        "down_hi"
    } else if deg <= -2.0 {
        "down_lo"
    } else {
        "flat"
    }
}

pub(crate) struct MlOut {
    pub frame: MlFrame,
    pub sighting: Option<Sighting>,
}

impl Runtime {
    pub(crate) fn row_meta(&self) -> (String, String, String) {
        let gen = self
            .sds
            .generation_id()
            .map(String::from)
            .unwrap_or_else(crate::generation_id);
        let digest = self.sds.combined_digest();
        let mode = serde_json::to_string(self.feedback.mode()).unwrap_or_else(|_| "live".into());
        (gen, digest, mode)
    }

    pub(crate) fn depth_scale(&self, label: &str) -> Option<f32> {
        let mode = self.ml.depth_mode();
        let scope = Scope::new(Track::Depth, &mode, label);
        self.feedback
            .estimate_factor(FeedbackPath::E, &self.sds, &scope, "scale", Window::Auto, self.world.tick)
            .filter(|est| is_stand_in(&mode) || est.key != scope.key_global())
            .filter(|est| !(matches!(est.state, DriftState::Shifted { .. }) && (est.window != Window::Span || est.n_eff < 4.0)))
            .map(|est| est.value as f32)
    }

    pub(crate) fn ml_analyse(&mut self, kf: &mut Keyframe, wanted: &[String], prefer: Option<u32>) -> Result<MlOut> {
        let rgb = kf.rgb()?;
        self.ml_analyse_with_rgb(kf, &rgb, wanted, prefer)
    }

    pub(crate) fn ml_analyse_with_rgb(&mut self, kf: &mut Keyframe, rgb: &Rgb, wanted: &[String], prefer: Option<u32>) -> Result<MlOut> {
        let right = kf.rgb_right()?;
        let lmode = self.ml.locate_mode();
        let floor_default = default_floor(&lmode);
        let tick = self.world.tick;
        let floors: BTreeMap<String, f32> = wanted
            .iter()
            .map(|l| {
                let s = Scope::new(Track::Locate, l, &lmode);
                let learned = self
                    .feedback
                    .estimate_mean(FeedbackPath::D, &self.sds, &s, "score", Window::Auto, tick)
                    .filter(|est| is_stand_in(&lmode) || est.key == s.key_secondary())
                    .filter(|est| !(matches!(est.state, DriftState::Shifted { .. }) && (est.window != Window::Span || est.n_eff < 4.0)))
                    .map(|est| {
                        let sd = est.se * est.n_eff.max(1.0).sqrt();
                        noto_sds::estimate::pred_lower(est.value, sd, est.n_eff, 0.0228) as f32
                    });
                let f = learned.unwrap_or(floor_default).max(floor_default.min(0.2));
                (l.clone(), f)
            })
            .collect();
        let ground0 = self.world.surface_m(kf.pose.x, kf.pose.z);
        let world = &self.world;
        let height = move |x: f32, z: f32| world.surface_m(x, z) - ground0;
        let frame = self.ml.analyse_with(rgb, right.as_ref(), &kf.intr, &kf.pose, Some(&kf.truth), wanted, &|l| floors.get(l).copied().unwrap_or(floor_default), prefer, Some(&height))?;
        let (dmode, tmode) = (self.ml.depth_mode(), self.ml.track_mode());
        let label = frame.target.as_ref().map(|t| t.label.clone()).or_else(|| wanted.first().cloned()).unwrap_or_default();

        let dscope = Scope::new(Track::Depth, &dmode, &label);
        let point = frame.points.iter().find(|p| wanted.contains(&p.label)).cloned();
        let lscope = Scope::new(Track::Locate, &label, &lmode);
        let tscope = Scope::new(Track::Tracker, &label, &tmode);
        let target_track = frame.target.as_ref().and_then(|t| frame.tracks.iter().find(|m| m.id == t.track_id));

        let mut sds_lane: BTreeMap<String, f32> = BTreeMap::new();
        let mut sds_det: BTreeMap<String, (f32, u64, String)> = BTreeMap::new();
        if let Some(p) = &point {
            if let Some((z, n, key, _)) = z_detail(self, &lscope, "score", p.score as f64) {
                sds_lane.insert("locate".into(), z as f32);
                sds_det.insert("locate".into(), (z as f32, n, key));
            }
        }
        if let Some(t) = target_track {
            if let Some(iou) = t.iou_prev {
                if let Some((z, n, key, _)) = z_detail(self, &tscope, "iou_prev", iou as f64) {
                    sds_lane.insert("tracker".into(), z as f32);
                    sds_det.insert("tracker".into(), (z as f32, n, key));
                }
            }
        }
        if let Some(m) = &frame.depth_metrics {
            if let Some((z, n, key, _)) = z_detail(self, &dscope, "absrel", m.absrel as f64) {
                sds_lane.insert("depth".into(), z as f32);
                sds_det.insert("depth".into(), (z as f32, n, key));
            }
        }

        let stereo_stats = self.ml.stereo_stats();
        {
            let sds = self.active_sds_mut();
            if let Some(m) = &frame.depth_metrics {
                sds.record_baseline(&dscope, "absrel", m.absrel as f64);
                sds.record_baseline(&dscope, "delta1", m.delta1 as f64);
            }
            if let Some(st) = stereo_stats {
                sds.record_baseline(&dscope, "stereo_valid", st.valid_frac() as f64);
            }
            if let Some(f) = &frame.align {
                sds.record_baseline(&dscope, "align_a", f.a as f64);
                sds.record_baseline(&dscope, "align_b", f.b as f64);
                sds.record_baseline(&dscope, "align_inliers", f.inlier_share as f64);
                sds.record_baseline(&dscope, "align_err", f.median_rel_err as f64);
            }
            if let Some(p) = &point {
                sds.record_baseline(&lscope, "score", p.score as f64);
                sds.record_baseline(&lscope, "margin", p.margin as f64);
                sds.record_baseline(&lscope, "presence", p.presence as f64);
                if let Some(e) = frame.locate_err_px {
                    sds.record_baseline(&lscope, "err_px", e as f64);
                }
            }
            sds.record_agreement(&tscope, "alive", target_track.is_some());
            if let Some(t) = target_track {
                if let Some(iou) = t.iou_prev {
                    sds.record_baseline(&tscope, "iou_prev", iou as f64);
                }
                sds.record_baseline(&tscope, "area_frac", t.area_frac as f64);
                sds.record_baseline(&tscope, "score", t.score as f64);
                if let Some(iou) = frame.track_iou_truth {
                    sds.record_baseline(&tscope, "iou_truth", iou as f64);
                }
            }
        }

        let scale = self.depth_scale(&label);
        let sighting = frame.target.as_ref().map(|t| Sighting {
            keyframe: kf.index,
            pose: kf.pose,
            heading: t.heading,
            sigma: noto_vision::centroid_sigma(kf.intr.width, kf.intr.fov_deg, 2.0),
            extent: target_track.map_or(0.0, |m| (kf.intr.relative_bearing(&kf.pose, m.bbox[2], m.cy) - kf.intr.relative_bearing(&kf.pose, m.bbox[0], m.cy)).abs()),
            depth_range_m: t.depth_range_m,
            track_id: t.track_id,
            score: t.score,
        });
        let head_tilt = kf.balance.as_ref().map(|b| b.head_tilt_deg);
        let feats = features(&FusionInput {
            width: kf.intr.width,
            height: kf.intr.height,
            depth: Some(&frame.depth),
            point: point.as_ref(),
            mask: target_track.map(|t| &t.mask),
            depth_range_m: frame.target.as_ref().and_then(|t| t.depth_range_m),
            depth_scale: scale,
            tri_range_m: None,
            residual_deg: None,
            head_tilt_deg: head_tilt,
        });
        let mut fl: Vec<String> = flags(&feats, &FlagCfg::default()).into_iter().map(String::from).collect();
        let fscope = Scope::new(Track::Fusion, &label, "");
        let mut zmax = 0.0f64;
        let mut zmax_det: Option<(u64, String)> = None;
        let num_feats = noto_vision::fusion::numeric(&feats);
        for &(axis, x) in &num_feats {
            if let Some((z, n, key, _)) = z_detail(self, &fscope, axis, x) {
                if z.abs() > zmax.abs() {
                    zmax = z;
                    zmax_det = Some((n, key));
                }
                if z.abs() > 2.5 {
                    fl.push(format!("{axis}_z"));
                }
            }
        }
        {
            let sds = self.active_sds_mut();
            for &(axis, x) in &num_feats {
                sds.record_baseline(&fscope, axis, x);
            }
        }
        sds_lane.insert("fusion".into(), zmax as f32);
        if let Some((n, key)) = zmax_det {
            sds_det.insert("fusion".into(), (zmax as f32, n, key));
        }
        if let Some(b) = &kf.balance {
            let sf = ((noto_core::balance::MARGIN_KEEP - b.margin) / 0.1).max(0.0);
            sds_lane.insert("balance".into(), sf);
            sds_det.insert("balance".into(), (sf, 1, format!("margin_shortfall:{}", b.source)));
        }

        let (w, h) = (kf.intr.width as f32, kf.intr.height as f32);
        let view = MlView {
            depth_mode: dmode.clone(),
            locate_mode: lmode.clone(),
            track_mode: tmode.clone(),
            depth_metrics: frame.depth_metrics,
            points: frame.points.iter().filter(|p| wanted.contains(&p.label) || p.score > 0.5).map(|p| PointView { label: p.label.clone(), x: p.px / w, y: p.py / h, score: p.score, margin: p.margin, presence: p.presence }).collect(),
            tracks: frame
                .tracks
                .iter()
                .map(|t| TrackView { id: t.id, label: t.label.clone(), bbox: [t.bbox[0] / w, t.bbox[1] / h, t.bbox[2] / w, t.bbox[3] / h], area_frac: t.area_frac, score: t.score, iou_prev: t.iou_prev, propagated: t.propagated })
                .collect(),
            target: frame.target.as_ref().map(|t| TargetView {
                track_id: t.track_id,
                label: t.label.clone(),
                rel_bearing_deg: t.rel_bearing_deg,
                depth_range_m: t.depth_range_m,
                depth_scale: scale,
                range_true_m: frame.range_true_m,
                ..Default::default()
            }),
            locate_err_px: frame.locate_err_px,
            track_iou_truth: frame.track_iou_truth,
            features: feats,
            flags: fl,
            pyramid: frame.pyramid,
            stage_ms: frame.stage_ms,
        };
        let ov = overlay(&frame, kf.intr.width, kf.intr.height, wanted);
        kf.ml_webp = encode_webp(&ov.data, ov.width, ov.height, 70.0).ok().map(|e| e.bytes);
        kf.ml = Some(view);
        kf.sds.extend(sds_lane);
        kf.sds_detail.extend(sds_det);
        Ok(MlOut { frame, sighting })
    }

    pub(crate) fn record_keyframe_row(&mut self, pass: &str, kf: &Keyframe, target: &str) {
        let ml = kf.ml.as_ref();
        let t = ml.and_then(|m| m.target.as_ref());
        let d = kf.decision.as_ref();
        let (generation, ledger_digest, sds_mode) = self.row_meta();
        let row = KeyframeRow {
            run_id: self.run_id.clone(),
            pass: pass.into(),
            index: kf.index,
            tick: kf.tick,
            time_s: kf.time_s,
            target: target.into(),
            depth_mode: ml.map(|m| m.depth_mode.clone()).unwrap_or_default(),
            depth_absrel: ml.and_then(|m| m.depth_metrics.map(|x| x.absrel as f64)),
            depth_delta1: ml.and_then(|m| m.depth_metrics.map(|x| x.delta1 as f64)),
            locate_mode: ml.map(|m| m.locate_mode.clone()).unwrap_or_default(),
            locate_score: ml.and_then(|m| m.points.iter().find(|p| p.label == target).map(|p| p.score as f64)),
            locate_err_px: ml.and_then(|m| m.locate_err_px.map(|x| x as f64)),
            track_mode: ml.map(|m| m.track_mode.clone()).unwrap_or_default(),
            track_id: t.map(|t| t.track_id),
            track_iou_prev: ml.and_then(|m| t.and_then(|t| m.tracks.iter().find(|x| x.id == t.track_id)).and_then(|x| x.iou_prev.map(|v| v as f64))),
            track_iou_truth: ml.and_then(|m| m.track_iou_truth.map(|x| x as f64)),
            tri_range_m: t.and_then(|t| t.tri_range_m.map(|x| x as f64)),
            tri_parallax_deg: t.and_then(|t| t.tri_parallax_deg.map(|x| x as f64)),
            depth_scale: t.and_then(|t| t.depth_scale.map(|x| x as f64)),
            range_used_m: t.and_then(|t| t.range_used_m.map(|x| x as f64)),
            range_true_m: t.and_then(|t| t.range_true_m.map(|x| x as f64)),
            decision: d.map(|d| d.decision.clone()).unwrap_or_default(),
            ts_method: d.map(|d| d.forecast_method.clone()).unwrap_or_default(),
            ts_residual: d.and_then(|d| d.residual_deg.map(|x| x as f64)),
            fusion_flags: ml.map(|m| m.flags.join(",")).unwrap_or_default(),
            balance_margin: kf.balance.as_ref().map(|b| b.margin as f64),
            effort_max: kf.balance.as_ref().map(|b| b.worst_ratio as f64),
            generation,
            ledger_digest,
            sds_mode,
        };
        if let Some(dv) = d {
            if dv.kind == "hold" {
                self.holds.push(crate::progress::HoldItem {
                    tick: kf.tick,
                    worker: 1,
                    domain: "decide",
                    code: dv.decision.clone(),
                    verdict: "held",
                    subject: target.to_string(),
                    measured: dv.fix.map(|f| f.0 as f64),
                    limit: None,
                    band: dv.band.map(|b| [b[0] as f64, b[1] as f64]),
                    unit: "m",
                    detail_ko: dv.note.clone(),
                });
            }
        }
        self.db.record_keyframe(row);
    }

    fn record_effects(&mut self, post: &noto_core::Posture, state: &str) {
        use noto_core::effects::{evaluate, live_scenario, Rule};
        let wear_now = self.world.wear.clone();
        let now_s = self.world.time_s;
        let mut wear_rates: Vec<(String, f64)> = Vec::new();
        for (joint, d) in &wear_now {
            if let Some(&(prev_d, prev_t)) = self.prev_wear.get(joint) {
                let dt = now_s - prev_t;
                if dt > 1e-3 && *d >= prev_d {
                    wear_rates.push((joint.clone(), ((*d - prev_d) as f64) / dt));
                }
            }
            self.prev_wear.insert(joint.clone(), (*d, now_s));
        }
        if !wear_now.is_empty() {
            let scope = Scope::new(Track::Effort, "wear", state);
            let sds = self.active_sds_mut();
            for (joint, d) in wear_now {
                sds.record_baseline(&scope, &format!("{joint}:life_used"), d as f64);
            }
            for (joint, rate) in wear_rates {
                sds.record_baseline(&scope, &format!("{joint}:life_rate"), rate);
                sds.record_baseline(&scope, &format!("{joint}:life_rate_per_h"), rate * 3600.0);
            }
        }
        let Some(s) = live_scenario(&self.world.body, post) else { return };
        let fx = evaluate(&self.world.body, &s);
        if fx.skipped.is_some() {
            return;
        }
        let at = format!("{}|{state}", post.ground.id());
        let sds = self.active_sds_mut();
        for c in &fx.cells {
            let Some(v) = c.value else { continue };
            let scope = Scope::new(Track::Balance, &format!("fx:{}", c.effect.id()), &at);
            match c.rule {
                Rule::Like { reference, tolerance } | Rule::AtLeast { reference, tolerance } => sds.record_reference(&scope, c.axis, v as f64, reference as f64, tolerance as f64),
                Rule::AtMost { limit } => {
                    sds.record_baseline(&scope, c.axis, v as f64);
                    sds.record_agreement(&scope, &format!("{}_ok", c.axis), v <= limit);
                }
                Rule::Raw => sds.record_baseline(&scope, c.axis, v as f64),
            }
        }
    }

    pub(crate) fn record_balance_events(&mut self) {
        let events = self.world.take_balance_events();
        let preset = self.world.cfg.body.id();
        let form = format!("{}:{:?}:f{}", preset, self.world.cfg.noto.base, u8::from(self.world.cfg.noto.fingers)).to_lowercase();
        let body_kg = self.world.body.mass_kg();
        let stg = self.current_step_label().unwrap_or_else(|| "idle".into());
        let holdout = self.pass == "holdout";
        let (generation, ledger_digest, sds_mode) = self.row_meta();

        if !events.is_empty() {
            let nn_lib = self.nn_live.library_for(&form, &self.world.body).to_vec();
            for ev in &events {
                let event_id = format!("{}#{}", self.run_id, ev.tick);
                let gnd = ev.posture.ground.id();
                let slp = slope_bucket(ev.posture.slope_deg);

                let task_li = ev.report.worst().map_or(0.0, |w| w.peak_ratio);
                let task_lmax = ev.report.joints.iter().map(|j| j.effort).fold(0.0f32, f32::max).max(task_li);
                let task_key = format!("{}/{}", ev.posture.source.id(), ev.state.replace('|', "/"));
                self.nn_live.record_shift_task(&task_key, ev.time_s as f32, task_li, task_lmax);

                if let Some(d) = &ev.dynamics {
                    self.ts.mark_phase_break(ev.tick);
                    self.record_effects(&ev.posture, &ev.state);
                    let matched = noto_core::nn::matching_phases_classified(&nn_lib, &ev.posture, body_kg);
                    if matched.is_empty() {
                        self.nn_live.clear_now();
                    } else {
                        let mut scored_list = Vec::with_capacity(matched.len());
                        let mut now_items = Vec::with_capacity(matched.len());
                        for (spec, phase, _dist, mc) in matched {
                            let cells = noto_core::nn::cells(&self.world.body, &ev.posture, &ev.report, Some(d));
                            let pr = noto_core::nn::score_phase(phase, &spec.seg_w, cells, &ev.report, Some(d));
                            self.nn_live.attach_scores(ev.tick, spec.id, spec.name_ko, &pr);
                            self.nn_live.record_series_sample(spec.id, &phase.id, ev.time_s as f32, pr.clone(), mc, true);
                            let bd = crate::progress::breakdown(&pr, &spec.seg_w);
                            for hi in crate::progress::hold_items_from_nn(ev.tick, 1, spec.id, &phase.id, &pr.gate_hits) {
                                self.holds.push(hi);
                            }
                            now_items.push(crate::nnlive::ActiveNow {
                                scenario: spec.id.to_string(),
                                phase: phase.id.clone(),
                                ladder: phase.ladder.map(Into::into),
                                since_tick: ev.tick,
                                state: ev.state.clone(),
                                score: (pr.score * 10.0).round() / 10.0,
                                ungated: (bd.ungated * 10.0).round() / 10.0,
                                item: pr.item,
                                gated: !pr.gates.is_empty(),
                                excused: !pr.beyond_human.is_empty(),
                                top_loss: bd.top_loss,
                            });
                            let subj = format!("{}:{}", spec.id.to_ascii_lowercase(), phase.id);
                            self.ts.record(1, "nn.score", &subj, ev.tick, pr.score);
                            self.ts.record(1, "nn.min_margin", &subj, ev.tick, pr.min_margin);
                            self.ts.record(1, "nn.peak_ratio", &subj, ev.tick, pr.peak_ratio);
                            self.ts.record(1, "nn.sustained_effort", &subj, ev.tick, pr.sustained_effort);
                            let (instance_id, span_id) = self.nn_live.active_span_info(spec.id, &phase.id);
                            self.db.record_nn_phase(NnPhaseRow {
                                event_id: event_id.clone(),
                                run_id: self.run_id.clone(),
                                pass: self.pass.clone(),
                                tick: ev.tick,
                                time_s: ev.time_s,
                                instance_id: instance_id.unwrap_or(0),
                                span_id: span_id.unwrap_or(0),
                                scenario: spec.id.to_string(),
                                phase: phase.id.clone(),
                                holdout,
                                form: form.clone(),
                                gnd: gnd.to_string(),
                                slope: slp.to_string(),
                                stg: stg.clone(),
                                wk: "main".into(),
                                score: pr.score as f64,
                                item: pr.item,
                                gate_codes: pr.gate_codes.join(","),
                                beyond_human: !pr.beyond_human.is_empty(),
                                static_margin: pr.static_margin as f64,
                                min_margin: pr.min_margin as f64,
                                peak_ratio: pr.peak_ratio as f64,
                                sustained_effort: pr.sustained_effort as f64,
                                worst_joint: pr.worst_joint.clone(),
                                generation: generation.clone(),
                                ledger_digest: ledger_digest.clone(),
                                sds_mode: sds_mode.clone(),
                            });
                            scored_list.push((spec.id, phase.id.clone(), pr, mc));
                        }
                        self.nn_live.set_now(now_items);
                        // Precompute phase-residual baselines before batch writes (Phase 2 – Part 2 · D5).
                        let phase_means: Vec<Option<f64>> = scored_list
                            .iter()
                            .map(|(scn_id, ph_id, _, _)| {
                                let legacy = Scope::new(Track::Balance, &format!("nn:{scn_id}"), ph_id);
                                self.sds.exact_baseline(&legacy, "score").map(|(m, _, _)| m)
                            })
                            .collect();
                        // Deduplicate shared upstream keys across multi-match events (e.g. S1/slope + S6/slope).
                        let sds = self.active_sds_mut();
                        sds.record_batch(&event_id, |b| {
                            for (idx, (scn_id, ph_id, pr, mc)) in scored_list.iter().enumerate() {
                                let ph_scope_tag = match mc {
                                    noto_core::nn::MatchClass::Dev => ph_id.clone(),
                                    noto_core::nn::MatchClass::Between => format!("{ph_id}@between"),
                                    noto_core::nn::MatchClass::HoldoutLike => format!("{ph_id}@holdout_like"),
                                };
                                let legacy = Scope::new(Track::Balance, &format!("nn:{scn_id}"), &ph_scope_tag);
                                b.record_baseline(&legacy, "score", pr.score as f64);
                                b.record_baseline(&legacy, "static_margin", pr.static_margin as f64);
                                b.record_baseline(&legacy, "min_margin", pr.min_margin as f64);
                                b.record_baseline(&legacy, "margin_dip", (pr.static_margin - pr.min_margin) as f64);
                                b.record_baseline(&legacy, "peak_ratio", pr.peak_ratio as f64);
                                b.record_baseline(&legacy, "sustained_effort", pr.sustained_effort as f64);
                                b.record_agreement(&legacy, "item_ok", pr.item >= 1);
                                b.record_agreement(&legacy, "beyond_human", !pr.beyond_human.is_empty());
                                b.record_extrema(&legacy, "min_margin", pr.min_margin as f64);
                                b.record_extrema(&legacy, "peak_ratio", pr.peak_ratio as f64);
                                if let Some(mu_ph) = phase_means[idx] {
                                    let prim = Scope::new(Track::Balance, &format!("nn:{scn_id}"), "");
                                    b.record_baseline(&prim, "score:phase_resid", pr.score as f64 - mu_ph);
                                }

                                let mut facets: Vec<(Facet, String)> = vec![
                                    (Facet::Ph, ph_scope_tag),
                                    (Facet::Form, form.clone()),
                                    (Facet::Gnd, gnd.to_string()),
                                    (Facet::Slope, slp.to_string()),
                                    (Facet::Wk, "main".into()),
                                ];
                                if holdout || *mc == noto_core::nn::MatchClass::HoldoutLike {
                                    facets.push((Facet::Set, "holdout".into()));
                                }
                                let v2 = ScopeKey::new(Track::Balance, &format!("nn:{}", scn_id.to_ascii_lowercase()), facets);
                                b.record_v2_baseline(&v2, "score", pr.score as f64);
                                b.record_v2_baseline(&v2, "static_margin", pr.static_margin as f64);
                                b.record_v2_baseline(&v2, "min_margin", pr.min_margin as f64);
                                b.record_v2_baseline(&v2, "margin_dip", (pr.static_margin - pr.min_margin) as f64);
                                b.record_v2_baseline(&v2, "peak_ratio", pr.peak_ratio as f64);
                                b.record_v2_baseline(&v2, "sustained_effort", pr.sustained_effort as f64);
                                b.record_v2_agreement(&v2, "item_ok", pr.item >= 1);
                                b.record_v2_agreement(&v2, "beyond_human", !pr.beyond_human.is_empty());
                                b.record_v2_extrema(&v2, "min_margin", pr.min_margin as f64);
                                b.record_v2_extrema(&v2, "peak_ratio", pr.peak_ratio as f64);

                                // Gate codes via record_gates on v2 chain.
                                let gate_items: Vec<(String, u8, bool, bool)> = ["infeasible", "margin_neg", "peak_over", "unreached", "head_tilt"]
                                    .iter()
                                    .map(|&code| {
                                        let hit = pr.gate_codes.contains(&code);
                                        let sole = hit && pr.gate_codes.len() == 1;
                                        (code.to_string(), if hit { 100 } else { 0 }, hit, sole)
                                    })
                                    .collect();
                                b.record_v2_gates(&v2, &gate_items);

                                if let Some(wj) = &pr.worst_joint {
                                    b.record_v2_transition(&v2, "worst", wj);
                                }
                                for c in &pr.cells {
                                    let axis = format!("{}>{}:{}", c.source.id(), c.target.id(), c.axis);
                                    if c.z.is_some() {
                                        b.record_v2_reference_detail(&v2, &axis, c.value as f64, c.reference.unwrap_or(0.0) as f64, c.tolerance.unwrap_or(1.0) as f64);
                                    } else {
                                        b.record_v2_exact_baseline(&v2, &axis, c.value as f64);
                                    }
                                }
                            }
                        });
                        // Legacy cell reference + raw recording (Phase 6 – Part 2 · D6 C3), keeping MatchClass scopes separate.
                        let sds = self.active_sds_mut();
                        for (scn_id, ph_id, pr, mc) in &scored_list {
                            let ph_tag = match mc {
                                noto_core::nn::MatchClass::Dev => ph_id.clone(),
                                noto_core::nn::MatchClass::Between => format!("{ph_id}@between"),
                                noto_core::nn::MatchClass::HoldoutLike => format!("{ph_id}@holdout_like"),
                            };
                            let legacy = Scope::new(Track::Balance, &format!("nn:{scn_id}"), &ph_tag);
                            for c in &pr.cells {
                                let axis = format!("{}>{}:{}", c.source.id(), c.target.id(), c.axis);
                                if c.z.is_some() {
                                    sds.record_reference(&legacy, &axis, c.value as f64, c.reference.unwrap_or(0.0) as f64, c.tolerance.unwrap_or(1.0) as f64);
                                } else {
                                    sds.record_baseline(&legacy, &axis, c.value as f64);
                                }
                            }
                        }
                    }
                } else {
                    // 50-tick resampled dwell event: extend PhaseSeries dwell without double-counting transitions,
                    // score statically into TsHub `nn.*`, and record SDS on `dwell:` prefixed axes only (Phase 3-P1 D3 & 6-P1 D3).
                    let matched = noto_core::nn::matching_phases_classified(&nn_lib, &ev.posture, body_kg);
                    if !matched.is_empty() {
                        for (spec, phase, _, mc) in &matched {
                            let cells = noto_core::nn::cells(&self.world.body, &ev.posture, &ev.report, None);
                            let pr = noto_core::nn::score_phase(phase, &spec.seg_w, cells, &ev.report, None);
                            self.nn_live.record_series_sample(spec.id, &phase.id, ev.time_s as f32, pr.clone(), *mc, false);
                            let subj = format!("{}:{}", spec.id.to_ascii_lowercase(), phase.id);
                            self.ts.record(1, "nn.score", &subj, ev.tick, pr.score);
                            self.ts.record(1, "nn.min_margin", &subj, ev.tick, pr.min_margin);
                            self.ts.record(1, "nn.peak_ratio", &subj, ev.tick, pr.peak_ratio);
                            self.ts.record(1, "nn.sustained_effort", &subj, ev.tick, pr.sustained_effort);
                        }
                        let static_margin = ev.report.margin as f64;
                        let peak_ratio = ev.report.worst().map_or(0.0, |w| w.peak_ratio) as f64;
                        let sds = self.active_sds_mut();
                        sds.record_batch(&format!("{event_id}:dwell"), |b| {
                            for (spec, phase, _, _) in &matched {
                                let v2 = ScopeKey::new(
                                    Track::Balance,
                                    &format!("nn:{}", spec.id.to_ascii_lowercase()),
                                    vec![(Facet::Ph, phase.id.clone()), (Facet::Form, form.clone())],
                                );
                                b.record_v2_exact_baseline(&v2, "dwell:static_margin", static_margin);
                                b.record_v2_exact_baseline(&v2, "dwell:peak_ratio", peak_ratio);
                            }
                        });
                    }
                }
            }

            for ev in events {
                let r = &ev.report;
                if r.source == Source::Quiet {
                    continue;
                }
                let src = r.source.id();
                let event_id = format!("{}#{}", self.run_id, ev.tick);
                let ff_scope = Scope::new(Track::Balance, &format!("{src}>foot"), &ev.state);
                let ctx = r.ref_ctx_for(body_kg, ev.posture.single_support);
                {
                    let sds = self.active_sds_mut();
                    if r.ff_predicted_m.abs() > 1e-4 && r.shift_uncomp_m.abs() > 1e-4 {
                        sds.record_calibration(&ff_scope, "ff", r.shift_uncomp_m as f64, r.ff_predicted_m as f64);
                    }
                    for c in &r.couplings {
                        let scope = Scope::new(Track::Balance, &format!("{src}>{}", c.target.id()), &ev.state);
                        let rf = reference(r.source, c.target, &c.axis, &ctx);
                        match &rf {
                            Some(rf) => sds.record_reference(&scope, &c.axis, c.value as f64, rf.value as f64, rf.tolerance as f64),
                            None => sds.record_baseline(&scope, &c.axis, c.value as f64),
                        }
                    }
                    for j in &r.joints {
                        let scope = Scope::new(Track::Effort, j.segment.id(), &format!("{src}|{}", ev.state));
                        sds.record_baseline(&scope, &format!("{}:peak_ratio", j.joint), j.peak_ratio as f64);
                        sds.record_baseline(&scope, &format!("{}:effort", j.joint), j.effort as f64);
                    }
                    if let Some(d) = &ev.dynamics {
                        let scope = Scope::new(Track::Effort, "dynamics", &format!("{src}|{}", ev.state));
                        sds.record_baseline(&scope, "min_margin", d.min_margin as f64);
                        sds.record_baseline(&scope, "margin_dip", (d.static_margin - d.min_margin) as f64);
                        sds.record_baseline(&scope, "peak_ratio", d.peak_ratio as f64);
                        sds.record_agreement(&scope, "feasible", d.feasible);
                        sds.record_agreement(&scope, "reached", d.reached);
                        for j in d.joints.iter().filter(|j| j.kind == "sim") {
                            sds.record_baseline(&scope, &format!("{}:peak_ratio", j.name), j.peak_ratio as f64);
                            sds.record_baseline(&scope, &format!("{}:lag_deg", j.name), j.max_lag_deg as f64);
                            if j.saturated_s > 0.0 || j.stalled_s > 0.0 {
                                sds.record_baseline(&scope, &format!("{}:sat_s", j.name), j.saturated_s as f64);
                            }
                        }
                    }
                }
                for c in &r.couplings {
                    let rf = reference(r.source, c.target, &c.axis, &ctx);
                    self.db.record_balance(BalanceRow {
                        event_id: event_id.clone(),
                        run_id: self.run_id.clone(),
                        pass: self.pass.clone(),
                        tick: ev.tick,
                        source: src.into(),
                        source_seg: r.source.segment().id().into(),
                        target: c.target.id().into(),
                        state: format!("{}|{}", ev.state, c.axis),
                        preset: preset.clone(),
                        comp: c.value as f64,
                        comp_unit: c.unit.clone(),
                        reference: rf.as_ref().map(|x| x.value as f64),
                        tolerance: rf.as_ref().map(|x| x.tolerance as f64),
                        torque_nm: c.torque_nm.unwrap_or(0.0) as f64,
                        effort: c.effort.unwrap_or(0.0) as f64,
                        effort_noff: c.effort_noff.unwrap_or(0.0) as f64,
                        margin: r.margin as f64,
                        margin_noff: r.margin_noff as f64,
                        ff_gain: r.ff_gain as f64,
                        generation: generation.clone(),
                        ledger_digest: ledger_digest.clone(),
                        sds_mode: sds_mode.clone(),
                    });
                }
            }
        }

        // Persist closed N:N spans (event counts `ev:<kind>:n` + `dwell_s:<ph>`) and closed instances (Phase 2 – Part 1 · D2/D3).
        let closed_spans = self.nn_live.drain_closed_spans();
        if !closed_spans.is_empty() {
            let sds = self.active_sds_mut();
            for sp in closed_spans {
                let mut facets = vec![(Facet::Ph, sp.phase.clone())];
                if !sp.run.form.is_empty() {
                    facets.push((Facet::Form, sp.run.form.clone()));
                }
                let v2 = ScopeKey::new(Track::Balance, &format!("nn:{}", sp.scenario.to_ascii_lowercase()), facets);
                let mut by_kind: BTreeMap<&str, u64> = BTreeMap::new();
                for m in &sp.marks {
                    *by_kind.entry(m.kind).or_insert(0) += 1;
                }
                sds.record_batch(&format!("span#{}", sp.span_id), |b| {
                    b.record_v2_exact_baseline(&v2, &format!("dwell_s:{}", sp.phase), sp.dwell_s);
                    for (k, n) in &by_kind {
                        b.record_v2_exact_baseline(&v2, &format!("ev:{k}:n"), *n as f64);
                    }
                });
            }
        }
        let closed_insts = self.nn_live.drain_closed_instances();
        if !closed_insts.is_empty() {
            let sds = self.active_sds_mut();
            for (rk, inst) in closed_insts {
                let mut facets = Vec::new();
                if !rk.form.is_empty() {
                    facets.push((Facet::Form, rk.form.clone()));
                }
                let v2 = ScopeKey::new(Track::Balance, &format!("nn:{}", inst.scenario.to_ascii_lowercase()), facets);
                sds.record_batch(&format!("inst#{}", inst.id), |b| {
                    if let Some(sc) = inst.score {
                        b.record_v2_exact_baseline(&v2, "scenario_score", sc as f64);
                    }
                    for pair in inst.phases_seen.windows(2) {
                        b.record_v2_transition(&v2, &pair[0], &pair[1]);
                    }
                });
                if let Some(sc) = inst.score {
                    let sc_scope = Scope::new(Track::Balance, &format!("nn:{}", inst.scenario), "");
                    sds.record_agreement(&sc_scope, "passed", sc >= 60.0 && inst.complete);
                }
            }
        }

        // Path F: Balance feed-forward gains (Inherit::Never across generations, cold default 1.0 when Off or Shifted-thin).
        let tick = self.world.tick;
        for s in Source::ALL {
            let scope = Scope::new(Track::Balance, &format!("{}>foot", s.id()), "");
            match self.feedback.estimate_factor(FeedbackPath::F, &self.sds, &scope, "ff", Window::Auto, tick) {
                Some(est)
                    if est.key.starts_with(&format!("balance|{}>foot|", s.id()))
                        && !(matches!(est.state, DriftState::Shifted { .. }) && (est.window != Window::Span || est.n_eff < 4.0)) =>
                {
                    self.world.ff_gains.insert(s, est.value as f32);
                }
                None if self.feedback.effective_mode(FeedbackPath::F) == crate::feedback::Mode::Off => {
                    self.world.ff_gains.remove(&s);
                }
                Some(est) if matches!(est.state, DriftState::Shifted { .. }) => {
                    self.world.ff_gains.remove(&s);
                }
                _ => {}
            }
        }
    }

    /// Evaluate sub-workers statically (`static_only: true`) using [`world_sim::crew::Worker::posture_now`]
    /// and record into separate `wk:cw<id>` SDS scopes so worker 1's dynamic averages are never polluted
    /// (Phase 6 – Part 2 · D3).
    pub fn record_crew_static_nn(&mut self) -> Vec<serde_json::Value> {
        let ff = noto_core::balance::FeedForward::default();
        let tick = self.world.tick;
        let mut results = Vec::new();
        let snapshots: Vec<(u32, String, noto_core::balance::Body, noto_core::balance::Posture, String)> = self
            .world
            .crew
            .iter()
            .map(|cw| {
                let (pitch, _) = self.world.chassis_attitude(cw.x, cw.z, cw.yaw);
                let gnd = self
                    .world
                    .terrain
                    .cell_of(cw.x, cw.z)
                    .map_or(noto_core::surface::Surface::Firm, |(cx, cz)| self.world.terrain.ground_at(cx, cz).surface());
                let (post, state) = cw.posture_now(pitch.to_degrees(), gnd);
                (cw.id, cw.form.id().to_string(), cw.body.clone(), post, state)
            })
            .collect();
        for (wid, wform, body, post, state) in snapshots {
            let lib = noto_core::nn::all_scenarios(&body);
            let rep = body.solve(&post, &ff, 1e3);
            let matched = noto_core::nn::matching_phases_classified(&lib, &post, body.mass_kg());
            let mut phases_out = Vec::new();
            for (spec, ph, _, _) in matched {
                let cs = noto_core::nn::cells(&body, &post, &rep, None);
                let pr = noto_core::nn::score_phase(ph, &spec.seg_w, cs, &rep, None);
                let v2 = ScopeKey::new(
                    Track::Balance,
                    &format!("nn:{}", spec.id.to_ascii_lowercase()),
                    vec![
                        (Facet::Ph, format!("{}@static", ph.id)),
                        (Facet::Form, wform.clone()),
                        (Facet::Wk, format!("cw{wid}")),
                    ],
                );
                let sds = self.active_sds_mut();
                sds.record_batch(&format!("cw{wid}#{tick}#{}/{}", spec.id, ph.id), |b| {
                    b.record_v2_exact_baseline(&v2, "static_score", pr.score as f64);
                    b.record_v2_exact_baseline(&v2, "static_margin", pr.static_margin as f64);
                });
                phases_out.push(json!({
                    "scenario": spec.id,
                    "phase": ph.id,
                    "score": (pr.score as f64 * 10.0).round() / 10.0,
                    "static_margin": (pr.static_margin as f64 * 1000.0).round() / 1000.0,
                    "cells": pr.cells.len(),
                }));
            }
            results.push(json!({
                "worker": wid,
                "form": wform,
                "state": state,
                "static_only": true,
                "phases": phases_out,
            }));
        }
        results
    }

    pub(crate) fn record_quest_run(&mut self, quest_id: &str, attempts: &[u32]) {
        let Some(q) = self.quests.get(quest_id).cloned() else { return };
        let kind = match &q.plan {
            noto_search::Plan::Fetch { .. } => "fetch",
            noto_search::Plan::Collect { .. } => "collect",
            noto_search::Plan::Goto { .. } => "goto",
            noto_search::Plan::Drop { .. } => "drop",
            noto_search::Plan::Stop => "stop",
        };
        let (target, destination) = match &q.plan {
            noto_search::Plan::Fetch { targets, destination } => (targets.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), Some(destination.place.clone())),
            noto_search::Plan::Collect { targets } => (targets.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), None),
            noto_search::Plan::Goto { place } => (place.place.clone(), Some(place.place.clone())),
            noto_search::Plan::Drop { items, place } => (items.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), place.as_ref().map(|p| p.place.clone())),
            noto_search::Plan::Stop => (String::new(), None),
        };
        let ok = q.state == noto_quest::State::Done;
        let steps: Vec<StepRecord> = q
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state != noto_quest::State::Pending)
            .map(|(i, s)| StepRecord {
                step: format!("{:?}", s.kind).to_lowercase(),
                attempts: attempts.get(i).copied().unwrap_or(1).max(1),
                ok: s.state == noto_quest::State::Done,
                fail_code: (s.state == noto_quest::State::Failed).then(|| q.failure.as_ref().map(|f| f.code.clone()).unwrap_or_default()),
                time_s: 0.0,
            })
            .collect();
        let r = q.result.clone();
        let (generation, ledger_digest, sds_mode) = self.row_meta();
        let run = QuestRun {
            run_id: self.run_id.clone(),
            session: self.session_id.clone(),
            pass: self.pass.clone(),
            quest_id: q.id.clone(),
            kind: kind.into(),
            target: target.clone(),
            destination,
            preset: self.world.cfg.body.id(),
            seed: self.cfg.seed,
            ok,
            failure: q.failure.as_ref().map(|f| f.code.clone()),
            ticks: r.as_ref().map_or(0, |r| r.ticks),
            time_s: r.as_ref().map_or(0.0, |r| r.time_s),
            energy_j: r.as_ref().map_or(0.0, |r| r.energy_j),
            steps: steps.clone(),
            created_ms: noto_sds::store::now_ms(),
            generation,
            ledger_digest,
            sds_mode,
        };
        let pass_copy = self.pass.clone();
        {
            let sds = self.active_sds_mut();
            for s in &steps {
                let scope = Scope::new(Track::Quest, &format!("{pass_copy}|{kind}"), &s.step);
                for i in 0..s.attempts.max(1) {
                    sds.record_agreement(&scope, "success", s.ok && i + 1 == s.attempts.max(1));
                }
            }
            sds.record_agreement(&Scope::new(Track::Quest, &format!("{pass_copy}|{kind}"), "quest"), "success", ok);
        }
        let stat = {
            self.db.record_run(run);
            self.db.stat(&self.pass, kind, &target, &self.world.cfg.body.id(), noto_sds::WHOLE_QUEST).cloned()
        };
        if let Some(st) = stat {
            let msg = format!("[{}] {kind} {target}: 성공 {}/{} ({:.0}%) · 프리셋 {}", self.pass, st.successes, st.attempts, st.rate * 100.0, st.preset);
            self.log("quest", Level::Info, msg, Some(json!({"pass": self.pass, "kind": kind, "target": target, "attempts": st.attempts, "successes": st.successes})));
        }
        for l in self.db.flush() {
            self.audit.info("sds", format!("[DB] {l}"));
        }
    }
}
