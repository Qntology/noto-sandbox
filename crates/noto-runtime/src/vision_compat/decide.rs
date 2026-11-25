use crate::ts::{forecast, Predictor};
use lv_spatial::pos2::{Pos2Filter, Pos2Model, Pos2Obs, CHI2_2DOF_99, POS2_RING_CAP};

fn cov3(c: [f64; 4]) -> [f32; 3] {
    [c[0] as f32, c[1] as f32, c[3] as f32]
}
use serde::{Deserialize, Serialize};

use super::camera::{wrap_angle, Pose};
use super::triangulate::{plan_step, triangulate, Bearing, Fix, TriCfg, TriStep};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecideCfg {
    pub reach_m: f32,
    pub hop_max_m: f32,
    pub search_turn_deg: f32,
    pub max_sigma_frac: f32,
    pub max_rays: usize,
    pub band_widen_sigma: f32,
    pub default_range_m: f32,
    pub close_range_m: f32,
    pub tri: TriCfg,
}

impl Default for DecideCfg {
    fn default() -> Self {
        DecideCfg {
            reach_m: 0.8,
            hop_max_m: 3.0,
            search_turn_deg: 50.0,
            max_sigma_frac: 0.15,
            max_rays: 4,
            band_widen_sigma: 2.0,
            default_range_m: 6.0,
            close_range_m: 1.6,
            tri: TriCfg::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sighting {
    pub keyframe: u32,
    pub pose: Pose,
    pub heading: f32,
    pub sigma: f32,
    /// The target's angular width (rad). Where a wide thing's visible centre sits moves with the view and with the
    /// mask by a share of that width, which `sigma` (the tracker's centroid noise) does not count.
    #[serde(default)]
    pub extent: f32,
    pub depth_range_m: Option<f32>,
    pub track_id: u32,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Decision {
    Search { turn_deg: f32 },
    LookDown { pitch_deg: f32, turn_deg: f32 },
    Triangulate(TriStep),
    Approach { to: (f32, f32), hop_m: f32, face: (f32, f32), look_pitch_deg: f32 },
    Arrive { at: (f32, f32), look_pitch_deg: f32 },
    Hold { reason: String },
    Explore { turned_deg: f32 },
}

impl Decision {
    pub fn label(&self) -> String {
        match self {
            Decision::Search { turn_deg } => format!("search(turn {turn_deg:+.0}°)"),
            Decision::LookDown { pitch_deg, turn_deg } => format!("look_down(pitch {pitch_deg:+.0}°, turn {turn_deg:+.0}°)"),
            Decision::Triangulate(t) => format!("tri_step({}, {:.2} m)", t.kind, t.step_m),
            Decision::Approach { hop_m, look_pitch_deg, .. } => format!("move({hop_m:.1} m, pitch {look_pitch_deg:+.0}°)"),
            Decision::Arrive { look_pitch_deg, .. } => format!("arrive(pitch {look_pitch_deg:+.0}°)"),
            Decision::Hold { reason } => format!("hold({reason})"),
            Decision::Explore { turned_deg } => format!("explore(after {turned_deg:.0}°)"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub bearing_rel_deg: f32,
    pub range_m: f32,
    pub pitch_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub x: f32,
    pub z: f32,
    pub range_m: f32,
    pub sigma_m: f32,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionReport {
    pub decision: Decision,
    pub estimate: Option<Estimate>,
    pub fix: Option<Fix>,
    pub fix_error: Option<String>,
    pub depth_scale: Option<f32>,
    pub residual_deg: Option<f32>,
    pub prediction: Option<Prediction>,
    pub forecast_method: String,
    pub band: Option<[f32; 4]>,
    #[serde(default)]
    pub cov: Option<[f32; 3]>,
    #[serde(default)]
    pub pos_model: Option<String>,
    #[serde(default)]
    pub realized_pos_err: Option<(f32, f32)>,
    pub note: String,
}

pub struct Decider {
    pub cfg: DecideCfg,
    rays: Vec<Sighting>,
    est_x: Vec<f64>,
    est_z: Vec<f64>,
    pos2: Pos2Filter,
    last_fix: Option<Fix>,
    prediction: Option<Prediction>,
    side: f32,
    lost: u32,
    turned_deg: f32,
}

impl Decider {
    pub fn new(cfg: DecideCfg) -> Decider {
        Decider {
            cfg,
            rays: Vec::new(),
            est_x: Vec::new(),
            est_z: Vec::new(),
            pos2: Pos2Filter::default(),
            last_fix: None,
            prediction: None,
            side: 1.0,
            lost: 0,
            turned_deg: 0.0,
        }
    }

    pub fn pos2(&self) -> &Pos2Filter {
        &self.pos2
    }

    pub fn side(&self) -> f32 {
        self.side
    }

    pub fn set_side(&mut self, side: f32) {
        self.side = if side < 0.0 { -1.0 } else { 1.0 };
    }

    pub fn reset(&mut self) {
        *self = Decider::new(self.cfg);
    }

    pub fn has_fix(&self) -> bool {
        self.last_fix.is_some()
    }

    pub fn track_id(&self) -> Option<u32> {
        self.rays.last().map(|r| r.track_id)
    }

    pub fn move_failed(&mut self) {
        self.side = -self.side;
        self.prediction = None;
    }

    fn forecast_position(
        &self,
        at_tick: Option<u64>,
        extra: &[&dyn Predictor],
    ) -> Option<(f32, f32, [f32; 4], [f32; 3], String)> {
        if let Some(est) = self.pos2.estimate(at_tick) {
            let sigmas = CHI2_2DOF_99.sqrt();
            let band = est.band_rect(sigmas, 0.0);
            let method = match est.model {
                Pos2Model::Static => "static_ivw".to_string(),
                Pos2Model::Dynamic => {
                    if self.est_x.len() >= 6 && !extra.is_empty() {
                        let fx = forecast(&self.est_x, 1, extra, &[]);
                        let fz = forecast(&self.est_z, 1, extra, &[]);
                        let b = [fx.p10[0] as f32, fx.p90[0] as f32, fz.p10[0] as f32, fz.p90[0] as f32];
                        return Some((fx.point[0] as f32, fz.point[0] as f32, b, cov3(est.cov), format!("dynamic:{}", fx.method)));
                    }
                    "dynamic_cv".to_string()
                }
            };
            return Some((est.mean[0] as f32, est.mean[1] as f32, band, cov3(est.cov), method));
        }
        if self.est_x.is_empty() {
            return None;
        }
        let fx = forecast(&self.est_x, 1, extra, &[]);
        let fz = forecast(&self.est_z, 1, extra, &[]);
        let sx = ((fx.p90[0] - fx.p10[0]) * 0.5 / 1.2816).max(0.02) as f32;
        let sz = ((fz.p90[0] - fz.p10[0]) * 0.5 / 1.2816).max(0.02) as f32;
        Some((
            fx.point[0] as f32,
            fz.point[0] as f32,
            [fx.p10[0] as f32, fx.p90[0] as f32, fz.p10[0] as f32, fz.p90[0] as f32],
            [sx * sx, 0.0, sz * sz],
            fx.method,
        ))
    }

    pub fn observe(&mut self, pose: &Pose, sighting: Option<Sighting>, holonomic: bool, extra: &[&dyn Predictor], depth_scale: Option<f32>) -> DecisionReport {
        let mut rep = DecisionReport {
            decision: Decision::Hold { reason: "start".into() },
            estimate: None,
            fix: None,
            fix_error: None,
            depth_scale: None,
            residual_deg: None,
            prediction: None,
            forecast_method: String::new(),
            band: None,
            cov: None,
            pos_model: None,
            realized_pos_err: None,
            note: String::new(),
        };
        let predicted = self.prediction.take();
        let calc_pitch = |eye_m: f32, dist_m: f32| -> f32 {
            let dz = (eye_m - 0.05).max(0.1);
            let direct = -dz.atan2(dist_m.max(0.2)).to_degrees();
            if dist_m <= 1.6 {
                direct.clamp(-60.0, -12.0)
            } else {
                -12.0
            }
        };

        let Some(s) = sighting else {
            self.lost += 1;
            if let Some((x, z, band, cov, method)) = self.forecast_position(None, extra) {
                let dist = (x - pose.x).hypot(z - pose.z);
                let rel = wrap_angle((z - pose.z).atan2(x - pose.x) - pose.yaw).to_degrees();
                rep.forecast_method = method.clone();
                rep.pos_model = Some(method);
                rep.band = Some(band);
                rep.cov = Some(cov);
                if self.lost == 1 && dist <= 1.8 {
                    let pitch_deg = calc_pitch(pose.eye_m, dist);
                    let turn_deg = if rel.abs() > 5.0 { rel } else { 0.0 };
                    rep.decision = Decision::LookDown { pitch_deg, turn_deg };
                    rep.note = format!("근거리({dist:.2} m) 대상 시야 이탈: 고개를 {pitch_deg:+.0}° 숙여 바닥 확인");
                    return rep;
                }
                if self.lost <= 2 && rel.abs() > 8.0 {
                    rep.decision = Decision::Search { turn_deg: rel };
                    rep.note = "추적을 놓쳐 예측 위치 쪽으로 회전".into();
                    return rep;
                }
            }
            if self.turned_deg + self.cfg.search_turn_deg * 0.5 >= 360.0 {
                rep.decision = Decision::Explore { turned_deg: self.turned_deg };
                rep.note = format!("{:.0}° 돌아도 대상 없음: 다른 시점으로 이동", self.turned_deg);
                self.turned_deg = 0.0;
                return rep;
            }
            self.turned_deg += self.cfg.search_turn_deg.abs();
            rep.decision = Decision::Search { turn_deg: self.cfg.search_turn_deg };
            rep.note = "대상 없음: 회전 탐색".into();
            return rep;
        };
        self.lost = 0;
        self.turned_deg = 0.0;
        if let Some(p) = predicted {
            rep.residual_deg = Some(wrap_angle(s.heading - pose.yaw).to_degrees() - p.bearing_rel_deg);
        }
        if self.rays.last().map_or(false, |r| r.track_id != s.track_id) {
            self.rays.clear();
        }
        self.rays.push(s);
        if self.rays.len() > self.cfg.max_rays {
            self.rays.remove(0);
        }
        let bearings: Vec<Bearing> = self.rays.iter().map(|r| Bearing { x: r.pose.x, z: r.pose.z, heading: r.heading, sigma: r.sigma }).collect();
        match triangulate(&bearings, self.cfg.tri.min_parallax_deg) {
            Ok(fix) => {
                if let Some(d) = s.depth_range_m.filter(|_| scale_usable(&fix, &s)) {
                    rep.depth_scale = Some(fix.range_m / d);
                }
                let obs_sigma = fix.sigma_m.max(0.02) as f64;
                if let Some(est) = self.pos2.estimate(Some(s.keyframe as u64)) {
                    rep.realized_pos_err = Some((fix.x - est.mean[0] as f32, fix.z - est.mean[1] as f32));
                    if self.pos2.len() >= 2 && !est.accepts(fix.x as f64, fix.z as f64, obs_sigma) {
                        let d2 = est.mahalanobis_sq(fix.x as f64, fix.z as f64, obs_sigma);
                        rep.note = format!(
                            "새 측량점이 99% 타원 밖 d²={d2:.2} > 9.21 ({:.1}, {:.1}) vs ({:.1}, {:.1})",
                            fix.x, fix.z, est.mean[0], est.mean[1]
                        );
                        rep.band = Some(est.band_rect(CHI2_2DOF_99.sqrt(), obs_sigma));
                        rep.cov = Some(cov3(est.cov));
                        rep.pos_model = Some(match est.model {
                            Pos2Model::Static => "static_ivw".into(),
                            Pos2Model::Dynamic => "dynamic_cv".into(),
                        });
                        rep.fix = Some(fix);
                        self.rays.clear();
                        self.rays.push(s);
                        rep.decision = Decision::Hold { reason: "band".into() };
                        return rep;
                    }
                }
                self.pos2.push(Pos2Obs { x: fix.x as f64, z: fix.z as f64, sigma_m: obs_sigma, tick: s.keyframe as u64 });
                self.est_x.push(fix.x as f64);
                self.est_z.push(fix.z as f64);
                if self.est_x.len() > POS2_RING_CAP {
                    self.est_x.remove(0);
                    self.est_z.remove(0);
                }
                self.last_fix = Some(fix.clone());
                rep.fix = Some(fix);
            }
            Err(e) => rep.fix_error = Some(e.code().to_string()),
        }

        let near = s.depth_range_m.map(|d| d * depth_scale.unwrap_or(1.0)).filter(|r| *r <= self.cfg.close_range_m);
        if let Some(r) = near {
            let at = (pose.x + r * s.heading.cos(), pose.z + r * s.heading.sin());
            rep.estimate = Some(Estimate { x: at.0, z: at.1, range_m: r, sigma_m: 0.25 * r, source: "depth".into() });
            if r <= self.cfg.reach_m + 0.25 {
                let look_pitch_deg = calc_pitch(pose.eye_m, r);
                rep.decision = Decision::Arrive { at, look_pitch_deg };
                rep.note = format!("가까움: 깊이 거리 {r:.2} m로 도달 (시선 {look_pitch_deg:+.0}°)");
            } else {
                let hop = r - self.cfg.reach_m;
                let next_r = r - hop;
                let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
                let to = (pose.x + hop * s.heading.cos(), pose.z + hop * s.heading.sin());
                self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
                rep.prediction = self.prediction;
                rep.decision = Decision::Approach { to, hop_m: hop, face: at, look_pitch_deg };
                rep.note = format!("가까움: 깊이 거리 {r:.2} m, {hop:.2} m 다가감 (시선 {look_pitch_deg:+.0}°)");
            }
            return rep;
        }

        let good = self.last_fix.as_ref().filter(|f| f.sigma_m <= (self.cfg.max_sigma_frac * f.range_m).max(0.25));
        if good.is_none() {
            let guess = s.depth_range_m.map(|d| d * depth_scale.unwrap_or(1.0)).unwrap_or(self.cfg.default_range_m);
            let step = plan_step(pose, s.heading, guess, holonomic, self.side, &self.cfg.tri);
            if let Some(f) = &self.last_fix {
                rep.note = format!("측량 오차 {:.2} m / 거리 {:.1} m: 기선 추가", f.sigma_m, f.range_m);
            } else {
                rep.note = format!("거리 미확정 (깊이 추정 {guess:.1} m): 삼각 측량 이동");
            }
            if matches!(rep.fix_error.as_deref(), Some("behind")) {
                self.rays.clear();
                self.rays.push(s);
            }
            let next_r = (step.face.0 - step.to.0).hypot(step.face.1 - step.to.1);
            self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: calc_pitch(pose.eye_m, next_r) });
            rep.prediction = self.prediction;
            rep.decision = Decision::Triangulate(step);
            return rep;
        }

        let (x, z, band, cov, method) = self.forecast_position(Some(s.keyframe as u64), extra).expect("a good fix implies a series");
        let sigma = good.map(|f| f.sigma_m).unwrap_or(1.0);
        let range = (x - pose.x).hypot(z - pose.z);
        rep.forecast_method = method.clone();
        rep.pos_model = Some(method);
        rep.band = Some(band);
        rep.cov = Some(cov);
        rep.estimate = Some(Estimate { x, z, range_m: range, sigma_m: sigma, source: if self.est_x.len() > 1 { "forecast".into() } else { "triangulation".into() } });
        if range <= self.cfg.reach_m + 0.25 {
            let look_pitch_deg = calc_pitch(pose.eye_m, range);
            rep.decision = Decision::Arrive { at: (x, z), look_pitch_deg };
            rep.note = format!("도달: 거리 {range:.2} m (시선 {look_pitch_deg:+.0}°)");
            return rep;
        }
        let hop = (range - self.cfg.reach_m).min(self.cfg.hop_max_m);
        let next_r = range - hop;
        let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
        let h = (z - pose.z).atan2(x - pose.x);
        let to = (pose.x + hop * h.cos(), pose.z + hop * h.sin());
        self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
        rep.prediction = self.prediction;
        rep.decision = Decision::Approach { to, hop_m: hop, face: (x, z), look_pitch_deg };
        rep.note = format!("예측 위치 ({x:.1}, {z:.1}) ±{sigma:.2} m 쪽으로 {hop:.1} m (시선 {look_pitch_deg:+.0}°)");
        rep
    }
}

impl Decider {
    pub fn fix_sigma_m(&self) -> Option<f32> {
        self.last_fix.as_ref().map(|f| f.sigma_m)
    }

    pub fn predict(&mut self, pose: &Pose, extra: &[&dyn Predictor], extra_sigma_m: f32) -> Option<DecisionReport> {
        let good = self.last_fix.clone().filter(|f| f.sigma_m <= (self.cfg.max_sigma_frac * f.range_m).max(0.25))?;
        let (x, z, band, cov, method) = self.forecast_position(None, extra)?;
        let sigma = good.sigma_m.hypot(extra_sigma_m);
        let range = (x - pose.x).hypot(z - pose.z);
        let mut rep = DecisionReport {
            decision: Decision::Hold { reason: "blind".into() },
            estimate: Some(Estimate { x, z, range_m: range, sigma_m: sigma, source: "forecast".into() }),
            fix: None,
            fix_error: None,
            depth_scale: None,
            residual_deg: None,
            prediction: None,
            forecast_method: method.clone(),
            band: Some(band),
            cov: Some(cov),
            pos_model: Some(method),
            realized_pos_err: None,
            note: String::new(),
        };
        self.prediction = None;
        let calc_pitch = |eye_m: f32, dist_m: f32| -> f32 {
            let dz = (eye_m - 0.05).max(0.1);
            let direct = -dz.atan2(dist_m.max(0.2)).to_degrees();
            if dist_m <= 1.6 { direct.clamp(-60.0, -12.0) } else { -12.0 }
        };
        if range <= self.cfg.reach_m + 0.25 {
            let look_pitch_deg = calc_pitch(pose.eye_m, range);
            rep.decision = Decision::Arrive { at: (x, z), look_pitch_deg };
            rep.note = format!("눈 감고 예측으로 도달: 거리 {range:.2} m (±{sigma:.2} m, 시선 {look_pitch_deg:+.0}°)");
            return Some(rep);
        }
        let hop = (range - self.cfg.reach_m).min(self.cfg.hop_max_m);
        let next_r = range - hop;
        let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
        let h = (z - pose.z).atan2(x - pose.x);
        let to = (pose.x + hop * h.cos(), pose.z + hop * h.sin());
        self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
        rep.prediction = self.prediction;
        rep.decision = Decision::Approach { to, hop_m: hop, face: (x, z), look_pitch_deg };
        rep.note = format!("눈 감고 예측 위치 ({x:.1}, {z:.1}) ±{sigma:.2} m 쪽으로 {hop:.1} m (ML 꺼짐)");
        Some(rep)
    }
}

/// Share of a target's angular width its visible centre wanders by (1σ) between views. Measured on the trees of the
/// v1.1 integrated test against their true trunks: 0.07, 0.11 and 0.32 of the width (RMS 0.2).
pub const LANDMARK_CENTRE_FRAC: f32 = 0.2;

/// Whether a fix may set the depth model's scale against the sighting `s` it ends on: rays that meet, a range the
/// scale is fit on, and a range error under 15 % once the target's own centre is counted. The fix's `sigma_m` comes
/// from the rays' centroid noise; the centre of a wide target adds `LANDMARK_CENTRE_FRAC` of its width to every
/// bearing, which scales the range error by the same ratio.
pub fn scale_usable(fix: &Fix, s: &Sighting) -> bool {
    let centre = s.sigma.hypot(LANDMARK_CENTRE_FRAC * s.extent) / s.sigma.max(1e-6);
    // Rays that meet can still be of two targets, when a track jumps from tree to tree: no depth model is off by that much.
    let near = s.depth_range_m.is_some_and(|d| d >= 0.5 && (0.5..=2.0).contains(&(fix.range_m / d)));
    fix.rays >= 3 && fix.residual_m <= 0.15 && fix.sigma_m * centre <= 0.15 * fix.range_m && (1.5..=9.0).contains(&fix.range_m) && near
}

pub fn centroid_sigma(width: u32, fov_deg: f32, px_err: f32) -> f32 {
    let f = width as f32 * 0.5 / (fov_deg.to_radians() * 0.5).tan();
    (px_err / f).max(1e-4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sight(k: u32, pose: Pose, target: (f32, f32), id: u32) -> Sighting {
        Sighting {
            keyframe: k,
            pose,
            heading: (target.1 - pose.z).atan2(target.0 - pose.x),
            sigma: 0.006,
            extent: 0.0,
            depth_range_m: Some(((target.0 - pose.x).hypot(target.1 - pose.z)) * 0.6),
            track_id: id,
            score: 0.8,
        }
    }

    fn play(target: (f32, f32), holonomic: bool, steps: usize) -> Vec<DecisionReport> {
        let mut d = Decider::new(DecideCfg::default());
        let mut pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut out = Vec::new();
        for k in 0..steps as u32 {
            let rep = d.observe(&pose, Some(sight(k, pose, target, 1)), holonomic, &[], None);
            match &rep.decision {
                Decision::Triangulate(t) => {
                    pose = Pose { x: t.to.0, z: t.to.1, yaw: (t.face.1 - t.to.1).atan2(t.face.0 - t.to.0), ..pose };
                }
                Decision::Approach { to, face, .. } => {
                    pose = Pose { x: to.0, z: to.1, yaw: (face.1 - to.1).atan2(face.0 - to.0), ..pose };
                }
                Decision::Arrive { .. } => {
                    out.push(rep);
                    break;
                }
                _ => {}
            }
            out.push(rep);
        }
        out
    }

    #[test]
    fn the_first_decision_is_a_triangulation_step_and_the_target_is_reached() {
        for holonomic in [true, false] {
            let reps = play((9.0, 2.0), holonomic, 20);
            assert!(matches!(reps[0].decision, Decision::Triangulate(_)), "{:?}", reps[0].decision);
            assert!(reps[0].fix.is_none(), "one sighting cannot give a range");
            let fix = reps.iter().find_map(|r| r.fix.clone()).expect("a fix after the step");
            assert!((fix.x - 9.0).abs() < 0.05 && (fix.z - 2.0).abs() < 0.05, "{fix:?}");
            let scale = reps.iter().find_map(|r| r.depth_scale).unwrap();
            assert!((scale - 1.0 / 0.6).abs() < 0.05, "{scale}");
            let last = reps.last().unwrap();
            assert!(matches!(last.decision, Decision::Arrive { .. }), "{holonomic}: {:?}", reps.iter().map(|r| r.decision.label()).collect::<Vec<_>>());
            for w in reps.windows(2) {
                if matches!(w[0].decision, Decision::Approach { .. }) {
                    let res = w[1].residual_deg.expect("an approach leaves a prediction");
                    assert!(res.abs() < 1.0, "{res}");
                }
            }
        }
    }

    #[test]
    fn a_lost_target_turns_toward_its_forecast_then_searches() {
        let mut d = Decider::new(DecideCfg::default());
        let reps = play((6.0, 0.0), true, 4);
        assert!(reps.iter().any(|r| r.fix.is_some()));
        let mut pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut r = d.observe(&pose, Some(sight(0, pose, (6.0, 0.0), 1)), true, &[], None);
        for k in 1..4 {
            if let Decision::Triangulate(t) = &r.decision {
                pose = Pose { x: t.to.0, z: t.to.1, yaw: 0.0, eye_m: 0.5 };
            }
            r = d.observe(&pose, Some(sight(k, pose, (6.0, 0.0), 1)), true, &[], None);
            if d.has_fix() {
                break;
            }
        }
        assert!(d.has_fix());
        let turned = Pose { yaw: std::f32::consts::PI, ..pose };
        let lost = d.observe(&turned, None, true, &[], None);
        match lost.decision {
            Decision::Search { turn_deg } => assert!(turn_deg.abs() > 150.0, "turn back toward the target: {turn_deg}"),
            other => panic!("{other:?}"),
        }
        d.observe(&turned, None, true, &[], None);
        let third = d.observe(&turned, None, true, &[], None);
        assert_eq!(third.decision, Decision::Search { turn_deg: DecideCfg::default().search_turn_deg });
    }

    #[test]
    fn a_full_turn_without_a_sighting_asks_for_another_viewpoint() {
        let mut d = Decider::new(DecideCfg::default());
        let pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut turns = 0.0;
        let explore = loop {
            match d.observe(&pose, None, false, &[], None).decision {
                Decision::Search { turn_deg } => turns += turn_deg,
                Decision::Explore { turned_deg } => break turned_deg,
                other => panic!("{other:?}"),
            }
            assert!(turns < 720.0, "never asks to move");
        };
        assert!(explore + DecideCfg::default().search_turn_deg >= 360.0 - 1e-3, "after about a full turn: {explore}");
        let t = (5.0, 1.0);
        let r = d.observe(&pose, Some(sight(0, pose, t, 3)), false, &[], None);
        assert!(matches!(r.decision, Decision::Triangulate(_)));
        assert!(matches!(d.observe(&pose, None, false, &[], None).decision, Decision::Search { .. }));
    }

    #[test]
    fn a_wide_targets_centre_keeps_a_short_baseline_from_setting_the_depth_scale() {
        // Three looks at a target 6 m ahead from 0.8 m of sideways baseline (about 8° of parallax).
        let target = (6.0, 0.0);
        let rays: Vec<Bearing> = [-0.4f32, 0.0, 0.4]
            .iter()
            .map(|&z| Bearing { x: 0.0, z, heading: (target.1 - z).atan2(target.0), sigma: 0.0073 })
            .collect();
        let fix = triangulate(&rays, 6.0).unwrap();
        let s = Sighting { extent: 0.0, sigma: 0.0073, ..sight(2, Pose { x: 0.0, z: 0.4, yaw: 0.0, eye_m: 0.5 }, target, 1) };
        assert!(scale_usable(&fix, &s), "a point-like target: the rays' noise is all there is ({fix:?})");
        assert!(!scale_usable(&fix, &Sighting { depth_range_m: Some(fix.range_m * 3.0), ..s.clone() }), "a range three times off the depth is another target");
        let tree = Sighting { extent: 10f32.to_radians(), ..s };
        assert!(!scale_usable(&fix, &tree), "a tree 10° wide: its centre alone moves the range by more than 15 %");
        let wide: Vec<Bearing> = [-2.0f32, 0.0, 2.0]
            .iter()
            .map(|&z| Bearing { x: 0.0, z, heading: (target.1 - z).atan2(target.0), sigma: 0.0073 })
            .collect();
        let fix = triangulate(&wide, 6.0).unwrap();
        assert!(scale_usable(&fix, &tree), "the same tree from 4 m of baseline ({fix:?})");
    }

    #[test]
    fn a_still_target_uses_the_weighted_mean() {
        let mut d = Decider::new(DecideCfg::default());
        // Seed 3 static fixes with different variances and verify inverse-variance weighted mean within 1e-6.
        let obs = [
            Pos2Obs { x: 6.00, z: 2.00, sigma_m: 0.20, tick: 1 },
            Pos2Obs { x: 6.10, z: 2.04, sigma_m: 0.10, tick: 2 },
            Pos2Obs { x: 5.96, z: 1.98, sigma_m: 0.05, tick: 3 },
        ];
        for o in obs {
            d.pos2.push(o);
            d.est_x.push(o.x);
            d.est_z.push(o.z);
        }
        let est = d.pos2.estimate(Some(4)).unwrap();
        let wsum: f64 = obs.iter().map(|o| 1.0 / (o.sigma_m * o.sigma_m)).sum();
        let wx: f64 = obs.iter().map(|o| o.x / (o.sigma_m * o.sigma_m)).sum::<f64>() / wsum;
        let wz: f64 = obs.iter().map(|o| o.z / (o.sigma_m * o.sigma_m)).sum::<f64>() / wsum;
        assert!((est.mean[0] - wx).abs() < 1e-6, "mean x {} != {wx}", est.mean[0]);
        assert!((est.mean[1] - wz).abs() < 1e-6, "mean z {} != {wz}", est.mean[1]);
        assert!(!est.accepts(9.5, 5.5, 0.10), "d² > 9.21 outlier must be rejected");

        // Verify Decider::observe emits Hold { reason: "band" } on a triangulated outlier (d² > 9.21).
        let p0 = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let p1 = Pose { x: 0.0, z: 2.5, yaw: 0.0, eye_m: 0.5 };
        let outlier = (9.5, 5.5);
        let _ = d.observe(&p0, Some(sight(10, p0, outlier, 9)), true, &[], None);
        let rep = d.observe(&p1, Some(sight(11, p1, outlier, 9)), true, &[], None);
        assert_eq!(rep.decision, Decision::Hold { reason: "band".into() }, "{rep:?}");
        assert!(rep.cov.is_some());
    }
}
