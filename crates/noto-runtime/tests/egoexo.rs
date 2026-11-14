//! The hand on an item it lifts, seen by the head camera (ego) and two fixed outside cameras (exo): how well the hand's
//! pose and its contacts come back from the head camera alone and from all three views fused.
//!
//! Each frame with the item in the hand (a log, a stone and a parcel, three seeds each): the hand's joint points (the
//! redrawn finger links, `World::own_arm_boxes`) are the truth; a view sees a point when its picture shows the robot
//! there at that depth; a seen point is measured with the stereo law along the line of sight (σ_Z = Z²·σ_d/(f·b),
//! σ_d = SGBM_SUBPIXEL_PX) and the pixel's angle across it (σ_uv = 1 px [추정: a keypoint's spread]). The ego estimate
//! fits the hand model to the head camera's points (`handfit`, Stage 1); the fused estimate fits it to the points fused
//! over the views by their information (`egoexo::fuse`). Both fits weigh each point by its information matrix. Stage 2
//! (`tactile`) reads the pads that touch from each fitted pose.
//!
//!   cargo test -p noto-runtime --test egoexo -- --ignored --nocapture

use noto_core::fit3::{mpjpe, pa_mpjpe, V3};
use noto_core::handfit::{fit_hand, HandFit, HandFitCfg, HandObs};
use noto_core::math::Vec3;
use noto_core::sim::ShapeKind;
use noto_core::tactile::{contact_probability, pad_geometry_mapped, TactileCfg};
use noto_core::{BaseKind, NotoConfig};
use noto_vision::egoexo::{fuse, inv3, occlusion_ratio, stereo_sigma_z, view_covariance, ViewObs};
use world_sim::render::{render_boxes_only, render_rays, Exposure, Obb, Rays, View, NEUTRAL_BG};
use world_sim::world::DropItem;
use world_sim::{Action, Camera, Frame, ItemKind, World, WorldConfig};

const SIGMA_D_PX: f64 = 0.27;
const SIGMA_UV_PX: f64 = 1.0;
const BASELINE_M: f64 = 0.06;
/// Noise draws per rendered frame (the pictures are the costly part; the fits are cheap).
const DRAWS: usize = 5;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn walker_with(kind: ItemKind, seed: u64) -> World {
    let mut noto = NotoConfig::preset(noto_core::hand::FingerCount::Five);
    noto.base = BaseKind::HumanFoot;
    let mut w = World::new(WorldConfig { seed, parcels: Some(0), noto, ..WorldConfig::default() }).unwrap();
    w.nodes.clear();
    w.drops.clear();
    w.agent.yaw = 0.0;
    w.drops.push(DropItem::whole(77, kind, w.agent.x + 0.6, w.agent.z));
    w
}

fn look_at(origin: [f32; 3], at: [f32; 3]) -> View {
    let d = [at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]];
    View { origin, yaw: d[2].atan2(d[0]), pitch: d[1].atan2(d[0].hypot(d[2])), roll: 0.0 }
}

/// Seen when the picture shows the robot (id 0, a depth) at the point's depth.
fn seen(f: &Frame, cam: &Camera, view: &View, p: [f32; 3]) -> Option<f32> {
    let (x, y, z) = cam.project_in(view, p)?;
    let (i, j) = (x as usize, y as usize);
    if x < 0.0 || y < 0.0 || i >= f.width || j >= f.height {
        return None;
    }
    let k = j * f.width + i;
    (f.ids[k] == 0 && f.depth[k].is_finite() && (f.depth[k] - z).abs() < 0.03).then_some(z)
}

fn observe(rng: &mut Rng, origin: [f32; 3], p: [f32; 3], f_px: f64) -> ViewObs {
    let o = origin.map(|v| v as f64);
    let q = p.map(|v| v as f64);
    let d = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
    let z = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let u = d.map(|v| v / z);
    let a = if u[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let e1 = {
        let c = [u[1] * a[2] - u[2] * a[1], u[2] * a[0] - u[0] * a[2], u[0] * a[1] - u[1] * a[0]];
        let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        c.map(|v| v / n)
    };
    let e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]];
    let (sz, st) = (stereo_sigma_z(z, f_px, BASELINE_M, SIGMA_D_PX), SIGMA_UV_PX / f_px);
    let (nz, n1, n2) = (rng.normal() * sz, rng.normal() * z * st, rng.normal() * z * st);
    let point = [0, 1, 2].map(|i| q[i] + u[i] * nz + e1[i] * n1 + e2[i] * n2);
    ViewObs { origin: o, point, sigma_z: sz, sigma_theta: st, weight: 1.0 }
}

/// The pads that touch the item at a fitted pose (P ≥ ½), as (finger, phalanx).
fn touching(hand: &noto_core::hand::Hand, fit: &HandFit, item: &Obb, cfg: &TactileCfg) -> Vec<(usize, usize)> {
    let to_item = |p: Vec3| -> Vec3 {
        let w = fit.wrist.apply([p.x as f64, p.y as f64, p.z as f64]);
        let r = [0, 1, 2].map(|i| (w[i] - item.center[i] as f64) as f32);
        let a = item.axes;
        Vec3::new(r[0] * a[0][0] + r[1] * a[0][1] + r[2] * a[0][2], r[0] * a[1][0] + r[1] * a[1][1] + r[2] * a[1][2], r[0] * a[2][0] + r[1] * a[2][1] + r[2] * a[2][2])
    };
    let half = Vec3::new(item.half[0], item.half[1], item.half[2]);
    pad_geometry_mapped(hand, &fit.angles, &to_item, Vec3::new(0.0, 0.0, 0.0), half, ShapeKind::Box)
        .into_iter()
        .filter(|g| contact_probability(g.gap_m, 0.0, cfg) >= 0.5)
        .map(|g| (g.finger, g.phalanx))
        .collect()
}

fn f1(est: &[(usize, usize)], truth: &[(usize, usize)]) -> Option<f32> {
    if est.is_empty() && truth.is_empty() {
        return None;
    }
    let tp = est.iter().filter(|e| truth.contains(e)).count() as f32;
    let p = if est.is_empty() { 0.0 } else { tp / est.len() as f32 };
    let r = if truth.is_empty() { 0.0 } else { tp / truth.len() as f32 };
    Some(if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 })
}

#[derive(Default)]
struct Acc {
    v: Vec<f64>,
}

impl Acc {
    fn add(&mut self, x: f64) {
        if x.is_finite() {
            self.v.push(x);
        }
    }

    fn mean(&self) -> String {
        if self.v.is_empty() {
            return "-".into();
        }
        format!("{:.2}", self.v.iter().sum::<f64>() / self.v.len() as f64)
    }

    fn median(&self) -> String {
        if self.v.is_empty() {
            return "-".into();
        }
        let mut v = self.v.clone();
        v.sort_by(f64::total_cmp);
        format!("{:.2}", v[v.len() / 2])
    }
}

/// One estimate's errors on one frame.
#[derive(Clone, Copy)]
struct Est {
    wrist_cm: f64,
    mpjpe_cm: f64,
    pa_cm: f64,
    f1: Option<f32>,
}

/// The truth of one frame: the right hand's finger links, in finger order, each finger's points from knuckle to tip
/// (world), or `None` when the redrawn hand is not on the item.
fn truth_points(w: &World, pose: &noto_core::dynamics::BodyPose, hand: &noto_core::hand::Hand) -> Option<Vec<Vec<[f32; 3]>>> {
    let links: Vec<_> = w.own_arm_boxes(pose).into_iter().filter(|b| b.name == "finger_r" || b.name == "thumb_r").collect();
    let counts: Vec<usize> = hand.fingers.iter().map(|f| f.segments.len()).collect();
    if links.len() != counts.iter().sum::<usize>() {
        return None;
    }
    let end = |b: &noto_core::dynamics::LinkBox, s: f32| [0, 1, 2].map(|i| b.center[i] + s * b.axes[2][i] * b.half[2]);
    let mut out = Vec::new();
    let mut at = 0;
    for c in &counts {
        let chain = &links[at..at + c];
        at += c;
        let mut pts = vec![w.body_to_world(end(&chain[0], -1.0))];
        pts.extend(chain.iter().map(|b| w.body_to_world(end(b, 1.0))));
        out.push(pts);
    }
    Some(out)
}

#[test]
#[ignore]
fn ego_and_outside_views_of_a_hand_lifting_an_item() {
    let cam = Camera { width: 320, height: 213, ..Camera::default() };
    let f_px = (cam.width as f64 * 0.5) / (cam.fov_deg as f64 * 0.5).to_radians().tan();
    let cfg = HandFitCfg::default();
    let tcfg = TactileCfg::default();
    let mut rng = Rng(0x5EED_E60E_0001);
    let (mut n, mut occl, mut truth_rms, mut ego_seen, mut fused_seen) = (0usize, Acc::default(), Acc::default(), Acc::default(), Acc::default());
    let (mut raw_ego, mut raw_fused_same) = (Acc::default(), Acc::default());
    let (mut fits, mut ego_only, mut fused_only, mut neither) = (0usize, 0usize, 0usize, 0usize);
    let mut paired: Vec<(Est, Est)> = Vec::new();
    // The same frames with outside cameras of 4x the focal length (a 1280 px sensor): their noise, the 320 px pictures' visibility.
    let mut paired_hi: Vec<(Est, Est)> = Vec::new();
    let mut by_phase: std::collections::BTreeMap<&'static str, Acc> = std::collections::BTreeMap::new();
    for kind in [ItemKind::Log, ItemKind::Stone, ItemKind::Parcel] {
        for seed in [7u64, 11, 13] {
            let mut w = walker_with(kind, seed);
            let (dx, dz) = (w.drops[0].x, w.drops[0].z);
            w.command(Action::Look { x: dx, z: dz }).unwrap();
            while w.busy() {
                w.tick();
            }
            w.command(Action::pick(77)).unwrap();
            let k = w.body.dims.k;
            let hand = w.body.working_hand(&w.hand).into_owned().scaled(k);
            let mut exo: Option<[View; 2]> = None;
            let mut prevs: Vec<[Vec<HandFit>; 3]> = (0..DRAWS).map(|_| [Vec::new(), Vec::new(), Vec::new()]).collect();
            while w.busy() {
                w.tick();
                let (Some(_), Some(pose)) = (w.hand_load(), w.body_pose()) else { continue };
                let Some(truth) = truth_points(&w, &pose, &hand) else { continue };
                let truth64: Vec<Vec<V3>> = truth.iter().map(|f| f.iter().map(|p| p.map(|v| v as f64)).collect()).collect();
                let exact = HandObs { points: truth64.iter().map(|f| f.iter().map(|p| Some(*p)).collect()).collect(), info: Vec::new() };
                let Some(truth_fit) = fit_hand(&hand, &exact, 1.0, &[], None, &cfg) else { continue };
                truth_rms.add(truth_fit.rms_m * 100.0);
                by_phase.entry(w.act_phase().map_or("-", |p| p.1)).or_default().add(truth_fit.rms_m * 100.0);
                let Some(item) = w.hand_load_boxes().into_iter().next() else { continue };
                let views = *exo.get_or_insert_with(|| {
                    let c = item.center;
                    [look_at([c[0] + 1.4, c[1] + 0.9, c[2] + 0.9], c), look_at([c[0] + 0.2, c[1] + 1.1, c[2] - 1.5], c)]
                });
                // The pictures: the head camera's own render, and the outside cameras' with the whole body and its hands.
                let hv = w.head_view(&cam);
                let ego_frame = w.render(&cam);
                let mut body: Vec<Obb> = w.robot_boxes(&pose);
                body.extend(w.own_view_boxes());
                let exo_frames: Vec<Frame> = views.iter().map(|v| render_rays(&w, &cam, &Rays::Pinhole { view: *v, exposure: Exposure::default() }, &body, &[], None)).collect();
                // The head camera's occlusion: the item's own pixels (drawn alone) that the hand (alone) covers in front.
                let hand_boxes: Vec<Obb> = w.own_view_boxes().into_iter().filter(|b| b.center != item.center).collect();
                let (lone, hands) = (render_boxes_only(&cam, &hv.view, &[item], NEUTRAL_BG), render_boxes_only(&cam, &hv.view, &hand_boxes, NEUTRAL_BG));
                let obj: Vec<bool> = lone.depth.iter().map(|d| d.is_finite()).collect();
                let front: Vec<bool> = hands.depth.iter().zip(&lone.depth).map(|(h, o)| h.is_finite() && (!o.is_finite() || h < o)).collect();
                if obj.iter().any(|o| *o) {
                    occl.add(occlusion_ratio(&obj, &front) as f64);
                }

                for prev in prevs.iter_mut() {
                let (mut ego_obs, mut fu_obs, mut hi_obs) = (HandObs::default(), HandObs::default(), HandObs::default());
                let (mut n_ego, mut n_fu, mut total) = (0, 0, 0);
                for f in &truth {
                    let (mut fe, mut fei, mut ff, mut ffi, mut fh, mut fhi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for p in f {
                        total += 1;
                        let t = p.map(|v| v as f64);
                        let e = seen(&ego_frame, &cam, &hv.view, *p).map(|_| observe(&mut rng, hv.view.origin, *p, f_px));
                        let mut all: Vec<ViewObs> = e.iter().copied().collect();
                        let mut all_hi = all.clone();
                        for (v, fr) in views.iter().zip(&exo_frames) {
                            if seen(fr, &cam, v, *p).is_some() {
                                all.push(observe(&mut rng, v.origin, *p, f_px));
                                all_hi.push(observe(&mut rng, v.origin, *p, 4.0 * f_px));
                            }
                        }
                        let fused = fuse(&all);
                        let fused_hi = fuse(&all_hi);
                        let dist = |a: V3| ((a[0] - t[0]).powi(2) + (a[1] - t[1]).powi(2) + (a[2] - t[2]).powi(2)).sqrt() * 100.0;
                        if let Some(o) = &e {
                            n_ego += 1;
                            raw_ego.add(dist(o.point));
                            if let Some((x, _)) = &fused {
                                raw_fused_same.add(dist(*x));
                            }
                        }
                        n_fu += fused.is_some() as usize;
                        fe.push(e.map(|o| o.point));
                        fei.push(e.and_then(|o| inv3(&view_covariance(&o))));
                        ff.push(fused.map(|(x, _)| x));
                        ffi.push(fused.and_then(|(_, c)| inv3(&c)));
                        fh.push(fused_hi.map(|(x, _)| x));
                        fhi.push(fused_hi.and_then(|(_, c)| inv3(&c)));
                    }
                    ego_obs.points.push(fe);
                    ego_obs.info.push(fei);
                    fu_obs.points.push(ff);
                    fu_obs.info.push(ffi);
                    hi_obs.points.push(fh);
                    hi_obs.info.push(fhi);
                }
                ego_seen.add(n_ego as f64 / total as f64 * 100.0);
                fused_seen.add(n_fu as f64 / total as f64 * 100.0);
                let truth_flat: Vec<V3> = truth64.iter().flatten().copied().collect();
                let truth_pads = touching(&hand, &truth_fit, &item, &tcfg);
                let mut run = |obs: &HandObs, prev: &mut Vec<HandFit>| -> Option<Est> {
                    let fit = fit_hand(&hand, obs, 1.0, prev, Some(truth_fit.wrist), &cfg).filter(|_| obs.seen() >= 4);
                    let Some(fit) = fit else {
                        prev.clear();
                        return None;
                    };
                    let pts: Vec<V3> = fit.points(&hand).into_iter().flatten().collect();
                    let (t, tt) = (fit.wrist.t, truth_fit.wrist.t);
                    let est = Est {
                        wrist_cm: ((t[0] - tt[0]).powi(2) + (t[1] - tt[1]).powi(2) + (t[2] - tt[2]).powi(2)).sqrt() * 100.0,
                        mpjpe_cm: mpjpe(&pts, &truth_flat) * 100.0,
                        pa_cm: pa_mpjpe(&pts, &truth_flat).unwrap_or(f64::NAN) * 100.0,
                        f1: f1(&touching(&hand, &fit, &item, &tcfg), &truth_pads),
                    };
                    prev.push(fit);
                    if prev.len() > 2 {
                        prev.remove(0);
                    }
                    Some(est)
                };
                let [pe, pf, ph] = prev;
                let (a, b, h) = (run(&ego_obs, pe), run(&fu_obs, pf), run(&hi_obs, ph));
                if let (Some(a), Some(h)) = (&a, h) {
                    paired_hi.push((Est { f1: a.f1, ..*a }, h));
                }
                match (a, b) {
                    (Some(a), Some(b)) => {
                        fits += 1;
                        paired.push((a, b));
                    }
                    (Some(_), None) => ego_only += 1,
                    (None, Some(_)) => fused_only += 1,
                    (None, None) => neither += 1,
                }
                }
                n += 1;
            }
        }
    }
    let col_of = |set: &[(Est, Est)], f: &dyn Fn(&(Est, Est)) -> f64| {
        let mut a = Acc::default();
        for p in set {
            a.add(f(p));
        }
        a
    };
    let col = |f: &dyn Fn(&(Est, Est)) -> f64| col_of(&paired, f);
    let hi = |f: &dyn Fn(&(Est, Est)) -> f64| col_of(&paired_hi, f);
    let f1v = |x: Option<f32>| x.map_or(f64::NAN, |v| v as f64);
    println!("[EGOEXO] 프레임 {n} (통나무 · 돌 · 소포 × 시드 3) · 손 모델 정답 맞춤 잔차 {} cm · 머리 카메라 가림 ρ_occ 평균 {} (중앙 {})", truth_rms.mean(), occl.mean(), occl.median());
    println!("[EGOEXO] 보이는 관절점: 머리 카메라 {} % · 세 시점 중 하나 이상 {} %", ego_seen.mean(), fused_seen.mean());
    println!("[EGOEXO] 머리 카메라가 본 점의 측정 오차: 머리 카메라만 {} cm · 세 시점 융합 {} cm", raw_ego.mean(), raw_fused_same.mean());
    println!("[EGOEXO] Stage 1 맞춤 (프레임 × 노이즈 {DRAWS}회): 둘 다 {fits} · 머리 카메라만 {ego_only} · 융합만 {fused_only} · 둘 다 실패 {neither} (관측점 4개 미만)");
    println!(
        "[EGOEXO] 둘 다 맞춘 {fits} 프레임 평균 (머리 카메라만 → 융합) — 손목 {} → {} cm · MPJPE {} → {} cm · PA-MPJPE {} → {} cm · 접촉 패드 F1 {} → {}",
        col(&|p| p.0.wrist_cm).mean(),
        col(&|p| p.1.wrist_cm).mean(),
        col(&|p| p.0.mpjpe_cm).mean(),
        col(&|p| p.1.mpjpe_cm).mean(),
        col(&|p| p.0.pa_cm).mean(),
        col(&|p| p.1.pa_cm).mean(),
        col(&|p| f1v(p.0.f1)).mean(),
        col(&|p| f1v(p.1.f1)).mean()
    );
    println!("[EGOEXO] 중앙값 — 손목 {} → {} cm · PA-MPJPE {} → {} cm", col(&|p| p.0.wrist_cm).median(), col(&|p| p.1.wrist_cm).median(), col(&|p| p.0.pa_cm).median(), col(&|p| p.1.pa_cm).median());
    println!(
        "[EGOEXO] 외부 카메라 4배 초점(1280 px) — {} 프레임 평균 (머리 카메라만 → 융합): 손목 {} → {} cm · PA-MPJPE {} → {} cm · 접촉 패드 F1 {} → {}",
        paired_hi.len(),
        hi(&|p| p.0.wrist_cm).mean(),
        hi(&|p| p.1.wrist_cm).mean(),
        hi(&|p| p.0.pa_cm).mean(),
        hi(&|p| p.1.pa_cm).mean(),
        hi(&|p| f1v(p.0.f1)).mean(),
        hi(&|p| f1v(p.1.f1)).mean()
    );
    let phases: Vec<String> = by_phase.iter().map(|(k, a)| format!("{k} {} cm ({})", a.mean(), a.v.len())).collect();
    println!("[EGOEXO] 단계별 정답 맞춤 잔차: {}", phases.join(" · "));
    assert!(n > 20, "the lifts have frames");
}
