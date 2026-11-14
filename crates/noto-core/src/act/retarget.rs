//! One recorded hand–object interaction played on other bodies and hands. The record is embodiment-free: the item's
//! pose over time, each wrist's pose, and the contacts on the item (item frame) with their forces. Played on a body,
//! the item keeps its path and each wrist keeps its place against the item (T_obj⁻¹·T_wrist, its offset scaled with the
//! hand) — so the wrench the hands put on the item is the demonstration's — and the body has to reach it:
//!
//! - the hand: the contacts go to the body's finger count (the thumb's stay; the other fingers' are merged in finger
//!   order into the remaining fingers, each at the force-weighted centre of its group, carrying the group's force, so the
//!   total force is kept exactly and the moment wherever the merged forces are parallel), and the hand closes on the
//!   item by its own actuation (`sim::top_grasp_shaped`: the coupled tendon's λ shares, or the direct drive's torque LP)
//!   — and the engine's own rule says how many hands that hold needs (`TopHands`);
//! - the body: per frame, each wrist target through `reach::arm_ik3` over a lean of the trunk and (on legs) a crouch of
//!   the hips, the pose with the least
//!   Σ(τ/τ_max)² + (1 − clamp(ZMP margin))² + a continuity term from the frame before, solved by `kin::solve` with the
//!   item's weight on the hands.

use serde::Serialize;

use crate::act::grasp::TopHands;
use crate::act::reach;
use crate::balance::{Body, G};
use crate::base::BaseKind;
use crate::fit3::{M3, V3};
use crate::hand::{FingerCount, NotoConfig};
use crate::joint::Actuation;
use crate::kin::{self, KinLoads};
use crate::parts::BodyPlan;
use crate::sim::{top_grasp_shaped, ShapeKind};

/// A rigid pose: p ↦ R·p + t.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Rigid {
    pub r: M3,
    pub t: V3,
}

impl Rigid {
    pub const ID: Rigid = Rigid { r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: [0.0; 3] };

    pub fn apply(&self, p: V3) -> V3 {
        [0, 1, 2].map(|i| (0..3).map(|j| self.r[i][j] * p[j]).sum::<f64>() + self.t[i])
    }

    pub fn inverse(&self) -> Rigid {
        let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| self.r[j][i]));
        let t = [0, 1, 2].map(|i| -(0..3).map(|j| rt[i][j] * self.t[j]).sum::<f64>());
        Rigid { r: rt, t }
    }

    /// self ∘ other.
    pub fn then(&self, other: &Rigid) -> Rigid {
        let r = [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| self.r[i][k] * other.r[k][j]).sum::<f64>()));
        Rigid { r, t: self.apply(other.t) }
    }
}

/// The wrist in the item's frame: T_obj⁻¹·T_wrist.
pub fn object_relative(obj: &Rigid, wrist: &Rigid) -> Rigid {
    obj.inverse().then(wrist)
}

/// A contact of the demonstration on the item, item frame: which finger (0 the thumb), where, the force on the item.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DemoContact {
    pub finger: usize,
    pub point: V3,
    pub force: V3,
}

/// The wrench the contacts put on the item about `com` (item frame): force, then moment.
pub fn wrench(contacts: &[DemoContact], com: V3) -> [f64; 6] {
    let mut w = [0.0; 6];
    for c in contacts {
        let r = [c.point[0] - com[0], c.point[1] - com[1], c.point[2] - com[2]];
        let f = c.force;
        let m = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
        for i in 0..3 {
            w[i] += f[i];
            w[3 + i] += m[i];
        }
    }
    w
}

/// The contacts of one hand on a hand of `fingers` fingers: the thumb's (finger 0) as they are, the others' merged in
/// finger order into `fingers − 1` groups, each one contact at its force-weighted centre carrying its summed force.
pub fn virtual_contacts(contacts: &[DemoContact], fingers: usize) -> Vec<DemoContact> {
    let mut others: Vec<usize> = contacts.iter().filter(|c| c.finger != 0).map(|c| c.finger).collect();
    others.sort_unstable();
    others.dedup();
    let slots = fingers.saturating_sub(1).max(1);
    if others.len() <= slots {
        return contacts.to_vec();
    }
    let mut out: Vec<DemoContact> = contacts.iter().filter(|c| c.finger == 0).copied().collect();
    for g in 0..slots {
        let (lo, hi) = (g * others.len() / slots, (g + 1) * others.len() / slots);
        let group: Vec<&DemoContact> = contacts.iter().filter(|c| others[lo..hi].contains(&c.finger)).collect();
        let (mut f, mut p, mut w) = ([0.0; 3], [0.0; 3], 0.0);
        for c in &group {
            let m = (c.force[0].powi(2) + c.force[1].powi(2) + c.force[2].powi(2)).sqrt();
            for i in 0..3 {
                f[i] += c.force[i];
                p[i] += m * c.point[i];
            }
            w += m;
        }
        if w > 0.0 {
            out.push(DemoContact { finger: g + 1, point: p.map(|v| v / w), force: f });
        }
    }
    out
}

/// The demonstration in the body frame it was recorded in (`reach`'s: x forward, y across with the right shoulder at
/// negative y, z up), sampled every `dt_s`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Demo {
    pub dt_s: f32,
    pub obj: Vec<Rigid>,
    /// Each frame's wrists (right, left); `None` while that hand is off the item.
    pub wrist: Vec<[Option<Rigid>; 2]>,
    /// Each hand's contacts on the item while it holds it.
    pub contacts: [Vec<DemoContact>; 2],
    pub size_m: [f32; 3],
    pub shape: ShapeKind,
    pub mass_kg: f32,
    pub mu: f32,
    /// The size of the demonstrating hand against the model hand (its k).
    pub hand_k: f32,
}

/// A body to play the demonstration on.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Embodiment {
    pub name: String,
    pub plan: BodyPlan,
    pub base: BaseKind,
    pub fingers: FingerCount,
    /// The hand's actuation; `None`: the finger count's preset.
    pub actuation: Option<Actuation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RetargetFrame {
    pub reachable: bool,
    pub trunk_deg: f32,
    /// How far the hips went down from standing, m.
    pub crouch_m: f32,
    pub shoulder_deg: [f32; 2],
    pub elbow_deg: [f32; 2],
    pub abd_deg: [f32; 2],
    pub zmp_margin: f32,
    pub effort_sum: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RetargetRollout {
    pub name: String,
    pub k: f32,
    pub fingers: u8,
    pub actuation: &'static str,
    /// The hand's squeeze capacity and what one hand holds up (N); the hands the engine's own rule asks for
    /// (`TopHands`: one when it holds `GRIP_SAFETY` times the weight, two when the pair holds the weight, 0: none) and
    /// their margin over `GRIP_SAFETY` × the weight. Played when it asks for no more hands than the demonstration used.
    pub squeeze_max_n: f32,
    pub hold_n: f32,
    pub hands: u8,
    pub hold_margin: f32,
    pub hold_ok: bool,
    /// The wrench the merged contacts lose against the demonstration's: |Δforce| (N) and |Δmoment| (N·m), worst hand.
    pub force_err_n: f32,
    pub moment_err_nm: f32,
    /// Frames whose wrist targets the body reached.
    pub reach_share: f32,
    pub min_zmp_margin: f32,
    pub max_effort: f32,
    pub frames: Vec<RetargetFrame>,
}

const TRUNK_DEG: [f32; 7] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
/// Hip drops tried on legs, as shares of the standing hip height.
const CROUCH: [f32; 5] = [0.0, 0.1, 0.2, 0.3, 0.4];
const W_CONT: f32 = 1e-3;

/// Plays `demo` on `e`.
pub fn retarget(demo: &Demo, e: &Embodiment) -> RetargetRollout {
    let body = Body::new(&e.plan, e.base, e.fingers);
    let k = body.dims.k;
    let mut cfg = NotoConfig::preset(e.fingers);
    if let Some(a) = e.actuation {
        cfg.joints.actuation = a;
    }
    let hand = cfg.build().expect("a preset hand builds").scaled(k);
    let closed = top_grasp_shaped(&hand, demo.size_m, demo.shape);
    let hands_on = demo.contacts.iter().filter(|c| !c.is_empty()).count().max(1) as u8;
    let th = TopHands::of(&closed, demo.mass_kg, demo.mu);
    let hold_ok = th.hands > 0 && th.hands <= hands_on;
    let com = [0.0; 3];
    let (mut force_err_n, mut moment_err_nm) = (0.0f32, 0.0f32);
    for cs in demo.contacts.iter().filter(|c| !c.is_empty()) {
        let (a, b) = (wrench(cs, com), wrench(&virtual_contacts(cs, hand.fingers.len()), com));
        let d = |i: usize| ((a[i] - b[i]).powi(2) + (a[i + 1] - b[i + 1]).powi(2) + (a[i + 2] - b[i + 2]).powi(2)).sqrt() as f32;
        force_err_n = force_err_n.max(d(0));
        moment_err_nm = moment_err_nm.max(d(3));
    }
    let scale = (k / demo.hand_k.max(1e-6)) as f64;
    let mut frames = Vec::with_capacity(demo.obj.len());
    let mut prev: Option<(f32, f32)> = None;
    for (obj, wrists) in demo.obj.iter().zip(&demo.wrist) {
        let targets: Vec<(usize, [f32; 3])> = wrists
            .iter()
            .enumerate()
            .filter_map(|(side, w)| {
                // The wrist's place in the item's frame, its offset scaled with the hand, back on the item's path.
                let rel = object_relative(obj, &(*w)?);
                let p = obj.apply(rel.t.map(|v| v * scale));
                Some((side, [p[0] as f32, p[1] as f32, p[2] as f32]))
            })
            .collect();
        let load = [0.0, 0.0, -demo.mass_kg * G / targets.len().max(1) as f32];
        let mut best: Option<(f32, RetargetFrame)> = None;
        let crouches: &[f32] = if body.legged() { &CROUCH } else { &CROUCH[..1] };
        for (&trunk, &drop) in TRUNK_DEG.iter().flat_map(|t| crouches.iter().map(move |d| (t, d))) {
            let mut pose = body.stand(0.0);
            pose.waist_deg = trunk - pose.pelvis_deg;
            let crouch_m = drop * body.dims.hip_m;
            pose.hip[1] -= crouch_m;
            let sh = reach::shoulder_point(&body, &pose);
            let mut ok = true;
            let mut loads = KinLoads::default();
            for &(side, at) in &targets {
                let s3 = [sh[0], body.shoulder_mount(side)[1], sh[1]];
                match reach::arm_ik3(&body.dims, s3, at, pose.trunk_deg(), side) {
                    Some((sd, ed, ad)) => {
                        pose.shoulder_deg[side] = sd;
                        pose.elbow_deg[side] = ed;
                        pose.shoulder_abd_deg[side] = ad;
                        loads.hand_force[side] = load;
                    }
                    None => ok = false,
                }
            }
            if !ok {
                continue;
            }
            let rep = kin::solve(&body, &pose, &loads);
            // A pose the body cannot hold (a joint out of its range or rating) only stands when no lean gives one it can.
            let cont = prev.map_or(0.0, |(t, c)| (trunk - t).powi(2) + (100.0 * (crouch_m - c)).powi(2));
            let cost = rep.effort_sum + (1.0 - rep.margin.clamp(0.0, 1.0)).powi(2) + W_CONT * cont + if rep.feasible { 0.0 } else { 1e3 };
            if best.as_ref().map_or(true, |(c, _)| cost < *c) {
                best = Some((cost, RetargetFrame { reachable: rep.feasible, trunk_deg: trunk, crouch_m, shoulder_deg: pose.shoulder_deg, elbow_deg: pose.elbow_deg, abd_deg: pose.shoulder_abd_deg, zmp_margin: rep.margin, effort_sum: rep.effort_sum }));
            }
        }
        let f = best.map(|(_, f)| f).unwrap_or(RetargetFrame { reachable: false, trunk_deg: f32::NAN, crouch_m: f32::NAN, shoulder_deg: [f32::NAN; 2], elbow_deg: [f32::NAN; 2], abd_deg: [f32::NAN; 2], zmp_margin: f32::NAN, effort_sum: f32::NAN });
        if f.reachable {
            prev = Some((f.trunk_deg, f.crouch_m));
        }
        frames.push(f);
    }
    let reached: Vec<&RetargetFrame> = frames.iter().filter(|f| f.reachable).collect();
    RetargetRollout {
        name: e.name.clone(),
        k,
        fingers: hand.fingers.len() as u8,
        actuation: match cfg.joints.actuation {
            Actuation::Direct => "direct",
            Actuation::Coupled { .. } => "coupled",
        },
        squeeze_max_n: closed.squeeze_max_n(),
        hold_n: th.hold_n,
        hands: th.hands,
        hold_margin: if th.hands > 0 { th.margin(th.hands) } else { 0.0 },
        hold_ok,
        force_err_n,
        moment_err_nm,
        reach_share: reached.len() as f32 / frames.len().max(1) as f32,
        min_zmp_margin: reached.iter().map(|f| f.zmp_margin).fold(f32::INFINITY, f32::min),
        max_effort: reached.iter().map(|f| f.effort_sum).fold(0.0, f32::max),
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wrist_keeps_its_place_on_the_item_and_merged_contacts_keep_the_force() {
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let obj = Rigid { r: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], t: [0.5, 0.0, 0.4] };
        let wrist = Rigid { r: Rigid::ID.r, t: [0.4, -0.15, 0.5] };
        let rel = object_relative(&obj, &wrist);
        let back = obj.then(&rel);
        for i in 0..3 {
            assert!((back.t[i] - wrist.t[i]).abs() < 1e-12);
        }
        // Moving the item and the wrist together keeps the relation.
        let lift = Rigid { r: Rigid::ID.r, t: [0.0, 0.0, 0.2] };
        let rel2 = object_relative(&lift.then(&obj), &lift.then(&wrist));
        assert!((0..3).all(|i| (rel2.t[i] - rel.t[i]).abs() < 1e-12));

        let five: Vec<DemoContact> = (0..5).map(|f| DemoContact { finger: f, point: [0.02 * f as f64, if f == 0 { -0.03 } else { 0.03 }, 0.0], force: [0.0, if f == 0 { 6.0 } else { -1.5 }, 0.2] }).collect();
        for n in [2usize, 3, 4, 5] {
            let v = virtual_contacts(&five, n);
            assert!(v.len() <= n.max(2));
            let (a, b) = (wrench(&five, [0.0; 3]), wrench(&v, [0.0; 3]));
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 1e-9, "the total force stays ({n} fingers)");
            }
            // The other fingers' forces are parallel here, so the moment stays too.
            for i in 3..6 {
                assert!((a[i] - b[i]).abs() < 1e-9, "the moment stays ({n} fingers): {} vs {}", a[i], b[i]);
            }
        }
    }

    #[test]
    fn a_lift_above_the_small_bodys_shoulder_is_the_tall_bodys_to_play() {
        // The item rises from 1.0 to 1.3 m, 0.35 m ahead; the wrists 0.1 m behind it, at the shoulders' width.
        let obj: Vec<Rigid> = (0..12).map(|i| Rigid { r: Rigid::ID.r, t: [0.35, 0.0, 1.0 + 0.3 * i as f64 / 11.0] }).collect();
        let wrist: Vec<[Option<Rigid>; 2]> = obj.iter().map(|o| [Some(Rigid { r: Rigid::ID.r, t: [o.t[0] - 0.1, -0.18, o.t[2] + 0.05] }), Some(Rigid { r: Rigid::ID.r, t: [o.t[0] - 0.1, 0.18, o.t[2] + 0.05] })]).collect();
        let c = |f: usize, y: f64| DemoContact { finger: f, point: [0.0, y, 0.0], force: [0.0, -y.signum() * 5.0, 0.0] };
        let demo = Demo { dt_s: 0.1, obj, wrist, contacts: [vec![c(0, -0.1), c(1, -0.1)], vec![c(0, 0.1), c(1, 0.1)]], size_m: [0.3, 0.2, 0.2], shape: ShapeKind::Box, mass_kg: 2.0, mu: 0.6, hand_k: 1.0 };
        let compact = retarget(&demo, &Embodiment { name: "compact".into(), plan: BodyPlan::default(), base: BaseKind::HumanFoot, fingers: FingerCount::Five, actuation: None });
        let standard = retarget(&demo, &Embodiment { name: "standard".into(), plan: BodyPlan::standard(), base: BaseKind::HumanFoot, fingers: FingerCount::Five, actuation: None });
        assert!(standard.k > compact.k);
        assert_eq!(compact.frames.len(), 12);
        assert!(standard.reach_share > 0.8, "the tall body plays it: {}", standard.reach_share);
        assert!(compact.reach_share < 0.3, "above its shoulders the small body cannot: {}", compact.reach_share);
        assert!(standard.min_zmp_margin > 0.0 && standard.max_effort > 0.0);
        let two = retarget(&demo, &Embodiment { name: "gripper".into(), plan: BodyPlan::default(), base: BaseKind::HumanFoot, fingers: FingerCount::Two, actuation: None });
        assert_eq!((two.fingers, two.actuation), (2, "direct"));
        assert!(two.squeeze_max_n > 0.0);
    }
}

