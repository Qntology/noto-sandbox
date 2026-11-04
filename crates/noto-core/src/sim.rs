use serde::{Deserialize, Serialize};

use crate::finger::{sample_points, FingerSpec};
use crate::hand::Hand;
use crate::joint::{Actuation, JointAngles};
use crate::math::Vec3;

pub const GRAVITY_MPS2: f32 = 9.81;

const CLOSE_STEPS: usize = 100;
const MIN_LEVER_M: f32 = 0.02;
const FORCE_CLOSURE_DIRS: usize = 512;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Payload {
    pub name: String,
    pub size_m: [f32; 3],
    pub mass_kg: f32,
    pub friction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspSetup {
    pub z_offset: f32,
    pub contact_margin_m: f32,
    pub safety_factor: f32,
}

impl Default for GraspSetup {
    fn default() -> Self {
        GraspSetup {
            z_offset: 0.03,
            contact_margin_m: 0.004,
            safety_factor: 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ContactSource {
    Finger { index: usize },
    Palm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub source: ContactSource,
    pub point: Vec3,
    pub normal: Vec3,
    pub force_n: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraspType {
    None,
    Pinch,
    Power,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspResult {
    pub contacts: Vec<Contact>,
    pub grasp_type: GraspType,
    pub force_closed: bool,
    pub friction_capacity_n: f32,
    pub required_n: f32,
    pub stable: bool,
    pub max_payload_kg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspOutcome {
    pub result: GraspResult,
    pub angles: Vec<Vec<JointAngles>>,
}

pub fn closure_angles(finger: &FingerSpec, actuation: Actuation, closure: f32) -> Vec<JointAngles> {
    let closure = closure.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(finger.joint_count());
    let mut prev_flex = 0.0;
    for (i, joint) in finger.joints.iter().enumerate() {
        let target = match actuation {
            Actuation::Direct => closure * joint.flex_max,
            Actuation::Coupled { ratio } => {
                if i == 0 {
                    closure * joint.flex_max
                } else {
                    ratio * prev_flex
                }
            }
        };
        let flex = joint.clamp_flex(target);
        out.push(JointAngles {
            flex,
            ..Default::default()
        });
        prev_flex = flex;
    }
    out
}

pub fn fk_all(hand: &Hand, angles: &[Vec<JointAngles>]) -> Vec<Vec<Vec3>> {
    hand.fingers
        .iter()
        .zip(angles)
        .map(|(f, a)| f.points(a))
        .collect()
}

fn box_sdf(p: Vec3, c: Vec3, h: Vec3) -> f32 {
    let d = p - c;
    let q = Vec3::new(d.x.abs() - h.x, d.y.abs() - h.y, d.z.abs() - h.z);
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

fn radial_normal(p: Vec3, center: Vec3) -> Vec3 {
    (p - center).normalized()
}

fn force_closed(normals: &[Vec3]) -> bool {
    if normals.len() < 2 {
        return false;
    }
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    for i in 0..FORCE_CLOSURE_DIRS {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / FORCE_CLOSURE_DIRS as f32;
        let r = (1.0 - y * y).sqrt();
        let theta = golden_angle * i as f32;
        let d = Vec3::new(theta.cos() * r, y, theta.sin() * r);
        if normals.iter().all(|n| n.dot(d) > 1e-3) {
            return false;
        }
    }
    true
}

pub fn grasp(hand: &Hand, payload: &Payload, setup: &GraspSetup) -> GraspOutcome {
    let actuation = hand.config.joints.actuation;
    let half = Vec3::new(
        payload.size_m[0] * 0.5,
        payload.size_m[1] * 0.5,
        payload.size_m[2] * 0.5,
    );
    let center = Vec3::new(0.0, half.y, setup.z_offset);
    let margin = setup.contact_margin_m;

    let mut angles: Vec<Vec<JointAngles>> = hand
        .fingers
        .iter()
        .map(|f| vec![JointAngles::default(); f.joint_count()])
        .collect();
    let mut contacts: Vec<Contact> = Vec::new();
    let mut blocked = vec![false; hand.fingers.len()];

    for step in 1..=CLOSE_STEPS {
        let closure = step as f32 / CLOSE_STEPS as f32;
        for (fi, finger) in hand.fingers.iter().enumerate() {
            if blocked[fi] {
                continue;
            }
            let candidate = closure_angles(finger, actuation, closure);
            let points = finger.points(&candidate);
            let touch = sample_points(&points)
                .into_iter()
                .filter(|p| p.y >= margin)
                .map(|p| (p, box_sdf(p, center, half)))
                .filter(|(_, d)| *d <= margin)
                .min_by(|a, b| a.1.total_cmp(&b.1));

            match touch {
                Some((p, _)) => {
                    blocked[fi] = true;
                    let lever = p.distance(finger.mount).max(MIN_LEVER_M);
                    contacts.push(Contact {
                        source: ContactSource::Finger { index: fi },
                        point: p,
                        normal: radial_normal(p, center),
                        force_n: finger.joints[0].max_torque / lever,
                    });
                }
                None => angles[fi] = candidate,
            }
        }
        if blocked.iter().all(|&b| b) {
            break;
        }
    }

    let footprint_on_palm = center.x - half.x <= hand.palm.width * 0.5
        && center.x + half.x >= -hand.palm.width * 0.5
        && center.z - half.z <= 0.0
        && center.z + half.z >= -hand.palm.length;
    if footprint_on_palm {
        let palm_point = Vec3::new(
            center.x,
            0.0,
            center.z.clamp(-hand.palm.length, 0.0),
        );
        let squeeze_n: f32 = contacts.iter().map(|c| c.force_n).sum();
        contacts.push(Contact {
            source: ContactSource::Palm,
            point: palm_point,
            normal: radial_normal(palm_point, center),
            force_n: squeeze_n,
        });
    }

    let result = score(&contacts, payload, setup);
    GraspOutcome { result, angles }
}

fn score(contacts: &[Contact], payload: &Payload, setup: &GraspSetup) -> GraspResult {
    let normals: Vec<Vec3> = contacts.iter().map(|c| c.normal).collect();
    let closed = force_closed(&normals);
    let palm_support = contacts
        .iter()
        .any(|c| matches!(c.source, ContactSource::Palm));
    let supported = palm_support || closed;
    let finger_contacts = contacts
        .iter()
        .filter(|c| matches!(c.source, ContactSource::Finger { .. }))
        .count();
    let friction_capacity_n: f32 = contacts
        .iter()
        .map(|c| c.force_n * payload.friction)
        .sum();
    let required_n = payload.mass_kg * GRAVITY_MPS2 * setup.safety_factor;
    let holdable = supported && finger_contacts >= 2;
    let max_payload_kg = if holdable {
        friction_capacity_n / (GRAVITY_MPS2 * setup.safety_factor)
    } else {
        0.0
    };
    let grasp_type = match finger_contacts {
        0 => GraspType::None,
        1 | 2 => GraspType::Pinch,
        _ => GraspType::Power,
    };
    GraspResult {
        contacts: contacts.to_vec(),
        grasp_type,
        force_closed: closed,
        friction_capacity_n,
        required_n,
        stable: holdable && friction_capacity_n >= required_n,
        max_payload_kg,
    }
}

pub const PHALANX_PAD_M: f32 = 0.009;
pub const PALM_PAD_M: f32 = 0.012;
pub const FACE_GAP_M: f32 = 0.001;
const PHALANX_SAMPLES: usize = 9;
const TOUCH_TOL_M: f32 = 0.0005;

/// What the fingers close on, in the hand frame of `top_grasp` (x across the knuckles, y away from the palm, z along
/// the fingers). Every shape fills the same bounding box (`half`), so the hand sits the same way on all of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShapeKind {
    #[default]
    Box,
    /// A round bar along `axis` (0 = across the knuckles: a log the fingers wrap round); its section is a circle of
    /// the mean of the other two half-sizes.
    Cylinder { axis: u8 },
    /// A lump: the box with its edges and corners rounded off by `round` × its smallest half-size (1 on a cube is a
    /// ball). Unlike a box–ellipsoid blend it has an exact distance, so "on the surface" means the same everywhere.
    Lump { round: f32 },
}

impl ShapeKind {
    pub fn id(self) -> &'static str {
        match self {
            ShapeKind::Box => "box",
            ShapeKind::Cylinder { .. } => "cylinder",
            ShapeKind::Lump { .. } => "lump",
        }
    }
}

fn axes_of(axis: u8) -> (usize, usize, usize) {
    match axis {
        1 => (1, 2, 0),
        2 => (2, 0, 1),
        _ => (0, 1, 2),
    }
}

fn comp(v: Vec3, i: usize) -> f32 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn from_comps(a: usize, va: f32, b: usize, vb: f32, c: usize, vc: f32) -> Vec3 {
    let mut out = [0.0f32; 3];
    out[a] = va;
    out[b] = vb;
    out[c] = vc;
    Vec3::new(out[0], out[1], out[2])
}

fn lump_radius(h: Vec3, round: f32) -> f32 {
    round.clamp(0.0, 1.0) * h.x.min(h.y).min(h.z).max(0.0)
}

/// Signed distance to the shape: positive outside, zero on the surface, negative inside.
pub fn shape_sdf(p: Vec3, c: Vec3, h: Vec3, shape: ShapeKind) -> f32 {
    match shape {
        ShapeKind::Box => box_sdf(p, c, h),
        ShapeKind::Cylinder { axis } => {
            let (a, b1, b2) = axes_of(axis);
            let d = p - c;
            let r = 0.5 * (comp(h, b1) + comp(h, b2));
            let da = comp(d, a).abs() - comp(h, a);
            let dr = comp(d, b1).hypot(comp(d, b2)) - r;
            da.max(dr).min(0.0) + da.max(0.0).hypot(dr.max(0.0))
        }
        ShapeKind::Lump { round } => {
            let r = lump_radius(h, round);
            box_sdf(p, c, Vec3::new(h.x - r, h.y - r, h.z - r)) - r
        }
    }
}

/// The nearest point of the shape's surface and its outward normal there.
pub fn shape_closest(p: Vec3, c: Vec3, h: Vec3, shape: ShapeKind) -> (Vec3, Vec3) {
    match shape {
        ShapeKind::Box => box_closest(p, c, h),
        ShapeKind::Cylinder { axis } => {
            let (a, b1, b2) = axes_of(axis);
            let d = p - c;
            let (ha, r) = (comp(h, a), 0.5 * (comp(h, b1) + comp(h, b2)));
            let (x, y, z) = (comp(d, a), comp(d, b1), comp(d, b2));
            let rho = y.hypot(z);
            let (uy, uz) = if rho > 1e-7 { (y / rho, z / rho) } else { (1.0, 0.0) };
            let q = c + from_comps(a, x.clamp(-ha, ha), b1, uy * rho.min(r), b2, uz * rho.min(r));
            let out = p - q;
            if out.length() > 1e-7 && (x.abs() > ha || rho > r) {
                return (q, out.normalized());
            }
            // Inside (or on the surface): out through whichever of the round side and the end is nearer.
            if ha - x.abs() < r - rho {
                (c + from_comps(a, x.signum() * ha, b1, y, b2, z), from_comps(a, x.signum(), b1, 0.0, b2, 0.0))
            } else {
                (c + from_comps(a, x, b1, uy * r, b2, uz * r), from_comps(a, 0.0, b1, uy, b2, uz))
            }
        }
        ShapeKind::Lump { round } => {
            let r = lump_radius(h, round);
            let (q, n) = box_closest(p, c, Vec3::new(h.x - r, h.y - r, h.z - r));
            (q + n.scale(r), n)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhalanxContact {
    pub finger: usize,
    pub phalanx: usize,
    pub point: Vec3,
    pub normal: Vec3,
    pub along: f32,
    pub jac: Vec<f32>,
    pub max_n: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TopGrasp {
    pub center: Vec3,
    pub half: Vec3,
    pub shape: ShapeKind,
    pub angles: Vec<Vec<JointAngles>>,
    pub closure: Vec<f32>,
    pub contacts: Vec<PhalanxContact>,
    pub finger_max_n: Vec<f32>,
    pub touching: usize,
}

impl TopGrasp {
    pub fn squeeze_max_n(&self) -> f32 {
        self.finger_max_n.iter().sum()
    }

    pub fn mean_closure(&self) -> f32 {
        let on: Vec<f32> = self.closure.iter().zip(&self.finger_max_n).filter(|(_, n)| **n > 1e-6).map(|(c, _)| *c).collect();
        let v = if on.is_empty() { &self.closure[..] } else { &on[..] };
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    }
}

fn box_closest(p: Vec3, c: Vec3, h: Vec3) -> (Vec3, Vec3) {
    let d = p - c;
    let q = Vec3::new(d.x.clamp(-h.x, h.x), d.y.clamp(-h.y, h.y), d.z.clamp(-h.z, h.z));
    let b = c + q;
    let out = p - b;
    if out.length() > 1e-7 {
        return (b, out.normalized());
    }
    // Inside: out through the nearest face (the phalanges never get here; a lump's inner box does).
    let gaps = [h.x - d.x.abs(), h.y - d.y.abs(), h.z - d.z.abs()];
    let k = (0..3).min_by(|&i, &j| gaps[i].total_cmp(&gaps[j])).unwrap_or(0);
    match k {
        0 => (Vec3::new(c.x + d.x.signum() * h.x, p.y, p.z), Vec3::new(d.x.signum(), 0.0, 0.0)),
        1 => (Vec3::new(p.x, c.y + d.y.signum() * h.y, p.z), Vec3::new(0.0, d.y.signum(), 0.0)),
        _ => (Vec3::new(p.x, p.y, c.z + d.z.signum() * h.z), Vec3::new(0.0, 0.0, d.z.signum())),
    }
}

fn phalanx_gaps(points: &[Vec3], c: Vec3, h: Vec3, shape: ShapeKind) -> Vec<(f32, f32)> {
    points
        .windows(2)
        .map(|w| {
            let gaps: Vec<(f32, f32)> = (0..PHALANX_SAMPLES)
                .map(|i| {
                    let s = i as f32 / (PHALANX_SAMPLES - 1) as f32;
                    let p = w[0] + (w[1] - w[0]).scale(s);
                    (shape_sdf(p, c, h, shape) - PHALANX_PAD_M, s)
                })
                .collect();
            let g = gaps.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
            let near: Vec<f32> = gaps.iter().filter(|x| x.0 <= g + 1e-4).map(|x| x.1).collect();
            let s = if near.is_empty() { 0.0 } else { 0.5 * (near[0] + near[near.len() - 1]) };
            (g, s)
        })
        .collect()
}

fn phalanx_point(finger: &FingerSpec, angles: &[JointAngles], k: usize, along: f32) -> Vec3 {
    let pts = finger.points(angles);
    pts[k] + (pts[k + 1] - pts[k]).scale(along)
}

fn phalanx_jacobian(finger: &FingerSpec, angles: &[JointAngles], k: usize, along: f32) -> Vec<Vec3> {
    let eps = 1e-3f32;
    (0..finger.joint_count())
        .map(|j| {
            if j > k {
                return Vec3::ZERO;
            }
            let mut a = angles.to_vec();
            let mut b = angles.to_vec();
            a[j].flex += eps;
            b[j].flex -= eps;
            (phalanx_point(finger, &a, k, along) - phalanx_point(finger, &b, k, along)).scale(0.5 / eps)
        })
        .collect()
}

pub fn finger_joint_torques(finger: &FingerSpec, angles: &[JointAngles], k: usize, along: f32, force_on_finger: Vec3) -> Vec<f32> {
    phalanx_jacobian(finger, angles, k.min(finger.joint_count().saturating_sub(1)), along).iter().map(|d| -d.dot(force_on_finger)).collect()
}

fn close_finger(finger: &FingerSpec, actuation: Actuation, c: Vec3, h: Vec3, shape: ShapeKind) -> (Vec<JointAngles>, Vec<(usize, f32)>) {
    let n = finger.joint_count();
    let touching = |angles: &[JointAngles]| -> Vec<(usize, f32, f32)> {
        phalanx_gaps(&finger.points(angles), c, h, shape).into_iter().enumerate().map(|(k, (g, s))| (k, g, s)).collect()
    };
    match actuation {
        Actuation::Coupled { .. } => {
            let mut prev = 0.0f32;
            let mut stop = 1.0f32;
            if touching(&closure_angles(finger, actuation, 0.0)).iter().any(|t| t.1 <= 0.0) {
                stop = 0.0;
            } else {
                for step in 1..=CLOSE_STEPS {
                    let u = step as f32 / CLOSE_STEPS as f32;
                    if touching(&closure_angles(finger, actuation, u)).iter().any(|t| t.1 <= 0.0) {
                        let (mut lo, mut hi) = (prev, u);
                        for _ in 0..24 {
                            let mid = 0.5 * (lo + hi);
                            if touching(&closure_angles(finger, actuation, mid)).iter().any(|t| t.1 <= 0.0) {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        stop = lo;
                        break;
                    }
                    prev = u;
                }
            }
            let angles = closure_angles(finger, actuation, stop);
            let hit = touching(&angles).into_iter().filter(|t| t.1 <= TOUCH_TOL_M && stop < 1.0).map(|t| (t.0, t.2)).collect();
            (angles, hit)
        }
        Actuation::Direct => {
            let mut frozen = vec![false; n];
            let mut angles = vec![JointAngles::default(); n];
            let target = |u: f32, j: usize| finger.joints[j].clamp_flex(u * finger.joints[j].flex_max);
            let with = |angles: &[JointAngles], frozen: &[bool], u: f32| -> Vec<JointAngles> {
                (0..n).map(|j| if frozen[j] { angles[j] } else { JointAngles { flex: target(u, j), ..Default::default() } }).collect()
            };
            let mut prev = 0.0f32;
            for step in 1..=CLOSE_STEPS {
                if frozen.iter().all(|f| *f) {
                    break;
                }
                let u = step as f32 / CLOSE_STEPS as f32;
                loop {
                    let cand = with(&angles, &frozen, u);
                    let first = touching(&cand).into_iter().filter(|t| t.1 <= 0.0 && !frozen[t.0]).map(|t| t.0).min();
                    let Some(k) = first else {
                        angles = cand;
                        break;
                    };
                    let (mut lo, mut hi) = (prev, u);
                    for _ in 0..24 {
                        let mid = 0.5 * (lo + hi);
                        if touching(&with(&angles, &frozen, mid)).iter().any(|t| t.0 == k && t.1 <= 0.0) {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    let at = with(&angles, &frozen, lo);
                    for j in 0..=k {
                        if !frozen[j] {
                            angles[j] = at[j];
                            frozen[j] = true;
                        }
                    }
                    for j in k + 1..n {
                        if !frozen[j] {
                            angles[j] = at[j];
                        }
                    }
                    if frozen.iter().all(|f| *f) {
                        break;
                    }
                }
                prev = u;
            }
            let hit = touching(&angles).into_iter().filter(|t| t.1 <= TOUCH_TOL_M).map(|t| (t.0, t.2)).collect();
            (angles, hit)
        }
    }
}

/// The simplex method on `max c·x` over `a·x ≤ b`, `x ≥ 0`, `b ≥ 0` (the origin is feasible), with Bland's rule so it
/// cannot cycle. Small dense problems only: a finger's contacts and joints.
fn simplex_max(a: &[Vec<f32>], b: &[f32], c: &[f32]) -> Vec<f32> {
    let (rows, cols) = (a.len(), c.len());
    let w = cols + rows + 1;
    let mut t = vec![vec![0.0f64; w]; rows + 1];
    for i in 0..rows {
        for j in 0..cols {
            t[i][j] = a[i][j] as f64;
        }
        t[i][cols + i] = 1.0;
        t[i][w - 1] = b[i].max(0.0) as f64;
    }
    for j in 0..cols {
        t[rows][j] = -(c[j] as f64);
    }
    let mut basis: Vec<usize> = (0..rows).map(|i| cols + i).collect();
    for _ in 0..64 * (rows + cols).max(1) {
        let Some(e) = (0..w - 1).find(|&j| t[rows][j] < -1e-12) else { break };
        let mut leave: Option<usize> = None;
        for i in 0..rows {
            if t[i][e] > 1e-12 {
                let r = t[i][w - 1] / t[i][e];
                let better = match leave {
                    None => true,
                    Some(l) => {
                        let rl = t[l][w - 1] / t[l][e];
                        r < rl - 1e-15 || (r <= rl + 1e-15 && basis[i] < basis[l])
                    }
                };
                if better {
                    leave = Some(i);
                }
            }
        }
        let Some(l) = leave else { break };
        let p = t[l][e];
        for v in t[l].iter_mut() {
            *v /= p;
        }
        let pivot = t[l].clone();
        for (i, row) in t.iter_mut().enumerate() {
            let f = row[e];
            if i != l && f != 0.0 {
                for (v, pv) in row.iter_mut().zip(&pivot) {
                    *v -= f * pv;
                }
            }
        }
        basis[l] = e;
    }
    let mut x = vec![0.0f32; cols];
    for (i, &v) in basis.iter().enumerate() {
        if v < cols {
            x[v] = t[i][w - 1].max(0.0) as f32;
        }
    }
    x
}

/// The most a direct-drive finger presses with all its contacts at once: the largest `Σλ` (λ ≥ 0) whose joint torques
/// `Σ_i λ_i·J_ij` stay inside every joint's rating, both ways. A contact no joint can press against gets nothing. The
/// v1.0 rule filled the farthest phalanx first, so its long lever could use up the knuckle before a nearer one pressed.
fn max_squeeze(jac: &[Vec<f32>], tau: &[f32]) -> Vec<f32> {
    let live: Vec<bool> = jac.iter().map(|j| j.iter().any(|v| *v > 1e-6)).collect();
    let mut a = Vec::with_capacity(2 * tau.len());
    let mut b = Vec::with_capacity(2 * tau.len());
    for (j, &t) in tau.iter().enumerate() {
        for s in [1.0f32, -1.0] {
            a.push(jac.iter().zip(&live).map(|(row, &on)| if on { s * row.get(j).copied().unwrap_or(0.0) } else { 0.0 }).collect());
            b.push(t.max(0.0));
        }
    }
    let c: Vec<f32> = live.iter().map(|&on| if on { 1.0 } else { 0.0 }).collect();
    simplex_max(&a, &b, &c)
}

pub fn top_grasp(hand: &Hand, size_m: [f32; 3]) -> TopGrasp {
    top_grasp_shaped(hand, size_m, ShapeKind::Box)
}

/// `top_grasp` on an item of the given shape, in the same bounding box: the phalanges stop on its surface, the
/// contacts take its normals, and the forces come from the same statics.
pub fn top_grasp_shaped(hand: &Hand, size_m: [f32; 3], shape: ShapeKind) -> TopGrasp {
    let actuation = hand.config.joints.actuation;
    let half = Vec3::new(size_m[0] * 0.5, size_m[1] * 0.5, size_m[2] * 0.5);
    let center = Vec3::new(0.0, PALM_PAD_M + half.y, -(PHALANX_PAD_M + FACE_GAP_M) - half.z);
    let mut out = TopGrasp { center, half, shape, angles: Vec::new(), closure: Vec::new(), contacts: Vec::new(), finger_max_n: Vec::new(), touching: 0 };
    for (fi, finger) in hand.fingers.iter().enumerate() {
        let (angles, hit) = close_finger(finger, actuation, center, half, shape);
        let q0_max = finger.joints.first().map_or(1.0, |j| j.flex_max.max(1e-6));
        out.closure.push(angles.first().map_or(0.0, |a| a.flex / q0_max));
        let mut cs: Vec<PhalanxContact> = hit
            .iter()
            .map(|&(k, along)| {
                let pts = finger.points(&angles);
                let p = pts[k] + (pts[k + 1] - pts[k]).scale(along);
                let (point, normal) = shape_closest(p, center, half, shape);
                let jac: Vec<f32> = phalanx_jacobian(finger, &angles, k, along).iter().map(|d| -normal.dot(*d)).collect();
                PhalanxContact { finger: fi, phalanx: k, point, normal, along, jac, max_n: 0.0 }
            })
            .collect();
        match actuation {
            Actuation::Coupled { .. } => {
                let c0 = out.closure[fi];
                let dc = if c0 < 0.99 { 0.01 } else { -0.01 };
                let (a, b) = (closure_angles(finger, actuation, c0), closure_angles(finger, actuation, c0 + dc));
                let dq0 = (b[0].flex - a[0].flex).abs().max(1e-6);
                let r: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (y.flex - x.flex).abs() / dq0).collect();
                let j_in: Vec<f32> = cs.iter().map(|c| c.jac.iter().zip(&r).map(|(j, r)| j * r).sum::<f32>()).collect();
                let s2: f32 = j_in.iter().filter(|j| **j > 1e-6).map(|j| j * j).sum();
                if s2 > 1e-12 {
                    let tau_in = finger.joints[0].max_torque;
                    let mut lam: Vec<f32> = j_in.iter().map(|j| if *j > 1e-6 { tau_in * j / s2 } else { 0.0 }).collect();
                    let mut scale = 1.0f32;
                    for (jj, spec) in finger.joints.iter().enumerate().skip(1) {
                        let t: f32 = cs.iter().zip(&lam).map(|(c, l)| l * c.jac[jj]).sum();
                        if t.abs() > spec.max_torque {
                            scale = scale.min(spec.max_torque / t.abs());
                        }
                    }
                    for (c, l) in cs.iter_mut().zip(lam.iter_mut()) {
                        *l *= scale;
                        c.max_n = *l;
                    }
                }
            }
            Actuation::Direct => {
                let jac: Vec<Vec<f32>> = cs.iter().map(|c| c.jac.clone()).collect();
                let tau: Vec<f32> = finger.joints.iter().map(|j| j.max_torque).collect();
                for (c, lam) in cs.iter_mut().zip(max_squeeze(&jac, &tau)) {
                    c.max_n = lam;
                }
            }
        }
        let total: f32 = cs.iter().map(|c| c.max_n).sum();
        if cs.iter().any(|c| c.max_n > 1e-6) {
            out.touching += 1;
        }
        out.finger_max_n.push(total);
        out.angles.push(angles);
        out.contacts.extend(cs);
    }
    out
}

#[derive(Clone, Debug)]
pub struct HandSim {
    hand: Hand,
    closure: Vec<f32>,
    angles: Vec<Vec<JointAngles>>,
}

impl HandSim {
    pub fn new(hand: Hand) -> Self {
        let n = hand.fingers.len();
        let angles = hand
            .fingers
            .iter()
            .map(|f| vec![JointAngles::default(); f.joint_count()])
            .collect();
        HandSim {
            hand,
            closure: vec![0.0; n],
            angles,
        }
    }

    pub fn hand(&self) -> &Hand {
        &self.hand
    }

    pub fn set_closure(&mut self, closure: &[f32]) {
        for (dst, src) in self.closure.iter_mut().zip(closure) {
            *dst = src.clamp(0.0, 1.0);
        }
    }

    pub fn closure(&self) -> &[f32] {
        &self.closure
    }

    pub fn step(&mut self, dt: f32) {
        let actuation = self.hand.config.joints.actuation;
        for ((finger, angles), &c) in self
            .hand
            .fingers
            .iter()
            .zip(self.angles.iter_mut())
            .zip(&self.closure)
        {
            let targets = closure_angles(finger, actuation, c);
            for ((joint, a), t) in finger.joints.iter().zip(angles.iter_mut()).zip(&targets) {
                let max_delta = joint.max_speed * dt;
                let delta = (t.flex - a.flex).clamp(-max_delta, max_delta);
                a.flex = joint.clamp_flex(a.flex + delta);
            }
        }
    }

    pub fn angles(&self) -> &[Vec<JointAngles>] {
        &self.angles
    }

    pub fn points(&self) -> Vec<Vec<Vec3>> {
        fk_all(&self.hand, &self.angles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::{FingerCount, NotoConfig};

    fn box_payload(mass_kg: f32) -> Payload {
        Payload {
            name: "test carton".into(),
            size_m: [0.08, 0.06, 0.06],
            mass_kg,
            friction: 0.6,
        }
    }

    fn build(count: FingerCount) -> Hand {
        NotoConfig::preset(count).build().unwrap()
    }

    #[test]
    fn box_sdf_is_negative_inside_and_positive_outside() {
        let h = Vec3::new(0.5, 0.5, 0.5);
        assert!(box_sdf(Vec3::ZERO, Vec3::ZERO, h) < 0.0);
        assert!((box_sdf(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, h) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn coupled_distal_joints_follow_ratio() {
        let hand = build(FingerCount::Five);
        let f = &hand.fingers[1];
        let a = closure_angles(f, Actuation::Coupled { ratio: 0.5 }, 1.0);
        let expected = 0.5 * a[0].flex;
        assert!((a[1].flex - expected).abs() < 1e-6);
    }

    #[test]
    fn closure_angles_respect_joint_limits() {
        let hand = build(FingerCount::Four);
        for f in &hand.fingers {
            for c in [0.0, 0.5, 1.0] {
                for (j, a) in f
                    .joints
                    .iter()
                    .zip(closure_angles(f, Actuation::Direct, c))
                {
                    assert!(a.flex >= j.flex_min && a.flex <= j.flex_max + 1e-6);
                }
            }
        }
    }

    #[test]
    fn five_finger_hand_holds_a_light_carton() {
        let hand = build(FingerCount::Five);
        let out = grasp(&hand, &box_payload(0.5), &GraspSetup::default());
        let r = &out.result;
        assert!(r.contacts.len() >= 2, "contacts: {}", r.contacts.len());
        assert!(r.stable, "capacity {} N vs required {} N", r.friction_capacity_n, r.required_n);
    }

    #[test]
    fn heavy_object_exceeds_capacity() {
        let hand = build(FingerCount::Two);
        let out = grasp(&hand, &box_payload(50.0), &GraspSetup::default());
        assert!(!out.result.stable);
        assert!(out.result.max_payload_kg < 50.0);
    }

    #[test]
    fn grasp_outcome_angles_stay_within_limits() {
        let hand = build(FingerCount::Three);
        let out = grasp(&hand, &box_payload(0.5), &GraspSetup::default());
        for (f, a) in hand.fingers.iter().zip(&out.angles) {
            for (j, x) in f.joints.iter().zip(a) {
                assert!(x.flex >= j.flex_min - 1e-6 && x.flex <= j.flex_max + 1e-6);
            }
        }
    }

    #[test]
    fn fingers_stop_on_the_box_with_face_normals_and_jt_forces() {
        for count in FingerCount::ALL {
            let hand = build(count);
            for size in [0.06f32, 0.10, 0.15] {
                let g = top_grasp(&hand, [size; 3]);
                assert_eq!(g.angles.len(), hand.fingers.len());
                assert!(g.touching >= 2, "{count:?} {size}: {} touching", g.touching);
                for c in &g.contacts {
                    assert!(box_sdf(c.point, g.center, g.half).abs() < 1e-4, "{c:?}");
                    assert!((c.normal.length() - 1.0).abs() < 1e-4);
                    assert!((c.point + c.normal.scale(0.01) - g.center).length() > (c.point - g.center).length());
                    assert!(c.max_n >= 0.0 && c.max_n < 200.0, "{c:?}");
                }
                for (f, a) in hand.fingers.iter().zip(&g.angles) {
                    for (gap, _) in phalanx_gaps(&f.points(a), g.center, g.half, g.shape) {
                        assert!(gap > -1e-3, "{count:?} {size}: {gap}");
                    }
                }
                assert!(g.squeeze_max_n() > 1.0);
            }
        }
    }

    const SHAPES: [ShapeKind; 3] = [ShapeKind::Box, ShapeKind::Cylinder { axis: 0 }, ShapeKind::Lump { round: 0.5 }];

    #[test]
    fn every_shape_has_a_signed_distance_its_nearest_point_and_normal_agree_with() {
        let (c, h) = (Vec3::new(0.01, 0.05, -0.06), Vec3::new(0.12, 0.05, 0.04));
        let mut d = crate::act::Draw::new(9);
        for shape in SHAPES {
            assert!(shape_sdf(c, c, h, shape) < 0.0, "{shape:?}: the centre is inside");
            for _ in 0..400 {
                let p = c + Vec3::new(0.3 * (d.uniform() - 0.5), 0.2 * (d.uniform() - 0.5), 0.2 * (d.uniform() - 0.5));
                let s = shape_sdf(p, c, h, shape);
                let (q, n) = shape_closest(p, c, h, shape);
                assert!(shape_sdf(q, c, h, shape).abs() < 2e-5, "{shape:?} {p:?}: the nearest point is on the surface");
                assert!((n.length() - 1.0).abs() < 1e-4, "{shape:?}: unit normal");
                assert!((p.distance(q) - s.abs()).abs() < 2e-5, "{shape:?} {p:?}: |sdf| {s} is the distance to it {}", p.distance(q));
                if s > 1e-4 {
                    assert!((q + n.scale(s)).distance(p) < 2e-5, "{shape:?} {p:?}: outside, the point is along the normal");
                }
                let e = 1e-3;
                let g = Vec3::new(
                    shape_sdf(p + Vec3::new(e, 0.0, 0.0), c, h, shape) - shape_sdf(p - Vec3::new(e, 0.0, 0.0), c, h, shape),
                    shape_sdf(p + Vec3::new(0.0, e, 0.0), c, h, shape) - shape_sdf(p - Vec3::new(0.0, e, 0.0), c, h, shape),
                    shape_sdf(p + Vec3::new(0.0, 0.0, e), c, h, shape) - shape_sdf(p - Vec3::new(0.0, 0.0, e), c, h, shape),
                )
                .scale(0.5 / e);
                if s > 2e-3 {
                    assert!(g.dot(n) > 0.99, "{shape:?} {p:?}: the normal is the gradient {g:?} vs {n:?}");
                }
            }
        }
        let lump = ShapeKind::Lump { round: 1.0 };
        let cube = Vec3::new(0.05, 0.05, 0.05);
        assert!((shape_sdf(Vec3::new(0.0, 0.0, 0.1), Vec3::ZERO, cube, lump) - 0.05).abs() < 1e-6, "a fully round cube is a ball");
        assert!((shape_sdf(Vec3::new(0.1, 0.1, 0.0), Vec3::ZERO, cube, ShapeKind::Cylinder { axis: 2 }) - (0.1f32.hypot(0.1) - 0.05)).abs() < 1e-6);
    }

    #[test]
    fn fingers_close_on_round_items_without_going_in_and_a_log_hooks_under() {
        for count in FingerCount::ALL {
            let hand = build(count);
            for (shape, size) in [(ShapeKind::Cylinder { axis: 0 }, [0.50f32, 0.14, 0.14]), (ShapeKind::Cylinder { axis: 0 }, [0.15; 3]), (ShapeKind::Lump { round: 0.5 }, [0.10; 3])] {
                let g = top_grasp_shaped(&hand, size, shape);
                assert_eq!(g.shape, shape);
                assert!(g.touching >= 2, "{count:?} {shape:?} {size:?}: {} touching", g.touching);
                for c in &g.contacts {
                    assert!(shape_sdf(c.point, g.center, g.half, shape).abs() < 1e-4, "{c:?}");
                    assert!(c.max_n >= 0.0 && c.max_n < 200.0, "{c:?}");
                    if let ShapeKind::Cylinder { .. } = shape {
                        assert!(c.normal.x.abs() < 1e-3, "the side of a log faces away from its axis: {c:?}");
                    }
                }
                for (f, a) in hand.fingers.iter().zip(&g.angles) {
                    for (gap, _) in phalanx_gaps(&f.points(a), g.center, g.half, shape) {
                        assert!(gap > -1e-3, "{count:?} {shape:?}: {gap}");
                    }
                }
            }
        }
        // A 14 cm log is wider than a finger can get round: the fingertips meet it at its widest, as they meet the box's
        // far face, and stop there. A 4 cm bar is narrow enough for them to go on underneath and hang it on the tips.
        let five = build(FingerCount::Five);
        let hook = |g: &TopGrasp| g.contacts.iter().map(|c| c.max_n * c.normal.y.max(0.0)).sum::<f32>();
        let log = top_grasp_shaped(&five, [0.50, 0.14, 0.14], ShapeKind::Cylinder { axis: 0 });
        let flat = top_grasp(&five, [0.50, 0.14, 0.14]);
        assert!(hook(&log) < 0.05 * log.squeeze_max_n() && (log.squeeze_max_n() / flat.squeeze_max_n() - 1.0).abs() < 0.05, "{} vs {} N, hook {}", log.squeeze_max_n(), flat.squeeze_max_n(), hook(&log));
        assert!((log.mean_closure() - flat.mean_closure()).abs() < 0.02);
        let bar = top_grasp_shaped(&five, [0.30, 0.04, 0.04], ShapeKind::Cylinder { axis: 0 });
        assert!(hook(&bar) > 0.5 * bar.squeeze_max_n() && bar.mean_closure() > log.mean_closure() + 0.1, "the tips go under a thin bar: hook {} of {} N", hook(&bar), bar.squeeze_max_n());
    }

    #[test]
    fn a_direct_finger_presses_with_every_contact_inside_every_joint_rating() {
        let mut cfg = NotoConfig::preset(FingerCount::Five);
        cfg.joints.actuation = Actuation::Direct;
        for hand in [build(FingerCount::Two), cfg.build().unwrap()] {
            for shape in SHAPES {
                for size in [[0.10f32; 3], [0.50, 0.14, 0.14]] {
                    let g = top_grasp_shaped(&hand, size, shape);
                    for (fi, f) in hand.fingers.iter().enumerate() {
                        let cs: Vec<&PhalanxContact> = g.contacts.iter().filter(|c| c.finger == fi).collect();
                        for (j, spec) in f.joints.iter().enumerate() {
                            let tau: f32 = cs.iter().map(|c| c.max_n * c.jac[j]).sum();
                            assert!(tau.abs() <= spec.max_torque + 1e-4, "{shape:?} {size:?} finger {fi} joint {j}: {tau} > {}", spec.max_torque);
                        }
                        // What the v1.0 rule (farthest contact first) pressed: never more than the optimum.
                        let mut room: Vec<f32> = f.joints.iter().map(|j| j.max_torque).collect();
                        let mut order: Vec<&&PhalanxContact> = cs.iter().collect();
                        order.sort_by(|a, b| b.phalanx.cmp(&a.phalanx));
                        let mut old = 0.0f32;
                        for c in order {
                            let lam = (0..=c.phalanx).filter(|&j| c.jac[j] > 1e-6).map(|j| room[j].max(0.0) / c.jac[j]).fold(f32::INFINITY, f32::min);
                            let lam = if lam.is_finite() { lam.max(0.0) } else { 0.0 };
                            for j in 0..=c.phalanx {
                                room[j] -= lam * c.jac[j];
                            }
                            old += lam;
                        }
                        assert!(g.finger_max_n[fi] >= old - 1e-3, "{shape:?} {size:?} finger {fi}: {} < {old}", g.finger_max_n[fi]);
                    }
                }
            }
        }
    }

    #[test]
    fn joint_torques_follow_the_contact_statics() {
        let hand = build(FingerCount::Five);
        let g = top_grasp(&hand, [0.10; 3]);
        let c = g.contacts.iter().max_by(|a, b| a.max_n.total_cmp(&b.max_n)).unwrap();
        let f = &hand.fingers[c.finger];
        let tau = finger_joint_torques(f, &g.angles[c.finger], c.phalanx, c.along, c.normal.scale(2.0));
        for (j, t) in tau.iter().enumerate() {
            assert!((t - 2.0 * c.jac[j]).abs() < 1e-3, "{j}: {t} vs {}", 2.0 * c.jac[j]);
        }
    }

    #[test]
    #[ignore]
    fn print_top_grasps() {
        for count in FingerCount::ALL {
            let hand = build(count);
            for size in [0.06f32, 0.079, 0.10, 0.15] {
                let g = top_grasp(&hand, [size; 3]);
                let old = grasp(&hand, &Payload { name: "o".into(), size_m: [size; 3], mass_kg: 1.0, friction: 0.6 }, &GraspSetup::default());
                let old_n: f32 = old.result.contacts.iter().filter(|c| matches!(c.source, ContactSource::Finger { .. })).map(|c| c.force_n).sum();
                println!("[TOP] {count:?} {size}: touching {} squeeze {:.1} N (sim::grasp {:.1} N) closure {:?} per finger {:?} phalanges {:?}", g.touching, g.squeeze_max_n(), old_n, g.closure.iter().map(|c| (c * 100.0).round()).collect::<Vec<_>>(), g.finger_max_n.iter().map(|n| (n * 10.0).round() / 10.0).collect::<Vec<_>>(), g.contacts.iter().map(|c| (c.finger, c.phalanx, (c.normal.x * 10.0).round(), (c.normal.y * 10.0).round(), (c.normal.z * 10.0).round())).collect::<Vec<_>>());
            }
            for (shape, size) in [(ShapeKind::Cylinder { axis: 0 }, [0.50f32, 0.14, 0.14]), (ShapeKind::Cylinder { axis: 0 }, [0.15; 3]), (ShapeKind::Cylinder { axis: 0 }, [0.30, 0.06, 0.06]), (ShapeKind::Cylinder { axis: 0 }, [0.30, 0.04, 0.04]), (ShapeKind::Lump { round: 0.5 }, [0.10; 3]), (ShapeKind::Lump { round: 0.5 }, [0.06; 3])] {
                let g = top_grasp_shaped(&hand, size, shape);
                let hook: f32 = g.contacts.iter().map(|c| c.max_n * c.normal.y.max(0.0)).sum();
                println!("[TOP-SHAPE] {count:?} {} {size:?}: touching {} squeeze {:.1} N hook {:.1} N closure {:?} per finger {:?} phalanges {:?}", shape.id(), g.touching, g.squeeze_max_n(), hook, g.closure.iter().map(|c| (c * 100.0).round()).collect::<Vec<_>>(), g.finger_max_n.iter().map(|n| (n * 10.0).round() / 10.0).collect::<Vec<_>>(), g.contacts.iter().map(|c| (c.finger, c.phalanx, (c.normal.y * 10.0).round(), (c.normal.z * 10.0).round(), (c.max_n * 10.0).round() / 10.0)).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn hand_sim_moves_toward_command_without_overshoot() {
        let mut sim = HandSim::new(build(FingerCount::Five));
        sim.set_closure(&[1.0; 5]);
        sim.step(0.01);
        let finger = sim.hand().fingers[0].clone();
        let actuation = sim.hand().config.joints.actuation;
        let first = sim.angles()[0][0].flex;
        assert!(first > 0.0 && first <= finger.joints[0].max_speed * 0.01 + 1e-6);
        for _ in 0..1000 {
            sim.step(0.01);
        }
        let target = closure_angles(&finger, actuation, 1.0);
        assert!((sim.angles()[0][0].flex - target[0].flex).abs() < 1e-5);
    }
}
