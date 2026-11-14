//! Stage 1 of the two-stream hand estimate: the hand's pose from observed joint points, on the hand model's own
//! kinematics. The unknowns are the wrist (a rigid transform of the hand frame) and every finger's joint angles; the
//! hand's size (β, here the body scale k the hand was built at) is known and fixed for the episode. The joint kinds are
//! the hard constraints — a hinge has no abduction and no joint has twist (`FingerSpec::joint_rotation` ignores them),
//! the knuckle's universal joint flexes and abducts — and every step is projected into the joint limits. A smoothness
//! term ties the angles to the previous frames (a second difference with two of them, a first with one).
//!
//! The result is frozen for Stage 2 (`crate::tactile`): the contact stream reads the fitted pose and never moves it.

use serde::Serialize;

use crate::fit3::{align_rigid, Similarity, M3, V3};
use crate::hand::Hand;
use crate::joint::JointAngles;
use crate::math::Vec3;

/// What the cameras saw of one hand: per finger, each FK point (`FingerSpec::points`: the knuckle, then the end of
/// every phalanx) in world coordinates, `None` where no view saw it; optionally each point's information matrix
/// (inverse covariance, 1/m²) — without one a point counts as a 1 cm isotropic measurement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HandObs {
    pub points: Vec<Vec<Option<V3>>>,
    pub info: Vec<Vec<Option<M3>>>,
}

impl HandObs {
    pub fn seen(&self) -> usize {
        self.points.iter().flatten().filter(|p| p.is_some()).count()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HandFitCfg {
    /// Weight of the angle smoothness term against the point residuals (rad against m).
    pub smooth: f64,
    pub iters: usize,
}

impl Default for HandFitCfg {
    fn default() -> Self {
        HandFitCfg { smooth: 0.004, iters: 40 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HandFit {
    /// Hand frame → world.
    pub wrist: Similarity,
    pub angles: Vec<Vec<JointAngles>>,
    /// RMS distance of the observed points to the fitted ones, m.
    pub rms_m: f64,
}

impl HandFit {
    /// Every FK point of every finger in world coordinates.
    pub fn points(&self, hand: &Hand) -> Vec<Vec<V3>> {
        model_points(hand, &self.angles, &self.wrist)
    }
}

fn v64(v: Vec3) -> V3 {
    [v.x as f64, v.y as f64, v.z as f64]
}

pub fn model_points(hand: &Hand, angles: &[Vec<JointAngles>], wrist: &Similarity) -> Vec<Vec<V3>> {
    hand.fingers.iter().zip(angles).map(|(f, a)| f.points(a).into_iter().map(|p| wrist.apply(v64(p))).collect()).collect()
}

/// The parameters of one fit: per finger, per joint, flex then (on a joint with abduction) abd; then the wrist's
/// rotation increment (a rotation vector) and translation.
struct Layout {
    slots: Vec<(usize, usize, bool)>,
}

impl Layout {
    fn of(hand: &Hand) -> Layout {
        let mut slots = Vec::new();
        for (fi, f) in hand.fingers.iter().enumerate() {
            for (ji, j) in f.joints.iter().enumerate() {
                slots.push((fi, ji, false));
                if j.kind.has_abduction() {
                    slots.push((fi, ji, true));
                }
            }
        }
        Layout { slots }
    }

    fn angles(&self, x: &[f64], hand: &Hand) -> Vec<Vec<JointAngles>> {
        let mut out: Vec<Vec<JointAngles>> = hand.fingers.iter().map(|f| vec![JointAngles::default(); f.joint_count()]).collect();
        for (k, &(fi, ji, abd)) in self.slots.iter().enumerate() {
            if abd {
                out[fi][ji].abd = x[k] as f32;
            } else {
                out[fi][ji].flex = x[k] as f32;
            }
        }
        out
    }

    fn pack(&self, angles: &[Vec<JointAngles>]) -> Vec<f64> {
        self.slots.iter().map(|&(fi, ji, abd)| angles.get(fi).and_then(|a| a.get(ji)).map_or(0.0, |a| if abd { a.abd } else { a.flex }) as f64).collect()
    }

    fn clamp(&self, x: &mut [f64], hand: &Hand) {
        for (k, &(fi, ji, abd)) in self.slots.iter().enumerate() {
            let s = &hand.fingers[fi].joints[ji];
            x[k] = if abd { x[k].clamp(-s.abd_limit as f64, s.abd_limit as f64) } else { x[k].clamp(s.flex_min as f64, s.flex_max as f64) };
        }
    }
}

fn rot_exp(w: V3) -> [[f64; 3]; 3] {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if th < 1e-12 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = [w[0] / th, w[1] / th, w[2] / th];
    let (s, c) = th.sin_cos();
    let v = 1.0 - c;
    [
        [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
        [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
        [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v],
    ]
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// The wrist after the increment (rotation vector, translation) in the last six parameters.
fn wrist_of(base: &Similarity, x: &[f64]) -> Similarity {
    let n = x.len();
    let w = [x[n - 6], x[n - 5], x[n - 4]];
    Similarity { s: base.s, r: matmul(&rot_exp(w), &base.r), t: [base.t[0] + x[n - 3], base.t[1] + x[n - 2], base.t[2] + x[n - 1]] }
}

/// The reference measurement a residual is scaled to: a point with this isotropic σ weighs as an unweighted one.
const SIGMA_REF_M: f64 = 0.01;

/// The upper Cholesky factor U of an information matrix W scaled by σ_ref² (W·σ_ref² = UᵀU), so that U·e is the
/// whitened residual in metres of a 1 cm measurement.
fn whitener(w: &M3) -> Option<M3> {
    let a: M3 = [0, 1, 2].map(|i| [0, 1, 2].map(|j| w[i][j] * SIGMA_REF_M * SIGMA_REF_M));
    let l00 = a[0][0].sqrt();
    if !(l00 > 0.0) {
        return None;
    }
    let l10 = a[1][0] / l00;
    let l20 = a[2][0] / l00;
    let l11 = (a[1][1] - l10 * l10).sqrt();
    if !(l11 > 0.0) {
        return None;
    }
    let l21 = (a[2][1] - l20 * l10) / l11;
    let l22 = (a[2][2] - l20 * l20 - l21 * l21).sqrt();
    if !(l22 > 0.0) {
        return None;
    }
    // W = L·Lᵀ with L lower; U = Lᵀ.
    Some([[l00, l10, l20], [0.0, l11, l21], [0.0, 0.0, l22]])
}

fn residuals(hand: &Hand, obs: &HandObs, lay: &Layout, base: &Similarity, x: &[f64], prev: &[Vec<f64>], cfg: &HandFitCfg) -> Vec<f64> {
    let pts = model_points(hand, &lay.angles(x, hand), &wrist_of(base, x));
    let mut r = Vec::new();
    for (fi, (fo, fm)) in obs.points.iter().zip(&pts).enumerate() {
        for (k, (o, m)) in fo.iter().zip(fm).enumerate() {
            if let Some(o) = o {
                let e = [m[0] - o[0], m[1] - o[1], m[2] - o[2]];
                match obs.info.get(fi).and_then(|f| f.get(k)).copied().flatten().and_then(|w| whitener(&w)) {
                    Some(u) => r.extend((0..3).map(|i| (0..3).map(|j| u[i][j] * e[j]).sum::<f64>())),
                    None => r.extend_from_slice(&e),
                }
            }
        }
    }
    let w = cfg.smooth.sqrt();
    let n = lay.slots.len();
    match prev {
        [.., a, b] => r.extend((0..n).map(|k| w * (x[k] - 2.0 * b[k] + a[k]))),
        [b] => r.extend((0..n).map(|k| w * (x[k] - b[k]))),
        [] => {}
    }
    r
}

/// Solves (A + μ·diag A) δ = g for a small dense symmetric A by Gaussian elimination.
fn solve(a: &[Vec<f64>], g: &[f64], mu: f64) -> Option<Vec<f64>> {
    let n = g.len();
    let mut m: Vec<Vec<f64>> = (0..n).map(|i| (0..=n).map(|j| if j == n { g[i] } else { a[i][j] + if i == j { mu * a[i][i].max(1e-9) } else { 0.0 } }).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-18 {
            return None;
        }
        m.swap(c, p);
        for i in 0..n {
            if i != c {
                let f = m[i][c] / m[c][c];
                if f != 0.0 {
                    for j in c..=n {
                        m[i][j] -= f * m[c][j];
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// Fits the hand's pose to `obs`. `scale` is the hand's size against its model (the body's k); `prev` the last fits
/// (oldest first, at most two used) for the smoothness term and the starting point; without one the wrist starts from
/// the knuckles the cameras saw (at least three), else from `wrist_guess`.
pub fn fit_hand(hand: &Hand, obs: &HandObs, scale: f64, prev: &[HandFit], wrist_guess: Option<Similarity>, cfg: &HandFitCfg) -> Option<HandFit> {
    if obs.seen() < 4 {
        return None;
    }
    let lay = Layout::of(hand);
    let prev_x: Vec<Vec<f64>> = prev.iter().rev().take(2).rev().map(|p| lay.pack(&p.angles)).collect();
    let base = match prev.last() {
        Some(p) => p.wrist,
        None => {
            let (src, dst): (Vec<V3>, Vec<V3>) = hand
                .fingers
                .iter()
                .zip(&obs.points)
                .filter_map(|(f, o)| o.first().copied().flatten().map(|q| (v64(f.mount).map(|v| v * scale), q)))
                .unzip();
            match (src.len() >= 3).then(|| align_rigid(&src, &dst)).flatten() {
                Some(r) => Similarity { s: scale, r: r.r, t: r.t },
                None => wrist_guess?,
            }
        }
    };
    let mut x: Vec<f64> = match prev_x.last() {
        Some(p) => p.clone(),
        None => {
            // A third closed: the middle of the usual range, so that no finger starts on a limit.
            let mut x0 = lay.pack(&hand.fingers.iter().map(|f| f.joints.iter().map(|j| JointAngles { flex: 0.35 * j.flex_max, ..JointAngles::default() }).collect()).collect::<Vec<_>>());
            lay.clamp(&mut x0, hand);
            x0
        }
    };
    x.extend_from_slice(&[0.0; 6]);
    let n = x.len();
    let cost = |x: &[f64]| residuals(hand, obs, &lay, &base, x, &prev_x, cfg).iter().map(|v| v * v).sum::<f64>();
    let mut cur = cost(&x);
    let mut mu = 1e-3;
    for _ in 0..cfg.iters {
        let r0 = residuals(hand, obs, &lay, &base, &x, &prev_x, cfg);
        let eps = 1e-5;
        let cols: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let (mut a, mut b) = (x.clone(), x.clone());
                a[k] += eps;
                b[k] -= eps;
                let (ra, rb) = (residuals(hand, obs, &lay, &base, &a, &prev_x, cfg), residuals(hand, obs, &lay, &base, &b, &prev_x, cfg));
                ra.iter().zip(&rb).map(|(p, q)| (p - q) / (2.0 * eps)).collect()
            })
            .collect();
        let jtj: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum()).collect()).collect();
        let g: Vec<f64> = (0..n).map(|i| -cols[i].iter().zip(&r0).map(|(a, b)| a * b).sum::<f64>()).collect();
        let mut improved = false;
        for _ in 0..8 {
            let Some(d) = solve(&jtj, &g, mu) else { break };
            let mut cand: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + b).collect();
            let m = lay.slots.len();
            lay.clamp(&mut cand[..m], hand);
            let c = cost(&cand);
            if c < cur {
                let gain = cur - c;
                x = cand;
                cur = c;
                mu = (mu * 0.3).max(1e-9);
                improved = gain > 1e-14;
                break;
            }
            mu *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let wrist = wrist_of(&base, &x);
    let angles = lay.angles(&x, hand);
    let fitted = model_points(hand, &angles, &wrist);
    let (mut se, mut k) = (0.0, 0usize);
    for (fo, fm) in obs.points.iter().zip(&fitted) {
        for (o, m) in fo.iter().zip(fm) {
            if let Some(o) = o {
                se += (0..3).map(|i| (m[i] - o[i]).powi(2)).sum::<f64>();
                k += 1;
            }
        }
    }
    Some(HandFit { wrist, angles, rms_m: (se / k.max(1) as f64).sqrt() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::{FingerCount, NotoConfig};

    fn pose(closure: f32, hand: &Hand) -> Vec<Vec<JointAngles>> {
        hand.fingers
            .iter()
            .enumerate()
            .map(|(i, f)| f.joints.iter().enumerate().map(|(j, s)| JointAngles { flex: closure * s.flex_max * (1.0 - 0.1 * j as f32), abd: if s.kind.has_abduction() { 0.1 * (i as f32 - 2.0) * s.abd_limit / 2.0 } else { 0.0 }, roll: 0.0 }).collect())
            .collect()
    }

    fn wrist() -> Similarity {
        let (c, s) = (0.4f64.cos(), 0.4f64.sin());
        Similarity { s: 1.0, r: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], t: [1.0, 0.8, -0.5] }
    }

    #[test]
    fn exact_points_give_back_the_pose_and_hidden_points_are_filled_by_the_model() {
        let hand = NotoConfig::preset(FingerCount::Five).build().unwrap();
        let truth = pose(0.5, &hand);
        let pts = model_points(&hand, &truth, &wrist());
        let obs = HandObs { points: pts.iter().map(|f| f.iter().map(|p| Some(*p)).collect()).collect(), info: Vec::new() };
        let fit = fit_hand(&hand, &obs, 1.0, &[], None, &HandFitCfg::default()).unwrap();
        assert!(fit.rms_m < 1e-4, "{}", fit.rms_m);
        for (a, b) in fit.angles.iter().flatten().zip(truth.iter().flatten()) {
            assert!((a.flex - b.flex).abs() < 0.01 && (a.abd - b.abd).abs() < 0.01, "{a:?} vs {b:?}");
        }
        // The fingertips hidden (an object in front of them): the model still places them.
        let hidden = HandObs { points: pts.iter().map(|f| f.iter().enumerate().map(|(k, p)| (k + 1 < f.len()).then_some(*p)).collect()).collect(), info: Vec::new() };
        let fit = fit_hand(&hand, &hidden, 1.0, &[], None, &HandFitCfg::default()).unwrap();
        let tips_err: f64 = fit.points(&hand).iter().zip(&pts).map(|(a, b)| (0..3).map(|i| (a.last().unwrap()[i] - b.last().unwrap()[i]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
        assert!(tips_err < 0.01, "hidden tips {tips_err} m off");
    }

    #[test]
    fn a_point_measured_badly_along_one_axis_pulls_only_where_it_is_sure() {
        let hand = NotoConfig::preset(FingerCount::Five).build().unwrap();
        let truth = pose(0.5, &hand);
        let pts = model_points(&hand, &truth, &wrist());
        let mut bad = pts.clone();
        // One fingertip read 5 cm off along x, by a view whose depth (x) error is 10 cm and whose lateral error 1 mm.
        bad[2].last_mut().unwrap()[0] += 0.05;
        let w: M3 = [[1.0 / 0.01, 0.0, 0.0], [0.0, 1e6, 0.0], [0.0, 0.0, 1e6]];
        let info: Vec<Vec<Option<M3>>> = bad.iter().enumerate().map(|(fi, f)| f.iter().enumerate().map(|(k, _)| (fi == 2 && k + 1 == f.len()).then_some(w)).collect()).collect();
        let tip = |fit: &HandFit| {
            let a = fit.points(&hand)[2].last().copied().unwrap();
            let b = pts[2].last().copied().unwrap();
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let plain = HandObs { points: bad.iter().map(|f| f.iter().map(|p| Some(*p)).collect()).collect(), info: Vec::new() };
        let weighted = HandObs { info, ..plain.clone() };
        let (a, b) = (fit_hand(&hand, &plain, 1.0, &[], None, &HandFitCfg::default()).unwrap(), fit_hand(&hand, &weighted, 1.0, &[], None, &HandFitCfg::default()).unwrap());
        assert!(tip(&b) < 0.5 * tip(&a), "weighted {:.4} m against unweighted {:.4} m", tip(&b), tip(&a));
    }

    #[test]
    fn the_joint_limits_hold_however_the_points_pull() {
        let hand = NotoConfig::preset(FingerCount::Three).build().unwrap();
        let truth = pose(0.4, &hand);
        let mut pts = model_points(&hand, &truth, &wrist());
        // A fingertip seen far behind the hand would need the finger bent backwards.
        pts[1].last_mut().unwrap()[1] += 0.3;
        let obs = HandObs { points: pts.iter().map(|f| f.iter().map(|p| Some(*p)).collect()).collect(), info: Vec::new() };
        let fit = fit_hand(&hand, &obs, 1.0, &[], None, &HandFitCfg::default()).unwrap();
        for (f, a) in hand.fingers.iter().zip(&fit.angles) {
            for (s, q) in f.joints.iter().zip(a) {
                assert!(q.flex >= s.flex_min - 1e-6 && q.flex <= s.flex_max + 1e-6);
                assert!(q.abd.abs() <= s.abd_limit + 1e-6);
                assert_eq!(q.roll, 0.0, "no twist");
                if !s.kind.has_abduction() {
                    assert_eq!(q.abd, 0.0, "a hinge does not abduct");
                }
            }
        }
        assert!(fit_hand(&hand, &HandObs { points: vec![vec![None; 4]; 3], info: Vec::new() }, 1.0, &[], None, &HandFitCfg::default()).is_none());
    }
}
