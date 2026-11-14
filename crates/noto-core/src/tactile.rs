//! Stage 2 of the two-stream hand estimate: on the hand pose Stage 1 fixed (`crate::handfit`, frozen — nothing here
//! moves it), which phalanx pads touch the item and what they carry. A pad's gap is the item's signed distance at the
//! phalanx's axis less the pad's radius (`sim::PHALANX_PAD_M`, bone and the 4.5 mm elastic layer); its contact
//! probability is a logistic of that gap, lowered by the pad's speed against the item (a pad sliding over a surface is
//! not holding it). The forces are the pad model's (`act::grasp::PadState`: the grip shared by the closing statics,
//! the friction ellipse per pad), weighted by the contact probability, and a hand's total drops to 0 the moment no pad
//! touches (the release gate).

use serde::{Deserialize, Serialize};

use crate::act::material::{PAD_E_PA, PAD_NU, PAD_RADIUS_M, PAD_THICKNESS_M};
use crate::contact::{FingerContacts, MAX_ACTIVE_PADS};
use crate::hand::Hand;
use crate::joint::JointAngles;
use crate::math::Vec3;
use crate::sim::{shape_sdf, ShapeKind, PHALANX_PAD_M};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TactileCfg {
    /// The gap at which a pad touches with probability ½, m (positive: within this much of the surface counts).
    pub touch_m: f32,
    /// The logistic's width in gap, m.
    pub soft_m: f32,
    /// The pad speed against the item that halves the logit's distance from touching, m/s.
    pub slip_mps: f32,
}

impl Default for TactileCfg {
    fn default() -> Self {
        TactileCfg { touch_m: 0.0015, soft_m: 0.0006, slip_mps: 0.05 }
    }
}

/// One phalanx against the item, in the hand frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PadGeom {
    pub finger: usize,
    pub phalanx: usize,
    /// Signed gap of the pad surface to the item, m (negative: pressed in).
    pub gap_m: f32,
    /// The pad's elastic indentation from the overlap, finite-thickness saturated: h·(1 − e^(−δ/h)).
    pub indentation_m: f32,
    /// The point on the phalanx's axis nearest the item.
    pub at: Vec3,
}

const AXIS_SAMPLES: usize = 9;

/// Every phalanx's gap to the item (centre `c`, half sizes `h`, `shape`, all in the hand frame) at the pose `angles`.
pub fn pad_geometry(hand: &Hand, angles: &[Vec<JointAngles>], c: Vec3, h: Vec3, shape: ShapeKind) -> Vec<PadGeom> {
    pad_geometry_mapped(hand, angles, &|p| p, c, h, shape)
}

/// `pad_geometry` with the hand's points carried by `to_item` into the frame the item is given in (an item turned
/// against the hand: its own frame, `c` and `h` axis-aligned there). `at` is in that frame too.
pub fn pad_geometry_mapped(hand: &Hand, angles: &[Vec<JointAngles>], to_item: &dyn Fn(Vec3) -> Vec3, c: Vec3, h: Vec3, shape: ShapeKind) -> Vec<PadGeom> {
    let mut out = Vec::new();
    for (fi, (finger, a)) in hand.fingers.iter().zip(angles).enumerate() {
        let pts: Vec<Vec3> = finger.points(a).into_iter().map(to_item).collect();
        for k in 0..pts.len().saturating_sub(1) {
            let (best, at) = (0..AXIS_SAMPLES)
                .map(|s| {
                    let p = pts[k] + (pts[k + 1] - pts[k]).scale(s as f32 / (AXIS_SAMPLES - 1) as f32);
                    (shape_sdf(p, c, h, shape), p)
                })
                .min_by(|x, y| x.0.total_cmp(&y.0))
                .unwrap_or((f32::INFINITY, pts[k]));
            let gap = best - PHALANX_PAD_M;
            let overlap = (-gap).max(0.0);
            out.push(PadGeom { finger: fi, phalanx: k, gap_m: gap, indentation_m: PAD_THICKNESS_M * (1.0 - (-overlap / PAD_THICKNESS_M).exp()), at });
        }
    }
    out
}

/// P(contact) = σ((touch − gap)/soft − speed/slip).
pub fn contact_probability(gap_m: f32, speed_mps: f32, cfg: &TactileCfg) -> f32 {
    let z = (cfg.touch_m - gap_m) / cfg.soft_m.max(1e-6) - speed_mps.max(0.0) / cfg.slip_mps.max(1e-6);
    1.0 / (1.0 + (-z).exp())
}

/// The elastic (Hertz) normal force of a pad pressed in by `indentation_m`: (4/3)·E*·√R·δ^(3/2), the inverse of
/// `material::pad_indentation_m` before its saturation.
pub fn hertz_force_n(indentation_m: f32) -> f32 {
    let e_star = PAD_E_PA / (1.0 - PAD_NU * PAD_NU);
    4.0 / 3.0 * e_star * PAD_RADIUS_M.sqrt() * indentation_m.max(0.0).powf(1.5)
}

/// One sample of the tactile stream: up to `MAX_ACTIVE_PADS` pads of the two hands (in `pads` order: finger, phalanx,
/// hand), each with its contact probability and expected force, and each hand's total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TactileStreamOut {
    pub n: u8,
    pub hand: [u8; MAX_ACTIVE_PADS],
    pub finger: [u8; MAX_ACTIVE_PADS],
    pub phalanx: [u8; MAX_ACTIVE_PADS],
    pub contact_prob: [f32; MAX_ACTIVE_PADS],
    /// P(contact) × |force| (normal and tangential), N.
    pub force_n: [f32; MAX_ACTIVE_PADS],
    /// Σ of the pads with P ≥ ½ of |force|, per hand (right, left), N.
    pub total_force_lr: [f32; 2],
}

/// The stream sample of one or two hands: each hand's pad geometry (from the frozen pose), the pads' speeds against
/// the item (m/s, same order; empty: at rest) and the pad model's contacts for that hand. A pad with no contact record
/// carries no force.
pub fn tactile_stream(hands: &[(&[PadGeom], &[f32], &FingerContacts)], cfg: &TactileCfg) -> TactileStreamOut {
    let mut out = TactileStreamOut::default();
    for (side, (geom, speed, contacts)) in hands.iter().enumerate().take(2) {
        for (i, g) in geom.iter().enumerate() {
            let p = contact_probability(g.gap_m, speed.get(i).copied().unwrap_or(0.0), cfg);
            let c = contacts.as_slice().iter().find(|c| c.finger as usize == g.finger && c.phalanx as usize == g.phalanx);
            let f = c.map_or(0.0, |c| (c.fn_n * c.fn_n + c.ft_n * c.ft_n).sqrt());
            if p >= 0.5 {
                out.total_force_lr[side] += f;
            }
            let k = out.n as usize;
            if k < MAX_ACTIVE_PADS && (p >= 0.05 || f > 0.0) {
                out.hand[k] = side as u8;
                out.finger[k] = g.finger as u8;
                out.phalanx[k] = g.phalanx as u8;
                out.contact_prob[k] = p;
                out.force_n[k] = p * f;
                out.n += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::act::material::pad_indentation_m;
    use crate::hand::{FingerCount, NotoConfig};
    use crate::sim::top_grasp_shaped;

    #[test]
    fn the_pads_a_closed_grasp_presses_are_the_ones_that_touch() {
        let hand = NotoConfig::preset(FingerCount::Five).build().unwrap();
        let tg = top_grasp_shaped(&hand, [0.08, 0.06, 0.06], ShapeKind::Box);
        let geom = pad_geometry(&hand, &tg.angles, tg.center, tg.half, tg.shape);
        let cfg = TactileCfg::default();
        let pressing: Vec<(usize, usize)> = tg.contacts.iter().filter(|c| c.max_n > 1e-6).map(|c| (c.finger, c.phalanx)).collect();
        assert!(!pressing.is_empty());
        for g in &geom {
            let p = contact_probability(g.gap_m, 0.0, &cfg);
            if pressing.contains(&(g.finger, g.phalanx)) {
                assert!(p > 0.5, "a pressing pad reads touching: {g:?} {p}");
            }
        }
        // Open the hand: nothing touches, and the release gate drops the total to 0.
        let open: Vec<Vec<JointAngles>> = hand.fingers.iter().map(|f| vec![JointAngles::default(); f.joint_count()]).collect();
        let far = pad_geometry(&hand, &open, tg.center + Vec3::new(0.0, 0.3, 0.0), tg.half, tg.shape);
        assert!(far.iter().all(|g| contact_probability(g.gap_m, 0.0, &cfg) < 0.01));
        let out = tactile_stream(&[(&far[..], &[][..], &FingerContacts::default())], &cfg);
        assert_eq!(out.total_force_lr, [0.0, 0.0]);
        // A pad sliding along the surface is less likely holding.
        assert!(contact_probability(0.0, 0.1, &cfg) < contact_probability(0.0, 0.0, &cfg));
    }

    #[test]
    fn the_hertz_force_inverts_the_pad_indentation() {
        for n in [0.5f32, 2.0, 8.0] {
            let d = pad_indentation_m(n);
            // Undo the finite-thickness saturation, then Hertz gives the force back.
            let dh = -PAD_THICKNESS_M * (1.0 - d / PAD_THICKNESS_M).ln();
            assert!((hertz_force_n(dh) - n).abs() < 0.02 * n, "{n} N → {d} m → {} N", hertz_force_n(dh));
        }
    }
}
