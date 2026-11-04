//! What a look can tell of an item from the depth inside its box, before a hand goes to it: which way it lies on the
//! ground, how long, wide and high it is, how its top is tilted and whether the top is flat. The box's pixels that
//! have depth go back through the camera (Z along the optical axis times the pixel's ray), the ones above the ground
//! are the item, their spread on the ground gives the long axis (the principal axis of their x–z scatter), the 5th
//! to 95th percentiles along and across it the extents, and a plane through the highest layer the top's tilt.
//!
//! The camera sees the top and the faces turned to it, not the far ones: an extent along the line of sight comes out
//! a little short, and an item seen end-on gives its width for its length. It knows images, the camera and its pose,
//! never the game state.

use serde::Serialize;

use crate::camera::{wrap_angle, Intrinsics, Pose};
use crate::frame::DepthMap;

/// Heights within this of the ground (m) are ground, not item.
pub const ITEM_LIFT_M: f32 = 0.02;
/// The fewest item points a pose is read from.
pub const ITEM_MIN_POINTS: usize = 24;
/// Long over wide past this is a bar (a log), not a block.
pub const BAR_RATIO: f32 = 2.0;
/// The box's own ground is believed over the camera's only past this difference (m): a step, not the matcher's bias.
pub const GROUND_STEP_M: f32 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ItemPose {
    /// The long axis on the ground, as a world heading folded into (−90°, 90°] (deg).
    pub yaw_deg: f32,
    /// The same, from the camera's heading: 0 is straight away from it, ±90 across its view (deg).
    pub rel_yaw_deg: f32,
    /// Along the long axis, across it and up from the ground (m).
    pub length_m: f32,
    pub width_m: f32,
    pub height_m: f32,
    /// The plane through the top layer: its tilt from level (deg).
    pub top_tilt_deg: f32,
    /// 1 − minor/major spread: 0 for a square or a disc, near 1 for a long bar. How much the yaw is worth.
    pub elongation: f32,
    /// "bar" or "block".
    pub form: &'static str,
    pub points: usize,
    /// The ground under it (m, in the pose's height frame) that the heights are from.
    pub ground_m: f32,
}

fn percentile(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    let i = ((v.len() as f32 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    v[i.min(v.len() - 1)]
}

/// The item inside `bbox` (normalised to the frame `intr` describes) of `depth`, seen from `pose`. The ground under
/// it is the lowest fifth of the box's points when the box takes in some ground round the item, else `ground_m`.
pub fn item_pose(depth: &DepthMap, intr: &Intrinsics, pose: &Pose, bbox: [f32; 4], ground_m: f32) -> Option<ItemPose> {
    let (w, h) = (intr.width as f32, intr.height as f32);
    let (sx, sy) = (depth.width as f32 / w, depth.height as f32 / h);
    let x0 = ((bbox[0] * w * sx).floor().max(0.0)) as u32;
    let x1 = ((bbox[2] * w * sx).ceil() as u32).min(depth.width.saturating_sub(1));
    let y0 = ((bbox[1] * h * sy).floor().max(0.0)) as u32;
    let y1 = ((bbox[3] * h * sy).ceil() as u32).min(depth.height.saturating_sub(1));
    let mut pts: Vec<[f32; 3]> = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = depth.at(x, y);
            if !(d.is_finite() && d > 0.0) {
                continue;
            }
            let ray = intr.ray(pose.yaw, (x as f32 + 0.5) / sx, (y as f32 + 0.5) / sy);
            pts.push([pose.x + d * ray[0], pose.eye_m + d * ray[1], pose.z + d * ray[2]]);
        }
    }
    if pts.len() < ITEM_MIN_POINTS {
        return None;
    }
    // The ground under it: the one the camera stands on (`ground_m`), unless the box's lowest points — the middle of its
    // low tail, which is the matcher's noise — put it on another level, a step or more away. Seen at a slant, ground
    // reads about 3 cm low from SGBM (a window square to the camera on a surface that is not), and an item measured
    // from there came out 3 cm too tall [측정: `item_pose_eval`, the renderer's depth gives 0.141 m for the 0.14 m log].
    let mut heights: Vec<f32> = pts.iter().map(|p| p[1]).collect();
    let low = percentile(&mut heights, 0.1);
    let mut bottom: Vec<f32> = heights.iter().copied().filter(|h| *h < low + 0.03).collect();
    let seen = if bottom.is_empty() { ground_m } else { percentile(&mut bottom, 0.5) };
    let ground = if (seen - ground_m).abs() > GROUND_STEP_M { seen } else { ground_m };
    let lifted: Vec<[f32; 3]> = pts.into_iter().filter(|p| p[1] > ground + ITEM_LIFT_M).collect();
    if lifted.len() < ITEM_MIN_POINTS {
        return None;
    }
    // Where a silhouette meets what lies behind it, the matcher hands the near depth to a few background pixels; past
    // the far edge those rays put it in the air above the top. The top is where most of the upper points are, and what
    // stands well above it is that smear.
    let mut hs: Vec<f32> = lifted.iter().map(|p| p[1] - ground).collect();
    let top_h = percentile(&mut hs, 0.75);
    let item: Vec<[f32; 3]> = lifted.into_iter().filter(|p| p[1] - ground <= 1.15 * top_h + 0.01).collect();
    if item.len() < ITEM_MIN_POINTS {
        return None;
    }
    let mut hs: Vec<f32> = item.iter().map(|p| p[1] - ground).collect();
    let height_m = percentile(&mut hs, 0.9);
    // The footprint from the top face: the same smear beside the silhouette floats between the ground and the top
    // (rays past the item's sides, ended at its depth), and taken with the rest it doubled a 10 cm stone. The top's
    // outline is a block's footprint; a round top gives a narrower band, which reads short across.
    let top_face: Vec<[f32; 3]> = item.iter().copied().filter(|p| p[1] - ground >= 0.7 * height_m).collect();
    let item = if top_face.len() >= ITEM_MIN_POINTS / 2 { top_face } else { item };
    let n = item.len() as f32;
    let (mx, mz) = (item.iter().map(|p| p[0]).sum::<f32>() / n, item.iter().map(|p| p[2]).sum::<f32>() / n);
    let (mut cxx, mut cxz, mut czz) = (0.0f32, 0.0f32, 0.0f32);
    for p in &item {
        let (dx, dz) = (p[0] - mx, p[2] - mz);
        cxx += dx * dx;
        cxz += dx * dz;
        czz += dz * dz;
    }
    let (cxx, cxz, czz) = (cxx / n, cxz / n, czz / n);
    // The principal axis of the 2 × 2 scatter and its two spreads.
    let theta = 0.5 * (2.0 * cxz).atan2(cxx - czz);
    let mid = 0.5 * (cxx + czz);
    let rad = (0.25 * (cxx - czz).powi(2) + cxz * cxz).sqrt();
    let (major, minor) = ((mid + rad).max(0.0), (mid - rad).max(0.0));
    let (ax, az) = theta.sin_cos();
    let (ux, uz) = (az, ax);
    let along: Vec<f32> = item.iter().map(|p| (p[0] - mx) * ux + (p[2] - mz) * uz).collect();
    let across: Vec<f32> = item.iter().map(|p| -(p[0] - mx) * uz + (p[2] - mz) * ux).collect();
    // The 5th–95th percentiles of an even footprint span 90 % of it.
    let spread = |v: &[f32]| {
        let mut v = v.to_vec();
        (percentile(&mut v, 0.95) - percentile(&mut v, 0.05)) / 0.9
    };
    let (length_m, width_m) = (spread(&along), spread(&across));
    // The top: a plane height = a·along + b·across + c through the upper third, fitted again without the points well
    // below it (the tops of the side faces the camera also sees), twice.
    let mut top: Vec<(f32, f32, f32)> = item.iter().zip(along.iter().zip(&across)).filter(|(p, _)| p[1] - ground > 0.67 * height_m).map(|(p, (u, v))| (*u, *v, p[1])).collect();
    let mut fit = plane(&top);
    for _ in 0..2 {
        let Some((a, b, c)) = fit else { break };
        let keep = (0.06 * height_m).max(0.008);
        top.retain(|&(u, v, z)| z - (a * u + b * v + c) > -keep);
        fit = plane(&top);
    }
    let top_tilt_deg = fit.map_or(0.0, |(a, b, _)| a.hypot(b).atan().to_degrees());
    let yaw = wrap_angle(uz.atan2(ux));
    let fold = |a: f32| {
        let mut a = wrap_angle(a);
        if a > std::f32::consts::FRAC_PI_2 {
            a -= std::f32::consts::PI;
        } else if a <= -std::f32::consts::FRAC_PI_2 {
            a += std::f32::consts::PI;
        }
        a
    };
    let elongation = if major > 1e-9 { 1.0 - (minor / major).sqrt() } else { 0.0 };
    Some(ItemPose {
        yaw_deg: fold(yaw).to_degrees(),
        rel_yaw_deg: fold(yaw - pose.yaw).to_degrees(),
        length_m,
        width_m,
        height_m,
        top_tilt_deg,
        elongation,
        form: if length_m > BAR_RATIO * width_m.max(1e-3) { "bar" } else { "block" },
        points: item.len(),
        ground_m: ground,
    })
}

/// The least-squares plane `z = a·u + b·v + c` through the points.
fn plane(p: &[(f32, f32, f32)]) -> Option<(f32, f32, f32)> {
    if p.len() < 6 {
        return None;
    }
    let n = p.len() as f32;
    let (mu, mv, mz) = (p.iter().map(|q| q.0).sum::<f32>() / n, p.iter().map(|q| q.1).sum::<f32>() / n, p.iter().map(|q| q.2).sum::<f32>() / n);
    let (mut suu, mut suv, mut svv, mut suz, mut svz) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(u, v, z) in p {
        let (u, v, z) = (u - mu, v - mv, z - mz);
        suu += u * u;
        suv += u * v;
        svv += v * v;
        suz += u * z;
        svz += v * z;
    }
    let det = suu * svv - suv * suv;
    if det.abs() < 1e-12 {
        return None;
    }
    let (a, b) = ((suz * svv - svz * suv) / det, (svz * suu - suz * suv) / det);
    Some((a, b, mz - a * mu - b * mv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;

    /// A depth map of a flat ground with a box `len` × `wid` × `hgt` lying `dist` ahead at heading `yaw` (world), seen
    /// by a camera at height `eye` pitched down `pitch_deg`: each pixel's depth is where its ray first meets either.
    fn scene(yaw: f32, len: f32, wid: f32, hgt: f32, dist: f32, eye: f32, pitch_deg: f32) -> (DepthMap, Intrinsics, Pose, [f32; 4]) {
        let intr = Intrinsics { width: 240, height: 160, fov_deg: 70.0, pitch_deg, roll_deg: 0.0 };
        let pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: eye };
        let (c, s) = (yaw.cos(), yaw.sin());
        let mut data = vec![f32::NAN; 240 * 160];
        let (mut bx0, mut by0, mut bx1, mut by1) = (1.0f32, 1.0f32, 0.0f32, 0.0f32);
        for y in 0..160u32 {
            for x in 0..240u32 {
                let r = intr.ray(0.0, x as f32 + 0.5, y as f32 + 0.5);
                let mut best = if r[1] < -1e-6 { eye / -r[1] } else { f32::INFINITY };
                let mut on_item = false;
                // The box in its own frame (u along, v across, height): slab intersection of the ray.
                let o = [-dist, eye, 0.0];
                let ou = o[0] * c + o[2] * s;
                let ov = -o[0] * s + o[2] * c;
                let du = r[0] * c + r[2] * s;
                let dv = -r[0] * s + r[2] * c;
                let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
                for (oo, dd, lo, hi) in [(ou, du, -0.5 * len, 0.5 * len), (ov, dv, -0.5 * wid, 0.5 * wid), (o[1], r[1], 0.0, hgt)] {
                    if dd.abs() < 1e-9 {
                        if oo < lo || oo > hi {
                            t0 = f32::INFINITY;
                        }
                        continue;
                    }
                    let (a, b) = ((lo - oo) / dd, (hi - oo) / dd);
                    t0 = t0.max(a.min(b));
                    t1 = t1.min(a.max(b));
                }
                if t0 <= t1 && t0 < best {
                    best = t0;
                    on_item = true;
                }
                if best.is_finite() {
                    data[(y * 240 + x) as usize] = best;
                }
                if on_item {
                    let (u, v) = ((x as f32 + 0.5) / 240.0, (y as f32 + 0.5) / 160.0);
                    bx0 = bx0.min(u);
                    by0 = by0.min(v);
                    bx1 = bx1.max(u);
                    by1 = by1.max(v);
                }
            }
        }
        let pad = 0.02;
        (DepthMap { width: 240, height: 160, kind: DepthKind::Metric, data }, intr, pose, [bx0 - pad, by0 - pad, bx1 + pad, by1 + pad])
    }

    #[test]
    fn a_log_lying_at_an_angle_reads_as_a_bar_at_that_angle_with_its_size() {
        for deg in [0.0f32, 25.0, -40.0, 70.0, 90.0] {
            let (d, intr, pose, bbox) = scene(deg.to_radians(), 0.50, 0.14, 0.14, 0.8, 1.45, -45.0);
            let p = item_pose(&d, &intr, &pose, bbox, 0.0).expect("an item in the box");
            let off = {
                let mut e = p.yaw_deg - deg;
                while e > 90.0 {
                    e -= 180.0;
                }
                while e <= -90.0 {
                    e += 180.0;
                }
                e.abs()
            };
            assert!(off < 3.0, "{deg}°: read {:.1}° ({p:?})", p.yaw_deg);
            assert_eq!(p.form, "bar", "{deg}°: {p:?}");
            assert!((p.length_m - 0.50).abs() < 0.08 && (p.width_m - 0.14).abs() < 0.05 && (p.height_m - 0.14).abs() < 0.02, "{deg}°: {p:?}");
            assert!(p.top_tilt_deg < 4.0 && p.ground_m.abs() < 0.03, "{deg}°: {p:?}");
        }
        let (d, intr, pose, bbox) = scene(0.3, 0.16, 0.12, 0.10, 0.7, 1.45, -45.0);
        let p = item_pose(&d, &intr, &pose, bbox, 0.0).unwrap();
        assert_eq!(p.form, "block", "{p:?}");
    }

    #[test]
    fn too_few_points_read_nothing() {
        let d = DepthMap { width: 8, height: 8, kind: DepthKind::Metric, data: vec![f32::NAN; 64] };
        let intr = Intrinsics { width: 8, height: 8, fov_deg: 60.0, pitch_deg: -40.0, roll_deg: 0.0 };
        assert!(item_pose(&d, &intr, &Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 1.4 }, [0.0, 0.0, 1.0, 1.0], 0.0).is_none());
    }
}
