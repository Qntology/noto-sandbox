//! One body point seen from several cameras at once: the head (ego) camera and fixed outside (exo) cameras. Each view
//! gives a point with an anisotropic error — along its line of sight the stereo law σ_Z = Z²·σ_d / (f·b), across it
//! the pixel's angle Z·σ_θ with σ_θ = σ_uv / f — and the views are fused by their information matrices, each scaled by
//! how much of the view the hand hides. The accuracy of the ego view alone is read against the fused reference as the
//! wrist's error and PA-MPJPE (mean joint error after the best similarity transform).
//!
//! All vectors are world coordinates: the line of sight u = (x − o)/‖x − o‖ is already the world direction, so the
//! covariance is built directly in the world frame (rotating it again by the camera's R would turn it twice).

use serde::{Deserialize, Serialize};

pub use noto_core::fit3::{align_similarity, mpjpe, pa_mpjpe, Similarity, M3, V3};

/// One view's observation of one point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewObs {
    /// The camera's optical centre.
    pub origin: V3,
    /// The point the view measured (its stereo depth along its ray).
    pub point: V3,
    /// σ_Z at this depth, m.
    pub sigma_z: f64,
    /// Angular error of a pixel, rad (σ_uv / f).
    pub sigma_theta: f64,
    /// (1 − occlusion) · valid-disparity share of the view; 0 drops it.
    pub weight: f64,
}

/// The stereo depth error at depth `z` for focal `f_px`, baseline `b_m` and disparity error `sigma_d_px`.
pub fn stereo_sigma_z(z: f64, f_px: f64, b_m: f64, sigma_d_px: f64) -> f64 {
    z * z * sigma_d_px / (f_px * b_m).max(1e-12)
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Σ = σ_Z²·uuᵀ + (Z·σ_θ)²·(I − uuᵀ), world frame.
pub fn view_covariance(o: &ViewObs) -> M3 {
    let d = sub(o.point, o.origin);
    let z = norm(d).max(1e-9);
    let u = [d[0] / z, d[1] / z, d[2] / z];
    let (a, b) = (o.sigma_z * o.sigma_z, (z * o.sigma_theta).powi(2));
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let uu = u[i] * u[j];
            m[i][j] = a * uu + b * ((i == j) as u8 as f64 - uu);
        }
    }
    m
}

pub fn inv3(m: &M3) -> Option<M3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ])
}

/// x̂ = (Σ w_c Σ_c⁻¹)⁻¹ Σ w_c Σ_c⁻¹ x_c and its covariance (Σ w_c Σ_c⁻¹)⁻¹; `None` without a weighted view.
pub fn fuse(obs: &[ViewObs]) -> Option<(V3, M3)> {
    let mut info = [[0.0; 3]; 3];
    let mut acc = [0.0; 3];
    let mut any = false;
    for o in obs.iter().filter(|o| o.weight > 0.0) {
        let Some(w) = inv3(&view_covariance(o)) else { continue };
        any = true;
        for i in 0..3 {
            for j in 0..3 {
                info[i][j] += o.weight * w[i][j];
                acc[i] += o.weight * w[i][j] * o.point[j];
            }
        }
    }
    if !any {
        return None;
    }
    let cov = inv3(&info)?;
    let x = [0, 1, 2].map(|i| (0..3).map(|j| cov[i][j] * acc[j]).sum());
    Some((x, cov))
}

/// ρ_occ: the share of the object's own pixels (rendered without the hand) that the hand covers.
pub fn occlusion_ratio(object: &[bool], hand: &[bool]) -> f32 {
    let n = object.iter().filter(|o| **o).count();
    if n == 0 {
        return 0.0;
    }
    object.iter().zip(hand).filter(|(o, h)| **o && **h).count() as f32 / n as f32
}

/// ν_valid: the share of the hand's pixels with a finite stereo depth.
pub fn valid_ratio(hand: &[bool], depth: &[f32]) -> f32 {
    let n = hand.iter().filter(|h| **h).count();
    if n == 0 {
        return 0.0;
    }
    hand.iter().zip(depth).filter(|(h, d)| **h && d.is_finite()).count() as f32 / n as f32
}

/// The ego view's accuracy against the multi-view reference: the wrist's error and PA-MPJPE, cm, and how much of the
/// object the hand hid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EgoExoPose {
    pub wrist_err_cm: f32,
    pub pa_mpjpe_cm: f32,
    pub occlusion_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_far_view_is_trusted_across_its_ray_not_along_it() {
        // Two cameras at right angles see the same point; each is sure across its ray and unsure along it.
        let truth = [0.0, 1.0, 2.0];
        let a = ViewObs { origin: [0.0, 1.0, 0.0], point: [0.0, 1.0, 2.3], sigma_z: 0.2, sigma_theta: 0.002, weight: 1.0 };
        let b = ViewObs { origin: [2.0, 1.0, 2.0], point: [-0.25, 1.0, 2.0], sigma_z: 0.2, sigma_theta: 0.002, weight: 1.0 };
        let (x, cov) = fuse(&[a, b]).unwrap();
        assert!(norm(sub(x, truth)) < 0.02, "the cross fixes both depths: {x:?}");
        assert!(cov[0][0] < 1e-3 && cov[2][2] < 1e-3);
        // Alone, a view keeps its depth error; a hidden view (weight 0) changes nothing.
        let (only_a, _) = fuse(&[a, ViewObs { weight: 0.0, ..b }]).unwrap();
        assert!((only_a[2] - 2.3).abs() < 1e-9);
        assert!(fuse(&[ViewObs { weight: 0.0, ..a }]).is_none());
        // The stereo law: four times the error at twice the depth.
        assert!((stereo_sigma_z(4.0, 300.0, 0.05, 0.27) / stereo_sigma_z(2.0, 300.0, 0.05, 0.27) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn occlusion_counts_the_object_pixels_the_hand_covers() {
        let object = [true, true, true, true, false];
        let hand = [true, false, false, true, true];
        assert_eq!(occlusion_ratio(&object, &hand), 0.5);
        assert_eq!(valid_ratio(&hand, &[1.0, 1.0, 1.0, f32::NAN, 2.0]), 2.0 / 3.0);
        assert_eq!(occlusion_ratio(&[false; 3], &[true; 3]), 0.0);
    }
}
