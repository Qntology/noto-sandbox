//! How closely two videos of the same episode move together: a reference (the simulator's render) and a candidate (a
//! render of another embodiment, a re-render, a synthesized video). A pixel moves at frame t when its colour changed by
//! more than `tau` (Euclidean, 0–255 RGB) since frame t − 1; Motion IoU is the intersection over union of the two
//! videos' moving pixels over the whole space-time volume. PSNR and SSIM (8 × 8 windows on luma) compare the frames
//! themselves.

use serde::{Deserialize, Serialize};

use crate::frame::Rgb;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotionMetrics {
    pub motion_iou: f32,
    /// Mean over frames, dB (frames that are identical count as 99 dB).
    pub psnr: f32,
    pub ssim: f32,
    /// Moving pixels in the reference and the candidate, over all frames.
    pub moving_ref: u64,
    pub moving_gen: u64,
}

/// The default motion threshold: a change of 12 in RGB distance (about 3 grey levels per channel above WebP noise).
pub const MOTION_TAU: f32 = 12.0;

fn same_size(a: &[Rgb], b: &[Rgb]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.width == y.width && x.height == y.height)
}

/// The moving pixels of frame `t` (t ≥ 1) of `v`.
pub fn motion_mask(v: &[Rgb], t: usize, tau: f32) -> Vec<bool> {
    let (a, b) = (&v[t - 1], &v[t]);
    let tau2 = tau * tau;
    a.data
        .chunks_exact(3)
        .zip(b.data.chunks_exact(3))
        .map(|(p, q)| {
            let d: f32 = (0..3).map(|c| (p[c] as f32 - q[c] as f32).powi(2)).sum();
            d > tau2
        })
        .collect()
}

/// Motion IoU of `gen` against `reference` (same length and frame size; `None` otherwise, or with fewer than two
/// frames). Two still videos agree: IoU 1.
pub fn motion_iou(reference: &[Rgb], gen: &[Rgb], tau: f32) -> Option<(f32, u64, u64)> {
    if !same_size(reference, gen) || reference.len() < 2 {
        return None;
    }
    let (mut inter, mut union, mut mr, mut mg) = (0u64, 0u64, 0u64, 0u64);
    for t in 1..reference.len() {
        let (a, b) = (motion_mask(reference, t, tau), motion_mask(gen, t, tau));
        for (x, y) in a.iter().zip(&b) {
            inter += (*x && *y) as u64;
            union += (*x || *y) as u64;
            mr += *x as u64;
            mg += *y as u64;
        }
    }
    Some((if union == 0 { 1.0 } else { inter as f32 / union as f32 }, mr, mg))
}

fn luma(f: &Rgb) -> Vec<f32> {
    f.data.chunks_exact(3).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect()
}

/// PSNR of two frames of one size, dB (99 when identical).
pub fn psnr(a: &Rgb, b: &Rgb) -> f32 {
    let n = a.data.len().max(1) as f32;
    let mse: f32 = a.data.iter().zip(&b.data).map(|(x, y)| (*x as f32 - *y as f32).powi(2)).sum::<f32>() / n;
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * (255.0f32 * 255.0 / mse).log10()
    }
}

/// SSIM of two frames' luma, the mean over non-overlapping 8 × 8 windows (C1 = (0.01·255)², C2 = (0.03·255)²).
pub fn ssim(a: &Rgb, b: &Rgb) -> f32 {
    let (la, lb) = (luma(a), luma(b));
    let (w, h) = (a.width as usize, a.height as usize);
    let (c1, c2) = ((0.01f32 * 255.0).powi(2), (0.03f32 * 255.0).powi(2));
    let (mut sum, mut n) = (0.0f64, 0u32);
    for y0 in (0..h.saturating_sub(7)).step_by(8) {
        for x0 in (0..w.saturating_sub(7)).step_by(8) {
            let px = |l: &[f32]| (0..8).flat_map(|dy| (0..8).map(move |dx| (y0 + dy) * w + x0 + dx)).map(|i| l[i]).collect::<Vec<f32>>();
            let (p, q) = (px(&la), px(&lb));
            let m = |v: &[f32]| v.iter().sum::<f32>() / 64.0;
            let (mp, mq) = (m(&p), m(&q));
            let vp = p.iter().map(|x| (x - mp).powi(2)).sum::<f32>() / 63.0;
            let vq = q.iter().map(|x| (x - mq).powi(2)).sum::<f32>() / 63.0;
            let cov = p.iter().zip(&q).map(|(x, y)| (x - mp) * (y - mq)).sum::<f32>() / 63.0;
            sum += (((2.0 * mp * mq + c1) * (2.0 * cov + c2)) / ((mp * mp + mq * mq + c1) * (vp + vq + c2))) as f64;
            n += 1;
        }
    }
    if n == 0 {
        1.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Motion IoU, mean PSNR and mean SSIM of `gen` against `reference`.
pub fn motion_metrics(reference: &[Rgb], gen: &[Rgb], tau: f32) -> Option<MotionMetrics> {
    let (iou, mr, mg) = motion_iou(reference, gen, tau)?;
    let k = reference.len() as f32;
    let psnr = reference.iter().zip(gen).map(|(a, b)| psnr(a, b)).sum::<f32>() / k;
    let ssim = reference.iter().zip(gen).map(|(a, b)| ssim(a, b)).sum::<f32>() / k;
    Some(MotionMetrics { motion_iou: iou, psnr, ssim, moving_ref: mr, moving_gen: mg })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey 32 × 16 video with a white 4 × 4 square at x = x0 + t·step.
    fn video(x0: usize, step: usize, frames: usize) -> Vec<Rgb> {
        (0..frames)
            .map(|t| {
                let mut data = vec![90u8; 32 * 16 * 3];
                for y in 6..10 {
                    for x in (x0 + t * step)..(x0 + t * step + 4) {
                        for c in 0..3 {
                            data[(y * 32 + x) * 3 + c] = 250;
                        }
                    }
                }
                Rgb::new(32, 16, data)
            })
            .collect()
    }

    #[test]
    fn the_same_motion_scores_one_and_a_different_one_less() {
        let a = video(2, 2, 5);
        let m = motion_metrics(&a, &a, MOTION_TAU).unwrap();
        assert_eq!(m.motion_iou, 1.0);
        assert_eq!(m.psnr, 99.0);
        assert!((m.ssim - 1.0).abs() < 1e-6);
        assert!(m.moving_ref > 0);
        // Half a step behind: of each frame's 4 changed columns, 2 are shared (IoU 1/3); a whole step behind, none.
        let late = video(1, 2, 5);
        let l = motion_metrics(&a, &late, MOTION_TAU).unwrap();
        assert!((l.motion_iou - 1.0 / 3.0).abs() < 1e-6, "{}", l.motion_iou);
        assert_eq!(motion_iou(&a, &video(0, 2, 5), MOTION_TAU).unwrap().0, 0.0);
        assert!(l.psnr < 99.0 && l.ssim < 1.0);
        // A still candidate shares no motion.
        let still = video(2, 0, 5);
        assert_eq!(motion_iou(&a, &still, MOTION_TAU).unwrap().0, 0.0);
        // Two still videos agree; videos of another size or length do not compare.
        assert_eq!(motion_iou(&still, &still, MOTION_TAU).unwrap().0, 1.0);
        assert!(motion_iou(&a, &a[..4], MOTION_TAU).is_none());
    }
}
