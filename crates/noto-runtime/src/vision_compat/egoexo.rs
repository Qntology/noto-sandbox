use serde::{Deserialize, Serialize};

pub use lv_spatial::egoexo::*;
pub use lv_spatial::fit3::{align_similarity, mpjpe, pa_mpjpe, Similarity, M3, V3};

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
    fn occlusion_counts_the_object_pixels_the_hand_covers() {
        let object = [true, true, true, true, false];
        let hand = [true, false, false, true, true];
        assert_eq!(occlusion_ratio(&object, &hand), 0.5);
        assert_eq!(valid_ratio(&hand, &[1.0, 1.0, 1.0, f32::NAN, 2.0]), 2.0 / 3.0);
        assert_eq!(occlusion_ratio(&[false; 3], &[true; 3]), 0.0);
    }
}
