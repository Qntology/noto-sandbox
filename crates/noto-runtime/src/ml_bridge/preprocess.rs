use image::{imageops::FilterType, DynamicImage};
use serde::{Deserialize, Serialize};

use super::config::Siglip2Config;

/// SigLIP2 patches across the frame's short side in a 1:2:4 pyramid: half of TIPSv2's 24 (`InputGrid::ShortSide`), so
/// the depth grid is exactly twice this one and a SAM mask of 16 px per patch is exactly 16 times it.
pub const PYRAMID_SHORT_SIDE: usize = 12;

#[derive(Clone, Debug, PartialEq)]
pub struct NaflexInput {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
    pub width: u32,
    pub height: u32,
}

pub fn naflex_grid(width: u32, height: u32, patch: usize, max_patches: usize) -> (usize, usize) {
    let scaled_dim = |scale: f64, size: u32| -> usize {
        let scaled = scale * size as f64;
        let divisible = (scaled / patch as f64).ceil() as usize * patch;
        divisible.max(patch)
    };

    let eps = 1e-5;
    let (mut lo, mut hi) = (eps, 100.0f64);
    while hi - lo >= eps {
        let mid = (lo + hi) / 2.0;
        let th = scaled_dim(mid, height);
        let tw = scaled_dim(mid, width);
        let n = (th / patch) * (tw / patch);
        if n <= max_patches {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let th = scaled_dim(lo, height);
    let tw = scaled_dim(lo, width);
    ((th / patch).max(1), (tw / patch).max(1))
}

/// The grid of a 1:2:4 pyramid's semantic layer: `short_side` patches across the frame's short side, the long side
/// rounded to the frame's aspect as TIPSv2 rounds its own (`InputGrid::ShortSide`), and the short side cut down until
/// the grid fits `max_patches`. `naflex_grid` fills the budget with whatever counts it lands on (19 × 13 for 384 × 256,
/// 18 × 14 for 640 × 480), which no other model's grid divides; this one gives 18 × 12 and 16 × 12, half of TIPSv2's
/// 36 × 24 and 32 × 24. Returns (grid_h, grid_w) like `naflex_grid`.
pub fn pyramid_grid(width: u32, height: u32, short_side: usize, max_patches: usize) -> (usize, usize) {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let budget = max_patches.max(1);
    let mut s = short_side.max(1);
    loop {
        let long = ((s as f64 * w.max(h) / w.min(h)).round() as usize).max(1);
        if s * long <= budget || s == 1 {
            let long = long.min(budget);
            return if width <= height { (long, s) } else { (s, long) };
        }
        s -= 1;
    }
}

/// How the dense (per-patch) SigLIP2 pass picks its grid: NaFlex's own (fill the patch budget) or the pyramid's
/// (aligned with the depth and mask layers).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DenseGrid {
    #[default]
    Naflex,
    Pyramid { short_side: usize },
}

impl DenseGrid {
    pub fn pyramid() -> DenseGrid {
        DenseGrid::Pyramid { short_side: PYRAMID_SHORT_SIDE }
    }

    /// (grid_h, grid_w) for a `width` × `height` image.
    pub fn grid_for(self, width: u32, height: u32, patch: usize, max_patches: usize) -> (usize, usize) {
        match self {
            DenseGrid::Naflex => naflex_grid(width, height, patch, max_patches),
            DenseGrid::Pyramid { short_side } => pyramid_grid(width, height, short_side, max_patches),
        }
    }
}

pub fn preprocess(image: &DynamicImage, cfg: &Siglip2Config) -> NaflexInput {
    let (grid_h, grid_w) = naflex_grid(image.width(), image.height(), cfg.patch_size, cfg.max_patches);
    preprocess_grid(image, cfg, grid_h, grid_w)
}

/// `preprocess` on a grid the caller chose (`DenseGrid::grid_for`): the image is stretched to grid × patch pixels.
pub fn preprocess_grid(image: &DynamicImage, cfg: &Siglip2Config, grid_h: usize, grid_w: usize) -> NaflexInput {
    let p = cfg.patch_size;
    let (grid_h, grid_w) = (grid_h.max(1), grid_w.max(1));
    let (rw, rh) = ((grid_w * p) as u32, (grid_h * p) as u32);
    let rgb = image.resize_exact(rw, rh, FilterType::Triangle).into_rgb8();
    let raw = rgb.as_raw();
    let stride = (rw as usize) * 3;

    // Exact 256-entry LUT matches `v as f32 / 127.5 - 1.0` bit-for-bit without per-pixel division.
    static U8_NORM_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    let lut = U8_NORM_LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = i as f32 / 127.5 - 1.0;
        }
        t
    });

    let patch_dim = p * p * 3;
    let mut patches = vec![0.0f32; grid_h * grid_w * patch_dim];
    let row_bytes = p * 3;
    let extract_row = |(gy, dst_grid_row): (usize, &mut [f32])| {
        let y0 = gy * p;
        for gx in 0..grid_w {
            let base = gx * patch_dim;
            let x0_b = gx * p * 3;
            for py in 0..p {
                let src = &raw[(y0 + py) * stride + x0_b..(y0 + py) * stride + x0_b + row_bytes];
                let dst = &mut dst_grid_row[base + py * row_bytes..base + (py + 1) * row_bytes];
                for (d, &b) in dst.iter_mut().zip(src.iter()) {
                    *d = lut[b as usize];
                }
            }
        }
    };
    if grid_h * grid_w >= 32 {
        use rayon::prelude::*;
        patches
            .par_chunks_exact_mut(grid_w * patch_dim)
            .enumerate()
            .for_each(extract_row);
    } else {
        patches
            .chunks_exact_mut(grid_w * patch_dim)
            .enumerate()
            .for_each(extract_row);
    }

    NaflexInput {
        patches,
        grid_h,
        grid_w,
        patch_dim,
        width: image.width(),
        height: image.height(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn grid_respects_patch_budget_and_aspect() {
        let (h, w) = naflex_grid(640, 480, 16, 256);
        assert!(h * w <= 256);
        assert!(h >= 1 && w >= 1);
        assert!(w > h);
        let (h2, w2) = naflex_grid(480, 640, 16, 256);
        assert!(h2 > w2);
    }

    #[test]
    fn square_image_fills_the_budget() {
        let (h, w) = naflex_grid(1000, 1000, 16, 256);
        assert_eq!((h, w), (16, 16));
    }

    #[test]
    fn the_pyramid_grid_is_half_the_depth_grid_and_fits_the_budget() {
        assert_eq!(naflex_grid(384, 256, 16, 256), (13, 19), "the unaligned grid this replaces");
        assert_eq!(naflex_grid(640, 480, 16, 256), (14, 18));
        assert_eq!(pyramid_grid(384, 256, PYRAMID_SHORT_SIDE, 256), (12, 18), "half of TIPSv2's 36 x 24");
        assert_eq!(pyramid_grid(640, 480, PYRAMID_SHORT_SIDE, 256), (12, 16), "half of TIPSv2's 32 x 24");
        assert_eq!(pyramid_grid(256, 384, PYRAMID_SHORT_SIDE, 256), (18, 12));
        assert_eq!(pyramid_grid(500, 500, PYRAMID_SHORT_SIDE, 256), (12, 12));
        assert_eq!(pyramid_grid(500, 500, 16, 256), (16, 16));
        let (h, w) = pyramid_grid(512, 256, PYRAMID_SHORT_SIDE, 256);
        assert_eq!((h, w), (11, 22), "a wide frame shrinks its short side to fit 256");
        let (h, w) = pyramid_grid(10_000, 10, PYRAMID_SHORT_SIDE, 256);
        assert!(h == 1 && w <= 256, "{h} x {w}");
        assert_eq!(DenseGrid::pyramid().grid_for(384, 256, 16, 256), (12, 18));
        assert_eq!(DenseGrid::default().grid_for(384, 256, 16, 256), (13, 19));
    }

    #[test]
    fn a_chosen_grid_sets_the_patch_count() {
        let cfg = Siglip2Config { patch_size: 4, max_patches: 64, ..Siglip2Config::default() };
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(48, 32, image::Rgb([10, 20, 30])));
        let input = preprocess_grid(&img, &cfg, 3, 5);
        assert_eq!((input.grid_h, input.grid_w), (3, 5));
        assert_eq!(input.patches.len(), 3 * 5 * input.patch_dim);
        assert_eq!((input.width, input.height), (48, 32));
    }

    #[test]
    fn preprocess_shapes_and_range() {
        let cfg = Siglip2Config {
            patch_size: 4,
            max_patches: 16,
            ..Siglip2Config::default()
        };
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 20, image::Rgb([255, 0, 128])));
        let input = preprocess(&img, &cfg);
        assert_eq!(input.patch_dim, 4 * 4 * 3);
        assert_eq!(input.patches.len(), input.grid_h * input.grid_w * input.patch_dim);
        assert!(input.patches.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert!((input.patches[0] - 1.0).abs() < 1e-6);
        assert!((input.patches[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn preprocess_large_grid_parallel_matches_serial_lut() {
        let cfg = Siglip2Config {
            patch_size: 16,
            max_patches: 256,
            ..Siglip2Config::default()
        };
        // 12x16 = 192 patches (>= 32) exercises rayon parallel extraction.
        let mut raw = vec![0u8; 320 * 240 * 3];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = ((i * 29 + 17) % 256) as u8;
        }
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(320, 240, raw).unwrap());
        let got = preprocess_grid(&img, &cfg, 12, 16);
        let rgb = img.resize_exact(16 * 16, 12 * 16, FilterType::Triangle).to_rgb8();
        let src = rgb.as_raw();
        let stride = 16 * 16 * 3;
        let patch_dim = 16 * 16 * 3;
        let mut want = vec![0.0f32; 12 * 16 * patch_dim];
        for gy in 0..12 {
            for gx in 0..16 {
                let base = (gy * 16 + gx) * patch_dim;
                for py in 0..16 {
                    for px in 0..16 {
                        for c in 0..3 {
                            let b = src[(gy * 16 + py) * stride + (gx * 16 + px) * 3 + c];
                            want[base + (py * 16 + px) * 3 + c] = b as f32 / 127.5 - 1.0;
                        }
                    }
                }
            }
        }
        assert_eq!(got.patches, want, "parallel preprocess_grid must be bit-identical to serial LUT");
    }
}
