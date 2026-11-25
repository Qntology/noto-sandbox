//! A perforated dense pass for the semantic layer (L1): only some patches of the grid go through SigLIP2, each with its
//! own cell's position embedding (the holes are left out of the encoder, not blacked out), and around the strongest
//! label peaks the closed neighbours are opened like a minesweeper board — they attend to every token computed so far
//! (`Siglip2::open_cells`). The pattern and the opening rule are fixed functions of the grid, the horizon and the heat:
//! no random drop, the same frame gives the same cells.
//!
//! Holes that were never opened take the lowest heat of their evaluated neighbours, so a hole never makes a peak of its
//! own and the heat grid keeps the layout `points_with_presence`, `gate_points` and `snap_points` read.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::camera::Intrinsics;
use crate::vision_compat::locate::{label_heats, Heat, LabelTexts, LocateCfg};

pub use lv_spatial::perforate::{
    cells_to_open, fill_holes, seeds, ExpandCfg, PerfPattern, PerforatedCfg,
};

/// The horizon's row on a grid of `rows` cell rows over a frame from `intr` (cells from the top; below 0 or past
/// `rows` when the horizon is out of the frame). Roll is ignored (the robot's head camera keeps it level).
pub fn horizon_row(intr: &Intrinsics, rows: u32) -> f32 {
    let (w, h) = (intr.width.max(1) as f32, intr.height.max(1) as f32);
    let half_h = (intr.fov_deg.to_radians() * 0.5).tan() * h / w;
    // The ray through screen height sy is level when sin(pitch) + cos(pitch)·sy = 0.
    let sy = -intr.pitch_deg.to_radians().tan();
    let y = (1.0 - sy / half_h.max(1e-6)) * 0.5;
    y * rows as f32
}

/// The cells (row-major) the first pass evaluates. `horizon` is the horizon's row (`horizon_row`); without one the
/// fovea assumes the default head camera's (−12° pitch: about a quarter of the way down).
pub fn first_cells(cols: u32, rows: u32, pattern: PerfPattern, horizon: Option<f32>) -> Vec<u32> {
    let horizon = horizon.unwrap_or(rows as f32 * 0.27);
    let mut out = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let checker = (c + r) % 2 == 0;
            let keep = match pattern {
                PerfPattern::Checker => checker,
                PerfPattern::All => true,
                PerfPattern::Fovea => {
                    let below = r as f32 + 0.5 - horizon;
                    let x = (c as f32 + 0.5) / cols as f32;
                    if below < -1.5 {
                        c % 2 == 0 && r % 2 == 0
                    } else if below <= 1.5 {
                        (x > 1.0 / 6.0 && x < 5.0 / 6.0) || checker
                    } else {
                        (x > 1.0 / 3.0 && x < 2.0 / 3.0) || checker
                    }
                }
            };
            if keep {
                out.push(r * cols + c);
            }
        }
    }
    out
}

/// What one perforated pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfReport {
    pub cells: usize,
    pub first: usize,
    pub seeds: usize,
    pub opened: usize,
    /// Tokens that went through the encoder over all passes.
    pub tokens_run: usize,
}

/// The heats of every label on the cells evaluated so far (NaN on closed cells), and each label's presence from the
/// pooled embedding of those cells.
fn evaluated_heats(sd: &crate::ml_bridge::SparseDense, texts: &LabelTexts, cfg: &LocateCfg, scale: f32, bias: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let cells: Vec<usize> = (0..sd.vectors.len()).filter(|&i| sd.vectors[i].is_some()).collect();
    let compact = crate::ml_bridge::DenseEmbedding {
        grid_h: 1,
        grid_w: cells.len(),
        vectors: cells.iter().map(|&i| sd.vectors[i].clone().unwrap_or_default()).collect(),
        attention: Vec::new(),
        pooled: sd.pooled.clone(),
    };
    label_heats(&compact, texts, cfg, scale, bias)
        .into_iter()
        .map(|(_, h, p)| {
            let mut full = vec![f32::NAN; sd.vectors.len()];
            for (k, &i) in cells.iter().enumerate() {
                full[i] = h.values[k];
            }
            (full, p)
        })
        .unzip()
}

/// The perforated dense pass of one frame on `cfg.grid`: the first cells of `perf.pattern`, the minesweeper rounds,
/// then each label's heat with the holes filled, and its presence from the pooled embedding of every evaluated cell.
pub fn perforated_heats(
    model: &crate::ml_bridge::Siglip2,
    image: &image::DynamicImage,
    texts: &LabelTexts,
    cfg: &LocateCfg,
    perf: &PerforatedCfg,
    horizon: Option<&Intrinsics>,
    scale: f32,
    bias: f32,
) -> Result<(Vec<(String, Heat, f32)>, PerfReport)> {
    let (gh, gw) = model.dense_grid(image.width(), image.height(), cfg.grid);
    let (cols, rows) = (gw as u32, gh as u32);
    let first = first_cells(cols, rows, perf.pattern, horizon.map(|i| horizon_row(i, rows)));
    let mut sd = model.embed_dense_sparse(image, cfg.grid, &first)?;
    let mut rep = PerfReport { cells: gh * gw, first: first.len(), ..PerfReport::default() };
    let (mut heats, mut presence) = evaluated_heats(&sd, texts, cfg, scale, bias);
    if let Some(x) = &perf.expand {
        let seed_cells = seeds(&heats, cols, rows, x);
        rep.seeds = seed_cells.len();
        for round in 0..x.rounds {
            let open: Vec<bool> = sd.vectors.iter().map(Option::is_some).collect();
            let add = cells_to_open(&heats, &open, &seed_cells, cols, rows, round, x);
            if add.is_empty() {
                break;
            }
            model.open_cells(&mut sd, &add)?;
            rep.opened += add.len();
            let next = evaluated_heats(&sd, texts, cfg, scale, bias);
            heats = next.0;
            presence = next.1;
        }
    }
    rep.tokens_run = sd.tokens_run;
    let open: Vec<bool> = sd.vectors.iter().map(Option::is_some).collect();
    for h in heats.iter_mut() {
        fill_holes(h, &open, cols, rows);
    }
    let out = texts.labels.iter().zip(heats).zip(presence).map(|(((id, _, _), values), p)| (id.clone(), Heat { cols, rows, values }, p)).collect();
    Ok((out, rep))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checkerboard_keeps_half_and_the_fovea_follows_the_horizon() {
        let checker = first_cells(18, 12, PerfPattern::Checker, None);
        assert_eq!(checker.len(), 108);
        let cam = Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0, roll_deg: 0.0 };
        let h = horizon_row(&cam, 12);
        assert!((h - 3.27).abs() < 0.05, "the −12° head camera sees the horizon a quarter down: {h}");
        let fovea = first_cells(18, 12, PerfPattern::Fovea, Some(h));
        let n = fovea.len();
        assert!((110..=150).contains(&n), "{n} of 216");
        let row = |r: u32| fovea.iter().filter(|&&i| i / 18 == r).count();
        assert!(row(0) <= 9 && row(1) == 0, "a quarter of the sky: {} {}", row(0), row(1));
        assert!(row(3) >= 14, "the horizon band is dense: {}", row(3));
        // Looking 30° further down the horizon leaves the frame and every row is near ground.
        let down = horizon_row(&Intrinsics { pitch_deg: -42.0, ..cam }, 12);
        assert!(down < 0.0, "{down}");
        let low = first_cells(18, 12, PerfPattern::Fovea, Some(down));
        assert!((0..12).all(|r| low.iter().filter(|&&i| i / 18 == r).count() >= 9), "no sky rows when the horizon is above the frame");
    }

    #[test]
    fn a_seed_opens_its_ring_and_strong_neighbours_grow_it_within_the_radius() {
        let (cols, rows) = (6u32, 5u32);
        let n = (cols * rows) as usize;
        let first = first_cells(cols, rows, PerfPattern::Checker, None);
        let mut open = vec![false; n];
        for &c in &first {
            open[c as usize] = true;
        }
        // One label with a peak at (2, 2) (open on the checkerboard), weaker heat around, nothing elsewhere.
        let mut h = vec![f32::NAN; n];
        for &c in &first {
            h[c as usize] = 0.05;
        }
        h[2 * 6 + 2] = 0.8;
        h[6 + 1] = 0.4;
        let cfg = ExpandCfg::default();
        let s = seeds(&[h.clone()], cols, rows, &cfg);
        assert_eq!(s, vec![14]);
        let ring = cells_to_open(&[h.clone()], &open, &s, cols, rows, 0, &cfg);
        assert_eq!(ring, vec![8, 13, 15, 20], "the closed cells of the 3 × 3 ring");
        for &c in &ring {
            open[c as usize] = true;
            h[c as usize] = 0.5;
        }
        let grow = cells_to_open(&[h.clone()], &open, &s, cols, rows, 1, &cfg);
        assert!(!grow.is_empty() && grow.iter().all(|&c| !open[c as usize]));
        assert!(grow.iter().all(|&c| ((c % 6) as i32 - 2).abs().max(((c / 6) as i32 - 2).abs()) <= 2), "within the radius: {grow:?}");
        // Below the seed threshold nothing starts.
        let quiet: Vec<f32> = h.iter().map(|v| v.min(0.1)).collect();
        assert!(seeds(&[quiet], cols, rows, &cfg).is_empty());
    }

    #[test]
    fn holes_take_their_lowest_evaluated_neighbour() {
        let open = vec![true, false, true, true];
        let mut v = vec![0.9, f32::NAN, 0.2, 0.5];
        fill_holes(&mut v, &open, 2, 2);
        assert_eq!(v, vec![0.9, 0.2, 0.2, 0.5]);
        let mut none = vec![f32::NAN; 4];
        fill_holes(&mut none, &[false; 4], 2, 2);
        assert_eq!(none, vec![0.0; 4]);
    }

    #[test]
    fn every_cell_through_the_perforated_path_is_the_full_pass() {
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = crate::ml_bridge::Siglip2Config { vision_hidden: 32, vision_intermediate: 64, vision_layers: 2, vision_heads: 4, patch_size: 4, max_patches: 16, vision_eps: 1e-6, text_hidden: 32, text_intermediate: 64, text_layers: 1, text_heads: 4, text_vocab: 16, text_max_len: 4, text_pad_id: 1, text_eps: 1e-6 };
        let model = crate::ml_bridge::Siglip2::from_var_builder(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev), &dev, None).unwrap();
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(24, 16, |x, y| image::Rgb([(x * 9 % 256) as u8, (y * 17 % 256) as u8, ((x + y) * 5 % 256) as u8])));
        let unit = |seed: f32| {
            let v: Vec<f32> = (0..32).map(|i| ((i as f32 + 1.0) * seed).sin()).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let texts = LabelTexts { labels: vec![("a".into(), unit(0.7), None), ("b".into(), unit(1.3), None)], background: vec![unit(2.1)] };
        let lc = LocateCfg { scoring: crate::vision_compat::locate::LocateScoring::Contrast, grid: crate::ml_bridge::preprocess::DenseGrid::Pyramid { short_side: 3 }, ..LocateCfg::default() };
        let full = model.embed_dense_batch_grid(&[image.clone()], lc.grid).unwrap().pop().unwrap();
        let want = label_heats(&full, &texts, &lc, 10.0, -1.0);
        let all = PerforatedCfg { pattern: PerfPattern::All, expand: None };
        let (got, rep) = perforated_heats(&model, &image, &texts, &lc, &all, None, 10.0, -1.0).unwrap();
        assert_eq!(rep.first, rep.cells);
        assert_eq!(got, want, "the same heat and presence, bit for bit");
        // A checkerboard evaluates half and fills the rest from evaluated neighbours, never above them.
        let half = PerforatedCfg { pattern: PerfPattern::Checker, expand: None };
        let (got, rep) = perforated_heats(&model, &image, &texts, &lc, &half, None, 10.0, -1.0).unwrap();
        assert_eq!(rep.tokens_run, rep.first);
        assert!(rep.first * 2 >= rep.cells && rep.first < rep.cells);
        for (_, h, _) in &got {
            assert!(h.values.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn the_tag_names_the_settings_back() {
        for cfg in [
            PerforatedCfg::default(),
            PerforatedCfg { pattern: PerfPattern::Checker, expand: None },
            PerforatedCfg { pattern: PerfPattern::Fovea, expand: None },
            PerforatedCfg { pattern: PerfPattern::Checker, expand: Some(ExpandCfg::default()) },
        ] {
            assert_eq!(PerforatedCfg::from_tag(&cfg.tag()), Some(cfg), "{}", cfg.tag());
        }
        assert_eq!(PerforatedCfg::default().tag(), "perf");
        assert_eq!(PerforatedCfg::from_tag("perf-ring"), None);
    }
}
