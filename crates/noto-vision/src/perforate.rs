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

use crate::camera::Intrinsics;
use crate::locate::{label_heats, Heat, LabelTexts, LocateCfg};

/// Which patches the first pass evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfPattern {
    /// Every other cell, a checkerboard (half the grid).
    Checker,
    /// Dense around the horizon where far, small objects stand; the middle of the near ground kept and its sides a
    /// checkerboard; a quarter of the sky above.
    Fovea,
    /// Every cell (the full pass through the perforated path; for checks).
    All,
}

/// The minesweeper step: seeds are each label's 3 × 3 local maxima of the first pass's heat at or above `seed_min`
/// (at most `seeds_per_label` per label, strongest first); every closed cell within one cell of a seed opens, and for
/// `rounds` − 1 more rounds the closed neighbours of opened cells whose heat reaches `grow_min` open too, never farther
/// than `radius` cells from their seed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpandCfg {
    pub seed_min: f32,
    pub grow_min: f32,
    pub radius: u32,
    pub seeds_per_label: usize,
    pub rounds: u32,
}

impl Default for ExpandCfg {
    fn default() -> Self {
        ExpandCfg { seed_min: 0.15, grow_min: 0.30, radius: 2, seeds_per_label: 2, rounds: 2 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerforatedCfg {
    pub pattern: PerfPattern,
    #[serde(default)]
    pub expand: Option<ExpandCfg>,
}

impl Default for PerforatedCfg {
    fn default() -> Self {
        PerforatedCfg { pattern: PerfPattern::Fovea, expand: Some(ExpandCfg::default()) }
    }
}

impl PerforatedCfg {
    /// The mode part: `perf` for the default, `perf-<pattern>[-x]` otherwise (`-x`: with the default expansion).
    pub fn tag(&self) -> String {
        if *self == PerforatedCfg::default() {
            return "perf".into();
        }
        let p = match self.pattern {
            PerfPattern::Checker => "checker",
            PerfPattern::Fovea => "fovea",
            PerfPattern::All => "all",
        };
        match self.expand {
            None => format!("perf-{p}"),
            Some(e) if e == ExpandCfg::default() => format!("perf-{p}-x"),
            Some(e) => format!("perf-{p}-x{}-{}-{}", (e.seed_min * 100.0).round(), (e.grow_min * 100.0).round(), e.radius),
        }
    }

    /// The settings `tag` names (the default expansion for `-x`; other expansions are not named back).
    pub fn from_tag(tag: &str) -> Option<PerforatedCfg> {
        let rest = tag.strip_prefix("perf")?;
        if rest.is_empty() {
            return Some(PerforatedCfg::default());
        }
        let mut parts = rest.strip_prefix('-')?.split('-');
        let pattern = match parts.next()? {
            "checker" => PerfPattern::Checker,
            "fovea" => PerfPattern::Fovea,
            "all" => PerfPattern::All,
            _ => return None,
        };
        let expand = match parts.next() {
            None => None,
            Some("x") => Some(ExpandCfg::default()),
            Some(_) => return None,
        };
        Some(PerforatedCfg { pattern, expand })
    }
}

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

fn neighbours(c: u32, r: u32, cols: u32, rows: u32, reach: i32) -> impl Iterator<Item = (u32, u32)> {
    (-reach..=reach).flat_map(move |dr| (-reach..=reach).map(move |dc| (dc, dr))).filter_map(move |(dc, dr)| {
        let (cc, rr) = (c as i32 + dc, r as i32 + dr);
        ((dc, dr) != (0, 0) && cc >= 0 && rr >= 0 && cc < cols as i32 && rr < rows as i32).then_some((cc as u32, rr as u32))
    })
}

/// The seeds of the minesweeper step: per label (`heats[label][cell]`, NaN on closed cells), the evaluated cells that
/// are 3 × 3 local maxima among evaluated cells with heat ≥ `seed_min`, strongest first, at most `seeds_per_label`.
pub fn seeds(heats: &[Vec<f32>], cols: u32, rows: u32, cfg: &ExpandCfg) -> Vec<u32> {
    let mut out = Vec::new();
    for h in heats {
        let mut local: Vec<(u32, f32)> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .filter_map(|(c, r)| {
                let v = h[(r * cols + c) as usize];
                let peak = v.is_finite() && v >= cfg.seed_min && neighbours(c, r, cols, rows, 1).all(|(cc, rr)| !(h[(rr * cols + cc) as usize] > v));
                peak.then_some((r * cols + c, v))
            })
            .collect();
        local.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.extend(local.into_iter().take(cfg.seeds_per_label).map(|(i, _)| i));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The closed cells to open next: those within one cell of a seed (`round` 0), or (later rounds) of an evaluated cell
/// within `radius` of a seed whose heat for any label reaches `grow_min`.
pub fn cells_to_open(heats: &[Vec<f32>], open: &[bool], seed_cells: &[u32], cols: u32, rows: u32, round: u32, cfg: &ExpandCfg) -> Vec<u32> {
    let mut want = vec![false; open.len()];
    let near_seed = |c: u32, r: u32| seed_cells.iter().any(|&s| (c as i32 - (s % cols) as i32).abs().max((r as i32 - (s / cols) as i32).abs()) <= cfg.radius as i32);
    for &s in seed_cells {
        let (c, r) = (s % cols, s / cols);
        if round == 0 {
            for (cc, rr) in neighbours(c, r, cols, rows, 1) {
                want[(rr * cols + cc) as usize] = true;
            }
        }
    }
    if round > 0 {
        for r in 0..rows {
            for c in 0..cols {
                let i = (r * cols + c) as usize;
                if !open[i] || !near_seed(c, r) || !heats.iter().any(|h| h[i] >= cfg.grow_min) {
                    continue;
                }
                for (cc, rr) in neighbours(c, r, cols, rows, 1) {
                    if near_seed(cc, rr) {
                        want[(rr * cols + cc) as usize] = true;
                    }
                }
            }
        }
    }
    (0..open.len()).filter(|&i| want[i] && !open[i]).map(|i| i as u32).collect()
}

/// Fills the closed cells of a heat grid with the lowest heat of their evaluated 8-neighbours (else the lowest
/// evaluated heat), so that no peak and no box grows out of a hole.
pub fn fill_holes(values: &mut [f32], open: &[bool], cols: u32, rows: u32) {
    let floor = values.iter().zip(open).filter(|(_, o)| **o).map(|(v, _)| *v).fold(f32::INFINITY, f32::min);
    let floor = if floor.is_finite() { floor } else { 0.0 };
    let filled: Vec<f32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (c, r)))
        .map(|(c, r)| {
            let i = (r * cols + c) as usize;
            if open[i] {
                return values[i];
            }
            neighbours(c, r, cols, rows, 1).map(|(cc, rr)| (rr * cols + cc) as usize).filter(|&j| open[j]).map(|j| values[j]).fold(f32::INFINITY, f32::min)
        })
        .map(|v| if v.is_finite() { v } else { floor })
        .collect();
    values.copy_from_slice(&filled);
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
fn evaluated_heats(sd: &noto_ml::SparseDense, texts: &LabelTexts, cfg: &LocateCfg, scale: f32, bias: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let cells: Vec<usize> = (0..sd.vectors.len()).filter(|&i| sd.vectors[i].is_some()).collect();
    let compact = noto_ml::DenseEmbedding {
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
    model: &noto_ml::Siglip2,
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
    if let Some(x) = &perf.expand {
        let (mut heats, _) = evaluated_heats(&sd, texts, cfg, scale, bias);
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
            heats = evaluated_heats(&sd, texts, cfg, scale, bias).0;
        }
    }
    rep.tokens_run = sd.tokens_run;
    let open: Vec<bool> = sd.vectors.iter().map(Option::is_some).collect();
    let (mut heats, presence) = evaluated_heats(&sd, texts, cfg, scale, bias);
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
        let cfg = noto_ml::Siglip2Config { vision_hidden: 32, vision_intermediate: 64, vision_layers: 2, vision_heads: 4, patch_size: 4, max_patches: 16, vision_eps: 1e-6, text_hidden: 32, text_intermediate: 64, text_layers: 1, text_heads: 4, text_vocab: 16, text_max_len: 4, text_pad_id: 1, text_eps: 1e-6 };
        let model = noto_ml::Siglip2::from_var_builder(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev), &dev, None).unwrap();
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(24, 16, |x, y| image::Rgb([(x * 9 % 256) as u8, (y * 17 % 256) as u8, ((x + y) * 5 % 256) as u8])));
        let unit = |seed: f32| {
            let v: Vec<f32> = (0..32).map(|i| ((i as f32 + 1.0) * seed).sin()).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let texts = LabelTexts { labels: vec![("a".into(), unit(0.7), None), ("b".into(), unit(1.3), None)], background: vec![unit(2.1)] };
        let lc = LocateCfg { scoring: crate::locate::LocateScoring::Contrast, grid: noto_ml::preprocess::DenseGrid::Pyramid { short_side: 3 }, ..LocateCfg::default() };
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
