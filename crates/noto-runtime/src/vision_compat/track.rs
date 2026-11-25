use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::frame::{DepthMap, Mask, Rgb};

pub use lv_spatial::track::{
    PointPrompt, StereoMaskTracker, StereoSpatialOut, StereoSpatialPipeline, StereoSpatialTier,
    TrackedMask,
};

pub trait MaskTracker: Send {
    fn mode(&self) -> &str;
    fn reset(&mut self);
    fn update(&mut self, rgb: &Rgb, depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>>;
    fn prepare(&mut self, _frames: &[&Rgb]) -> Result<()> {
        Ok(())
    }
    /// Samples masks on the 1 : 2 : 4 pyramid's grid while `cfg` is set (`None`: the tracker's own downscale).
    fn set_pyramid(&mut self, _cfg: Option<&super::pyramid::PyramidCfg>) {}
    fn device(&self) -> String {
        "cpu".into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionCfg {
    pub downscale: u32,
    pub color_tol: f32,
    pub depth_jump: f32,
    pub min_area_frac: f32,
    pub max_area_frac: f32,
    pub search_px: i32,
    pub refind_tol: f32,
    pub max_tracks: usize,
    pub max_misses: u32,
}

impl Default for RegionCfg {
    fn default() -> Self {
        RegionCfg {
            downscale: 2,
            color_tol: 40.0,
            depth_jump: 0.06,
            min_area_frac: 0.0004,
            max_area_frac: 0.25,
            search_px: 14,
            refind_tol: 46.0,
            max_tracks: 8,
            max_misses: 2,
        }
    }
}

struct Live {
    id: u32,
    label: String,
    mask: Mask,
    color: [f32; 3],
    cx: f32,
    cy: f32,
    age: u32,
    misses: u32,
}

pub struct RegionTracker {
    pub cfg: RegionCfg,
    /// The size each label's region may grow to (`with_extent`); `None`: only `max_area_frac` bounds it.
    pub extent: Option<ExtentClamp>,
    tracks: Vec<Live>,
    next_id: u32,
}

/// How far a region may grow from its seed: no farther than `radius_m[label]` (m) on the item at the seed's depth,
/// i.e. f·R/Z pixels each way — so a region cannot run onto a neighbour beyond the item's own size, however alike
/// their colour and depth. A label without a radius, or a seed without depth, grows as before.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtentClamp {
    /// The camera's focal length in frame pixels.
    pub focal_px: f32,
    pub radius_m: std::collections::HashMap<String, f32>,
}

impl Default for RegionTracker {
    fn default() -> Self {
        RegionTracker::new(RegionCfg::default())
    }
}

struct Work {
    w: u32,
    h: u32,
    rgb: Vec<[f32; 3]>,
    depth: Option<Vec<f32>>,
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn median3(px: &[[f32; 3]], w: u32, h: u32) -> Vec<[f32; 3]> {
    let mut out = px.to_vec();
    let mut win: [Vec<f32>; 3] = [Vec::with_capacity(9), Vec::with_capacity(9), Vec::with_capacity(9)];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            for c in win.iter_mut() {
                c.clear();
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (sx, sy) = ((x + dx).clamp(0, w as i32 - 1), (y + dy).clamp(0, h as i32 - 1));
                    let p = px[(sy as u32 * w + sx as u32) as usize];
                    for k in 0..3 {
                        win[k].push(p[k]);
                    }
                }
            }
            let mut m = [0.0f32; 3];
            for k in 0..3 {
                win[k].sort_by(f32::total_cmp);
                m[k] = win[k][4];
            }
            out[(y as u32 * w + x as u32) as usize] = m;
        }
    }
    out
}

impl Work {
    fn new(rgb: &Rgb, depth: Option<&DepthMap>, k: u32) -> Work {
        let k = k.max(1);
        let (w, h) = ((rgb.width / k).max(1), (rgb.height / k).max(1));
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let mut s = [0.0f32; 3];
                let mut n = 0.0;
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = ((x * k + dx).min(rgb.width - 1), (y * k + dy).min(rgb.height - 1));
                        let p = rgb.px(sx, sy);
                        s[0] += p[0] as f32;
                        s[1] += p[1] as f32;
                        s[2] += p[2] as f32;
                        n += 1.0;
                    }
                }
                px.push([s[0] / n, s[1] / n, s[2] / n]);
            }
        }
        let px = median3(&px, w, h);
        let depth = depth.map(|d| {
            let mut v = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    v.push(d.at_scaled((x * k) as f32 + k as f32 * 0.5, (y * k) as f32 + k as f32 * 0.5, rgb.width, rgb.height));
                }
            }
            v
        });
        Work { w, h, rgb: px, depth }
    }

    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.w + x) as usize
    }

    /// The box (x0, y0, x1, y1, work pixels) a region seeded at `seed` may fill: `half` work pixels each way.
    fn bounds(&self, seed: (u32, u32), half: Option<f32>) -> [u32; 4] {
        match half {
            Some(r) if r.is_finite() => {
                let r = r.max(1.0);
                let (x, y) = (seed.0 as f32, seed.1 as f32);
                [(x - r).max(0.0) as u32, (y - r).max(0.0) as u32, ((x + r) as u32).min(self.w - 1), ((y + r) as u32).min(self.h - 1)]
            }
            _ => [0, 0, self.w - 1, self.h - 1],
        }
    }

    fn grow(&self, seed: (u32, u32), cfg: &RegionCfg, half: Option<f32>) -> (Mask, [f32; 3], bool) {
        let [bx0, by0, bx1, by1] = self.bounds(seed, half);
        let mut mask = Mask::empty(self.w, self.h);
        let max_px = (cfg.max_area_frac * (self.w * self.h) as f32) as u32 + 1;
        let mut mean = self.rgb[self.idx(seed.0, seed.1)];
        let mut n = 1.0f32;
        let mut stack = vec![seed];
        mask.set(seed.0, seed.1);
        let mut count = 1;
        while let Some((x, y)) = stack.pop() {
            let here = self.idx(x, y);
            let neigh = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
            for (nx, ny) in neigh {
                if nx >= self.w || ny >= self.h || mask.get(nx, ny) || nx < bx0 || nx > bx1 || ny < by0 || ny > by1 {
                    continue;
                }
                let j = self.idx(nx, ny);
                let c = self.rgb[j];
                if let Some(d) = &self.depth {
                    let (a, b) = (d[here], d[j]);
                    if !b.is_finite() {
                        continue;
                    }
                    if a.is_finite() && (a - b).abs() > cfg.depth_jump * a.min(b).max(0.1) {
                        continue;
                    }
                }
                if dist(c, mean) > cfg.color_tol {
                    continue;
                }
                mask.set(nx, ny);
                count += 1;
                n += 1.0;
                for k in 0..3 {
                    mean[k] += (c[k] - mean[k]) / n;
                }
                if count >= max_px {
                    return (mask, mean, true);
                }
                stack.push((nx, ny));
            }
        }
        (mask, mean, false)
    }

    fn local_median(&self, x: f32, y: f32, r: i32) -> Option<[f32; 3]> {
        let (cx, cy) = (x as i32, y as i32);
        let mut ch: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || py < 0 || px >= self.w as i32 || py >= self.h as i32 {
                    continue;
                }
                let c = self.rgb[self.idx(px as u32, py as u32)];
                for k in 0..3 {
                    ch[k].push(c[k]);
                }
            }
        }
        if ch[0].is_empty() {
            return None;
        }
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            ch[k].sort_by(f32::total_cmp);
            out[k] = ch[k][ch[k].len() / 2];
        }
        Some(out)
    }

    fn best_seed(&self, x: f32, y: f32, color: Option<[f32; 3]>, r: i32) -> Option<((u32, u32), f32)> {
        let (cx, cy) = (x as i32, y as i32);
        let mut best: Option<((u32, u32), f32)> = None;
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || py < 0 || px >= self.w as i32 || py >= self.h as i32 {
                    continue;
                }
                let i = self.idx(px as u32, py as u32);
                if self.depth.as_ref().map_or(false, |d| !d[i].is_finite()) {
                    continue;
                }
                let colour_cost = color.map_or(0.0, |c| dist(self.rgb[i], c));
                let cost = colour_cost + 0.6 * ((dx * dx + dy * dy) as f32).sqrt();
                if best.map_or(true, |(_, b)| cost < b) {
                    best = Some(((px as u32, py as u32), cost));
                }
            }
        }
        best
    }
}

impl RegionTracker {
    pub fn new(cfg: RegionCfg) -> RegionTracker {
        RegionTracker { cfg, extent: None, tracks: Vec::new(), next_id: 1 }
    }

    /// This tracker with each label's region kept to the item's size (`ExtentClamp`).
    pub fn with_extent(mut self, extent: ExtentClamp) -> RegionTracker {
        self.extent = Some(extent);
        self
    }

    /// The half-size, in work pixels, a `label` region seeded at `seed` may take (`None`: unbounded).
    fn half(&self, work: &Work, label: &str, seed: (u32, u32), k: u32) -> Option<f32> {
        let e = self.extent.as_ref()?;
        let r = *e.radius_m.get(label)?;
        let z = work.depth.as_ref()?[work.idx(seed.0, seed.1)];
        (z.is_finite() && z > 0.0).then(|| e.focal_px * r / z / k as f32)
    }

    fn accept(&self, work: &Work, m: &Mask, capped: bool) -> bool {
        let frac = m.area() as f32 / (work.w * work.h) as f32;
        !capped && frac >= self.cfg.min_area_frac && frac <= self.cfg.max_area_frac
    }

    fn out(&self, t: &Live, k: u32, frame: (u32, u32), iou_prev: Option<f32>, propagated: bool, score: f32) -> TrackedMask {
        let b = t.mask.bbox().unwrap_or([0, 0, 0, 0]);
        let kf = k as f32;
        TrackedMask {
            id: t.id,
            label: t.label.clone(),
            mask: t.mask.clone(),
            cx: t.cx * kf,
            cy: t.cy * kf,
            bbox: [b[0] as f32 * kf, b[1] as f32 * kf, (b[2] + 1) as f32 * kf, (b[3] + 1) as f32 * kf],
            area_frac: t.mask.area() as f32 / (frame.0 / k * (frame.1 / k)).max(1) as f32,
            score,
            iou_prev,
            age: t.age,
            propagated,
        }
    }
}

impl MaskTracker for RegionTracker {
    fn mode(&self) -> &str {
        if self.extent.is_some() {
            "region-grow+clamp"
        } else {
            "region-grow"
        }
    }

    fn reset(&mut self) {
        self.tracks.clear();
    }

    fn update(&mut self, rgb: &Rgb, depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>> {
        if self.tracks.is_empty() && prompts.is_empty() {
            return Ok(Vec::new());
        }
        let k = self.cfg.downscale.max(1);
        let work = Work::new(rgb, depth, k);
        let kf = k as f32;
        let mut out = Vec::new();
        let mut used = vec![false; prompts.len()];
        let old = std::mem::take(&mut self.tracks);
        for mut t in old {
            let near = |p: &PointPrompt| -> bool {
                let (x, y) = (p.px / kf, p.py / kf);
                let (xi, yi) = (x as u32, y as u32);
                if xi < t.mask.width && yi < t.mask.height && t.mask.get(xi, yi) {
                    return true;
                }
                let Some(b) = t.mask.bbox() else { return false };
                let (bw, bh) = ((b[2] - b[0] + 1) as f32, (b[3] - b[1] + 1) as f32);
                let (gx, gy) = ((0.5 * bw).max(6.0), (0.5 * bh).max(6.0));
                x >= b[0] as f32 - gx && x <= b[2] as f32 + gx && y >= b[1] as f32 - gy && y <= b[3] as f32 + gy
            };
            let prompt_hit = prompts.iter().enumerate().find(|(i, p)| !used[*i] && p.label == t.label && near(p)).map(|(i, p)| (i, p.clone()));
            let (seed, propagated) = match &prompt_hit {
                Some((i, p)) => {
                    used[*i] = true;
                    (work.best_seed(p.px / kf, p.py / kf, Some(t.color), 2).map(|s| s.0), false)
                }
                None => {
                    let s = work.best_seed(t.cx, t.cy, Some(t.color), self.cfg.search_px);
                    (s.filter(|(seed, _)| dist(work.rgb[work.idx(seed.0, seed.1)], t.color) <= self.cfg.refind_tol).map(|s| s.0), true)
                }
            };
            let Some(seed) = seed else { continue };
            if propagated && t.misses >= self.cfg.max_misses {
                continue;
            }
            let half = self.half(&work, &t.label, seed, k);
            let (mask, color, capped) = work.grow(seed, &self.cfg, half);
            if !self.accept(&work, &mask, capped) {
                continue;
            }
            t.misses = if propagated { t.misses + 1 } else { 0 };
            let iou = t.mask.iou(&mask);
            let coherence = (1.0 - dist(color, t.color) / (2.0 * self.cfg.refind_tol)).clamp(0.0, 1.0);
            let score = if propagated { 0.6 * coherence } else { 0.4 + 0.6 * coherence };
            let (cx, cy) = mask.centroid().expect("accepted masks are not empty");
            t.mask = mask;
            t.color = [0.7 * t.color[0] + 0.3 * color[0], 0.7 * t.color[1] + 0.3 * color[1], 0.7 * t.color[2] + 0.3 * color[2]];
            t.cx = cx;
            t.cy = cy;
            t.age += 1;
            out.push(self.out(&t, k, (rgb.width, rgb.height), Some(iou), propagated, score));
            self.tracks.push(t);
        }
        for (i, p) in prompts.iter().enumerate() {
            if used[i] || self.tracks.len() >= self.cfg.max_tracks {
                continue;
            }
            let on_other = self.tracks.iter().any(|t| {
                let (x, y) = ((p.px / kf) as u32, (p.py / kf) as u32);
                x < t.mask.width && y < t.mask.height && t.mask.get(x, y)
            });
            if on_other {
                continue;
            }
            let around = work.local_median(p.px / kf, p.py / kf, 2);
            let Some((seed, _)) = work.best_seed(p.px / kf, p.py / kf, around, 2) else { continue };
            let half = self.half(&work, &p.label, seed, k);
            let (mask, color, capped) = work.grow(seed, &self.cfg, half);
            if !self.accept(&work, &mask, capped) {
                continue;
            }
            let (cx, cy) = mask.centroid().expect("accepted masks are not empty");
            let t = Live { id: self.next_id, label: p.label.clone(), mask, color, cx, cy, age: 1, misses: 0 };
            self.next_id += 1;
            out.push(self.out(&t, k, (rgb.width, rgb.height), None, false, 0.4 + 0.6 * p.score.clamp(0.0, 1.0)));
            self.tracks.push(t);
        }
        Ok(out)
    }
}

/// Mask tracker backed by `lv_spatial::StereoSpatialTier` and pure `lv_spatial::mask_ops` post-processing.
pub struct StereoTierTracker {
    tier: StereoSpatialTier,
    inner: RegionTracker,
}

impl StereoTierTracker {
    pub fn new(tier: StereoSpatialTier) -> Self {
        Self {
            tier,
            inner: RegionTracker::new(RegionCfg::default()),
        }
    }

    pub fn with_extent(mut self, extent: ExtentClamp) -> Self {
        self.inner = self.inner.with_extent(extent);
        self
    }

    pub fn tier(&self) -> StereoSpatialTier {
        self.tier
    }

    pub fn set_tier(&mut self, tier: StereoSpatialTier) {
        self.tier = tier;
    }
}

impl MaskTracker for StereoTierTracker {
    fn mode(&self) -> &str {
        match self.tier {
            StereoSpatialTier::StereoFastTick => "stereo-fast-tick",
            StereoSpatialTier::StereoMetricGrasp => "stereo-metric-grasp",
            StereoSpatialTier::StereoStageLive => "stereo-stage-live",
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn update(&mut self, rgb: &Rgb, depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>> {
        let mut masks = self.inner.update(rgb, depth, prompts)?;
        if matches!(self.tier, StereoSpatialTier::StereoMetricGrasp) {
            let params = lv_spatial::mask_ops::postprocess::PostprocessParams {
                holes: 2,
                dots: 2,
                border_fix: 0,
                grow: 0,
            };
            for t in &mut masks {
                if let Ok(refined) = lv_spatial::mask_ops::postprocess::apply(&t.mask.to_gray(), &params) {
                    t.mask = Mask::from_gray(&refined);
                }
            }
        }
        Ok(masks)
    }

    fn set_pyramid(&mut self, cfg: Option<&super::pyramid::PyramidCfg>) {
        self.inner.set_pyramid(cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision_compat::frame::DepthKind;

    fn scene(x0: u32) -> (Rgb, DepthMap) {
        let (w, h) = (80u32, 60u32);
        let mut rgb = Vec::new();
        let mut depth = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let block = x >= x0 && x < x0 + 20 && (20..40).contains(&y);
                let fleck = block && (x + y) % 7 == 0;
                rgb.extend_from_slice(&if fleck { [180, 120, 90] } else if block { [128, 128, 132] } else { [70, 130, 60] });
                depth.push(if block { 4.0 } else { 6.0 + y as f32 * 0.01 });
            }
        }
        (Rgb::new(w, h, rgb), DepthMap { width: w, height: h, kind: DepthKind::Metric, data: depth })
    }

    #[test]
    fn a_prompt_on_the_block_segments_the_block_and_follows_it() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let first = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 40.0, py: 30.0, score: 0.8 }]).unwrap();
        assert_eq!(first.len(), 1);
        let m = &first[0];
        assert!(m.mask.area() >= 300 && m.mask.area() <= 400, "area {}", m.mask.area());
        assert!((m.cx - 40.0).abs() < 2.0 && (m.cy - 30.0).abs() < 2.0, "{} {}", m.cx, m.cy);
        let (rgb2, depth2) = scene(36);
        let next = t.update(&rgb2, Some(&depth2), &[]).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].id, m.id);
        assert!(next[0].propagated);
        assert!((next[0].cx - 46.0).abs() < 2.0, "{}", next[0].cx);
        let iou = next[0].iou_prev.unwrap();
        assert!(iou > 0.4 && iou < 0.9, "moved block overlaps partly: {iou}");
    }

    #[test]
    fn a_track_without_prompts_is_dropped_after_its_misses_and_a_nearby_prompt_keeps_it() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let first = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 40.0, py: 30.0, score: 0.8 }]).unwrap();
        let id = first[0].id;
        let (rgb2, depth2) = scene(44);
        let near = t.update(&rgb2, Some(&depth2), &[PointPrompt { label: "stone".into(), px: 53.0, py: 30.0, score: 0.8 }]).unwrap();
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].id, id, "same object, same id");
        assert!(!near[0].propagated);
        for _ in 0..2 {
            assert_eq!(t.update(&rgb2, Some(&depth2), &[]).unwrap().len(), 1);
        }
        assert!(t.update(&rgb2, Some(&depth2), &[]).unwrap().is_empty(), "dropped after two keyframes without support");
    }

    #[test]
    fn the_item_size_keeps_a_region_off_a_touching_look_alike() {
        // Two grey blocks side by side at the same depth (an item and a boulder touching it): the region runs over
        // both; kept to the item's size at its depth it stays on the one the prompt is on.
        let (w, h) = (80u32, 60u32);
        let mut rgb = Vec::new();
        let mut depth = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let block = (20..60).contains(&x) && (20..40).contains(&y);
                rgb.extend_from_slice(&if block { [128, 128, 132] } else { [70, 130, 60] });
                depth.push(if block { 4.0 } else { 6.0 + y as f32 * 0.01 });
            }
        }
        let (rgb, depth) = (Rgb::new(w, h, rgb), DepthMap { width: w, height: h, kind: DepthKind::Metric, data: depth });
        let prompt = [PointPrompt { label: "stone".into(), px: 30.0, py: 30.0, score: 0.8 }];
        let mut free_tracker = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        assert_eq!(free_tracker.mode(), "region-grow");
        let free = free_tracker.update(&rgb, Some(&depth), &prompt).unwrap();
        assert!(free[0].mask.area() >= 780, "both blocks (less the corners the median filter rounds): {}", free[0].mask.area());
        // f = 80 px, R = 0.5 m at 4 m: 10 px each way from the seed.
        let clamp = ExtentClamp { focal_px: 80.0, radius_m: [("stone".to_string(), 0.5)].into_iter().collect() };
        let mut kept_tracker = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() }).with_extent(clamp.clone());
        assert_eq!(kept_tracker.mode(), "region-grow+clamp");
        let kept = kept_tracker.update(&rgb, Some(&depth), &prompt).unwrap();
        let b = kept[0].bbox;
        assert!(b[0] >= 20.0 && b[2] <= 41.5 && kept[0].mask.area() < 500, "{b:?} {}", kept[0].mask.area());
        // A label with no size grows as before.
        let other = [PointPrompt { label: "tree".into(), ..prompt[0].clone() }];
        let tree = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() }).with_extent(clamp).update(&rgb, Some(&depth), &other).unwrap();
        assert!(tree[0].mask.area() >= 780, "{}", tree[0].mask.area());
    }

    #[test]
    fn a_prompt_on_the_ground_alone_is_too_big_to_be_an_object() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let got = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 5.0, py: 55.0, score: 0.8 }]).unwrap();
        assert!(got.is_empty(), "the grass region covers most of the frame and is refused");
    }
}
