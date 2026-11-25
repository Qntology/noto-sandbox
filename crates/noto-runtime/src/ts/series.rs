//! Timestamped time-series sample specification, regularization, and span alignment
//! (Phase 3 – Part 1 · D1 & D4).

use serde::{Deserialize, Serialize};

/// Aggregation and interpolation semantics for a time-series channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    /// Instantaneous state variable (dedup: last, LOCF up to `hold`).
    Gauge,
    /// Interval maximum (dedup: max, bin max, no LOCF).
    Peak,
    /// Interval minimum (dedup: min, bin min, no LOCF).
    Trough,
    /// Monotone cumulative counter (dedup: max, linear interpolation, never extrapolate).
    Counter,
    /// Event-scoped evaluation score (dedup: mean, bin mean, no LOCF).
    Score,
    /// Discrete event marker count (dedup: sum, bin sum).
    Event,
}

/// Quality bitflags carried on every [`TsSample`].
pub mod q {
    pub const OBSERVED: u16 = 1;
    pub const HELD: u16 = 2;
    pub const IMPUTED: u16 = 4;
    pub const GAP_BEFORE: u16 = 8;
    pub const PHASE_BREAK: u16 = 16;
    pub const RESET: u16 = 32;
}

/// Compact 12-byte timestamped sample (16 bytes on disk with a 4-byte key id).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TsSample {
    /// Tick offset from run start.
    pub dtick: u32,
    /// Measured value.
    pub value: f32,
    /// Sub-tick offset in milliseconds (e.g. 20 ms dynamics rollout samples).
    pub sub_ms: u16,
    /// Quality bitmask (`q::*`).
    pub q: u16,
}

impl TsSample {
    pub fn observed(dtick: u32, value: f32) -> Self {
        Self { dtick, value, sub_ms: 0, q: q::OBSERVED }
    }

    pub fn with_q(dtick: u32, value: f32, q_bits: u16) -> Self {
        Self { dtick, value, sub_ms: 0, q: q_bits }
    }
}

/// Static specification of a registered time-series channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ChannelSpec {
    pub id: &'static str,
    pub kind: ChannelKind,
    pub unit: &'static str,
    pub period: u32,
    pub hold: u32,
    pub cross_phase: bool,
    pub decision_ok: bool,
    pub crew: bool,
}

/// Regularized evenly-spaced grid returned by [`regularize`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegSeries {
    pub t0: u32,
    pub period: u32,
    pub values: Vec<f64>,
    pub mask: Vec<bool>,
    /// Contiguous valid index ranges `[start, end)` that do not cross gaps or phase breaks.
    pub segments: Vec<(usize, usize)>,
    /// Fraction of valid grid points produced by LOCF or linear interpolation.
    pub imputed_share: f64,
}

/// Half-open phase span reference `[start, end)` in relative ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanRef {
    pub span_id: u32,
    pub start: u32,
    pub end: Option<u32>,
}

/// Samples aligned to a single phase span in both `tau` (ticks since start) and
/// normalized phase progress `u ∈ [0, 1]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpanView {
    pub span_id: u32,
    pub start: u32,
    pub end: u32,
    pub tau: Vec<u32>,
    pub u: Vec<f64>,
    pub values: Vec<f64>,
    pub mask: Vec<bool>,
    pub raw_count: usize,
    pub counter_delta: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct DedupPoint {
    dtick: u32,
    value: f64,
    count: u32,
    q: u16,
}

/// Deduplicate same-tick samples according to `kind` and drop out-of-order (late) samples.
/// Returns `(deduped_points, late_count)`.
pub fn dedup_samples(s: &[TsSample], kind: ChannelKind) -> (Vec<(u32, f64, u16)>, usize) {
    let (pts, late) = dedup_internal(s, kind);
    (pts.into_iter().map(|p| (p.dtick, p.value, p.q)).collect(), late)
}

fn dedup_internal(s: &[TsSample], kind: ChannelKind) -> (Vec<DedupPoint>, usize) {
    let mut out: Vec<DedupPoint> = Vec::with_capacity(s.len());
    let mut late = 0usize;
    for sample in s {
        let v = sample.value as f64;
        if !v.is_finite() {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if sample.dtick < last.dtick {
                late += 1;
                continue;
            }
            if sample.dtick == last.dtick {
                last.q |= sample.q;
                match kind {
                    ChannelKind::Gauge => {
                        last.value = v;
                        last.count += 1;
                    }
                    ChannelKind::Peak | ChannelKind::Counter => {
                        if v > last.value {
                            last.value = v;
                        }
                        last.count += 1;
                    }
                    ChannelKind::Trough => {
                        if v < last.value {
                            last.value = v;
                        }
                        last.count += 1;
                    }
                    ChannelKind::Score => {
                        let sum = last.value * last.count as f64 + v;
                        last.count += 1;
                        last.value = sum / last.count as f64;
                    }
                    ChannelKind::Event => {
                        last.value += v;
                        last.count += 1;
                    }
                }
                continue;
            }
        }
        out.push(DedupPoint { dtick: sample.dtick, value: v, count: 1, q: sample.q });
    }
    (out, late)
}

fn has_break_in(breaks: &[u32], lo_excl: u32, hi_incl: u32) -> bool {
    let idx = breaks.partition_point(|&b| b <= lo_excl);
    idx < breaks.len() && breaks[idx] <= hi_incl
}

fn last_break_in(breaks: &[u32], lo_excl: u32, hi_incl: u32) -> Option<u32> {
    let idx = breaks.partition_point(|&b| b <= hi_incl);
    if idx == 0 {
        return None;
    }
    let b = breaks[idx - 1];
    (b > lo_excl).then_some(b)
}

/// Regularize a timestamped sample stream onto the uniform grid `g_k = t0 + k * period` for `k in 0..n`.
///
/// - `Gauge`: exact observation at `g_k`, otherwise LOCF from latest observation within `hold` ticks
///   that does not cross any boundary in `breaks` (or `q::PHASE_BREAK` / `q::RESET`).
/// - `Peak` / `Trough` / `Score` / `Event`: aggregate samples in `(g_{k-1}, g_k]` (or `dtick == g_0`
///   for `k == 0`, clipped after any break inside the bin). No LOCF.
/// - `Counter`: exact observation or linear interpolation between bracketing observations without
///   crossing `q::RESET`; never extrapolates past the first or last observation.
pub fn regularize(
    s: &[TsSample],
    kind: ChannelKind,
    t0: u32,
    period: u32,
    n: usize,
    hold: u32,
    breaks: &[u32],
) -> RegSeries {
    let period = period.max(1);
    if n == 0 {
        return RegSeries { t0, period, values: vec![], mask: vec![], segments: vec![], imputed_share: 0.0 };
    }
    let (pts, _late) = dedup_internal(s, kind);

    let mut all_breaks: Vec<u32> = breaks.to_vec();
    let mut reset_breaks: Vec<u32> = Vec::new();
    for p in &pts {
        if (p.q & (q::PHASE_BREAK | q::RESET)) != 0 {
            all_breaks.push(p.dtick);
        }
        if (p.q & q::RESET) != 0 {
            reset_breaks.push(p.dtick);
        }
    }
    all_breaks.sort_unstable();
    all_breaks.dedup();
    reset_breaks.sort_unstable();
    reset_breaks.dedup();

    let mut values = vec![f64::NAN; n];
    let mut mask = vec![false; n];
    let mut valid_count = 0usize;
    let mut imputed_count = 0usize;

    for k in 0..n {
        let g = t0.saturating_add((k as u32).saturating_mul(period));
        match kind {
            ChannelKind::Gauge => {
                // Find latest point with dtick <= g
                let pos = pts.partition_point(|p| p.dtick <= g);
                if pos > 0 {
                    let p = pts[pos - 1];
                    if p.dtick == g {
                        values[k] = p.value;
                        mask[k] = true;
                        valid_count += 1;
                    } else if g - p.dtick <= hold && !has_break_in(&all_breaks, p.dtick, g) {
                        values[k] = p.value;
                        mask[k] = true;
                        valid_count += 1;
                        imputed_count += 1;
                    }
                }
            }
            ChannelKind::Peak | ChannelKind::Trough | ChannelKind::Score | ChannelKind::Event => {
                let (mut lo_incl, hi_incl) = if k == 0 {
                    (g, g)
                } else {
                    (g.saturating_sub(period).saturating_add(1), g)
                };
                if k > 0 {
                    if let Some(b) = last_break_in(&all_breaks, g.saturating_sub(period), g) {
                        lo_incl = lo_incl.max(b);
                    }
                }
                let start = pts.partition_point(|p| p.dtick < lo_incl);
                let end = pts.partition_point(|p| p.dtick <= hi_incl);
                let slice = &pts[start..end];
                if slice.is_empty() {
                    if kind == ChannelKind::Event {
                        values[k] = 0.0;
                        mask[k] = true;
                        valid_count += 1;
                    }
                } else {
                    let v = match kind {
                        ChannelKind::Peak => slice.iter().map(|p| p.value).fold(f64::NEG_INFINITY, f64::max),
                        ChannelKind::Trough => slice.iter().map(|p| p.value).fold(f64::INFINITY, f64::min),
                        ChannelKind::Score => {
                            let (sum, cnt) = slice.iter().fold((0.0, 0u64), |(s, c), p| (s + p.value * p.count as f64, c + p.count as u64));
                            sum / cnt.max(1) as f64
                        }
                        ChannelKind::Event => slice.iter().map(|p| p.value).sum(),
                        _ => unreachable!(),
                    };
                    values[k] = v;
                    mask[k] = true;
                    valid_count += 1;
                }
            }
            ChannelKind::Counter => {
                let pos = pts.partition_point(|p| p.dtick <= g);
                if pos > 0 && pts[pos - 1].dtick == g {
                    values[k] = pts[pos - 1].value;
                    mask[k] = true;
                    valid_count += 1;
                } else if pos > 0 && pos < pts.len() {
                    let left = pts[pos - 1];
                    let right = pts[pos];
                    if !has_break_in(&reset_breaks, left.dtick, right.dtick) && right.dtick > left.dtick {
                        let frac = (g - left.dtick) as f64 / (right.dtick - left.dtick) as f64;
                        values[k] = left.value + (right.value - left.value) * frac;
                        mask[k] = true;
                        valid_count += 1;
                        imputed_count += 1;
                    }
                }
            }
        }
    }

    // Build contiguous valid segments without crossing breaks (Counter splits on RESET only;
    // non-Counter channels split on any phase break or reset).
    let seg_breaks = if kind == ChannelKind::Counter { &reset_breaks } else { &all_breaks };
    let mut segments = Vec::new();
    let mut cur_start: Option<usize> = None;
    for k in 0..n {
        if !mask[k] {
            if let Some(st) = cur_start.take() {
                segments.push((st, k));
            }
            continue;
        }
        if let Some(st) = cur_start {
            let g_prev = t0.saturating_add(((k - 1) as u32).saturating_mul(period));
            let g_now = t0.saturating_add((k as u32).saturating_mul(period));
            if has_break_in(seg_breaks, g_prev, g_now) {
                segments.push((st, k));
                cur_start = Some(k);
            }
        } else {
            cur_start = Some(k);
        }
    }
    if let Some(st) = cur_start {
        segments.push((st, n));
    }

    let imputed_share = if valid_count == 0 { 0.0 } else { imputed_count as f64 / valid_count as f64 };
    RegSeries { t0, period, values, mask, segments, imputed_share }
}

/// Align a sample slice to a single phase span `[span.start, end)` and resample onto `u_bins`
/// normalized progress bins `u ∈ [0, 1]`.
pub fn align_span(s: &[TsSample], span: &SpanRef, u_bins: usize) -> SpanView {
    let end = span.end.unwrap_or_else(|| {
        s.iter()
            .map(|p| p.dtick)
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(span.start)
    });
    let len = end.saturating_sub(span.start);
    let inside: Vec<&TsSample> = s
        .iter()
        .filter(|p| p.value.is_finite() && p.dtick >= span.start && p.dtick < end)
        .collect();
    let tau: Vec<u32> = inside.iter().map(|p| p.dtick - span.start).collect();
    let start_val = s.iter().find(|p| p.value.is_finite() && p.dtick == span.start).map(|p| p.value as f64);
    let end_val = s
        .iter()
        .rev()
        .find(|p| p.value.is_finite() && p.dtick == end)
        .or_else(|| inside.last().copied())
        .map(|p| p.value as f64);
    let counter_delta = match (start_val, end_val) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };

    let bins = u_bins.max(1);
    let u: Vec<f64> = (0..bins).map(|i| if bins == 1 { 0.0 } else { i as f64 / (bins - 1) as f64 }).collect();
    let mut sums = vec![0.0f64; bins];
    let mut counts = vec![0usize; bins];
    for p in &inside {
        let frac = if len == 0 { 0.0 } else { (p.dtick - span.start) as f64 / len as f64 };
        let b = ((frac * bins as f64).floor() as usize).min(bins - 1);
        sums[b] += p.value as f64;
        counts[b] += 1;
    }
    let mut values = vec![f64::NAN; bins];
    let mut mask = vec![false; bins];
    for i in 0..bins {
        if counts[i] > 0 {
            values[i] = sums[i] / counts[i] as f64;
            mask[i] = true;
        }
    }
    SpanView {
        span_id: span.span_id,
        start: span.start,
        end,
        tau,
        u,
        values,
        mask,
        raw_count: inside.len(),
        counter_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regularize_gauge_holds_then_gaps() {
        let samples = vec![
            TsSample::observed(0, 10.0),
            TsSample::observed(10, 20.0),
        ];
        // period=2, n=7 -> grid ticks 0, 2, 4, 6, 8, 10, 12; hold=4
        let reg = regularize(&samples, ChannelKind::Gauge, 0, 2, 7, 4, &[]);
        // 0: obs 10, 2: held 10 (dt=2<=4), 4: held 10 (dt=4<=4), 6: gap, 8: gap, 10: obs 20, 12: held 20 (dt=2<=4)
        assert_eq!(reg.mask, vec![true, true, true, false, false, true, true]);
        assert!((reg.values[0] - 10.0).abs() < 1e-6);
        assert!((reg.values[2] - 10.0).abs() < 1e-6);
        assert!(reg.values[3].is_nan());
        assert!(reg.values[4].is_nan());
        assert!((reg.values[5] - 20.0).abs() < 1e-6);
        assert!((reg.values[6] - 20.0).abs() < 1e-6);
        assert_eq!(reg.segments, vec![(0, 3), (5, 7)]);
        assert!((reg.imputed_share - 3.0 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn duplicates_follow_channel_kind() {
        let dup = vec![
            TsSample::observed(5, 2.0),
            TsSample::observed(5, 8.0),
            TsSample::observed(5, 5.0),
            TsSample::observed(3, 99.0), // late -> dropped
        ];
        let (g, late) = dedup_samples(&dup, ChannelKind::Gauge);
        assert_eq!(late, 1);
        assert_eq!(g.len(), 1);
        assert!((g[0].1 - 5.0).abs() < 1e-9);

        let (pk, _) = dedup_samples(&dup, ChannelKind::Peak);
        assert!((pk[0].1 - 8.0).abs() < 1e-9);

        let (tr, _) = dedup_samples(&dup, ChannelKind::Trough);
        assert!((tr[0].1 - 2.0).abs() < 1e-9);

        let (ct, _) = dedup_samples(&dup, ChannelKind::Counter);
        assert!((ct[0].1 - 8.0).abs() < 1e-9);

        let (sc, _) = dedup_samples(&dup, ChannelKind::Score);
        assert!((sc[0].1 - 5.0).abs() < 1e-9);

        let (ev, _) = dedup_samples(&dup, ChannelKind::Event);
        assert!((ev[0].1 - 15.0).abs() < 1e-9);
    }

    #[test]
    fn counter_interpolates_but_never_extrapolates() {
        let samples = vec![
            TsSample::observed(2, 10.0),
            TsSample::observed(6, 30.0),
        ];
        // Grid: 0, 2, 4, 6, 8
        let reg = regularize(&samples, ChannelKind::Counter, 0, 2, 5, 0, &[]);
        assert_eq!(reg.mask, vec![false, true, true, true, false]);
        assert!(reg.values[0].is_nan());
        assert!((reg.values[1] - 10.0).abs() < 1e-9);
        assert!((reg.values[2] - 20.0).abs() < 1e-9);
        assert!((reg.values[3] - 30.0).abs() < 1e-9);
        assert!(reg.values[4].is_nan());
    }

    #[test]
    fn phase_boundary_stops_hold_and_counter_splits_exactly() {
        let gauge_samples = vec![
            TsSample::observed(0, 1.0),
            TsSample::observed(6, 2.0),
        ];
        // Break at tick 4 stops LOCF of tick 0 sample from filling tick 4.
        let reg = regularize(&gauge_samples, ChannelKind::Gauge, 0, 2, 5, 6, &[4]);
        // 0: obs 1, 2: held 1, 4: break -> NaN, 6: obs 2, 8: held 2
        assert_eq!(reg.mask, vec![true, true, false, true, true]);
        assert!(reg.values[2].is_nan());
        assert_eq!(reg.segments, vec![(0, 2), (3, 5)]);

        // Counter with boundary snapshots at phase boundaries 0, 10, 20:
        let ctr = vec![
            TsSample::observed(0, 0.0),
            TsSample::observed(5, 2.5),
            TsSample::with_q(10, 6.0, q::OBSERVED | q::PHASE_BREAK),
            TsSample::observed(15, 9.0),
            TsSample::observed(20, 15.0),
        ];
        let sp1 = align_span(&ctr, &SpanRef { span_id: 1, start: 0, end: Some(10) }, 5);
        let sp2 = align_span(&ctr, &SpanRef { span_id: 2, start: 10, end: Some(20) }, 5);
        let total = 15.0 - 0.0;
        let split_sum = sp1.counter_delta.unwrap() + sp2.counter_delta.unwrap();
        assert!((split_sum - total).abs() < 1e-6, "split {split_sum} vs total {total}");
    }
}
