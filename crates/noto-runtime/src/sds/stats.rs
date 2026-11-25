use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Exact running min/max extrema preserved alongside Welford moments (Phase 2 – Part 1 · D3).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MinMax {
    pub min: f64,
    pub max: f64,
    pub n: u64,
}

impl Default for MinMax {
    fn default() -> Self {
        Self { min: f64::INFINITY, max: f64::NEG_INFINITY, n: 0 }
    }
}

impl MinMax {
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.n == 0 {
            self.min = x;
            self.max = x;
        } else {
            if x < self.min {
                self.min = x;
            }
            if x > self.max {
                self.max = x;
            }
        }
        self.n += 1;
    }

    pub fn merge(&mut self, other: &MinMax) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
        } else {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
            self.n += other.n;
        }
    }
}

/// Observation-sequence exponentially weighted window (Phase 2 – Part 2 · D1).
/// With target effective sample size `m`, `λ = (m - 1) / (m + 1)` so `w² / w2 → m` in steady state.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EwWin {
    pub w: f64,
    pub w2: f64,
    pub mean: f64,
    pub m: f64,
    #[serde(default)]
    pub c1: f64,
}

impl EwWin {
    pub fn lambda_for(target_m: f64) -> f64 {
        let m = target_m.max(2.0);
        (m - 1.0) / (m + 1.0)
    }

    pub fn push(&mut self, x: f64, prev: Option<f64>, target_m: f64) {
        let lam = Self::lambda_for(target_m);
        let old_mean = self.mean;
        self.w = lam * self.w + 1.0;
        self.w2 = lam * lam * self.w2 + 1.0;
        self.mean = old_mean + (x - old_mean) / self.w;
        self.m = lam * self.m + (x - old_mean) * (x - self.mean);
        if let Some(p) = prev {
            self.c1 = lam * self.c1 + (p - old_mean) * (x - self.mean);
        }
    }

    pub fn n_eff(&self) -> f64 {
        if self.w2 > 0.0 {
            (self.w * self.w) / self.w2
        } else {
            0.0
        }
    }

    pub fn variance(&self) -> f64 {
        let den = self.w - self.w2 / self.w.max(1e-12);
        if den <= 1e-9 {
            0.0
        } else {
            (self.m / den).max(0.0)
        }
    }

    pub fn sd(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Lag-1 autocorrelation estimate clamped to `[0, 0.95]`.
    pub fn rho(&self) -> f64 {
        if self.m <= 1e-12 {
            0.0
        } else {
            (self.c1 / self.m).clamp(0.0, 0.95)
        }
    }

    /// Exact offline merge of `older` followed by `newer` (which consumed `n_newer` sequential observations).
    pub fn merge_offline(older: &EwWin, newer: &EwWin, n_newer: u64, target_m: f64) -> EwWin {
        if older.w <= 0.0 {
            return newer.clone();
        }
        if newer.w <= 0.0 {
            return older.clone();
        }
        let lam = Self::lambda_for(target_m);
        let a = lam.powi(n_newer.min(100_000) as i32);
        let a2 = a * a;
        let w1 = a * older.w;
        let w2_1 = a2 * older.w2;
        let w = w1 + newer.w;
        let w2 = w2_1 + newer.w2;
        let delta = newer.mean - older.mean;
        let mean = older.mean + delta * (newer.w / w);
        let m = a * older.m + newer.m + delta * delta * (w1 * newer.w / w);
        let c1 = a * older.c1 + newer.c1;
        EwWin { w, w2, mean, m, c1 }
    }

    /// Seed an exponential window from legacy lifetime Welford moments, capping weight at `min(n, m)`.
    pub fn seed_from(n: u64, mean: f64, variance: f64, target_m: f64) -> EwWin {
        let m_eff = (n as f64).min(target_m.max(1.0));
        if m_eff <= 0.0 {
            return EwWin::default();
        }
        // In steady state n_eff = w²/w2 ≈ m_eff; set w = m_eff, w2 = m_eff so w - w2/w = m_eff - 1
        let w = m_eff;
        let w2 = m_eff;
        let m = if m_eff > 1.0 { variance.max(0.0) * (w - w2 / w) } else { 0.0 };
        EwWin { w, w2, mean, m, c1: 0.0 }
    }
}

/// Two-sided CUSUM drift state (Phase 2 – Part 2 · D2).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DriftState {
    #[default]
    Stable,
    Suspect,
    Shifted {
        since: u64,
        up: bool,
        size: f64,
    },
}

/// Post-changepoint segment moments (`StatSpan`, Phase 2 – Part 2 · D1/D2).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StatSpan {
    pub start: u64,
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

impl StatSpan {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n - 1) as f64).max(0.0)
        }
    }

    pub fn sd(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Optional temporal state attached beside lifetime [`Welford`] moments (Phase 2 – Part 2 · D1/D2).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Temporal {
    pub seq: u64,
    pub short: EwWin,
    pub long: EwWin,
    #[serde(default)]
    pub cusum: [f64; 2],
    #[serde(default)]
    pub zero_at: u64,
    #[serde(default)]
    pub state: DriftState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seg: Option<StatSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<f64>,
    #[serde(default)]
    pub seeded: bool,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub resid_w: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub resid_m2: f64,
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

impl Temporal {
    pub fn seed_from(w: &Welford, m_s: f64, m_l: f64) -> Temporal {
        Temporal {
            seq: w.n,
            short: EwWin::seed_from(w.n, w.mean, w.variance(), m_s),
            long: EwWin::seed_from(w.n, w.mean, w.variance(), m_l),
            cusum: [0.0, 0.0],
            zero_at: w.n,
            state: DriftState::Stable,
            seg: None,
            prev: w.ring.last().copied(),
            seeded: true,
            resid_w: 0.0,
            resid_m2: 0.0,
        }
    }

    /// Push one observation with sequence-decay windows `(m_s, m_l)` and two-sided CUSUM `(k, h)`.
    /// Returns `Some((prev_state, next_state))` whenever [`DriftState`] transitions.
    pub fn push(&mut self, x: f64, m_s: f64, m_l: f64, k: f64, h: f64) -> Option<(DriftState, DriftState)> {
        self.push_with_expected(x, None, m_s, m_l, k, h)
    }

    /// Push with an optional phase-expected profile mean for residual CUSUM (Phase 2 – Part 2 · D5).
    pub fn push_with_expected(&mut self, x: f64, expected_mean: Option<f64>, m_s: f64, m_l: f64, k: f64, h: f64) -> Option<(DriftState, DriftState)> {
        if !x.is_finite() {
            return None;
        }
        self.seq += 1;
        let prev_state = self.state;

        // Evaluate CUSUM against pre-update long window (or expected profile mean + residual SD).
        let (ready, ref_mean, sd) = if let Some(exp_mu) = expected_mean {
            let ready = self.resid_w >= 8.0;
            let r_sd = if self.resid_w > 0.0 {
                (self.resid_m2 / self.resid_w).sqrt().max(1e-4)
            } else {
                self.long.sd().max(1e-4)
            };
            (ready, exp_mu, r_sd)
        } else {
            let n_l = self.long.n_eff();
            let ready = n_l >= 8.0;
            let pred_sd = self.long.sd().max(1e-4) * (1.0 + 1.0 / n_l.max(1.0)).sqrt();
            (ready, self.long.mean, pred_sd)
        };

        if ready {
            // Winsorize standardized residual to [-4, 4] so a single spike reaches at most Suspect
            // (4 - 0.5 = 3.5 <= 5.0) and cannot falsely declare a step shift alone.
            let r = ((x - ref_mean) / sd).clamp(-4.0, 4.0);
            self.cusum[0] = (self.cusum[0] + r - k).max(0.0);
            self.cusum[1] = (self.cusum[1] - r - k).max(0.0);
            if self.cusum[0] == 0.0 && self.cusum[1] == 0.0 {
                self.zero_at = self.seq;
            }
            let peak = self.cusum[0].max(self.cusum[1]);
            let up = self.cusum[0] >= self.cusum[1];
            match self.state {
                DriftState::Stable => {
                    if peak > h {
                        let size = (x - ref_mean).abs() / sd;
                        self.state = DriftState::Shifted { since: self.zero_at, up, size };
                        let mut sp = StatSpan { start: self.seq, ..StatSpan::default() };
                        sp.push(x);
                        self.seg = Some(sp);
                        self.cusum = [0.0, 0.0];
                        self.zero_at = self.seq;
                    } else if peak > h * 0.5 {
                        self.state = DriftState::Suspect;
                    }
                }
                DriftState::Suspect => {
                    if peak > h {
                        let size = (self.short.mean - ref_mean).abs().max((x - ref_mean).abs()) / sd;
                        self.state = DriftState::Shifted { since: self.zero_at, up, size };
                        let mut sp = StatSpan { start: self.seq, ..StatSpan::default() };
                        sp.push(x);
                        self.seg = Some(sp);
                        self.cusum = [0.0, 0.0];
                        self.zero_at = self.seq;
                    } else if peak <= h * 0.25 {
                        self.state = DriftState::Stable;
                    }
                }
                DriftState::Shifted { .. } => {
                    if let Some(sp) = self.seg.as_mut() {
                        sp.push(x);
                        if sp.n >= (m_s.round() as u64).max(6) {
                            let long_gap = (self.long.mean - sp.mean).abs();
                            if long_gap <= 0.5 * sp.sd().max(self.long.sd()).max(1e-4) {
                                self.state = DriftState::Stable;
                                self.seg = None;
                                self.cusum = [0.0, 0.0];
                                self.zero_at = self.seq;
                            }
                        }
                    }
                }
            }
        }

        if let Some(exp_mu) = expected_mean {
            let lambda = if m_l <= 1.0 { 0.0 } else { (m_l - 1.0) / (m_l + 1.0) };
            let res = x - exp_mu;
            self.resid_w = lambda * self.resid_w + 1.0;
            self.resid_m2 = lambda * self.resid_m2 + res * res;
        }

        let prev = self.prev;
        self.short.push(x, prev, m_s);
        self.long.push(x, prev, m_l);
        self.prev = Some(x);

        (self.state != prev_state).then_some((prev_state, self.state))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(default)]
    pub ring: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<Temporal>,
}

impl Welford {
    pub fn push(&mut self, x: f64, ring_len: usize) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
        self.ring.push(x);
        self.trim(ring_len);
    }

    /// Push one observation updating both lifetime Welford and its [`Temporal`] state.
    pub fn push_temporal(&mut self, x: f64, ring_len: usize, m_s: f64, m_l: f64, k: f64, h: f64) -> Option<(DriftState, DriftState)> {
        if !x.is_finite() {
            return None;
        }
        if self.t.is_none() && self.n > 0 {
            self.t = Some(Temporal::seed_from(self, m_s, m_l));
        }
        self.push(x, ring_len);
        let t = self.t.get_or_insert_with(Temporal::default);
        t.push(x, m_s, m_l, k, h)
    }

    /// Ensure temporal state exists (seeding from lifetime stats if loaded from a legacy ledger).
    pub fn temporal_or_seeded(&self, m_s: f64, m_l: f64) -> Temporal {
        match &self.t {
            Some(t) => t.clone(),
            None => Temporal::seed_from(self, m_s, m_l),
        }
    }

    fn trim(&mut self, ring_len: usize) {
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    pub fn sd(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    pub fn recent_mean(&self) -> f64 {
        if self.ring.is_empty() {
            return self.mean;
        }
        self.ring.iter().sum::<f64>() / self.ring.len() as f64
    }

    /// How far the ring's mean sits from the values before it, in standard errors of that difference.
    pub fn drift_z(&self) -> f64 {
        let (n, k) = (self.n as f64, self.ring.len() as f64);
        if k < 1.0 || n < k + 2.0 {
            return 0.0;
        }
        let before = n - k;
        let ring_mean = self.recent_mean();
        let before_mean = (n * self.mean - k * ring_mean) / before;
        let ring_m2: f64 = self.ring.iter().map(|x| (x - ring_mean).powi(2)).sum();
        let gap = ring_mean - before_mean;
        let within = (self.m2 - gap * gap * before * k / n).max(ring_m2);
        let se = (within / (n - 2.0) * (1.0 / k + 1.0 / before)).sqrt();
        if !(se > 0.0) {
            return 0.0;
        }
        gap / se
    }

    /// Geometric mean of positive values from moments alone: `mean / sqrt(1 + cv²)`.
    pub fn geo_mean(&self) -> f64 {
        if self.n == 0 || self.mean <= 0.0 {
            return self.mean;
        }
        self.mean / (1.0 + self.variance() / (self.mean * self.mean)).sqrt()
    }

    /// Geometric mean of the ring (or arithmetic mean if any value is non-positive).
    pub fn recent_geo_mean(&self) -> f64 {
        if self.ring.is_empty() || self.ring.iter().any(|x| *x <= 0.0) {
            return self.recent_mean();
        }
        (self.ring.iter().map(|x| x.ln()).sum::<f64>() / self.ring.len() as f64).exp()
    }

    pub fn merge(&mut self, other: &Welford, ring_len: usize) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
        } else {
            let na = self.n as f64;
            let nb = other.n as f64;
            let n = na + nb;
            let delta = other.mean - self.mean;
            self.mean += delta * (nb / n);
            self.m2 += other.m2 + delta * delta * (na * nb / n);
            self.n += other.n;
            self.ring.extend(other.ring.iter().cloned());
            if let Some(ot) = &other.t {
                // When run buffer inherited base's `t`, `ot.seq >= self.t.seq` and `ot` already carries
                // the exact continuation with no double weighting; adopt `ot`.
                match &self.t {
                    Some(st) if ot.seq >= st.seq => self.t = Some(ot.clone()),
                    None => self.t = Some(ot.clone()),
                    Some(_) => {}
                }
            }
        }
        self.trim(ring_len);
    }

    /// Exact offline merge of two independent Welford + Temporal histories (older followed by newer).
    pub fn merge_offline(&mut self, other: &Welford, ring_len: usize, m_s: f64, m_l: f64) {
        if other.n == 0 {
            return;
        }
        let prev_t = self.t.clone();
        self.merge(other, ring_len);
        if let (Some(ta), Some(tb)) = (prev_t, &other.t) {
            self.t = Some(Temporal {
                seq: ta.seq + tb.seq,
                short: EwWin::merge_offline(&ta.short, &tb.short, other.n, m_s),
                long: EwWin::merge_offline(&ta.long, &tb.long, other.n, m_l),
                cusum: tb.cusum,
                zero_at: ta.seq + tb.zero_at,
                state: tb.state,
                seg: tb.seg.clone(),
                prev: tb.prev.or(ta.prev),
                seeded: ta.seeded || tb.seeded,
                resid_w: tb.resid_w,
                resid_m2: tb.resid_m2,
            });
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DecayShape {
    pub n: usize,
    pub top1: f64,
    pub top2: f64,
    pub margin: f64,
    pub top_gap_ratio: f64,
    pub tail_flatness: f64,
    pub entropy_norm: f64,
}

pub fn decay_shape(scores: &[f32]) -> Option<DecayShape> {
    let mut v: Vec<f64> = scores.iter().filter(|s| s.is_finite()).map(|s| *s as f64).collect();
    if v.len() < 2 {
        return None;
    }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let (top1, top2) = (v[0], v[1]);
    let margin = top1 - top2;
    let tail = &v[1..];
    let tail_median = tail[tail.len() / 2];
    let span = (top1 - tail_median).abs();
    let top_gap_ratio = if span > 1e-9 { (margin / span).clamp(0.0, 1.0) } else { 0.0 };
    let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let tail_var = tail.iter().map(|x| (x - tail_mean).powi(2)).sum::<f64>() / tail.len() as f64;
    let full = (v[0] - v[n - 1]).abs();
    let tail_flatness = if full > 1e-9 { (tail_var.sqrt() / full).clamp(0.0, 1.0) } else { 1.0 };
    let sd = tail_var.sqrt();
    let entropy_norm = if sd <= 1e-9 {
        1.0
    } else {
        let exps: Vec<f64> = v.iter().map(|x| ((x - v[0]) / sd).exp()).collect();
        let s: f64 = exps.iter().sum::<f64>().max(1e-12);
        let h: f64 = exps.iter().map(|e| e / s).filter(|p| *p > 1e-12).map(|p| -p * p.ln()).sum();
        let hmax = (n as f64).ln();
        if hmax > 1e-9 {
            (h / hmax).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Some(DecayShape { n, top1, top2, margin, top_gap_ratio, tail_flatness, entropy_norm })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecayStat {
    pub margin: Welford,
    pub top_gap_ratio: Welford,
    pub tail_flatness: Welford,
    pub entropy_norm: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionStat {
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub margin: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateStat {
    pub evaluated: u64,
    pub blocked: u64,
    pub sole: u64,
    #[serde(default)]
    pub severity: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeStat {
    #[serde(default)]
    pub baseline: HashMap<String, Welford>,
    #[serde(default)]
    pub calibration: HashMap<String, Welford>,
    #[serde(default)]
    pub decay: HashMap<String, DecayStat>,
    #[serde(default)]
    pub confusion: HashMap<String, ConfusionStat>,
    #[serde(default)]
    pub agreement: HashMap<String, Welford>,
    #[serde(default)]
    pub transition: HashMap<String, u64>,
    #[serde(default)]
    pub gates: HashMap<String, GateStat>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extrema: HashMap<String, MinMax>,
    pub updated_at: i64,
}

impl ScopeStat {
    pub fn absorb(&mut self, other: ScopeStat, ring: usize, now_ms: i64) {
        for (k, v) in other.baseline {
            self.baseline.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.calibration {
            self.calibration.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.decay {
            let d = self.decay.entry(k).or_default();
            d.margin.merge(&v.margin, ring);
            d.top_gap_ratio.merge(&v.top_gap_ratio, ring);
            d.tail_flatness.merge(&v.tail_flatness, ring);
            d.entropy_norm.merge(&v.entropy_norm, ring);
        }
        for (k, v) in other.confusion {
            let c = self.confusion.entry(k).or_default();
            c.ties += v.ties;
            c.a_wins += v.a_wins;
            c.b_wins += v.b_wins;
            c.margin.merge(&v.margin, ring);
        }
        for (k, v) in other.agreement {
            self.agreement.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.transition {
            *self.transition.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.gates {
            let g = self.gates.entry(k).or_default();
            g.evaluated += v.evaluated;
            g.blocked += v.blocked;
            g.sole += v.sole;
            g.severity.merge(&v.severity, ring);
        }
        for (k, v) in other.extrema {
            self.extrema.entry(k).or_default().merge(&v);
        }
        self.updated_at = now_ms;
    }

    pub fn observations(&self) -> u64 {
        self.baseline.values().map(|w| w.n).sum::<u64>()
            + self.calibration.values().map(|w| w.n).sum::<u64>()
            + self.confusion.values().map(|c| c.ties).sum::<u64>()
            + self.gates.values().map(|g| g.evaluated).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    #[test]
    fn welford_matches_the_textbook_formulas() {
        let xs = [0.31, 0.44, 0.52, 0.38, 0.61, 0.47, 0.55, 0.29];
        let mut w = Welford::default();
        for x in xs {
            w.push(x, 4);
        }
        let (m, s) = naive(&xs);
        assert!((w.mean - m).abs() < 1e-12);
        assert!((w.sd() - s).abs() < 1e-12);
        assert_eq!(w.ring.len(), 4);
        assert_eq!(w.ring, xs[4..].to_vec(), "the ring keeps the newest values");
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let mut w = Welford::default();
        w.push(f64::NAN, 4);
        w.push(f64::INFINITY, 4);
        assert_eq!(w.n, 0);
        w.push(1.0, 4);
        assert_eq!(w.n, 1);
        assert_eq!(w.sd(), 0.0);
    }

    #[test]
    fn merge_equals_pushing_everything_in_order() {
        let a = [1.0, 2.0, 4.0, 8.0, 3.0];
        let b = [5.0, 7.0, 2.0, 9.0];
        let mut all = Welford::default();
        for x in a.iter().chain(b.iter()) {
            all.push(*x, 6);
        }
        let (mut wa, mut wb) = (Welford::default(), Welford::default());
        a.iter().for_each(|x| wa.push(*x, 6));
        b.iter().for_each(|x| wb.push(*x, 6));
        wa.merge(&wb, 6);
        assert_eq!(wa.n, all.n);
        assert!((wa.mean - all.mean).abs() < 1e-12);
        assert!((wa.sd() - all.sd()).abs() < 1e-12);
        assert_eq!(wa.ring, all.ring);
        let before = wa.clone();
        wa.merge(&Welford::default(), 6);
        assert_eq!(wa, before);
        let mut empty = Welford::default();
        empty.merge(&before, 6);
        assert_eq!(empty, before);
    }

    #[test]
    fn ew_window_and_merge_are_exact() {
        let m = 16.0;
        let mut w = EwWin::default();
        let mut prev = None;
        for i in 0..2000 {
            let x = 0.5 + 0.05 * ((i as f64 * 0.7).sin());
            w.push(x, prev, m);
            prev = Some(x);
        }
        let neff = w.n_eff();
        assert!((neff - m).abs() / m < 0.02, "steady-state n_eff={neff} vs target m={m}");

        // Exact offline merge vs sequential push
        let xs_a = [0.3, 0.5, 0.4, 0.6, 0.45, 0.52, 0.48];
        let xs_b = [0.55, 0.70, 0.62, 0.68, 0.71];
        let mut seq = EwWin::default();
        for &x in xs_a.iter().chain(xs_b.iter()) {
            seq.push(x, None, m);
        }
        let (mut wa, mut wb) = (EwWin::default(), EwWin::default());
        for &x in &xs_a {
            wa.push(x, None, m);
        }
        for &x in &xs_b {
            wb.push(x, None, m);
        }
        let merged = EwWin::merge_offline(&wa, &wb, xs_b.len() as u64, m);
        assert!((merged.w - seq.w).abs() < 1e-9);
        assert!((merged.w2 - seq.w2).abs() < 1e-9);
        assert!((merged.mean - seq.mean).abs() < 1e-9);
        assert!((merged.m - seq.m).abs() < 1e-9);
    }

    #[test]
    fn forgetting_follows_a_step_change() {
        let m_s = 12.0;
        let m_l = 96.0;
        let mut w = Welford::default();
        for i in 0..200 {
            let noise = if i % 2 == 0 { 0.05 } else { -0.05 };
            w.push_temporal(0.5 + noise, 8, m_s, m_l, 0.5, 5.0);
        }
        // 2-sigma step from 0.5 -> 0.6 (sd ≈ 0.05) -> +0.10 = +2σ
        let mut within_at = None;
        for step in 1..=(2 * m_s as usize) {
            let noise = if step % 2 == 0 { 0.05 } else { -0.05 };
            w.push_temporal(0.6 + noise, 8, m_s, m_l, 0.5, 5.0);
            let t = w.t.as_ref().unwrap();
            let est = t.seg.as_ref().map(|s| s.mean).unwrap_or(t.short.mean);
            if (est - 0.6).abs() < 0.05 && within_at.is_none() {
                within_at = Some(step);
            }
        }
        assert!(within_at.is_some(), "segment/short window should track 2σ shift within 2*m_s");
        assert!((w.mean - 0.6).abs() > 0.05, "lifetime mean lags behind step change: {}", w.mean);
    }

    #[test]
    fn cusum_alarm_budget_and_delay() {
        // Stable normal series: false alarm rate over 80-obs session should be very low; 1σ shift detected within m_s.
        struct Rng(u64);
        impl Rng {
            fn normal(&mut self) -> f64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u1 = (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u2 = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            }
        }
        let mut rng = Rng(42);
        let mut alarms = 0usize;
        let trials = 200usize;
        let m_s = 16.0;
        let m_l = 128.0;
        for _ in 0..trials {
            let mut w = Welford::default();
            for _ in 0..80 {
                w.push_temporal(rng.normal(), 8, m_s, m_l, 0.5, 5.0);
            }
            let mut fired = false;
            for _ in 0..60 {
                let tr = w.push_temporal(rng.normal(), 8, m_s, m_l, 0.5, 5.0);
                if matches!(tr, Some((_, DriftState::Shifted { .. }))) {
                    fired = true;
                    break;
                }
            }
            if fired {
                alarms += 1;
            }
        }
        let fpr = alarms as f64 / trials as f64;
        assert!(fpr < 0.08, "CUSUM false alarm rate too high: {fpr:.3}");

        // 1.5σ shift detection delay
        let mut delays = Vec::new();
        for _ in 0..100 {
            let mut w = Welford::default();
            for _ in 0..100 {
                w.push_temporal(rng.normal(), 8, m_s, m_l, 0.5, 5.0);
            }
            for step in 1..=40 {
                w.push_temporal(1.5 + rng.normal(), 8, m_s, m_l, 0.5, 5.0);
                if matches!(w.t.as_ref().unwrap().state, DriftState::Shifted { .. }) {
                    delays.push(step as f64);
                    break;
                }
            }
        }
        let mean_delay = delays.iter().sum::<f64>() / delays.len().max(1) as f64;
        assert!(delays.len() >= 95 && mean_delay <= m_s, "detected {}/100, mean delay {mean_delay:.1}", delays.len());
    }

    #[test]
    fn drift_z_flags_a_recent_shift() {
        let mut w = Welford::default();
        for _ in 0..30 {
            for x in [0.48, 0.52] {
                w.push(x, 6);
            }
        }
        assert!(w.drift_z().abs() < 0.3);
        for _ in 0..6 {
            w.push(0.9, 6);
        }
        assert!(w.drift_z() > 2.0, "drift {}", w.drift_z());
    }

    #[test]
    fn decay_shape_separates_a_clear_winner_from_a_flat_list() {
        let peaked = decay_shape(&[0.9, 0.2, 0.18, 0.15, 0.1, 0.05]).unwrap();
        let flat = decay_shape(&[0.52, 0.5, 0.49, 0.5, 0.48, 0.47]).unwrap();
        assert!(peaked.margin > flat.margin * 5.0);
        assert!(peaked.top_gap_ratio > flat.top_gap_ratio);
        assert!(flat.entropy_norm > peaked.entropy_norm);
        assert!(decay_shape(&[0.5]).is_none());
        assert!(decay_shape(&[f32::NAN, 0.5]).is_none());
    }

    #[test]
    fn absorb_adds_counts_and_merges_statistics() {
        let mut a = ScopeStat::default();
        let mut b = ScopeStat::default();
        a.baseline.entry("x".into()).or_default().push(1.0, 4);
        b.baseline.entry("x".into()).or_default().push(3.0, 4);
        a.extrema.entry("x".into()).or_default().push(1.0);
        b.extrema.entry("x".into()).or_default().push(3.0);
        b.transition.insert("a>b".into(), 2);
        a.transition.insert("a>b".into(), 1);
        b.gates.entry("g".into()).or_default().evaluated = 5;
        a.absorb(b, 4, 123);
        assert_eq!(a.baseline["x"].n, 2);
        assert!((a.baseline["x"].mean - 2.0).abs() < 1e-12);
        assert_eq!(a.extrema["x"], MinMax { min: 1.0, max: 3.0, n: 2 });
        assert_eq!(a.transition["a>b"], 3);
        assert_eq!(a.gates["g"].evaluated, 5);
        assert_eq!(a.updated_at, 123);
        assert_eq!(a.observations(), 2 + 5);
    }
}
