//! Realized-error book (`ErrBook`), Adaptive Conformal Inference (`Aci`), two-sided CUSUM
//! level-shift detector (`Cusum`), and multi-horizon pending forecast ring (`PendingRing`)
//! (Phase 3 – Part 2 · D1, D3, D4, D5).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::fusion::{HistoryMse, MethodWeight, HISTORY_CAP};

/// Canonical multi-horizon evaluation steps (Phase 3 – Part 2 · D5).
pub const HORIZONS: [usize; 5] = [1, 2, 4, 8, 16];

/// Standard normal CDF approximation (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
pub fn norm_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return 0.5;
    }
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    (0.5 * (1.0 + sign * y)).clamp(0.0, 1.0)
}

/// Winkler/Gneiting proper interval score at nominal 80% coverage (`alpha = 0.20`).
pub fn interval_score_80(y: f64, lo: f64, hi: f64) -> f64 {
    let width = (hi - lo).max(0.0);
    let pen_lo = if y < lo { 10.0 * (lo - y) } else { 0.0 };
    let pen_hi = if y > hi { 10.0 * (y - hi) } else { 0.0 };
    width + pen_lo + pen_hi
}

/// Online Adaptive Conformal Inference (ACI) band multiplier targeting `alpha` miscoverage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aci {
    pub k: f64,
    pub eta: f64,
    pub alpha: f64,
    pub min_k: f64,
    pub max_k: f64,
}

impl Default for Aci {
    fn default() -> Self {
        Aci {
            k: 1.0,
            eta: 0.05,
            alpha: 0.20,
            min_k: 0.5,
            max_k: 3.0,
        }
    }
}

impl Aci {
    pub fn step(&mut self, missed: bool) -> f64 {
        let m = if missed { 1.0 } else { 0.0 };
        let ln_k = self.k.max(1e-6).ln() + self.eta * (m - self.alpha);
        self.k = ln_k.exp().clamp(self.min_k, self.max_k);
        self.k
    }
}

/// Two-sided standardized-residual CUSUM for level-shift detection (Phase 3 – Part 2 · D4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cusum {
    pub up: f64,
    pub dn: f64,
    pub kc: f64,
    pub hc: f64,
}

impl Default for Cusum {
    fn default() -> Self {
        Cusum { up: 0.0, dn: 0.0, kc: 0.5, hc: 4.0 }
    }
}

impl Cusum {
    pub fn new(kc: f64, hc: f64) -> Self {
        Cusum { up: 0.0, dn: 0.0, kc, hc }
    }

    pub fn reset(&mut self) {
        self.up = 0.0;
        self.dn = 0.0;
    }

    /// Push standardized residual `z`. Returns `Some(+1.0)` on upward shift or `Some(-1.0)` on downward shift
    /// and resets accumulators.
    pub fn push(&mut self, z: f64) -> Option<f64> {
        if !z.is_finite() {
            return None;
        }
        let zc = z.clamp(-4.0, 4.0);
        self.up = (self.up + zc - self.kc).max(0.0);
        self.dn = (self.dn - zc - self.kc).max(0.0);
        if self.up > self.hc {
            self.reset();
            Some(1.0)
        } else if self.dn > self.hc {
            self.reset();
            Some(-1.0)
        } else {
            None
        }
    }
}

/// Per-scope discounted realized error book and ACI calibrator (Phase 3 – Part 2 · D1/D3/D5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrBook {
    pub lambda: f64,
    pub huber_c: f64,
    pub r: BTreeMap<String, f64>,
    pub n_eff: BTreeMap<String, f64>,
    pub fused: f64,
    pub fused_n: f64,
    pub k: f64,
    pub aci: Aci,
    pub h_r: [BTreeMap<String, f64>; 5],
    pub h_n: [BTreeMap<String, f64>; 5],
    pub h_fused: [f64; 5],
    pub h_fused_n: [f64; 5],
    pub h_aci: [Aci; 5],
    pub h_hits: [u64; 5],
    pub h_total: [u64; 5],
    pub model_hits: u64,
    pub cal_hits: u64,
    pub gate_hits: u64,
    pub total_evals: u64,
    pub pit_sum: f64,
    pub pit_sq_sum: f64,
    pub pit_n: u64,
}

impl Default for ErrBook {
    fn default() -> Self {
        Self::new(0.9, 0.05)
    }
}

impl ErrBook {
    pub fn new(lambda: f64, eta: f64) -> Self {
        let mk_aci = || Aci { eta, ..Aci::default() };
        ErrBook {
            lambda: lambda.clamp(0.5, 0.995),
            huber_c: 3.0,
            r: BTreeMap::new(),
            n_eff: BTreeMap::new(),
            fused: 0.0,
            fused_n: 0.0,
            k: 1.0,
            aci: mk_aci(),
            h_r: std::array::from_fn(|_| BTreeMap::new()),
            h_n: std::array::from_fn(|_| BTreeMap::new()),
            h_fused: [0.0; 5],
            h_fused_n: [0.0; 5],
            h_aci: std::array::from_fn(|_| mk_aci()),
            h_hits: [0; 5],
            h_total: [0; 5],
            model_hits: 0,
            cal_hits: 0,
            gate_hits: 0,
            total_evals: 0,
            pit_sum: 0.0,
            pit_sq_sum: 0.0,
            pit_n: 0,
        }
    }

    /// Seed initial MSE for a predictor from persisted SDS history (capped at `HISTORY_CAP`).
    pub fn seed_method(&mut self, method: &str, mse: f64, n: u64) {
        if !mse.is_finite() || mse < 0.0 || n == 0 {
            return;
        }
        let ne = (n as f64).min(HISTORY_CAP);
        self.r.entry(method.to_string()).or_insert(mse);
        self.n_eff.entry(method.to_string()).or_insert(ne);
        self.h_r[0].entry(method.to_string()).or_insert(mse);
        self.h_n[0].entry(method.to_string()).or_insert(ne);
    }

    pub fn seed_k(&mut self, k: f64) {
        if k.is_finite() {
            self.k = k.clamp(self.aci.min_k, self.aci.max_k);
            self.aci.k = self.k;
            self.h_aci[0].k = self.k;
        }
    }

    fn huber_sq(&self, err: f64, s_ref: f64) -> f64 {
        let cap = (self.huber_c * s_ref.max(1e-4)).max(1e-3);
        let clipped = err.clamp(-cap, cap);
        clipped * clipped
    }

    /// Discounted 1-step realized error update across all active predictors and fused forecast.
    /// Returns `(pit, is80)` for diagnostics.
    pub fn update(&mut self, weights: &[MethodWeight], y: f64, e_f: f64, s_model: f64) -> (f64, f64) {
        let s_before = self.s_fused(s_model).max(1e-4);
        let pit = norm_cdf(e_f / s_before);
        let half80 = self.k * 1.2816 * s_before;
        let p50 = y - e_f;
        let is80 = interval_score_80(y, p50 - half80, p50 + half80);

        let lam = self.lambda;
        let e2_f = self.huber_sq(e_f, s_before);
        if self.fused_n <= 0.0 {
            self.fused = e2_f;
            self.fused_n = 1.0;
        } else {
            self.fused = lam * self.fused + (1.0 - lam) * e2_f;
            self.fused_n = lam * self.fused_n + 1.0;
        }
        self.h_fused[0] = self.fused;
        self.h_fused_n[0] = self.fused_n;

        for w in weights {
            if !w.point1.is_finite() {
                continue;
            }
            let e_m = y - w.point1;
            let e2 = self.huber_sq(e_m, s_before);
            let prev_n = self.n_eff.get(&w.method).copied().unwrap_or(0.0);
            if prev_n <= 0.0 {
                self.r.insert(w.method.clone(), e2);
                self.n_eff.insert(w.method.clone(), 1.0);
            } else {
                let prev_r = self.r.get(&w.method).copied().unwrap_or(e2);
                self.r.insert(w.method.clone(), lam * prev_r + (1.0 - lam) * e2);
                self.n_eff.insert(w.method.clone(), lam * prev_n + 1.0);
            }
            self.h_r[0].insert(w.method.clone(), self.r[&w.method]);
            self.h_n[0].insert(w.method.clone(), self.n_eff[&w.method]);
        }

        self.pit_sum += pit;
        self.pit_sq_sum += pit * pit;
        self.pit_n += 1;
        (pit, is80)
    }

    /// Update realized error for a specific horizon index `h_idx` (`0..5` -> `H = [1,2,4,8,16]`).
    pub fn update_horizon(&mut self, h_idx: usize, y: f64, p50: f64, s_h: f64, points: &[(String, [f64; 5])]) -> bool {
        if h_idx >= HORIZONS.len() || !y.is_finite() || !p50.is_finite() {
            return true;
        }
        let s_ref = self.s_for_horizon(h_idx, s_h).max(1e-4);
        let k_h = self.h_aci[h_idx].k;
        let half = k_h * 1.2816 * s_ref;
        let inside = (y - p50).abs() <= half;
        self.h_total[h_idx] += 1;
        if inside {
            self.h_hits[h_idx] += 1;
        }
        let lam = self.lambda;
        let e2_f = self.huber_sq(y - p50, s_ref);
        if self.h_fused_n[h_idx] <= 0.0 {
            self.h_fused[h_idx] = e2_f;
            self.h_fused_n[h_idx] = 1.0;
        } else {
            self.h_fused[h_idx] = lam * self.h_fused[h_idx] + (1.0 - lam) * e2_f;
            self.h_fused_n[h_idx] = lam * self.h_fused_n[h_idx] + 1.0;
        }
        for (m, pts) in points {
            let pm = pts[h_idx];
            if !pm.is_finite() {
                continue;
            }
            let e2 = self.huber_sq(y - pm, s_ref);
            let pn = self.h_n[h_idx].get(m).copied().unwrap_or(0.0);
            if pn <= 0.0 {
                self.h_r[h_idx].insert(m.clone(), e2);
                self.h_n[h_idx].insert(m.clone(), 1.0);
            } else {
                let pr = self.h_r[h_idx].get(m).copied().unwrap_or(e2);
                self.h_r[h_idx].insert(m.clone(), lam * pr + (1.0 - lam) * e2);
                self.h_n[h_idx].insert(m.clone(), lam * pn + 1.0);
            }
        }
        self.h_aci[h_idx].step(!inside);
        if h_idx == 0 {
            self.fused = self.h_fused[0];
            self.fused_n = self.h_fused_n[0];
            self.r = self.h_r[0].clone();
            self.n_eff = self.h_n[0].clone();
            self.k = self.h_aci[0].k;
            self.aci.k = self.k;
        }
        inside
    }

    /// Blended 1-step spread `s` combining realized discounted RMS `sqrt(fused)` with model band `s_model`.
    pub fn s_fused(&self, s_model: f64) -> f64 {
        let sm = s_model.max(1e-4);
        if self.fused_n <= 0.0 {
            return sm;
        }
        let s_r = self.fused.max(0.0).sqrt();
        let w = (self.fused_n / HISTORY_CAP).clamp(0.0, 1.0);
        w * s_r + (1.0 - w) * sm
    }

    pub fn s_for_horizon(&self, h_idx: usize, s_model_h: f64) -> f64 {
        let sm = s_model_h.max(1e-4);
        let n = self.h_fused_n[h_idx];
        if n <= 0.0 {
            return sm;
        }
        let s_r = self.h_fused[h_idx].max(0.0).sqrt();
        let w = (n / HISTORY_CAP).clamp(0.0, 1.0);
        w * s_r + (1.0 - w) * sm
    }

    /// Precision-weighted gate spread `s_g = sqrt((nu0*s0^2 + n_eff*s_f^2) / (nu0 + n_eff))` (Phase 3 – Part 2 · D4).
    pub fn s_gate(&self, s0: Option<f64>, nu0: f64, s_model: f64) -> f64 {
        let s_f = self.s_fused(s_model).max(1e-4);
        let Some(prior) = s0.filter(|x| x.is_finite() && *x > 0.0) else {
            return s_f;
        };
        let n = self.fused_n.max(0.0);
        let nu = nu0.max(0.0);
        if nu + n <= 0.0 {
            return prior.max(s_f);
        }
        ((nu * prior * prior + n * s_f * s_f) / (nu + n)).sqrt().max(1e-4)
    }

    pub fn aci_step(&mut self, missed: bool) -> f64 {
        self.k = self.aci.step(missed);
        self.h_aci[0].k = self.k;
        self.k
    }

    pub fn record_coverage(&mut self, model_in: bool, cal_in: bool, gate_in: bool) {
        self.total_evals += 1;
        if model_in {
            self.model_hits += 1;
        }
        if cal_in {
            self.cal_hits += 1;
        }
        if gate_in {
            self.gate_hits += 1;
        }
    }

    /// Halve effective sample counts on a detected level shift so weights adapt rapidly.
    pub fn halve_n_eff(&mut self) {
        for v in self.n_eff.values_mut() {
            *v *= 0.5;
        }
        self.fused_n *= 0.5;
    }

    /// Export discounted realized MSEs in `HistoryMse` form for `forecast(...)`.
    pub fn realized(&self) -> Vec<HistoryMse> {
        self.r
            .iter()
            .filter_map(|(m, &mse)| {
                let n = self.n_eff.get(m).copied().unwrap_or(0.0);
                (mse.is_finite() && n >= 0.5).then(|| HistoryMse {
                    method: m.clone(),
                    mse,
                    n: n.round().max(1.0) as u64,
                })
            })
            .collect()
    }

    pub fn realized_rmse(&self, method: &str) -> Option<(f64, f64)> {
        let mse = self.r.get(method).copied()?;
        let n = self.n_eff.get(method).copied().unwrap_or(0.0);
        (mse.is_finite() && n > 0.0).then_some((mse.max(0.0).sqrt(), n))
    }

    pub fn pit_stats(&self) -> (Option<f64>, Option<f64>) {
        if self.pit_n == 0 {
            return (None, None);
        }
        let n = self.pit_n as f64;
        let mean = self.pit_sum / n;
        let sd = if self.pit_n >= 2 {
            ((self.pit_sq_sum - n * mean * mean) / (n - 1.0)).max(0.0).sqrt()
        } else {
            0.0
        };
        (Some(mean), Some(sd))
    }
}

/// Multi-horizon pending forecast issued at `(issue_idx, tick)` awaiting future realizations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pending {
    pub issue_idx: u64,
    pub tick: u64,
    pub p50: [f64; 5],
    pub s: [f64; 5],
    pub points: Vec<(String, [f64; 5])>,
}

/// Realized multi-horizon score emitted when `cur_idx == issue_idx + HORIZONS[h_idx]`.
#[derive(Clone, Debug)]
pub struct HorizonScore {
    pub h: usize,
    pub h_idx: usize,
    pub dt_ticks: u64,
    pub err_fused: f64,
    pub inside80: bool,
    pub method_sq: Vec<(String, f64)>,
}

/// Bounded queue of pending multi-horizon forecasts (Phase 3 – Part 2 · D5).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PendingRing {
    pub items: VecDeque<Pending>,
}

impl PendingRing {
    const CAP: usize = 32;

    /// Record a multi-horizon forecast conditioned on history ending at `issue_idx`
    /// so that horizon `h` targets observation index `issue_idx + h`.
    pub fn issue(
        &mut self,
        issue_idx: u64,
        tick: u64,
        point: &[f64],
        p10: &[f64],
        p90: &[f64],
        weights: &[MethodWeight],
    ) {
        let pick = |v: &[f64], h: usize| -> f64 {
            v.get(h - 1).or_else(|| v.last()).copied().unwrap_or(0.0)
        };
        let mut p50 = [0.0; 5];
        let mut s = [0.0; 5];
        for (i, &h) in HORIZONS.iter().enumerate() {
            p50[i] = pick(point, h);
            let half = 0.5 * (pick(p90, h) - pick(p10, h)).max(0.0);
            s[i] = (half / 1.2816).max(1e-4);
        }
        let points: Vec<(String, [f64; 5])> = weights
            .iter()
            .map(|w| {
                let mut arr = [0.0; 5];
                for (i, &h) in HORIZONS.iter().enumerate() {
                    arr[i] = w.points.get(h - 1).copied().unwrap_or(w.point1);
                }
                (w.method.clone(), arr)
            })
            .collect();
        self.items.push_back(Pending { issue_idx, tick, p50, s, points });
        while self.items.len() > Self::CAP {
            self.items.pop_front();
        }
    }

    /// Score all pending forecasts whose target index `issue_idx + h` equals `cur_idx`.
    /// Set `min_h` to `1` to score all horizons `[1,2,4,8,16]`, or `2` when `h=1` is already scored synchronously.
    pub fn score_ready_from(
        &mut self,
        cur_idx: u64,
        cur_tick: u64,
        y: f64,
        min_h: usize,
        book: &mut ErrBook,
    ) -> Vec<HorizonScore> {
        let mut out = Vec::new();
        for p in &self.items {
            if cur_idx <= p.issue_idx {
                continue;
            }
            let diff = (cur_idx - p.issue_idx) as usize;
            if diff < min_h {
                continue;
            }
            if let Some(h_idx) = HORIZONS.iter().position(|&h| h == diff) {
                let inside = book.update_horizon(h_idx, y, p.p50[h_idx], p.s[h_idx], &p.points);
                let method_sq: Vec<(String, f64)> = p
                    .points
                    .iter()
                    .map(|(m, arr)| (m.clone(), (y - arr[h_idx]).powi(2)))
                    .collect();
                out.push(HorizonScore {
                    h: HORIZONS[h_idx],
                    h_idx,
                    dt_ticks: cur_tick.saturating_sub(p.tick),
                    err_fused: y - p.p50[h_idx],
                    inside80: inside,
                    method_sq,
                });
            }
        }
        // Drop pending items older than max horizon (16).
        while self.items.front().is_some_and(|p| cur_idx.saturating_sub(p.issue_idx) >= 16) {
            self.items.pop_front();
        }
        out
    }

    /// Score all pending forecasts across all horizons `H = [1, 2, 4, 8, 16]`.
    pub fn score_ready(&mut self, cur_idx: u64, cur_tick: u64, y: f64, book: &mut ErrBook) -> Vec<HorizonScore> {
        self.score_ready_from(cur_idx, cur_tick, y, 1, book)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forgetting_tracks_a_switch_in_the_best_method() {
        let lambda = 0.9;
        let mut book = ErrBook::new(lambda, 0.05);
        // Phase 1: method "a" is exact (err=0.01), "b" is noisy (err=0.20).
        for _ in 0..40 {
            let w = vec![
                MethodWeight { method: "a".into(), rmse: 0.01, history_rmse: None, history_n: 0, weight: 0.9, point1: 0.99, points: vec![0.99], rev: "v1".into() },
                MethodWeight { method: "b".into(), rmse: 0.20, history_rmse: None, history_n: 0, weight: 0.1, point1: 0.80, points: vec![0.80], rev: "v1".into() },
            ];
            book.update(&w, 1.0, 0.02, 0.05);
        }
        assert!(book.r["a"] < book.r["b"]);

        // Phase 2: switch — "b" becomes accurate (err=0.01) and "a" degrades (err=0.20).
        let max_steps = (2.0 * (2.0_f64.ln() / (1.0 - lambda))).ceil() as usize; // <= 14 steps
        let mut switched_at = None;
        for step in 1..=max_steps * 2 {
            let w = vec![
                MethodWeight { method: "a".into(), rmse: 0.20, history_rmse: None, history_n: 0, weight: 0.5, point1: 0.80, points: vec![0.80], rev: "v1".into() },
                MethodWeight { method: "b".into(), rmse: 0.01, history_rmse: None, history_n: 0, weight: 0.5, point1: 0.99, points: vec![0.99], rev: "v1".into() },
            ];
            book.update(&w, 1.0, 0.02, 0.05);
            if book.r["b"] < book.r["a"] {
                switched_at = Some(step);
                break;
            }
        }
        assert!(
            switched_at.is_some_and(|s| s <= max_steps),
            "forgetting must invert best method within 2*ln(2)/(1-λ)={max_steps} updates, got {switched_at:?}"
        );
    }

    #[test]
    fn aci_converges_to_target_coverage() {
        let mut aci = Aci::default();
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut normal = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        // True SD = 2.0, uncalibrated s = 1.0 -> ACI must expand k toward ~2.0 so coverage hits ~80%.
        let mut hits = 0usize;
        let total = 400usize;
        for t in 0..500 {
            let y = 2.0 * normal();
            let inside = y.abs() <= aci.k * 1.2816;
            if t >= 100 && inside {
                hits += 1;
            }
            aci.step(!inside);
        }
        let cov = hits as f64 / total as f64;
        assert!((0.74..=0.86).contains(&cov), "ACI coverage {cov:.3} out of [0.74, 0.86], k={:.2}", aci.k);
    }

    #[test]
    fn cusum_detects_step_shift_without_false_alarm() {
        let mut c = Cusum::default();
        for i in 0..40 {
            let z = if i % 2 == 0 { 0.4 } else { -0.4 };
            assert!(c.push(z).is_none());
        }
        let mut triggered = None;
        for _ in 0..10 {
            if let Some(dir) = c.push(2.0) {
                triggered = Some(dir);
                break;
            }
        }
        assert_eq!(triggered, Some(1.0));
    }
}
