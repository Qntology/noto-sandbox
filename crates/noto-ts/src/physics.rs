//! Physics-model forecast adapter (`PhysicsPredictor`), physical state integration,
//! and first-passage / RUL calculators (Phase 3 – Part 3 · D2 & D5, Phase 4 – Part 2 · D4).

use serde::{Deserialize, Serialize};

use crate::fusion::{Band, Predictor};
use crate::registry::{Caps, Channels, Exog};

/// Adapter exposing a deterministic physical trajectory (e.g. dynamics rollout, path-preview acceleration,
/// or odometry σ(d) curve) as a [`Predictor`] that competes in [`crate::forecast`] via inverse-MSE weighting.
///
/// `trace` must cover both the historical slice (`0..n`) and the future horizon (`n..n + horizon`) so
/// rolling-origin backtesting at `split < n` evaluates the physical model's true historical residuals.
#[derive(Clone, Debug)]
pub struct PhysicsPredictor {
    pub name: String,
    pub rev: String,
    pub trace: Vec<f64>,
    pub sigma_floor: f64,
}

impl PhysicsPredictor {
    pub fn new(name: impl Into<String>, trace: Vec<f64>, sigma_floor: f64) -> Self {
        Self {
            name: name.into(),
            rev: "phys-v1".into(),
            trace,
            sigma_floor: sigma_floor.max(1e-4),
        }
    }

    fn recent_resid_sd(&self, series: &[f64]) -> f64 {
        let n = series.len().min(self.trace.len());
        if n < 2 {
            return self.sigma_floor;
        }
        let start = n.saturating_sub(16);
        let cnt = (n - start) as f64;
        let mse = series[start..n]
            .iter()
            .zip(&self.trace[start..n])
            .map(|(y, m)| (y - m).powi(2))
            .sum::<f64>()
            / cnt.max(1.0);
        mse.sqrt().max(self.sigma_floor)
    }
}

impl Predictor for PhysicsPredictor {
    fn name(&self) -> &str {
        &self.name
    }

    fn rev(&self) -> String {
        self.rev.clone()
    }

    fn min_history(&self) -> usize {
        2
    }

    fn caps(&self) -> Caps {
        Caps {
            min_obs: 2,
            max_horizon: usize::MAX,
            channels: Channels::Univariate,
            exog: Exog::KnownFuture,
            statics: false,
            quantiles: &[0.1, 0.5, 0.9],
            irregular_time: true,
            cost: 0,
        }
    }

    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        if series.len() < self.min_history() {
            return Err("물리 예측기 입력 이력이 부족합니다".into());
        }
        let offset = series.len();
        if self.trace.len() < offset + horizon {
            return Err(format!(
                "물리 궤적 길이({}) < 필요 길이({})",
                self.trace.len(),
                offset + horizon
            ));
        }
        let point = self.trace[offset..offset + horizon].to_vec();
        if point.iter().any(|v| !v.is_finite()) {
            return Err("물리 예측기 출력에 비유한 값이 있습니다".into());
        }
        let sd = self.recent_resid_sd(series);
        let mut lo = Vec::with_capacity(horizon);
        let mut hi = Vec::with_capacity(horizon);
        for (i, &p) in point.iter().enumerate() {
            let half = 1.2816 * sd * ((i + 1) as f64).sqrt().min(3.0);
            lo.push(p - half);
            hi.push(p + half);
        }
        Ok(Band { point, lo: Some(lo), hi: Some(hi), quantiles: None })
    }
}

/// First-order physical state law driven by a forecasted forcing series (Phase 4 – Part 2 · D4).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum State {
    /// Thermal lag: `dh/dt = (drive - h) / tau_s`, where `drive = effort²` and limit is `1.0`.
    Heat { tau_s: f64 },
    /// Cumulative Miner wear damage: `dw/dt = max(0, drive)^alpha` (or direct wear rate when `alpha == 1.0`).
    Wear { alpha: f64 },
    /// Battery energy depletion: `dE/dt = max(0, drive)` toward `cap_j`.
    Energy { cap_j: f64 },
}

/// Distribution of first-passage crossing times (seconds) for a physical limit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstPassage {
    pub p50_s: Option<f64>,
    pub p10_s: Option<f64>,
    pub p90_s: Option<f64>,
    /// Approximate probability that the limit is crossed within the integrated horizon `[0, 1]`.
    pub p_within_horizon: f64,
}

/// Closed-form time (s) for first-order thermal state `h(t) = e2 + (h0 - e2) * exp(-t / tau_s)`
/// to reach thermal limit `1.0` under constant squared effort `e2` (Phase 3 – Part 3 · D5).
pub fn heat_crossing_s(h0: f64, e2: f64, tau_s: f64) -> Option<f64> {
    if !h0.is_finite() || !e2.is_finite() || !tau_s.is_finite() || tau_s <= 0.0 {
        return None;
    }
    if h0 >= 1.0 {
        return Some(0.0);
    }
    if e2 <= 1.0 + 1e-9 {
        return None;
    }
    let ratio = (e2 - h0) / (e2 - 1.0);
    if ratio <= 1.0 {
        Some(0.0)
    } else {
        Some(tau_s * ratio.ln())
    }
}

fn integrate_one(state: State, x0: f64, drive: &[f64], dt_s: f64, limit: f64) -> Option<f64> {
    if x0 >= limit {
        return Some(0.0);
    }
    if dt_s <= 0.0 || drive.is_empty() {
        return None;
    }
    let mut x = x0;
    for (k, &u_raw) in drive.iter().enumerate() {
        let u = if u_raw.is_finite() { u_raw } else { 0.0 };
        let prev = x;
        match state {
            State::Heat { tau_s } => {
                let a = (dt_s / tau_s.max(1e-6)).min(1.0);
                x += a * (u.max(0.0) - x);
            }
            State::Wear { alpha } => {
                x += u.max(0.0).powf(alpha.max(0.0)) * dt_s;
            }
            State::Energy { cap_j } => {
                let _ = cap_j;
                x += u.max(0.0) * dt_s;
            }
        }
        if x >= limit {
            let frac = if (x - prev).abs() > 1e-12 {
                ((limit - prev) / (x - prev)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            return Some((k as f64 + frac) * dt_s);
        }
    }
    None
}

/// Integrate physical state trajectories driven by `(drive_p10, drive_p50, drive_p90)` to estimate
/// first-passage crossing times `(p10_s [earliest], p50_s [median], p90_s [latest])`.
pub fn integrate_state(
    state: State,
    x0: f64,
    drive_p10: &[f64],
    drive_p50: &[f64],
    drive_p90: &[f64],
    dt_s: f64,
    limit: f64,
) -> FirstPassage {
    // Higher forcing crosses sooner -> p10_s (earliest 10th percentile time) comes from p90 forcing.
    let t_early = integrate_one(state, x0, drive_p90, dt_s, limit);
    let t_med = integrate_one(state, x0, drive_p50, dt_s, limit);
    let t_late = integrate_one(state, x0, drive_p10, dt_s, limit);
    let p_within_horizon = match (t_early.is_some(), t_med.is_some(), t_late.is_some()) {
        (true, true, true) => 0.90,
        (true, true, false) => 0.65,
        (true, false, false) => 0.25,
        (false, false, false) => 0.05,
        _ => 0.50,
    };
    FirstPassage {
        p50_s: t_med,
        p10_s: t_early,
        p90_s: t_late,
        p_within_horizon,
    }
}

/// Integrate one deterministic driver trajectory into a physical state path (Phase 4 – Part 2 · D4).
pub fn integrate_path(state: State, x0: f64, driver: &[f64], dt_s: f64) -> Vec<f64> {
    let mut x = x0;
    driver
        .iter()
        .map(|&u_raw| {
            let u = if u_raw.is_finite() { u_raw.max(0.0) } else { 0.0 };
            x = match state {
                State::Heat { tau_s } => x + (dt_s / tau_s.max(1e-6)).min(1.0) * (u - x),
                State::Wear { alpha } => x + u.powf(alpha.max(0.0)) * dt_s,
                State::Energy { .. } => x + u * dt_s,
            };
            x
        })
        .collect()
}

/// Hybrid physics-driven first-passage summary extending beyond the finite forecast horizon
/// by holding the final predicted forcing level constant when `extrapolated` is needed (Phase 4 – Part 2 · D4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysPassage {
    pub series: String,
    pub threshold: f64,
    /// `[p10_s (earliest), p50_s (median), p90_s (latest)]` crossing times in seconds.
    pub t_s: [Option<f64>; 3],
    pub horizon_s: f64,
    pub extrapolated: bool,
}

fn first_passage_with_extrap(state: State, x0: f64, drive: &[f64], dt_s: f64, limit: f64) -> (Option<f64>, bool) {
    if let Some(t) = integrate_one(state, x0, drive, dt_s, limit) {
        return (Some(t), false);
    }
    let Some(&last_u) = drive.last() else {
        return (None, false);
    };
    let u = if last_u.is_finite() { last_u.max(0.0) } else { 0.0 };
    let path = integrate_path(state, x0, drive, dt_s);
    let x_end = path.last().copied().unwrap_or(x0);
    let h_s = drive.len() as f64 * dt_s;
    let extra = match state {
        State::Heat { tau_s } => heat_crossing_s(x_end, u, tau_s),
        State::Wear { alpha } => {
            let rate = u.powf(alpha.max(0.0));
            (rate > 1e-12 && x_end < limit).then(|| (limit - x_end) / rate)
        }
        State::Energy { .. } => (u > 1e-12 && x_end < limit).then(|| (limit - x_end) / u),
    };
    match extra {
        Some(dt_extra) if dt_extra.is_finite() => (Some(h_s + dt_extra), true),
        _ => (None, false),
    }
}

/// Hybrid physics-driven predictor (`PhysDriven`, Phase 4 – Part 2 · D4):
/// combines forecasted forcing quantile paths `(p10, p50, p90)` with exact engine state integration.
#[derive(Clone, Debug)]
pub struct PhysDriven {
    pub series: String,
    pub state: State,
    pub threshold: f64,
    pub dt_s: f64,
}

impl PhysDriven {
    pub fn new(series: impl Into<String>, state: State, threshold: f64, dt_s: f64) -> Self {
        Self {
            series: series.into(),
            state,
            threshold,
            dt_s: dt_s.max(1e-3),
        }
    }

    pub fn first_passage(&self, x0: f64, drive_p10: &[f64], drive_p50: &[f64], drive_p90: &[f64]) -> PhysPassage {
        let horizon_s = drive_p50.len() as f64 * self.dt_s;
        let (t10, e10) = first_passage_with_extrap(self.state, x0, drive_p90, self.dt_s, self.threshold);
        let (t50, e50) = first_passage_with_extrap(self.state, x0, drive_p50, self.dt_s, self.threshold);
        let (t90, e90) = first_passage_with_extrap(self.state, x0, drive_p10, self.dt_s, self.threshold);
        PhysPassage {
            series: self.series.clone(),
            threshold: self.threshold,
            t_s: [t10, t50, t90],
            horizon_s,
            extrapolated: e10 || e50 || e90,
        }
    }
}

/// Remaining useful life (seconds) `[p10_s, p50_s, p90_s]` from current `life_used` in `[0, 1]`
/// and forecasted wear-rate quantiles `[rate_p10, rate_p50, rate_p90]` (life_used / s).
/// Returns `None` when wear rate is zero (no load).
pub fn wear_rul_s(life_used: f64, rate_p10_p50_p90: [f64; 3]) -> [Option<f64>; 3] {
    let rem = (1.0 - life_used).max(0.0);
    let calc = |rate: f64| -> Option<f64> {
        if !rate.is_finite() || rate <= 1e-12 {
            None
        } else {
            Some(rem / rate)
        }
    };
    // Highest wear rate -> shortest (earliest) RUL p10; lowest wear rate -> longest RUL p90.
    [calc(rate_p10_p50_p90[2]), calc(rate_p10_p50_p90[1]), calc(rate_p10_p50_p90[0])]
}

/// Combined shift-change time (s) `[p10_s, p50_s, p90_s]` taking the elementwise minimum
/// across thermal saturation, battery depletion, and wear budget crossing times.
pub fn shift_change_s(t_heat: [Option<f64>; 3], t_battery: [Option<f64>; 3], t_wear: [Option<f64>; 3]) -> [Option<f64>; 3] {
    std::array::from_fn(|i| {
        [t_heat[i], t_battery[i], t_wear[i]]
            .into_iter()
            .flatten()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .reduce(f64::min)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast;

    #[test]
    fn physics_predictor_joins_fusion_and_wins_on_exact_trace() {
        let trace: Vec<f64> = (0..24).map(|i| 0.5 + 0.25 * (i as f64 * 0.4).sin()).collect();
        let hist = &trace[..18];
        let phys = PhysicsPredictor::new("physics", trace.clone(), 1e-3);
        let f = forecast(hist, 4, &[&phys], &[]);
        let pw = f.weights.iter().find(|w| w.method == "physics").expect("physics joined");
        assert!(pw.weight > 0.5, "exact physics model must dominate fusion weight: {:?}", f.weights);
        for (pred, truth) in f.point.iter().zip(&trace[18..22]) {
            assert!((pred - truth).abs() < 0.05, "pred={pred} vs truth={truth}");
        }
    }

    #[test]
    fn heat_crossing_matches_discrete_ema_within_one_tick() {
        let tau = 60.0;
        let h0 = 0.25;
        let e2 = 1.44; // effort = 1.2
        let dt = 0.1;
        let closed = heat_crossing_s(h0, e2, tau).expect("crosses");
        let mut h = h0;
        let mut t = 0.0;
        while h < 1.0 && t < 500.0 {
            h += (dt / tau).min(1.0) * (e2 - h);
            t += dt;
        }
        assert!((closed - t).abs() <= dt * 1.5, "closed={closed:.3} vs discrete={t:.3}");
        let fp = integrate_state(
            State::Heat { tau_s: tau },
            h0,
            &vec![1.2; 1200],
            &vec![e2; 1200],
            &vec![1.7; 1200],
            dt,
            1.0,
        );
        assert!(fp.p10_s.unwrap() < fp.p50_s.unwrap() && fp.p50_s.unwrap() < fp.p90_s.unwrap());
        assert!((fp.p50_s.unwrap() - closed).abs() <= dt * 1.5);
    }

    #[test]
    fn wear_rul_is_monotone_and_finite() {
        assert_eq!(wear_rul_s(0.2, [0.0, 0.0, 0.0]), [None, None, None]);
        let rates = [1e-5, 2e-5, 4e-5];
        let mut prev = f64::INFINITY;
        for step in 0..10 {
            let used = step as f64 * 0.08;
            let rul = wear_rul_s(used, rates);
            assert!(rul.iter().all(|x| x.is_some_and(|v| v.is_finite() && v >= 0.0)));
            let [r10, r50, r90] = [rul[0].unwrap(), rul[1].unwrap(), rul[2].unwrap()];
            assert!(r10 <= r50 && r50 <= r90, "quantile ordering: {rul:?}");
            assert!(r50 < prev, "RUL must strictly decrease as wear accumulates");
            prev = r50;
        }
        let sc = shift_change_s(
            [Some(200.0), Some(300.0), None],
            [Some(400.0), Some(500.0), Some(600.0)],
            [None, Some(250.0), Some(400.0)],
        );
        assert_eq!(sc, [Some(200.0), Some(250.0), Some(400.0)]);
    }
}
