use serde::{Deserialize, Serialize};

use super::fusion::{forecast, Forecast, HistoryMse, Predictor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateConfig {
    pub widen_sd: f64,
    pub min_history: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { widen_sd: 1.0, min_history: 6 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalOutcome {
    Abstained,
    Accepted,
    Warn,
    Held,
}

/// Calibrated gate band parameters (Phase 3 – Part 2 · D4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateBand {
    /// Adaptive conformal multiplier `k` (targeting nominal 80% coverage at `±1.2816·k·s_g`).
    pub k: f64,
    /// Two-sided gate threshold in standardized units (default `2.33` ≈ 98% nominal).
    pub z_g: f64,
    /// Precision-weighted 1-step residual spread `s_g`.
    pub s_g: f64,
}

impl Default for GateBand {
    fn default() -> Self {
        GateBand { k: 1.0, z_g: 2.33, s_g: 0.05 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVerdict {
    pub outcome: TemporalOutcome,
    pub observed: f64,
    pub expected: Option<f64>,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub z: Option<f64>,
    pub method: String,
    pub reason: String,
    /// Whether `observed` fell inside the calibrated 80% band (`|z| <= 1.2816`).
    #[serde(default)]
    pub inside80: bool,
    /// Continuous severity `0..=100` scaling across `|z| in [1.2816, z_g]`.
    #[serde(default)]
    pub severity: u8,
}

impl TemporalVerdict {
    pub fn passes(&self) -> bool {
        self.outcome != TemporalOutcome::Held
    }
}

/// Evaluate a candidate observation `observed` against forecast `f` using calibrated [`GateBand`]
/// (Phase 3 – Part 2 · D4).
pub fn temporal_verdict(observed: f64, f: &Forecast, band: &GateBand) -> TemporalVerdict {
    if !observed.is_finite() || f.point.is_empty() {
        return TemporalVerdict {
            outcome: TemporalOutcome::Held,
            observed,
            expected: None,
            lo: None,
            hi: None,
            z: None,
            method: f.method.clone(),
            reason: "관측값이 유한하지 않음".into(),
            inside80: false,
            severity: 100,
        };
    }
    let p50 = f.point[0];
    let scale = (band.k.max(0.1) * band.s_g.max(1e-6)).max(1e-6);
    let half_gate = band.z_g.max(1.2816) * scale;
    let lo = p50 - half_gate;
    let hi = p50 + half_gate;
    let z = (observed - p50) / scale;
    let az = z.abs();
    let inside80 = az <= 1.2816;
    let inside_gate = az <= band.z_g.max(1.2816);
    let outcome = if inside80 {
        TemporalOutcome::Accepted
    } else if inside_gate {
        TemporalOutcome::Warn
    } else {
        TemporalOutcome::Held
    };
    let sev = if inside80 {
        0
    } else {
        let span = (band.z_g - 1.2816).max(1e-3);
        (((az - 1.2816) / span).clamp(0.0, 1.0) * 100.0).round() as u8
    };
    let reason = match outcome {
        TemporalOutcome::Accepted => format!("예측 띠 안 (z={z:+.2})"),
        TemporalOutcome::Warn => format!("경고 띠: z={z:+.2} (80% 밖 · 게이트 ±{:.2} 안)", band.z_g),
        TemporalOutcome::Held => {
            if observed > hi {
                format!("예측 띠 위로 벗어남 (관측 {observed:.3} > 상한 {hi:.3}, z={z:+.2})")
            } else {
                format!("예측 띠 아래로 벗어남 (관측 {observed:.3} < 하한 {lo:.3}, z={z:+.2})")
            }
        }
        TemporalOutcome::Abstained => "판정 생략".into(),
    };
    TemporalVerdict {
        outcome,
        observed,
        expected: Some(p50),
        lo: Some(lo),
        hi: Some(hi),
        z: Some(z),
        method: f.method.clone(),
        reason,
        inside80,
        severity: sev,
    }
}

pub fn temporal_gate(
    history: &[f64],
    observed: f64,
    sds_sd: Option<f64>,
    extra: &[&dyn Predictor],
    past_errors: &[HistoryMse],
    cfg: &GateConfig,
) -> TemporalVerdict {
    temporal_gate_with_forecast(history, observed, sds_sd, extra, past_errors, cfg).0
}

/// Evaluates the temporal gate and returns the 1-step [`Forecast`] computed for it,
/// avoiding a redundant second forecast call when the caller logs fusion MSE weights.
pub fn temporal_gate_with_forecast(
    history: &[f64],
    observed: f64,
    sds_sd: Option<f64>,
    extra: &[&dyn Predictor],
    past_errors: &[HistoryMse],
    cfg: &GateConfig,
) -> (TemporalVerdict, Option<Forecast>) {
    if !observed.is_finite() {
        return (
            TemporalVerdict {
                outcome: TemporalOutcome::Held,
                observed,
                expected: None,
                lo: None,
                hi: None,
                z: None,
                method: "none".into(),
                reason: "관측값이 유한하지 않음".into(),
                inside80: false,
                severity: 100,
            },
            None,
        );
    }
    let usable = history.iter().filter(|v| v.is_finite()).count();
    if usable < cfg.min_history {
        return (
            TemporalVerdict {
                outcome: TemporalOutcome::Abstained,
                observed,
                expected: None,
                lo: None,
                hi: None,
                z: None,
                method: "none".into(),
                reason: format!("이력 {usable}점 < {}점이라 판정하지 않음", cfg.min_history),
                inside80: true,
                severity: 0,
            },
            None,
        );
    }
    let f = forecast(history, 1, extra, past_errors);
    let widen = cfg.widen_sd * sds_sd.unwrap_or(0.0).max(0.0);
    let (p50, lo, hi) = (f.point[0], f.p10[0] - widen, f.p90[0] + widen);
    let half = ((hi - lo) / 2.0).max(1e-9);
    let z = (observed - p50) / (half / 1.2816);
    let inside = observed >= lo && observed <= hi;
    let verdict = TemporalVerdict {
        outcome: if inside { TemporalOutcome::Accepted } else { TemporalOutcome::Held },
        observed,
        expected: Some(p50),
        lo: Some(lo),
        hi: Some(hi),
        z: Some(z),
        method: f.method.clone(),
        reason: if inside {
            "예측 띠 안".into()
        } else if observed > hi {
            format!("예측 띠 위로 벗어남 (관측 {observed:.3} > 상한 {hi:.3})")
        } else {
            format!("예측 띠 아래로 벗어남 (관측 {observed:.3} < 하한 {lo:.3})")
        },
        inside80: observed >= f.p10[0] && observed <= f.p90[0],
        severity: if inside { 0 } else { ((z.abs() - 1.2816) / 2.5 * 100.0).clamp(0.0, 100.0).round() as u8 },
    };
    (verdict, Some(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(n: usize, level: f64) -> Vec<f64> {
        (0..n).map(|i| level + 0.02 * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0).collect()
    }

    #[test]
    fn abstains_without_enough_history() {
        let v = temporal_gate(&[0.5, 0.5, 0.5], 0.9, None, &[], &[], &GateConfig::default());
        assert_eq!(v.outcome, TemporalOutcome::Abstained);
        assert!(v.passes());
        assert!(v.reason.contains("3점"));
    }

    #[test]
    fn a_consistent_observation_passes_and_a_spike_is_held() {
        let h = steady(20, 0.55);
        let ok = temporal_gate(&h, 0.56, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(ok.outcome, TemporalOutcome::Accepted, "{:?}", ok);
        let spike = temporal_gate(&h, 0.95, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(spike.outcome, TemporalOutcome::Held);
        assert!(!spike.passes());
        assert!(spike.reason.contains("위로"));
        assert!(spike.z.unwrap() > 3.0, "{:?}", spike.z);
        let dip = temporal_gate(&h, 0.1, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(dip.outcome, TemporalOutcome::Held);
        assert!(dip.reason.contains("아래로"));
    }

    #[test]
    fn a_larger_sds_spread_widens_the_accepted_range() {
        let h = steady(20, 0.55);
        let obs = 0.68;
        let tight = temporal_gate(&h, obs, Some(0.0), &[], &[], &GateConfig::default());
        let wide = temporal_gate(&h, obs, Some(0.15), &[], &[], &GateConfig::default());
        assert_eq!(tight.outcome, TemporalOutcome::Held, "{:?}", tight);
        assert_eq!(wide.outcome, TemporalOutcome::Accepted, "{:?}", wide);
        assert!(wide.hi.unwrap() > tight.hi.unwrap());
    }

    #[test]
    fn the_second_consistent_high_score_is_accepted() {
        let mut h = steady(20, 0.2);
        let first = temporal_gate(&h, 0.7, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(first.outcome, TemporalOutcome::Held, "first sighting is held");
        h.push(0.7);
        let second = temporal_gate(&h, 0.71, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(second.outcome, TemporalOutcome::Accepted, "{:?}", second);
    }

    #[test]
    fn non_finite_observations_are_held() {
        let v = temporal_gate(&steady(10, 0.5), f64::NAN, None, &[], &[], &GateConfig::default());
        assert_eq!(v.outcome, TemporalOutcome::Held);
    }

    #[test]
    fn the_gate_holds_about_alpha_of_true_values() {
        let mut state = 0xfeed_face_1234_5678u64;
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
        let z_g = 2.33; // nominal 2*(1 - Phi(2.33)) ≈ 0.0198
        let expected_alpha = 2.0 * (1.0 - crate::ts::calib::norm_cdf(z_g));
        let trials = 1000usize;
        let mut held = 0usize;
        let mut warned = 0usize;
        for _ in 0..trials {
            let f = Forecast {
                point: vec![0.6],
                p10: vec![0.6 - 1.2816 * 0.05],
                p90: vec![0.6 + 1.2816 * 0.05],
                weights: vec![],
                rmse_fused: 0.05,
                holdout: 4,
                method: "test".into(),
                notes: vec![],
                quantiles: None,
            };
            let obs = 0.6 + 0.05 * normal();
            let v = temporal_verdict(obs, &f, &GateBand { k: 1.0, z_g, s_g: 0.05 });
            if v.outcome == TemporalOutcome::Held {
                held += 1;
            } else if v.outcome == TemporalOutcome::Warn {
                warned += 1;
            }
        }
        let rate = held as f64 / trials as f64;
        let se = (expected_alpha * (1.0 - expected_alpha) / trials as f64).sqrt();
        assert!(
            (rate - expected_alpha).abs() <= 3.0 * se.max(0.008),
            "held rate {rate:.4} far from nominal {expected_alpha:.4} (se={se:.4})"
        );
        assert!(warned > 0, "Warn outcome should fire between 80% and gate band");
    }
}
