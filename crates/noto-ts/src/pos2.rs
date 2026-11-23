//! 2D probabilistic target position filter (Phase 3 – Part 2 · D6, Phase 4 – Part 2 · D3).
//!
//! Static targets use inverse-variance weighted mean + combined Fisher/scatter 2×2 covariance
//! with Mahalanobis chi-square gating (`d² <= 9.21`, df=2, 99%). Sustained directional innovation
//! (two-axis CUSUM) switches seamlessly to a 2D constant-velocity Kalman filter for moving targets.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::calib::Cusum;

/// 99% chi-square critical value for 2 degrees of freedom (`d² <= 9.21034`).
pub const CHI2_2DOF_99: f64 = 9.21034;

/// Maximum observations retained in the bounded ring (`CAP = 32`).
pub const POS2_RING_CAP: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pos2Model {
    Static,
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pos2Obs {
    pub x: f64,
    pub z: f64,
    pub sigma_m: f64,
    pub tick: u64,
}

/// 2D position estimate with symmetric 2×2 covariance `[cxx, cxz, czz]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pos2Estimate {
    pub mean: [f64; 2],
    /// Symmetric covariance `[cxx, cxz, czz]` in m².
    pub cov: [f32; 3],
    pub n: usize,
    pub model: Pos2Model,
    pub chi2_gate: f64,
}

impl Pos2Estimate {
    /// Mahalanobis squared distance `d² = Δᵀ (Σ + σ² I)⁻¹ Δ` for a candidate sighting `(x, z, sigma_m)`.
    pub fn mahalanobis_sq(&self, x: f64, z: f64, sigma_m: f64) -> f64 {
        let s2 = sigma_m.max(1e-3).powi(2);
        let a = (self.cov[0] as f64).max(1e-9) + s2;
        let b = self.cov[1] as f64;
        let d = (self.cov[2] as f64).max(1e-9) + s2;
        let det = (a * d - b * b).max(1e-12);
        let dx = x - self.mean[0];
        let dz = z - self.mean[1];
        (d * dx * dx - 2.0 * b * dx * dz + a * dz * dz) / det
    }

    pub fn accepts(&self, x: f64, z: f64, sigma_m: f64) -> bool {
        self.mahalanobis_sq(x, z, sigma_m) <= self.chi2_gate
    }

    /// Axis-aligned bounding rectangle `[x_lo, x_hi, z_lo, z_hi]` at `sigmas` standard deviations
    /// (backward compatible with `DecisionReport.band`).
    pub fn band_rect(&self, sigmas: f64, obs_sigma_m: f64) -> [f32; 4] {
        let sx = ((self.cov[0] as f64).max(0.0) + obs_sigma_m * obs_sigma_m).sqrt() * sigmas;
        let sz = ((self.cov[2] as f64).max(0.0) + obs_sigma_m * obs_sigma_m).sqrt() * sigmas;
        [
            (self.mean[0] - sx) as f32,
            (self.mean[0] + sx) as f32,
            (self.mean[1] - sz) as f32,
            (self.mean[1] + sz) as f32,
        ]
    }
}

/// 2D constant-velocity Kalman state `[x, z, vx, vz]` (in metres and metres/tick) for moving targets.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CvKalman {
    x: [f64; 4],
    p: [[f64; 4]; 4],
    last_tick: u64,
    initialized: bool,
}

impl CvKalman {
    fn init(&mut self, obs: Pos2Obs, vx: f64, vz: f64) {
        let s2 = obs.sigma_m.max(0.02).powi(2);
        self.x = [obs.x, obs.z, vx, vz];
        self.p = [
            [s2, 0.0, 0.0, 0.0],
            [0.0, s2, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.05],
        ];
        self.last_tick = obs.tick;
        self.initialized = true;
    }

    fn predict_at(&self, tick: u64) -> ([f64; 4], [[f64; 4]; 4]) {
        let dt = tick.saturating_sub(self.last_tick).max(1) as f64;
        let mut x = self.x;
        x[0] += x[2] * dt;
        x[1] += x[3] * dt;
        // F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], Q = q_acc * G G^T
        let q = 1e-4;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;
        let p = self.p;
        // FP
        let mut fp = [[0.0; 4]; 4];
        for j in 0..4 {
            fp[0][j] = p[0][j] + dt * p[2][j];
            fp[1][j] = p[1][j] + dt * p[3][j];
            fp[2][j] = p[2][j];
            fp[3][j] = p[3][j];
        }
        // FP F^T + Q
        let mut pn = [[0.0; 4]; 4];
        for i in 0..4 {
            pn[i][0] = fp[i][0] + dt * fp[i][2];
            pn[i][1] = fp[i][1] + dt * fp[i][3];
            pn[i][2] = fp[i][2];
            pn[i][3] = fp[i][3];
        }
        pn[0][0] += 0.25 * dt4 * q;
        pn[0][2] += 0.5 * dt3 * q;
        pn[2][0] += 0.5 * dt3 * q;
        pn[2][2] += dt2 * q;
        pn[1][1] += 0.25 * dt4 * q;
        pn[1][3] += 0.5 * dt3 * q;
        pn[3][1] += 0.5 * dt3 * q;
        pn[3][3] += dt2 * q;
        (x, pn)
    }

    fn update(&mut self, obs: Pos2Obs) {
        let (xp, pp) = self.predict_at(obs.tick);
        let r = obs.sigma_m.max(0.01).powi(2);
        let s00 = pp[0][0] + r;
        let s01 = pp[0][1];
        let s11 = pp[1][1] + r;
        let det = (s00 * s11 - s01 * s01).max(1e-12);
        let inv00 = s11 / det;
        let inv01 = -s01 / det;
        let inv11 = s00 / det;
        // K = P H^T S^{-1} (4x2)
        let mut k = [[0.0; 2]; 4];
        for i in 0..4 {
            k[i][0] = pp[i][0] * inv00 + pp[i][1] * inv01;
            k[i][1] = pp[i][0] * inv01 + pp[i][1] * inv11;
        }
        let y0 = obs.x - xp[0];
        let y1 = obs.z - xp[1];
        for i in 0..4 {
            self.x[i] = xp[i] + k[i][0] * y0 + k[i][1] * y1;
        }
        // P = (I - K H) P
        let mut pn = pp;
        for i in 0..4 {
            for j in 0..4 {
                pn[i][j] = pp[i][j] - k[i][0] * pp[0][j] - k[i][1] * pp[1][j];
            }
        }
        self.p = pn;
        self.last_tick = obs.tick;
    }
}

/// Stateful 2D position estimator with static-to-dynamic CUSUM switching.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pos2Filter {
    obs: VecDeque<Pos2Obs>,
    model: Pos2Model,
    cusum_x: Cusum,
    cusum_z: Cusum,
    kf: CvKalman,
    pub chi2_gate: f64,
}

impl Default for Pos2Filter {
    fn default() -> Self {
        Pos2Filter {
            obs: VecDeque::new(),
            model: Pos2Model::Static,
            cusum_x: Cusum::new(0.5, 4.0),
            cusum_z: Cusum::new(0.5, 4.0),
            kf: CvKalman::default(),
            chi2_gate: CHI2_2DOF_99,
        }
    }
}

impl Pos2Filter {
    pub fn clear(&mut self) {
        self.obs.clear();
        self.model = Pos2Model::Static;
        self.cusum_x.reset();
        self.cusum_z.reset();
        self.kf = CvKalman::default();
    }

    pub fn len(&self) -> usize {
        self.obs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    pub fn model(&self) -> Pos2Model {
        self.model
    }

    pub fn observations(&self) -> &VecDeque<Pos2Obs> {
        &self.obs
    }

    /// Current estimate at `at_tick` (for `Static` targets, tick is unused; for `Dynamic` targets,
    /// extrapolates constant velocity to `at_tick`).
    pub fn estimate(&self, at_tick: Option<u64>) -> Option<Pos2Estimate> {
        if self.obs.is_empty() {
            return None;
        }
        if self.model == Pos2Model::Dynamic && self.kf.initialized {
            let tick = at_tick.unwrap_or(self.kf.last_tick);
            let (xp, pp) = self.kf.predict_at(tick);
            return Some(Pos2Estimate {
                mean: [xp[0], xp[1]],
                cov: [pp[0][0] as f32, pp[0][1] as f32, pp[1][1] as f32],
                n: self.obs.len(),
                model: Pos2Model::Dynamic,
                chi2_gate: self.chi2_gate,
            });
        }
        // Static inverse-variance weighted mean & combined covariance (Fisher info⁻¹ + weighted scatter).
        let mut w_sum = 0.0;
        let mut wx = 0.0;
        let mut wz = 0.0;
        for o in &self.obs {
            let s2 = o.sigma_m.max(1e-3).powi(2);
            let w = 1.0 / s2;
            w_sum += w;
            wx += w * o.x;
            wz += w * o.z;
        }
        let mx = wx / w_sum;
        let mz = wz / w_sum;
        let fisher_var = 1.0 / w_sum;
        let (mut cxx, mut cxz, mut czz) = (fisher_var, 0.0, fisher_var);
        if self.obs.len() >= 2 {
            // Add empirical weighted excess scatter so mis-specified sensor sigma never causes overconfidence.
            let mut sxx = 0.0;
            let mut sxz = 0.0;
            let mut szz = 0.0;
            let mut w2_sum = 0.0;
            for o in &self.obs {
                let w = 1.0 / o.sigma_m.max(1e-3).powi(2);
                let dx = o.x - mx;
                let dz = o.z - mz;
                sxx += w * dx * dx;
                sxz += w * dx * dz;
                szz += w * dz * dz;
                w2_sum += w * w;
            }
            let denom = (w_sum - w2_sum / w_sum).max(1e-9);
            let n_eff = (w_sum * w_sum / w2_sum.max(1e-12)).max(1.0);
            // Only add positive excess scatter above the nominal measurement variance.
            let mean_s2 = self.obs.len() as f64 / w_sum;
            let ex_x = (sxx / denom - mean_s2).max(0.0);
            let ex_z = (szz / denom - mean_s2).max(0.0);
            cxx += ex_x / n_eff;
            czz += ex_z / n_eff;
            cxz += (sxz / denom) / n_eff;
        }
        Some(Pos2Estimate {
            mean: [mx, mz],
            cov: [cxx as f32, cxz as f32, czz as f32],
            n: self.obs.len(),
            model: Pos2Model::Static,
            chi2_gate: self.chi2_gate,
        })
    }

    /// Push a new sighting `(x, z, sigma_m, tick)`, returning whether it triggered a Static->Dynamic switch.
    pub fn push(&mut self, obs: Pos2Obs) -> bool {
        if !obs.x.is_finite() || !obs.z.is_finite() || !obs.sigma_m.is_finite() {
            return false;
        }
        let mut switched = false;
        if self.model == Pos2Model::Static {
            if let Some(est) = self.estimate(Some(obs.tick)) {
                if self.obs.len() >= 2 {
                    let s2 = obs.sigma_m.max(1e-3).powi(2);
                    let sx = ((est.cov[0] as f64).max(0.0) + s2).sqrt().max(1e-3);
                    let sz = ((est.cov[2] as f64).max(0.0) + s2).sqrt().max(1e-3);
                    let zx = (obs.x - est.mean[0]) / sx;
                    let zz = (obs.z - est.mean[1]) / sz;
                    if self.cusum_x.push(zx).is_some() || self.cusum_z.push(zz).is_some() {
                        self.model = Pos2Model::Dynamic;
                        switched = true;
                        let (vx, vz) = if let Some(prev) = self.obs.back() {
                            let dt = obs.tick.saturating_sub(prev.tick).max(1) as f64;
                            ((obs.x - prev.x) / dt, (obs.z - prev.z) / dt)
                        } else {
                            (0.0, 0.0)
                        };
                        self.kf.init(obs, vx, vz);
                    }
                }
            }
        } else if self.kf.initialized {
            self.kf.update(obs);
        }

        self.obs.push_back(obs);
        while self.obs.len() > POS2_RING_CAP {
            self.obs.pop_front();
        }
        switched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos2_static_equals_weighted_mean() {
        let mut f = Pos2Filter::default();
        let pts = [
            Pos2Obs { x: 4.0, z: 2.0, sigma_m: 0.10, tick: 1 },
            Pos2Obs { x: 4.2, z: 2.1, sigma_m: 0.20, tick: 2 },
            Pos2Obs { x: 3.9, z: 1.95, sigma_m: 0.05, tick: 3 },
        ];
        for p in pts {
            f.push(p);
        }
        let est = f.estimate(None).unwrap();
        assert_eq!(est.model, Pos2Model::Static);
        let (mut wx, mut wz, mut wsum) = (0.0, 0.0, 0.0);
        for p in pts {
            let w = 1.0 / (p.sigma_m * p.sigma_m);
            wx += w * p.x;
            wz += w * p.z;
            wsum += w;
        }
        assert!((est.mean[0] - wx / wsum).abs() < 1e-6);
        assert!((est.mean[1] - wz / wsum).abs() < 1e-6);
        // Outlier far outside 99% ellipse (d² > 9.21) is rejected.
        assert!(!est.accepts(5.0, 3.0, 0.10));
        assert!(est.mahalanobis_sq(5.0, 3.0, 0.10) > CHI2_2DOF_99);
    }

    #[test]
    fn pos2_gate_coverage() {
        let mut state = 0x55aa_1122_3344_7788u64;
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
        let trials = 600usize;
        let mut accepted = 0usize;
        for _ in 0..trials {
            let mut f = Pos2Filter::default();
            for t in 0..5 {
                f.push(Pos2Obs { x: 3.0 + 0.12 * normal(), z: 5.0 + 0.12 * normal(), sigma_m: 0.12, tick: t });
            }
            let est = f.estimate(None).unwrap();
            let nx = 3.0 + 0.12 * normal();
            let nz = 5.0 + 0.12 * normal();
            if est.accepts(nx, nz, 0.12) {
                accepted += 1;
            }
        }
        let rate = accepted as f64 / trials as f64;
        assert!((0.96..=1.0).contains(&rate), "nominal 99% chi2 gate acceptance rate {rate:.3}");
    }

    #[test]
    fn pos2_switches_to_dynamic_on_moving_target() {
        let mut f = Pos2Filter::default();
        for t in 0..12u64 {
            f.push(Pos2Obs { x: 1.0 + 0.15 * t as f64, z: 2.0, sigma_m: 0.08, tick: t });
        }
        assert_eq!(f.model(), Pos2Model::Dynamic, "sustained motion must switch filter to Dynamic CV Kalman");
        let pred = f.estimate(Some(13)).unwrap();
        assert!(pred.mean[0] > 1.0 + 0.15 * 11.0, "dynamic predictor extrapolates velocity: {}", pred.mean[0]);
    }
}
