//! Exogenous residualization wrapper `ExogResidual<P>` (Phase 4 – Part 2 · D2 Path A).
//!
//! Fits closed-form ridge regression `y = b0 + X * beta + r` (`K <= 4` exogenous covariates)
//! with leave-one-out / last-point cross-validation over a fixed lambda grid, forecasts the
//! residual `r` with any univariate [`Predictor`] `P`, and adds `b0 + X_future * beta` back.
//! Never touches neural weights and falls back cleanly when `n < K + 3`.

use serde::{Deserialize, Serialize};

use super::fusion::{Band, Predictor};
use super::registry::{Caps, Exog, SeriesCtx};

/// Candidate ridge regularization strengths evaluated by cross-validation.
pub const RIDGE_GRID: [f64; 5] = [1e-3, 1e-2, 0.1, 1.0, 10.0];
/// Maximum number of exogenous covariates accepted by closed-form ridge regression.
pub const MAX_EXOG_K: usize = 4;

/// Diagnostic fit summary from [`ExogResidual`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExogFit {
    pub intercept: f64,
    pub beta: Vec<f64>,
    pub lambda: f64,
    pub used: bool,
    pub resid_ac1: Option<f64>,
    pub note: Option<String>,
}

/// Solve symmetric positive-definite linear system `A x = b` via Gaussian elimination with partial pivoting.
pub(crate) fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return None;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-12 || !best.is_finite() {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let diag = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / diag;
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = ((i + 1)..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// Fit ridge regression `y = intercept + X * beta` with penalty `lambda` on slopes (intercept unpenalized).
pub fn fit_ridge(y: &[f64], x: &[Vec<f64>], lambda: f64) -> Option<(f64, Vec<f64>)> {
    let n = y.len();
    if n == 0 || x.len() != n {
        return None;
    }
    let k = x[0].len();
    if k == 0 || k > MAX_EXOG_K || x.iter().any(|row| row.len() != k) {
        return None;
    }
    let p = k + 1;
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        if !y[i].is_finite() || x[i].iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut row = Vec::with_capacity(p);
        row.push(1.0);
        row.extend_from_slice(&x[i]);
        for r in 0..p {
            xty[r] += row[r] * y[i];
            for c in 0..p {
                xtx[r][c] += row[r] * row[c];
            }
        }
    }
    for j in 1..p {
        xtx[j][j] += lambda.max(1e-9);
    }
    let theta = solve_linear(xtx, xty)?;
    Some((theta[0], theta[1..].to_vec()))
}

fn lag1_autocorr(r: &[f64]) -> Option<f64> {
    if r.len() < 4 {
        return None;
    }
    let mean = r.iter().sum::<f64>() / r.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..r.len() {
        let d = r[i] - mean;
        den += d * d;
        if i + 1 < r.len() {
            num += d * (r[i + 1] - mean);
        }
    }
    (den > 1e-12).then_some((num / den).clamp(-1.0, 1.0))
}

/// Select `lambda` from [`RIDGE_GRID`] by leave-one-out validation across the tail of `(y, x)`.
pub fn fit_ridge_cv(y: &[f64], x: &[Vec<f64>]) -> ExogFit {
    let n = y.len();
    let k = x.first().map_or(0, Vec::len);
    if k == 0 || k > MAX_EXOG_K || x.len() != n || x.iter().any(|r| r.len() != k) {
        return ExogFit {
            intercept: 0.0,
            beta: Vec::new(),
            lambda: 0.0,
            used: false,
            resid_ac1: None,
            note: Some("외생 공변량 차원 부적합 (1..=4 필요)".into()),
        };
    }
    if n < k + 3 {
        return ExogFit {
            intercept: 0.0,
            beta: vec![0.0; k],
            lambda: 0.0,
            used: false,
            resid_ac1: None,
            note: Some(format!("외생 회귀 관측 부족 ({n} < {}) — 단변량 예측기로 대체", k + 3)),
        };
    }
    let folds = n.min(8);
    let mut best_lam = RIDGE_GRID[0];
    let mut best_mse = f64::INFINITY;
    for &lam in &RIDGE_GRID {
        let mut sq_err = 0.0;
        let mut cnt = 0usize;
        for f in 0..folds {
            let hold = n - 1 - f;
            let mut y_tr = Vec::with_capacity(n - 1);
            let mut x_tr = Vec::with_capacity(n - 1);
            for i in 0..n {
                if i != hold {
                    y_tr.push(y[i]);
                    x_tr.push(x[i].clone());
                }
            }
            if let Some((b0, beta)) = fit_ridge(&y_tr, &x_tr, lam) {
                let pred = b0 + x[hold].iter().zip(&beta).map(|(xi, bi)| xi * bi).sum::<f64>();
                sq_err += (y[hold] - pred).powi(2);
                cnt += 1;
            }
        }
        if cnt > 0 {
            let mse = sq_err / cnt as f64;
            if mse < best_mse {
                best_mse = mse;
                best_lam = lam;
            }
        }
    }
    match fit_ridge(y, x, best_lam) {
        Some((intercept, beta)) => {
            let resid: Vec<f64> = (0..n)
                .map(|i| y[i] - (intercept + x[i].iter().zip(&beta).map(|(xi, bi)| xi * bi).sum::<f64>()))
                .collect();
            ExogFit {
                intercept,
                beta,
                lambda: best_lam,
                used: true,
                resid_ac1: lag1_autocorr(&resid),
                note: None,
            }
        }
        None => ExogFit {
            intercept: 0.0,
            beta: vec![0.0; k],
            lambda: best_lam,
            used: false,
            resid_ac1: None,
            note: Some("외생 능형 회귀 풀기 실패 — 단변량 예측기로 대체".into()),
        },
    }
}

/// Exogenous residualization wrapper around any univariate [`Predictor`].
pub struct ExogResidual<P: Predictor> {
    pub inner: P,
    label: String,
}

impl<P: Predictor> ExogResidual<P> {
    pub fn new(inner: P) -> Self {
        let label = format!("exog_{}", inner.name());
        Self { inner, label }
    }

    /// Fit exogenous regression on `ctx`, forecast the residual via `inner`, and return `(Band, ExogFit)`.
    pub fn predict_with_fit(&self, ctx: &SeriesCtx<'_>, horizon: usize) -> Result<(Band, ExogFit), String> {
        let (Some(x_past), Some(x_fut)) = (ctx.exog_past, ctx.exog_future) else {
            let band = self.inner.predict(ctx.y, horizon)?;
            return Ok((
                band,
                ExogFit {
                    intercept: 0.0,
                    beta: Vec::new(),
                    lambda: 0.0,
                    used: false,
                    resid_ac1: None,
                    note: Some("외생 행렬 없음 — 내부 단변량 예측기 사용".into()),
                },
            ));
        };
        if x_fut.len() < horizon {
            let band = self.inner.predict(ctx.y, horizon)?;
            return Ok((
                band,
                ExogFit {
                    intercept: 0.0,
                    beta: Vec::new(),
                    lambda: 0.0,
                    used: false,
                    resid_ac1: None,
                    note: Some("미래 외생 길이 부족 — 내부 단변량 예측기 사용".into()),
                },
            ));
        }
        let fit = fit_ridge_cv(ctx.y, x_past);
        if !fit.used {
            let band = self.inner.predict(ctx.y, horizon)?;
            return Ok((band, fit));
        }
        let resid: Vec<f64> = (0..ctx.y.len())
            .map(|i| ctx.y[i] - (fit.intercept + x_past[i].iter().zip(&fit.beta).map(|(a, b)| a * b).sum::<f64>()))
            .collect();
        let mut band = self.inner.predict(&resid, horizon)?;
        for h in 0..horizon {
            let trend = fit.intercept + x_fut[h].iter().zip(&fit.beta).map(|(a, b)| a * b).sum::<f64>();
            band.point[h] += trend;
            if let Some(lo) = band.lo.as_mut() {
                if h < lo.len() {
                    lo[h] += trend;
                }
            }
            if let Some(hi) = band.hi.as_mut() {
                if h < hi.len() {
                    hi[h] += trend;
                }
            }
            if let Some((_, rows)) = band.quantiles.as_mut() {
                for row in rows {
                    if h < row.len() {
                        row[h] += trend;
                    }
                }
            }
        }
        Ok((band, fit))
    }
}

impl<P: Predictor> Predictor for ExogResidual<P> {
    fn name(&self) -> &str {
        &self.label
    }

    fn rev(&self) -> String {
        format!("exog-v1+{}", self.inner.rev())
    }

    fn min_history(&self) -> usize {
        self.inner.min_history()
    }

    fn max_horizon(&self) -> usize {
        self.inner.max_horizon()
    }

    fn caps(&self) -> Caps {
        let mut c = self.inner.caps();
        c.exog = Exog::KnownFuture;
        c
    }

    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        self.inner.predict(series, horizon)
    }

    fn predict_ctx(&self, ctx: &SeriesCtx<'_>, horizon: usize) -> Result<Band, String> {
        Ok(self.predict_with_fit(ctx, horizon)?.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ts::fusion::{forecast, HoltPredictor};

    #[test]
    fn exog_residual_recovers_known_beta() {
        // Synthetic 400-trial check on y_t = 0.2 + 0.5 * depth_t + AR(1) residual (phi = 0.6).
        let mut state: u64 = 0x517cc1b727220a95;
        let mut randn = || {
            let mut u = 0.0;
            for _ in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u += (state >> 11) as f64 / (1u64 << 53) as f64;
            }
            u - 6.0
        };
        let trials = 400usize;
        let mut beta_err_sum = 0.0;
        let mut sq_exog = 0.0;
        let mut sq_uni = 0.0;
        let mut worse_count = 0usize;
        let wrapper = ExogResidual::new(HoltPredictor);
        for _ in 0..trials {
            let n = 25usize;
            let mut depth = Vec::with_capacity(n + 1);
            let mut y = Vec::with_capacity(n + 1);
            let mut r = 0.0;
            for t in 0..=n {
                let d = 1.0 + 3.0 * ((t as f64 * 0.45).sin().abs()) + 0.3 * randn();
                r = 0.6 * r + 0.05 * randn();
                depth.push(vec![d]);
                y.push(0.2 + 0.5 * d + r);
            }
            let ctx = SeriesCtx {
                y: &y[..n],
                t_s: None,
                exog_past: Some(&depth[..n]),
                exog_future: Some(&depth[n..n + 1]),
                statics: &[],
            };
            let (band, fit) = wrapper.predict_with_fit(&ctx, 1).expect("exog forecast");
            assert!(fit.used);
            beta_err_sum += (fit.beta[0] - 0.5).abs();
            let err_e = (band.point[0] - y[n]).powi(2);
            let uni = forecast(&y[..n], 1, &[], &[]);
            let err_u = (uni.point[0] - y[n]).powi(2);
            sq_exog += err_e;
            sq_uni += err_u;
            if err_e > err_u {
                worse_count += 1;
            }
        }
        let mean_beta_err = beta_err_sum / trials as f64;
        let rmse_exog = (sq_exog / trials as f64).sqrt();
        let rmse_uni = (sq_uni / trials as f64).sqrt();
        assert!(mean_beta_err < 0.05, "beta recovery error too large: {mean_beta_err:.4}");
        assert!(rmse_exog < rmse_uni * 0.5, "exog RMSE {rmse_exog:.4} should strongly beat univariate {rmse_uni:.4}");
        assert!((worse_count as f64 / trials as f64) < 0.20, "worse fraction too high: {worse_count}/{trials}");
    }

    #[test]
    fn exog_residual_falls_back_gracefully_when_history_too_short() {
        let wrapper = ExogResidual::new(HoltPredictor);
        let y = [1.0, 1.2, 1.3];
        let x_past = vec![vec![1.0, 2.0]; 3];
        let x_fut = vec![vec![1.5, 2.5]];
        let ctx = SeriesCtx {
            y: &y,
            t_s: None,
            exog_past: Some(&x_past),
            exog_future: Some(&x_fut),
            statics: &[],
        };
        let (band, fit) = wrapper.predict_with_fit(&ctx, 1).expect("fallback works");
        assert!(!fit.used);
        assert!(fit.note.is_some());
        assert_eq!(band.point.len(), 1);
    }
}
