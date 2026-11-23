//! Simulator forecast adapters T1 (affine recalibration) and T2 (pre-fit ridge residual)
//! with `ENGINE_REV` invalidation, Group B/Truth control-boundary guard, and adoption gating
//! (Phase 4 – Part 2 · D6).

use serde::{Deserialize, Serialize};

use crate::exog::{fit_ridge_cv, ExogFit};
use crate::fusion::Band;
use crate::registry::Group;

/// Guard enforcing that control-bound adapters never train on Group B (human benchmarks) or Truth.
pub fn adapter_accepts_group(group: Group, for_control: bool) -> bool {
    if for_control {
        matches!(group, Group::A)
    } else {
        true
    }
}

/// Evaluation metrics used to gate adoption of a T1/T2 adapter against T0 baseline.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdapterEval {
    pub rmse_base: f64,
    pub rmse_adapted: f64,
    pub cov_base: f64,
    pub cov_adapted: f64,
    pub adopted: bool,
}

/// Check adoption rule: adapted 1-step RMSE must not worsen vs baseline, and nominal 80% coverage
/// must not drop by more than `0.02`.
pub fn evaluate_adoption(
    truth: &[f64],
    base_pred: &[f64],
    base_band: &[(f64, f64)],
    adapted_pred: &[f64],
    adapted_band: &[(f64, f64)],
) -> AdapterEval {
    let n = truth.len().min(base_pred.len()).min(adapted_pred.len());
    if n == 0 {
        return AdapterEval { rmse_base: 0.0, rmse_adapted: 0.0, cov_base: 0.0, cov_adapted: 0.0, adopted: false };
    }
    let rmse = |preds: &[f64]| -> f64 {
        (truth[..n].iter().zip(&preds[..n]).map(|(y, p)| (y - p).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let cov = |bands: &[(f64, f64)]| -> f64 {
        let m = n.min(bands.len());
        if m == 0 {
            return 1.0;
        }
        truth[..m].iter().zip(&bands[..m]).filter(|(y, (lo, hi))| *y >= lo && *y <= hi).count() as f64 / m as f64
    };
    let rmse_base = rmse(base_pred);
    let rmse_adapted = rmse(adapted_pred);
    let cov_base = cov(base_band);
    let cov_adapted = cov(adapted_band);
    let adopted = rmse_adapted <= rmse_base + 1e-9 && cov_adapted >= cov_base - 0.02;
    AdapterEval { rmse_base, rmse_adapted, cov_base, cov_adapted, adopted }
}

/// T1 output affine recalibration adapter `y_cal = a * y_hat + b` stamped with `engine_rev`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineAdapter {
    pub series_kind: String,
    pub engine_rev: String,
    pub rev: String,
    pub a: f64,
    pub b: f64,
    pub n_train: usize,
    pub for_control: bool,
}

impl AffineAdapter {
    /// Fit closed-form OLS `y ≈ a * y_hat + b` on dev-seed training pairs `(y_hat, y_true)`.
    /// Refuses `Group::B` / `Group::Truth` when `for_control` is true.
    pub fn fit(
        series_kind: impl Into<String>,
        engine_rev: impl Into<String>,
        group: Group,
        for_control: bool,
        y_hat: &[f64],
        y_true: &[f64],
    ) -> Result<Self, String> {
        if !adapter_accepts_group(group, for_control) {
            return Err("제어용 어댑터는 Group B 또는 Truth 계열로 학습할 수 없습니다".into());
        }
        let n = y_hat.len().min(y_true.len());
        if n < 4 {
            return Err(format!("T1 아핀 어댑터 학습 표본 부족 ({n} < 4)"));
        }
        let mx = y_hat[..n].iter().sum::<f64>() / n as f64;
        let my = y_true[..n].iter().sum::<f64>() / n as f64;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for i in 0..n {
            let dx = y_hat[i] - mx;
            sxx += dx * dx;
            sxy += dx * (y_true[i] - my);
        }
        let a = if sxx > 1e-12 { (sxy / sxx).clamp(0.1, 10.0) } else { 1.0 };
        let b = my - a * mx;
        let rev_str = engine_rev.into();
        let sk = series_kind.into();
        Ok(Self {
            rev: format!("t1-{rev_str}"),
            series_kind: sk,
            engine_rev: rev_str,
            a,
            b,
            n_train: n,
            for_control,
        })
    }

    pub fn is_valid_for(&self, current_engine_rev: &str) -> bool {
        self.engine_rev == current_engine_rev
    }

    pub fn apply_scalar(&self, y_hat: f64) -> f64 {
        self.a * y_hat + self.b
    }

    pub fn apply_band(&self, band: &Band, current_engine_rev: &str) -> Result<Band, String> {
        if !self.is_valid_for(current_engine_rev) {
            return Err(format!(
                "T1 어댑터 무효: 학습 개정({}) != 현재 개정({})",
                self.engine_rev, current_engine_rev
            ));
        }
        let map_vec = |v: &[f64]| v.iter().map(|&x| self.apply_scalar(x)).collect::<Vec<_>>();
        Ok(Band {
            point: map_vec(&band.point),
            lo: band.lo.as_ref().map(|v| map_vec(v)),
            hi: band.hi.as_ref().map(|v| map_vec(v)),
            quantiles: band.quantiles.as_ref().map(|(lv, rows)| (lv.clone(), rows.iter().map(|row| map_vec(row)).collect())),
        })
    }
}

/// T2 pre-fit ridge residual adapter `y_cal = y_hat + b0 + X * beta` stamped with `engine_rev`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualAdapter {
    pub series_kind: String,
    pub engine_rev: String,
    pub rev: String,
    pub fit: ExogFit,
    pub n_train: usize,
    pub for_control: bool,
}

impl ResidualAdapter {
    /// Fit ridge regression predicting baseline residual `e = y_true - y_hat` from covariates `x`.
    pub fn fit(
        series_kind: impl Into<String>,
        engine_rev: impl Into<String>,
        group: Group,
        for_control: bool,
        y_hat: &[f64],
        y_true: &[f64],
        x: &[Vec<f64>],
    ) -> Result<Self, String> {
        if !adapter_accepts_group(group, for_control) {
            return Err("제어용 T2 어댑터는 Group B 또는 Truth 계열로 학습할 수 없습니다".into());
        }
        let n = y_hat.len().min(y_true.len()).min(x.len());
        let resid: Vec<f64> = (0..n).map(|i| y_true[i] - y_hat[i]).collect();
        let fit = fit_ridge_cv(&resid, &x[..n]);
        if !fit.used {
            return Err(fit.note.unwrap_or_else(|| "T2 잔차 어댑터 학습 실패".into()));
        }
        let rev_str = engine_rev.into();
        let sk = series_kind.into();
        Ok(Self {
            rev: format!("t2-{rev_str}"),
            series_kind: sk,
            engine_rev: rev_str,
            fit,
            n_train: n,
            for_control,
        })
    }

    pub fn is_valid_for(&self, current_engine_rev: &str) -> bool {
        self.engine_rev == current_engine_rev
    }

    pub fn correction(&self, cov: &[f64]) -> f64 {
        self.fit.intercept + cov.iter().zip(&self.fit.beta).map(|(a, b)| a * b).sum::<f64>()
    }

    pub fn apply_band(&self, band: &Band, x_future: &[Vec<f64>], current_engine_rev: &str) -> Result<Band, String> {
        if !self.is_valid_for(current_engine_rev) {
            return Err(format!(
                "T2 어댑터 무효: 학습 개정({}) != 현재 개정({})",
                self.engine_rev, current_engine_rev
            ));
        }
        let h = band.point.len().min(x_future.len());
        let mut out = band.clone();
        for i in 0..h {
            let corr = self.correction(&x_future[i]);
            out.point[i] += corr;
            if let Some(lo) = out.lo.as_mut() {
                if i < lo.len() {
                    lo[i] += corr;
                }
            }
            if let Some(hi) = out.hi.as_mut() {
                if i < hi.len() {
                    hi[i] += corr;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_refuse_group_b_and_invalidate_on_engine_rev_change() {
        let yhat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ytrue: Vec<f64> = yhat.iter().map(|v| 1.2 * v + 0.5).collect();
        assert!(AffineAdapter::fit("nn.score", "rev1", Group::B, true, &yhat, &ytrue).is_err());
        assert!(AffineAdapter::fit("truth.range", "rev1", Group::Truth, true, &yhat, &ytrue).is_err());
        let t1 = AffineAdapter::fit("heat:shoulder_r", "rev1", Group::A, true, &yhat, &ytrue).unwrap();
        assert!((t1.a - 1.2).abs() < 1e-6 && (t1.b - 0.5).abs() < 1e-6);
        assert!(t1.is_valid_for("rev1"));
        assert!(!t1.is_valid_for("rev2"));
        let band = Band { point: vec![2.0], lo: Some(vec![1.5]), hi: Some(vec![2.5]), quantiles: None };
        let cal = t1.apply_band(&band, "rev1").unwrap();
        assert!((cal.point[0] - 2.9).abs() < 1e-6);
        assert!(t1.apply_band(&band, "rev2").is_err());

        let cov: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64 * 0.25]).collect();
        let base: Vec<f64> = (0..12).map(|i| 0.5 + 0.1 * i as f64).collect();
        let truth: Vec<f64> = (0..12).map(|i| base[i] + 0.3 + 0.4 * cov[i][0]).collect();
        assert!(ResidualAdapter::fit("nn.score", "rev1", Group::B, true, &base, &truth, &cov).is_err());
        let t2 = ResidualAdapter::fit("locate.score", "rev1", Group::A, true, &base, &truth, &cov).unwrap();
        assert!(t2.is_valid_for("rev1") && !t2.is_valid_for("rev2"));
        let eval_pred: Vec<f64> = (0..12).map(|i| base[i] + t2.correction(&cov[i])).collect();
        let b_base: Vec<(f64, f64)> = base.iter().map(|&p| (p - 0.5, p + 0.5)).collect();
        let b_adap: Vec<(f64, f64)> = eval_pred.iter().map(|&p| (p - 0.5, p + 0.5)).collect();
        let ev = evaluate_adoption(&truth, &base, &b_base, &eval_pred, &b_adap);
        assert!(ev.adopted && ev.rmse_adapted < ev.rmse_base * 0.1);
    }
}
