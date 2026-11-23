use crate::fusion::{Band, Predictor};
use crate::registry::{Caps, Channels, Exog};
use crate::ttm::TtmForecaster;

const TTM_QUANTILES_DEFAULT: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

pub struct TtmPredictor<'a>(pub &'a TtmForecaster);

/// Conformal-calibrated TTM predictor wrapper scaling `(q50 - q10)` and `(q90 - q50)`
/// by empirical `(c_lo, c_hi)` (Phase 4 – Part 1 · D4).
pub struct CalibratedTtmPredictor<'a> {
    pub model: &'a TtmForecaster,
    pub c_lo: f64,
    pub c_hi: f64,
}

impl<'a> TtmPredictor<'a> {
    pub fn calibrated(&self, c_lo: f64, c_hi: f64) -> CalibratedTtmPredictor<'a> {
        CalibratedTtmPredictor { model: self.0, c_lo: c_lo.clamp(0.5, 3.0), c_hi: c_hi.clamp(0.5, 3.0) }
    }
}

fn build_band(model: &TtmForecaster, series: &[f64], horizon: usize, c_lo: f64, c_hi: f64) -> Result<Band, String> {
    let out = model.forecast(series)?;
    let h = horizon.min(out.point.len());
    let point: Vec<f64> = out.point.iter().take(h).copied().collect();
    let pick = |level: f64| -> Option<Vec<f64>> {
        let q = out.quantiles.as_ref()?;
        let i = out.quantile_levels.iter().position(|l| (l - level).abs() < 1e-6)?;
        Some(q.get(i)?.iter().take(h).copied().collect())
    };
    let raw_lo = pick(0.1);
    let raw_hi = pick(0.9);
    let lo = raw_lo.map(|v| {
        point
            .iter()
            .zip(&v)
            .map(|(&p50, &q10)| p50 - c_lo * (p50 - q10).max(0.0))
            .collect()
    });
    let hi = raw_hi.map(|v| {
        point
            .iter()
            .zip(&v)
            .map(|(&p50, &q90)| p50 + c_hi * (q90 - p50).max(0.0))
            .collect()
    });
    let quantiles = out.quantiles.as_ref().map(|rows| {
        let truncated: Vec<Vec<f64>> = rows.iter().map(|r| r.iter().take(h).copied().collect()).collect();
        (out.quantile_levels.clone(), truncated)
    });
    Ok(Band { point, lo, hi, quantiles })
}

fn ttm_caps(model: &TtmForecaster) -> Caps {
    Caps {
        min_obs: model.min_history(),
        max_horizon: model.prediction_length(),
        channels: Channels::Independent,
        exog: Exog::None,
        statics: false,
        quantiles: if model.config.multi_quantile_head { &TTM_QUANTILES_DEFAULT } else { &[] },
        irregular_time: false,
        cost: 2,
    }
}

impl Predictor for TtmPredictor<'_> {
    fn name(&self) -> &str {
        "ttm"
    }

    fn rev(&self) -> String {
        format!("r3-{}-{}-{}", self.0.context_length(), self.0.prediction_length(), self.0.fp8())
    }

    fn min_history(&self) -> usize {
        self.0.min_history()
    }

    fn max_horizon(&self) -> usize {
        self.0.prediction_length()
    }

    fn caps(&self) -> Caps {
        ttm_caps(self.0)
    }

    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        build_band(self.0, series, horizon, 1.0, 1.0)
    }
}

impl Predictor for CalibratedTtmPredictor<'_> {
    fn name(&self) -> &str {
        "ttm"
    }

    fn rev(&self) -> String {
        format!("r3-{}-{}-{}", self.model.context_length(), self.model.prediction_length(), self.model.fp8())
    }

    fn min_history(&self) -> usize {
        self.model.min_history()
    }

    fn max_horizon(&self) -> usize {
        self.model.prediction_length()
    }

    fn caps(&self) -> Caps {
        ttm_caps(self.model)
    }

    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        build_band(self.model, series, horizon, self.c_lo, self.c_hi)
    }
}
