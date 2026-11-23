//! Predictor registry, capability declarations, and series eligibility rules
//! (Phase 4 – Part 2 · D1, D5).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channels {
    Univariate,
    Independent,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exog {
    None,
    PastOnly,
    KnownFuture,
}

/// Static capabilities declared by a [`Predictor`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Caps {
    pub min_obs: usize,
    pub max_horizon: usize,
    pub channels: Channels,
    pub exog: Exog,
    pub statics: bool,
    pub quantiles: &'static [f64],
    pub irregular_time: bool,
    /// Cost tier: `0` closed-form, `1` grid search, `2` neural forward pass.
    pub cost: u8,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            min_obs: 1,
            max_horizon: usize::MAX,
            channels: Channels::Univariate,
            exog: Exog::None,
            statics: false,
            quantiles: &[],
            irregular_time: true,
            cost: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictorSpec {
    pub id: String,
    pub rev: String,
    pub caps: Caps,
}

impl PredictorSpec {
    pub fn key(&self) -> String {
        format!("{}@{}", self.id, self.rev)
    }

    /// Canonical SDS error axis name `fusion_mse:{id}@{rev}` (Phase 4 – Part 2 · D1).
    pub fn mse_axis(&self) -> String {
        format!("fusion_mse:{}@{}", self.id, self.rev)
    }
}

/// Provenance group for forecast series: Group B & Truth must never enter decision files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Group {
    A,
    B,
    Truth,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    Uniform { dt_s: f64 },
    Event,
    Rollout,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeriesSpec {
    pub kind: String,
    pub group: Group,
    pub cadence: Cadence,
    pub monotone: bool,
    pub need_horizon: usize,
}

/// Rich forecasting input context carrying values, optional timestamps, exogenous matrices,
/// and static key-value tags (Phase 4 – Part 2 · D2).
#[derive(Clone, Debug)]
pub struct SeriesCtx<'a> {
    pub y: &'a [f64],
    pub t_s: Option<&'a [f64]>,
    pub exog_past: Option<&'a [Vec<f64>]>,
    pub exog_future: Option<&'a [Vec<f64>]>,
    pub statics: &'a [(&'a str, &'a str)],
}

impl<'a> SeriesCtx<'a> {
    pub fn simple(y: &'a [f64]) -> Self {
        SeriesCtx { y, t_s: None, exog_past: None, exog_future: None, statics: &[] }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eligibility {
    pub eligible: bool,
    pub skip_reason: Option<String>,
}

/// Evaluate whether a predictor with `caps` is eligible to forecast `spec` given observed slice `y`,
/// missing fraction `missing_share`, and minimum variance `var_floor` (Phase 4 – Part 2 · D5).
pub fn eligible(spec: &SeriesSpec, y: &[f64], missing_share: f64, var_floor: f64, caps: &Caps) -> Eligibility {
    let finite: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    if !caps.irregular_time && !matches!(spec.cadence, Cadence::Uniform { .. }) {
        return Eligibility {
            eligible: false,
            skip_reason: Some("등간격 계열이 아니라 불참(사건·롤아웃 계열)".into()),
        };
    }
    let min_req = if caps.cost >= 2 { caps.min_obs.max(8) } else { caps.min_obs };
    if finite.len() < min_req {
        return Eligibility {
            eligible: false,
            skip_reason: Some(format!("관측 {}점 < 최소 {}점", finite.len(), min_req)),
        };
    }
    if missing_share > 0.30 {
        return Eligibility {
            eligible: false,
            skip_reason: Some(format!("결측 비율 {:.0}% > 상한 30%", missing_share * 100.0)),
        };
    }
    if spec.monotone && caps.cost >= 2 {
        return Eligibility {
            eligible: false,
            skip_reason: Some("단조 누적 계열은 증분 변환 전까지 신경망 모델 제외".into()),
        };
    }
    if finite.len() >= 2 && var_floor > 0.0 {
        let m = finite.iter().sum::<f64>() / finite.len() as f64;
        let var = finite.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (finite.len() - 1) as f64;
        if var < var_floor && caps.cost >= 2 {
            return Eligibility {
                eligible: false,
                skip_reason: Some(format!("분산 {:.2e} < 분해능 하한 {:.2e}", var, var_floor)),
            };
        }
    }
    if spec.need_horizon > caps.max_horizon {
        return Eligibility {
            eligible: false,
            skip_reason: Some(format!("필요 지평 {} > 최대 지평 {}", spec.need_horizon, caps.max_horizon)),
        };
    }
    Eligibility { eligible: true, skip_reason: None }
}

/// Registry of active predictors keyed by `id@rev` (rejecting duplicate keys so error histories never mix).
#[derive(Clone, Debug, Default)]
pub struct PredictorRegistry {
    specs: BTreeMap<String, PredictorSpec>,
}

impl PredictorRegistry {
    pub fn with_builtins() -> Self {
        let mut r = Self::default();
        let _ = r.register(PredictorSpec {
            id: "last".into(),
            rev: "v1".into(),
            caps: Caps { min_obs: 1, max_horizon: usize::MAX, channels: Channels::Univariate, exog: Exog::None, statics: false, quantiles: &[], irregular_time: true, cost: 0 },
        });
        let _ = r.register(PredictorSpec {
            id: "ewma".into(),
            rev: "v1".into(),
            caps: Caps { min_obs: 2, max_horizon: usize::MAX, channels: Channels::Univariate, exog: Exog::None, statics: false, quantiles: &[], irregular_time: true, cost: 0 },
        });
        let _ = r.register(PredictorSpec {
            id: "holt".into(),
            rev: "v1".into(),
            caps: Caps { min_obs: 3, max_horizon: usize::MAX, channels: Channels::Univariate, exog: Exog::None, statics: false, quantiles: &[0.1, 0.5, 0.9], irregular_time: true, cost: 1 },
        });
        r
    }

    pub fn register(&mut self, spec: PredictorSpec) -> Result<(), String> {
        let k = spec.key();
        if self.specs.contains_key(&k) {
            return Err(format!("예측기 '{k}'가 이미 등록되어 있습니다"));
        }
        self.specs.insert(k, spec);
        Ok(())
    }

    pub fn unregister(&mut self, id: &str, rev: &str) {
        self.specs.remove(&format!("{id}@{rev}"));
    }

    pub fn remove_by_id(&mut self, id: &str) {
        self.specs.retain(|_, v| v.id != id);
    }

    pub fn specs(&self) -> Vec<&PredictorSpec> {
        self.specs.values().collect()
    }

    pub fn all(&self) -> impl Iterator<Item = &PredictorSpec> {
        self.specs.values()
    }

    pub fn get(&self, id: &str, rev: &str) -> Option<&PredictorSpec> {
        self.specs.get(&format!("{id}@{rev}"))
    }

    pub fn mse_axes(&self) -> Vec<String> {
        self.specs.values().map(PredictorSpec::mse_axis).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_keeps_history_per_model_rev() {
        let mut reg = PredictorRegistry::with_builtins();
        let ttm_a = PredictorSpec {
            id: "ttm".into(),
            rev: "r3-52-16-aaa11111".into(),
            caps: Caps { min_obs: 8, max_horizon: 16, channels: Channels::Independent, exog: Exog::None, statics: false, quantiles: &[0.1, 0.5, 0.9], irregular_time: false, cost: 2 },
        };
        let ttm_b = PredictorSpec { rev: "r3-52-16-bbb22222".into(), ..ttm_a.clone() };
        reg.register(ttm_a.clone()).unwrap();
        assert!(reg.register(ttm_a.clone()).is_err(), "duplicate id@rev must be rejected");
        reg.register(ttm_b.clone()).unwrap();
        assert_ne!(ttm_a.mse_axis(), ttm_b.mse_axis());
        let axes = reg.mse_axes();
        assert!(axes.contains(&ttm_a.mse_axis()));
        assert!(axes.contains(&ttm_b.mse_axis()));
    }

    #[test]
    fn eligible_enforces_cadence_monotone_and_variance_rules() {
        let ttm_caps = Caps {
            min_obs: 8,
            max_horizon: 16,
            channels: Channels::Independent,
            exog: Exog::None,
            statics: false,
            quantiles: &[0.1, 0.5, 0.9],
            irregular_time: false,
            cost: 2,
        };
        let y: Vec<f64> = (0..20).map(|i| 0.5 + 0.05 * (i as f64).sin()).collect();
        let ev_spec = SeriesSpec { kind: "vision.score".into(), group: Group::A, cadence: Cadence::Event, monotone: false, need_horizon: 4 };
        assert!(!eligible(&ev_spec, &y, 0.0, 1e-5, &ttm_caps).eligible);

        let mono_spec = SeriesSpec { kind: "wear.life_used".into(), group: Group::A, cadence: Cadence::Uniform { dt_s: 5.0 }, monotone: true, need_horizon: 4 };
        assert!(!eligible(&mono_spec, &y, 0.0, 1e-5, &ttm_caps).eligible);

        let flat = vec![0.5; 20];
        let uni_spec = SeriesSpec { kind: "bal.margin".into(), group: Group::A, cadence: Cadence::Uniform { dt_s: 0.2 }, monotone: false, need_horizon: 4 };
        assert!(!eligible(&uni_spec, &flat, 0.0, 1e-5, &ttm_caps).eligible);
        assert!(eligible(&uni_spec, &y, 0.0, 1e-5, &ttm_caps).eligible);
    }
}
