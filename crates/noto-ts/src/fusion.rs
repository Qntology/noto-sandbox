use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::holt::holt_damped;
use crate::registry::{Caps, Channels, Exog, SeriesCtx};
use crate::series::RegSeries;

pub const HOLDOUT: usize = 4;
pub const MIN_POINTS: usize = 6;
pub const HISTORY_CAP: f64 = 8.0;
/// Minimum historical error count needed for a predictor to participate via SDS history when
/// live length >= min_history but backtest split < min_history (Phase 4 – Part 1 · D1).
pub const HISTORY_ONLY_MIN_N: u64 = 4;

/// Participation verdict for an extra predictor such as Granite TTM (Phase 4 – Part 1 · D1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtmEligibility {
    pub real_points: usize,
    pub backtest_points: usize,
    pub padded: usize,
    pub dropped_by_align: usize,
    pub joins: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Band {
    pub point: Vec<f64>,
    pub lo: Option<Vec<f64>>,
    pub hi: Option<Vec<f64>>,
    /// Optional full quantile levels and trajectories `(levels, [q_idx][h_idx])` (Phase 4 – Part 1 · D4).
    pub quantiles: Option<(Vec<f64>, Vec<Vec<f64>>)>,
}

pub trait Predictor {
    fn name(&self) -> &str;
    fn rev(&self) -> String {
        "v1".into()
    }
    fn min_history(&self) -> usize;
    fn max_horizon(&self) -> usize {
        usize::MAX
    }
    fn caps(&self) -> Caps {
        Caps {
            min_obs: self.min_history(),
            max_horizon: self.max_horizon(),
            channels: Channels::Univariate,
            exog: Exog::None,
            statics: false,
            quantiles: &[],
            irregular_time: true,
            cost: 0,
        }
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String>;
    fn predict_ctx(&self, ctx: &SeriesCtx<'_>, horizon: usize) -> Result<Band, String> {
        self.predict(ctx.y, horizon)
    }
}

pub struct HoltPredictor;

impl Predictor for HoltPredictor {
    fn name(&self) -> &str {
        "holt"
    }
    fn min_history(&self) -> usize {
        3
    }
    fn caps(&self) -> Caps {
        Caps {
            min_obs: 3,
            max_horizon: usize::MAX,
            channels: Channels::Univariate,
            exog: Exog::None,
            statics: false,
            quantiles: &[0.1, 0.5, 0.9],
            irregular_time: true,
            cost: 1,
        }
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let (point, sd, _) = holt_damped(series, horizon);
        let lo = point.iter().zip(&sd).map(|(p, s)| p - 1.2816 * s).collect();
        let hi = point.iter().zip(&sd).map(|(p, s)| p + 1.2816 * s).collect();
        Ok(Band { point, lo: Some(lo), hi: Some(hi), quantiles: None })
    }
}

pub struct LastPredictor;

impl Predictor for LastPredictor {
    fn name(&self) -> &str {
        "last"
    }
    fn min_history(&self) -> usize {
        1
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let last = *series.last().ok_or("empty series")?;
        Ok(Band { point: vec![last; horizon], lo: None, hi: None, quantiles: None })
    }
}

pub struct EwmaPredictor {
    pub alpha: f64,
}

impl Predictor for EwmaPredictor {
    fn name(&self) -> &str {
        "ewma"
    }
    fn min_history(&self) -> usize {
        2
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let mut it = series.iter();
        let mut level = *it.next().ok_or("empty series")?;
        for v in it {
            level = self.alpha * v + (1.0 - self.alpha) * level;
        }
        Ok(Band { point: vec![level; horizon], lo: None, hi: None, quantiles: None })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryMse {
    pub method: String,
    pub mse: f64,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodWeight {
    pub method: String,
    pub rmse: f64,
    pub history_rmse: Option<f64>,
    pub history_n: u64,
    pub weight: f64,
    /// 1-step point prediction produced by this predictor on the current call (Phase 3 – Part 2 · D1).
    #[serde(default)]
    pub point1: f64,
    /// Multi-step point predictions produced by this predictor (Phase 3 – Part 2 · D5).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<f64>,
    #[serde(default)]
    pub rev: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecast {
    pub point: Vec<f64>,
    pub p10: Vec<f64>,
    pub p90: Vec<f64>,
    pub weights: Vec<MethodWeight>,
    pub rmse_fused: f64,
    pub holdout: usize,
    pub method: String,
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<(Vec<f64>, Vec<Vec<f64>>)>,
}

fn variance(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let m = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (y.len() - 1) as f64
}

pub fn forecast(series: &[f64], horizon: usize, extra: &[&dyn Predictor], history: &[HistoryMse]) -> Forecast {
    forecast_with_prior(series, horizon, extra, history, None)
}

/// Fused forecast accepting an optional prior 1-step residual spread `prior_s0` so short histories
/// (`n = 1..5`) never collapse to zero-width bands (Phase 3 – Part 2 · D6).
pub fn forecast_with_prior(
    series: &[f64],
    horizon: usize,
    extra: &[&dyn Predictor],
    history: &[HistoryMse],
    prior_s0: Option<f64>,
) -> Forecast {
    let horizon = horizon.max(1);
    let mut notes = Vec::new();
    let clean: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.len() != series.len() {
        notes.push(format!("비유한 값 {}개를 제외했습니다", series.len() - clean.len()));
    }
    let n = clean.len();
    if n == 0 {
        return Forecast {
            point: vec![0.0; horizon],
            p10: vec![0.0; horizon],
            p90: vec![0.0; horizon],
            weights: vec![],
            rmse_fused: 0.0,
            holdout: 0,
            method: "none".into(),
            notes: vec!["이력이 없습니다".into()],
            quantiles: None,
        };
    }

    let holt = HoltPredictor;
    let last = LastPredictor;
    let ewma = EwmaPredictor { alpha: 0.3 };
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut pool: Vec<&dyn Predictor> = Vec::with_capacity(3 + extra.len());
    for p in [&last as &dyn Predictor, &ewma, &holt].into_iter().chain(extra.iter().copied()) {
        if seen_names.insert(p.name().to_string()) {
            pool.push(p);
        } else {
            notes.push(format!("{}: 중복 예측기 이름을 건너뛰었습니다", p.name()));
        }
    }

    if n < MIN_POINTS {
        for p in extra {
            notes.push(format!("{}: 이력 {n}점(최소 {}점·백테스트 시작 {MIN_POINTS}점 미만)이라 제외했습니다", p.name(), p.min_history()));
        }
        let sample_var = variance(&clean);
        let (pt, sd, method) = if let Some(s0) = prior_s0.filter(|s| s.is_finite() && *s > 0.0) {
            if n >= 3 {
                let b = ewma.predict(&clean, horizon).unwrap();
                let nu0 = 4.0;
                let df = (n - 1) as f64;
                let sd = ((nu0 * s0 * s0 + df * sample_var) / (nu0 + df)).sqrt().max(1e-4);
                notes.push(format!("이력 {n}점 < {MIN_POINTS}점이라 EWMA + 사전 산포(s0={s0:.3})를 사용했습니다"));
                (b.point[0], sd, "ewma")
            } else {
                notes.push(format!("이력 {n}점 < {MIN_POINTS}점이라 마지막 값 + 사전 산포(s0={s0:.3})를 사용했습니다"));
                (*clean.last().unwrap(), s0.max(sample_var.sqrt()).max(1e-4), "last")
            }
        } else {
            notes.push(format!("이력 {n}점 < {MIN_POINTS}점이라 백테스트 없이 마지막 값을 사용했습니다"));
            (*clean.last().unwrap(), sample_var.sqrt(), "last")
        };
        let widen = |h: usize, sign: f64| pt + sign * 1.2816 * sd * (h as f64).sqrt();
        return Forecast {
            point: vec![pt; horizon],
            p10: (1..=horizon).map(|h| widen(h, -1.0)).collect(),
            p90: (1..=horizon).map(|h| widen(h, 1.0)).collect(),
            weights: vec![],
            rmse_fused: sd,
            holdout: 0,
            method: method.into(),
            notes,
            quantiles: None,
        };
    }

    let hz = HOLDOUT.min(n / 3).max(2);
    let split = n - hz;
    let eps = (0.01 * variance(&clean)).max(1e-12);

    struct Scored {
        idx: usize,
        rmse: f64,
        eff_mse: f64,
        history: Option<(f64, u64)>,
    }
    let mut scored: Vec<Scored> = Vec::new();
    for (idx, p) in pool.iter().enumerate() {
        if hz > p.max_horizon() {
            notes.push(format!("{}: 백테스트 수평선 {hz} > 최대 {}이라 제외했습니다", p.name(), p.max_horizon()));
            continue;
        }
        let hist = history.iter().find(|h| h.method == p.name() && h.n > 0 && h.mse.is_finite());
        if split < p.min_history() {
            if n >= p.min_history() {
                if let Some(h) = hist.filter(|h| h.n >= HISTORY_ONLY_MIN_N) {
                    notes.push(format!("{}: 백테스트 분할 {split}점 < 최소 {}점이나 과거 오차({}건)로 가중 참여", p.name(), p.min_history(), h.n));
                    scored.push(Scored { idx, rmse: h.mse.max(0.0).sqrt(), eff_mse: h.mse.max(0.0), history: Some((h.mse.max(0.0).sqrt(), h.n)) });
                    continue;
                }
            }
            notes.push(format!("{}: 이력 {n}점(백테스트 분할 {split}점이 최소 {}점 미만)이라 제외했습니다", p.name(), p.min_history()));
            continue;
        }

        // Fast predictors (cost < 2) use 1-step rolling-origin holdout over `split..n` (Phase 3 – Part 2 · D1);
        // expensive neural predictors run a single multi-step backtest from `split`.
        let mse_res: Result<f64, String> = if p.caps().cost < 2 {
            let mut sq = 0.0;
            let mut cnt = 0usize;
            for t in split..n {
                if t < p.min_history() {
                    continue;
                }
                match p.predict(&clean[..t], 1) {
                    Ok(b) if !b.point.is_empty() && b.point[0].is_finite() => {
                        sq += (b.point[0] - clean[t]).powi(2);
                        cnt += 1;
                    }
                    Ok(_) => return_err_none(&mut notes, p.name(), "백테스트 출력이 올바르지 않아 제외했습니다"),
                    Err(e) => return_err_none(&mut notes, p.name(), &format!("백테스트 실패({e})")),
                }
            }
            if cnt > 0 {
                Ok(sq / cnt as f64)
            } else {
                Err("rolling holdout empty".into())
            }
        } else {
            match p.predict(&clean[..split], hz) {
                Ok(b) if b.point.len() >= hz && b.point.iter().all(|v| v.is_finite()) => {
                    Ok(b.point.iter().zip(&clean[split..]).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / hz as f64)
                }
                Ok(_) => {
                    notes.push(format!("{}: 백테스트 출력이 올바르지 않아 제외했습니다", p.name()));
                    Err("bad output".into())
                }
                Err(e) => {
                    notes.push(format!("{}: 백테스트 실패({e})", p.name()));
                    Err(e)
                }
            }
        };
        let Ok(mse) = mse_res else { continue };
        let eff_mse = match hist {
            Some(h) => {
                let nh = (h.n as f64).min(HISTORY_CAP);
                (nh * h.mse + mse) / (nh + 1.0)
            }
            None => mse,
        };
        scored.push(Scored { idx, rmse: mse.sqrt(), eff_mse, history: hist.map(|h| (h.mse.max(0.0).sqrt(), h.n)) });
    }
    if scored.is_empty() {
        notes.push("백테스트를 통과한 예측기가 없어 마지막 값을 사용했습니다".into());
        let l = *clean.last().unwrap();
        let sd = variance(&clean).sqrt();
        return Forecast {
            point: vec![l; horizon],
            p10: vec![l - 1.2816 * sd; horizon],
            p90: vec![l + 1.2816 * sd; horizon],
            weights: vec![],
            rmse_fused: sd,
            holdout: hz,
            method: "last".into(),
            notes,
            quantiles: None,
        };
    }

    struct Live {
        w: MethodWeight,
        band: Band,
    }
    let mut live: Vec<Live> = Vec::new();
    for s in &scored {
        let p = pool[s.idx];
        if horizon > p.max_horizon() {
            notes.push(format!("{}: 최대 수평선 {}을 넘어 제외했습니다", p.name(), p.max_horizon()));
            continue;
        }
        match p.predict(&clean, horizon) {
            Ok(b) if b.point.len() >= horizon && b.point.iter().all(|v| v.is_finite()) => {
                let pts: Vec<f64> = b.point.iter().take(horizon).copied().collect();
                let pt1 = pts[0];
                live.push(Live {
                    w: MethodWeight {
                        method: p.name().to_string(),
                        rmse: s.rmse,
                        history_rmse: s.history.map(|h| h.0),
                        history_n: s.history.map_or(0, |h| h.1),
                        weight: 0.0,
                        point1: pt1,
                        points: pts,
                        rev: p.rev(),
                    },
                    band: b,
                });
            }
            Ok(_) => notes.push(format!("{}: 예측 출력이 올바르지 않아 제외했습니다", p.name())),
            Err(e) => notes.push(format!("{}: 예측 실패({e})", p.name())),
        }
    }
    let eff_of = |name: &str| scored.iter().find(|s| pool[s.idx].name() == name).map(|s| s.eff_mse).unwrap_or(f64::INFINITY);
    let inv: Vec<f64> = live.iter().map(|l| 1.0 / (eff_of(&l.w.method) + eps)).collect();
    let total: f64 = inv.iter().sum::<f64>().max(1e-300);
    for (l, w) in live.iter_mut().zip(&inv) {
        l.w.weight = w / total;
    }
    let rmse_fused = live.iter().map(|l| l.w.weight * eff_of(&l.w.method)).sum::<f64>().sqrt();

    let mut point = vec![0.0; horizon];
    for l in &live {
        for (i, v) in l.band.point.iter().enumerate().take(horizon) {
            point[i] += v * l.w.weight;
        }
    }
    let band_src = live
        .iter()
        .filter(|l| l.band.lo.is_some() && l.band.hi.is_some())
        .max_by(|a, b| a.w.weight.total_cmp(&b.w.weight));
    let quantiles = band_src.and_then(|l| l.band.quantiles.clone());
    let mut p10 = vec![0.0; horizon];
    let mut p90 = vec![0.0; horizon];
    for i in 0..horizon {
        let (lo_off, hi_off) = match band_src {
            Some(l) => (
                (l.band.point[i] - l.band.lo.as_ref().unwrap()[i]).max(0.0),
                (l.band.hi.as_ref().unwrap()[i] - l.band.point[i]).max(0.0),
            ),
            None => (0.0, 0.0),
        };
        let sig = 1.2816 * rmse_fused * (((i + 1) as f64) / hz as f64).sqrt() * std::f64::consts::SQRT_2;
        p90[i] = point[i] + (hi_off * hi_off + sig * sig).sqrt();
        p10[i] = point[i] - (lo_off * lo_off + sig * sig).sqrt();
    }
    let label = live.iter().map(|l| format!("{} {:.2}", l.w.method, l.w.weight)).collect::<Vec<_>>().join(", ");
    Forecast {
        point,
        p10,
        p90,
        weights: live.into_iter().map(|l| l.w).collect(),
        rmse_fused,
        holdout: hz,
        method: format!("fusion({label})"),
        notes,
        quantiles,
    }
}

fn return_err_none(notes: &mut Vec<String>, name: &str, msg: &str) {
    notes.push(format!("{name}: {msg}"));
}

/// Forecast from a regularized [`RegSeries`], feeding only its last contiguous valid segment
/// into [`forecast`] so gaps and phase breaks never contaminate predictions (Phase 3 – Part 1 · D1).
/// On a gap-free dense series with zero imputation, output matches [`forecast`] bit-for-bit.
pub fn forecast_reg(reg: &RegSeries, horizon: usize, extra: &[&dyn Predictor], history: &[HistoryMse]) -> Forecast {
    let Some(&(start, end)) = reg.segments.last() else {
        return forecast(&[], horizon, extra, history);
    };
    let slice = &reg.values[start..end];
    let mut out = forecast(slice, horizon, extra, history);
    if reg.segments.len() > 1 || reg.imputed_share > 0.0 {
        out.notes.push(format!(
            "등간격 분절 {}개 중 마지막 구간({}점) 사용 · 보간 비율 {:.0}%",
            reg.segments.len(),
            slice.len(),
            reg.imputed_share * 100.0
        ));
    }
    out
}

/// First crossing time (seconds) where the upper (`p90` if `rising`) or lower (`p10` if `!rising`)
/// forecast quantile reaches `limit`, assuming uniform spacing `dt_s` per step (Phase 3 – Part 3 · D2).
pub fn time_to_cross(f: &Forecast, limit: f64, rising: bool, dt_s: f64) -> Option<f64> {
    if !limit.is_finite() || !dt_s.is_finite() || dt_s <= 0.0 {
        return None;
    }
    let q = if rising { &f.p90 } else { &f.p10 };
    q.iter()
        .position(|v| v.is_finite() && if rising { *v >= limit } else { *v <= limit })
        .map(|k| (k as f64 + 1.0) * dt_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::{q, regularize, ChannelKind, TsSample};

    struct BiasPredictor {
        name: &'static str,
        bias: f64,
    }
    impl Predictor for BiasPredictor {
        fn name(&self) -> &str {
            self.name
        }
        fn min_history(&self) -> usize {
            3
        }
        fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
            let last = *series.last().unwrap_or(&0.0);
            Ok(Band { point: vec![last + self.bias; horizon], lo: None, hi: None, quantiles: None })
        }
    }

    struct FailingPredictor;
    impl Predictor for FailingPredictor {
        fn name(&self) -> &str {
            "broken"
        }
        fn min_history(&self) -> usize {
            2
        }
        fn predict(&self, _: &[f64], _: usize) -> Result<Band, String> {
            Err("boom".into())
        }
    }

    #[test]
    fn fusion_favours_the_better_predictor_on_a_ramp() {
        let ramp: Vec<f64> = (0..25).map(|i| 1.0 + 0.4 * i as f64).collect();
        let f = forecast(&ramp, 4, &[], &[]);
        let w_holt = f.weights.iter().find(|w| w.method == "holt").unwrap().weight;
        let w_last = f.weights.iter().find(|w| w.method == "last").unwrap().weight;
        assert!(w_holt > w_last, "holt {w_holt} should beat last {w_last}");
        let sum: f64 = f.weights.iter().map(|w| w.weight).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(f.point[0] > 10.6 && f.point[3] > f.point[0]);
        assert!(f.weights.iter().all(|w| w.point1.is_finite() && w.points.len() == 4));
    }

    #[test]
    fn fusion_handles_short_and_noisy_histories_safely() {
        let empty = forecast(&[], 3, &[], &[]);
        assert_eq!(empty.point.len(), 3);
        let short = forecast(&[0.5, 0.52, 0.51], 4, &[], &[]);
        assert_eq!(short.method, "last");
        assert!(short.p10.iter().zip(&short.p90).all(|(a, b)| a <= b));
        let single_with_prior = forecast_with_prior(&[0.5], 3, &[], &[], Some(0.05));
        assert!(single_with_prior.p90[0] > single_with_prior.p10[0], "prior_s0 must prevent zero-width bands at n=1");
        let with_nan = forecast(&[1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2, &[], &[]);
        assert!(with_nan.point.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extra_predictor_joins_and_failures_are_skipped_with_notes() {
        let series: Vec<f64> = (0..20).map(|i| 0.5 + 0.01 * (i % 3) as f64).collect();
        let good = BiasPredictor { name: "good", bias: 0.0 };
        let bad = BiasPredictor { name: "bad", bias: 2.0 };
        let fail = FailingPredictor;
        let f = forecast(&series, 3, &[&good, &bad, &fail], &[]);
        let wg = f.weights.iter().find(|w| w.method == "good").unwrap().weight;
        let wb = f.weights.iter().find(|w| w.method == "bad").unwrap().weight;
        assert!(wg > wb * 10.0, "good={wg}, bad={wb}");
        assert!(f.notes.iter().any(|n| n.contains("broken")));
    }

    #[test]
    fn past_errors_steer_weights_when_the_current_holdout_is_small() {
        let series: Vec<f64> = (0..12).map(|_| 0.8).collect();
        let a = BiasPredictor { name: "a", bias: 0.0 };
        let b = BiasPredictor { name: "b", bias: 0.0 };
        let hist = [
            HistoryMse { method: "a".into(), mse: 0.0001, n: 20 },
            HistoryMse { method: "b".into(), mse: 0.25, n: 20 },
        ];
        let f = forecast(&series, 2, &[&a, &b], &hist);
        let wa = f.weights.iter().find(|w| w.method == "a").unwrap().weight;
        let wb = f.weights.iter().find(|w| w.method == "b").unwrap().weight;
        assert!(wa > wb * 5.0, "a={wa}, b={wb}");
    }

    #[test]
    fn weights_follow_who_predicts_the_next_value() {
        // Two predictors look identical on holdout (`bias = 0`), but realized 1-step errors show "b" is biased.
        let series: Vec<f64> = (0..16).map(|i| 0.6 + 0.01 * ((i % 3) as f64 - 1.0)).collect();
        let a = BiasPredictor { name: "a", bias: 0.0 };
        let b = BiasPredictor { name: "b", bias: 0.0 };
        let mut book = crate::calib::ErrBook::default();
        for _ in 0..30 {
            let f = forecast(&series, 1, &[&a, &b], &book.realized());
            // Realize `y` matching "a" closely while shifting "b"'s simulated prediction by +0.15.
            let mut ws = f.weights.clone();
            for w in &mut ws {
                if w.method == "b" {
                    w.point1 += 0.15;
                }
            }
            book.update(&ws, series[15], 0.01, 0.03);
        }
        let f_after = forecast(&series, 1, &[&a, &b], &book.realized());
        let wa = f_after.weights.iter().find(|w| w.method == "a").unwrap().weight;
        let wb = f_after.weights.iter().find(|w| w.method == "b").unwrap().weight;
        assert!(wb * 3.0 <= wa, "biased predictor b ({wb:.4}) must drop below 1/3 of a ({wa:.4})");
    }

    #[test]
    fn history_counts_at_most_the_cap_against_the_new_run() {
        let series: Vec<f64> = (0..16).map(|i| 0.5 + 0.02 * (i % 2) as f64).collect();
        let a = BiasPredictor { name: "a", bias: 0.0 };
        let b = BiasPredictor { name: "b", bias: 0.5 };
        let h_small = [HistoryMse { method: "b".into(), mse: 0.0, n: 8 }];
        let h_huge = [HistoryMse { method: "b".into(), mse: 0.0, n: 100_000 }];
        let f_small = forecast(&series, 2, &[&a, &b], &h_small);
        let f_huge = forecast(&series, 2, &[&a, &b], &h_huge);
        let wb_s = f_small.weights.iter().find(|w| w.method == "b").unwrap().weight;
        let wb_h = f_huge.weights.iter().find(|w| w.method == "b").unwrap().weight;
        assert!((wb_s - wb_h).abs() < 1e-12, "cap must hold: {wb_s} vs {wb_h}");
    }

    #[test]
    fn forecast_reg_matches_forecast_on_dense_series() {
        let vals: Vec<f64> = (0..24).map(|i| 0.65 + 0.02 * (i as f64 * 0.3).sin()).collect();
        let samples: Vec<TsSample> = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| TsSample { dtick: (i as u32) * 2, value: v as f32, sub_ms: 0, q: q::OBSERVED })
            .collect();
        let reg = regularize(&samples, ChannelKind::Gauge, 0, 2, vals.len(), 6, &[]);
        let f_raw = forecast(&reg.values, 4, &[], &[]);
        let f_reg = forecast_reg(&reg, 4, &[], &[]);
        assert_eq!(f_raw.point, f_reg.point);
        assert_eq!(f_raw.p10, f_reg.p10);
        assert_eq!(f_raw.p90, f_reg.p90);
        assert_eq!(f_raw.method, f_reg.method);
        assert_eq!(f_raw.notes, f_reg.notes);
    }
}
