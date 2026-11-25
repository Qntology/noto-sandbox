use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use noto_runtime::ts::ttm::{TtmConfig, TtmForecaster};
use noto_runtime::ts::{forecast, TtmPredictor};

struct RandomBackend {
    seed: u64,
}

fn fnv(name: &str, seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl SimpleBackend for RandomBackend {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let dims = s.dims().to_vec();
        let n: usize = dims.iter().product();
        let mut state = fnv(name, self.seed);
        let fan_in = dims.last().copied().unwrap_or(1).max(1) as f32;
        let scale = 1.0 / fan_in.sqrt();
        let norm_weight = dims.len() == 1 && name.ends_with("weight");
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                if norm_weight {
                    1.0 + 0.05 * u
                } else {
                    scale * u
                }
            })
            .collect();
        Tensor::from_vec(vals, s, dev)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        Err(candle_core::Error::Msg(format!("get_unchecked({name}) is not supported by the random backend")))
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn config(extra: serde_json::Value) -> TtmConfig {
    let mut v = serde_json::json!({
        "context_length": 52,
        "prediction_length": 16,
        "patch_length": 4,
        "patch_stride": 4,
        "num_input_channels": 1,
        "d_model": 16,
        "num_layers": 2,
        "expansion_factor": 2,
        "mode": "common_channel",
        "gated_attn": true,
        "norm_mlp": "LayerNorm",
        "scaling": "std",
        "decoder_num_layers": 2,
        "decoder_d_model": 8,
        "decoder_mode": "common_channel",
        "use_decoder": true,
        "adaptive_patching_levels": 0
    });
    if let (Some(base), Some(add)) = (v.as_object_mut(), extra.as_object()) {
        for (k, val) in add {
            base.insert(k.clone(), val.clone());
        }
    }
    serde_json::from_value(v).expect("config parses")
}

fn model(seed: u64, extra: serde_json::Value) -> TtmForecaster {
    let vb = VarBuilder::from_backend(Box::new(RandomBackend { seed }), DType::F32, Device::Cpu);
    TtmForecaster::from_varbuilder(config(extra), vb, std::path::PathBuf::from("random")).expect("model builds")
}

fn series(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.1 * (i as f64 * 0.4).sin() + 0.01 * (i % 3) as f64).collect()
}

#[test]
fn random_weight_model_forecasts_the_configured_horizon_with_finite_values() {
    let m = model(1, serde_json::json!({}));
    assert_eq!((m.context_length(), m.prediction_length(), m.min_history()), (52, 16, 8));
    let out = m.forecast(&series(52)).expect("forecast");
    assert_eq!(out.point.len(), 16);
    assert!(out.point.iter().all(|v| v.is_finite()), "{:?}", out.point);
    let again = m.forecast(&series(52)).unwrap();
    assert_eq!(out.point, again.point);
    let other = model(2, serde_json::json!({})).forecast(&series(52)).unwrap();
    assert_ne!(out.point, other.point);
}

#[test]
fn shorter_histories_are_padded_and_still_finite() {
    let m = model(3, serde_json::json!({}));
    for n in [8, 20, 51, 80] {
        let out = m.forecast(&series(n)).unwrap_or_else(|e| panic!("n={n}: {e}"));
        assert_eq!(out.point.len(), 16, "n={n}");
        assert!(out.point.iter().all(|v| v.is_finite()), "n={n}: {:?}", out.point);
    }
}

#[test]
fn bad_inputs_and_unsupported_configs_are_refused() {
    let m = model(4, serde_json::json!({}));
    assert!(m.forecast(&series(7)).unwrap_err().contains("너무 짧습니다"));
    let mut bad = series(30);
    bad[5] = f64::NAN;
    assert!(m.forecast(&bad).unwrap_err().contains("NaN"));
    assert!(TtmForecaster::validate(&config(serde_json::json!({"self_attn": true}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({"adaptive_patching_levels": 1}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({"mode": "mix_channel"}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({}))).is_ok());
}

#[test]
fn the_fusion_can_use_the_model_as_one_predictor() {
    let m = model(5, serde_json::json!({}));
    let ttm = TtmPredictor(&m);
    let y = series(60);
    let f = forecast(&y, 3, &[&ttm], &[]);
    assert!(f.weights.iter().any(|w| w.method == "ttm"), "{:?} / {:?}", f.weights, f.notes);
    assert!(f.point.iter().all(|v| v.is_finite()));
    let sum: f64 = f.weights.iter().map(|w| w.weight).sum();
    assert!((sum - 1.0).abs() < 1e-9);
    let wt = f.weights.iter().find(|w| w.method == "ttm").unwrap().weight;
    let best_other = f.weights.iter().filter(|w| w.method != "ttm").map(|w| w.weight).fold(0.0, f64::max);
    assert!(best_other >= wt, "ttm {wt} vs best other {best_other}");
    let long = forecast(&y, 20, &[&ttm], &[]);
    assert!(long.weights.iter().all(|w| w.method != "ttm"));
    assert!(long.notes.iter().any(|n| n.contains("ttm")), "{:?}", long.notes);
    assert_eq!(long.point.len(), 20);
}

#[test]
fn a_missing_weight_folder_gives_a_readable_error() {
    let err = TtmForecaster::load(std::path::Path::new("/nonexistent/ttm"), &Device::Cpu).unwrap_err();
    assert!(err.contains("config.json") || err.contains("없습니다"), "{err}");
}

#[test]
fn fusion_notes_why_ttm_sat_out() {
    let m = model(6, serde_json::json!({}));
    let ttm = TtmPredictor(&m);
    for n in 6..=10 {
        let y = series(n);
        let f = forecast(&y, 2, &[&ttm], &[]);
        assert!(f.weights.iter().all(|w| w.method != "ttm"), "n={n}: should not join without history");
        assert!(f.notes.iter().any(|s| s.contains("ttm") && s.contains("분할")), "n={n}: {:?}", f.notes);
        let elig = m.eligibility(n, 2, 0);
        assert!(!elig.joins && elig.reason.is_some(), "n={n}: {elig:?}");
    }
    // With sufficient SDS history (n>=4) and real_points >= min_history (8), history-only weighting lets TTM join
    let hist = [noto_runtime::ts::HistoryMse { method: "ttm".into(), mse: 0.01, n: 4 }];
    let f = forecast(&series(9), 2, &[&ttm], &hist);
    assert!(f.weights.iter().any(|w| w.method == "ttm"), "{:?} / {:?}", f.weights, f.notes);
    assert!(m.eligibility(9, 2, 4).joins);
}

#[test]
fn ttm_fingerprint_tracks_config_and_unverified_weights() {
    let m1 = model(7, serde_json::json!({"prediction_filter_length": 8}));
    let fp = m1.fingerprint();
    assert_eq!(fp.fp8.len(), 8);
    assert!(!fp.verified, "random backend weights are not official R3 weights");
    assert!(fp.warnings.iter().any(|w| w.contains("prediction_filter_length")), "{:?}", fp.warnings);
    let m2 = model(7, serde_json::json!({"d_model": 32}));
    assert_ne!(m1.fp8(), m2.fp8(), "different configs produce different fp8");
}

#[test]
fn padded_tokens_stay_in_distribution() {
    use noto_runtime::ts::ttm::{prepare_scaled, PadMode, TtmInputPolicy};
    // High-offset narrow-variance series (like vision score around 0.85 ± 0.02):
    // under Mean padding, padded tokens sit at 0.0 (mean) instead of -42σ!
    let s: Vec<f64> = (0..12).map(|i| 0.85 + 0.02 * (i as f64 * 0.5).sin()).collect();
    for mode in [PadMode::Mean, PadMode::Edge] {
        let policy = TtmInputPolicy { pad: mode, scale_floor: None, min_real: 8 };
        let (scaled, obs_start, loc, scale) = prepare_scaled(&s, 52, &policy);
        assert_eq!(scaled.len(), 52);
        assert_eq!(obs_start, 40);
        assert!((loc - 0.85).abs() < 0.05);
        assert!(scale > 0.0 && scale < 0.05);
        for (i, &v) in scaled[..obs_start].iter().enumerate() {
            assert!(v.abs() <= 3.0, "mode={mode:?} pad[{i}]={v} out of distribution");
        }
    }
    // Zero padding produces an extreme negative outlier for high-offset narrow series.
    let zero_pol = TtmInputPolicy { pad: PadMode::Zero, scale_floor: None, min_real: 8 };
    let (z_scaled, obs_start, _, _) = prepare_scaled(&s, 52, &zero_pol);
    assert!(z_scaled[0].abs() > 10.0 && obs_start == 40);
}

#[test]
fn ttm_batch_equals_sequential() {
    let m = model(19, serde_json::json!({}));
    let s1: Vec<f64> = (0..18).map(|i| 0.70 + 0.05 * (i as f64 * 0.3).sin()).collect();
    let s2: Vec<f64> = (0..32).map(|i| 10.0 + 0.2 * i as f64).collect();
    let s3: Vec<f64> = (0..52).map(|i| 0.5 + 0.1 * (i as f64 * 0.17).cos()).collect();
    let seq1 = m.forecast(&s1).unwrap();
    let seq2 = m.forecast(&s2).unwrap();
    let seq3 = m.forecast(&s3).unwrap();
    m.clear_cache();
    let batch = m.forecast_batch(&[s1.as_slice(), s2.as_slice(), s3.as_slice()]).unwrap();
    assert_eq!(batch.len(), 3);
    for (a, b) in [(&seq1, &batch[0]), (&seq2, &batch[1]), (&seq3, &batch[2])] {
        assert_eq!(a.point.len(), b.point.len());
        for (x, y) in a.point.iter().zip(&b.point) {
            assert!((x - y).abs() < 1e-4, "batch mismatch: {x} vs {y}");
        }
    }
}

#[test]
fn ttm_cache_hits_on_unchanged_history() {
    let m = model(23, serde_json::json!({}));
    let s: Vec<f64> = (0..20).map(|i| 0.6 + 0.04 * (i as f64 * 0.4).sin()).collect();
    let a = m.forecast(&s).unwrap();
    let b = m.forecast(&s).unwrap();
    assert_eq!(a.point, b.point);
    let stats = m.call_stats();
    assert_eq!(stats.calls, 1, "second call must hit cache: {stats:?}");
    assert_eq!(stats.cache_hits, 1, "expected 1 cache hit: {stats:?}");
    let mut s2 = s.clone();
    s2.push(0.62);
    let _ = m.forecast(&s2).unwrap();
    let stats2 = m.call_stats();
    assert_eq!(stats2.calls, 2);
    assert_eq!(stats2.cache_hits, 1);
}

#[test]
fn ttm_fingerprint_rejects_hash_mismatch() {
    use noto_runtime::ts::ttm::{PadMode, TtmInputPolicy};
    let m1 = model(31, serde_json::json!({}));
    let fp1 = m1.fp8().to_string();
    let m2 = m1.clone().with_policy(TtmInputPolicy { pad: PadMode::Edge, scale_floor: None, min_real: 8 });
    assert_ne!(fp1, m2.fp8(), "changing input policy must change fp8 to isolate SDS MSE");
    assert!(!m2.fingerprint().verified);
}
