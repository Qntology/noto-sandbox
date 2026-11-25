use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Padding strategy for histories shorter than `context_length` (Phase 4 – Part 1 · D2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PadMode {
    Zero,
    #[default]
    Mean,
    Edge,
}

impl PadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PadMode::Zero => "zero",
            PadMode::Mean => "mean",
            PadMode::Edge => "edge",
        }
    }
}

/// Unified input normalization and padding policy (Phase 4 – Part 1 · D2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TtmInputPolicy {
    pub pad: PadMode,
    pub scale_floor: Option<f64>,
    pub min_real: usize,
}

impl Default for TtmInputPolicy {
    fn default() -> Self {
        TtmInputPolicy {
            pad: PadMode::Mean,
            scale_floor: None,
            min_real: 8,
        }
    }
}

impl TtmInputPolicy {
    pub fn tag(&self) -> String {
        match self.scale_floor {
            Some(sf) => format!("{}:min{}:sf{:.1e}", self.pad.as_str(), self.min_real, sf),
            None => format!("{}:min{}", self.pad.as_str(), self.min_real),
        }
    }
}

/// Runtime call telemetry for a loaded [`TtmForecaster`] (Phase 4 – Part 1 · D5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TtmCallStats {
    pub calls: u64,
    pub cache_hits: u64,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub device: String,
}

/// Preferred device override from `NOTO_TTM_DEVICE` (`"cpu"` or `"shared"`).
pub fn ttm_device_pref() -> &'static str {
    match std::env::var("NOTO_TTM_DEVICE").ok().as_deref().map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("cpu") => "cpu",
        Some("shared") => "shared",
        _ => "shared",
    }
}

/// Parity probe comparing a target device against CPU on 3 fixed synthetic series (Phase 4 – Part 1 · D5).
/// Returns max absolute deviation (`Ok(max_diff)`) or an error message if it exceeds `tol`.
pub fn ttm_parity_check(dir: &Path, device: &Device, tol: f64) -> std::result::Result<f64, String> {
    if matches!(device, Device::Cpu) {
        return Ok(0.0);
    }
    let cpu = TtmForecaster::load(dir, &Device::Cpu)?;
    let dev = TtmForecaster::load(dir, device)?;
    let probes: [Vec<f64>; 3] = [
        (0..52).map(|i| 0.6 + 0.08 * (i as f64 * 0.25).sin()).collect(),
        (0..20).map(|i| 0.3 + 0.015 * i as f64).collect(),
        (0..12).map(|i| 0.75 + 0.02 * ((i % 3) as f64 - 1.0)).collect(),
    ];
    let mut max_diff = 0.0f64;
    for s in &probes {
        let a = cpu.forecast(s)?;
        let b = dev.forecast(s)?;
        for (x, y) in a.point.iter().zip(&b.point) {
            if !y.is_finite() {
                return Err("TTM GPU 출력에 비유한 값이 포함됨".into());
            }
            max_diff = max_diff.max((x - y).abs());
        }
    }
    if max_diff > tol {
        Err(format!("TTM 장치 대조 오차 {max_diff:.2e} > 허용치 {tol:.2e}"))
    } else {
        Ok(max_diff)
    }
}

/// Official Granite TTM-R3 (52-16) revision and file hashes (matching `src-tauri/src/models.rs`).
pub const OFFICIAL_REVISION: &str = "7d99b7e35b3dc18f9db3b2b73cc9699dea6c9508";
pub const OFFICIAL_CONFIG_SHA256: &str = "2c69af1a0ba3b3582b5a84adb309c3eb0ed24dec204576fcf23d7894c3a10f07";
pub const OFFICIAL_WEIGHTS_SHA256: &str = "710403c80572c022702ae6a2fad069c309c22b04ea96f17838f0ce7c8e4bba9c";

/// Cryptographic identity & config audit of a loaded TTM model (Phase 4 – Part 1 · D6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtmFingerprint {
    pub revision: String,
    pub config_sha256: String,
    pub weights_sha256: String,
    pub fp8: String,
    pub context: usize,
    pub prediction: usize,
    pub quantile_levels: Vec<f64>,
    pub input_policy: String,
    pub verified: bool,
    pub warnings: Vec<String>,
}

fn sha256_file(path: &Path) -> std::result::Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn sha256_files(paths: &[PathBuf]) -> std::result::Result<String, String> {
    if paths.len() == 1 {
        return sha256_file(&paths[0]);
    }
    let mut h = Sha256::new();
    for p in paths {
        let fh = sha256_file(p)?;
        h.update(p.file_name().and_then(|n| n.to_str()).unwrap_or("").as_bytes());
        h.update(b":");
        h.update(fh.as_bytes());
        h.update(b"\n");
    }
    Ok(format!("{:x}", h.finalize()))
}

fn compute_fp8(config_sha: &str, weights_sha: &str, input_policy: &str) -> String {
    let mut h = Sha256::new();
    h.update(config_sha.as_bytes());
    h.update(b"|");
    h.update(weights_sha.as_bytes());
    h.update(b"|");
    h.update(input_policy.as_bytes());
    let full = format!("{:x}", h.finalize());
    full.chars().take(8).collect()
}

fn config_warnings(cfg: &TtmConfig) -> Vec<String> {
    let mut w = Vec::new();
    if let Some(pfl) = cfg.prediction_filter_length {
        if pfl != cfg.prediction_length {
            w.push(format!("TTM config: prediction_filter_length={pfl} 는 현재 무시됩니다 (prediction_length={})", cfg.prediction_length));
        }
    }
    if cfg.patch_length > 0 && cfg.context_length % cfg.patch_length != 0 {
        let dropped = cfg.context_length % cfg.patch_length;
        w.push(format!("TTM config: context_length={} 이 patch_length={} 의 배수가 아니어서 선두 {dropped}점이 버려집니다", cfg.context_length, cfg.patch_length));
    }
    w
}

fn d1() -> usize {
    1
}
fn d2() -> usize {
    2
}
fn d3() -> usize {
    3
}
fn d4() -> usize {
    4
}
fn d8() -> usize {
    8
}
fn d16() -> usize {
    16
}
fn d64() -> usize {
    64
}
fn dtrue() -> bool {
    true
}
fn d_eps() -> f64 {
    1e-5
}
fn d_mq_eps() -> f64 {
    1e-6
}
fn s_common() -> String {
    "common_channel".into()
}
fn s_layernorm() -> String {
    "LayerNorm".into()
}
fn s_median() -> String {
    "median".into()
}
fn s_pool() -> String {
    "pool".into()
}
fn s_add() -> String {
    "add".into()
}
fn s_sincos() -> String {
    "sincos".into()
}
fn s_std() -> serde_json::Value {
    serde_json::Value::String("std".into())
}
fn default_q() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtmConfig {
    #[serde(default = "d64")]
    pub context_length: usize,
    #[serde(default = "d8")]
    pub patch_length: usize,
    #[serde(default = "d8")]
    pub patch_stride: usize,
    #[serde(default = "d1")]
    pub num_input_channels: usize,
    #[serde(default = "d16")]
    pub prediction_length: usize,
    #[serde(default = "d16")]
    pub d_model: usize,
    #[serde(default = "d2")]
    pub expansion_factor: usize,
    #[serde(default = "d3")]
    pub num_layers: usize,
    #[serde(default = "s_common")]
    pub mode: String,
    #[serde(default = "dtrue")]
    pub gated_attn: bool,
    #[serde(default = "s_layernorm")]
    pub norm_mlp: String,
    #[serde(default)]
    pub self_attn: bool,
    #[serde(default)]
    pub use_positional_encoding: bool,
    #[serde(default = "s_sincos")]
    pub positional_encoding_type: String,
    #[serde(default = "s_std")]
    pub scaling: serde_json::Value,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub adaptive_patching_levels: usize,
    #[serde(default)]
    pub resolution_prefix_tuning: bool,
    #[serde(default = "d8")]
    pub decoder_num_layers: usize,
    #[serde(default = "d8")]
    pub decoder_d_model: usize,
    #[serde(default)]
    pub decoder_adaptive_patching_levels: usize,
    #[serde(default)]
    pub decoder_raw_residual: bool,
    #[serde(default = "s_common")]
    pub decoder_mode: String,
    #[serde(default = "dtrue")]
    pub use_decoder: bool,
    #[serde(default)]
    pub enable_forecast_channel_mixing: bool,
    #[serde(default)]
    pub categorical_vocab_size_list: Option<Vec<usize>>,
    #[serde(default)]
    pub prediction_filter_length: Option<usize>,
    #[serde(default)]
    pub multi_scale: bool,
    #[serde(default)]
    pub register_tokens: usize,
    #[serde(default)]
    pub fft_length: usize,
    #[serde(default = "dtrue")]
    pub use_fft_embedding: bool,
    #[serde(default)]
    pub get_one_freq_emb: bool,
    #[serde(default)]
    pub fft_ignore_dc: bool,
    #[serde(default = "d4")]
    pub fft_d_freq_min: usize,
    #[serde(default = "d64")]
    pub fft_d_freq_max: usize,
    #[serde(default)]
    pub multi_quantile_head: bool,
    #[serde(default)]
    pub residual_context_length: Option<usize>,
    #[serde(default)]
    pub trend_patch_length: Option<usize>,
    #[serde(default)]
    pub trend_patch_stride: Option<usize>,
    #[serde(default)]
    pub trend_d_model: Option<usize>,
    #[serde(default)]
    pub trend_decoder_d_model: Option<usize>,
    #[serde(default)]
    pub trend_num_layers: Option<usize>,
    #[serde(default)]
    pub trend_decoder_num_layers: Option<usize>,
    #[serde(default)]
    pub trend_register_tokens: Option<usize>,
    #[serde(default)]
    pub trend_fft_length: Option<usize>,
    #[serde(default)]
    pub trend_multi_scale: Option<bool>,
    #[serde(default)]
    pub trend_adaptive_patching_levels: Option<usize>,
    #[serde(default)]
    pub decompose: bool,
    #[serde(default = "d8")]
    pub mq_hidden: usize,
    #[serde(default = "d3")]
    pub mq_kernel_size: usize,
    #[serde(default = "d_mq_eps")]
    pub mq_eps: f64,
    #[serde(default)]
    pub mq_use_decoder_pool: bool,
    #[serde(default = "s_median")]
    pub mq_q50_type: String,
    #[serde(default = "s_pool")]
    pub mq_cond_path: String,
    #[serde(default = "s_add")]
    pub mq_cond_mode: String,
    #[serde(default = "d8")]
    pub mq_decoder_d_model: usize,
    #[serde(default)]
    pub combine_quantiles_via_variance: bool,
    #[serde(default = "default_q")]
    pub quantile_levels: Vec<f64>,
    #[serde(default)]
    pub enable_base_norm_always: bool,
    #[serde(default)]
    pub gate_mode: Option<String>,
    #[serde(default)]
    pub gate_groups: Option<usize>,
    #[serde(default)]
    pub use_register_context_gating: bool,
    #[serde(default)]
    pub num_patches: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Scaling {
    Std,
    Mean,
    Nop,
}

impl Scaling {
    fn from_value(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::String(s) if s == "std" => Scaling::Std,
            serde_json::Value::String(s) if s == "mean" => Scaling::Mean,
            serde_json::Value::Bool(true) => Scaling::Std,
            _ => Scaling::Nop,
        }
    }

    /// Scale the observed slice first, then pad the prefix according to `policy` so padded tokens
    /// never become extreme out-of-distribution values (Phase 4 – Part 1 · D2).
    /// Returns `(scaled_ctx, obs_start, loc, scale)`.
    fn prepare_input(&self, series: &[f64], sl: usize, policy: &TtmInputPolicy) -> (Vec<f64>, usize, f64, f64) {
        let obs_slice = if series.len() >= sl {
            &series[series.len() - sl..]
        } else {
            series
        };
        let obs_start = sl.saturating_sub(obs_slice.len());
        let var_floor = policy.scale_floor.unwrap_or(1e-5).max(1e-12);
        let (loc, scale) = match self {
            Scaling::Nop => (0.0, 1.0),
            Scaling::Std => {
                let cnt = obs_slice.len().max(1) as f64;
                let loc = obs_slice.iter().sum::<f64>() / cnt;
                let var = obs_slice.iter().map(|x| (x - loc) * (x - loc)).sum::<f64>() / cnt;
                let scale = (var + var_floor).sqrt();
                (loc, scale)
            }
            Scaling::Mean => {
                let cnt = obs_slice.len().max(1) as f64;
                let s: f64 = obs_slice.iter().map(|x| x.abs()).sum();
                let scale = (s / cnt).max(var_floor.sqrt().min(1e-10));
                (0.0, scale)
            }
        };
        let scaled_obs: Vec<f64> = obs_slice.iter().map(|&x| (x - loc) / scale).collect();
        let pad_val = match policy.pad {
            PadMode::Mean => 0.0,
            PadMode::Edge => scaled_obs.first().copied().unwrap_or(0.0),
            PadMode::Zero => (0.0 - loc) / scale,
        };
        let mut out = vec![pad_val; sl];
        out[obs_start..].copy_from_slice(&scaled_obs);
        (out, obs_start, loc, scale)
    }
}

/// Public helper exposing prepared `(scaled_context, obs_start, loc, scale)` for tests & diagnostics.
pub fn prepare_scaled(series: &[f64], context_length: usize, policy: &TtmInputPolicy) -> (Vec<f64>, usize, f64, f64) {
    Scaling::Std.prepare_input(series, context_length.max(1), policy)
}

/// Top-`k` DFT frequency indices computed strictly over the observed slice so frequency selection
/// is completely invariant to padding length (Phase 4 – Part 1 · D2).
pub fn fft_top_indices(obs: &[f64], k: usize, max_bins: usize, ignore_dc: bool) -> Vec<usize> {
    fft_top_indices_lut(obs, k, max_bins, ignore_dc, None)
}

/// Same as [`fft_top_indices`], using an L1-resident cos/sin twiddle LUT of length `obs.len()` when given
/// (valid only for unpadded full-context slices; results are identical to the trig path).
pub fn fft_top_indices_lut(obs: &[f64], k: usize, max_bins: usize, ignore_dc: bool, lut: Option<(&[f64], &[f64])>) -> Vec<usize> {
    let n = obs.len().max(1);
    let lut = lut.filter(|(c, s)| c.len() == n && s.len() == n);
    let bins = (n / 2 + 1).min(max_bins.max(1));
    let mut mag: Vec<(usize, f64)> = (0..bins)
        .map(|kf| {
            if ignore_dc && kf == 0 {
                return (0, 0.0);
            }
            let mut re = 0.0;
            let mut im = 0.0;
            if let Some((cos_lut, sin_lut)) = lut {
                let step = kf % n;
                let mut idx = 0usize;
                for &v in obs {
                    re += v * cos_lut[idx];
                    im += v * sin_lut[idx];
                    idx += step;
                    if idx >= n {
                        idx -= n;
                    }
                }
            } else {
                for (tt, &v) in obs.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (kf * tt) as f64 / n as f64;
                    re += v * a.cos();
                    im += v * a.sin();
                }
            }
            (kf, (re * re + im * im).sqrt())
        })
        .collect();
    mag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    let mut out: Vec<usize> = mag.iter().take(k).map(|(i, _)| *i).collect();
    while out.len() < k {
        out.push(0);
    }
    out
}

#[derive(Debug, Clone)]
struct SubConfig {
    context_length: usize,
    patch_length: usize,
    d_model: usize,
    decoder_d_model: usize,
    num_layers: usize,
    decoder_num_layers: usize,
    register_tokens: usize,
    fft_length: usize,
    multi_scale: bool,
    scaling: Scaling,
    num_patches: usize,
    n_base_patches: usize,
}

fn multiscale_patches(context_length: usize, patch_length: usize) -> usize {
    let mut total = 0;
    let mut i = 0u32;
    loop {
        let factor = 1usize << i;
        let ds = context_length / factor;
        if ds < patch_length {
            break;
        }
        total += ds / patch_length;
        i += 1;
    }
    total
}

impl SubConfig {
    fn derive(cfg: &TtmConfig, which: &str) -> std::result::Result<Self, String> {
        let mut s = SubConfig {
            context_length: cfg.context_length,
            patch_length: cfg.patch_length,
            d_model: cfg.d_model,
            decoder_d_model: cfg.decoder_d_model,
            num_layers: cfg.num_layers,
            decoder_num_layers: cfg.decoder_num_layers,
            register_tokens: cfg.register_tokens,
            fft_length: cfg.fft_length,
            multi_scale: cfg.multi_scale,
            scaling: Scaling::from_value(&cfg.scaling),
            num_patches: 0,
            n_base_patches: 0,
        };
        let mut stride = cfg.patch_stride;
        if which == "trend" {
            if let Some(v) = cfg.trend_patch_length {
                s.patch_length = v;
            }
            if let Some(v) = cfg.trend_patch_stride {
                stride = v;
            }
            if let Some(v) = cfg.trend_d_model {
                s.d_model = v;
            }
            if let Some(v) = cfg.trend_decoder_d_model {
                s.decoder_d_model = v;
            }
            if let Some(v) = cfg.trend_num_layers {
                s.num_layers = v;
            }
            if let Some(v) = cfg.trend_decoder_num_layers {
                s.decoder_num_layers = v;
            }
            if let Some(v) = cfg.trend_register_tokens {
                s.register_tokens = v;
            }
            if let Some(v) = cfg.trend_fft_length {
                s.fft_length = v;
            }
            if let Some(v) = cfg.trend_multi_scale {
                s.multi_scale = v;
            }
            if cfg.trend_adaptive_patching_levels.unwrap_or(0) > 0 {
                return Err("TTM: trend_adaptive_patching_levels > 0 은 지원하지 않습니다".into());
            }
            s.scaling = Scaling::Nop;
        } else if which == "residual" {
            if let Some(v) = cfg.residual_context_length {
                s.context_length = v;
            }
            s.scaling = Scaling::Nop;
        }
        if s.patch_length != stride {
            return Err("TTM: patch_length 와 patch_stride 가 달라 지원하지 않습니다".into());
        }
        if s.context_length <= s.patch_length {
            return Err("TTM: context_length 가 patch_length 이하입니다".into());
        }
        s.n_base_patches = (s.context_length - s.patch_length) / s.patch_length + 1;
        let mut np = if s.multi_scale {
            multiscale_patches(s.context_length, s.patch_length)
        } else {
            s.n_base_patches
        };
        np += s.register_tokens;
        if s.fft_length > 0 {
            np += if cfg.get_one_freq_emb { 1 } else { s.fft_length };
        }
        s.num_patches = match (which, cfg.num_patches) {
            ("trend", _) => np,
            (_, Some(given)) => given,
            _ => np,
        };
        Ok(s)
    }
}

#[derive(Debug, Clone)]
enum GateKind {
    Softmax,
    Sigmoid,
    Glu,
    GroupSigmoid(usize),
}

#[derive(Debug, Clone)]
struct Gate {
    layer: Linear,
    kind: GateKind,
    size: usize,
}

impl Gate {
    fn load(cfg: &TtmConfig, size: usize, vb: VarBuilder) -> Result<Self> {
        let mode = cfg.gate_mode.clone().unwrap_or_else(|| "softmax".into());
        let kind = match mode.as_str() {
            "sigmoid" => GateKind::Sigmoid,
            "glu" => GateKind::Glu,
            "group_sigmoid" => {
                let req = cfg.gate_groups.unwrap_or(8);
                let mut g = 1;
                if req > 1 {
                    if size % req == 0 {
                        g = req;
                    } else {
                        for cand in (1..=req.min(size)).rev() {
                            if size % cand == 0 {
                                g = cand;
                                break;
                            }
                        }
                    }
                }
                GateKind::GroupSigmoid(g)
            }
            _ => GateKind::Softmax,
        };
        let out = if matches!(kind, GateKind::Glu) { 2 * size } else { size };
        Ok(Self {
            layer: candle_nn::linear(size, out, vb.pp("attn_layer"))?,
            kind,
            size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.apply(&self.layer)?;
        match &self.kind {
            GateKind::Softmax => x * candle_nn::ops::softmax_last_dim(&logits)?,
            GateKind::Sigmoid => x * candle_nn::ops::sigmoid(&logits)?,
            GateKind::Glu => {
                let parts = logits.chunk(2, D::Minus1)?;
                &parts[0] * candle_nn::ops::sigmoid(&parts[1])?
            }
            GateKind::GroupSigmoid(g) => {
                let dims = logits.dims().to_vec();
                let mut shape = dims.clone();
                let last = shape.pop().unwrap_or(self.size);
                shape.push(*g);
                shape.push(last / g);
                let grouped = logits.reshape(shape.clone())?.mean_keepdim(D::Minus1)?;
                let gate = candle_nn::ops::sigmoid(&grouped)?.broadcast_as(shape)?.reshape(dims)?;
                x * gate
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Norm {
    Layer(LayerNorm),
    /// Pre-folded inference BatchNorm: `scale = w / sqrt(var + eps)`, `bias = b - mean * scale`.
    Batch { scale: Tensor, bias: Tensor },
}

impl Norm {
    fn load(cfg: &TtmConfig, d: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.norm_mlp.to_lowercase().contains("batch") {
            let bv = vb.pp("norm").pp("batchnorm");
            let w = bv.get(d, "weight")?;
            let b = bv.get(d, "bias")?;
            let mean = bv.get(d, "running_mean")?;
            let var = bv.get(d, "running_var")?;
            let scale = w.broadcast_div(&(var + cfg.norm_eps)?.sqrt()?)?;
            let bias = b.broadcast_sub(&mean.broadcast_mul(&scale)?)?;
            Ok(Norm::Batch { scale, bias })
        } else {
            Ok(Norm::Layer(candle_nn::layer_norm(d, cfg.norm_eps, vb.pp("norm"))?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Norm::Layer(l) => x.apply(l),
            Norm::Batch { scale, bias } => x.broadcast_mul(scale)?.broadcast_add(bias),
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(i: usize, o: usize, expansion: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(i, i * expansion, vb.pp("fc1"))?,
            fc2: candle_nn::linear(i * expansion, o, vb.pp("fc2"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct MixerLayer {
    patch_norm: Option<Norm>,
    patch_mlp: Option<Mlp>,
    patch_gate: Option<Gate>,
    feat_norm: Norm,
    feat_mlp: Mlp,
    feat_gate: Option<Gate>,
}

impl MixerLayer {
    fn load(cfg: &TtmConfig, num_patches: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let (patch_norm, patch_mlp, patch_gate) = if num_patches > 1 {
            let pv = vb.pp("patch_mixer");
            (
                Some(Norm::load(cfg, d, pv.pp("norm"))?),
                Some(Mlp::load(num_patches, num_patches, cfg.expansion_factor, pv.pp("mlp"))?),
                if cfg.gated_attn {
                    Some(Gate::load(cfg, num_patches, pv.pp("gating_block"))?)
                } else {
                    None
                },
            )
        } else {
            (None, None, None)
        };
        let fv = vb.pp("feature_mixer");
        Ok(Self {
            patch_norm,
            patch_mlp,
            patch_gate,
            feat_norm: Norm::load(cfg, d, fv.pp("norm"))?,
            feat_mlp: Mlp::load(d, d, cfg.expansion_factor, fv.pp("mlp"))?,
            feat_gate: if cfg.gated_attn {
                Some(Gate::load(cfg, d, fv.pp("gating_block"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        if let (Some(norm), Some(mlp)) = (&self.patch_norm, &self.patch_mlp) {
            let r = h.clone();
            let mut y = norm.forward(&h)?.transpose(2, 3)?.contiguous()?;
            y = y.apply(mlp)?;
            if let Some(g) = &self.patch_gate {
                y = g.forward(&y)?;
            }
            h = (y.transpose(2, 3)? + r)?;
        }
        let r = h.clone();
        let mut y = self.feat_norm.forward(&h)?.apply(&self.feat_mlp)?;
        if let Some(g) = &self.feat_gate {
            y = g.forward(&y)?;
        }
        y + r
    }
}

#[derive(Debug, Clone)]
struct Block {
    layers: Vec<MixerLayer>,
}

impl Block {
    fn load(cfg: &TtmConfig, n: usize, num_patches: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(MixerLayer::load(cfg, num_patches, d, vb.pp("mixers").pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for l in self.layers.iter() {
            h = l.forward(&h)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
enum Patcher {
    Single(Linear),
    Multi(Vec<Linear>),
}

#[derive(Debug, Clone)]
enum FftEmbed {
    Table(Tensor),
    Mlp(Linear, Linear),
}

#[derive(Debug, Clone)]
struct FftTokens {
    k: usize,
    max_bins: usize,
    context_length: usize,
    cos_lut: Vec<f64>,
    sin_lut: Vec<f64>,
    embed: FftEmbed,
    one: Option<(Linear, usize)>,
    ignore_dc: bool,
}

#[derive(Debug, Clone)]
struct CpuLinear {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

impl CpuLinear {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((out_dim, in_dim), "weight")?.to_vec2::<f32>()?;
        let b = vb.get(out_dim, "bias")?.to_vec1::<f32>()?;
        Ok(Self { w, b })
    }

    fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        self.w
            .iter()
            .zip(&self.b)
            .map(|(row, &bias)| {
                let mut acc = bias;
                for (&w, &v) in row.iter().zip(x.iter()) {
                    acc += w * v;
                }
                acc
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct MqHead {
    dw_w: Vec<f32>,
    dw_b: f32,
    in_w: f32,
    in_b: f32,
    pw1_w: Vec<f32>,
    pw1_b: Vec<f32>,
    pool: Option<CpuLinear>,
    flat: Option<(CpuLinear, CpuLinear)>,
    pw2_w: Vec<Vec<f32>>,
    pw2_b: Vec<f32>,
    hidden: usize,
    concat: bool,
    use_cond: bool,
    q50_median: bool,
    k_side: usize,
    eps: f64,
    num_q: usize,
}

impl MqHead {
    fn load(cfg: &TtmConfig, decoder_d_model: usize, num_patches: usize, vb: VarBuilder) -> Result<Self> {
        let q = cfg.quantile_levels.len();
        let h = cfg.mq_hidden;
        let ks = cfg.mq_kernel_size;
        let use_cond = cfg.mq_use_decoder_pool;
        let concat = use_cond && cfg.mq_cond_mode == "concat";
        let pw2_in = if concat { 2 * h } else { h };
        let dw_w = vb.get((1, 1, ks), "dw.weight")?.flatten_all()?.to_vec1::<f32>()?;
        let dw_b = vb.get(1, "dw.bias")?.to_vec1::<f32>()?[0];
        let in_w = vb.get(1, "inorm.weight")?.to_vec1::<f32>()?[0];
        let in_b = vb.get(1, "inorm.bias")?.to_vec1::<f32>()?[0];
        let pw1_w = vb.get((h, 1, 1), "pw1.weight")?.flatten_all()?.to_vec1::<f32>()?;
        let pw1_b = vb.get(h, "pw1.bias")?.to_vec1::<f32>()?;
        let pw2_w = vb.get((q, pw2_in, 1), "pw2.weight")?.reshape((q, pw2_in))?.to_vec2::<f32>()?;
        let pw2_b = vb.get(q, "pw2.bias")?.to_vec1::<f32>()?;
        let (pool, flat) = if use_cond {
            if cfg.mq_cond_path == "pool" {
                (Some(CpuLinear::load(decoder_d_model, h, vb.pp("dec_proj"))?), None)
            } else {
                let md = cfg.mq_decoder_d_model;
                (
                    None,
                    Some((
                        CpuLinear::load(decoder_d_model, md, vb.pp("dec_tok_proj"))?,
                        CpuLinear::load(num_patches * md, h, vb.pp("dec_flat_to_hidden"))?,
                    )),
                )
            }
        } else {
            (None, None)
        };
        Ok(Self {
            dw_w,
            dw_b,
            in_w,
            in_b,
            pw1_w,
            pw1_b,
            pool,
            flat,
            pw2_w,
            pw2_b,
            hidden: h,
            concat,
            use_cond,
            q50_median: cfg.mq_q50_type == "median",
            k_side: q / 2,
            eps: cfg.mq_eps,
            num_q: q,
        })
    }

    fn forward(&self, mean_hat: &[f64], dec: &Tensor) -> Result<Vec<Vec<f64>>> {
        let t = mean_hat.len();
        let ks = self.dw_w.len();
        let pad = ks / 2;
        let mut x1 = vec![0f64; t];
        for i in 0..t {
            let mut s = self.dw_b as f64;
            for j in 0..ks {
                let idx = i as i64 + j as i64 - pad as i64;
                if idx >= 0 && (idx as usize) < t {
                    s += self.dw_w[j] as f64 * mean_hat[idx as usize];
                }
            }
            x1[i] = s;
        }
        let m = x1.iter().sum::<f64>() / t as f64;
        let v = x1.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / t as f64;
        let x2: Vec<f64> = x1
            .iter()
            .map(|a| (a - m) / (v + 1e-5).sqrt() * self.in_w as f64 + self.in_b as f64)
            .collect();
        let gelu = |z: f64| 0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2));
        let mut hmat: Vec<Vec<f64>> = (0..self.hidden)
            .map(|j| x2.iter().map(|a| gelu(self.pw1_w[j] as f64 * a + self.pw1_b[j] as f64)).collect())
            .collect();
        if self.use_cond {
            let dec_rows: Vec<Vec<f32>> = dec.to_vec2::<f32>()?;
            let cond: Vec<f32> = if let Some(p) = &self.pool {
                let n = dec_rows.len().max(1) as f32;
                let d = dec_rows.first().map_or(0, Vec::len);
                let mut avg = vec![0.0f32; d];
                for row in &dec_rows {
                    for (a, &v) in avg.iter_mut().zip(row.iter()) {
                        *a += v;
                    }
                }
                for a in &mut avg {
                    *a /= n;
                }
                p.forward_vec(&avg)
            } else if let Some((tok, fl)) = &self.flat {
                let mut flat_in = Vec::with_capacity(dec_rows.len() * tok.b.len());
                for row in &dec_rows {
                    flat_in.extend_from_slice(&tok.forward_vec(row));
                }
                fl.forward_vec(&flat_in)
            } else {
                vec![0.0; self.hidden]
            };
            if self.concat {
                for j in 0..self.hidden {
                    hmat.push(vec![cond[j] as f64; t]);
                }
            } else {
                for j in 0..self.hidden {
                    for i in 0..t {
                        hmat[j][i] += cond[j] as f64;
                    }
                }
            }
        }
        let o: Vec<Vec<f64>> = (0..self.num_q)
            .map(|qi| {
                if qi == 0 && !self.q50_median {
                    return Vec::new();
                }
                (0..t)
                    .map(|i| {
                        let mut s = self.pw2_b[qi] as f64;
                        for (c, row) in hmat.iter().enumerate() {
                            s += self.pw2_w[qi][c] as f64 * row[i];
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        let softplus = |z: f64| if z > 20.0 { z } else { z.exp().ln_1p() };
        let k = self.k_side;
        let mut q50: Vec<f64> = mean_hat.to_vec();
        if self.q50_median {
            for i in 0..t {
                q50[i] += o[0][i];
            }
        }
        let mut down = vec![vec![0f64; t]; k];
        let mut up = vec![vec![0f64; t]; k];
        for i in 0..t {
            let mut dc = 0.0;
            let mut uc = 0.0;
            for j in 0..k {
                dc += softplus(o[1 + j][i]) + self.eps;
                uc += softplus(o[1 + k + j][i]) + self.eps;
                down[j][i] = q50[i] - dc;
                up[j][i] = q50[i] + uc;
            }
        }
        let mut out = Vec::with_capacity(2 * k + 1);
        for j in (0..k).rev() {
            out.push(down[j].clone());
        }
        out.push(q50);
        for j in 0..k {
            out.push(up[j].clone());
        }
        Ok(out)
    }
}

fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x > 6.0 {
        return s;
    }
    let t = 1.0 / (1.0 + 0.5 * x);
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    s * y
}

#[derive(Debug, Clone)]
struct Predictor {
    sc: SubConfig,
    patcher: Patcher,
    register: Option<Tensor>,
    fft: Option<FftTokens>,
    pos: Option<Tensor>,
    base_norm: Option<LayerNorm>,
    encoder: Block,
    adapter: Option<Linear>,
    decoder: Option<Block>,
    head: Linear,
    mq: Option<MqHead>,
    device: Device,
}

impl Predictor {
    fn load(cfg: &TtmConfig, sc: SubConfig, vb: VarBuilder) -> Result<Self> {
        let ev = vb.pp("backbone").pp("encoder");
        let d = sc.d_model;
        let patcher = if sc.multi_scale {
            let mut n = 0;
            let mut s = sc.context_length;
            while s >= sc.patch_length {
                n += 1;
                s /= 2;
            }
            let mut ps = Vec::with_capacity(n);
            for i in 0..n {
                ps.push(candle_nn::linear(sc.patch_length, d, ev.pp("patcher").pp("projectors").pp(i))?);
            }
            Patcher::Multi(ps)
        } else {
            Patcher::Single(candle_nn::linear(sc.patch_length, d, ev.pp("patcher"))?)
        };
        let register = if sc.register_tokens > 0 {
            Some(ev.get((sc.register_tokens, d), "add_tokens.patch_tokens")?)
        } else {
            None
        };
        let fft = if sc.fft_length > 0 {
            let fv = ev.pp("add_fft_tokens");
            let max_bins = sc.context_length / 2 + 1;
            let k = sc.fft_length.min(max_bins);
            let (dim, one) = if cfg.get_one_freq_emb {
                let raw = (d / k.max(1)).max(1);
                let df = raw.min(cfg.fft_d_freq_max).max(cfg.fft_d_freq_min);
                (df, Some((candle_nn::linear(k * df, d, fv.pp("freq_concat_proj"))?, df)))
            } else {
                (d, None)
            };
            let embed = if cfg.use_fft_embedding {
                FftEmbed::Table(fv.get((max_bins, dim), "freq_embedding.weight")?)
            } else {
                FftEmbed::Mlp(
                    candle_nn::linear(1, dim, fv.pp("freq_index_mlp").pp(0))?,
                    candle_nn::linear(dim, dim, fv.pp("freq_index_mlp").pp(2))?,
                )
            };
            let sl = sc.context_length.max(1);
            let mut cos_lut = Vec::with_capacity(sl);
            let mut sin_lut = Vec::with_capacity(sl);
            for m in 0..sl {
                let a = -2.0 * std::f64::consts::PI * (m as f64) / (sl as f64);
                cos_lut.push(a.cos());
                sin_lut.push(a.sin());
            }
            Some(FftTokens {
                k,
                max_bins,
                context_length: sl,
                cos_lut,
                sin_lut,
                embed,
                one,
                ignore_dc: cfg.fft_ignore_dc,
            })
        } else {
            None
        };
        let pos = if cfg.use_positional_encoding {
            let pv = ev.pp("positional_encoder");
            if pv.contains_tensor("position_enc") {
                Some(pv.get((sc.num_patches, d), "position_enc")?)
            } else {
                Some(sincos_pe(sc.num_patches, d, vb.device())?)
            }
        } else {
            None
        };
        let base_norm = if sc.multi_scale || cfg.enable_base_norm_always {
            Some(candle_nn::layer_norm(sc.num_patches * d, cfg.norm_eps, ev.pp("base_norm"))?)
        } else {
            None
        };
        let encoder = Block::load(cfg, sc.num_layers, sc.num_patches, d, ev.pp("mlp_mixer_encoder"))?;
        let (adapter, decoder) = if cfg.use_decoder {
            let dv = vb.pp("decoder");
            let adapter = if d != sc.decoder_d_model {
                Some(candle_nn::linear(d, sc.decoder_d_model, dv.pp("adapter"))?)
            } else {
                None
            };
            (
                adapter,
                Some(Block::load(
                    cfg,
                    sc.decoder_num_layers,
                    sc.num_patches,
                    sc.decoder_d_model,
                    dv.pp("decoder_block"),
                )?),
            )
        } else {
            (None, None)
        };
        let head_d = if cfg.use_decoder { sc.decoder_d_model } else { d };
        let head = candle_nn::linear(
            sc.num_patches * head_d,
            cfg.prediction_length,
            vb.pp("head").pp("base_forecast_block"),
        )?;
        let mq = if cfg.multi_quantile_head {
            Some(MqHead::load(cfg, head_d, sc.num_patches, vb.pp("multi_quantile_head_block"))?)
        } else {
            None
        };
        Ok(Self {
            sc,
            patcher,
            register,
            fft,
            pos,
            base_norm,
            encoder,
            adapter,
            decoder,
            head,
            mq,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, series: &[f64], policy: &TtmInputPolicy) -> Result<(Vec<f64>, Option<Vec<Vec<f64>>>)> {
        let mut out = self.forward_batch(&[series], policy)?;
        Ok(out.remove(0))
    }

    /// Batched channel-independent forward pass `(B, 1, n_patches, d)` (Phase 4 – Part 1 · D5, Part 2 · D2).
    fn forward_batch(&self, batch: &[&[f64]], policy: &TtmInputPolicy) -> Result<Vec<(Vec<f64>, Option<Vec<Vec<f64>>>)>> {
        let bsz = batch.len();
        if bsz == 0 {
            return Ok(Vec::new());
        }
        let sl = self.sc.context_length;
        let pl = self.sc.patch_length;
        let n = self.sc.n_base_patches;
        let start = sl - n * pl;
        let d = self.sc.d_model;

        let mut prepared: Vec<(Vec<f64>, usize, f64, f64)> = Vec::with_capacity(bsz);
        for s in batch {
            prepared.push(self.sc.scaling.prepare_input(s, sl, policy));
        }

        let mut x = match &self.patcher {
            Patcher::Single(lin) => {
                let mut flat = Vec::with_capacity(bsz * n * pl);
                for (scaled, _, _, _) in &prepared {
                    flat.extend(scaled[start..].iter().map(|v| *v as f32));
                }
                let t = Tensor::from_vec(flat, (bsz, 1, n, pl), &self.device)?;
                t.apply(lin)?
            }
            Patcher::Multi(projs) => {
                let s_len = sl - start;
                let mut outs: Vec<Tensor> = Vec::new();
                let mut i = 0u32;
                loop {
                    let factor = 1usize << i;
                    if s_len < factor || s_len / factor < pl {
                        break;
                    }
                    let ds_len = s_len / factor;
                    let np = ds_len / pl;
                    if np == 0 {
                        break;
                    }
                    let mut flat = Vec::with_capacity(bsz * np * pl);
                    for (scaled, _, _, _) in &prepared {
                        let body = &scaled[start..];
                        let ds: Vec<f64> = (0..ds_len)
                            .map(|w| body[w * factor..(w + 1) * factor].iter().sum::<f64>() / factor as f64)
                            .collect();
                        flat.extend(ds[ds.len() - np * pl..].iter().map(|v| *v as f32));
                    }
                    let t = Tensor::from_vec(flat, (bsz, 1, np, pl), &self.device)?;
                    let proj = projs
                        .get(i as usize)
                        .ok_or_else(|| candle_core::Error::Msg("TTM multi-scale projector 부족".into()))?;
                    outs.push(t.apply(proj)?);
                    i += 1;
                }
                outs.reverse();
                Tensor::cat(&outs, 2)?
            }
        };

        if let Some(r) = &self.register {
            let reg = r.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((bsz, 1, r.dim(0)?, r.dim(1)?))?.contiguous()?;
            x = Tensor::cat(&[&reg, &x], 2)?;
        }
        if let Some(f) = &self.fft {
            let bins = f.max_bins;
            let mut fft_rows: Vec<Tensor> = Vec::with_capacity(bsz);
            for (scaled, obs_start, _, _) in &prepared {
                // Frequency selection runs strictly on the observed slice so padding never distorts FFT tokens.
                let obs_slice = &scaled[*obs_start..sl];
                let lut = (obs_slice.len() == f.context_length).then(|| (f.cos_lut.as_slice(), f.sin_lut.as_slice()));
                let idx_usize = fft_top_indices_lut(obs_slice, f.k, bins, f.ignore_dc, lut);
                let idx: Vec<u32> = idx_usize.into_iter().map(|v| v as u32).collect();
                let emb = match &f.embed {
                    FftEmbed::Table(tab) => tab.index_select(&Tensor::from_vec(idx, f.k, &self.device)?, 0)?,
                    FftEmbed::Mlp(l1, l2) => {
                        let v: Vec<f32> = idx.iter().map(|i| *i as f32 / bins as f32).collect();
                        Tensor::from_vec(v, (f.k, 1), &self.device)?.apply(l1)?.relu()?.apply(l2)?
                    }
                };
                let tok = match &f.one {
                    Some((proj, df)) => emb.reshape((1, f.k * df))?.apply(proj)?,
                    None => emb,
                };
                fft_rows.push(tok.unsqueeze(0)?.unsqueeze(0)?);
            }
            let fft_refs: Vec<&Tensor> = fft_rows.iter().collect();
            let fft_cat = Tensor::cat(&fft_refs, 0)?;
            x = Tensor::cat(&[&fft_cat, &x], 2)?;
        }
        if let Some(p) = &self.pos {
            x = x.broadcast_add(p)?;
        }
        let p_total = x.dim(2)?;
        if p_total != self.sc.num_patches {
            return Err(candle_core::Error::Msg(format!(
                "TTM 패치 수 불일치: {} != {}",
                p_total, self.sc.num_patches
            )));
        }
        if let Some(bn) = &self.base_norm {
            x = x.reshape((bsz, 1, p_total * d))?.apply(bn)?.reshape((bsz, 1, p_total, d))?;
        }
        let mut h = self.encoder.forward(&x)?;
        if let Some(a) = &self.adapter {
            h = h.apply(a)?;
        }
        if let Some(dec) = &self.decoder {
            h = dec.forward(&h)?;
        }
        let dd = h.dim(3)?;
        let flat = h.reshape((bsz, 1, p_total * dd))?;
        let head_out = flat.apply(&self.head)?; // (bsz, 1, pred_len)
        let pred_len = head_out.dim(2)?;
        let y_all = head_out.flatten_all()?.to_vec1::<f32>()?;

        let mut results = Vec::with_capacity(bsz);
        for bi in 0..bsz {
            let (_, _, loc, scale) = prepared[bi];
            let y: Vec<f64> = y_all[bi * pred_len..(bi + 1) * pred_len].iter().map(|v| *v as f64).collect();
            let (point, quant) = match &self.mq {
                Some(mq) => {
                    let dec2 = h.i((bi, 0))?;
                    let q = mq.forward(&y, &dec2)?;
                    let med = q[q.len() / 2].clone();
                    (med, Some(q))
                }
                None => (y, None),
            };
            let inv = |v: &Vec<f64>| -> Vec<f64> { v.iter().map(|a| a * scale + loc).collect() };
            results.push((inv(&point), quant.map(|q| q.iter().map(|r| inv(r)).collect())));
        }
        Ok(results)
    }
}

fn sincos_pe(n: usize, d: usize, device: &Device) -> Result<Tensor> {
    let mut pe = vec![0f64; n * d];
    for p in 0..n {
        for i in (0..d).step_by(2) {
            let div = (-(10000f64.ln()) * i as f64 / d as f64).exp();
            pe[p * d + i] = (p as f64 * div).sin();
            if i + 1 < d {
                pe[p * d + i + 1] = (p as f64 * div).cos();
            }
        }
    }
    let mean = pe.iter().sum::<f64>() / pe.len() as f64;
    let var = pe.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (pe.len() as f64 - 1.0).max(1.0);
    let sd = var.sqrt() * 10.0;
    let v: Vec<f32> = pe.iter().map(|x| ((x - mean) / sd) as f32).collect();
    Tensor::from_vec(v, (n, d), device)
}

fn lower_median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    let mid = (s.len() - 1) / 2;
    let (_, val, _) = s.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    *val
}

pub fn robust_lowess_like(x: &[f64], frac: f64, iters: usize) -> Vec<f64> {
    let t = x.len();
    if t < 3 {
        return x.to_vec();
    }
    let eps = 1e-8;
    let mut w = ((frac * t as f64).round() as usize).max(5);
    if w % 2 == 0 {
        w += 1;
    }
    let cap = if t % 2 == 1 { t } else { t - 1 };
    w = w.min(cap);
    let r = (w - 1) / 2;
    let sigma = 0.25 * w as f64;
    let mut k: Vec<f64> = (0..w).map(|i| (-0.5 * ((i as f64 - r as f64) / sigma).powi(2)).exp()).collect();
    let ks: f64 = k.iter().sum::<f64>() + 1e-12;
    for v in k.iter_mut() {
        *v /= ks;
    }
    let reflect = |src: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(t + 2 * r);
        for i in (1..=r).rev() {
            out.push(src[i]);
        }
        out.extend_from_slice(src);
        for i in 0..r {
            out.push(src[t - 2 - i]);
        }
        out
    };
    let conv = |y: &[f64], wts: Option<&[f64]>| -> Vec<f64> {
        let ones = vec![1.0; t];
        let wv: &[f64] = wts.unwrap_or(&ones);
        let yw: Vec<f64> = y.iter().zip(wv).map(|(a, b)| a * b).collect();
        let yp = reflect(&yw);
        let wp = reflect(wv);
        (0..t)
            .map(|i| {
                let mut num = 0.0;
                let mut den = 0.0;
                for j in 0..w {
                    num += k[j] * yp[i + j];
                    den += k[j] * wp[i + j];
                }
                num / (den + eps)
            })
            .collect()
    };
    let xm = x.iter().sum::<f64>() / t as f64;
    let recenter = |tr: Vec<f64>| -> Vec<f64> {
        let tm = tr.iter().sum::<f64>() / t as f64;
        tr.into_iter().map(|v| v - tm + xm).collect()
    };
    let mut trend = recenter(conv(x, None));
    for _ in 0..iters {
        let resid: Vec<f64> = x.iter().zip(trend.iter()).map(|(a, b)| a - b).collect();
        let med = lower_median(&resid);
        let rc: Vec<f64> = resid.iter().map(|v| v - med).collect();
        let abs: Vec<f64> = rc.iter().map(|v| v.abs()).collect();
        let mad = lower_median(&abs) + eps;
        let wts: Vec<f64> = rc
            .iter()
            .map(|v| {
                let u = v / (6.0 * mad);
                let c = (u * u).clamp(0.0, 1.0);
                (1.0 - c) * (1.0 - c)
            })
            .collect();
        trend = recenter(conv(x, Some(&wts)));
    }
    trend
}

#[derive(Debug, Clone)]
enum Kind {
    Plain(Predictor),
    Decomposed {
        trend: Predictor,
        resid: Predictor,
        scaling: Scaling,
        residual_context: Option<usize>,
        via_variance: bool,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TtmOutput {
    pub point: Vec<f64>,
    pub quantile_levels: Vec<f64>,
    pub quantiles: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Default)]
struct TtmCacheState {
    map: HashMap<u64, TtmOutput>,
    order: VecDeque<u64>,
    calls: u64,
    cache_hits: u64,
    lat_ms: VecDeque<f32>,
}

impl TtmCacheState {
    const CACHE_CAP: usize = 128;
    const LAT_CAP: usize = 256;

    fn hash_input(series: &[f64], policy_tag: &str, fp8: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: &[u8]| {
            for b in bytes {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        mix(fp8.as_bytes());
        mix(b"|");
        mix(policy_tag.as_bytes());
        mix(&(series.len() as u64).to_le_bytes());
        for &v in series {
            mix(&v.to_bits().to_le_bytes());
        }
        h
    }

    fn get(&mut self, key: u64) -> Option<TtmOutput> {
        if let Some(v) = self.map.get(&key).cloned() {
            self.cache_hits = self.cache_hits.saturating_add(1);
            Some(v)
        } else {
            None
        }
    }

    fn insert(&mut self, key: u64, val: TtmOutput, elapsed_ms: f32) {
        self.calls = self.calls.saturating_add(1);
        self.lat_ms.push_back(elapsed_ms);
        while self.lat_ms.len() > Self::LAT_CAP {
            self.lat_ms.pop_front();
        }
        if !self.map.contains_key(&key) {
            self.order.push_back(key);
            while self.order.len() > Self::CACHE_CAP {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
        self.map.insert(key, val);
    }

    fn stats(&self, device: &Device) -> TtmCallStats {
        let mut sorted: Vec<f32> = self.lat_ms.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| -> f32 {
            if sorted.is_empty() {
                0.0
            } else {
                let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            }
        };
        let dev_str = match device {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(_) => "CUDA".to_string(),
            Device::Metal(_) => "Metal".to_string(),
            Device::Rocm(_) => "ROCm".to_string(),
            Device::OpenCl(_) => "OpenCL".to_string(),
            Device::Vulkan(_) => "Vulkan".to_string(),
        };
        TtmCallStats {
            calls: self.calls,
            cache_hits: self.cache_hits,
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
            device: dev_str,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TtmForecaster {
    pub config: TtmConfig,
    kind: Kind,
    pub dir: PathBuf,
    fingerprint: TtmFingerprint,
    policy: TtmInputPolicy,
    state: Arc<Mutex<TtmCacheState>>,
}

impl TtmForecaster {
    pub fn validate(cfg: &TtmConfig) -> std::result::Result<(), String> {
        if cfg.mode != "common_channel" || cfg.decoder_mode != "common_channel" {
            return Err("TTM: mix_channel 모드는 지원하지 않습니다".into());
        }
        if cfg.self_attn {
            return Err("TTM: self_attn 은 지원하지 않습니다".into());
        }
        if cfg.adaptive_patching_levels > 0 || cfg.decoder_adaptive_patching_levels > 0 {
            return Err("TTM: adaptive patching 은 지원하지 않습니다".into());
        }
        if cfg.resolution_prefix_tuning {
            return Err("TTM: resolution_prefix_tuning 은 지원하지 않습니다".into());
        }
        if cfg.decoder_raw_residual {
            return Err("TTM: decoder_raw_residual 은 지원하지 않습니다".into());
        }
        if cfg.enable_forecast_channel_mixing || cfg.categorical_vocab_size_list.is_some() {
            return Err("TTM: forecast channel mixing / categorical 입력은 지원하지 않습니다".into());
        }
        if cfg.use_register_context_gating {
            return Err("TTM: register context gating 은 지원하지 않습니다".into());
        }
        if cfg.multi_quantile_head {
            let q = &cfg.quantile_levels;
            if q.len() % 2 != 1 || (q[q.len() / 2] - 0.5).abs() > 1e-9 {
                return Err("TTM: quantile_levels 는 0.5 를 중심으로 한 홀수 개여야 합니다".into());
            }
        }
        Ok(())
    }

    pub fn from_varbuilder(cfg: TtmConfig, vb: VarBuilder, dir: PathBuf) -> std::result::Result<Self, String> {
        Self::validate(&cfg)?;
        let kind = if cfg.decompose {
            let tsc = SubConfig::derive(&cfg, "trend")?;
            let rsc = SubConfig::derive(&cfg, "residual")?;
            Kind::Decomposed {
                trend: Predictor::load(&cfg, tsc, vb.pp("trend_forecaster")).map_err(|e| format!("TTM trend 로드 실패: {}", e))?,
                resid: Predictor::load(&cfg, rsc, vb.pp("residual_forecaster")).map_err(|e| format!("TTM residual 로드 실패: {}", e))?,
                scaling: Scaling::from_value(&cfg.scaling),
                residual_context: cfg.residual_context_length,
                via_variance: cfg.combine_quantiles_via_variance,
            }
        } else {
            let sc = SubConfig::derive(&cfg, "plain")?;
            Kind::Plain(Predictor::load(&cfg, sc, vb).map_err(|e| format!("TTM 로드 실패: {}", e))?)
        };
        let policy = TtmInputPolicy::default();
        let input_policy = policy.tag();
        let cfg_digest = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "ctx={} pred={} pl={} d={} q={:?} dec={}",
                    cfg.context_length, cfg.prediction_length, cfg.patch_length, cfg.d_model, cfg.quantile_levels, cfg.decompose
                )
                .as_bytes()
            )
        );
        let weights_digest = format!("{:x}", Sha256::digest(dir.to_string_lossy().as_bytes()));
        let fp8 = compute_fp8(&cfg_digest, &weights_digest, &input_policy);
        let fingerprint = TtmFingerprint {
            revision: "custom".into(),
            config_sha256: cfg_digest,
            weights_sha256: weights_digest,
            fp8,
            context: cfg.context_length,
            prediction: cfg.prediction_length,
            quantile_levels: if cfg.multi_quantile_head { cfg.quantile_levels.clone() } else { vec![] },
            input_policy,
            verified: false,
            warnings: config_warnings(&cfg),
        };
        Ok(Self { config: cfg, kind, dir, fingerprint, policy, state: Arc::new(Mutex::new(TtmCacheState::default())) })
    }

    pub fn load(dir: &Path, device: &Device) -> std::result::Result<Self, String> {
        let cfg_path = dir.join("config.json");
        let v = crate::io::read_json(&cfg_path)?;
        let cfg: TtmConfig = serde_json::from_value(v).map_err(|e| format!("TTM config: {}", e))?;
        let weight_files = crate::io::safetensor_files(dir)?;
        let config_sha256 = sha256_file(&cfg_path)?;
        let weights_sha256 = sha256_files(&weight_files)?;
        let vb = crate::io::load_varbuilder(dir, device)?;
        let mut model = Self::from_varbuilder(cfg, vb, dir.to_path_buf())?;
        let verified = config_sha256 == OFFICIAL_CONFIG_SHA256 && weights_sha256 == OFFICIAL_WEIGHTS_SHA256;
        let revision = if verified { OFFICIAL_REVISION.to_string() } else { "custom".to_string() };
        let fp8 = compute_fp8(&config_sha256, &weights_sha256, &model.fingerprint.input_policy);
        let mut warnings = model.fingerprint.warnings.clone();
        if !verified {
            warnings.push(format!("TTM 가중치/설정 해시가 공식 R3 명세와 다릅니다 (fp8={fp8})"));
        }
        model.fingerprint = TtmFingerprint {
            revision,
            config_sha256,
            weights_sha256,
            fp8,
            context: model.config.context_length,
            prediction: model.config.prediction_length,
            quantile_levels: if model.config.multi_quantile_head { model.config.quantile_levels.clone() } else { vec![] },
            input_policy: model.fingerprint.input_policy,
            verified,
            warnings,
        };
        Ok(model)
    }

    pub fn fingerprint(&self) -> &TtmFingerprint {
        &self.fingerprint
    }

    pub fn fp8(&self) -> &str {
        &self.fingerprint.fp8
    }

    /// Evaluate whether TTM can join a fusion call of `n_real` points and requested `horizon` (Phase 4 – Part 1 · D1).
    pub fn eligibility(&self, n_real: usize, horizon: usize, history_n: u64) -> crate::fusion::TtmEligibility {
        let hz = crate::fusion::HOLDOUT.min(n_real / 3).max(2);
        let backtest_points = n_real.saturating_sub(hz);
        let ctx = self.config.context_length;
        let padded = ctx.saturating_sub(n_real);
        let pl = self.config.patch_length.max(1);
        let dropped_by_align = ctx % pl;
        let min_h = self.min_history();
        let max_hz = self.prediction_length();
        let (joins, reason) = if n_real < crate::fusion::MIN_POINTS {
            (false, Some(format!("이력 {n_real}점 < 백테스트 최소 {}점", crate::fusion::MIN_POINTS)))
        } else if horizon > max_hz {
            (false, Some(format!("예측 수평선 {horizon} > 최대 {max_hz}")))
        } else if backtest_points < min_h {
            if n_real >= min_h && history_n >= crate::fusion::HISTORY_ONLY_MIN_N {
                (true, None)
            } else {
                (false, Some(format!("이력 {n_real}점 (백테스트 분할 {backtest_points}점 < 최소 {min_h}점, 11점부터 참여)")))
            }
        } else {
            (true, None)
        };
        crate::fusion::TtmEligibility { real_points: n_real, backtest_points, padded, dropped_by_align, joins, reason }
    }

    pub fn context_length(&self) -> usize {
        self.config.context_length
    }

    pub fn device(&self) -> &Device {
        match &self.kind {
            Kind::Plain(p) => &p.device,
            Kind::Decomposed { trend, .. } => &trend.device,
        }
    }

    pub fn prediction_length(&self) -> usize {
        self.config.prediction_length
    }

    pub fn min_history(&self) -> usize {
        self.policy.min_real.max(1)
    }

    pub fn input_policy(&self) -> &TtmInputPolicy {
        &self.policy
    }

    pub fn with_policy(mut self, policy: TtmInputPolicy) -> Self {
        let tag = policy.tag();
        self.policy = policy;
        let fp8 = compute_fp8(&self.fingerprint.config_sha256, &self.fingerprint.weights_sha256, &tag);
        self.fingerprint.input_policy = tag;
        self.fingerprint.fp8 = fp8;
        self.state = Arc::new(Mutex::new(TtmCacheState::default()));
        self
    }

    pub fn call_stats(&self) -> TtmCallStats {
        self.state
            .lock()
            .map(|s| s.stats(self.device()))
            .unwrap_or_default()
    }

    pub fn clear_cache(&self) {
        if let Ok(mut st) = self.state.lock() {
            st.map.clear();
            st.order.clear();
        }
    }

    pub fn forecast(&self, series: &[f64]) -> std::result::Result<TtmOutput, String> {
        let key = TtmCacheState::hash_input(series, &self.fingerprint.input_policy, &self.fingerprint.fp8);
        if let Ok(mut st) = self.state.lock() {
            if let Some(hit) = st.get(key) {
                return Ok(hit);
            }
        }
        let t0 = Instant::now();
        let out = self.forecast_inner(series)?;
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        if let Ok(mut st) = self.state.lock() {
            st.insert(key, out.clone(), ms);
        }
        Ok(out)
    }

    /// Batched channel-independent forecast over multiple independent series (Phase 4 – Part 1 · D5).
    /// Matches individual `forecast` calls within f32 rounding tolerance while sharing one forward pass.
    pub fn forecast_batch(&self, batch: &[&[f64]]) -> std::result::Result<Vec<TtmOutput>, String> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        for s in batch {
            if s.len() < self.min_history() {
                return Err(format!("TTM 입력이 너무 짧습니다 ({} < {})", s.len(), self.min_history()));
            }
            if s.iter().any(|v| !v.is_finite()) {
                return Err("TTM 입력에 NaN/Inf 가 있습니다".into());
            }
        }
        let t0 = Instant::now();
        let levels = self.config.quantile_levels.clone();
        let results: Vec<TtmOutput> = match &self.kind {
            Kind::Plain(p) => {
                let raw = p.forward_batch(batch, &self.policy).map_err(|e| e.to_string())?;
                raw.into_iter()
                    .map(|(point, quantiles)| TtmOutput { point, quantile_levels: levels.clone(), quantiles })
                    .collect()
            }
            Kind::Decomposed { .. } => {
                let mut out = Vec::with_capacity(batch.len());
                for s in batch {
                    out.push(self.forecast_inner(s)?);
                }
                out
            }
        };
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        if let Ok(mut st) = self.state.lock() {
            for (s, out) in batch.iter().zip(&results) {
                let key = TtmCacheState::hash_input(s, &self.fingerprint.input_policy, &self.fingerprint.fp8);
                st.insert(key, out.clone(), ms / batch.len() as f32);
            }
        }
        Ok(results)
    }

    fn forecast_inner(&self, series: &[f64]) -> std::result::Result<TtmOutput, String> {
        if series.len() < self.min_history() {
            return Err(format!("TTM 입력이 너무 짧습니다 ({} < {})", series.len(), self.min_history()));
        }
        if series.iter().any(|v| !v.is_finite()) {
            return Err("TTM 입력에 NaN/Inf 가 있습니다".into());
        }
        let levels = self.config.quantile_levels.clone();
        match &self.kind {
            Kind::Plain(p) => {
                let (point, q) = p.forward(series, &self.policy).map_err(|e| e.to_string())?;
                Ok(TtmOutput {
                    point,
                    quantile_levels: levels,
                    quantiles: q,
                })
            }
            Kind::Decomposed {
                trend,
                resid,
                scaling,
                residual_context,
                via_variance,
            } => {
                let ctx = self.config.context_length.max(1);
                let (scaled, obs_start, loc, scale) = scaling.prepare_input(series, ctx, &self.policy);
                // Run LOWESS strictly on the observed slice so prefix padding never biases trend extraction.
                let obs_slice = &scaled[obs_start..ctx];
                let tr_obs = robust_lowess_like(obs_slice, 0.15, 2);
                let mut resid_obs: Vec<f64> = obs_slice.iter().zip(&tr_obs).map(|(a, b)| a - b).collect();
                let l_res = residual_context.map(|r| r.min(ctx)).unwrap_or(ctx);
                if resid_obs.len() > l_res {
                    resid_obs = resid_obs[resid_obs.len() - l_res..].to_vec();
                }
                let (tp, tq) = trend.forward(&tr_obs, &self.policy).map_err(|e| e.to_string())?;
                let (rp, rq) = resid.forward(&resid_obs, &self.policy).map_err(|e| e.to_string())?;
                let point: Vec<f64> = tp.iter().zip(rp.iter()).map(|(a, b)| (a + b) * scale + loc).collect();
                let quantiles = match (tq, rq) {
                    (Some(tq), Some(rq)) => {
                        let qn = tq.len();
                        let k = qn / 2;
                        let t = tq[0].len();
                        let mut out = vec![vec![0f64; t]; qn];
                        for i in 0..t {
                            let t50 = tq[k][i];
                            let r50 = rq[k][i];
                            for qi in 0..qn {
                                let v = if *via_variance {
                                    if qi == k {
                                        t50 + r50
                                    } else {
                                        let tw = tq[qi][i] - t50;
                                        let rw = rq[qi][i] - r50;
                                        let mag = (tw * tw + rw * rw + 1e-8).sqrt();
                                        let sgn = (tw + rw).signum() * if tw + rw == 0.0 { 0.0 } else { 1.0 };
                                        t50 + r50 + sgn * mag
                                    }
                                } else {
                                    tq[qi][i] + rq[qi][i]
                                };
                                out[qi][i] = v * scale + loc;
                            }
                        }
                        Some(out)
                    }
                    _ => None,
                };
                Ok(TtmOutput {
                    point,
                    quantile_levels: levels,
                    quantiles,
                })
            }
        }
    }
}
