use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Context};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv2d, conv2d_no_bias, linear, Conv2d, Conv2dConfig, Linear, VarBuilder,
};
use rayon::prelude::*;

use noto_ml::device::{accel_op_ok, cpu_packed_enabled, AccelOp};
use noto_ml::encoder::FastLayerNorm;
use noto_ml::quant::{Family, QLinear, QWeights, QuantPlan, QuantStats};
use noto_ml::vision::antialias_bilinear_weights;

use crate::depth::{clamp_metric, DepthModel, DepthScene, SceneHeads};

fn to_candle(e: anyhow::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{e:#}"))
}
use crate::frame::{DepthKind, DepthMap, NormalMap, Rgb, SegMap};

pub const MODE: &str = "tipsv2-b14-dpt";
pub const DEFAULT_DIR: &str = "models/tipsv2-b14-dpt";
pub const DIR_ENV: &str = "TIPS_DIR";
pub const HEADS_ENV: &str = "NOTO_TIPS_HEADS";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

pub const DEFAULT_SHORT_SIDE_PATCHES: usize = 24;

pub const ADE_CLASSES: usize = 150;
pub const ADE_SKY: usize = 2;
pub const ADE_TREE: usize = 4;
pub const ADE_ROCK: usize = 34;
pub const ADE_BOX: usize = 41;
pub const ADE_GROUND: [usize; 11] = [3, 6, 9, 11, 13, 29, 46, 52, 54, 91, 94];
pub const ADE_VEGETATION: [usize; 4] = [4, 17, 66, 72];
pub const ADE_FAR: [usize; 4] = [2, 16, 68, 48];

pub const ADE_NAMES: [&str; ADE_CLASSES] = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
];

pub fn ade_name(i: usize) -> &'static str {
    ADE_NAMES.get(i).copied().unwrap_or("?")
}

pub fn is_ade_ground(i: usize) -> bool {
    ADE_GROUND.contains(&i)
}

pub fn ade_colour(i: usize) -> [u8; 3] {
    match i {
        ADE_SKY => [120, 170, 230],
        ADE_TREE | 17 | 66 | 72 => [40, 140, 50],
        ADE_ROCK => [130, 130, 136],
        ADE_BOX => [230, 150, 60],
        9 | 29 => [110, 170, 70],
        13 | 46 | 52 | 91 | 94 => [150, 110, 70],
        3 | 6 | 11 | 54 => [170, 160, 150],
        16 | 68 => [90, 100, 120],
        21 | 26 | 60 | 128 => [40, 90, 170],
        _ => {
            let h = (i as u32).wrapping_mul(2_654_435_761);
            [(64 + (h >> 24) % 160) as u8, (64 + (h >> 16) % 160) as u8, (64 + (h >> 8) % 160) as u8]
        }
    }
}

pub const NORMAL_RIGHT: usize = 0;

const DPT_LEVELS: usize = 4;
const FUSION_UPSAMPLE: usize = 2;
const DOWN_KERNEL: usize = 3;
const SAME_PAD_3X3: usize = 1;

#[derive(Clone, Copy, Debug)]
enum LevelScale {
    Up(usize),
    Keep,
    Down(usize),
}

const LEVEL_SCALES: [LevelScale; DPT_LEVELS] = [LevelScale::Up(4), LevelScale::Up(2), LevelScale::Keep, LevelScale::Down(2)];

#[derive(Clone, Debug, PartialEq)]
pub struct TipsConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_hidden: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub register_tokens: usize,
    pub layer_norm_eps: f64,
    pub tap_blocks: [usize; DPT_LEVELS],
    pub post_process_channels: [usize; DPT_LEVELS],
    pub dpt_channels: usize,
    pub depth_bins: usize,
    pub min_depth: f32,
    pub max_depth: f32,
    pub readout_gelu_tanh: bool,
    pub seg_classes: usize,
}

impl Default for TipsConfig {
    fn default() -> Self {
        TipsConfig {
            hidden: 768,
            layers: 12,
            heads: 12,
            mlp_hidden: 3072,
            patch_size: 14,
            image_size: 448,
            register_tokens: 1,
            layer_norm_eps: 1e-6,
            tap_blocks: [2, 5, 8, 11],
            post_process_channels: [96, 192, 384, 768],
            dpt_channels: 256,
            depth_bins: 256,
            min_depth: 1e-3,
            max_depth: 10.0,
            readout_gelu_tanh: true,
            seg_classes: ADE_CLASSES,
        }
    }
}

impl TipsConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_json(raw: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        if let Some(f) = v["vision_fn"].as_str() {
            if f != "vit_base" {
                bail!("only the ViT-B/14 backbone (vision_fn \"vit_base\") is ported, the config asks for {f:?}");
            }
        }
        if let Some(f) = v["ffn_layer"].as_str() {
            if f != "mlp" {
                bail!("only the GELU MLP feed-forward is ported, the config asks for {f:?}");
            }
        }
        let d = TipsConfig::default();
        let us = |keys: &[&str], fallback: usize| keys.iter().find_map(|k| v[*k].as_u64()).map_or(fallback, |x| x as usize);
        let fl = |keys: &[&str], fallback: f64| keys.iter().find_map(|k| v[*k].as_f64()).unwrap_or(fallback);
        let levels = |key: &str| -> anyhow::Result<Option<[usize; DPT_LEVELS]>> {
            let Some(a) = v[key].as_array() else { return Ok(None) };
            let n: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
            match <[usize; DPT_LEVELS]>::try_from(n) {
                Ok(n) if n.len() == a.len() => Ok(Some(n)),
                _ => bail!("{key} must hold {DPT_LEVELS} non-negative integers"),
            }
        };
        let hidden = us(&["embed_dim", "hidden_size"], d.hidden);
        let mlp_hidden = match v["mlp_ratio"].as_f64() {
            Some(r) => (hidden as f64 * r) as usize,
            None => us(&["intermediate_size"], d.mlp_hidden),
        };
        let tap_blocks = match levels("out_indices")? {
            Some(stages) if stages.contains(&0) => bail!("out_indices counts stages from the patch embedding (0); DPT needs block outputs (1 and up)"),
            Some(stages) => stages.map(|s| s - 1),
            None => d.tap_blocks,
        };
        Ok(TipsConfig {
            hidden,
            layers: us(&["num_layers", "num_hidden_layers"], d.layers),
            heads: us(&["num_heads", "num_attention_heads"], d.heads),
            mlp_hidden,
            patch_size: us(&["patch_size"], d.patch_size),
            image_size: us(&["img_size", "image_size"], d.image_size),
            register_tokens: us(&["num_register_tokens"], d.register_tokens),
            layer_norm_eps: fl(&["layer_norm_eps"], d.layer_norm_eps),
            tap_blocks,
            post_process_channels: levels("post_process_channels")?.unwrap_or(d.post_process_channels),
            dpt_channels: us(&["channels", "dpt_channels"], d.dpt_channels),
            depth_bins: us(&["num_depth_bins"], d.depth_bins),
            min_depth: fl(&["min_depth"], d.min_depth as f64) as f32,
            max_depth: fl(&["max_depth"], d.max_depth as f64) as f32,
            readout_gelu_tanh: match ["readout_act", "readout_activation"].iter().find_map(|k| v[*k].as_str()) {
                Some(a) => a.contains("tanh"),
                None => d.readout_gelu_tanh,
            },
            seg_classes: v["id2label"].as_object().map_or(d.seg_classes, |m| m.len()),
        })
    }

    pub fn position_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.heads == 0 || !self.hidden.is_multiple_of(self.heads) {
            bail!("hidden size {} does not split into {} heads", self.hidden, self.heads);
        }
        if self.patch_size == 0 || !self.image_size.is_multiple_of(self.patch_size) {
            bail!("training size {} is not a whole number of {} px patches", self.image_size, self.patch_size);
        }
        if !self.tap_blocks.windows(2).all(|w| w[0] < w[1]) || self.tap_blocks[DPT_LEVELS - 1] >= self.layers {
            bail!("tap blocks {:?} must increase and stay below {} layers", self.tap_blocks, self.layers);
        }
        if self.depth_bins < 2 || !(self.min_depth > 0.0 && self.min_depth < self.max_depth) {
            bail!("depth bins need at least two centres with 0 < min < max, got {} in {}..{}", self.depth_bins, self.min_depth, self.max_depth);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputGrid {
    ShortSide(usize),
    Fixed { cols: usize, rows: usize },
    /// `ratio` times the SigLIP2 grid of the frame (`noto_ml::preprocess::pyramid_grid`), so every SigLIP2 patch
    /// covers exactly `ratio` × `ratio` of these patches (2: the 1 : 2 pyramid, 1: the fast 1 : 1).
    Pyramid { l1_short_side: usize, l1_budget: usize, ratio: usize },
}

impl Default for InputGrid {
    fn default() -> Self {
        InputGrid::ShortSide(DEFAULT_SHORT_SIDE_PATCHES)
    }
}

impl InputGrid {
    pub fn pyramid(cfg: &crate::pyramid::PyramidCfg) -> InputGrid {
        InputGrid::Pyramid { l1_short_side: cfg.l1_short_side, l1_budget: cfg.l1_budget, ratio: cfg.l2_ratio }
    }

    pub fn grid_for(self, width: u32, height: u32) -> (usize, usize) {
        match self {
            InputGrid::Fixed { cols, rows } => (cols.max(1), rows.max(1)),
            InputGrid::Pyramid { l1_short_side, l1_budget, ratio } => {
                let (rows, cols) = noto_ml::preprocess::pyramid_grid(width, height, l1_short_side, l1_budget);
                (cols * ratio.max(1), rows * ratio.max(1))
            }
            InputGrid::ShortSide(n) => {
                let n = n.max(1) as f64;
                let (w, h) = (width.max(1) as f64, height.max(1) as f64);
                let long = |a: f64, b: f64| ((n * a / b).round() as usize).max(1);
                if width <= height {
                    (n as usize, long(h, w))
                } else {
                    (long(w, h), n as usize)
                }
            }
        }
    }
}

struct Attention {
    qkv: QLinear,
    proj: QLinear,
    heads: usize,
    head_dim: usize,
    cpu_packed: bool,
}

impl Attention {
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let d = cfg.hidden;
        Ok(Attention {
            qkv: w.pp("qkv").linear(d, 3 * d).map_err(to_candle)?,
            proj: w.pp("proj").linear(d, d).map_err(to_candle)?,
            heads: cfg.heads,
            head_dim: d / cfg.heads,
            cpu_packed: w.is_cpu_packed(),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let scale = (self.head_dim as f64).powf(-0.5);

        // Fastpath 1: CPU packed 64-key streaming attention (zero transposes, zero [B,H,N,N] score alloc).
        if self.cpu_packed && x.device().is_cpu() && x.dtype() == DType::F32 {
            let qkv_3 = qkv.reshape((b, n, 3, d))?;
            let q_c = qkv_3.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
            let k_c = qkv_3.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
            let v_c = qkv_3.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
            let (qs, ql) = q_c.storage_and_layout();
            let (ks, kl) = k_c.storage_and_layout();
            let (vs, vl) = v_c.storage_and_layout();
            if let (
                candle_core::Storage::Cpu(candle_core::CpuStorage::F32(qv)),
                candle_core::Storage::Cpu(candle_core::CpuStorage::F32(kv)),
                candle_core::Storage::Cpu(candle_core::CpuStorage::F32(vv)),
            ) = (&*qs, &*ks, &*vs)
            {
                let cfg = noto_cpu_kernels::CpuStreamAttnCfg {
                    num_heads: self.heads,
                    num_kv_heads: self.heads,
                    head_dim: self.head_dim,
                    scale: scale as f32,
                    causal: false,
                    past_offset: 0,
                };
                let mut ctx = vec![0.0f32; b * n * d];
                let (q0, k0, v0) = (ql.start_offset(), kl.start_offset(), vl.start_offset());
                for bi in 0..b {
                    noto_cpu_kernels::stream_attention_f32(
                        &qv[q0 + bi * n * d..q0 + (bi + 1) * n * d],
                        &kv[k0 + bi * n * d..k0 + (bi + 1) * n * d],
                        &vv[v0 + bi * n * d..v0 + (bi + 1) * n * d],
                        n,
                        n,
                        &cfg,
                        &mut ctx[bi * n * d..(bi + 1) * n * d],
                    );
                }
                let ctx_t = Tensor::from_vec(ctx, (b, n, d), x.device())?;
                return self.proj.forward(&ctx_t);
            }
        }

        // Fastpath 2: GPU FlashAttention in (b, n, heads, head_dim) layout (zero transpose copies).
        if accel_op_ok(x.device(), AccelOp::FlashAttn) {
            let qkv_4 = qkv.reshape((b, n, 3, self.heads, self.head_dim))?;
            let q = qkv_4.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
            let k = qkv_4.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
            let v = qkv_4.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
            let fa_cfg = candle_nn::flash_attn::FlashAttnConfig::new(scale as f32);
            let ctx = candle_nn::flash_attn::flash_attn_with_config(&q, &k, &v, &fa_cfg)?.reshape((b, n, d))?;
            return self.proj.forward(&ctx);
        }

        let qkv_4 = qkv.reshape((b, n, 3, self.heads, self.head_dim))?;
        let split = |i: usize| -> candle_core::Result<Tensor> {
            qkv_4.narrow(2, i, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, n, d))?;
        self.proj.forward(&ctx)
    }
}

struct Block {
    norm1: FastLayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: FastLayerNorm,
    fc1: QLinear,
    fc2: QLinear,
    ls2: Tensor,
}

impl Block {
    /// Norms and the LayerScale vectors stay F32 whatever the plan (they scale every channel of the residual).
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let (d, eps) = (cfg.hidden, cfg.layer_norm_eps);
        let vb = w.vb();
        let mlp = w.pp("mlp");
        let packed = w.is_cpu_packed();
        Ok(Block {
            norm1: FastLayerNorm::new(d, eps, vb.pp("norm1"), packed)?,
            attn: Attention::new(cfg, &w.pp("attn"))?,
            ls1: vb.pp("ls1").get(d, "gamma")?,
            norm2: FastLayerNorm::new(d, eps, vb.pp("norm2"), packed)?,
            fc1: mlp.pp("fc1").linear(d, cfg.mlp_hidden).map_err(to_candle)?,
            fc2: mlp.pp("fc2").linear(cfg.mlp_hidden, d).map_err(to_candle)?,
            ls2: vb.pp("ls2").get(d, "gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?)?.broadcast_mul(&self.ls1)?)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        &x + mlp.broadcast_mul(&self.ls2)?
    }
}

struct Tap {
    cls: Tensor,
    patches: Tensor,
    rows: usize,
    cols: usize,
}

struct Encoder {
    /// Non-overlapping stride=patch_size Conv2d(3, d, p, s=p) stored as a Linear(3*p*p, d) so every
    /// backend runs native GEMM instead of host conv2d fallback.
    patch_proj: QLinear,
    cls_token: Tensor,
    register_tokens: Option<Tensor>,
    pos_embed: Tensor,
    pos_cache: Mutex<HashMap<(usize, usize), Tensor>>,
    blocks: Vec<Block>,
    norm: FastLayerNorm,
    side: usize,
    hidden: usize,
    patch_size: usize,
    tap_blocks: [usize; DPT_LEVELS],
}

impl Encoder {
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let vb = w.vb();
        let (d, side, p) = (cfg.hidden, cfg.position_side(), cfg.patch_size);
        let proj_vb = vb.pp("patch_embed").pp("proj");
        let pw = proj_vb.get((d, 3, p, p), "weight")?.reshape((d, 3 * p * p))?;
        let pb = proj_vb.get(d, "bias")?;
        let patch_proj = if w.is_cpu_packed() {
            QLinear::from_dense_packed(&pw, Some(&pb))?
        } else {
            QLinear::Dense(Linear::new(pw, Some(pb)))
        };
        let register_tokens = match cfg.register_tokens {
            0 => None,
            r => Some(vb.get((1, r, d), "register_tokens")?),
        };
        Ok(Encoder {
            patch_proj,
            cls_token: vb.get((1, 1, d), "cls_token")?,
            register_tokens,
            pos_embed: vb.get((1, 1 + side * side, d), "pos_embed")?,
            pos_cache: Mutex::new(HashMap::new()),
            blocks: (0..cfg.layers).map(|i| Block::new(cfg, &w.pp(format!("blocks.{i}")))).collect::<candle_core::Result<_>>()?,
            norm: FastLayerNorm::new(d, cfg.layer_norm_eps, vb.pp("norm"), w.is_cpu_packed())?,
            side,
            hidden: d,
            patch_size: p,
            tap_blocks: cfg.tap_blocks,
        })
    }

    fn positions(&self, rows: usize, cols: usize) -> candle_core::Result<Tensor> {
        let (side, d) = (self.side, self.hidden);
        if (rows, cols) == (side, side) {
            return Ok(self.pos_embed.clone());
        }
        let key = (rows, cols);
        if let Some(cached) = self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(cached.clone());
        }
        let dev = self.pos_embed.device();
        let wy = Tensor::from_vec(antialias_bilinear_weights(side, rows), (rows, side), dev)?;
        let wx = Tensor::from_vec(antialias_bilinear_weights(side, cols), (cols, side), dev)?;
        let table = self.pos_embed.narrow(1, 1, side * side)?.reshape((side, side * d))?;
        let by_rows = wy.matmul(&table)?.reshape((rows, side, d))?;
        let grid = wx.broadcast_matmul(&by_rows)?.reshape((1, rows * cols, d))?;
        let res = Tensor::cat(&[&self.pos_embed.narrow(1, 0, 1)?, &grid], 1)?;
        self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, res.clone());
        Ok(res)
    }

    fn taps(&self, pixels: &Tensor) -> candle_core::Result<Vec<Tap>> {
        let (b, _, h, w) = pixels.dims4()?;
        let p = self.patch_size;
        let (rows, cols, d) = (h / p, w / p, self.hidden);
        let n = rows * cols;
        // Exact equivalence to stride=p Conv2d(3, d, p, s=p): (b, 3, rows, p, cols, p) -> (b, rows*cols, 3*p*p) -> Linear.
        let patch_in = pixels
            .reshape((b, 3, rows, p, cols, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, n, 3 * p * p))?;
        let patches = self.patch_proj.forward(&patch_in)?;
        let cls = self.cls_token.broadcast_as((b, 1, d))?;
        let mut x = Tensor::cat(&[&cls, &patches], 1)?.broadcast_add(&self.positions(rows, cols)?)?;
        let registers = match &self.register_tokens {
            Some(r) => {
                let r = r.broadcast_as((b, r.dim(1)?, d))?;
                x = Tensor::cat(&[&x.narrow(1, 0, 1)?, &r, &x.narrow(1, 1, n)?], 1)?;
                r.dim(1)?
            }
            None => 0,
        };
        let last = self.tap_blocks[DPT_LEVELS - 1];
        let mut taps = Vec::with_capacity(DPT_LEVELS);
        for (i, block) in self.blocks.iter().enumerate().take(last + 1) {
            x = block.forward(&x)?;
            if self.tap_blocks.contains(&i) {
                let normed = self.norm.forward(&x)?;
                taps.push(Tap {
                    cls: normed.narrow(1, 0, 1)?.squeeze(1)?,
                    patches: normed.narrow(1, 1 + registers, n)?,
                    rows,
                    cols,
                });
            }
        }
        Ok(taps)
    }
}

/// Non-overlapping `ConvTranspose2d(ch, ch, k=s, s=s, pad=0)` implemented as a native GEMM + pixel-shuffle
/// so every backend (CUDA, ROCm, Vulkan, Metal, CPU) runs on device without conv_transpose2d fallback.
struct PixelShuffleUp {
    proj: Linear,
    bias: Tensor,
    ch: usize,
    scale: usize,
}

impl PixelShuffleUp {
    fn new(ch: usize, scale: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        // ConvTranspose2d weight shape is (c_in=ch, c_out=ch, scale, scale);
        // transpose to (c_out * scale * scale, c_in) for row-major Linear(ch -> ch*scale*scale).
        let w = vb
            .get((ch, ch, scale, scale), "weight")?
            .permute((1, 2, 3, 0))?
            .contiguous()?
            .reshape((ch * scale * scale, ch))?;
        let bias = vb.get(ch, "bias")?.reshape((1, ch, 1, 1))?;
        Ok(PixelShuffleUp { proj: Linear::new(w, None), bias, ch, scale })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, _, h, w) = x.dims4()?;
        let s = self.scale;
        let ch = self.ch;
        let x_hwc = x.permute((0, 2, 3, 1))?.contiguous()?;
        let y = self
            .proj
            .forward(&x_hwc)?
            .reshape((b, h, w, ch, s, s))?
            .permute((0, 3, 1, 4, 2, 5))?
            .contiguous()?
            .reshape((b, ch, h * s, w * s))?;
        y.broadcast_add(&self.bias)
    }
}

struct ReassembleLevel {
    readout: QLinear,
    /// 1x1 `out_projection` stored as Linear(d -> ch) and applied directly on (b, n, d) before NCHW reshape.
    out_projection: Linear,
    out_channels: usize,
    resize: Resize,
    gelu_tanh: bool,
}

enum Resize {
    Up(PixelShuffleUp),
    Keep,
    Down(Conv2d),
}

impl ReassembleLevel {
    /// The readout projection may be quantized (a plan's `tv.readout`); the 1×1 projection and the resampling
    /// layers stay F32.
    fn new(level: usize, cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let vb = w.vb();
        let (d, ch) = (cfg.hidden, cfg.post_process_channels[level]);
        let resize_vb = vb.pp("resize_layers").pp(level);
        let resize = match LEVEL_SCALES[level] {
            LevelScale::Up(s) => Resize::Up(PixelShuffleUp::new(ch, s, resize_vb)?),
            LevelScale::Keep => Resize::Keep,
            LevelScale::Down(s) => {
                let down = Conv2dConfig { stride: s, padding: SAME_PAD_3X3, ..Default::default() };
                Resize::Down(conv2d(ch, ch, DOWN_KERNEL, down, resize_vb)?)
            }
        };
        let out_vb = vb.pp("out_projections").pp(level);
        let out_w = out_vb.get((ch, d, 1, 1), "weight")?.reshape((ch, d))?;
        let out_b = out_vb.get(ch, "bias")?;
        Ok(ReassembleLevel {
            readout: w.pp("readout_projects").pp(level).linear(2 * d, d).map_err(to_candle)?,
            out_projection: Linear::new(out_w, Some(out_b)),
            out_channels: ch,
            resize,
            gelu_tanh: cfg.readout_gelu_tanh,
        })
    }

    fn forward(&self, tap: &Tap) -> candle_core::Result<Tensor> {
        let (b, n, d) = tap.patches.dims3()?;
        let readout = tap.cls.unsqueeze(1)?.broadcast_as((b, n, d))?;
        let x = self.readout.forward(&Tensor::cat(&[&tap.patches, &readout], 2)?.contiguous()?)?;
        let x = if self.gelu_tanh { x.gelu()? } else { x.gelu_erf()? };
        let x = self.out_projection.forward(&x)?;
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, self.out_channels, tap.rows, tap.cols))?;
        match &self.resize {
            Resize::Up(up) => up.forward(&x),
            Resize::Keep => Ok(x),
            Resize::Down(down) => down.forward(&x),
        }
    }
}

struct ResidualUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualUnit {
    fn new(ch: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let same = Conv2dConfig { padding: SAME_PAD_3X3, ..Default::default() };
        Ok(ResidualUnit { conv1: conv2d_no_bias(ch, ch, 3, same, vb.pp("conv1"))?, conv2: conv2d_no_bias(ch, ch, 3, same, vb.pp("conv2"))? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.conv2.forward(&self.conv1.forward(&x.relu()?)?.relu()?)?;
        x + y
    }
}

struct FusionBlock {
    residual_unit: Option<ResidualUnit>,
    main_unit: ResidualUnit,
    /// 1x1 `out_conv` executed as a direct `[ch, ch] @ [b, ch, h*w]` GEMM without host conv2d fallback.
    out_w: Tensor,
    out_b: Tensor,
}

impl FusionBlock {
    fn new(ch: usize, has_skip: bool, vb: VarBuilder) -> candle_core::Result<Self> {
        let out_vb = vb.pp("out_conv");
        let out_w = out_vb.get((ch, ch, 1, 1), "weight")?.reshape((ch, ch))?;
        let out_b = out_vb.get(ch, "bias")?.reshape((1, ch, 1, 1))?;
        Ok(FusionBlock {
            residual_unit: if has_skip { Some(ResidualUnit::new(ch, vb.pp("residual_unit"))?) } else { None },
            main_unit: ResidualUnit::new(ch, vb.pp("main_unit"))?,
            out_w,
            out_b,
        })
    }

    fn forward(&self, x: &Tensor, skip: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        if let (Some(unit), Some(skip)) = (&self.residual_unit, skip) {
            let (_, _, h, w) = x.dims4()?;
            let skip = if skip.dims() != x.dims() { skip.upsample_bilinear2d(h, w, false)? } else { skip.clone() };
            x = (x + unit.forward(&skip)?)?;
        }
        let x = self.main_unit.forward(&x)?;
        let (b, ch, h, w) = x.dims4()?;
        let (oh, ow) = (h * FUSION_UPSAMPLE, w * FUSION_UPSAMPLE);
        let x = x.upsample_bilinear2d(oh, ow, true)?;
        let y = self.out_w.broadcast_matmul(&x.reshape((b, ch, oh * ow))?)?.reshape((b, ch, oh, ow))?;
        y.broadcast_add(&self.out_b)
    }
}

struct Decoder {
    reassemble: Vec<ReassembleLevel>,
    convs: Vec<Conv2d>,
    fusion: Vec<FusionBlock>,
    project: Conv2d,
}

impl Decoder {
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let vb = w.vb();
        let ch = cfg.dpt_channels;
        let same = Conv2dConfig { padding: SAME_PAD_3X3, ..Default::default() };
        Ok(Decoder {
            reassemble: (0..DPT_LEVELS).map(|i| ReassembleLevel::new(i, cfg, &w.pp("reassemble"))).collect::<candle_core::Result<_>>()?,
            convs: (0..DPT_LEVELS).map(|i| conv2d_no_bias(cfg.post_process_channels[i], ch, 3, same, vb.pp("convs").pp(i))).collect::<candle_core::Result<_>>()?,
            fusion: (0..DPT_LEVELS).map(|i| FusionBlock::new(ch, i > 0, vb.pp("fusion_blocks").pp(i))).collect::<candle_core::Result<_>>()?,
            project: conv2d(ch, ch, 3, same, vb.pp("project"))?,
        })
    }

    fn forward(&self, taps: &[Tap]) -> candle_core::Result<Tensor> {
        let levels: Vec<Tensor> = taps
            .iter()
            .zip(&self.reassemble)
            .zip(&self.convs)
            .map(|((tap, level), conv)| conv.forward(&level.forward(tap)?))
            .collect::<candle_core::Result<_>>()?;
        let mut out = self.fusion[0].forward(&levels[DPT_LEVELS - 1], None)?;
        for (i, block) in self.fusion.iter().enumerate().skip(1) {
            out = block.forward(&out, Some(&levels[DPT_LEVELS - 1 - i]))?;
        }
        self.project.forward(&out)
    }
}

struct DepthHead {
    dec: Decoder,
    classifier: Linear,
    centers_col: Tensor,
    centers_pair: Tensor,
    min_depth: f64,
}

impl DepthHead {
    /// The bin classifier stays F32: 256 logits through a ReLU and a normalization make the depth and its spread.
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        let vb = w.vb();
        let centers = bin_centers(cfg.min_depth, cfg.max_depth, cfg.depth_bins, vb.device())?;
        let centers_col = centers.reshape((cfg.depth_bins, 1))?;
        let centers_sq = centers.sqr()?.reshape((cfg.depth_bins, 1))?;
        let centers_pair = Tensor::cat(&[&centers_col, &centers_sq], 1)?;
        Ok(DepthHead {
            dec: Decoder::new(cfg, w)?,
            classifier: linear(cfg.dpt_channels, cfg.depth_bins, vb.pp("depth_head"))?,
            centers_col,
            centers_pair,
            min_depth: cfg.min_depth as f64,
        })
    }

    fn forward(&self, taps: &[Tap], spread: bool) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let out = self.dec.forward(taps)?.relu()?;
        let logits = self.classifier.forward(&out.permute((0, 2, 3, 1))?.contiguous()?)?;
        if spread {
            let (mean, sd) = decode_bins_spread(&logits, &self.centers_pair, self.min_depth)?;
            Ok((mean.unsqueeze(1)?, Some(sd.unsqueeze(1)?)))
        } else {
            Ok((decode_bins(&logits, &self.centers_col, self.min_depth)?.unsqueeze(1)?, None))
        }
    }
}

struct NormalsHead {
    dec: Decoder,
    out: Linear,
}

impl NormalsHead {
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        Ok(NormalsHead { dec: Decoder::new(cfg, w)?, out: linear(cfg.dpt_channels, 3, w.vb().pp("normals_head"))? })
    }

    fn forward(&self, taps: &[Tap]) -> candle_core::Result<Tensor> {
        let x = self.out.forward(&self.dec.forward(taps)?.permute((0, 2, 3, 1))?.contiguous()?)?;
        unit_last(&x)?.permute((0, 3, 1, 2))?.contiguous()
    }
}

struct SegHead {
    dec: Decoder,
    out: Linear,
}

impl SegHead {
    fn new(cfg: &TipsConfig, w: &QWeights) -> candle_core::Result<Self> {
        Ok(SegHead { dec: Decoder::new(cfg, w)?, out: linear(cfg.dpt_channels, cfg.seg_classes, w.vb().pp("segmentation_head"))? })
    }

    fn forward(&self, taps: &[Tap]) -> candle_core::Result<Tensor> {
        self.out.forward(&self.dec.forward(taps)?.permute((0, 2, 3, 1))?.contiguous()?)?.permute((0, 3, 1, 2))?.contiguous()
    }
}

fn unit_last(x: &Tensor) -> candle_core::Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + 1e-12)?;
    x.broadcast_div(&n)
}

fn unit_channels(x: &Tensor) -> candle_core::Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-12)?;
    x.broadcast_div(&n)
}

fn flip_w(t: &Tensor) -> candle_core::Result<Tensor> {
    let w = t.dim(D::Minus1)?;
    let idx = Tensor::from_vec((0..w as u32).rev().collect::<Vec<u32>>(), w, t.device())?;
    t.index_select(&idx, t.rank() - 1)
}

fn merge_views(t: Tensor, b: usize, views: usize, mirror: Option<usize>) -> candle_core::Result<Tensor> {
    if views < 2 {
        return Ok(t);
    }
    let a = t.narrow(0, 0, b)?;
    let mut f = flip_w(&t.narrow(0, b, b)?)?;
    if let Some(c) = mirror {
        let ch = f.dim(1)?;
        let sign: Vec<f32> = (0..ch).map(|k| if k == c { -1.0 } else { 1.0 }).collect();
        f = f.broadcast_mul(&Tensor::from_vec(sign, (1, ch, 1, 1), f.device())?)?;
    }
    (a + f)? * 0.5
}

fn bin_centers(min: f32, max: f32, bins: usize, device: &Device) -> candle_core::Result<Tensor> {
    let step = (max as f64 - min as f64) / (bins - 1) as f64;
    let centers: Vec<f32> = (0..bins).map(|i| (min as f64 + step * i as f64) as f32).collect();
    Tensor::from_vec(centers, bins, device)
}

fn decode_bins(logits: &Tensor, centers_col: &Tensor, min_depth: f64) -> candle_core::Result<Tensor> {
    let p = (logits.relu()? + min_depth)?;
    let p = p.broadcast_div(&p.sum_keepdim(D::Minus1)?)?;
    let col = if centers_col.rank() == 1 {
        centers_col.reshape((centers_col.dim(0)?, 1))?
    } else {
        centers_col.clone()
    };
    p.broadcast_matmul(&col)?.squeeze(D::Minus1)
}

fn decode_bins_spread(logits: &Tensor, centers_pair: &Tensor, min_depth: f64) -> candle_core::Result<(Tensor, Tensor)> {
    let p = (logits.relu()? + min_depth)?;
    let p = p.broadcast_div(&p.sum_keepdim(D::Minus1)?)?;
    let moments = p.broadcast_matmul(centers_pair)?;
    let mean = moments.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let second = moments.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let var = (second - mean.sqr()?)?.relu()?;
    Ok((mean, var.sqrt()?))
}

pub struct TipsDepth {
    cfg: TipsConfig,
    encoder: Encoder,
    head: DepthHead,
    normals_head: Option<NormalsHead>,
    seg_head: Option<SegHead>,
    device: Device,
    pub input_grid: InputGrid,
    /// The pyramid's grid, which overrides `input_grid` while a pipeline runs the 1 : 2 : 4 alignment.
    pub pyramid_grid: Option<InputGrid>,
    pub max_rel_spread: Option<f32>,
    pub flip_tta: bool,
    /// `tipsv2-b14-dpt`, with the plan's tag when the encoder or the readout is quantized (SDS scopes, calibration).
    mode: String,
    quant: QuantPlan,
    quant_stats: QuantStats,
}

struct RunOut {
    depth: DepthMap,
    spread: Option<DepthMap>,
    normals: Option<NormalMap>,
    seg: Option<SegMap>,
}

pub fn heads_from_env() -> SceneHeads {
    match std::env::var(HEADS_ENV).ok().as_deref().map(str::trim) {
        Some("depth") | Some("none") => SceneHeads::NONE,
        Some("normals") => SceneHeads { normals: true, segmentation: false },
        Some("segmentation") | Some("seg") => SceneHeads { normals: false, segmentation: true },
        _ => SceneHeads::ALL,
    }
}

impl TipsDepth {
    pub fn load(dir: &Path, device: &Device) -> anyhow::Result<Self> {
        Self::load_with(dir, device, heads_from_env())
    }

    /// [`TipsDepth::load_quant`] with the plan for `device` (`NOTO_QUANT`, else the measured default,
    /// `noto_ml::QuantPlan::for_load`).
    pub fn load_with(dir: &Path, device: &Device, heads: SceneHeads) -> anyhow::Result<Self> {
        Self::load_quant(dir, device, heads, &QuantPlan::for_load(device)?)
    }

    /// The checkpoint with the encoder and readout weights of `plan` quantized once from its F32 values.
    pub fn load_quant(dir: &Path, device: &Device, heads: SceneHeads, plan: &QuantPlan) -> anyhow::Result<Self> {
        let weights = dir.join(WEIGHTS_FILE);
        if !weights.exists() {
            bail!(
                "TIPSv2 DPT weights not found at {}. Download google/tipsv2-b14-dpt ({WEIGHTS_FILE}, optionally \
                 {CONFIG_FILE}) into that folder, or point ${DIR_ENV} at the folder that has them (default {DEFAULT_DIR}).",
                weights.display()
            );
        }
        let config_path = dir.join(CONFIG_FILE);
        let cfg = if config_path.exists() { TipsConfig::from_file(&config_path)? } else { TipsConfig::default() };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)? };
        let packed = plan.is_f32() && cpu_packed_enabled(device);
        let w = QWeights::with_plan(vb, &weights, Family::Tips, plan)?.with_cpu_packed(packed);
        let built = Self::build(cfg, &w, device, heads).with_context(|| format!("loading {}", weights.display()))?;
        w.flush_cache();
        Ok(built)
    }

    pub fn from_var_builder(cfg: TipsConfig, vb: VarBuilder, device: &Device) -> anyhow::Result<Self> {
        Self::from_var_builder_with(cfg, vb, device, SceneHeads::NONE)
    }

    pub fn from_var_builder_with(cfg: TipsConfig, vb: VarBuilder, device: &Device, heads: SceneHeads) -> anyhow::Result<Self> {
        Self::build(cfg, &QWeights::plain(vb, Family::Tips), device, heads)
    }

    fn build(cfg: TipsConfig, w: &QWeights, device: &Device, heads: SceneHeads) -> anyhow::Result<Self> {
        cfg.validate()?;
        let vb = w.vb();
        let encoder = Encoder::new(&cfg, &w.pp("vision_encoder"))?;
        let head = DepthHead::new(&cfg, &w.pp("depth_head"))?;
        let has = |head: &str| vb.contains_tensor(&format!("{head}.project.weight"));
        let normals_head = if heads.normals && has("normals_head") { Some(NormalsHead::new(&cfg, &w.pp("normals_head"))?) } else { None };
        let seg_head = if heads.segmentation && has("segmentation_head") { Some(SegHead::new(&cfg, &w.pp("segmentation_head"))?) } else { None };
        let quant = w.plan().clone();
        let mut mode = format!("{MODE}{}", quant.mode_suffix(Family::Tips));
        if w.is_cpu_packed() {
            mode.push_str("+cpupack");
        }
        Ok(TipsDepth {
            cfg,
            encoder,
            head,
            normals_head,
            seg_head,
            device: device.clone(),
            input_grid: InputGrid::default(),
            pyramid_grid: None,
            max_rel_spread: None,
            flip_tta: false,
            mode,
            quant,
            quant_stats: w.stats(),
        })
    }

    pub fn config(&self) -> &TipsConfig {
        &self.cfg
    }

    /// The plan the weights run with.
    pub fn quant_plan(&self) -> &QuantPlan {
        &self.quant
    }

    /// What the load quantized (formats, bytes, time).
    pub fn quant_stats(&self) -> &QuantStats {
        &self.quant_stats
    }

    /// The grid a `width` × `height` frame runs on: the pyramid's while one is set, else `input_grid`.
    pub fn grid_for(&self, width: u32, height: u32) -> (usize, usize) {
        self.pyramid_grid.unwrap_or(self.input_grid).grid_for(width, height)
    }

    pub fn heads(&self) -> SceneHeads {
        SceneHeads { normals: self.normals_head.is_some(), segmentation: self.seg_head.is_some() }
    }

    pub fn set_readout_gelu_tanh(&mut self, on: bool) {
        self.cfg.readout_gelu_tanh = on;
        let decoders = std::iter::once(&mut self.head.dec).chain(self.normals_head.as_mut().map(|h| &mut h.dec)).chain(self.seg_head.as_mut().map(|h| &mut h.dec));
        for dec in decoders {
            for level in &mut dec.reassemble {
                level.gelu_tanh = on;
            }
        }
    }

    pub fn predict(&self, rgb: &Rgb) -> anyhow::Result<DepthMap> {
        let mut out = self.predict_batch(&[rgb])?;
        Ok(out.pop().expect("one map per frame"))
    }

    pub fn predict_batch(&self, frames: &[&Rgb]) -> anyhow::Result<Vec<DepthMap>> {
        Ok(self.run_batch(frames, false, SceneHeads::NONE)?.into_iter().map(|r| r.depth).collect())
    }

    pub fn predict_batch_spread(&self, frames: &[&Rgb]) -> anyhow::Result<Vec<(DepthMap, DepthMap)>> {
        Ok(self.run_batch(frames, true, SceneHeads::NONE)?.into_iter().map(|r| (r.depth, r.spread.expect("spread was asked for"))).collect())
    }

    pub fn scene_batch(&self, frames: &[&Rgb], want: SceneHeads) -> anyhow::Result<Vec<DepthScene>> {
        Ok(self
            .run_batch(frames, false, want)?
            .into_iter()
            .map(|r| {
                let mut s = DepthScene { depth: r.depth, normals: r.normals, segmentation: r.seg };
                s.mask_sky();
                s
            })
            .collect())
    }

    fn run_batch(&self, frames: &[&Rgb], want_spread: bool, want: SceneHeads) -> anyhow::Result<Vec<RunOut>> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (i, f) in frames.iter().enumerate() {
            if f.width == 0 || f.height == 0 {
                bail!("empty keyframe ({}x{})", f.width, f.height);
            }
            groups.entry((f.width, f.height)).or_default().push(i);
        }
        let want = want.and(self.heads());
        let policy = noto_ml::BatchPolicy::for_device(&self.device);
        let p = self.cfg.patch_size;
        let views = if self.flip_tta { 2 } else { 1 };
        let mut out: Vec<Option<RunOut>> = (0..frames.len()).map(|_| None).collect();
        let spread = want_spread || self.max_rel_spread.is_some();
        for (&(w, h), idx) in &groups {
            let (cols, rows) = self.grid_for(w, h);
            for chunk in idx.chunks((policy.image_rows(cols * rows) / views).max(1)) {
                let b = chunk.len();
                let (in_w, in_h) = ((cols * p) as u32, (rows * p) as u32);
                let mut pixels = model_input_batch(chunk, frames, in_w, in_h, &self.device)?;
                if views > 1 {
                    pixels = Tensor::cat(&[&pixels, &flip_w(&pixels)?], 0)?;
                }
                let taps = self.encoder.taps(&pixels)?;
                let (depth, sd) = self.head.forward(&taps, spread)?;
                let depth = merge_views(depth, b, views, None)?.upsample_bilinear2d(h as usize, w as usize, false)?;
                let sd = sd.map(|t| merge_views(t, b, views, None)?.upsample_bilinear2d(h as usize, w as usize, false)).transpose()?;
                let normals = match (&self.normals_head, want.normals) {
                    (Some(head), true) => Some(unit_channels(&merge_views(head.forward(&taps)?, b, views, Some(NORMAL_RIGHT))?)?),
                    _ => None,
                };
                let seg = match (&self.seg_head, want.segmentation) {
                    (Some(head), true) => {
                        let logits = merge_views(head.forward(&taps)?, b, views, None)?;
                        let m = logits.max_keepdim(1)?;
                        let shifted = logits.broadcast_sub(&m)?;
                        let lse = shifted.exp()?.sum_keepdim(1)?.log()?;
                        let sky = shifted.narrow(1, ADE_SKY, 1)?.broadcast_sub(&lse)?.exp()?.squeeze(1)?.contiguous()?;
                        Some((logits.argmax(1)?, sky))
                    }
                    _ => None,
                };
                // Bulk D2H transfer once per output tensor instead of 5*B separate GPU syncs.
                let frame_px = (w * h) as usize;
                let depth_all: Vec<f32> = depth.flatten_all()?.to_vec1()?;
                let sd_all: Option<Vec<f32>> = match &sd {
                    Some(t) => Some(t.flatten_all()?.to_vec1()?),
                    None => None,
                };
                let normals_all: Option<(usize, usize, Vec<f32>)> = match &normals {
                    Some(t) => {
                        let (_, _, nh, nw) = t.dims4()?;
                        let v: Vec<f32> = t.permute((0, 2, 3, 1))?.contiguous()?.flatten_all()?.to_vec1()?;
                        Some((nh, nw, v))
                    }
                    None => None,
                };
                let seg_all: Option<(usize, usize, Vec<u32>, Vec<f32>)> = match &seg {
                    Some((labels, sky)) => {
                        let (_, sh, sw) = labels.dims3()?;
                        let l_all: Vec<u32> = labels.flatten_all()?.to_vec1()?;
                        let s_all: Vec<f32> = sky.flatten_all()?.to_vec1()?;
                        Some((sh, sw, l_all, s_all))
                    }
                    None => None,
                };
                for (k, &i) in chunk.iter().enumerate() {
                    let mut map = DepthMap {
                        width: w,
                        height: h,
                        kind: DepthKind::Metric,
                        data: depth_all[k * frame_px..(k + 1) * frame_px].to_vec(),
                    };
                    clamp_metric(&mut map, self.cfg.max_depth);
                    let sd = sd_all.as_ref().map(|v| DepthMap {
                        width: w,
                        height: h,
                        kind: DepthKind::Metric,
                        data: v[k * frame_px..(k + 1) * frame_px].to_vec(),
                    });
                    if let (Some(limit), Some(s)) = (self.max_rel_spread, &sd) {
                        for (d, e) in map.data.iter_mut().zip(&s.data) {
                            if *e > limit * *d {
                                *d = f32::INFINITY;
                            }
                        }
                    }
                    let normals = normals_all.as_ref().map(|(nh, nw, v)| {
                        let stride = nh * nw * 3;
                        NormalMap {
                            width: *nw as u32,
                            height: *nh as u32,
                            data: v[k * stride..(k + 1) * stride].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                        }
                    });
                    let seg = seg_all.as_ref().map(|(sh, sw, l_all, s_all)| {
                        let stride = sh * sw;
                        SegMap {
                            width: *sw as u32,
                            height: *sh as u32,
                            labels: l_all[k * stride..(k + 1) * stride].iter().map(|&c| c.min(255) as u8).collect(),
                            sky: s_all[k * stride..(k + 1) * stride].to_vec(),
                        }
                    });
                    out[i] = Some(RunOut { depth: map, spread: if want_spread { sd } else { None }, normals, seg });
                }
            }
        }
        noto_ml::trim_idle_pool(&self.device);
        Ok(out.into_iter().map(|m| m.expect("every frame is in one group")).collect())
    }
}

fn u8_to_unit_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = i as f32 / 255.0;
        }
        t
    })
}

fn rgb_to_chw_f32(rgb: &Rgb, width: u32, height: u32) -> anyhow::Result<Vec<f32>> {
    let resized;
    let data: &[u8] = if (rgb.width, rgb.height) == (width, height) {
        &rgb.data
    } else {
        let src = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(rgb.width, rgb.height, rgb.data.as_slice())
            .context("RGB buffer does not match its size")?;
        resized = image::imageops::resize(&src, width, height, image::imageops::FilterType::Triangle);
        resized.as_raw()
    };
    let lut = u8_to_unit_lut();
    let plane = (width * height) as usize;
    let mut chw = vec![0f32; 3 * plane];
    let (r_plane, rest) = chw.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    for (i, px) in data.chunks_exact(3).enumerate() {
        r_plane[i] = lut[px[0] as usize];
        g_plane[i] = lut[px[1] as usize];
        b_plane[i] = lut[px[2] as usize];
    }
    Ok(chw)
}

fn model_input_batch(chunk: &[usize], frames: &[&Rgb], width: u32, height: u32, device: &Device) -> anyhow::Result<Tensor> {
    let b = chunk.len();
    if b == 1 {
        let chw = rgb_to_chw_f32(frames[chunk[0]], width, height)?;
        return Ok(Tensor::from_vec(chw, (1, 3, height as usize, width as usize), device)?);
    }
    let per_frame: Vec<Vec<f32>> = chunk
        .par_iter()
        .map(|&i| rgb_to_chw_f32(frames[i], width, height))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let frame_len = 3 * (width * height) as usize;
    let mut flat = Vec::with_capacity(b * frame_len);
    for f in per_frame {
        flat.extend_from_slice(&f);
    }
    Ok(Tensor::from_vec(flat, (b, 3, height as usize, width as usize), device)?)
}

impl DepthModel for TipsDepth {
    fn mode(&self) -> &str {
        &self.mode
    }

    fn device(&self) -> String {
        noto_ml::device_label(&self.device)
    }

    fn max_depth_m(&self) -> Option<f32> {
        Some(self.cfg.max_depth)
    }

    fn infer(&mut self, rgb: &Rgb) -> anyhow::Result<DepthMap> {
        self.predict(rgb)
    }

    fn infer_batch(&mut self, frames: &[&Rgb]) -> anyhow::Result<Vec<DepthMap>> {
        self.predict_batch(frames)
    }

    fn scene_heads(&self) -> SceneHeads {
        self.heads()
    }

    fn infer_scene_batch(&mut self, frames: &[&Rgb], want: SceneHeads) -> anyhow::Result<Vec<DepthScene>> {
        self.scene_batch(frames, want)
    }

    fn set_pyramid(&mut self, cfg: Option<&crate::pyramid::PyramidCfg>) {
        self.pyramid_grid = cfg.map(InputGrid::pyramid);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::{Arc, Mutex};

    use candle_core::Shape;
    use candle_nn::var_builder::SimpleBackend;

    use super::*;

    const DEPTH_PREFIXES: [&str; 2] = ["vision_encoder.", "depth_head."];
    const UNUSED_AT_INFERENCE: [&str; 1] = ["vision_encoder.mask_token"];
    const WEIGHT_RANGE: f32 = 0.05;

    type Shapes = BTreeMap<String, Vec<usize>>;
    type Requested = Arc<Mutex<BTreeSet<String>>>;

    fn checkpoint_shapes() -> Shapes {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tipsv2_b14_dpt_header.json");
        let raw = std::fs::read_to_string(&path).unwrap();
        let header: BTreeMap<String, serde_json::Value> = serde_json::from_str(&raw).unwrap();
        header
            .into_iter()
            .map(|(name, meta)| {
                assert_eq!(meta["dtype"], "F32", "{name}");
                let shape = meta["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                (name, shape)
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    enum Fill {
        Random,
        Zero,
    }

    struct HeaderBackend {
        shapes: Shapes,
        fill: Fill,
        requested: Requested,
    }

    impl HeaderBackend {
        fn tensor(&self, name: &str, want: Option<&Shape>, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            let Some(shape) = self.shapes.get(name) else { candle_core::bail!("{name} is not a depth-path tensor of the checkpoint") };
            if let Some(want) = want {
                if want.dims() != shape.as_slice() {
                    candle_core::bail!("{name}: the port asks for {:?}, the checkpoint has {shape:?}", want.dims());
                }
            }
            self.requested.lock().unwrap().insert(name.to_string());
            match self.fill {
                Fill::Random => Tensor::rand(-WEIGHT_RANGE, WEIGHT_RANGE, shape.as_slice(), dev)?.to_dtype(dtype),
                Fill::Zero => Tensor::zeros((), dtype, dev)?.broadcast_as(shape.as_slice()),
            }
        }
    }

    impl SimpleBackend for HeaderBackend {
        fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            self.tensor(name, Some(&s), dtype, dev)
        }

        fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            self.tensor(name, None, dtype, dev)
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.shapes.contains_key(name)
        }
    }

    fn header_model(fill: Fill) -> (TipsDepth, Shapes, Requested) {
        let shapes: Shapes = checkpoint_shapes().into_iter().filter(|(n, _)| DEPTH_PREFIXES.iter().any(|p| n.starts_with(p))).collect();
        let requested = Arc::new(Mutex::new(BTreeSet::new()));
        let backend = HeaderBackend { shapes: shapes.clone(), fill, requested: requested.clone() };
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
        let model = TipsDepth::from_var_builder(TipsConfig::default(), vb, &Device::Cpu).unwrap();
        (model, shapes, requested)
    }

    fn gradient(width: u32, height: u32) -> Rgb {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, ((x + y) * 7 % 256) as u8]);
            }
        }
        Rgb::new(width, height, data)
    }

    #[test]
    fn runs_on_every_name_and_shape_of_the_real_checkpoint() {
        let (mut model, _, _) = header_model(Fill::Random);
        model.input_grid = InputGrid::Fixed { cols: 4, rows: 3 };
        let cfg = model.config().clone();
        let map = model.infer(&gradient(50, 37)).unwrap();
        assert_eq!((map.width, map.height), (50, 37));
        assert_eq!(map.kind, DepthKind::Metric);
        assert_eq!(map.data.len(), 50 * 37);
        assert!(map.data.iter().all(|d| d.is_finite() && (cfg.min_depth..=cfg.max_depth).contains(d)), "depth outside the bins");
        assert_eq!(model.mode(), "tipsv2-b14-dpt");
        assert_eq!(model.max_depth_m(), Some(10.0));
        assert_eq!(model.device(), "CPU");
    }

    #[test]
    fn every_depth_path_tensor_of_the_checkpoint_is_read() {
        let (_model, shapes, requested) = header_model(Fill::Zero);
        let requested = requested.lock().unwrap();
        let expected: BTreeSet<String> = shapes.keys().filter(|n| !UNUSED_AT_INFERENCE.contains(&n.as_str())).cloned().collect();
        let unread: Vec<&String> = expected.difference(&requested).collect();
        assert!(unread.is_empty(), "checkpoint tensors the port ignores: {unread:?}");
        assert!(requested.iter().all(|n| expected.contains(n)), "the port read a tensor it documents as unused");
        assert_eq!(requested.len(), 175 + 52, "176 encoder tensors minus mask_token, 52 depth-head tensors");
    }

    fn full_model(fill: Fill) -> (TipsDepth, Shapes, Requested) {
        let shapes = checkpoint_shapes();
        let requested = Arc::new(Mutex::new(BTreeSet::new()));
        let backend = HeaderBackend { shapes: shapes.clone(), fill, requested: requested.clone() };
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
        let model = TipsDepth::from_var_builder_with(TipsConfig::default(), vb, &Device::Cpu, SceneHeads::ALL).unwrap();
        (model, shapes, requested)
    }

    fn mirror(rgb: &Rgb) -> Rgb {
        let mut data = Vec::with_capacity(rgb.data.len());
        for y in 0..rgb.height {
            for x in (0..rgb.width).rev() {
                data.extend_from_slice(&rgb.px(x, y));
            }
        }
        Rgb::new(rgb.width, rgb.height, data)
    }

    #[test]
    fn the_normals_and_segmentation_heads_read_every_tensor_and_share_the_encoder() {
        let (model, shapes, requested) = full_model(Fill::Zero);
        assert_eq!(model.heads(), SceneHeads::ALL);
        let requested = requested.lock().unwrap();
        let expected: BTreeSet<String> = shapes.keys().filter(|n| !UNUSED_AT_INFERENCE.contains(&n.as_str())).cloned().collect();
        let unread: Vec<&String> = expected.difference(&requested).collect();
        assert!(unread.is_empty(), "checkpoint tensors the port ignores: {unread:?}");
        assert_eq!(requested.len(), 175 + 3 * 52, "one encoder, three DPT decoders of 52 tensors");
        assert_eq!(model.config().seg_classes, ADE_CLASSES);
        assert_eq!(ade_name(ADE_SKY), "sky");
        assert_eq!((ade_name(ADE_TREE), ade_name(ADE_ROCK), ade_name(ADE_BOX)), ("tree", "rock", "box"));
    }

    #[test]
    fn a_scene_has_unit_normals_ade_labels_and_no_depth_on_the_sky() {
        let (mut model, _, _) = full_model(Fill::Random);
        model.input_grid = InputGrid::Fixed { cols: 4, rows: 3 };
        let s = model.scene_batch(&[&gradient(50, 37)], SceneHeads::ALL).unwrap().pop().unwrap();
        let n = s.normals.as_ref().expect("normals");
        assert!(n.data.iter().all(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0).abs() < 1e-3), "unit normals");
        let seg = s.segmentation.as_ref().expect("segmentation");
        assert_eq!((seg.width, seg.height), (n.width, n.height), "both at the decoder's resolution");
        assert!(seg.labels.iter().all(|&l| (l as usize) < ADE_CLASSES) && seg.sky.iter().all(|p| (0.0..=1.0).contains(p)));
        for y in 0..37 {
            for x in 0..50 {
                let (sx, sy) = ((x * seg.width / 50).min(seg.width - 1), (y * seg.height / 37).min(seg.height - 1));
                assert_eq!(seg.is_sky((sy * seg.width + sx) as usize), s.depth.at(x, y).is_infinite(), "sky pixels and only those lose their depth");
            }
        }
        let only = model.scene_batch(&[&gradient(50, 37)], SceneHeads::NONE).unwrap().pop().unwrap();
        assert!(only.normals.is_none() && only.segmentation.is_none(), "heads run only when asked");
    }

    #[test]
    fn flip_averaging_is_mirror_symmetric() {
        let (mut model, _, _) = full_model(Fill::Random);
        model.input_grid = InputGrid::Fixed { cols: 4, rows: 3 };
        model.flip_tta = true;
        let img = gradient(56, 42);
        let a = model.scene_batch(&[&img], SceneHeads::ALL).unwrap().pop().unwrap();
        let b = model.scene_batch(&[&mirror(&img)], SceneHeads::ALL).unwrap().pop().unwrap();
        for y in 0..42 {
            for x in 0..56 {
                let (da, db) = (a.depth.at(55 - x, y), b.depth.at(x, y));
                assert!(da.is_infinite() == db.is_infinite() && (!da.is_finite() || (da - db).abs() <= 1e-4 * da.max(1.0)), "depth at {x},{y}: {da} vs {db}");
            }
        }
        let (na, nb) = (a.normals.unwrap(), b.normals.unwrap());
        for y in 0..na.height {
            for x in 0..na.width {
                let (p, q) = (na.at(na.width - 1 - x, y), nb.at(x, y));
                assert!((p[0] + q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4 && (p[2] - q[2]).abs() < 1e-4, "normal at {x},{y}: {p:?} vs {q:?}");
            }
        }
        let (sa, sb) = (a.segmentation.unwrap(), b.segmentation.unwrap());
        for y in 0..sa.height {
            for x in 0..sa.width {
                assert!((sa.sky[(y * sa.width + sa.width - 1 - x) as usize] - sb.sky[(y * sb.width + x) as usize]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn bins_decode_to_metres() {
        let cfg = TipsConfig::default();
        let (m, n) = (cfg.min_depth as f64, cfg.depth_bins);
        let centers = bin_centers(cfg.min_depth, cfg.max_depth, n, &Device::Cpu).unwrap();
        let c = centers.to_vec1::<f32>().unwrap();
        assert!((c[0] as f64 - 0.001).abs() < 1e-7 && (c[n - 1] as f64 - 10.0).abs() < 1e-6);
        let sum_c: f64 = c.iter().map(|&x| x as f64).sum();

        let (big, k) = (1000.0f32, 200usize);
        let mut logits = vec![-3.0f32; n];
        let mut peaked = vec![0.0f32; n];
        peaked[k] = big;
        logits.extend(peaked);
        let t = Tensor::from_vec(logits, (1, 2, n), &Device::Cpu).unwrap();
        let depth = decode_bins(&t, &centers, m).unwrap().to_vec2::<f32>().unwrap();

        let uniform = sum_c / n as f64;
        assert!((uniform - (0.001 + 10.0) / 2.0).abs() < 1e-5);
        assert!((depth[0][0] as f64 - uniform).abs() < 1e-4, "{} vs {uniform}", depth[0][0]);
        let expected = (m * sum_c + big as f64 * c[k] as f64) / (n as f64 * m + big as f64);
        assert!((depth[0][1] as f64 - expected).abs() < 1e-4, "{} vs {expected}", depth[0][1]);
        assert!((depth[0][1] - c[k]).abs() < 0.01, "a confident bin dominates");
    }

    #[test]
    fn position_resampling_matches_torch_antialias() {
        let w = |src, dst| antialias_bilinear_weights(src, dst);
        assert_eq!(w(3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(w(2, 4), vec![1.0, 0.0, 0.75, 0.25, 0.25, 0.75, 0.0, 1.0]);
        let down = w(4, 2);
        let expect = [3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0, 0.0, 0.0, 1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0];
        assert!(down.iter().zip(expect).all(|(a, b)| (a - b).abs() < 1e-6), "{down:?}");
        for row in w(32, 24).chunks(32) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn default_grid_turns_keyframes_into_504_by_336() {
        assert_eq!(InputGrid::default().grid_for(384, 256), (36, 24));
        assert_eq!(InputGrid::ShortSide(24).grid_for(256, 384), (24, 36));
        assert_eq!(InputGrid::Fixed { cols: 4, rows: 3 }.grid_for(1000, 10), (4, 3));
    }

    #[test]
    fn the_pyramid_grid_is_twice_siglip2s_and_matches_the_default_on_keyframes() {
        let pyr = InputGrid::pyramid(&crate::pyramid::PyramidCfg::default());
        for (w, h) in [(384, 256), (640, 480), (256, 384)] {
            let (gh, gw) = noto_ml::preprocess::pyramid_grid(w, h, 12, 256);
            assert_eq!(pyr.grid_for(w, h), (2 * gw, 2 * gh), "{w}x{h}");
            assert_eq!(pyr.grid_for(w, h), InputGrid::default().grid_for(w, h), "{w}x{h}: TIPSv2 runs on its own grid");
        }
        assert_eq!(pyr.grid_for(500, 500), (24, 24));
        assert_eq!(InputGrid::pyramid(&crate::pyramid::PyramidCfg::fast()).grid_for(384, 256), (18, 12));
        let (mut model, _, _) = header_model(Fill::Random);
        model.input_grid = InputGrid::Fixed { cols: 4, rows: 3 };
        assert_eq!(model.grid_for(384, 256), (4, 3));
        model.set_pyramid(Some(&crate::pyramid::PyramidCfg::fast()));
        assert_eq!(model.grid_for(384, 256), (18, 12), "the pyramid overrides the grid while it is set");
        model.set_pyramid(None);
        assert_eq!(model.grid_for(384, 256), (4, 3), "and gives the old one back");
    }

    #[test]
    fn config_reads_python_names_and_backbone_stages() {
        let c = TipsConfig::from_json(r#"{"out_indices": [3, 6, 9, 12], "num_depth_bins": 128, "img_size": 448}"#).unwrap();
        assert_eq!(c.tap_blocks, [2, 5, 8, 11]);
        assert_eq!(c.depth_bins, 128);
        assert_eq!(c.hidden, 768);
        assert!(TipsConfig::from_json(r#"{"vision_fn": "vit_large"}"#).is_err());
        assert!(TipsConfig::from_json(r#"{"out_indices": [0, 6, 9, 12]}"#).is_err());
    }

    #[test]
    fn missing_weights_name_the_folder_and_files() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/no-such-model");
        let err = TipsDepth::load(&dir, &Device::Cpu).err().unwrap().to_string();
        assert!(err.contains("model.safetensors") && err.contains("no-such-model") && err.contains("TIPS_DIR"), "{err}");
    }
}
