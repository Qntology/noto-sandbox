use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{Device, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::config::Siglip2Config;
use crate::encoder::{to_candle, EncoderLayer, LayerKv, Mlp};
use crate::preprocess::NaflexInput;
use crate::quant::{Family, QLinear, QWeights};

/// MapHead's packed `in_proj` (query, key and value rows). F32: the checkpoint's `[3D, D]` weight sliced per call as
/// before (bit for bit the old head); quantized: one [`QLinear`] per slice.
enum InProj {
    Raw { w: Tensor, b: Tensor, d: usize },
    Q([QLinear; 3]),
}

impl InProj {
    fn new(w: &QWeights, d: usize) -> Result<InProj> {
        if w.qtype("in_proj_weight", d) == crate::quant::QType::F32 {
            w.record_dense(3 * d, d);
            return Ok(InProj::Raw { w: w.vb().get((3 * d, d), "in_proj_weight")?, b: w.vb().get(3 * d, "in_proj_bias")?, d });
        }
        let part = |i: usize| w.linear_rows("in_proj_weight", "in_proj_bias", 3 * d, i * d, d, d).map_err(to_candle);
        Ok(InProj::Q([part(0)?, part(1)?, part(2)?]))
    }

    /// Projection `i` (0 query, 1 key, 2 value) of `x`.
    fn forward(&self, i: usize, x: &Tensor) -> Result<Tensor> {
        match self {
            InProj::Raw { w, b, d } => x.broadcast_matmul(&w.narrow(0, i * d, *d)?.t()?.contiguous()?)?.broadcast_add(&b.narrow(0, i * d, *d)?),
            InProj::Q(l) => l[i].forward(x),
        }
    }
}

pub struct MapHead {
    probe: Tensor,
    in_proj: InProj,
    out: QLinear,
    ln: LayerNorm,
    mlp: Mlp,
    heads: usize,
    hidden: usize,
}

impl MapHead {
    fn new(cfg: &Siglip2Config, w: &QWeights) -> Result<Self> {
        let d = cfg.vision_hidden;
        let attn = w.pp("attention");
        Ok(MapHead {
            probe: w.vb().get((1, 1, d), "probe")?,
            in_proj: InProj::new(&attn, d)?,
            out: attn.pp("out_proj").linear(d, d).map_err(to_candle)?,
            ln: layer_norm(d, cfg.vision_eps, w.vb().pp("layernorm"))?,
            mlp: Mlp::with_weights(d, cfg.vision_intermediate, &w.pp("mlp"))?,
            heads: cfg.vision_heads,
            hidden: d,
        })
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let d = self.hidden;
        let (b, n, _) = tokens.dims3()?;
        let hd = d / self.heads;

        // Contiguous: from a broadcast batch Metal's matmul gets every row after the first wrong.
        let probe = self.probe.broadcast_as((b, 1, d))?.contiguous()?;
        let q = self.in_proj.forward(0, &probe)?;
        let k = self.in_proj.forward(1, tokens)?;
        let v = self.in_proj.forward(2, tokens)?;

        let split = |t: Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.heads, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = split(q, 1)?;
        let k = split(k, n)?;
        let v = split(v, n)?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine((hd as f64).powf(-0.5), 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, 1, d))?;

        let attended = self.out.forward(&ctx)?;
        let normed = self.ln.forward(&attended)?;
        let pooled = (&attended + self.mlp.forward(&normed)?)?;
        pooled.reshape((b, d))
    }

    fn dense(&self, tokens: &Tensor) -> Result<Tensor> {
        let v = self.in_proj.forward(2, tokens)?;
        let attended = self.out.forward(&v)?;
        let normed = self.ln.forward(&attended)?;
        &attended + self.mlp.forward(&normed)?
    }

    fn attention(&self, tokens: &Tensor) -> Result<Tensor> {
        let d = self.hidden;
        let (b, n, _) = tokens.dims3()?;
        let hd = d / self.heads;
        let probe = self.probe.broadcast_as((b, 1, d))?.contiguous()?;
        let q = self.in_proj.forward(0, &probe)?;
        let k = self.in_proj.forward(1, tokens)?;
        let q = q.reshape((b, 1, self.heads, hd))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, n, self.heads, hd))?.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine((hd as f64).powf(-0.5), 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.mean(1)?.reshape((b, n))
    }
}

pub struct VisionTower {
    patch_proj: Linear,
    position: Tensor,
    layers: Vec<EncoderLayer>,
    post_ln: LayerNorm,
    head: MapHead,
    side: usize,
    hidden: usize,
    device: Device,
    pos_antialias: bool,
    /// The position table resampled to each grid the tower has seen (the frames of a run share a few grids), so a
    /// keyframe no longer repeats the two resampling matmuls. Same values, computed once.
    pos_cache: Mutex<HashMap<(usize, usize, bool), Tensor>>,
}

pub fn antialias_bilinear_weights(src: usize, dst: usize) -> Vec<f32> {
    let scale = src as f64 / dst as f64;
    let (support, inv_scale) = if scale >= 1.0 { (scale, 1.0 / scale) } else { (1.0, 1.0) };
    let triangle = |x: f64| (1.0 - x.abs()).max(0.0);
    let mut weights = vec![0f32; dst * src];
    for o in 0..dst {
        let center = scale * (o as f64 + 0.5);
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as i64).min(src as i64).max(lo as i64) as usize;
        let taps: Vec<f64> = (lo..hi).map(|j| triangle((j as f64 - center + 0.5) * inv_scale)).collect();
        let total: f64 = taps.iter().sum();
        if total > 0.0 {
            for (k, t) in taps.iter().enumerate() {
                weights[o * src + lo + k] = (t / total) as f32;
            }
        }
    }
    weights
}

impl VisionTower {
    pub fn new(cfg: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        Self::with_weights(cfg, &QWeights::plain(vb, Family::Siglip2), false)
    }

    pub fn with_weights(cfg: &Siglip2Config, w: &QWeights, fused_qkv: bool) -> Result<Self> {
        let d = cfg.vision_hidden;
        let vb = w.vb();
        let emb = vb.pp("embeddings");
        let patch_dim = cfg.patch_size * cfg.patch_size * 3;
        let mut layers = Vec::with_capacity(cfg.vision_layers);
        for i in 0..cfg.vision_layers {
            layers.push(EncoderLayer::with_weights(d, cfg.vision_intermediate, cfg.vision_heads, cfg.vision_eps, &w.pp("encoder").pp(format!("layers.{i}")), fused_qkv)?);
        }
        Ok(VisionTower {
            patch_proj: linear(patch_dim, d, emb.pp("patch_embedding"))?,
            position: emb
                .pp("position_embedding")
                .get((cfg.max_patches, d), "weight")?,
            layers,
            post_ln: layer_norm(d, cfg.vision_eps, vb.pp("post_layernorm"))?,
            head: MapHead::new(cfg, &w.pp("head"))?,
            side: cfg.position_side(),
            hidden: d,
            device: vb.device().clone(),
            pos_antialias: true,
            pos_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn set_position_antialias(&mut self, on: bool) {
        self.pos_antialias = on;
    }

    pub fn position_antialias(&self) -> bool {
        self.pos_antialias
    }

    fn positions(&self, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let key = (grid_h, grid_w, self.pos_antialias);
        if let Some(t) = self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(t.clone());
        }
        let t = self.resample_positions(grid_h, grid_w)?;
        self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, t.clone());
        Ok(t)
    }

    fn resample_positions(&self, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let d = self.hidden;
        let table = self.position.reshape((self.side, self.side, d))?;
        if grid_h == self.side && grid_w == self.side {
            return table.reshape((1, grid_h * grid_w, d));
        }
        if self.pos_antialias {
            let side = self.side;
            let wy = Tensor::from_vec(antialias_bilinear_weights(side, grid_h), (grid_h, side), &self.device)?;
            let wx = Tensor::from_vec(antialias_bilinear_weights(side, grid_w), (grid_w, side), &self.device)?;
            let by_rows = wy.matmul(&table.reshape((side, side * d))?)?.reshape((grid_h, side, d))?;
            return wx.broadcast_matmul(&by_rows)?.reshape((1, grid_h * grid_w, d));
        }
        table
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .upsample_bilinear2d(grid_h, grid_w, false)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((1, grid_h * grid_w, d))
    }

    fn tokens(&self, input: &NaflexInput) -> Result<Tensor> {
        let (gh, gw, pd) = (input.grid_h, input.grid_w, input.patch_dim);
        let patches = Tensor::from_slice(&input.patches, (1, gh * gw, pd), &self.device)?;
        let mut x = self.patch_proj.forward(&patches)?;
        x = x.broadcast_add(&self.positions(gh, gw)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, None)?;
        }
        self.post_ln.forward(&x)
    }

    pub fn forward_dense(&self, input: &NaflexInput) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.tokens(input)?;
        let dense = self.head.dense(&x)?.squeeze(0)?;
        let attn = self.head.attention(&x)?.squeeze(0)?;
        let pooled = self.head.forward(&x)?.squeeze(0)?;
        Ok((dense, attn, pooled))
    }

    pub fn forward(&self, input: &NaflexInput) -> Result<Tensor> {
        self.forward_batch(&[input])
    }

    pub fn forward_dense_batch(&self, inputs: &[&NaflexInput]) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.batch_tokens(inputs)?;
        Ok((self.head.dense(&x)?, self.head.attention(&x)?, self.head.forward(&x)?))
    }

    fn batch_tokens(&self, inputs: &[&NaflexInput]) -> Result<Tensor> {
        let first = match inputs.first() {
            Some(f) => f,
            None => candle_core::bail!("a vision batch needs at least one image"),
        };
        let (gh, gw, pd) = (first.grid_h, first.grid_w, first.patch_dim);
        let n = gh * gw;
        let mut flat: Vec<f32> = Vec::with_capacity(inputs.len() * n * pd);
        for i in inputs {
            if (i.grid_h, i.grid_w, i.patch_dim) != (gh, gw, pd) {
                candle_core::bail!("batched images must share one patch grid ({gh}x{gw}), got {}x{}", i.grid_h, i.grid_w);
            }
            flat.extend_from_slice(&i.patches);
        }
        let patches = Tensor::from_vec(flat, (inputs.len(), n, pd), &self.device)?;
        let mut x = self.patch_proj.forward(&patches)?;
        x = x.broadcast_add(&self.positions(gh, gw)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, None)?;
        }
        self.post_ln.forward(&x)
    }

    pub fn forward_batch(&self, inputs: &[&NaflexInput]) -> Result<Tensor> {
        self.head.forward(&self.batch_tokens(inputs)?)
    }

    /// The patches of `input` at `cells` (row-major grid cells) through the patch projection, each with its own cell's
    /// position embedding.
    fn cell_tokens(&self, input: &NaflexInput, cells: &[u32]) -> Result<Tensor> {
        let (gh, gw, pd) = (input.grid_h, input.grid_w, input.patch_dim);
        let n = gh * gw;
        let mut flat = Vec::with_capacity(cells.len() * pd);
        for &c in cells {
            let c = c as usize;
            if c >= n {
                candle_core::bail!("cell {c} is outside the {gh}x{gw} grid");
            }
            flat.extend_from_slice(&input.patches[c * pd..(c + 1) * pd]);
        }
        let patches = Tensor::from_vec(flat, (1, cells.len(), pd), &self.device)?;
        let rows = Tensor::from_slice(cells, cells.len(), &self.device)?;
        let pos = self.positions(gh, gw)?.index_select(&rows, 1)?;
        self.patch_proj.forward(&patches)?.broadcast_add(&pos)
    }

    /// The encoder over some of the grid's patches only: the tokens of `cells` attend to each other and nothing else
    /// (the holes are not black patches, they are absent). With every cell in grid order this is `forward_dense`'s pass.
    pub fn forward_sparse(&self, input: &NaflexInput, cells: &[u32]) -> Result<SparsePass> {
        let mut x = self.cell_tokens(input, cells)?;
        let mut kv = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (y, layer_kv) = layer.forward_kv(&x, None)?;
            x = y;
            kv.push(layer_kv);
        }
        Ok(SparsePass { cells: cells.to_vec(), tokens: self.post_ln.forward(&x)?, kv })
    }

    /// Opens more cells on top of `first`: the tokens of `cells` attend, at every layer, to each other and to `first`'s
    /// tokens (their keys and values as `first` computed them). A cell of `first` listed again is computed anew (with the
    /// opened ones around it) and leaves the context. `first`'s other tokens keep their features.
    pub fn extend_sparse(&self, input: &NaflexInput, first: &SparsePass, cells: &[u32]) -> Result<SparsePass> {
        let again: std::collections::HashSet<u32> = cells.iter().copied().collect();
        let keep: Vec<u32> = first.cells.iter().enumerate().filter(|(_, c)| !again.contains(c)).map(|(i, _)| i as u32).collect();
        let context: Option<Vec<LayerKv>> = if keep.len() == first.cells.len() {
            None
        } else if keep.is_empty() {
            Some(Vec::new())
        } else {
            let rows = Tensor::from_slice(&keep, keep.len(), &self.device)?;
            Some(first.kv.iter().map(|kv| kv.select(&rows)).collect::<Result<_>>()?)
        };
        let mut x = self.cell_tokens(input, cells)?;
        let mut kv = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let ctx = match &context {
                None => Some(&first.kv[i]),
                Some(v) => v.get(i),
            };
            let (y, layer_kv) = layer.forward_kv(&x, ctx)?;
            x = y;
            kv.push(layer_kv);
        }
        Ok(SparsePass { cells: cells.to_vec(), tokens: self.post_ln.forward(&x)?, kv })
    }

    /// `first` with `second` (an `extend_sparse` of it) folded in: the cells `second` computed anew take their new
    /// tokens, keys and values, so a further `extend_sparse` attends to every cell opened so far.
    pub fn merge_sparse(&self, first: &SparsePass, second: &SparsePass) -> Result<SparsePass> {
        let again: std::collections::HashSet<u32> = second.cells.iter().copied().collect();
        let keep: Vec<u32> = first.cells.iter().enumerate().filter(|(_, c)| !again.contains(c)).map(|(i, _)| i as u32).collect();
        let mut cells: Vec<u32> = keep.iter().map(|&i| first.cells[i as usize]).collect();
        cells.extend_from_slice(&second.cells);
        if keep.is_empty() {
            return Ok(SparsePass { cells, tokens: second.tokens.clone(), kv: second.kv.clone() });
        }
        let rows = Tensor::from_slice(&keep, keep.len(), &self.device)?;
        let tokens = Tensor::cat(&[&first.tokens.index_select(&rows, 1)?, &second.tokens], 1)?;
        let kv = first
            .kv
            .iter()
            .zip(&second.kv)
            .map(|(a, b)| {
                let a = a.select(&rows)?;
                Ok(LayerKv { k: Tensor::cat(&[&a.k, &b.k], 2)?, v: Tensor::cat(&[&a.v, &b.v], 2)? })
            })
            .collect::<Result<_>>()?;
        Ok(SparsePass { cells, tokens, kv })
    }

    /// The head's per-token value path (dense features) on some tokens (1, n, d).
    pub fn head_dense(&self, tokens: &Tensor) -> Result<Tensor> {
        self.head.dense(tokens)
    }

    /// The head's pooled embedding of some tokens (1, n, d).
    pub fn head_pooled(&self, tokens: &Tensor) -> Result<Tensor> {
        self.head.forward(tokens)
    }
}

/// An encoder pass over some cells of a patch grid (`VisionTower::forward_sparse` / `extend_sparse`): the cells, their
/// tokens after the last layer (1, cells, hidden), and every layer's keys and values for a later pass to attend to.
#[derive(Clone, Debug)]
pub struct SparsePass {
    pub cells: Vec<u32>,
    pub tokens: Tensor,
    kv: Vec<LayerKv>,
}
