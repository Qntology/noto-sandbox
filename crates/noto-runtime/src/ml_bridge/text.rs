use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, Module, VarBuilder};

use super::config::Siglip2Config;
use super::encoder::{to_candle, EncoderLayer, FastLayerNorm};
use super::quant::{Family, QLinear, QWeights};
use super::vocab::{TokenTable, VocabStats};

pub struct TextTower {
    tokens: TokenTable,
    position: Tensor,
    layers: Vec<EncoderLayer>,
    final_ln: FastLayerNorm,
    head: QLinear,
    max_len: usize,
    hidden: usize,
    device: Device,
}

impl TextTower {
    pub fn new(cfg: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        Self::with_weights(cfg, &QWeights::plain(vb, Family::Siglip2), None, false)
    }

    /// `tokens`: the token table to use (`None`: the whole table from `w`, on the device).
    pub fn with_weights(cfg: &Siglip2Config, w: &QWeights, tokens: Option<TokenTable>, fused_qkv: bool) -> Result<Self> {
        let d = cfg.text_hidden;
        let vb = w.vb();
        let emb = vb.pp("embeddings");
        let mut layers = Vec::with_capacity(cfg.text_layers);
        for i in 0..cfg.text_layers {
            layers.push(EncoderLayer::with_weights(d, cfg.text_intermediate, cfg.text_heads, cfg.text_eps, &w.pp("encoder").pp(format!("layers.{i}")), fused_qkv)?);
        }
        let tokens = match tokens {
            Some(t) => t,
            None => TokenTable::full(embedding(cfg.text_vocab, d, emb.pp("token_embedding"))?, cfg.text_vocab),
        };
        Ok(TextTower {
            tokens,
            position: emb
                .pp("position_embedding")
                .get((cfg.text_max_len, d), "weight")?,
            layers,
            final_ln: FastLayerNorm::new(d, cfg.text_eps, vb.pp("final_layer_norm"), w.is_cpu_packed())?,
            head: w.pp("head").linear(d, d).map_err(to_candle)?,
            max_len: cfg.text_max_len,
            hidden: d,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, len) = ids.dims2()?;
        let ids: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        self.forward_ids(&ids, (b, len), mask)
    }

    /// `ids` row-major in `shape` (rows, length), `mask` on the device.
    pub fn forward_ids(&self, ids: &[u32], shape: (usize, usize), mask: &Tensor) -> Result<Tensor> {
        let (b, len) = shape;
        let pos = self.position.narrow(0, 0, len)?.unsqueeze(0)?;
        let tokens = self.tokens.lookup(ids, shape, &self.device).map_err(to_candle)?;
        let mut x = tokens.broadcast_add(&pos)?;

        let additive = mask
            .affine(-1.0, 1.0)?
            .affine(-1e9, 0.0)?
            .reshape((b, 1, 1, len))?;
        for layer in &self.layers {
            x = layer.forward(&x, Some(&additive))?;
        }
        let x = self.final_ln.forward(&x)?;
        let pooled = x.narrow(1, len - 1, 1)?.reshape((b, self.hidden))?;
        self.head.forward(&pooled)
    }

    /// Puts the rows of `ids` on the device ahead of use (an active token table; nothing for the whole one).
    pub fn prime(&self, ids: &[u32]) -> Result<usize> {
        self.tokens.prime(ids).map_err(to_candle)
    }

    pub fn vocab_stats(&self) -> VocabStats {
        self.tokens.stats()
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
