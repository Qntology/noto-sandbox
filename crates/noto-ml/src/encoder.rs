use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};

use crate::quant::{Family, QLinear, QWeights};

/// The query, key and value projections: three layers (the checkpoint's own), or one over the stacked rows (fused,
/// one GEMM instead of three; the rows are the same, the sums may round in another order).
enum Qkv {
    Split { q: QLinear, k: QLinear, v: QLinear },
    Fused(QLinear),
}

pub struct Attention {
    qkv: Qkv,
    out: QLinear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_weights(hidden, heads, &QWeights::plain(vb, Family::Siglip2), false)
    }

    pub fn with_weights(hidden: usize, heads: usize, w: &QWeights, fused_qkv: bool) -> Result<Self> {
        let qkv = if fused_qkv {
            Qkv::Fused(w.linear_concat(&["q_proj", "k_proj", "v_proj"], hidden, hidden).map_err(to_candle)?)
        } else {
            Qkv::Split {
                q: w.pp("q_proj").linear(hidden, hidden).map_err(to_candle)?,
                k: w.pp("k_proj").linear(hidden, hidden).map_err(to_candle)?,
                v: w.pp("v_proj").linear(hidden, hidden).map_err(to_candle)?,
            }
        };
        Ok(Attention { qkv, out: w.pp("out_proj").linear(hidden, hidden).map_err(to_candle)?, heads, head_dim: hidden / heads })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = match &self.qkv {
            Qkv::Split { q, k, v } => (split(q.forward(x)?)?, split(k.forward(x)?)?, split(v.forward(x)?)?),
            Qkv::Fused(qkv) => {
                let all = qkv.forward(x)?;
                let d = self.heads * self.head_dim;
                (split(all.narrow(2, 0, d)?)?, split(all.narrow(2, d, d)?)?, split(all.narrow(2, 2 * d, d)?)?)
            }
        };

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, self.heads * self.head_dim))?;
        self.out.forward(&ctx)
    }
}

pub struct Mlp {
    fc1: QLinear,
    fc2: QLinear,
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_weights(hidden, intermediate, &QWeights::plain(vb, Family::Siglip2))
    }

    pub fn with_weights(hidden: usize, intermediate: usize, w: &QWeights) -> Result<Self> {
        Ok(Mlp {
            fc1: w.pp("fc1").linear(hidden, intermediate).map_err(to_candle)?,
            fc2: w.pp("fc2").linear(intermediate, hidden).map_err(to_candle)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

pub struct EncoderLayer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        heads: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::with_weights(hidden, intermediate, heads, eps, &QWeights::plain(vb, Family::Siglip2), false)
    }

    pub fn with_weights(hidden: usize, intermediate: usize, heads: usize, eps: f64, w: &QWeights, fused_qkv: bool) -> Result<Self> {
        let vb = w.vb();
        Ok(EncoderLayer {
            ln1: layer_norm(hidden, eps, vb.pp("layer_norm1"))?,
            attn: Attention::with_weights(hidden, heads, &w.pp("self_attn"), fused_qkv)?,
            ln2: layer_norm(hidden, eps, vb.pp("layer_norm2"))?,
            mlp: Mlp::with_weights(hidden, intermediate, &w.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.ln1.forward(x)?, mask)?)?;
        &h + self.mlp.forward(&self.ln2.forward(&h)?)?
    }
}

pub(crate) fn to_candle(e: anyhow::Error) -> candle_core::Error {
    candle_core::Error::Msg(format!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn block_keeps_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let layer = EncoderLayer::new(16, 32, 4, 1e-6, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 5, 16), &dev).unwrap();
        let y = layer.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn masked_tokens_get_no_attention() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let attn = Attention::new(8, 2, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 3, 8), &dev).unwrap();
        let mask = Tensor::from_vec(vec![0f32, 0.0, -1e9], (1, 1, 1, 3), &dev).unwrap();
        let full = attn.forward(&x, Some(&mask)).unwrap();
        let mut changed = x.to_vec3::<f32>().unwrap();
        changed[0][2] = vec![9.0; 8];
        let x2 = Tensor::new(changed, &dev).unwrap();
        let full2 = attn.forward(&x2, Some(&mask)).unwrap();
        let a = full.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
        let b = full2.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
        for (ra, rb) in a[0].iter().zip(&b[0]) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_fused_projection_attends_like_the_split_one() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let split = Attention::new(16, 4, vb.clone()).unwrap();
        // The VarMap's weights start at zero: give them values so the comparison means something.
        for (name, var) in vm.data().lock().unwrap().iter() {
            let seed = name.len() as f64;
            var.set(&Tensor::randn(0f32, 0.3, var.dims().to_vec(), &dev).unwrap().affine(1.0, seed * 1e-3).unwrap()).unwrap();
        }
        let split = { drop(split); Attention::new(16, 4, vb.clone()).unwrap() };
        let fused = Attention::with_weights(16, 4, &QWeights::plain(vb, Family::Siglip2), true).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 5, 16), &dev).unwrap();
        let (a, b) = (split.forward(&x, None).unwrap(), fused.forward(&x, None).unwrap());
        let (a, b) = (a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert!(a.iter().any(|v| v.abs() > 1e-3), "the layer reads its input");
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}
