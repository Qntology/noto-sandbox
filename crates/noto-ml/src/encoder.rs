use candle_core::{DType, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};

use crate::device::{accel_op_ok, AccelOp};
use crate::quant::{Family, QLinear, QWeights};

/// LayerNorm with an optional cached CPU 1-pass SIMD fastpath (active when `QWeights::is_cpu_packed()`).
pub struct FastLayerNorm {
    inner: LayerNorm,
    cpu: Option<(Vec<f32>, Option<Vec<f32>>, f32)>,
}

impl FastLayerNorm {
    pub fn new(hidden: usize, eps: f64, vb: VarBuilder, cpu_packed: bool) -> Result<Self> {
        let inner = layer_norm(hidden, eps, vb)?;
        let cpu = if cpu_packed && inner.weight().device().is_cpu() && inner.weight().dtype() == DType::F32 {
            let w = inner.weight().flatten_all()?.to_vec1::<f32>()?;
            let b = match inner.bias() {
                Some(bt) => Some(bt.flatten_all()?.to_vec1::<f32>()?),
                None => None,
            };
            Some((w, b, eps as f32))
        } else {
            None
        };
        Ok(FastLayerNorm { inner, cpu })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some((w, b, eps)) = &self.cpu {
            if x.device().is_cpu() && x.dtype() == DType::F32 {
                let dims = x.dims();
                let h = w.len();
                if !dims.is_empty() && dims[dims.len() - 1] == h {
                    let x_c = x.contiguous()?;
                    let (storage, layout) = x_c.storage_and_layout();
                    if let candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) = &*storage {
                        let start = layout.start_offset();
                        let total = x.elem_count();
                        let mut out = vec![0.0f32; total];
                        noto_cpu_kernels::layer_norm(&v[start..start + total], h, w, b.as_deref(), *eps, &mut out);
                        return Tensor::from_vec(out, dims, x.device());
                    }
                }
            }
        }
        self.inner.forward(x)
    }
}

pub(crate) fn fast_gelu_tanh(x: &Tensor, cpu_packed: bool) -> Result<Tensor> {
    if cpu_packed && x.device().is_cpu() && x.dtype() == DType::F32 {
        let dims = x.dims();
        let x_c = x.contiguous()?;
        let (storage, layout) = x_c.storage_and_layout();
        if let candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) = &*storage {
            let start = layout.start_offset();
            let total = x.elem_count();
            let mut out = v[start..start + total].to_vec();
            noto_cpu_kernels::gelu_tanh_inplace(&mut out);
            return Tensor::from_vec(out, dims, x.device());
        }
    }
    x.gelu()
}

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
    cpu_packed: bool,
}

/// One layer's keys and values for some tokens, (batch, heads, tokens, head_dim): what the tokens of a later pass attend
/// to besides each other.
#[derive(Clone, Debug)]
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
}

impl LayerKv {
    /// The tokens at `rows` (a u32 index tensor) of these keys and values.
    pub fn select(&self, rows: &Tensor) -> Result<LayerKv> {
        Ok(LayerKv { k: self.k.index_select(rows, 2)?, v: self.v.index_select(rows, 2)? })
    }
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
        Ok(Attention {
            qkv,
            out: w.pp("out_proj").linear(hidden, hidden).map_err(to_candle)?,
            heads,
            head_dim: hidden / heads,
            cpu_packed: w.is_cpu_packed(),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        Ok(self.forward_inner(x, mask, None, false)?.0)
    }

    /// `forward`, also returning this pass's keys and values. With `context` the queries attend to those keys and values
    /// too (tokens of an earlier pass, placed before this pass's own); without it this is `forward`, the same operations.
    pub fn forward_kv(&self, x: &Tensor, mask: Option<&Tensor>, context: Option<&LayerKv>) -> Result<(Tensor, LayerKv)> {
        let (out, kv) = self.forward_inner(x, mask, context, true)?;
        Ok((out, kv.expect("need_kv=true returns LayerKv")))
    }

    fn forward_inner(&self, x: &Tensor, mask: Option<&Tensor>, context: Option<&LayerKv>, need_kv: bool) -> Result<(Tensor, Option<LayerKv>)> {
        if mask.is_some() && context.is_some() {
            candle_core::bail!("an attention mask does not cover an earlier pass's tokens");
        }
        let (b, n, _) = x.dims3()?;
        let d = self.heads * self.head_dim;
        let scale = (self.head_dim as f64).powf(-0.5);

        // Compute linear Q, K, V in (b, n, d) first so zero-transpose paths (CPU streaming & GPU FlashAttn) avoid copies.
        let (q_lin, k_lin, v_lin) = match &self.qkv {
            Qkv::Split { q, k, v } => (q.forward(x)?, k.forward(x)?, v.forward(x)?),
            Qkv::Fused(qkv) => {
                let all = qkv.forward(x)?.reshape((b, n, 3, d))?;
                (
                    all.narrow(2, 0, 1)?.squeeze(2)?,
                    all.narrow(2, 1, 1)?.squeeze(2)?,
                    all.narrow(2, 2, 1)?.squeeze(2)?,
                )
            }
        };

        // Fastpath 1: CPU packed streaming 64-key panel attention (0 score matrix allocs, 0 transposes when context=None).
        if self.cpu_packed && x.device().is_cpu() && x.dtype() == DType::F32 {
            let (k_src, v_src, seq_kv, kv_out) = if let Some(c) = context {
                let k_cur = k_lin.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                let v_cur = v_lin.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                let k_all = Tensor::cat(&[&c.k, &k_cur], 2)?;
                let v_all = Tensor::cat(&[&c.v, &v_cur], 2)?;
                let skv = k_all.dim(2)?;
                let k_bnd = k_all.transpose(1, 2)?.contiguous()?.reshape((b, skv, d))?;
                let v_bnd = v_all.transpose(1, 2)?.contiguous()?.reshape((b, skv, d))?;
                (k_bnd, v_bnd, skv, if need_kv { Some(LayerKv { k: k_cur, v: v_cur }) } else { None })
            } else {
                let kv = if need_kv {
                    let k_cur = k_lin.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                    let v_cur = v_lin.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
                    Some(LayerKv { k: k_cur, v: v_cur })
                } else {
                    None
                };
                (k_lin.contiguous()?, v_lin.contiguous()?, n, kv)
            };
            let q_c = q_lin.contiguous()?;
            let mask_vec: Option<Vec<f32>> = match mask {
                Some(m) if m.dims() == [b, 1, 1, seq_kv] => Some(m.flatten_all()?.to_vec1::<f32>()?),
                _ => None,
            };
            if mask.is_none() || mask_vec.is_some() {
                let (qs, ql) = q_c.storage_and_layout();
                let (ks, kl) = k_src.storage_and_layout();
                let (vs, vl) = v_src.storage_and_layout();
                if let (
                    candle_core::Storage::Cpu(candle_core::CpuStorage::F32(qv)),
                    candle_core::Storage::Cpu(candle_core::CpuStorage::F32(kv_v)),
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
                    let q0 = ql.start_offset();
                    let k0 = kl.start_offset();
                    let v0 = vl.start_offset();
                    for bi in 0..b {
                        let q_sl = &qv[q0 + bi * n * d..q0 + (bi + 1) * n * d];
                        let k_sl = &kv_v[k0 + bi * seq_kv * d..k0 + (bi + 1) * seq_kv * d];
                        let v_sl = &vv[v0 + bi * seq_kv * d..v0 + (bi + 1) * seq_kv * d];
                        let m_sl = mask_vec.as_deref().map(|mv| &mv[bi * seq_kv..(bi + 1) * seq_kv]);
                        noto_cpu_kernels::stream_attention_f32_masked(
                            q_sl,
                            k_sl,
                            v_sl,
                            n,
                            seq_kv,
                            &cfg,
                            m_sl,
                            &mut ctx[bi * n * d..(bi + 1) * n * d],
                        );
                    }
                    let ctx_t = Tensor::from_vec(ctx, (b, n, d), x.device())?;
                    return Ok((self.out.forward(&ctx_t)?, kv_out));
                }
            }
        }

        // Fastpath 2: GPU FlashAttention (unmasked, head_dim <= 256; zero transpose copies when context=None && !need_kv).
        if mask.is_none() && self.head_dim <= 256 && accel_op_ok(x.device(), AccelOp::FlashAttn) {
            let fa_cfg = candle_nn::flash_attn::FlashAttnConfig::new(scale as f32);
            if context.is_none() && !need_kv {
                let q4 = q_lin.contiguous()?.reshape((b, n, self.heads, self.head_dim))?;
                let k4 = k_lin.contiguous()?.reshape((b, n, self.heads, self.head_dim))?;
                let v4 = v_lin.contiguous()?.reshape((b, n, self.heads, self.head_dim))?;
                if let Ok(ctx) = candle_nn::flash_attn::flash_attn_with_config(&q4, &k4, &v4, &fa_cfg) {
                    let ctx = ctx.reshape((b, n, d))?;
                    return Ok((self.out.forward(&ctx)?, None));
                }
            } else {
                let split = |t: &Tensor| -> Result<Tensor> {
                    t.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()
                };
                let (q, k, v) = (split(&q_lin)?, split(&k_lin)?, split(&v_lin)?);
                let (k_all, v_all) = match context {
                    Some(c) => (Tensor::cat(&[&c.k, &k], 2)?, Tensor::cat(&[&c.v, &v], 2)?),
                    None => (k.clone(), v.clone()),
                };
                if let Ok(ctx) = candle_nn::flash_attn::flash_attn_bhsd(&q, &k_all, &v_all, &fa_cfg) {
                    let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, n, d))?;
                    return Ok((self.out.forward(&ctx)?, if need_kv { Some(LayerKv { k, v }) } else { None }));
                }
            }
        }

        // Standard unfused attention fallback (always used for masked text attention on GPU and non-packed CPU).
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(q_lin)?, split(k_lin)?, split(v_lin)?);
        let (k_all, v_all) = match context {
            Some(c) => (Tensor::cat(&[&c.k, &k], 2)?, Tensor::cat(&[&c.v, &v], 2)?),
            None => (k.clone(), v.clone()),
        };
        let mut scores = q.matmul(&k_all.transpose(2, 3)?)?.affine(scale, 0.0)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v_all)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, d))?;
        Ok((self.out.forward(&ctx)?, if need_kv { Some(LayerKv { k, v }) } else { None }))
    }
}

pub struct Mlp {
    fc1: QLinear,
    fc2: QLinear,
    cpu_packed: bool,
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_weights(hidden, intermediate, &QWeights::plain(vb, Family::Siglip2))
    }

    pub fn with_weights(hidden: usize, intermediate: usize, w: &QWeights) -> Result<Self> {
        Ok(Mlp {
            fc1: w.pp("fc1").linear(hidden, intermediate).map_err(to_candle)?,
            fc2: w.pp("fc2").linear(intermediate, hidden).map_err(to_candle)?,
            cpu_packed: w.is_cpu_packed(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = if self.fc1.is_cpu_packed() || matches!(x.device(), candle_core::Device::Cuda(_) | candle_core::Device::Rocm(_)) {
            self.fc1.forward_gelu(x, false)?
        } else {
            fast_gelu_tanh(&self.fc1.forward(x)?, self.cpu_packed)?
        };
        self.fc2.forward(&h)
    }
}

pub struct EncoderLayer {
    ln1: FastLayerNorm,
    attn: Attention,
    ln2: FastLayerNorm,
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
        let packed = w.is_cpu_packed();
        Ok(EncoderLayer {
            ln1: FastLayerNorm::new(hidden, eps, vb.pp("layer_norm1"), packed)?,
            attn: Attention::with_weights(hidden, heads, &w.pp("self_attn"), fused_qkv)?,
            ln2: FastLayerNorm::new(hidden, eps, vb.pp("layer_norm2"), packed)?,
            mlp: Mlp::with_weights(hidden, intermediate, &w.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.ln1.forward(x)?, mask)?)?;
        &h + self.mlp.forward(&self.ln2.forward(&h)?)?
    }

    /// `forward` (no mask), also returning the layer's keys and values; `context` is `Attention::forward_kv`'s.
    pub fn forward_kv(&self, x: &Tensor, context: Option<&LayerKv>) -> Result<(Tensor, LayerKv)> {
        let (a, kv) = self.attn.forward_kv(&self.ln1.forward(x)?, None, context)?;
        let h = (x + a)?;
        Ok(((&h + self.mlp.forward(&self.ln2.forward(&h)?)?)?, kv))
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
    fn tokens_opened_later_attend_like_a_mask_that_hides_them_from_the_first() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let build = |vb: &VarBuilder| (EncoderLayer::new(16, 32, 4, 1e-6, vb.pp("l0")).unwrap(), EncoderLayer::new(16, 32, 4, 1e-6, vb.pp("l1")).unwrap());
        drop(build(&vb));
        for (name, var) in vm.data().lock().unwrap().iter() {
            if !name.contains("norm") {
                var.set(&Tensor::randn(0f32, 0.3, var.dims().to_vec(), &dev).unwrap()).unwrap();
            }
        }
        let (l0, l1) = build(&vb);
        let x = Tensor::randn(0f32, 1.0, (1, 5, 16), &dev).unwrap();
        // The first three tokens never see the last two; the last two see everything.
        let mut m = vec![0f32; 25];
        for i in 0..3 {
            for j in 3..5 {
                m[i * 5 + j] = -1e9;
            }
        }
        let mask = Tensor::from_vec(m, (1, 1, 5, 5), &dev).unwrap();
        let full = l1.forward(&l0.forward(&x, Some(&mask)).unwrap(), Some(&mask)).unwrap();
        let (a0, kv0) = l0.forward_kv(&x.narrow(1, 0, 3).unwrap(), None).unwrap();
        let (a1, kv1) = l1.forward_kv(&a0, None).unwrap();
        let (b0, _) = l0.forward_kv(&x.narrow(1, 3, 2).unwrap(), Some(&kv0)).unwrap();
        let (b1, _) = l1.forward_kv(&b0, Some(&kv1)).unwrap();
        let steps = Tensor::cat(&[&a1, &b1], 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let full = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(full.iter().any(|v| v.abs() > 1e-2));
        for (p, q) in full.iter().zip(&steps) {
            assert!((p - q).abs() < 1e-4, "{p} vs {q}");
        }
        // Without context the pass is `forward`, bit for bit.
        assert_eq!(l0.forward(&x, None).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), l0.forward_kv(&x, None).unwrap().0.flatten_all().unwrap().to_vec1::<f32>().unwrap());
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
