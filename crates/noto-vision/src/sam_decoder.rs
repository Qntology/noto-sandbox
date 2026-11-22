use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::transformer::TwoWayTransformer;
use candle_transformers::models::segment_anything::{linear, LayerNorm2d, Linear};

struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("layers");
        let layers = (0..num_layers)
            .map(|i| {
                let in_dim = if i == 0 { input_dim } else { hidden_dim };
                let out_dim = if i + 1 == num_layers { output_dim } else { hidden_dim };
                linear(vb.pp(i), in_dim, out_dim, true)
            })
            .collect::<Result<_>>()?;
        Ok(Mlp { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?;
            }
        }
        Ok(xs)
    }
}

/// Non-overlapping 2x2 stride-2 ConvTranspose2d as GEMM + pixel-shuffle (native device op on all backends).
/// Keeps live `VarMap` tensors un-detached so post-build `Var::set()` in tests remains visible.
enum FastUpscale2x2 {
    Packed { proj: candle_nn::Linear, bias: Tensor, c_out: usize },
    Live { weight: Tensor, bias: Tensor, c_in: usize, c_out: usize },
}

impl FastUpscale2x2 {
    fn new(c_in: usize, c_out: usize, vb: VarBuilder) -> Result<Self> {
        let w_raw = vb.get((c_in, c_out, 2, 2), "weight")?;
        let b_raw = vb.get(c_out, "bias")?;
        if w_raw.is_variable() || b_raw.is_variable() {
            return Ok(FastUpscale2x2::Live { weight: w_raw, bias: b_raw, c_in, c_out });
        }
        let w = w_raw
            .permute((1, 2, 3, 0))?
            .contiguous()?
            .reshape((c_out * 4, c_in))?;
        let bias = b_raw.reshape((1, c_out, 1, 1))?;
        Ok(FastUpscale2x2::Packed { proj: candle_nn::Linear::new(w, None), bias, c_out })
    }
}

impl Module for FastUpscale2x2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = x.dims4()?;
        let x_hwc = x.permute((0, 2, 3, 1))?.contiguous()?;
        match self {
            FastUpscale2x2::Packed { proj, bias, c_out } => {
                let y = proj
                    .forward(&x_hwc)?
                    .reshape((b, h, w, *c_out, 2, 2))?
                    .permute((0, 3, 1, 4, 2, 5))?
                    .contiguous()?
                    .reshape((b, *c_out, h * 2, w * 2))?;
                y.broadcast_add(bias)
            }
            FastUpscale2x2::Live { weight, bias, c_in, c_out } => {
                let wt = weight
                    .permute((1, 2, 3, 0))?
                    .contiguous()?
                    .reshape((*c_out * 4, *c_in))?;
                let y = candle_nn::Linear::new(wt, None)
                    .forward(&x_hwc)?
                    .reshape((b, h, w, *c_out, 2, 2))?
                    .permute((0, 3, 1, 4, 2, 5))?
                    .contiguous()?
                    .reshape((b, *c_out, h * 2, w * 2))?;
                y.broadcast_add(&bias.reshape((1, *c_out, 1, 1))?)
            }
        }
    }
}

pub struct MaskDecoder {
    iou_token: candle_nn::Embedding,
    mask_tokens: candle_nn::Embedding,
    iou_prediction_head: Mlp,
    output_upscaling_conv1: FastUpscale2x2,
    output_upscaling_ln: LayerNorm2d,
    output_upscaling_conv2: FastUpscale2x2,
    num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<Mlp>,
    transformer: TwoWayTransformer,
}

impl MaskDecoder {
    pub fn new(transformer_dim: usize, num_multimask_outputs: usize, iou_head_depth: usize, iou_head_hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let num_mask_tokens = num_multimask_outputs + 1;
        let iou_prediction_head = Mlp::new(transformer_dim, iou_head_hidden_dim, num_mask_tokens, iou_head_depth, vb.pp("iou_prediction_head"))?;
        let iou_token = candle_nn::embedding(1, transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens = candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
        let output_upscaling_conv1 = FastUpscale2x2::new(transformer_dim, transformer_dim / 4, vb.pp("output_upscaling.0"))?;
        let output_upscaling_ln = LayerNorm2d::new(transformer_dim / 4, 1e-6, vb.pp("output_upscaling.1"))?;
        let output_upscaling_conv2 = FastUpscale2x2::new(transformer_dim / 4, transformer_dim / 8, vb.pp("output_upscaling.3"))?;
        let vb_o = vb.pp("output_hypernetworks_mlps");
        let output_hypernetworks_mlps = (0..num_mask_tokens).map(|i| Mlp::new(transformer_dim, transformer_dim, transformer_dim / 8, 3, vb_o.pp(i))).collect::<Result<_>>()?;
        let transformer = TwoWayTransformer::new(2, transformer_dim, 8, 2048, vb.pp("transformer"))?;
        Ok(MaskDecoder { iou_token, mask_tokens, iou_prediction_head, output_upscaling_conv1, output_upscaling_ln, output_upscaling_conv2, num_mask_tokens, output_hypernetworks_mlps, transformer })
    }

    pub fn forward(&self, image_embeddings: &Tensor, image_pe: &Tensor, sparse: &Tensor, dense: &Tensor, multimask_output: bool) -> Result<(Tensor, Tensor)> {
        let (masks, iou_pred) = self.predict_masks(image_embeddings, image_pe, sparse, dense)?;
        if multimask_output {
            Ok((masks.i((.., 1..))?, iou_pred.i((.., 1..))?))
        } else {
            Ok((masks.i((.., 0..1))?, iou_pred.i((.., 0..1))?))
        }
    }

    fn predict_masks(&self, image_embeddings: &Tensor, image_pe: &Tensor, sparse: &Tensor, dense: &Tensor) -> Result<(Tensor, Tensor)> {
        let output_tokens = Tensor::cat(&[self.iou_token.embeddings(), self.mask_tokens.embeddings()], 0)?;
        let (d1, d2) = output_tokens.dims2()?;
        let output_tokens = output_tokens.unsqueeze(0)?.expand((sparse.dim(0)?, d1, d2))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse], 1)?;
        let src = repeat_interleave(image_embeddings, tokens.dim(0)?, 0)?;
        let src = src.broadcast_add(dense)?;
        let pos_src = repeat_interleave(image_pe, tokens.dim(0)?, 0)?;
        let (b, c, h, w) = src.dims4()?;
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., 0))?.contiguous()?;
        let mask_tokens_out = hs.i((.., 1..1 + self.num_mask_tokens))?.contiguous()?;
        let src = src.transpose(1, 2)?.reshape((b, c, h, w))?;
        let upscaled = self.output_upscaling_conv1.forward(&src)?.apply(&self.output_upscaling_ln)?.gelu()?.apply(&self.output_upscaling_conv2)?.gelu()?;
        let hyper_in: Vec<Tensor> = self
            .output_hypernetworks_mlps
            .iter()
            .enumerate()
            .map(|(i, mlp)| mlp.forward(&mask_tokens_out.i((.., i))?.contiguous()?))
            .collect::<Result<_>>()?;
        let hyper_in = Tensor::stack(&hyper_in, 1)?.contiguous()?;
        let (b, c, h, w) = upscaled.dims4()?;
        let masks = hyper_in.matmul(&upscaled.reshape((b, c, h * w))?)?.reshape((b, (), h, w))?;
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        Ok((masks, iou_pred))
    }
}

fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let img = img.unsqueeze(dim + 1)?;
    let mut dims = img.dims().to_vec();
    dims[dim + 1] = repeats;
    img.broadcast_as(dims)?.flatten(dim, dim + 1)
}
