//! Thin bridge to `premium_discount_quant::kquant` translating Candle's `GgmlDType`.

use anyhow::anyhow;
use candle_core::quantized::GgmlDType;
use premium_discount_quant::kquant as pd;

pub use pd::rel;

fn to_format(dtype: GgmlDType) -> anyhow::Result<pd::Format> {
    match dtype {
        GgmlDType::F32 => Ok(pd::Format::F32),
        GgmlDType::F16 => Ok(pd::Format::F16),
        GgmlDType::Q8_0 => Ok(pd::Format::Q8_0),
        GgmlDType::Q6K => Ok(pd::Format::Q6K),
        GgmlDType::Q5K => Ok(pd::Format::Q5K),
        GgmlDType::Q4K => Ok(pd::Format::Q4K),
        _ => Err(anyhow!("{dtype:?} is not supported (Q8_0, Q6_K, Q5_K, Q4_K, F16, F32)")),
    }
}

pub fn encode(
    xs: &[f32],
    cols: usize,
    dtype: GgmlDType,
    importance: Option<&[f32]>,
) -> anyhow::Result<Vec<u8>> {
    pd::encode(xs, cols, to_format(dtype)?, importance).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
pub mod testing {
    /// Deterministic standard-normal pseudo-random vector for quantization/vocab unit tests.
    pub fn normals(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let (u1, u2) = (next(), next());
            let r = (-2.0 * u1.ln()).sqrt();
            let th = std::f64::consts::TAU * u2;
            out.push((r * th.cos()) as f32);
            if out.len() < n {
                out.push((r * th.sin()) as f32);
            }
        }
        out
    }
}
