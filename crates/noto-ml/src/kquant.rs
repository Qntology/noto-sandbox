//! GGML block encoders (Q8_0 · Q6_K · Q5_K · Q4_K) for weights read straight from the F32 safetensors a model loads.
//! A weight is quantized **once**, from its original values: nothing here reads a quantized file back.
//!
//! The per-block scale search is gravity-ocr's `quant.py` (the one blueprint-sandbox measured and kept):
//! - importance `w = rms(block) + |x|`, so the few large weights that make most of a product stay exact, times an
//!   optional per-column importance (an activation mean square, llama.cpp's `imatrix`);
//! - symmetric (Q8_0, Q6_K): 19 candidate scales `(nmax + 0.1 s) / amax` (s = −9 … 9), each rounded and refitted by
//!   weighted least squares `d = Σ w x q / Σ w q²`, the one with the least weighted error kept;
//! - asymmetric (Q4_K, Q5_K): 21 candidates `(nmax − 1 + 0.1 s) / (max − min(min, 0))` (s = 0 … 20), (scale, offset)
//!   solved by weighted least squares; GGML stores the offset as a non-negative minimum, so a positive one falls back
//!   to the scale alone;
//! - the sub-block scales and minimums of a 256 super block are quantized against an F16 super scale as GGML does
//!   (6 bits for Q4_K and Q5_K, signed 8 bits for Q6_K), and the values are rounded again with the stored scales.
//!
//! The bit layout is GGML's, so candle multiplies the bytes on the device (the tests read them back with candle's own
//! dequantizer). Rounding is round-half-to-even (numpy's `rint`).

use anyhow::{bail, Result};
use candle_core::quantized::GgmlDType;
use half::f16;
use rayon::prelude::*;

const QK_K: usize = 256;
const QK8_0: usize = 32;
/// Elements per parallel chunk.
const PAR_ELEMS: usize = 1 << 16;

/// The formats this module writes.
pub fn supported(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::Q8_0 | GgmlDType::Q6K | GgmlDType::Q5K | GgmlDType::Q4K)
}

/// Row-major rows `xs` of `cols` columns as `dtype` GGML bytes. `importance` weighs the columns (all 1 without it).
pub fn encode(xs: &[f32], cols: usize, dtype: GgmlDType, importance: Option<&[f32]>) -> Result<Vec<u8>> {
    if cols == 0 || !xs.len().is_multiple_of(cols) {
        bail!("{} values do not split into rows of {cols}", xs.len());
    }
    if !supported(dtype) {
        bail!("{dtype:?} is not written here (Q8_0, Q6_K, Q5_K, Q4_K, F16, F32)");
    }
    if !cols.is_multiple_of(dtype.block_size()) {
        bail!("{cols} columns are not a multiple of the {dtype:?} block {}", dtype.block_size());
    }
    if let Some(w) = importance {
        if w.len() != cols {
            bail!("{} importances for {cols} columns", w.len());
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            bail!("an importance is negative or not finite");
        }
    }
    Ok(match dtype {
        GgmlDType::F32 => xs.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlDType::F16 => xs.iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        _ => {
            let per = (PAR_ELEMS / cols).max(1) * cols;
            let parts: Vec<Vec<u8>> = xs
                .par_chunks(per)
                .map(|rows| {
                    let mut out = Vec::with_capacity(rows.len() / dtype.block_size() * dtype.type_size());
                    for row in rows.chunks_exact(cols) {
                        encode_row(row, dtype, importance, &mut out);
                    }
                    out
                })
                .collect();
            parts.concat()
        }
    })
}

fn encode_row(row: &[f32], dtype: GgmlDType, importance: Option<&[f32]>, out: &mut Vec<u8>) {
    let bs = dtype.block_size();
    for (b, x) in row.chunks_exact(bs).enumerate() {
        let colw = importance.map(|w| &w[b * bs..(b + 1) * bs]);
        match dtype {
            GgmlDType::Q8_0 => q8_0(x, colw, out),
            GgmlDType::Q6K => q6_k(x, colw, out),
            GgmlDType::Q5K => q45_k(x, colw, 31, out),
            GgmlDType::Q4K => q45_k(x, colw, 15, out),
            _ => unreachable!("supported() filtered the format"),
        }
    }
}

/// `quant.py` `_importance`: `rms + |x|` per block, times the column importance when it is not all zero.
fn importance(x: &[f32], colw: Option<&[f32]>, w: &mut [f32]) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    for (wi, xi) in w.iter_mut().zip(x) {
        *wi = rms + xi.abs();
    }
    if let Some(c) = colw {
        let total: f32 = w.iter().zip(c).map(|(a, b)| a * b).sum();
        if total > 0.0 && total.is_finite() {
            w.iter_mut().zip(c).for_each(|(a, b)| *a *= b);
        }
    }
}

/// `quant.py` `_fit_symmetric`: the weighted least-squares scale of the rounding to q ∈ [`qmin`, `nmax`].
pub(crate) fn fit_symmetric(x: &[f32], w: &[f32], nmax: i32, qmin: i32) -> f32 {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    if amax <= 0.0 || !amax.is_finite() {
        return 0.0;
    }
    let (lo, hi) = (qmin as f32, nmax as f32);
    let (mut best_err, mut best_d) = (f32::INFINITY, 0f32);
    for s in -9..=9 {
        let inv = (nmax as f32 + 0.1 * s as f32) / amax;
        let (mut num, mut den) = (0f32, 0f32);
        for (&xi, &wi) in x.iter().zip(w) {
            let q = (xi * inv).round_ties_even().clamp(lo, hi);
            num += wi * xi * q;
            den += wi * q * q;
        }
        let d = if den > 0.0 { num / den } else { 0.0 };
        let mut err = 0f32;
        for (&xi, &wi) in x.iter().zip(w) {
            let q = (xi * inv).round_ties_even().clamp(lo, hi);
            let e = d * q - xi;
            err += wi * e * e;
        }
        if err < best_err {
            best_err = err;
            best_d = d;
        }
    }
    best_d
}

/// `quant.py` `_fit_asymmetric`: (d, m ≥ 0) of `x ≈ d q − m` with q ∈ [0, `nmax`].
pub(crate) fn fit_asymmetric(x: &[f32], w: &[f32], nmax: i32) -> (f32, f32) {
    let lo = x.iter().fold(0f32, |a, &v| a.min(v));
    let hi = x.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let range = hi - lo;
    let safe = if range > 0.0 { range } else { 1.0 };
    let sum_w: f32 = w.iter().sum();
    let sum_x: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
    let top = nmax as f32;
    let (mut best_err, mut best_d, mut best_m) = (f32::INFINITY, 0f32, 0f32);
    for s in 0..=20 {
        let inv = (top - 1.0 + 0.1 * s as f32) / safe;
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for (&xi, &wi) in x.iter().zip(w) {
            let q = ((xi - lo) * inv).round_ties_even().clamp(0.0, top);
            sl += wi * q;
            sl2 += wi * q * q;
            sxl += wi * xi * q;
        }
        let det = sum_w * sl2 - sl * sl;
        let scale_only = || if sl2 > 0.0 { sxl / sl2 } else { 0.0 };
        // All roundings equal (a constant block): a non-zero step gets the scale alone, an all-zero step the offset alone
        // (a negative constant block = −m). quant.py leaves both at zero there.
        let (mut d, mut off) = if det > 0.0 {
            ((sum_w * sxl - sum_x * sl) / det, (sl2 * sum_x - sl * sxl) / det)
        } else if sl2 > 0.0 {
            (scale_only(), 0.0)
        } else {
            (0.0, if sum_w > 0.0 { (sum_x / sum_w).min(0.0) } else { 0.0 })
        };
        if off > 0.0 {
            d = scale_only();
            off = 0.0;
        }
        let mut err = 0f32;
        for (&xi, &wi) in x.iter().zip(w) {
            let q = ((xi - lo) * inv).round_ties_even().clamp(0.0, top);
            let e = d * q + off - xi;
            err += wi * e * e;
        }
        if err < best_err {
            best_err = err;
            best_d = d;
            best_m = -off;
        }
    }
    (best_d.max(0.0), best_m.max(0.0))
}

/// Sub-block scales (≥ 0) as an F16 super scale and `levels`-step integers; the F16 rounding goes up so that the super
/// scale times `levels` covers the largest one.
fn super_scale<const N: usize>(v: &[f32; N], levels: u8) -> (f16, [u8; N]) {
    let max = v.iter().fold(0f32, |a, &b| a.max(b));
    let mut idx = [0u8; N];
    if max <= 0.0 || !max.is_finite() {
        return (f16::ZERO, idx);
    }
    let top = levels as f32;
    let mut d = f16::from_f32(max / top);
    if d.to_f32() * top < max {
        d = f16::from_bits(d.to_bits() + 1);
    }
    let df = d.to_f32();
    for (i, &x) in v.iter().enumerate() {
        idx[i] = (x / df).round_ties_even().clamp(0.0, top) as u8;
    }
    (d, idx)
}

/// GGML `get_scale_min_k4`: the j-th 6-bit scale and minimum packed in 12 bytes.
fn scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q8_0: blocks of 32, an F16 scale, −127 … 127.
fn q8_0(x: &[f32], colw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut w = [0f32; QK8_0];
    importance(x, colw, &mut w);
    let d = f16::from_f32(fit_symmetric(x, &w, 127, -127).max(0.0));
    let df = d.to_f32();
    out.extend_from_slice(&d.to_bits().to_le_bytes());
    for &xi in x {
        let q = if df == 0.0 { 0 } else { (xi / df).round_ties_even().clamp(-127.0, 127.0) as i8 };
        out.push(q as u8);
    }
}

/// Q4_K (`nmax` 15) and Q5_K (`nmax` 31): a super block of 256 = 8 blocks of 32, a 6-bit scale and minimum per block.
fn q45_k(x: &[f32], colw: Option<&[f32]>, nmax: i32, out: &mut Vec<u8>) {
    let mut w = [0f32; 32];
    let (mut ds, mut ms) = ([0f32; 8], [0f32; 8]);
    for j in 0..8 {
        let xb = &x[32 * j..32 * j + 32];
        importance(xb, colw.map(|c| &c[32 * j..32 * j + 32]), &mut w);
        (ds[j], ms[j]) = fit_asymmetric(xb, &w, nmax);
    }
    let (d, ls) = super_scale(&ds, 63);
    let (dmin, lm) = super_scale(&ms, 63);
    let mut scales = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            scales[j] = ls[j];
            scales[j + 4] = lm[j];
        } else {
            scales[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            scales[j - 4] |= (ls[j] >> 4) << 6;
            scales[j] |= (lm[j] >> 4) << 6;
        }
    }
    // Round the values again with the stored scales (the formula of candle's `to_float`).
    let (df, dmf) = (d.to_f32(), dmin.to_f32());
    let top = nmax as f32;
    let mut l = [0u8; QK_K];
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &scales);
        let dd = df * sc as f32;
        let dm = dmf * m as f32;
        if dd == 0.0 {
            continue;
        }
        for i in 0..32 {
            l[32 * j + i] = ((x[32 * j + i] + dm) / dd).round_ties_even().clamp(0.0, top) as u8;
        }
    }
    out.extend_from_slice(&d.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin.to_bits().to_le_bytes());
    out.extend_from_slice(&scales);
    let mut qs = [0u8; QK_K / 2];
    if nmax == 31 {
        let mut qh = [0u8; QK_K / 8];
        let (mut m1, mut m2) = (1u8, 2u8);
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let (mut l1, mut l2) = (l[n + j], l[n + j + 32]);
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[n / 2 + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
        out.extend_from_slice(&qh);
    } else {
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                qs[n / 2 + j] = l[n + j] | (l[n + j + 32] << 4);
            }
        }
    }
    out.extend_from_slice(&qs);
}

/// Q6_K: a super block of 256 = 16 blocks of 16, a signed 8-bit scale per block, values −32 … 31.
fn q6_k(x: &[f32], colw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut w = [0f32; 16];
    let mut ds = [0f32; 16];
    for s in 0..16 {
        let xb = &x[16 * s..16 * s + 16];
        importance(xb, colw.map(|c| &c[16 * s..16 * s + 16]), &mut w);
        ds[s] = fit_symmetric(xb, &w, 31, -32).max(0.0);
    }
    // As GGML: a negative super scale reaches the −128 step of the signed byte (block scale = super × negative ≥ 0).
    let max = ds.iter().fold(0f32, |a, &b| a.max(b));
    let mut scales = [0i8; 16];
    let mut d = f16::ZERO;
    if max > 0.0 {
        d = f16::from_f32(-max / 128.0);
        if d.to_f32().abs() * 128.0 < max {
            d = f16::from_bits(d.to_bits() + 1);
        }
        let df = d.to_f32();
        for s in 0..16 {
            scales[s] = (ds[s] / df).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
    let df = d.to_f32();
    let mut l = [32u8; QK_K];
    for s in 0..16 {
        let dd = df * scales[s] as f32;
        if dd == 0.0 {
            continue;
        }
        for i in 0..16 {
            let q = (x[16 * s + i] / dd).round_ties_even().clamp(-32.0, 31.0) as i32;
            l[16 * s + i] = (q + 32) as u8;
        }
    }
    let (mut ql, mut qh) = ([0u8; QK_K / 2], [0u8; QK_K / 4]);
    for (half, j) in (0..QK_K).step_by(128).enumerate() {
        for k in 0..32 {
            let (a, b, c, e) = (l[j + k], l[j + k + 32], l[j + k + 64], l[j + k + 96]);
            ql[64 * half + k] = (a & 0xF) | ((c & 0xF) << 4);
            ql[64 * half + k + 32] = (b & 0xF) | ((e & 0xF) << 4);
            qh[32 * half + k] = (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((e >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(scales.iter().map(|s| *s as u8));
    out.extend_from_slice(&d.to_bits().to_le_bytes());
}

/// GGML bytes (`rows` × `cols`) back to F32 with candle's dequantizer (diagnostics and tests).
pub fn decode(dtype: GgmlDType, bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    use candle_core::quantized::ggml_file::qtensor_from_ggml;
    use candle_core::Device;
    let half = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    Ok(match dtype {
        GgmlDType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        GgmlDType::F16 => bytes.chunks_exact(2).map(|b| f16::from_bits(half(b)).to_f32()).collect(),
        _ => {
            // `qtensor_from_ggml` reads the bytes as blocks, so they go into a u32-aligned buffer first.
            let mut words = vec![0u32; bytes.len().div_ceil(4)];
            let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
            raw[..bytes.len()].copy_from_slice(bytes);
            let q = qtensor_from_ggml(dtype, &raw[..bytes.len()], vec![rows, cols], &Device::Cpu)?;
            q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?
        }
    })
}

/// ‖a − b‖ / ‖b‖
pub fn rel(a: &[f32], b: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

/// The relative weight error ‖Ŵ − W‖ / ‖W‖ of one quantization (diagnostics).
pub fn relative_error(xs: &[f32], cols: usize, dtype: GgmlDType, importance: Option<&[f32]>) -> Result<f32> {
    let q = encode(xs, cols, dtype, importance)?;
    let back = decode(dtype, &q, xs.len() / cols, cols)?;
    Ok(rel(&back, xs))
}

#[cfg(test)]
pub(crate) mod testing {
    /// Deterministic standard normals (Box–Muller over a 64-bit LCG), the same for every run and platform.
    pub fn normals(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n).map(|_| ((-2.0 * next().ln()).sqrt() * (std::f64::consts::TAU * next()).cos()) as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::normals;
    use super::*;

    /// Gaussian rows with rare large values (heavy tails, like real weights).
    fn heavy(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut xs = normals(rows * cols, seed);
        for (i, x) in xs.iter_mut().enumerate() {
            *x *= 0.02;
            if i % 97 == 0 {
                *x *= 6.0;
            }
        }
        xs
    }

    /// candle's own quantizer (`from_float`).
    fn candle_plain(xs: &[f32], dtype: GgmlDType) -> Vec<u8> {
        let mut q = dtype.cpu_zeros(xs.len());
        q.from_float(xs);
        unsafe { std::slice::from_raw_parts(q.as_ptr(), q.storage_size_in_bytes()) }.to_vec()
    }

    #[test]
    fn the_block_fits_match_gravity_ocr_quant_py() {
        // gravity_ocr/quant.py on the same input (numpy f32), as measured for blueprint-sandbox.
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 1.3).sin() * 0.05 + if i % 7 == 0 { 0.2 } else { 0.0 }).collect();
        let mut w = [0f32; 32];
        importance(&x, None, &mut w);
        let close = |got: f32, want: f32| (got - want).abs() <= want.abs() * 2e-5;
        let (d, m) = fit_asymmetric(&x, &w, 15);
        assert!(close(d, 1.925078221e-02) && close(m, 4.800733924e-02), "q4 {d} {m}");
        let (d, m) = fit_asymmetric(&x, &w, 31);
        assert!(close(d, 9.597460739e-03) && close(m, 4.951747879e-02), "q5 {d} {m}");
        let x16: Vec<f32> = x[..16].iter().map(|v| v - 0.01).collect();
        let mut w16 = [0f32; 16];
        importance(&x16, None, &mut w16);
        let d = fit_symmetric(&x16, &w16, 31, -32);
        assert!(close(d, 6.606251933e-03), "q6 {d}");
        // quant.py's q8 reaches −128 (Q8_0 here keeps GGML's −127).
        let d = fit_symmetric(&x, &w, 127, -128);
        assert!(close(d, 1.904964098e-03), "q8 {d}");
    }

    #[test]
    fn candle_reads_the_layout_and_the_error_beats_its_own_quantizer() -> Result<()> {
        let cols = 1024;
        let xs = heavy(32, cols, 7);
        for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let ours = encode(&xs, cols, dtype, None)?;
            assert_eq!(ours.len(), xs.len() / dtype.block_size() * dtype.type_size(), "{dtype:?} bytes");
            let mine = rel(&decode(dtype, &ours, 32, cols)?, &xs);
            let theirs = rel(&decode(dtype, &candle_plain(&xs, dtype), 32, cols)?, &xs);
            // A wrong layout reads back tens of percent off (the normal 4-bit error of a 32 block is about 9 %).
            let bound = match dtype {
                GgmlDType::Q4K => 0.11,
                GgmlDType::Q5K => 0.055,
                GgmlDType::Q6K => 0.03,
                _ => 0.008,
            };
            assert!(mine < bound, "{dtype:?} error {mine}");
            assert!(mine <= theirs * 1.02, "{dtype:?}: this search {mine:.5}, candle {theirs:.5}");
        }
        let e: Vec<f32> = [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8_0].iter().map(|d| relative_error(&xs, cols, *d, None).unwrap()).collect();
        assert!(e.windows(2).all(|w| w[0] > w[1]), "more bits, less error: {e:?}");
        Ok(())
    }

    #[test]
    fn zero_and_constant_blocks_stay_finite() -> Result<()> {
        let cols = 256;
        let mut xs = vec![0f32; cols];
        xs.extend(std::iter::repeat_n(0.03f32, cols));
        xs.extend(std::iter::repeat_n(-0.03f32, cols));
        for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let back = decode(dtype, &encode(&xs, cols, dtype, None)?, 3, cols)?;
            assert!(back.iter().all(|v| v.is_finite()), "{dtype:?}");
            assert!(back[..cols].iter().all(|v| *v == 0.0), "{dtype:?}: a zero row stays zero");
            let err = rel(&back[cols..], &xs[cols..]);
            assert!(err < 0.02, "{dtype:?}: constant rows off by {err}");
        }
        Ok(())
    }

    #[test]
    fn column_importance_moves_the_error_to_unimportant_columns() -> Result<()> {
        let cols = 512;
        let xs = heavy(64, cols, 11);
        // Even columns a hundred times as important. The weighting is relative inside a block, so it has to differ there.
        let imp: Vec<f32> = (0..cols).map(|c| if c % 2 == 0 { 100.0 } else { 1.0 }).collect();
        let even = |v: &[f32]| -> Vec<f32> { v.iter().step_by(2).copied().collect() };
        for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let plain = decode(dtype, &encode(&xs, cols, dtype, None)?, 64, cols)?;
            let weighted = decode(dtype, &encode(&xs, cols, dtype, Some(&imp))?, 64, cols)?;
            let (a, b) = (rel(&even(&weighted), &even(&xs)), rel(&even(&plain), &even(&xs)));
            assert!(a < b, "{dtype:?}: important columns {a:.5} (weighted) vs {b:.5} (uniform)");
        }
        assert!(encode(&xs, cols, GgmlDType::Q4K, Some(&imp[..10])).is_err(), "a wrong importance length is refused");
        Ok(())
    }

    #[test]
    fn unsupported_or_misaligned_requests_are_refused() {
        assert!(encode(&[0.0; 256], 256, GgmlDType::Q3K, None).is_err(), "no Q3_K");
        assert!(encode(&[0.0; 200], 100, GgmlDType::Q8_0, None).is_err(), "columns not a block multiple");
        assert!(encode(&[0.0; 10], 3, GgmlDType::F32, None).is_err());
        let b = encode(&[1.5, -2.0], 2, GgmlDType::F16, None).unwrap();
        assert_eq!(decode(GgmlDType::F16, &b, 1, 2).unwrap(), vec![1.5, -2.0]);
    }
}
