use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};

pub const DEVICE_ENV: &str = "NOTO_DEVICE";
pub const BATCH_ROWS_ENV: &str = "NOTO_BATCH_ROWS";
pub const FIXED_SHAPE_ENV: &str = "NOTO_FIXED_SHAPE";
pub const FLASH_ATTN_ENV: &str = "NOTO_FLASH_ATTN";
pub const CPU_PACKED_ENV: &str = "NOTO_CPU_PACKED";
pub const ACCEL_CHECK_ENV: &str = "NOTO_ACCEL_CHECK";
pub const POOL_KEEP_BYTES: usize = 256 << 20;
pub const VULKAN_TRIM_MIN_BYTES: usize = 64 << 20;

/// GPU accelerated operation classes validated at runtime against a CPU oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccelOp {
    Matmul,
    Softmax,
    QuantQ8,
    FlashAttn,
    Conv2d,
}

impl AccelOp {
    pub const ALL: [AccelOp; 5] = [
        AccelOp::Matmul,
        AccelOp::Softmax,
        AccelOp::QuantQ8,
        AccelOp::FlashAttn,
        AccelOp::Conv2d,
    ];

    pub fn name(self) -> &'static str {
        match self {
            AccelOp::Matmul => "matmul",
            AccelOp::Softmax => "softmax",
            AccelOp::QuantQ8 => "q8_0",
            AccelOp::FlashAttn => "flash_attn",
            AccelOp::Conv2d => "conv2d",
        }
    }
}

fn device_ordinal(device: &Device) -> usize {
    use candle_core::DeviceLocation;
    match device.location() {
        DeviceLocation::Cpu => 0,
        DeviceLocation::Cuda { gpu_id }
        | DeviceLocation::Metal { gpu_id }
        | DeviceLocation::Rocm { gpu_id }
        | DeviceLocation::Vulkan { gpu_id }
        | DeviceLocation::OpenCl { gpu_id } => gpu_id,
    }
}

fn accel_cache() -> &'static Mutex<HashMap<(Backend, usize, AccelOp), bool>> {
    static CACHE: OnceLock<Mutex<HashMap<(Backend, usize, AccelOp), bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn accel_faults() -> &'static Mutex<HashMap<AccelOp, bool>> {
    static FAULTS: OnceLock<Mutex<HashMap<AccelOp, bool>>> = OnceLock::new();
    FAULTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fault-injection hook for tests: forces the self-check for `op` to panic inside `catch_unwind`
/// and cache a negative verdict until `reset_accel_cache()` is called.
pub fn set_accel_fault(op: AccelOp, fail: bool) {
    accel_faults().lock().unwrap_or_else(|p| p.into_inner()).insert(op, fail);
    if fail {
        accel_cache().lock().unwrap_or_else(|p| p.into_inner()).retain(|(_, _, o), _| *o != op);
    }
}

/// Clears cached per-device acceleration self-check verdicts and active fault injections.
pub fn reset_accel_cache() {
    accel_cache().lock().unwrap_or_else(|p| p.into_inner()).clear();
    accel_faults().lock().unwrap_or_else(|p| p.into_inner()).clear();
}

fn cpu_packed_override() -> &'static Mutex<Option<bool>> {
    static OVERRIDE: OnceLock<Mutex<Option<bool>>> = OnceLock::new();
    OVERRIDE.get_or_init(|| Mutex::new(None))
}

/// Test/runtime override for CPU packed-weight kernels (`None` defers to `NOTO_CPU_PACKED`).
pub fn set_cpu_packed_override(v: Option<bool>) {
    *cpu_packed_override().lock().unwrap_or_else(|p| p.into_inner()) = v;
}

/// Whether packed M-/thread-invariant CPU f32 kernels are enabled on `device`.
pub fn cpu_packed_enabled(device: &Device) -> bool {
    if !device.is_cpu() {
        return false;
    }
    if let Some(v) = *cpu_packed_override().lock().unwrap_or_else(|p| p.into_inner()) {
        return v;
    }
    std::env::var(CPU_PACKED_ENV).map_or(true, |v| v.trim() != "0" && !v.trim().eq_ignore_ascii_case("off"))
}

fn run_accel_probe(device: &Device, op: AccelOp) -> Result<bool> {
    use candle_core::{DType, Tensor};
    use candle_nn::ops::softmax_last_dim;

    if accel_faults().lock().unwrap_or_else(|p| p.into_inner()).get(&op).copied().unwrap_or(false) {
        panic!("injected accel fault for {op:?}");
    }
    const TOL: f32 = 1e-3;
    let same_tol = |on: Tensor, cpu: Tensor, tol: f32| -> Result<bool> {
        let on = on.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let cpu = cpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if on.len() != cpu.len() {
            return Ok(false);
        }
        Ok(on.iter().zip(&cpu).all(|(p, q)| p.is_finite() && q.is_finite() && (p - q).abs() <= tol * p.abs().max(q.abs()).max(1.0)))
    };
    let same = |on: Tensor, cpu: Tensor| same_tol(on, cpu, TOL);
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 31 + seed) % 97) as f32 / 48.0 - 1.0).collect()
    };

    match op {
        AccelOp::Matmul => {
            const N: usize = 32;
            let a = Tensor::from_vec(det(N * N, 1), (N, N), &Device::Cpu)?;
            let b = Tensor::from_vec(det(N * N, 5), (N, N), &Device::Cpu)?;
            let got = a.to_device(device)?.matmul(&b.to_device(device)?)?;
            same(got, a.matmul(&b)?)
        }
        AccelOp::Softmax => {
            const N: usize = 32;
            let a = Tensor::from_vec(det(N * N, 3), (N, N), &Device::Cpu)?;
            let got = softmax_last_dim(&a.to_device(device)?)?;
            same(got, softmax_last_dim(&a)?)
        }
        AccelOp::QuantQ8 => {
            use candle_core::quantized::GgmlDType;
            // Verify all 4 pyramid quantization formats (Q8_0, Q4K, Q5K, Q6K) on ROCm w8a16 / Vulkan GpuQuant / CUDA / Metal.
            let (rows, cols) = (32usize, 256usize);
            let w = det(rows * cols, 7);
            let x_cpu = Tensor::from_vec(det(4 * cols, 11), (1, 4, cols), &Device::Cpu)?;
            let x_dev = x_cpu.to_device(device)?;
            for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
                let bytes = crate::kquant::encode(&w, cols, dtype, None)?;
                let mut words = vec![0u64; bytes.len().div_ceil(8)];
                let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
                raw[..bytes.len()].copy_from_slice(&bytes);
                let qt_cpu = candle_core::quantized::ggml_file::qtensor_from_ggml(
                    dtype,
                    &raw[..bytes.len()],
                    vec![rows, cols],
                    &Device::Cpu,
                )?;
                let qt_dev = candle_core::quantized::ggml_file::qtensor_from_ggml(
                    dtype,
                    &raw[..bytes.len()],
                    vec![rows, cols],
                    device,
                )?;
                let ref_out = candle_core::quantized::QMatMul::from_qtensor(qt_cpu)?.forward_exact(&x_cpu)?;
                let qm_dev = candle_core::quantized::QMatMul::from_qtensor(qt_dev)?;
                // On candle@708b2f2, CUDA/ROCm/Metal run native exact kernels; Vulkan runs GpuQuant VRAM shaders via forward().
                let got = if matches!(device, Device::Vulkan(_)) {
                    candle_core::Module::forward(&qm_dev, &x_dev)?
                } else {
                    qm_dev.forward_exact(&x_dev)?
                };
                if !same_tol(got, ref_out, 5e-3)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        AccelOp::FlashAttn => {
            // Only probe fused GPU FlashAttention on backends that have native kernels (CUDA/ROCm/Vulkan).
            if !matches!(device, Device::Cuda(_) | Device::Rocm(_) | Device::Vulkan(_)) {
                return Ok(false);
            }
            let (b, s, h, d) = (1usize, 16usize, 4usize, 64usize);
            let q = Tensor::from_vec(det(b * s * h * d, 13), (b, s, h, d), &Device::Cpu)?;
            let k = Tensor::from_vec(det(b * s * h * d, 17), (b, s, h, d), &Device::Cpu)?;
            let v = Tensor::from_vec(det(b * s * h * d, 19), (b, s, h, d), &Device::Cpu)?;
            let scale = (d as f32).powf(-0.5);
            let cfg_strict = candle_nn::flash_attn::FlashAttnConfig::new(scale).fused(true).strict(true);
            let got = candle_nn::flash_attn::flash_attn_with_config(&q.to_device(device)?, &k.to_device(device)?, &v.to_device(device)?, &cfg_strict)?;
            let q_bhsd = q.transpose(1, 2)?.contiguous()?;
            let k_bhsd = k.transpose(1, 2)?.contiguous()?;
            let v_bhsd = v.transpose(1, 2)?.contiguous()?;
            let scores = q_bhsd.matmul(&k_bhsd.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
            let ref_out = softmax_last_dim(&scores)?.matmul(&v_bhsd)?.transpose(1, 2)?.contiguous()?;
            same(got, ref_out)
        }
        AccelOp::Conv2d => {
            let x = Tensor::from_vec(det(1 * 8 * 8 * 8, 23), (1, 8, 8, 8), &Device::Cpu)?;
            let w = Tensor::from_vec(det(8 * 8 * 9, 29), (8, 8, 3, 3), &Device::Cpu)?;
            let got = x.to_device(device)?.conv2d(&w.to_device(device)?, 1, 1, 1, 1)?;
            let ref_out = x.conv2d(&w, 1, 1, 1, 1)?;
            same(got, ref_out)
        }
    }
}

/// Checks whether `op` is safe and numerically accurate on `device`, caching the verdict per
/// `(backend, ordinal, op)` with `catch_unwind` isolation against panicking GPU drivers/kernels.
pub fn accel_op_ok(device: &Device, op: AccelOp) -> bool {
    let backend = Backend::of(device);
    if backend == Backend::Cpu {
        return false;
    }
    if op == AccelOp::FlashAttn {
        if let Ok(v) = std::env::var(FLASH_ATTN_ENV) {
            let t = v.trim();
            if t == "0" || t.eq_ignore_ascii_case("off") || t.eq_ignore_ascii_case("false") {
                return false;
            }
        }
    }
    if std::env::var(ACCEL_CHECK_ENV).is_ok_and(|v| v.trim() == "0" || v.trim().eq_ignore_ascii_case("off")) {
        return !accel_faults().lock().unwrap_or_else(|p| p.into_inner()).get(&op).copied().unwrap_or(false);
    }
    let key = (backend, device_ordinal(device), op);
    if let Some(&ok) = accel_cache().lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return ok;
    }
    let dev = device.clone();
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_accel_probe(&dev, op)))
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(false);
    accel_cache().lock().unwrap_or_else(|p| p.into_inner()).insert(key, ok);
    ok
}

/// One-line diagnostic summary of verified GPU ops on `device`.
pub fn accel_diagnose(device: &Device) -> String {
    if device.is_cpu() {
        return format!("CPU (packed={})", cpu_packed_enabled(device));
    }
    let parts: Vec<String> = AccelOp::ALL
        .iter()
        .map(|&op| format!("{}={}", op.name(), if accel_op_ok(device, op) { "ok" } else { "fallback" }))
        .collect();
    format!("{} [{}]", device_label(device), parts.join(", "))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

pub const GPU_ORDER: [Backend; 4] = [Backend::Cuda, Backend::Rocm, Backend::Vulkan, Backend::Metal];

impl Backend {
    pub const ALL: [Backend; 5] = [
        Backend::Cpu,
        Backend::Cuda,
        Backend::Metal,
        Backend::Vulkan,
        Backend::Rocm,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
        }
    }

    pub fn parse(s: &str) -> Option<Backend> {
        Backend::ALL
            .into_iter()
            .find(|b| b.name().eq_ignore_ascii_case(s.trim()))
    }

    pub fn of(device: &Device) -> Backend {
        match device {
            Device::Cuda(_) => Backend::Cuda,
            Device::Metal(_) => Backend::Metal,
            Device::Vulkan(_) => Backend::Vulkan,
            Device::Rocm(_) => Backend::Rocm,
            _ => Backend::Cpu,
        }
    }

    pub fn compiled(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Cuda => cfg!(feature = "cuda"),
            Backend::Metal => cfg!(feature = "metal"),
            Backend::Vulkan => cfg!(feature = "vulkan"),
            Backend::Rocm => cfg!(feature = "rocm"),
        }
    }

    pub fn device(self, ordinal: usize) -> Result<Device> {
        static CACHE: OnceLock<Mutex<HashMap<(Backend, usize), Device>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(d) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&(self, ordinal)) {
            return Ok(d.clone());
        }
        let d = self.open(ordinal)?;
        let mut g = cache.lock().unwrap_or_else(|p| p.into_inner());
        Ok(g.entry((self, ordinal)).or_insert(d).clone())
    }

    pub fn open(self, ordinal: usize) -> Result<Device> {
        match self {
            Backend::Cpu => Ok(Device::Cpu),
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let Ok(d) = std::panic::catch_unwind(|| Device::new_cuda(ordinal)) else {
                        bail!("CUDA device {ordinal} panicked on open");
                    };
                    return Ok(d?);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    bail!("this build has no CUDA support (enable the `cuda` feature)")
                }
            }
            Backend::Metal => {
                #[cfg(feature = "metal")]
                {
                    return open_metal(ordinal);
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = ordinal;
                    bail!("this build has no Metal support (enable the `metal` feature)")
                }
            }
            Backend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let Ok(d) = std::panic::catch_unwind(|| open_best_vulkan(ordinal)) else {
                        bail!("Vulkan device {ordinal} panicked on open");
                    };
                    return d;
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    let _ = ordinal;
                    bail!("this build has no Vulkan support (enable the `vulkan` feature)")
                }
            }
            Backend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    let Ok(d) = std::panic::catch_unwind(|| Device::new_rocm(ordinal)) else {
                        bail!("ROCm device {ordinal} panicked on open");
                    };
                    let d = d?;
                    ROCM_USED.store(true, Ordering::SeqCst);
                    ROCM_ORDINAL.store(ordinal, Ordering::SeqCst);
                    return Ok(d);
                }
                #[cfg(not(feature = "rocm"))]
                {
                    let _ = ordinal;
                    bail!("this build has no ROCm support (enable the `rocm` feature)")
                }
            }
        }
    }
}

#[cfg(feature = "vulkan")]
fn open_best_vulkan(ordinal: usize) -> Result<Device> {
    if ordinal != 0 {
        return Ok(Device::new_vulkan(ordinal)?);
    }
    let count = candle_core::vulkan_backend::device_count().unwrap_or(0).min(8);
    if count <= 1 {
        return Ok(Device::new_vulkan(0)?);
    }
    let mut best_hw: Option<(u64, Device)> = None;
    let mut best_cpu: Option<(u64, Device)> = None;
    for id in 0..count {
        if let Ok(d) = Device::new_vulkan(id) {
            let free = d.mem_info().ok().flatten().map_or(0, |(f, _)| f as u64);
            let is_cpu = d.as_vulkan_device().is_ok_and(|v| v.is_cpu_device());
            let slot = if is_cpu { &mut best_cpu } else { &mut best_hw };
            if slot.as_ref().is_none_or(|(b, _)| free > *b) {
                *slot = Some((free, d));
            }
        }
    }
    match best_hw.or(best_cpu) {
        Some((_, d)) => Ok(d),
        None => Ok(Device::new_vulkan(0)?),
    }
}

/// The Metal kernels assume an Apple GPU. Elsewhere (the Intel runner's paravirtual device) the device opens, copies
/// and means work, and a matmul or a softmax gives other numbers without an error.
#[cfg(feature = "metal")]
fn metal_kernels_hold(device: &Device) -> Result<bool> {
    use candle_core::Tensor;
    use candle_nn::ops::softmax_last_dim;

    const N: usize = 32;
    const TOL: f32 = 1e-3;
    let values = |seed: usize| (0..N * N).map(|i| ((i * 31 + seed) % 97) as f32 / 24.0 - 2.0).collect::<Vec<f32>>();
    let (a, b) = (Tensor::from_vec(values(1), (N, N), &Device::Cpu)?, Tensor::from_vec(values(5), (N, N), &Device::Cpu)?);
    let (x, y) = (a.to_device(device)?, b.to_device(device)?);
    let same = |on: Tensor, cpu: Tensor| -> Result<bool> {
        let (on, cpu): (Vec<f32>, Vec<f32>) = (on.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?, cpu.flatten_all()?.to_vec1()?);
        Ok(on.iter().zip(&cpu).all(|(p, q)| (p - q).abs() <= TOL * p.abs().max(q.abs()).max(1.0)))
    };
    Ok(same(x.matmul(&y)?, a.matmul(&b)?)? && same(softmax_last_dim(&x)?, softmax_last_dim(&a)?)?)
}

#[cfg(feature = "metal")]
fn open_metal(ordinal: usize) -> Result<Device> {
    // The fork panics where there is no device instead of returning the error.
    let Ok(device) = std::panic::catch_unwind(|| Device::new_metal(ordinal)) else {
        bail!("no Metal device");
    };
    let device = device?;
    if !metal_kernels_hold(&device)? {
        bail!("this Metal device gives other numbers than the CPU (the kernels assume an Apple GPU)");
    }
    Ok(device)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub backend: Backend,
    pub ordinal: usize,
    pub reason: String,
}

fn parse_request(raw: &str) -> Result<(Option<Backend>, usize)> {
    let raw = raw.trim().to_ascii_lowercase();
    let (name, ordinal) = match raw.split_once(':') {
        Some((n, o)) => (n.to_string(), o.trim().parse::<usize>().map_err(|_| anyhow::anyhow!("{DEVICE_ENV}={raw}: ordinal is not a number"))?),
        None => (raw.clone(), 0),
    };
    match name.as_str() {
        "" | "auto" | "gpu" => Ok((None, ordinal)),
        n => match Backend::parse(n) {
            Some(b) => Ok((Some(b), ordinal)),
            None => bail!("{DEVICE_ENV}={raw}: unknown backend (cpu, cuda, rocm, vulkan, metal, gpu)"),
        },
    }
}

pub fn select(request: Option<&str>) -> Selection {
    let mut notes = Vec::new();
    let (wanted, ordinal) = match request.map(parse_request) {
        Some(Ok(r)) => r,
        Some(Err(e)) => {
            notes.push(e.to_string());
            (None, 0)
        }
        None => (None, 0),
    };
    if let Some(b) = wanted {
        if b == Backend::Cpu {
            return Selection { backend: b, ordinal: 0, reason: format!("{DEVICE_ENV}=cpu") };
        }
        match b.device(ordinal) {
            Ok(_) => return Selection { backend: b, ordinal, reason: format!("{DEVICE_ENV}={}:{ordinal}", b.name()) },
            Err(e) => notes.push(format!("{} 장치 {ordinal}을 열 수 없음: {e}", b.name())),
        }
    }
    for b in GPU_ORDER {
        if !b.compiled() {
            continue;
        }
        match b.device(ordinal) {
            Ok(_) => {
                notes.push(format!("이 빌드의 첫 GPU 백엔드 {}:{ordinal}", b.name()));
                return Selection { backend: b, ordinal, reason: notes.join("; ") };
            }
            Err(e) => notes.push(format!("{}: {e}", b.name())),
        }
    }
    if notes.is_empty() {
        notes.push("GPU 기능 없이 빌드됨".into());
    }
    Selection { backend: Backend::Cpu, ordinal: 0, reason: format!("CPU ({})", notes.join("; ")) }
}

pub fn preferred() -> Selection {
    select(std::env::var(DEVICE_ENV).ok().as_deref())
}

pub fn device_label(device: &Device) -> String {
    let ord = device_ordinal(device);
    match device {
        Device::Cuda(_) => format!("CUDA:{ord}"),
        Device::Rocm(r) => match r.name() {
            Ok(n) if !n.is_empty() => format!("ROCm:{ord} ({n})"),
            _ => format!("ROCm:{ord}"),
        },
        Device::Vulkan(v) => format!("Vulkan:{ord} ({}, {})", v.name(), v.device_type()),
        Device::Metal(_) => if ord == 0 { "Metal".into() } else { format!("Metal:{ord}") },
        _ => "CPU".into(),
    }
}

pub fn device_family(device: &Device) -> &'static str {
    Backend::of(device).name()
}

pub fn mem_info(device: &Device) -> Option<(u64, u64)> {
    device.mem_info().ok().flatten().map(|(f, t)| (f as u64, t as u64))
}

pub fn trim_idle_pool(device: &Device) {
    match device {
        #[cfg(feature = "rocm")]
        Device::Rocm(rd) => {
            // Only trim ROCm pool when idle bytes exceed 320 MiB, keeping a 256 MiB hot pool
            // to avoid hipMallocAsync/hipFreeAsync stalls between inference frames.
            const HIGH_WATER: usize = 320 << 20;
            if rd.pool_idle_bytes().unwrap_or(0) >= HIGH_WATER {
                let _ = rd.trim_memory_pool_to(POOL_KEEP_BYTES);
            }
        }
        Device::Cuda(_) => {
            let _ = device.trim_memory_pool(POOL_KEEP_BYTES);
        }
        Device::Vulkan(_) => {
            if device.pool_idle_bytes().unwrap_or(0) >= VULKAN_TRIM_MIN_BYTES {
                let _ = device.trim_memory_pool(0);
            }
        }
        _ => {}
    }
}

/// Full GPU pool teardown: synchronizes, releases scratch workspaces, trims pool to 0,
/// and frees ROCm cached resources.
pub fn drain_gpu_pool(device: &Device) {
    if device.is_cpu() {
        return;
    }
    let _ = device.synchronize_all();
    let _ = device.release_workspaces();
    let _ = device.trim_memory_pool(0);
    #[cfg(feature = "rocm")]
    if let Device::Rocm(rd) = device {
        let _ = rd.release_cached_resources();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPolicy {
    pub max_rows: usize,
    pub token_budget: usize,
    pub fixed_shape: bool,
}

impl BatchPolicy {
    pub fn for_device(device: &Device) -> BatchPolicy {
        let gpu = !device.is_cpu();
        let mut p = BatchPolicy {
            max_rows: if gpu { 16 } else { 8 },
            token_budget: if gpu { 2048 } else { 1024 },
            fixed_shape: matches!(device, Device::Cuda(_) | Device::Rocm(_)),
        };
        if let Some(rows) = std::env::var(BATCH_ROWS_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|r| *r > 0) {
            p.max_rows = rows;
            p.token_budget = p.token_budget.max(rows);
        }
        if let Ok(v) = std::env::var(FIXED_SHAPE_ENV) {
            p.fixed_shape = v.trim() != "0";
        }
        p
    }

    pub fn image_rows(&self, tokens: usize) -> usize {
        (self.token_budget / tokens.max(1)).clamp(1, self.max_rows.max(1))
    }
}

static ROCM_USED: AtomicBool = AtomicBool::new(false);
static ROCM_ORDINAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn rocm_exit_now_if_used(code: u32) {
    #[cfg(all(windows, feature = "rocm"))]
    {
        use std::io::Write;
        if !ROCM_USED.load(Ordering::SeqCst) {
            return;
        }
        if std::env::var("NOTO_ROCM_FAST_EXIT").map(|v| v.trim() == "0").unwrap_or(false) {
            return;
        }
        let ord = ROCM_ORDINAL.load(Ordering::SeqCst);
        if let Ok(d) = Backend::Rocm.device(ord) {
            drain_gpu_pool(&d);
        }
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
            TerminateProcess(GetCurrentProcess(), code);
        }
    }
    let _ = code;
    let _ = (ROCM_USED.load(Ordering::Relaxed), ROCM_ORDINAL.load(Ordering::Relaxed));
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub backend: Backend,
    pub compiled: bool,
    pub available: bool,
    pub detail: String,
}

pub fn probe_backends() -> Vec<BackendStatus> {
    Backend::ALL
        .into_iter()
        .map(|backend| {
            let compiled = backend.compiled();
            if !compiled {
                return BackendStatus {
                    backend,
                    compiled,
                    available: false,
                    detail: "not compiled into this build".into(),
                };
            }
            match backend.device(0) {
                Ok(dev) => {
                    let mem = mem_info(&dev).map(|(f, t)| format!(", {:.1} / {:.1} GB free", f as f64 / 1e9, t as f64 / 1e9)).unwrap_or_default();
                    BackendStatus {
                        backend,
                        compiled,
                        available: true,
                        detail: format!("{}{mem}", device_label(&dev)),
                    }
                }
                Err(e) => BackendStatus {
                    backend,
                    compiled,
                    available: false,
                    detail: e.to_string(),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for b in Backend::ALL {
            assert_eq!(Backend::parse(b.name()), Some(b));
        }
        assert_eq!(Backend::parse("CUDA"), Some(Backend::Cuda));
        assert_eq!(Backend::parse("tpu"), None);
    }

    #[test]
    fn cpu_is_always_available() {
        assert!(Backend::Cpu.compiled());
        assert!(Backend::Cpu.open(0).is_ok());
        assert_eq!(Backend::of(&Backend::Cpu.device(0).unwrap()), Backend::Cpu);
    }

    #[test]
    fn uncompiled_backend_reports_missing_feature() {
        if !Backend::Rocm.compiled() {
            let err = Backend::Rocm.open(0).unwrap_err().to_string();
            assert!(err.contains("rocm"));
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn a_metal_device_opens_only_if_it_gives_the_cpus_numbers() {
        match Backend::Metal.open(0) {
            Ok(d) => {
                assert!(metal_kernels_hold(&d).unwrap());
                println!("[GPU] Metal opened: {}", device_label(&d));
            }
            Err(e) => println!("[GPU] Metal refused: {e}"),
        }
    }

    #[test]
    fn device_requests_parse_and_fall_back() {
        assert_eq!(parse_request("cuda:1").unwrap(), (Some(Backend::Cuda), 1));
        assert_eq!(parse_request(" GPU ").unwrap(), (None, 0));
        assert!(parse_request("tpu").is_err());
        assert!(parse_request("cuda:x").is_err());
        assert_eq!(select(Some("cpu")).backend, Backend::Cpu);
        let s = select(None);
        assert!(s.backend.device(s.ordinal).is_ok(), "{s:?}");
        if !Backend::Rocm.compiled() {
            let s = select(Some("rocm"));
            assert_ne!(s.backend, Backend::Rocm);
            assert!(s.reason.contains("rocm"), "{}", s.reason);
        }
    }

    #[test]
    fn the_cache_hands_out_one_device_per_backend() {
        let a = Backend::Cpu.device(0).unwrap();
        let b = Backend::Cpu.device(0).unwrap();
        assert!(a.same_device(&b));
        assert_eq!(device_label(&a), "CPU");
        assert_eq!(device_family(&a), "cpu");
        assert!(mem_info(&a).is_none());
        trim_idle_pool(&a);
    }

    #[test]
    fn batch_policy_is_dynamic_on_the_cpu() {
        let p = BatchPolicy::for_device(&Device::Cpu);
        if std::env::var(FIXED_SHAPE_ENV).is_err() {
            assert!(!p.fixed_shape);
        }
        assert!(p.image_rows(256) >= 1 && p.image_rows(256) <= p.max_rows);
        assert_eq!(p.image_rows(100_000), 1);
    }

    #[test]
    fn accel_check_and_fault_injection_work() {
        let require_gpu = std::env::var("NOTO_REQUIRE_GPU")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        assert!(!accel_op_ok(&Device::Cpu, AccelOp::FlashAttn));
        assert!(accel_diagnose(&Device::Cpu).starts_with("CPU"));
        let sel = select(None);
        if sel.backend == Backend::Cpu {
            assert!(!require_gpu, "NOTO_REQUIRE_GPU=1 but no GPU device opened ({})", sel.reason);
            return;
        }
        let dev = sel.backend.device(sel.ordinal).unwrap();
        reset_accel_cache();
        #[cfg(feature = "vulkan")]
        let vk_native_before = dev
            .as_vulkan_device()
            .is_ok_and(|v| !v.is_cpu_device())
            .then(candle_core::vulkan_backend::shaders::native_exec_count);

        println!("[ACCEL] {}", accel_diagnose(&dev));
        assert!(accel_op_ok(&dev, AccelOp::Matmul), "matmul should pass on {:?}", sel.backend);
        assert!(accel_op_ok(&dev, AccelOp::Softmax), "softmax should pass on {:?}", sel.backend);
        assert!(accel_op_ok(&dev, AccelOp::QuantQ8), "pyramid quant (Q8_0/Q4K/Q5K/Q6K) should pass on {:?}", sel.backend);
        assert!(accel_op_ok(&dev, AccelOp::FlashAttn), "flash_attn should pass on {:?}", sel.backend);
        assert!(accel_op_ok(&dev, AccelOp::Conv2d), "conv2d should pass on {:?}", sel.backend);

        #[cfg(feature = "vulkan")]
        if let Some(before) = vk_native_before {
            let after = candle_core::vulkan_backend::shaders::native_exec_count();
            assert!(
                after > before,
                "Vulkan hardware GPU must execute native GpuQuant/FlashAttn shaders (before={before}, after={after})"
            );
            println!("[ACCEL] Vulkan native shader dispatches: {}", after - before);
        }

        // Gate 5-A: fault injection must panic inside catch_unwind and persist negative cache until reset_accel_cache().
        set_accel_fault(AccelOp::FlashAttn, true);
        assert!(!accel_op_ok(&dev, AccelOp::FlashAttn), "fault injection must disable FlashAttn");
        set_accel_fault(AccelOp::FlashAttn, false);
        assert!(
            !accel_op_ok(&dev, AccelOp::FlashAttn),
            "negative verdict must stay cached after clearing fault until reset_accel_cache()"
        );
        reset_accel_cache();
        assert!(accel_op_ok(&dev, AccelOp::FlashAttn), "FlashAttn should pass again after reset_accel_cache()");
        drain_gpu_pool(&dev);
    }
}
