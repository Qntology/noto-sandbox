//! Pyramid quantization straight from the F32 safetensors a model loads (no GGUF, no re-quantization).
//!
//! A [`QuantPlan`] gives every linear weight one of four tiers: F32 (tier 1), Q8_0 (tier 2), Q6_K or Q5_K (tier 3) and
//! Q4_K (tier 4). The plan names groups of tensors (vision attention in the middle layers, the MaskCLIP head, the DPT
//! readout, ...) so that a measurement can move one group at a time. [`QWeights`] wraps the model's `VarBuilder`: a
//! weight its plan keeps in F32 loads as before (`candle_nn::linear`, byte for byte the old model), any other one is read
//! as F32 from the memory-mapped original, quantized once by [`crate::kquant`] and multiplied by candle's quantized
//! matmul without quantizing the activations ([`QMatMul::forward_exact`]: f32 accumulation).
//!
//! Norms, biases, positions, patch projections, LayerScale vectors, the SigLIP temperature and bias, convolutions and
//! embedding tables never go through here: they stay F32. So does every weight of a backend whose quantized matmul
//! would not run on the device (Vulkan keeps Q8_0 only, see [`QuantPlan::for_device`]).

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::kquant;
use crate::parallel::{fnv1a, FNV_OFFSET};

/// `NOTO_QUANT`: the plan the model loaders use when the caller gives none (`f32`, `pyramid`, `q8`, or `<base>;<group>=<type>,...`).
pub const QUANT_ENV: &str = "NOTO_QUANT";
/// `NOTO_QUANT_CACHE=0`: disable the on-disk quantized weight cache beside the safetensors file.
pub const QUANT_CACHE_ENV: &str = "NOTO_QUANT_CACHE";

const QCACHE_MAGIC: &[u8; 8] = b"NOTOQC01";

fn qtype_code(q: QType) -> u8 {
    match q {
        QType::F32 => 0,
        QType::Q8_0 => 1,
        QType::Q6K => 2,
        QType::Q5K => 3,
        QType::Q4K => 4,
    }
}

fn qtype_from_code(c: u8) -> Option<QType> {
    match c {
        0 => Some(QType::F32),
        1 => Some(QType::Q8_0),
        2 => Some(QType::Q6K),
        3 => Some(QType::Q5K),
        4 => Some(QType::Q4K),
        _ => None,
    }
}

struct QuantDiskCache {
    path: PathBuf,
    file_len: u64,
    mtime_nanos: u128,
    plan_hash: u64,
    entries: HashMap<String, (QType, usize, usize, Vec<u8>)>,
    dirty: bool,
}

impl QuantDiskCache {
    fn enabled() -> bool {
        std::env::var(QUANT_CACHE_ENV).map_or(true, |v| v.trim() != "0" && !v.trim().eq_ignore_ascii_case("off"))
    }

    fn open(file: &Path, family: Family, plan: &QuantPlan) -> Option<Self> {
        if !Self::enabled() {
            return None;
        }
        let meta = std::fs::metadata(file).ok()?;
        let file_len = meta.len();
        let mtime_nanos = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        let parent = file.parent()?;
        let stem = file.file_stem()?.to_string_lossy();
        let fam = match family {
            Family::Siglip2 => "siglip2",
            Family::Tips => "tips",
        };
        let plan_hash = plan.hash();
        let cache_dir = parent.join(".qcache");
        let safe_tag: String = plan
            .tag
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let path = cache_dir.join(format!("{stem}_{fam}_{safe_tag}_{plan_hash:016x}.nqb"));
        let mut cache = Self {
            path,
            file_len,
            mtime_nanos,
            plan_hash,
            entries: HashMap::new(),
            dirty: false,
        };
        let _ = cache.try_load();
        Some(cache)
    }

    fn try_load(&mut self) -> Option<()> {
        let mut f = std::fs::File::open(&self.path).ok()?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).ok()?;
        if buf.len() < 8 + 8 + 16 + 8 + 4 + 8 {
            return None;
        }
        let payload_len = buf.len() - 8;
        let expected_hash = u64::from_le_bytes(buf[payload_len..].try_into().ok()?);
        if fnv1a(FNV_OFFSET, &buf[..payload_len]) != expected_hash {
            return None;
        }
        if &buf[0..8] != QCACHE_MAGIC {
            return None;
        }
        let mut pos = 8usize;
        let read_u64 = |b: &[u8], p: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let read_u128 = |b: &[u8], p: &mut usize| -> Option<u128> {
            let v = u128::from_le_bytes(b.get(*p..*p + 16)?.try_into().ok()?);
            *p += 16;
            Some(v)
        };
        let read_u32 = |b: &[u8], p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let flen = read_u64(&buf, &mut pos)?;
        let mtime = read_u128(&buf, &mut pos)?;
        let phash = read_u64(&buf, &mut pos)?;
        if flen != self.file_len || mtime != self.mtime_nanos || phash != self.plan_hash {
            return None;
        }
        let count = read_u32(&buf, &mut pos)? as usize;
        let mut entries = HashMap::with_capacity(count);
        for _ in 0..count {
            let name_len = read_u32(&buf, &mut pos)? as usize;
            let name = std::str::from_utf8(buf.get(pos..pos + name_len)?).ok()?.to_string();
            pos += name_len;
            let q = qtype_from_code(*buf.get(pos)?)?;
            pos += 1;
            let rows = read_u64(&buf, &mut pos)? as usize;
            let cols = read_u64(&buf, &mut pos)? as usize;
            let blen = read_u64(&buf, &mut pos)? as usize;
            let data = buf.get(pos..pos + blen)?.to_vec();
            pos += blen;
            entries.insert(name, (q, rows, cols, data));
        }
        if pos != payload_len {
            return None;
        }
        self.entries = entries;
        Some(())
    }

    fn get(&self, key: &str, q: QType, rows: usize, cols: usize) -> Option<&[u8]> {
        let (cq, cr, cc, data) = self.entries.get(key)?;
        if *cq == q && *cr == rows && *cc == cols {
            Some(data.as_slice())
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, q: QType, rows: usize, cols: usize, bytes: Vec<u8>) {
        self.entries.insert(key, (q, rows, cols, bytes));
        self.dirty = true;
    }

    fn flush(&mut self) {
        if !self.dirty || self.entries.is_empty() {
            return;
        }
        let Some(parent) = self.path.parent() else { return };
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(QCACHE_MAGIC);
        buf.extend_from_slice(&self.file_len.to_le_bytes());
        buf.extend_from_slice(&self.mtime_nanos.to_le_bytes());
        buf.extend_from_slice(&self.plan_hash.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (name, (q, rows, cols, data)) in &self.entries {
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.push(qtype_code(*q));
            buf.extend_from_slice(&(*rows as u64).to_le_bytes());
            buf.extend_from_slice(&(*cols as u64).to_le_bytes());
            buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
            buf.extend_from_slice(data);
        }
        let hash = fnv1a(FNV_OFFSET, &buf);
        buf.extend_from_slice(&hash.to_le_bytes());
        let tmp = self.path.with_extension(format!("tmp.{}", std::process::id()));
        if let Ok(mut f) = std::fs::File::create(&tmp) {
            if f.write_all(&buf).is_ok() && f.sync_all().is_ok() {
                drop(f);
                let _ = std::fs::rename(&tmp, &self.path);
                self.dirty = false;
            } else {
                drop(f);
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
}

/// The format of one weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    F32,
    Q8_0,
    Q6K,
    Q5K,
    Q4K,
}

impl QType {
    pub const ALL: [QType; 5] = [QType::F32, QType::Q8_0, QType::Q6K, QType::Q5K, QType::Q4K];

    pub fn name(self) -> &'static str {
        match self {
            QType::F32 => "f32",
            QType::Q8_0 => "q8_0",
            QType::Q6K => "q6k",
            QType::Q5K => "q5k",
            QType::Q4K => "q4k",
        }
    }

    pub fn parse(s: &str) -> Option<QType> {
        let s = s.trim().to_ascii_lowercase().replace('_', "");
        QType::ALL.into_iter().find(|q| q.name().replace('_', "") == s)
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            QType::F32 => GgmlDType::F32,
            QType::Q8_0 => GgmlDType::Q8_0,
            QType::Q6K => GgmlDType::Q6K,
            QType::Q5K => GgmlDType::Q5K,
            QType::Q4K => GgmlDType::Q4K,
        }
    }

    /// The pyramid tier: 1 lossless, 2 the 8-bit gates, 3 the 5–6-bit attention and FFN outputs, 4 the 4-bit expansions.
    pub fn tier(self) -> u8 {
        match self {
            QType::F32 => 1,
            QType::Q8_0 => 2,
            QType::Q6K | QType::Q5K => 3,
            QType::Q4K => 4,
        }
    }

    /// Stored bits per weight.
    pub fn bits(self) -> f32 {
        let g = self.ggml();
        g.type_size() as f32 * 8.0 / g.block_size() as f32
    }
}

/// Which model a weight belongs to (the group names differ).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Siglip2,
    Tips,
}

/// The group of a SigLIP2 weight (`vision_model.…` / `text_model.…`), or `None` for a tensor that is never quantized.
/// `sv`/`st` = vision/text tower; `edge` = the first two and the last three vision layers (the first and the last text
/// layer), `mid` = the rest.
pub fn siglip2_group(name: &str) -> Option<&'static str> {
    let vision = name.starts_with("vision_model.");
    if !vision && !name.starts_with("text_model.") {
        return None;
    }
    if !name.ends_with(".weight") && !name.ends_with("in_proj_weight") {
        return None;
    }
    if name.contains(".head.") || name.ends_with("head.weight") {
        return Some(if vision { "sv.head" } else { "st.head" });
    }
    let layer: usize = name.split(".layers.").nth(1)?.split('.').next()?.parse().ok()?;
    let edge = if vision { layer < 2 || layer >= 9 } else { layer == 0 || layer >= 11 };
    let part = if name.contains(".self_attn.") {
        "attn"
    } else if name.contains(".mlp.fc1.") {
        "fc1"
    } else if name.contains(".mlp.fc2.") {
        "fc2"
    } else {
        return None;
    };
    Some(match (vision, edge, part) {
        (true, true, "attn") => "sv.attn.edge",
        (true, true, _) => "sv.mlp.edge",
        (true, false, "attn") => "sv.attn.mid",
        (true, false, "fc1") => "sv.fc1.mid",
        (true, false, _) => "sv.fc2.mid",
        (false, true, "attn") => "st.attn.edge",
        (false, true, _) => "st.mlp.edge",
        (false, false, "attn") => "st.attn.mid",
        (false, false, "fc1") => "st.fc1.mid",
        (false, false, _) => "st.fc2.mid",
    })
}

/// The group of a TIPSv2 weight, or `None` (convolutions, norms, LayerScale, positions, the bin classifiers).
/// `edge` = blocks 0, 1, 10 and 11; `mid` = blocks 2–9.
pub fn tips_group(name: &str) -> Option<&'static str> {
    if !name.ends_with(".weight") {
        return None;
    }
    if name.contains(".reassemble.readout_projects.") {
        return Some("tv.readout");
    }
    let rest = name.strip_prefix("vision_encoder.blocks.")?;
    let block: usize = rest.split('.').next()?.parse().ok()?;
    let edge = block < 2 || block >= 10;
    let part = if rest.contains(".attn.") {
        "attn"
    } else if rest.contains(".mlp.fc1.") {
        "fc1"
    } else if rest.contains(".mlp.fc2.") {
        "fc2"
    } else {
        return None;
    };
    Some(match (edge, part) {
        (true, "attn") => "tv.attn.edge",
        (true, _) => "tv.mlp.edge",
        (false, "attn") => "tv.attn.mid",
        (false, "fc1") => "tv.fc1.mid",
        (false, _) => "tv.fc2.mid",
    })
}

/// Every group, with its tier in the instruction's pyramid (§2.2): the MaskCLIP head, the edge layers and the DPT
/// readout at 8 bits, attention and FFN outputs of the middle layers at 6 bits, their FFN expansions at 4 bits.
pub const PYRAMID: [(&str, QType); 17] = [
    ("sv.head", QType::Q8_0),
    ("sv.attn.edge", QType::Q8_0),
    ("sv.mlp.edge", QType::Q8_0),
    ("sv.attn.mid", QType::Q6K),
    ("sv.fc2.mid", QType::Q6K),
    ("sv.fc1.mid", QType::Q4K),
    ("st.head", QType::Q8_0),
    ("st.attn.edge", QType::Q8_0),
    ("st.mlp.edge", QType::Q8_0),
    ("st.attn.mid", QType::Q6K),
    ("st.fc2.mid", QType::Q6K),
    ("st.fc1.mid", QType::Q4K),
    ("tv.readout", QType::Q8_0),
    ("tv.attn.edge", QType::Q8_0),
    ("tv.mlp.edge", QType::Q8_0),
    ("tv.attn.mid", QType::Q6K),
    ("tv.fc2.mid", QType::Q6K),
];

/// The tier-4 group of TIPSv2 (kept out of [`PYRAMID`]'s fixed-size table for readability).
pub const PYRAMID_TIPS_FC1: (&str, QType) = ("tv.fc1.mid", QType::Q4K);

fn group_of(family: Family, name: &str) -> Option<&'static str> {
    match family {
        Family::Siglip2 => siglip2_group(name),
        Family::Tips => tips_group(name),
    }
}

/// A tier for every group. Groups it does not name stay F32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantPlan {
    /// What the plan is called in modes and fingerprints: empty for F32, else a preset name or `custom-<hash>`.
    pub tag: String,
    pub groups: BTreeMap<String, QType>,
}

impl Default for QuantPlan {
    fn default() -> Self {
        QuantPlan::f32()
    }
}

impl QuantPlan {
    pub fn f32() -> QuantPlan {
        QuantPlan { tag: String::new(), groups: BTreeMap::new() }
    }

    /// The instruction's four tiers (§2.2).
    pub fn pyramid() -> QuantPlan {
        let mut groups: BTreeMap<String, QType> = PYRAMID.iter().map(|(g, q)| (g.to_string(), *q)).collect();
        groups.insert(PYRAMID_TIPS_FC1.0.into(), PYRAMID_TIPS_FC1.1);
        QuantPlan { tag: "pyr".into(), groups }
    }

    /// Every quantizable weight at Q8_0.
    pub fn q8() -> QuantPlan {
        let mut groups: BTreeMap<String, QType> = PYRAMID.iter().map(|(g, _)| (g.to_string(), QType::Q8_0)).collect();
        groups.insert(PYRAMID_TIPS_FC1.0.into(), QType::Q8_0);
        QuantPlan { tag: "q8".into(), groups }
    }

    /// `f32`, `pyramid`, `q8`, optionally followed by `;group=type,group=type` (a group prefix such as `sv` or `tv.fc1`
    /// sets every group under it). A plan with overrides is tagged by a hash of what it ends up with.
    pub fn parse(spec: &str) -> Result<QuantPlan> {
        let mut parts = spec.split(';');
        let base = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        let mut plan = match base.as_str() {
            "" | "f32" | "off" | "none" => QuantPlan::f32(),
            "pyramid" | "pyr" => QuantPlan::pyramid(),
            "q8" | "q8_0" => QuantPlan::q8(),
            other => bail!("{QUANT_ENV}: unknown plan {other:?} (f32, pyramid, q8)"),
        };
        let mut changed = false;
        for part in parts {
            for item in part.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (g, t) = item.split_once('=').ok_or_else(|| anyhow!("{QUANT_ENV}: {item:?} is not group=type"))?;
                let q = QType::parse(t).ok_or_else(|| anyhow!("{QUANT_ENV}: unknown type {t:?} (f32, q8_0, q6k, q5k, q4k)"))?;
                let g = g.trim();
                let known: Vec<&str> = PYRAMID.iter().map(|(n, _)| *n).chain([PYRAMID_TIPS_FC1.0]).filter(|n| *n == g || n.starts_with(&format!("{g}."))).collect();
                if known.is_empty() {
                    bail!("{QUANT_ENV}: no group is called {g:?} or starts with it");
                }
                for n in known {
                    plan.groups.insert(n.to_string(), q);
                }
                changed = true;
            }
        }
        plan.groups.retain(|_, q| *q != QType::F32);
        if plan.groups.is_empty() {
            return Ok(QuantPlan::f32());
        }
        if changed {
            plan.tag = format!("custom-{:08x}", plan.hash() as u32);
        }
        Ok(plan)
    }

    /// The plan `NOTO_QUANT` names (F32 when it is not set).
    pub fn from_env() -> Result<QuantPlan> {
        match std::env::var(QUANT_ENV) {
            Ok(v) => QuantPlan::parse(&v),
            Err(_) => Ok(QuantPlan::f32()),
        }
    }

    /// The plan a model loads with on `device`: the one `NOTO_QUANT` names, else the measured default — every
    /// quantizable weight at Q8_0 on a GPU (the only allocation of the screen that kept the L1 and L2 numbers within
    /// the acceptance limits, `docs/research/ml-quant.md`), F32 on the CPU (no device memory to save, and the CPU
    /// dequantizes a block weight at every call).
    pub fn for_load(device: &Device) -> Result<QuantPlan> {
        match std::env::var(QUANT_ENV) {
            Ok(v) => QuantPlan::parse(&v),
            Err(_) if device.is_cpu() => Ok(QuantPlan::f32()),
            Err(_) => Ok(QuantPlan::q8()),
        }
    }

    pub fn is_f32(&self) -> bool {
        self.groups.is_empty()
    }

    /// A stable hash of the assignment (fingerprints, custom tags).
    pub fn hash(&self) -> u64 {
        let mut h = fnv1a(FNV_OFFSET, b"noto-quant-v1");
        for (g, q) in &self.groups {
            h = fnv1a(h, g.as_bytes());
            h = fnv1a(h, q.name().as_bytes());
        }
        h
    }

    /// The plan this device can run: Vulkan multiplies only Q8_0 on the device (the fork runs other block formats on
    /// the host), so its K-quant groups go to Q8_0.
    pub fn for_device(&self, device: &Device) -> QuantPlan {
        if matches!(device, Device::Vulkan(_)) {
            self.q8_only()
        } else {
            self.clone()
        }
    }

    /// Every K-quant group at Q8_0 instead (tagged apart when anything moved).
    pub fn q8_only(&self) -> QuantPlan {
        if self.groups.values().all(|q| matches!(q, QType::Q8_0)) {
            return self.clone();
        }
        let groups = self.groups.keys().map(|g| (g.clone(), QType::Q8_0)).collect();
        QuantPlan { tag: format!("{}-q8", self.tag), groups }
    }

    /// The type of weight `name` (`cols` input columns) of `family`. A width the format's block does not divide stays F32.
    pub fn qtype(&self, family: Family, name: &str, cols: usize) -> QType {
        let Some(q) = group_of(family, name).and_then(|g| self.groups.get(g).copied()) else { return QType::F32 };
        if q == QType::F32 || !cols.is_multiple_of(q.ggml().block_size()) {
            return QType::F32;
        }
        q
    }

    /// The part of a model's mode name that tells a quantized model's statistics apart (`""` for F32).
    pub fn mode_suffix(&self, family: Family) -> String {
        let prefix = match family {
            Family::Siglip2 => ["sv.", "st."],
            Family::Tips => ["tv.", "tv."],
        };
        if self.groups.keys().any(|g| prefix.iter().any(|p| g.starts_with(p))) {
            format!("+q-{}", self.tag)
        } else {
            String::new()
        }
    }

    /// Whether any weight of the given tower prefix (`sv`, `st`, `tv`) is quantized.
    pub fn touches(&self, prefix: &str) -> bool {
        self.groups.keys().any(|g| g.starts_with(&format!("{prefix}.")))
    }
}

/// What a load quantized, per format: tensors, the F32 bytes they replace and the bytes they take.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct QuantStats {
    pub tensors: BTreeMap<String, (u32, u64, u64)>,
    pub quantize_ms: f64,
}

impl QuantStats {
    fn add(&mut self, q: QType, f32_bytes: u64, bytes: u64) {
        let e = self.tensors.entry(q.name().to_string()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += f32_bytes;
        e.2 += bytes;
    }

    /// (F32 bytes of the linear weights, bytes they take after the plan).
    pub fn totals(&self) -> (u64, u64) {
        self.tensors.values().fold((0, 0), |(a, b), (_, f, q)| (a + f, b + q))
    }

    pub fn summary(&self) -> String {
        let (f, q) = self.totals();
        let parts: Vec<String> = self.tensors.iter().map(|(k, (n, _, b))| format!("{k} {n}×{:.1} MB", *b as f64 / 1e6)).collect();
        format!("linear {:.1} → {:.1} MB ({}), {:.1} s quantizing", f as f64 / 1e6, q as f64 / 1e6, parts.join(" · "), self.quantize_ms / 1e3)
    }
}

/// A linear layer whose weight is an F32 tensor, CPU packed NR=32 panels, or quantized blocks with an F32 bias.
pub enum QLinear {
    Dense(Linear),
    CpuPacked { w: Arc<noto_cpu_kernels::PackedW>, b: Option<Vec<f32>> },
    Quant { w: QMatMul, b: Option<Tensor>, use_native_fwd: bool },
}

impl QLinear {
    pub fn is_quantized(&self) -> bool {
        matches!(self, QLinear::Quant { .. })
    }

    pub fn is_cpu_packed(&self) -> bool {
        matches!(self, QLinear::CpuPacked { .. })
    }

    pub fn from_dense_packed(weight: &Tensor, bias: Option<&Tensor>) -> candle_core::Result<QLinear> {
        let (out_dim, in_dim) = weight.dims2()?;
        let w_flat = weight.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let packed = Arc::new(noto_cpu_kernels::PackedW::pack(&w_flat, out_dim, in_dim));
        let b = match bias {
            Some(bt) => Some(bt.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        Ok(QLinear::CpuPacked { w: packed, b })
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            QLinear::Dense(l) => l.forward(x),
            QLinear::CpuPacked { w, b } => {
                if !x.device().is_cpu() || x.dtype() != DType::F32 {
                    candle_core::bail!("QLinear::CpuPacked requires CPU F32 tensor, got {:?} {:?}", x.device(), x.dtype());
                }
                let dims = x.dims();
                if dims.is_empty() || dims[dims.len() - 1] != w.k {
                    candle_core::bail!("QLinear::CpuPacked input {:?} incompatible with k={}", dims, w.k);
                }
                let m: usize = dims[..dims.len() - 1].iter().product();
                let mut out_dims = dims.to_vec();
                *out_dims.last_mut().unwrap() = w.n;
                if m == 0 {
                    return Tensor::zeros(out_dims, DType::F32, x.device());
                }
                let x_c = x.contiguous()?;
                let (storage, layout) = x_c.storage_and_layout();
                let x_slice = match &*storage {
                    candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) => {
                        let start = layout.start_offset();
                        &v[start..start + m * w.k]
                    }
                    _ => unreachable!("checked CPU F32"),
                };
                let mut out = vec![0.0f32; m * w.n];
                let threads = rayon::current_num_threads().max(1);
                noto_cpu_kernels::gemm_bias(
                    x_slice,
                    m,
                    w,
                    b.as_deref(),
                    &mut out,
                    noto_cpu_kernels::Blocking::default(),
                    noto_cpu_kernels::Epilogue::Store,
                    threads,
                );
                Tensor::from_vec(out, out_dims, x.device())
            }
            QLinear::Quant { w, b, use_native_fwd } => {
                // Vulkan and Metal Q8_0 native kernels accept F32 activations directly without activation
                // quantization; CUDA and ROCm use `forward_exact` (f32 accumulation, tiled dequant).
                let y = if *use_native_fwd { w.forward(x)? } else { w.forward_exact(x)? };
                match b {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

/// The model's `VarBuilder` with the plan and a CPU view of the same safetensors file (the quantization source).
#[derive(Clone)]
pub struct QWeights<'a> {
    vb: VarBuilder<'a>,
    src: Option<Arc<MmapedSafetensors>>,
    cache: Option<Arc<Mutex<QuantDiskCache>>>,
    family: Family,
    plan: Arc<QuantPlan>,
    stats: Arc<Mutex<QuantStats>>,
    cpu_packed: bool,
}

impl<'a> QWeights<'a> {
    /// No quantization source: every weight as `vb` gives it (tests with a `VarMap`, F32 plans).
    pub fn plain(vb: VarBuilder<'a>, family: Family) -> QWeights<'a> {
        QWeights {
            vb,
            src: None,
            cache: None,
            family,
            plan: Arc::new(QuantPlan::f32()),
            stats: Arc::default(),
            cpu_packed: false,
        }
    }

    /// `vb` over `file`, which is also memory-mapped for the CPU side of the quantization when the plan needs it.
    pub fn with_plan(vb: VarBuilder<'a>, file: &Path, family: Family, plan: &QuantPlan) -> Result<QWeights<'a>> {
        let plan = plan.for_device(vb.device());
        let (src, cache) = if plan.is_f32() {
            (None, None)
        } else {
            let src = Some(Arc::new(
                unsafe { MmapedSafetensors::new(file) }.with_context(|| format!("mapping {}", file.display()))?,
            ));
            let cache = QuantDiskCache::open(file, family, &plan).map(|c| Arc::new(Mutex::new(c)));
            (src, cache)
        };
        Ok(QWeights {
            vb,
            src,
            cache,
            family,
            plan: Arc::new(plan),
            stats: Arc::default(),
            cpu_packed: false,
        })
    }

    /// Enables M-/thread-invariant CPU NR=32 packed f32 weights for dense linear layers on CPU.
    pub fn with_cpu_packed(mut self, on: bool) -> QWeights<'a> {
        self.cpu_packed = on && self.vb.device().is_cpu();
        self
    }

    pub fn is_cpu_packed(&self) -> bool {
        self.cpu_packed
    }

    /// Persists newly quantized weights to the on-disk `.qcache` file atomically (no-op if unchanged or disabled).
    pub fn flush_cache(&self) {
        if let Some(c) = &self.cache {
            c.lock().unwrap_or_else(|p| p.into_inner()).flush();
        }
    }

    pub fn pp<S: ToString>(&self, s: S) -> QWeights<'a> {
        QWeights { vb: self.vb.pp(s), ..self.clone() }
    }

    pub fn vb(&self) -> &VarBuilder<'a> {
        &self.vb
    }

    pub fn device(&self) -> &Device {
        self.vb.device()
    }

    pub fn plan(&self) -> &QuantPlan {
        &self.plan
    }

    pub fn stats(&self) -> QuantStats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The type weight `name` under this prefix (`cols` input columns) loads as (F32 without a quantization source).
    pub fn qtype(&self, name: &str, cols: usize) -> QType {
        if self.src.is_none() {
            return QType::F32;
        }
        self.plan.qtype(self.family, &self.full(name), cols)
    }

    fn full(&self, name: &str) -> String {
        let p = self.vb.prefix();
        if p.is_empty() {
            name.to_string()
        } else {
            format!("{p}.{name}")
        }
    }

    /// The F32 values of `rows` rows from `start` of the original 2D weight `name` (under this prefix).
    fn source_rows(&self, name: &str, start: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let src = self.src.as_ref().ok_or_else(|| anyhow!("no quantization source"))?;
        let full = self.full(name);
        let t = src.load(&full, &Device::Cpu).with_context(|| format!("reading {full}"))?.to_dtype(DType::F32)?;
        let (r, c) = t.dims2()?;
        if c != cols || start + rows > r {
            bail!("{full} is {r}×{c}, wanted rows {start}..{} of {cols} columns", start + rows);
        }
        Ok(t.narrow(0, start, rows)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn quantized_from_bytes(&self, bytes: &[u8], rows: usize, cols: usize, q: QType, bias: Option<Tensor>, elapsed_ms: f64) -> Result<QLinear> {
        // `qtensor_from_ggml` reads the bytes as blocks: give it an aligned buffer.
        let mut words = vec![0u64; bytes.len().div_ceil(8)];
        let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
        raw[..bytes.len()].copy_from_slice(bytes);
        let qt = qtensor_from_ggml(q.ggml(), &raw[..bytes.len()], vec![rows, cols], self.device())?;
        let mut s = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        s.add(q, (rows * cols * 4) as u64, bytes.len() as u64);
        s.quantize_ms += elapsed_ms;
        let use_native_fwd = matches!(self.device(), Device::Vulkan(_) | Device::Metal(_));
        Ok(QLinear::Quant { w: QMatMul::from_qtensor(qt)?, b: bias, use_native_fwd })
    }

    fn quantized_cached<F>(&self, cache_key: String, rows: usize, cols: usize, q: QType, bias: Option<Tensor>, load_f32: F) -> Result<QLinear>
    where
        F: FnOnce() -> Result<Vec<f32>>,
    {
        let t0 = Instant::now();
        if let Some(c) = &self.cache {
            let hit = c
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get(&cache_key, q, rows, cols)
                .map(|s| s.to_vec());
            if let Some(bytes) = hit {
                return self.quantized_from_bytes(&bytes, rows, cols, q, bias, t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let values = load_f32()?;
        let bytes = kquant::encode(&values, cols, q.ggml(), None)?;
        if let Some(c) = &self.cache {
            c.lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(cache_key, q, rows, cols, bytes.clone());
        }
        self.quantized_from_bytes(&bytes, rows, cols, q, bias, t0.elapsed().as_secs_f64() * 1e3)
    }

    /// Counts an F32 linear weight a layer loaded by itself (MapHead's packed projection) in the stats.
    pub fn record_dense(&self, rows: usize, cols: usize) {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).add(QType::F32, (rows * cols * 4) as u64, (rows * cols * 4) as u64);
    }

    fn dense_linear(&self, w: Tensor, b: Option<Tensor>) -> Result<QLinear> {
        if self.cpu_packed {
            Ok(QLinear::from_dense_packed(&w, b.as_ref())?)
        } else {
            Ok(QLinear::Dense(Linear::new(w, b)))
        }
    }

    /// `candle_nn::linear(in_dim, out_dim)` under this prefix, quantized or CPU-packed when the plan says so.
    pub fn linear(&self, in_dim: usize, out_dim: usize) -> Result<QLinear> {
        let q = self.plan.qtype(self.family, &self.full("weight"), in_dim);
        if q == QType::F32 || self.src.is_none() {
            self.record_dense(out_dim, in_dim);
            if self.cpu_packed {
                let w = self.vb.get((out_dim, in_dim), "weight")?;
                let b = self.vb.get(out_dim, "bias")?;
                return self.dense_linear(w, Some(b));
            }
            return Ok(QLinear::Dense(candle_nn::linear(in_dim, out_dim, self.vb.clone())?));
        }
        let b = self.vb.get(out_dim, "bias")?;
        let key = self.full("weight");
        self.quantized_cached(key, out_dim, in_dim, q, Some(b), || self.source_rows("weight", 0, out_dim, in_dim))
    }

    /// One linear layer over the concatenated outputs of `parts` (each `out_each` × `in_dim`, with bias): the fused
    /// QKV projection. The plan's type is the first part's (the parts share a group).
    pub fn linear_concat(&self, parts: &[&str], in_dim: usize, out_each: usize) -> Result<QLinear> {
        let first = parts.first().ok_or_else(|| anyhow!("no parts to concatenate"))?;
        let q = self.plan.qtype(self.family, &self.pp(first).full("weight"), in_dim);
        let biases = parts.iter().map(|p| self.vb.pp(p).get(out_each, "bias")).collect::<candle_core::Result<Vec<_>>>()?;
        let b = Tensor::cat(&biases, 0)?;
        if q == QType::F32 || self.src.is_none() {
            let ws = parts.iter().map(|p| self.vb.pp(p).get((out_each, in_dim), "weight")).collect::<candle_core::Result<Vec<_>>>()?;
            self.record_dense(out_each * parts.len(), in_dim);
            return self.dense_linear(Tensor::cat(&ws, 0)?, Some(b));
        }
        let key = format!("{}.concat[{}]", self.vb.prefix(), parts.join(","));
        self.quantized_cached(key, parts.len() * out_each, in_dim, q, Some(b), || {
            let mut w = Vec::with_capacity(parts.len() * out_each * in_dim);
            for p in parts {
                w.extend(self.pp(p).source_rows("weight", 0, out_each, in_dim)?);
            }
            Ok(w)
        })
    }

    /// Rows `start..start + rows` of the 2D weight `w_name` with the same rows of the bias `b_name` (MapHead's packed
    /// `in_proj`). Dense: the F32 rows, transposed once here.
    pub fn linear_rows(&self, w_name: &str, b_name: &str, total_rows: usize, start: usize, rows: usize, in_dim: usize) -> Result<QLinear> {
        let q = self.plan.qtype(self.family, &self.full(w_name), in_dim);
        let b = self.vb.get(total_rows, b_name)?.narrow(0, start, rows)?.contiguous()?;
        if q == QType::F32 || self.src.is_none() {
            let w = self.vb.get((total_rows, in_dim), w_name)?.narrow(0, start, rows)?.contiguous()?;
            self.record_dense(rows, in_dim);
            return self.dense_linear(w, Some(b));
        }
        let key = format!("{}[{start}..{}]", self.full(w_name), start + rows);
        self.quantized_cached(key, rows, in_dim, q, Some(b), || self.source_rows(w_name, start, rows, in_dim))
    }

    /// Splits a packed `[parts * rows, in_dim]` weight + bias into `parts` [`QLinear`]s in a single file load.
    pub fn linear_split_rows(&self, w_name: &str, b_name: &str, parts: usize, rows: usize, in_dim: usize) -> Result<Vec<QLinear>> {
        let total_rows = parts * rows;
        let q = self.plan.qtype(self.family, &self.full(w_name), in_dim);
        let b_all = self.vb.get(total_rows, b_name)?;
        if q == QType::F32 || self.src.is_none() {
            let w_all = self.vb.get((total_rows, in_dim), w_name)?;
            let mut out = Vec::with_capacity(parts);
            for i in 0..parts {
                let w = w_all.narrow(0, i * rows, rows)?.contiguous()?;
                let b = b_all.narrow(0, i * rows, rows)?.contiguous()?;
                self.record_dense(rows, in_dim);
                out.push(self.dense_linear(w, Some(b))?);
            }
            return Ok(out);
        }
        let mut loaded: Option<Vec<f32>> = None;
        let mut out = Vec::with_capacity(parts);
        for i in 0..parts {
            let start = i * rows;
            let b = b_all.narrow(0, start, rows)?.contiguous()?;
            let key = format!("{}[{start}..{}]", self.full(w_name), start + rows);
            let layer = self.quantized_cached(key, rows, in_dim, q, Some(b), || {
                if loaded.is_none() {
                    loaded = Some(self.source_rows(w_name, 0, total_rows, in_dim)?);
                }
                let all = loaded.as_ref().unwrap();
                Ok(all[start * in_dim..(start + rows) * in_dim].to_vec())
            })?;
            out.push(layer);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquant::testing::normals;
    use candle_nn::VarMap;

    #[test]
    fn groups_follow_the_tier_table() {
        assert_eq!(siglip2_group("vision_model.encoder.layers.5.mlp.fc1.weight"), Some("sv.fc1.mid"));
        assert_eq!(siglip2_group("vision_model.encoder.layers.10.self_attn.q_proj.weight"), Some("sv.attn.edge"));
        assert_eq!(siglip2_group("vision_model.encoder.layers.1.mlp.fc2.weight"), Some("sv.mlp.edge"));
        assert_eq!(siglip2_group("vision_model.head.attention.in_proj_weight"), Some("sv.head"));
        assert_eq!(siglip2_group("vision_model.head.mlp.fc1.weight"), Some("sv.head"));
        assert_eq!(siglip2_group("text_model.head.weight"), Some("st.head"));
        assert_eq!(siglip2_group("text_model.encoder.layers.11.mlp.fc1.weight"), Some("st.mlp.edge"));
        assert_eq!(siglip2_group("text_model.encoder.layers.4.self_attn.out_proj.weight"), Some("st.attn.mid"));
        for never in [
            "vision_model.embeddings.patch_embedding.weight",
            "text_model.embeddings.token_embedding.weight",
            "vision_model.encoder.layers.3.layer_norm1.weight",
            "vision_model.encoder.layers.3.mlp.fc1.bias",
            "vision_model.post_layernorm.weight",
            "logit_scale",
        ] {
            assert_eq!(siglip2_group(never), None, "{never}");
        }
        assert_eq!(tips_group("vision_encoder.blocks.4.mlp.fc1.weight"), Some("tv.fc1.mid"));
        assert_eq!(tips_group("vision_encoder.blocks.11.attn.qkv.weight"), Some("tv.attn.edge"));
        assert_eq!(tips_group("depth_head.reassemble.readout_projects.2.weight"), Some("tv.readout"));
        for never in ["vision_encoder.blocks.4.ls1.gamma", "vision_encoder.patch_embed.proj.weight", "depth_head.fusion_blocks.0.main_unit.conv1.weight", "depth_head.depth_head.weight", "vision_encoder.pos_embed"] {
            assert_eq!(tips_group(never), None, "{never}");
        }
    }

    #[test]
    fn plans_parse_and_tag_what_they_end_up_with() {
        assert!(QuantPlan::parse("f32").unwrap().is_f32());
        assert_eq!(QuantPlan::parse("pyramid").unwrap(), QuantPlan::pyramid());
        let p = QuantPlan::parse("f32;sv.fc1=q4k").unwrap();
        assert_eq!(p.groups.len(), 1, "sv.fc1 covers sv.fc1.mid only: {:?}", p.groups);
        let p = QuantPlan::parse("f32;sv=q8_0").unwrap();
        assert_eq!(p.groups.len(), 6, "every vision group: {:?}", p.groups);
        assert!(p.tag.starts_with("custom-"));
        assert_ne!(p.tag, QuantPlan::parse("f32;sv=q6k").unwrap().tag, "a different assignment, a different tag");
        assert_eq!(p.tag, QuantPlan::parse("f32;sv=q8_0").unwrap().tag, "the same assignment, the same tag");
        assert!(QuantPlan::parse("pyramid;tv=f32,sv=f32,st=f32").unwrap().is_f32(), "all back to F32 is the F32 plan");
        assert!(QuantPlan::parse("pyramid;nope=q8_0").is_err());
        assert!(QuantPlan::parse("pyramid;sv=q3k").is_err());
        assert!(QuantPlan::parse("bogus").is_err());
        let pyr = QuantPlan::pyramid();
        assert_eq!(pyr.qtype(Family::Siglip2, "vision_model.encoder.layers.5.mlp.fc1.weight", 768), QType::Q4K);
        assert_eq!(pyr.qtype(Family::Siglip2, "vision_model.encoder.layers.5.mlp.fc1.weight", 100), QType::F32, "a width the block does not divide stays F32");
        assert_eq!(pyr.mode_suffix(Family::Tips), "+q-pyr");
        assert_eq!(QuantPlan::parse("f32;sv=q8_0").unwrap().mode_suffix(Family::Tips), "", "a SigLIP2-only plan leaves the depth model's mode alone");
        assert_eq!(QuantPlan::f32().mode_suffix(Family::Siglip2), "");
    }

    #[test]
    fn the_default_plan_is_q8_on_a_gpu_and_f32_on_the_cpu() {
        if std::env::var(QUANT_ENV).is_err() {
            assert!(QuantPlan::for_load(&Device::Cpu).unwrap().is_f32(), "no device memory to save, and the CPU dequantizes every call");
        }
        let q8 = QuantPlan::q8();
        assert!(q8.groups.values().all(|q| *q == QType::Q8_0));
        assert_eq!(q8.groups.len(), PYRAMID.len() + 1, "every group the pyramid names");
        assert_eq!(q8.mode_suffix(Family::Siglip2), "+q-q8");
    }

    #[test]
    fn vulkan_keeps_q8_0_only() {
        let p = QuantPlan::pyramid();
        assert_eq!(p.for_device(&Device::Cpu), p, "the CPU runs every format");
        let v = p.q8_only();
        assert!(v.groups.values().all(|q| *q == QType::Q8_0) && v.groups.len() == p.groups.len());
        assert_eq!(v.tag, "pyr-q8", "statistics of the Vulkan plan stay apart");
        let q8 = QuantPlan::q8();
        assert_eq!(q8.q8_only(), q8);
    }

    fn file_with(vars: &[(&str, Vec<usize>)], seed: u64) -> (std::path::PathBuf, VarMap) {
        let dir = std::env::temp_dir().join(format!("noto-ml-quant-{seed}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.safetensors");
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        for (k, (name, shape)) in vars.iter().enumerate() {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = normals(n, seed + k as u64).into_iter().map(|x| x * 0.05).collect();
            vb.get(shape.clone(), name).unwrap();
            vm.data().lock().unwrap()[*name].set(&Tensor::from_vec(v, shape.clone(), &Device::Cpu).unwrap()).unwrap();
        }
        vm.save(&path).unwrap();
        (path, vm)
    }

    #[test]
    fn a_quantized_linear_is_close_and_the_f32_plan_is_the_old_layer() -> Result<()> {
        let pre = "vision_model.encoder.layers.4.mlp.fc1";
        let (path, _) = file_with(&[(&format!("{pre}.weight"), vec![512, 256]), (&format!("{pre}.bias"), vec![512])], 3);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu)? };
        let x = Tensor::from_vec(normals(2 * 7 * 256, 99), (2, 7, 256), &Device::Cpu)?;
        let old = candle_nn::linear(256, 512, vb.pp(pre))?.forward(&x)?;
        let same = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &QuantPlan::f32())?.pp(pre).linear(256, 512)?;
        assert!(!same.is_quantized());
        assert_eq!(same.forward(&x)?.flatten_all()?.to_vec1::<f32>()?, old.flatten_all()?.to_vec1::<f32>()?, "F32 plan = the old layer, bit for bit");
        let mut last = f32::INFINITY;
        for q in [QType::Q4K, QType::Q5K, QType::Q6K, QType::Q8_0] {
            let plan = QuantPlan::parse(&format!("f32;sv.fc1={}", q.name()))?;
            let w = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &plan)?;
            let l = w.pp(pre).linear(256, 512)?;
            assert!(l.is_quantized(), "{q:?}");
            let got = l.forward(&x)?;
            assert_eq!(got.dims(), old.dims());
            let err = kquant::rel(&got.flatten_all()?.to_vec1::<f32>()?, &old.flatten_all()?.to_vec1::<f32>()?);
            assert!(err < last, "{q:?}: more bits, less error ({err} after {last})");
            assert!(err < 0.12, "{q:?}: output error {err}");
            last = err;
            let (f, b) = w.stats().totals();
            assert_eq!(f, 512 * 256 * 4);
            assert!((b as f32 / f as f32 - q.bits() / 32.0).abs() < 1e-3, "{q:?}: {b} bytes");
        }
        Ok(())
    }

    #[test]
    fn fused_and_row_slices_match_their_parts() -> Result<()> {
        let pre = "vision_model.encoder.layers.3.self_attn";
        let mut vars: Vec<(String, Vec<usize>)> = Vec::new();
        for p in ["q_proj", "k_proj", "v_proj"] {
            vars.push((format!("{pre}.{p}.weight"), vec![256, 256]));
            vars.push((format!("{pre}.{p}.bias"), vec![256]));
        }
        vars.push(("vision_model.head.attention.in_proj_weight".into(), vec![768, 256]));
        vars.push(("vision_model.head.attention.in_proj_bias".into(), vec![768]));
        let refs: Vec<(&str, Vec<usize>)> = vars.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
        let (path, _) = file_with(&refs, 21);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu)? };
        let x = Tensor::from_vec(normals(5 * 256, 5), (1, 5, 256), &Device::Cpu)?;
        for plan in [QuantPlan::f32(), QuantPlan::parse("f32;sv.attn=q8_0,sv.head=q8_0")?] {
            let w = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &plan)?;
            let fused = w.pp(pre).linear_concat(&["q_proj", "k_proj", "v_proj"], 256, 256)?.forward(&x)?;
            for (i, p) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                let one = w.pp(pre).pp(p).linear(256, 256)?.forward(&x)?;
                let part = fused.narrow(2, i * 256, 256)?;
                let err = kquant::rel(&part.flatten_all()?.to_vec1::<f32>()?, &one.flatten_all()?.to_vec1::<f32>()?);
                assert!(err < 1e-5, "{p} of the fused projection ({:?}): {err}", plan.tag);
            }
            let head = w.pp("vision_model.head.attention");
            let hv = vb.pp("vision_model.head.attention");
            let full = Linear::new(hv.get((768, 256), "in_proj_weight")?, Some(hv.get(768, "in_proj_bias")?)).forward(&x)?;
            for k in 0..3 {
                let rows = head.linear_rows("in_proj_weight", "in_proj_bias", 768, k * 256, 256, 256)?.forward(&x)?;
                let err = kquant::rel(&rows.flatten_all()?.to_vec1::<f32>()?, &full.narrow(2, k * 256, 256)?.flatten_all()?.to_vec1::<f32>()?);
                let bound = if plan.is_f32() { 1e-6 } else { 0.02 };
                assert!(err < bound, "rows {k} ({:?}): {err}", plan.tag);
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_packed_linear_is_m_invariant_and_qcache_roundtrips() -> Result<()> {
        let pre = "vision_model.encoder.layers.4.mlp.fc1";
        let (path, _) = file_with(&[(&format!("{pre}.weight"), vec![128, 64]), (&format!("{pre}.bias"), vec![128])], 77);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu)? };
        let x4 = Tensor::from_vec(normals(4 * 64, 123), (4, 64), &Device::Cpu)?;
        let x1 = x4.narrow(0, 2, 1)?.contiguous()?;
        let ref_lin = candle_nn::linear(64, 128, vb.pp(pre))?;
        let packed = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &QuantPlan::f32())?
            .with_cpu_packed(true)
            .pp(pre)
            .linear(64, 128)?;
        assert!(packed.is_cpu_packed());
        let y4 = packed.forward(&x4)?;
        let y1 = packed.forward(&x1)?;
        let row2 = y4.narrow(0, 2, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let single = y1.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(row2, single, "CpuPacked must be bit-identical between M=1 and M=4");
        let y_ref = ref_lin.forward(&x4)?.flatten_all()?.to_vec1::<f32>()?;
        let max_diff = y4
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "CpuPacked vs Dense max_diff={max_diff}");

        // Verify disk .qcache write + reload produces bit-identical Q8_0 outputs.
        let plan = QuantPlan::parse("f32;sv.fc1=q8_0")?;
        let w1 = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &plan)?;
        let l1 = w1.pp(pre).linear(64, 128)?;
        w1.flush_cache();
        let w2 = QWeights::with_plan(vb.clone(), &path, Family::Siglip2, &plan)?;
        let l2 = w2.pp(pre).linear(64, 128)?;
        assert_eq!(
            l1.forward(&x4)?.flatten_all()?.to_vec1::<f32>()?,
            l2.forward(&x4)?.flatten_all()?.to_vec1::<f32>()?,
            "cached Q8_0 reload must be bit-identical"
        );
        Ok(())
    }
}
