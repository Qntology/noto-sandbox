//! The SigLIP2 text tower's token table without the 786 MB on the device.
//!
//! `text_model.embeddings.token_embedding.weight` is 256,000 Gemma tokens × 768 F32. The table stays in the memory-
//! mapped safetensors file (no device memory, pages read when a row is needed); the device holds only the rows the
//! vocabulary uses: the special tokens, every token of the texts a caller primes it with (the lexicon's phrases, the
//! prompt templates and background classes, the dictionary, command endings), and any other token the moment a text
//! needs it (a free-form command). The tokenizer is the full one, so nothing becomes `<unk>`.
//!
//! A row is gathered with the same `index_select` the full `Embedding` uses, from the same F32 bytes, so the text
//! vectors are bit for bit those of the full table (the phrase bank's fingerprint does not change).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Module};
use serde::{Deserialize, Serialize};

/// `NOTO_VOCAB`: `active` (default: rows on demand) or `full` (the whole table on the device).
pub const VOCAB_ENV: &str = "NOTO_VOCAB";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VocabPolicy {
    /// The whole table on the device (the old load).
    Full,
    /// Only the rows in use on the device, the rest read from the mapped file when a text needs them.
    #[default]
    Active,
}

impl VocabPolicy {
    pub fn from_env() -> VocabPolicy {
        match std::env::var(VOCAB_ENV).map(|v| v.trim().to_ascii_lowercase()) {
            Ok(v) if v == "full" => VocabPolicy::Full,
            _ => VocabPolicy::Active,
        }
    }
}

/// F32 rows of a 2D tensor read straight from the mapped file.
pub struct MappedRows {
    st: MmapedSafetensors,
    name: String,
    rows: usize,
    cols: usize,
}

impl MappedRows {
    pub fn open(file: &Path, name: &str, rows: usize, cols: usize) -> Result<MappedRows> {
        let st = unsafe { MmapedSafetensors::new(file) }.with_context(|| format!("mapping {}", file.display()))?;
        {
            let view = st.get(name).with_context(|| format!("{name} in {}", file.display()))?;
            if format!("{:?}", view.dtype()) != "F32" || view.shape() != [rows, cols] || view.data().len() != rows * cols * 4 {
                bail!("{name} is {:?} {:?}, expected F32 [{rows}, {cols}]", view.dtype(), view.shape());
            }
        }
        Ok(MappedRows { st, name: name.to_string(), rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The rows `ids`, concatenated.
    pub fn read(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let view = self.st.get(&self.name)?;
        let data = view.data();
        let width = self.cols * 4;
        let mut out = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let id = id as usize;
            if id >= self.rows {
                bail!("token {id} is outside the {} rows of {}", self.rows, self.name);
            }
            out.extend(data[id * width..(id + 1) * width].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        Ok(out)
    }
}

#[derive(Default)]
struct Active {
    table: Option<Tensor>,
    index: HashMap<u32, u32>,
    added_on_demand: u64,
}

/// What the device holds of the token table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VocabStats {
    pub policy: VocabPolicy,
    pub vocab: usize,
    pub rows_on_device: usize,
    pub device_bytes: u64,
    /// Rows a text needed after priming (read from the file then).
    pub added_on_demand: u64,
}

/// The rows in use, on the device, and the mapped file the others come from.
pub struct ActiveTable {
    rows: Arc<MappedRows>,
    state: RwLock<Active>,
    device: Device,
}

/// The token embedding: the whole table, or the rows in use.
pub enum TokenTable {
    Full(Embedding, usize),
    Active(ActiveTable),
}

impl TokenTable {
    pub fn full(embedding: Embedding, vocab: usize) -> TokenTable {
        TokenTable::Full(embedding, vocab)
    }

    pub fn active(rows: MappedRows, device: &Device) -> TokenTable {
        TokenTable::Active(ActiveTable { rows: Arc::new(rows), state: RwLock::new(Active::default()), device: device.clone() })
    }

    fn add(state: &mut Active, rows: &MappedRows, device: &Device, ids: &[u32]) -> Result<usize> {
        let mut new: Vec<u32> = ids.iter().copied().filter(|id| !state.index.contains_key(id)).collect();
        new.sort_unstable();
        new.dedup();
        if new.is_empty() {
            return Ok(0);
        }
        let values = rows.read(&new)?;
        let block = Tensor::from_vec(values, (new.len(), rows.cols()), device)?;
        let base = state.index.len() as u32;
        state.table = Some(match state.table.take() {
            Some(t) => Tensor::cat(&[&t, &block], 0)?,
            None => block,
        });
        for (k, id) in new.iter().enumerate() {
            state.index.insert(*id, base + k as u32);
        }
        Ok(new.len())
    }

    /// Puts the rows of `ids` on the device ahead of use. Returns how many were new.
    pub fn prime(&self, ids: &[u32]) -> Result<usize> {
        match self {
            TokenTable::Full(..) => Ok(0),
            TokenTable::Active(a) => {
                let mut s = a.state.write().unwrap_or_else(|p| p.into_inner());
                Self::add(&mut s, &a.rows, &a.device, ids)
            }
        }
    }

    /// The embeddings of `ids` (row-major `shape`), `[.., dim]`.
    pub fn lookup(&self, ids: &[u32], shape: (usize, usize), device: &Device) -> Result<Tensor> {
        match self {
            TokenTable::Full(e, _) => Ok(e.forward(&Tensor::from_slice(ids, shape, device)?)?),
            TokenTable::Active(a) => {
                let missing = {
                    let s = a.state.read().unwrap_or_else(|p| p.into_inner());
                    ids.iter().any(|id| !s.index.contains_key(id))
                };
                if missing {
                    let mut s = a.state.write().unwrap_or_else(|p| p.into_inner());
                    let n = Self::add(&mut s, &a.rows, &a.device, ids)?;
                    s.added_on_demand += n as u64;
                }
                let s = a.state.read().unwrap_or_else(|p| p.into_inner());
                let idx: Vec<u32> = ids.iter().map(|id| s.index[id]).collect();
                let table = s.table.as_ref().ok_or_else(|| anyhow!("an empty token table"))?;
                let idx = Tensor::from_vec(idx, ids.len(), &a.device)?;
                Ok(table.index_select(&idx, 0)?.reshape((shape.0, shape.1, a.rows.cols()))?)
            }
        }
    }

    pub fn stats(&self) -> VocabStats {
        match self {
            TokenTable::Full(e, vocab) => {
                let (n, d) = e.embeddings().dims2().unwrap_or((0, 0));
                VocabStats { policy: VocabPolicy::Full, vocab: *vocab, rows_on_device: n, device_bytes: (n * d * 4) as u64, added_on_demand: 0 }
            }
            TokenTable::Active(a) => {
                let s = a.state.read().unwrap_or_else(|p| p.into_inner());
                let n = s.index.len();
                VocabStats { policy: VocabPolicy::Active, vocab: a.rows.rows(), rows_on_device: n, device_bytes: (n * a.rows.cols() * 4) as u64, added_on_demand: s.added_on_demand }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_bridge::kquant::testing::normals;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    /// One file per test: a mapped file cannot be rewritten on Windows while another test reads it.
    fn table_file(rows: usize, cols: usize) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("noto-ml-vocab-{rows}x{cols}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tok.safetensors");
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        vb.get((rows, cols), "tok.weight").unwrap();
        vm.data().lock().unwrap()["tok.weight"].set(&Tensor::from_vec(normals(rows * cols, 4), (rows, cols), &Device::Cpu).unwrap()).unwrap();
        vm.save(&path).unwrap();
        path
    }

    #[test]
    fn active_rows_give_the_full_tables_embeddings_bit_for_bit() -> Result<()> {
        let (rows, cols) = (300, 16);
        let path = table_file(rows, cols);
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &dev)? };
        let full = TokenTable::full(candle_nn::embedding(rows, cols, vb.pp("tok"))?, rows);
        let active = TokenTable::active(MappedRows::open(&path, "tok.weight", rows, cols)?, &dev);
        assert_eq!(active.prime(&[1, 7, 7, 250])?, 3, "duplicates are one row");
        let ids: Vec<u32> = vec![7, 1, 250, 299, 0, 7, 3, 1];
        let a = full.lookup(&ids, (2, 4), &dev)?;
        let b = active.lookup(&ids, (2, 4), &dev)?;
        assert_eq!(a.dims(), b.dims());
        assert_eq!(a.flatten_all()?.to_vec1::<f32>()?, b.flatten_all()?.to_vec1::<f32>()?);
        let s = active.stats();
        assert_eq!(s.rows_on_device, 6, "the primed rows and the ones this text needed");
        assert_eq!(s.added_on_demand, 3);
        assert_eq!(s.device_bytes, 6 * 16 * 4);
        assert_eq!(full.stats().rows_on_device, rows);
        assert!(active.lookup(&[rows as u32], (1, 1), &dev).is_err(), "a token outside the table is an error");
        Ok(())
    }

    #[test]
    fn a_table_of_another_shape_is_refused() {
        let path = table_file(10, 8);
        assert!(MappedRows::open(&path, "tok.weight", 10, 8).is_ok());
        assert!(MappedRows::open(&path, "tok.weight", 11, 8).is_err());
        assert!(MappedRows::open(&path, "nope", 10, 8).is_err());
    }
}
