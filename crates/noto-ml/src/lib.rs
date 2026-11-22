pub mod classify;
pub mod config;
pub mod device;
pub mod encoder;
pub mod kquant;
pub mod model;
pub mod parallel;
pub mod preprocess;
pub mod quant;
pub mod text;
pub mod tokenizer;
pub mod vision;
pub mod vocab;

pub use classify::{classify, Anchors, DictEntry, Dictionary, Prediction};
pub use config::Siglip2Config;
pub use device::{
    accel_diagnose, accel_op_ok, cpu_packed_enabled, device_label, preferred, probe_backends, reset_accel_cache,
    select, set_accel_fault, set_cpu_packed_override, trim_idle_pool, AccelOp, Backend, BackendStatus, BatchPolicy,
    Selection,
};
pub use model::{DenseEmbedding, LoadOpts, Siglip2, SparseDense};
pub use parallel::{BatchOpts, ImageEmbedder, PhraseBank, TextEmbedder, Tile};
pub use quant::{QLinear, QType, QWeights, QuantPlan, QuantStats};
pub use vocab::{VocabPolicy, VocabStats};
