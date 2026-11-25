pub mod ml_bridge;
pub mod sds;
pub mod ts;
pub mod vision_compat;

mod actsds;
mod benchsds;
pub mod api;
pub mod cargo;
pub mod chase;
mod api_test;
pub mod config;
pub mod crew;
mod driver;
pub mod embedding;
pub mod feedback;
pub mod foreuse;
pub mod mcp;
pub mod ml;
mod mldrive;
mod mlsds;
pub mod nnlive;
pub mod perception;
pub mod precision;
pub mod progress;
pub mod record;
pub mod rigs;
pub mod runtime;
pub mod sdsview;
pub mod stereo;
pub mod testrun;
pub mod tshub;
pub mod views;
pub mod vision;
pub mod vocab;

pub use config::{CameraCfg, GateCfg, NoiseCfg, RuntimeConfig};
pub use embedding::{BankedPhrases, EmbeddingCfg, EmbeddingPerception, SiglipPhrases};
pub use perception::{Candidate, EntKind, EntityTable, OraclePerception, Recognizer, ScanInput, ScanOutput};
pub use precision::{GateCounters, GateResult, Judged, Landmark, LmState, PrecisionGates};
pub use ml::{DepthEngine, LocateEngine, MlPipeline};
pub use record::{RecordCfg, Recording};
pub use testrun::{TestCfg, TestPhase, TestSession};
pub use vision::{DriftRate, Memory, VisionCfg, VisionMark};
pub use runtime::{label_ko, Detection, EvalCounters, EvalSummary, FrameOut, FrameView, Runtime, ScanReport, Submission, TsView};
pub use vocab::vocab_texts;

/// Automatic 12-hex SHA-256 fingerprint of engine source files (Phase 2 – Part 3 · D2).
pub const ENGINE_FP: &str = env!("NOTO_ENGINE_FP");

/// Combined generation identifier `<ENGINE_REV>+<ENGINE_FP>`.
pub fn generation_id() -> String {
    format!("{}+{}", noto_core::bench::ENGINE_REV, ENGINE_FP)
}

pub fn world_terrain_kind(id: &str) -> Result<world_sim::TerrainKind, String> {
    world_sim::TerrainKind::from_id(id).ok_or_else(|| format!("알 수 없는 지형 '{id}' (open 또는 rugged)"))
}
