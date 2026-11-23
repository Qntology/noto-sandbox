pub mod audit;
pub mod catalog;
pub mod estimate;
pub mod pool;
pub mod records;
pub mod schema;
pub mod scope;
pub mod stats;
pub mod store;

pub use audit::{verify_dir, verify_file, AnchorEntry, AuditLog, Level as AuditLevel, LogLine, SessionVerifyReport, VerifyError, VerifyOk};
pub use audit::Level;
pub use catalog::{classify, is_control_allowed, lookup, AxisSpec, ContactLog, Provenance, Reader, AXES};
pub use estimate::{chi2_quantile, effective_n, factor_ci, mean_ci, norm_quantile, pred_lower, spread_ci, t_quantile, welch_overlap, wilson_ci};
pub use pool::{heterogeneity_i2, pooled_agreement, pooled_baseline, Pooled};
pub use records::{BalanceRow, KeyframeRow, MirrorBatch, NnPhaseRow, QuestRun, StepRecord, SuccessStat, TestDb, FILE_NN_PHASES, WHOLE_QUEST};
pub use schema::{chain_keys, generate_docs, registry_rev, schema_for, validate_key, FacetSpec, TrackSchema};
pub use scope::{key_depth, AxisKind, Facet, Ledger, MinObsRule, Scope, ScopeKey, TemporalPolicy, Track, Window};
pub use stats::{decay_shape, DecayShape, DriftState, EwWin, MinMax, ScopeStat, StatSpan, Temporal, Welford};
pub use store::{
    write_atomic, AreaAxis, AreaReport, AxisStatus, Batch, ConfusionStatus, GateRate, GenIndex, GenStatus, GenerationEntry, LedgerStatus, Level as SdsLevel,
    OpenOptions, OtherRevs, ReferenceReport, ScopePolicy, SdsEstimate, SdsStatus, SdsStore, SDS_RECIPE, UNSTAMPED,
};
