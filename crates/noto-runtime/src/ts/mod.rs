pub use lv_sds::{io, ttm};
pub use lv_sds::ttm::{
    ttm_device_pref, ttm_parity_check, PadMode, TtmCallStats, TtmFingerprint, TtmInputPolicy,
};
pub use lv_spatial::pos2;
pub use lv_spatial::pos2::{
    Pos2Estimate, Pos2Filter, Pos2Model, Pos2Obs, CHI2_2DOF_99, POS2_RING_CAP,
};
pub use premium_discount_sds::temporal as holt;
pub use premium_discount_sds::temporal::{holt_damped, HoltFit};

pub mod adapter;
pub mod calib;
pub mod exog;
pub mod fusion;
pub mod gate;
pub mod physics;
pub mod registry;
pub mod series;
pub mod ttm_predictor;

pub use adapter::{
    adapter_accepts_group, evaluate_adoption, AdapterEval, AffineAdapter, ResidualAdapter,
};
pub use calib::{interval_score_80, norm_cdf, Aci, Cusum, ErrBook, HorizonScore, Pending, PendingRing, HORIZONS};
pub use exog::{fit_ridge, fit_ridge_cv, ExogFit, ExogResidual, MAX_EXOG_K, RIDGE_GRID};
pub use fusion::{
    forecast, forecast_reg, forecast_with_prior, time_to_cross, Band, EwmaPredictor, Forecast,
    HistoryMse, HoltPredictor, LastPredictor, MethodWeight, Predictor, TtmEligibility, HISTORY_CAP,
    HISTORY_ONLY_MIN_N, HOLDOUT, MIN_POINTS,
};
pub use gate::{
    temporal_gate, temporal_gate_with_forecast, temporal_verdict, GateBand, GateConfig,
    TemporalOutcome, TemporalVerdict,
};
pub use physics::{
    heat_crossing_s, integrate_path, integrate_state, shift_change_s, wear_rul_s, FirstPassage,
    PhysDriven, PhysPassage, PhysicsPredictor, State,
};
pub use registry::{
    eligible, Cadence, Caps, Channels, Eligibility, Exog, Group, PredictorRegistry,
    PredictorSpec, SeriesCtx, SeriesSpec,
};
pub use series::{
    align_span, dedup_samples, regularize, ChannelKind, ChannelSpec, RegSeries, SpanRef, SpanView,
    TsSample,
};
pub use ttm_predictor::{CalibratedTtmPredictor, TtmPredictor};
