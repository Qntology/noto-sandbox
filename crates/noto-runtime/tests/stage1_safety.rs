use std::collections::BTreeMap;

use noto_runtime::feedback::{FeedbackPath, Mode, SdsMode};
use noto_runtime::{Runtime, RuntimeConfig};
use noto_runtime::sds::{Scope, Track};
use serde_json::json;

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon")
}

fn new_rt(seed: u64) -> Runtime {
    Runtime::new(RuntimeConfig { seed, ..RuntimeConfig::default() }, &lexicon(), None).expect("runtime")
}

#[test]
fn off_mode_matches_a_cold_start() {
    let mut cold = new_rt(11);
    let mut hot_off = new_rt(11);
    // Pollute hot_off's SDS ledgers across all 8 feedback paths (A-H)
    {
        let sds = hot_off.sds_mut();
        for _ in 0..30 {
            sds.record_baseline(&Scope::new(Track::Vision, "tree", ""), "score", 0.92);
            sds.record_baseline(&Scope::new(Track::Vision, "tree", ""), "score_all", 0.92);
            sds.record_agreement(&Scope::new(Track::Vision, "tree", ""), "touch", false);
            sds.record_baseline(&Scope::new(Track::Forecast, "tree", ""), "fusion_mse:holt", 0.5);
            sds.record_baseline(&Scope::new(Track::Locate, "tree", "stand-in"), "score", 0.95);
            sds.record_calibration(&Scope::new(Track::Depth, "stand-in", "tree"), "scale", 2.5, 1.0);
            sds.record_calibration(&Scope::new(Track::Balance, "accel>foot", ""), "ff", 2.2, 1.0);
            sds.record_baseline(&Scope::new(Track::Memory, "wheel", "drift"), "drift_var_per_m2", 0.05);
            sds.record_baseline(&Scope::new(Track::Memory, "wheel", "obs"), "scatter_var_m2", 0.02);
        }
    }
    hot_off.set_sds_mode(SdsMode { default: Mode::Off, overrides: BTreeMap::new(), snapshot: None });

    cold.submit("나무 1개 캐와");
    hot_off.submit("나무 1개 캐와");
    cold.step(120);
    hot_off.step(120);

    let sc = cold.snapshot();
    let sh = hot_off.snapshot();
    assert_eq!(sc.tick, sh.tick);
    assert!((sc.agent.pos.x - sh.agent.pos.x).abs() < 1e-6 && (sc.agent.pos.z - sh.agent.pos.z).abs() < 1e-6, "agent pose diverged: {:?} vs {:?}", sc.agent, sh.agent);
    assert_eq!(cold.api_call("nn_status", &json!({})).unwrap()["scenarios"], hot_off.api_call("nn_status", &json!({})).unwrap()["scenarios"]);
}

#[test]
fn frozen_mode_ignores_writes_during_the_run() {
    let mut rt = new_rt(13);
    {
        let sds = rt.sds_mut();
        for _ in 0..12 {
            sds.record_baseline(&Scope::new(Track::Memory, "wheel", "drift"), "drift_var_per_m2", 0.0025);
        }
    }
    rt.set_sds_mode(SdsMode { default: Mode::Frozen, overrides: BTreeMap::new(), snapshot: None });
    let snap_id = rt.feedback().mode().snapshot.clone().expect("frozen snapshot digest");
    assert_eq!(snap_id.len(), 64);

    let status_before = rt.api_call("sds_feedback", &json!({})).unwrap();
    assert_eq!(status_before["mode"]["default"], "frozen");
    let before_rev = rt.sds().rev();

    rt.submit("나무 1개 캐와");
    rt.step(60);

    assert!(rt.sds().rev() > before_rev, "live SDS run buffer should still record observations");
    assert_eq!(rt.feedback().mode().snapshot.as_deref(), Some(snap_id.as_str()));
}

#[test]
fn forecast_scope_separates_ttm_fingerprints() {
    let mut sds = noto_runtime::sds::SdsStore::memory();
    let s1 = Scope::new(Track::Forecast, "tree", "ttm@aaaaaaaa");
    let s2 = Scope::new(Track::Forecast, "tree", "ttm@bbbbbbbb");
    for _ in 0..10 {
        sds.record_baseline(&s1, "fusion_mse:ttm@aaaaaaaa", 0.01);
        sds.record_baseline(&s2, "fusion_mse:ttm@bbbbbbbb", 0.25);
    }
    let fb = noto_runtime::feedback::FeedbackHub::live();
    let e1 = noto_runtime::precision::past_errors_with(&sds, "tree", Some("aaaaaaaa"), &fb, 1);
    let e2 = noto_runtime::precision::past_errors_with(&sds, "tree", Some("bbbbbbbb"), &fb, 1);
    let m1 = e1.iter().find(|h| h.method == "ttm").unwrap().mse;
    let m2 = e2.iter().find(|h| h.method == "ttm").unwrap().mse;
    assert!((m1 - 0.01).abs() < 1e-6 && (m2 - 0.25).abs() < 1e-6, "m1={m1} m2={m2}");
    let e3 = noto_runtime::precision::past_errors_with(&sds, "tree", Some("cccccccc"), &fb, 1);
    assert!(e3.iter().all(|h| h.method != "ttm"), "unseen fp8 must not inherit another fp8's mse: {e3:?}");
}

#[test]
fn every_written_axis_is_catalogued() {
    let mut rt = new_rt(19);
    rt.sds_mut().enable_contact_log();
    rt.submit("나무 1개 창고로");
    rt.step(180);
    let log = rt.sds().contact_snapshot().expect("contact log");
    assert!(!log.writes.is_empty(), "expected runtime SDS writes");
    let uncat = log.uncatalogued_writes();
    assert!(uncat.is_empty(), "uncatalogued SDS writes found: {uncat:?}");
}

#[test]
fn sds_feedback_api_switches_modes_and_logs_audit() {
    let mut rt = new_rt(23);
    let st = rt.api_call("sds_feedback", &json!({"default": "off", "overrides": {"A": "live"}})).unwrap();
    assert_eq!(st["mode"]["default"], "off");
    assert_eq!(st["mode"]["overrides"]["A"], "live");
    assert_eq!(rt.feedback().effective_mode(FeedbackPath::A), Mode::Live);
    assert_eq!(rt.feedback().effective_mode(FeedbackPath::F), Mode::Off);
    let (lines, _) = rt.log_tail(0, 50, Some("fb"));
    assert!(!lines.is_empty(), "expected fb audit log entry");
}
