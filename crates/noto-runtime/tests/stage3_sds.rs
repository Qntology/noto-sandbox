use std::path::PathBuf;

use noto_core::{BaseKind, BodyPlan};
use noto_core::parts::Tier;
use noto_runtime::config::{GateCfg, RuntimeConfig};
use noto_runtime::feedback::{FeedbackHub, FeedbackPath, Mode, SdsMode};
use noto_runtime::nnlive::{NnLive, NnView};
use noto_runtime::precision::PrecisionGates;
use noto_runtime::Runtime;
use noto_sds::{DriftState, Facet, Scope, ScopeKey, SdsStore, Track};
use serde_json::json;

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file")
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let d = std::env::temp_dir().join(format!("noto-stage3-{tag}-{}-{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn decision_reads_unchanged_after_migration() {
    let mut sds = SdsStore::memory();
    let v_scope = Scope::new(Track::Vision, "tree", "");
    let l_scope = Scope::new(Track::Locate, "tree", "stand-in");
    let d_scope = Scope::new(Track::Depth, "stand-in", "tree");
    let b_scope = Scope::new(Track::Balance, "accel>foot", "flat|move");
    let g_scope = Scope::new(Track::Memory, "wheel", "drift");
    let h_scope = Scope::new(Track::Memory, "wheel", "obs");
    let f_scope = Scope::new(Track::Forecast, "tree", "");

    for i in 0..30 {
        let off = ((i % 3) as f64 - 1.0) * 0.01;
        sds.record_baseline(&v_scope, "score", 0.78 + off);
        sds.record_baseline(&v_scope, "score_all", 0.75 + off);
        sds.record_agreement(&v_scope, "touch", i % 5 != 0);
        sds.record_baseline(&l_scope, "score", 0.72 + off);
        sds.record_calibration(&d_scope, "scale", 4.4 + off, 4.0);
        sds.record_calibration(&b_scope, "ff", 0.11 + off * 0.1, 0.10);
        sds.record_baseline(&g_scope, "drift_var_per_m2", 0.002 + off * 0.0001);
        sds.record_baseline(&h_scope, "scatter_var_m2", 0.001 + off * 0.0001);
        sds.record_baseline(&f_scope, "fusion_mse:ewma", 0.01 + off * 0.001);
    }

    let fb = FeedbackHub::live();
    let before_a = fb.baseline(FeedbackPath::A, &sds, &v_scope, "score", 1);
    let before_b = fb.baseline(FeedbackPath::B, &sds, &v_scope, "score_all", 1);
    let before_c = fb.fusion_mse(FeedbackPath::C, &sds, &f_scope, "ewma", 1);
    let before_d = fb.baseline(FeedbackPath::D, &sds, &l_scope, "score", 1);
    let before_e = fb.calibration(FeedbackPath::E, &sds, &d_scope, "scale", 1).map(|x| x.0);
    let before_f = fb.calibration(FeedbackPath::F, &sds, &b_scope, "ff", 1).map(|x| x.0);
    let before_g = fb.baseline(FeedbackPath::G, &sds, &g_scope, "drift_var_per_m2", 1);
    let before_h = fb.baseline(FeedbackPath::H, &sds, &h_scope, "scatter_var_m2", 1);

    // Save to disk and reopen (triggers migration / load path while preserving decision keys).
    let dir = temp_dir("mig");
    let mut disk = SdsStore::open(&dir);
    for i in 0..30 {
        let off = ((i % 3) as f64 - 1.0) * 0.01;
        disk.record_baseline(&v_scope, "score", 0.78 + off);
        disk.record_baseline(&v_scope, "score_all", 0.75 + off);
        disk.record_agreement(&v_scope, "touch", i % 5 != 0);
        disk.record_baseline(&l_scope, "score", 0.72 + off);
        disk.record_calibration(&d_scope, "scale", 4.4 + off, 4.0);
        disk.record_calibration(&b_scope, "ff", 0.11 + off * 0.1, 0.10);
        disk.record_baseline(&g_scope, "drift_var_per_m2", 0.002 + off * 0.0001);
        disk.record_baseline(&h_scope, "scatter_var_m2", 0.001 + off * 0.0001);
        disk.record_baseline(&f_scope, "fusion_mse:ewma", 0.01 + off * 0.001);
    }
    disk.flush();
    drop(disk);

    let reopened = SdsStore::open(&dir);
    let after_a = fb.baseline(FeedbackPath::A, &reopened, &v_scope, "score", 2);
    let after_b = fb.baseline(FeedbackPath::B, &reopened, &v_scope, "score_all", 2);
    let after_c = fb.fusion_mse(FeedbackPath::C, &reopened, &f_scope, "ewma", 2);
    let after_d = fb.baseline(FeedbackPath::D, &reopened, &l_scope, "score", 2);
    let after_e = fb.calibration(FeedbackPath::E, &reopened, &d_scope, "scale", 2).map(|x| x.0);
    let after_f = fb.calibration(FeedbackPath::F, &reopened, &b_scope, "ff", 2).map(|x| x.0);
    let after_g = fb.baseline(FeedbackPath::G, &reopened, &g_scope, "drift_var_per_m2", 2);
    let after_h = fb.baseline(FeedbackPath::H, &reopened, &h_scope, "scatter_var_m2", 2);

    assert_eq!(before_a.map(|x| (x.0 * 1e6).round()), after_a.map(|x| (x.0 * 1e6).round()));
    assert_eq!(before_b.map(|x| (x.0 * 1e6).round()), after_b.map(|x| (x.0 * 1e6).round()));
    assert_eq!(before_c.map(|x| (x.0 * 1e6).round()), after_c.map(|x| (x.0 * 1e6).round()));
    assert_eq!(before_d.map(|x| (x.0 * 1e6).round()), after_d.map(|x| (x.0 * 1e6).round()));
    assert_eq!(before_e.map(|x| (x * 1e6).round()), after_e.map(|x| (x * 1e6).round()));
    assert_eq!(before_f.map(|x| (x * 1e6).round()), after_f.map(|x| (x * 1e6).round()));
    assert_eq!(before_g.map(|x| (x.0 * 1e6).round()), after_g.map(|x| (x.0 * 1e6).round()));
    assert_eq!(before_h.map(|x| (x.0 * 1e6).round()), after_h.map(|x| (x.0 * 1e6).round()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn nn_view_matches_old_nnlive() {
    let mut sds = SdsStore::memory();
    let s4_carry = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "carry".into())]);
    let s4_b05 = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "brake_0.5".into())]);
    let s4_b25 = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "brake_2.5".into())]);
    sds.record_v2_baseline(&s4_carry, "score", 90.0);
    sds.record_v2_baseline(&s4_b05, "score", 80.0);

    let live = NnLive::default();
    let v1 = NnView::from_store(&sds, "all", &live);
    let sc1 = v1["scenarios"][0]["score"].as_f64().expect("S4 score");
    assert!((sc1 - 86.3).abs() < 0.051, "expected 86.3, got {sc1}");

    sds.record_v2_baseline(&s4_b25, "score", 40.0);
    let v2 = NnView::from_store(&sds, "all", &live);
    let sc2 = v2["scenarios"][0]["score"].as_f64().expect("S4 score with worse rung");
    assert!((sc2 - 71.3).abs() < 0.051, "expected 71.3, got {sc2}");
}

#[test]
fn phase_obs_joins_score_margin_load_and_survives_restart() {
    let dir = temp_dir("nn-restart");
    let cfg = RuntimeConfig { seed: 7, parcels: Some(0), ..RuntimeConfig::default() };
    let score_before: f64;
    {
        let mut rt = Runtime::new(cfg.clone(), &lexicon(), Some(&dir)).unwrap();
        // Give the robot cargo and drive + brake to trigger S4 (carry/brake) balance events with dynamics.
        let sub = rt.submit("통나무 1개 창고로");
        assert!(sub.ok, "{:?}", sub.error);
        rt.run_until_idle(1400);
        rt.sds_mut().flush();

        // Check that at least one v2 N:N phase key joins score + min_margin + peak_ratio + gates + extrema.
        let keys = rt.sds().scope_keys_for_track(Track::Balance);
        let nn_v2_keys: Vec<&String> = keys.iter().filter(|k| k.starts_with("balance/nn:") && k.contains(";ph=")).collect();
        assert!(!nn_v2_keys.is_empty(), "expected v2 N:N phase keys in SDS: {keys:?}");
        let joined = nn_v2_keys.iter().any(|k| {
            rt.sds().merged_scope_by_key(Track::Balance.ledger(), k).is_some_and(|st| {
                st.baseline.contains_key("score")
                    && st.baseline.contains_key("min_margin")
                    && st.baseline.contains_key("peak_ratio")
                    && st.agreement.contains_key("item_ok")
                    && st.extrema.contains_key("min_margin")
                    && !st.gates.is_empty()
            })
        });
        assert!(joined, "a v2 N:N phase scope must join score, min_margin, peak_ratio, item_ok, extrema, and gates");

        // Verify NnPhaseRow was recorded in TestDb mirror batch.
        let batch = rt.test_db_mut().take_mirror();
        assert!(!batch.nn_phases.is_empty(), "NnPhaseRow must be recorded for matched phases");
        assert!(!batch.nn_phases[0].event_id.is_empty());
        assert!(!batch.nn_phases[0].generation.is_empty());

        let all_before = rt.api_call("nn_status", &json!({"window": "all"})).unwrap();
        let scns = all_before["scenarios"].as_array().unwrap();
        assert!(!scns.is_empty(), "expected scenarios in nn_status(window=all)");
        score_before = scns[0]["score"].as_f64().unwrap();
    }

    // Reopen a fresh Runtime from the same directory: window=all must survive, window=run must be empty!
    let mut rt2 = Runtime::new(cfg, &lexicon(), Some(&dir)).unwrap();
    let all_after = rt2.api_call("nn_status", &json!({"window": "all"})).unwrap();
    let run_after = rt2.api_call("nn_status", &json!({"window": "run"})).unwrap();
    let scns_all = all_after["scenarios"].as_array().unwrap();
    let scns_run = run_after["scenarios"].as_array().unwrap();
    assert!(!scns_all.is_empty(), "window=all must survive disk restart");
    assert!((scns_all[0]["score"].as_f64().unwrap() - score_before).abs() < 1e-6);
    assert!(scns_run.is_empty(), "window=run must be empty right after restart, got {scns_run:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn floor_is_unchanged_on_a_steady_label_and_drift_falls_back_or_restarts() {
    let g = PrecisionGates::new(GateCfg::default());
    let fb = FeedbackHub::live();
    let mut sds = SdsStore::memory();
    let scope = Scope::new(Track::Vision, "tree", "");

    // Steady high-score observations around 0.80 ± 0.015 -> raises floor above fixed 0.30 up to cap 0.55.
    for i in 0..45 {
        let v = 0.80 + 0.015 * ((i % 3) as f64 - 1.0);
        sds.record_baseline(&scope, "score", v);
    }
    let (steady_floor, src) = g.floor_for_with("tree", &sds, &fb, 10);
    assert!((steady_floor - 0.55).abs() < 1e-4, "steady floor expected 0.55, got {steady_floor} ({src})");
    assert_eq!(fb.drift_state(FeedbackPath::A, &sds, &scope, "score"), DriftState::Stable);

    // Sudden step drop to 0.45 (>> 1σ drop) -> CUSUM shifts; immediately after shift (seg n < 4) must drop back
    // to cold fixed floor 0.30 (never clinging to stale 0.55!), then once segment accumulates >= 4 points, tracks ~0.40-0.44!
    let mut saw_shifted = false;
    let mut saw_fallback = false;
    for i in 0..12 {
        sds.record_baseline(&scope, "score", 0.45 + 0.005 * ((i % 2) as f64 - 0.5));
        if matches!(fb.drift_state(FeedbackPath::A, &sds, &scope, "score"), DriftState::Shifted { .. }) {
            saw_shifted = true;
            let (fl, fl_src) = g.floor_for_with("tree", &sds, &fb, 50 + i);
            if fl <= 0.301 {
                saw_fallback = true;
            }
            assert!(fl < 0.50, "post-shift floor must not stay at stale 0.55: got {fl} ({fl_src})");
        }
    }
    assert!(saw_shifted, "step drop must trigger CUSUM Shifted state");
    assert!(saw_fallback, "thin segment immediately after shift must fall back to cold fixed floor");
    let (post_floor, _) = g.floor_for_with("tree", &sds, &fb, 99);
    assert!(post_floor > 0.30 && post_floor < 0.48, "settled segment floor should reflect new 0.45 distribution: {post_floor}");
}

#[test]
#[should_panic(expected = "provenance guard")]
fn decisions_read_drift_only_on_decision_axes_blocks_reference() {
    let sds = SdsStore::memory();
    let fb = FeedbackHub::live();
    let ref_scope = Scope::new(Track::Balance, "nn:S4", "brake_1.5");
    let _ = fb.drift_state(FeedbackPath::F, &sds, &ref_scope, "score");
}

#[test]
#[should_panic(expected = "provenance guard")]
fn decisions_read_drift_only_on_decision_axes_blocks_truth() {
    let sds = SdsStore::memory();
    let fb = FeedbackHub::live();
    let truth_scope = Scope::new(Track::Depth, "stand-in", "tree");
    let _ = fb.drift_state(FeedbackPath::E, &sds, &truth_scope, "absrel");
}

#[test]
fn every_control_read_goes_through_the_registry() {
    // Allowlist of non-decision display/test files that read SDS directly (Phase 2 – Part 3 · D5).
    let allowlist = [
        "feedback.rs",
        "actsds.rs",
        "mldrive.rs",
        "mlsds.rs",
        "testrun.rs",
        "nnlive.rs",
        "benchsds.rs",
    ];
    let read_tokens = [
        "adaptive_baseline(",
        "adaptive_baseline_scoped(",
        "agreement_rate(",
        "agreement_rate_scoped(",
        "calibration_factor(",
        "estimate_mean(",
        "estimate_rate(",
        "estimate_spread(",
        "estimate_factor(",
        "fusion_mse(",
    ];
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(&src_dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".rs") || allowlist.contains(&name.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let code = text.split("#[cfg(test)]").next().unwrap_or(&text);
        for tok in read_tokens {
            // In precision.rs / vision.rs / runtime.rs, calls go through `fb.<method>` (FeedbackHub), NEVER `sds.<method>`.
            let direct = format!("sds.{tok}");
            assert!(
                !code.contains(&direct),
                "direct SDS read `{direct}` found outside feedback.rs in {name}"
            );
        }
    }
}

#[test]
fn reference_perturbation_leaves_control_untouched() {
    let cfg = RuntimeConfig { seed: 11, parcels: Some(0), ..RuntimeConfig::default() };
    let mut rt_a = Runtime::new(cfg.clone(), &lexicon(), None).unwrap();
    let mut rt_b = Runtime::new(cfg, &lexicon(), None).unwrap();

    // Seed identical Experience observations in both runtimes.
    let v_scope = Scope::new(Track::Vision, "tree", "");
    let b_scope = Scope::new(Track::Balance, "accel>foot", "");
    for i in 0..25 {
        for rt in [&mut rt_a, &mut rt_b] {
            rt.sds_mut().record_baseline(&v_scope, "score", 0.72 + 0.01 * ((i % 3) as f64 - 1.0));
            rt.sds_mut().record_calibration(&b_scope, "ff", 0.11, 0.10);
        }
    }

    // Perturb Group B reference axes (nn:* and :dev/:in_band) wildly ONLY in rt_b!
    let nn_scope = Scope::new(Track::Balance, "nn:S4", "brake_1.5");
    let couple_scope = Scope::new(Track::Balance, "arm_lift>foot", "flat|pick");
    for _ in 0..40 {
        rt_b.sds_mut().record_baseline(&nn_scope, "score", 12.5);
        rt_b.sds_mut().record_reference(&couple_scope, "com_shift_frac", 9.9, 0.2, 0.05);
    }

    rt_a.submit("통나무 1개 창고로");
    rt_b.submit("통나무 1개 창고로");
    for _ in 0..250 {
        rt_a.tick();
        rt_b.tick();
    }

    assert_eq!(rt_a.traj_digests(), rt_b.traj_digests(), "reference perturbation must never alter control trajectory");
    assert_eq!(rt_a.feedback().snapshot_values(), rt_b.feedback().snapshot_values(), "A-H values must be identical");
}

#[test]
fn holdout_writes_never_feed_control() {
    let cfg = RuntimeConfig { seed: 7, parcels: Some(0), ..RuntimeConfig::default() };
    let mut rt = Runtime::new(cfg, &lexicon(), None).unwrap();

    // Seed control SDS with some baseline observations.
    let v_scope = Scope::new(Track::Vision, "tree", "");
    for _ in 0..15 {
        rt.sds_mut().record_baseline(&v_scope, "score", 0.75);
    }
    let obs_before = rt.sds().total_observations();
    let eval_before = rt.eval_sds().total_observations();
    assert_eq!(eval_before, 0);

    // Switch to holdout pass and run simulation + scan + quest actions.
    rt.set_pass("holdout");
    let sub = rt.submit("통나무 1개 창고로");
    assert!(sub.ok);
    for _ in 0..180 {
        rt.tick();
    }
    let _ = rt.scan_now("holdout_check");

    assert_eq!(
        rt.sds().total_observations(),
        obs_before,
        "holdout pass must NEVER write to control sds"
    );
    assert!(
        rt.eval_sds().total_observations() > 0,
        "holdout pass must write its observations to isolated eval_sds"
    );
}

#[test]
fn same_seed_same_snapshot_reproduces_nn() {
    let dir = temp_dir("replay");
    let cfg = RuntimeConfig { seed: 7, parcels: Some(0), ..RuntimeConfig::default() };
    let mut rt = Runtime::new(cfg, &lexicon(), Some(&dir)).unwrap();
    rt.test_cfg.record.max_video_frames = 0;
    rt.test_cfg.record.max_keyframes = 6;
    rt.set_sds_mode(SdsMode { default: Mode::Frozen, overrides: Default::default(), snapshot: None });

    let plan = BodyPlan::uniform(Tier::Mid);
    rt.test_start("통나무 1개 창고로", 7, plan, Some(BaseKind::Wheel)).unwrap();
    let _ = rt.test_advance(500).unwrap();
    let st = rt.api_test_status().unwrap();
    let manifest_val = st["manifest"].clone();
    assert!(manifest_val.is_object(), "manifest must be recorded on test_start");
    let orig_digests = manifest_val["traj_digests"].as_array().unwrap();
    assert_eq!(orig_digests.len(), 10, "expected 10 fifty-tick trajectory digests over 500 ticks");

    // Replay using the `replay` API command and assert bit-identical trajectory digests.
    let rep = rt.api_call("replay", &json!({"manifest": manifest_val, "max_ticks": 500})).unwrap();
    assert_eq!(rep["matched"], json!(true), "replay mismatch: {rep}");
    assert!(rep["first_mismatch_tick"].is_null());
    assert_eq!(rep["original_digests"], rep["replay_digests"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn audit_verify_and_generations_api_work() {
    let dir = temp_dir("audit-gen");
    let cfg = RuntimeConfig { seed: 7, parcels: Some(0), ..RuntimeConfig::default() };
    let mut rt = Runtime::new(cfg, &lexicon(), Some(&dir)).unwrap();
    rt.sds_mut().record_baseline(&Scope::new(Track::Vision, "tree", ""), "score", 0.8);
    rt.sds_mut().flush();
    // Trigger runtime flush_sds via reset_world or ticks so sds_seal + anchor are written.
    let _ = rt.reset_world();

    let verify = rt.api_call("audit_verify", &json!({})).unwrap();
    let files = verify["files"].as_array().unwrap();
    assert!(!files.is_empty(), "audit_verify must list session file");
    assert!(files[0]["broken_at"].is_null(), "audit chain broken: {files:?}");
    assert_eq!(files[0]["anchor_ok"], json!(true));
    assert_eq!(files[0]["seal_ok"], json!(true));

    let gens = rt.api_call("sds_generations", &json!({})).unwrap();
    assert!(gens["current"].as_str().unwrap().contains(noto_runtime::ENGINE_FP));

    let arch = rt.api_call("sds_archive", &json!({"action": "purge"})).unwrap();
    assert_eq!(arch["action"], json!("purge"));
    let list = rt.api_call("sds_archive", &json!({"action": "list"})).unwrap();
    assert!(!list["entries"].as_array().unwrap().is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}
