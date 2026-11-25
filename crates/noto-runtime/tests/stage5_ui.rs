//! Stage 5 verification suite: Phase 3-P3 (forecast uses), Phase 5-P1 (progress/N:N/holds),
//! and Phase 5-P2 (SDS explorer, ts_view until_tick, test_status_delta, audit legacy compatibility).

use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig};
use noto_runtime::config::RuntimeConfig;
use noto_runtime::perception::{Candidate, EntKind};
use noto_runtime::Runtime;
use serde_json::json;

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json"))
        .expect("lexicon")
}

fn make_rt(seed: u64, base: BaseKind) -> Runtime {
    let mut cfg = RuntimeConfig::default();
    cfg.seed = seed;
    cfg.noto = NotoConfig { base, fingers: FingerCount::Five, ..cfg.noto };
    cfg.body = BodyPlan::named("mid").unwrap_or_default();
    Runtime::new(cfg, &lexicon(), None).expect("runtime")
}

#[test]
fn forecast_off_is_bit_identical() {
    let mut a = make_rt(11, BaseKind::HumanFoot);
    let mut b = make_rt(11, BaseKind::HumanFoot);
    assert!(!a.world().forecast_control && !b.world().forecast_control);
    let _ = a.api_command_submit("원목 1개 가져와");
    let _ = b.api_command_submit("원목 1개 가져와");
    for _ in 0..160 {
        a.tick();
        b.tick();
    }
    assert_eq!(a.traj_digests(), b.traj_digests(), "Cold trust runs must be bit-identical");
    assert_eq!(a.world().agent.x.to_bits(), b.world().agent.x.to_bits());
    assert_eq!(a.world().agent.z.to_bits(), b.world().agent.z.to_bits());
}

#[test]
fn the_current_phase_clears_when_the_body_goes_quiet() {
    let mut rt = make_rt(7, BaseKind::HumanFoot);
    // Drive the robot briefly so a balance/N:N phase activates.
    let _ = rt.api_command_submit("창고로 가");
    for _ in 0..35 {
        rt.tick();
    }
    // Cancel and let the body settle quietly for > 15 ticks.
    rt.cancel();
    for _ in 0..40 {
        rt.tick();
    }
    let step = rt.api_sim_step_focus(1, 1).expect("sim_step");
    let now = step["progress"]["nn"]["now"].as_array().expect("nn.now array");
    assert!(now.is_empty(), "when body is quiet, progress.nn.now must clear: {now:?}");
}

#[test]
fn two_scenarios_can_be_current_at_once() {
    let mut rt = make_rt(7, BaseKind::HumanFoot);
    // Load cargo onto a legged body and walk: triggers S2 (carry_walk) AND S4 (carry) / S3 simultaneously!
    rt.world_mut().agent.carried.push((world_sim::ItemKind::Log, 2.0));
    let _ = rt.api_command_submit("창고로 가");
    let mut max_concurrent = 0usize;
    let mut seen_pair: Vec<String> = Vec::new();
    for _ in 0..120 {
        let step = rt.api_sim_step_focus(2, 1).expect("sim_step");
        if let Some(arr) = step["progress"]["nn"]["now"].as_array() {
            let scns: std::collections::BTreeSet<&str> = arr
                .iter()
                .filter_map(|x| x["scenario"].as_str())
                .collect();
            if scns.len() > max_concurrent {
                max_concurrent = scns.len();
                seen_pair = scns.into_iter().map(String::from).collect();
            }
        }
        if max_concurrent >= 2 {
            break;
        }
    }
    assert!(
        max_concurrent >= 2,
        "legged carry motion must activate >= 2 concurrent scenarios (got {max_concurrent}: {seen_pair:?})"
    );
}

#[test]
fn a_sub_worker_reports_not_computed() {
    let mut rt = make_rt(3, BaseKind::HumanFoot);
    rt.crew_add(1, "humanoid", None).expect("add crew");
    let w2_id = rt.world().crew[0].id;
    let step = rt.api_sim_step_focus(1, w2_id).expect("sim_step focus crew");
    assert_eq!(step["progress"]["focus"].as_u64(), Some(w2_id as u64));
    assert_eq!(step["progress"]["nn"]["status"].as_str(), Some("not_computed"));
}

#[test]
fn phase_states_cover_skipped_and_unrecorded() {
    // Wheeled body skips legged-only scenarios/phases (e.g. S3 weight_shift / foot_lift),
    // and offline/unrecorded scenarios appear as unrecorded or not_observed.
    let mut rt = make_rt(5, BaseKind::Wheel);
    rt.tick();
    let st = rt.api_call("nn_status", &json!({})).expect("nn_status");
    let states = st["scenario_states"].as_array().expect("scenario_states");
    let mut saw_skipped = false;
    let mut saw_unrecorded_or_not_obs = false;
    for sc in states {
        let s = sc["state"].as_str().unwrap_or("");
        if s == "skipped_form" {
            saw_skipped = true;
        }
        if s == "no_record" || s == "unseen" {
            saw_unrecorded_or_not_obs = true;
        }
        if let Some(phs) = sc["phases"].as_array() {
            for p in phs {
                let ps = p["state"].as_str().unwrap_or("");
                if ps == "skipped_form" {
                    saw_skipped = true;
                }
                if ps == "no_record" || ps == "unseen" {
                    saw_unrecorded_or_not_obs = true;
                }
            }
        }
    }
    assert!(saw_skipped, "wheeled body must mark legged-only phases as skipped_form: {st}");
    assert!(saw_unrecorded_or_not_obs, "catalog must report no_record/unseen scenarios: {st}");
}

#[test]
fn progress_payload_stays_small() {
    let mut rt = make_rt(9, BaseKind::HumanFoot);
    rt.crew_add(7, "mixed", None).expect("add 7 crew (total 8 workers)");
    let step = rt.api_sim_step_focus(2, 1).expect("sim_step");
    let bytes = serde_json::to_vec(&step["progress"]).unwrap().len();
    assert!(
        bytes < 8192,
        "progress payload with 8 workers must remain compact (<8 KiB), got {bytes} bytes"
    );
    assert_eq!(step["progress"]["workers"].as_array().unwrap().len(), 8);
}

#[test]
fn ts_view_until_tick_uses_only_past_points() {
    let mut rt = make_rt(4, BaseKind::HumanFoot);
    // Feed 12 landmark sightings at ticks 10..=120 with a level change near the end.
    let make_c = |score: f32| Candidate {
        label: "tree".into(),
        kind: EntKind::Source,
        link: "log".into(),
        score,
        margin: 0.25,
        runner_up: None,
        bbox: [0.4, 0.4, 0.6, 0.6],
        pixels: 120,
        depth_m: 3.0,
        world: Some((5.0, 5.0)),
        truth: None,
        truth_label: None,
    };
    let mut lm_id = 0u32;
    let mut g = noto_runtime::PrecisionGates::new(rt.cfg.gates.clone());
    let mut sds = noto_runtime::sds::SdsStore::memory();
    for i in 1..=12u64 {
        let sc = if i <= 8 { 0.80 + 0.01 * ((i % 3) as f32 - 1.0) } else { 0.50 };
        let j = g.judge(vec![make_c(sc)], i * 10, &mut sds).pop().unwrap();
        lm_id = j.landmark.unwrap();
    }
    // Use the public Runtime::scan / gates via a direct scan or inspect via PrecisionGates on Runtime:
    // Let's also feed via Runtime::api_perception_scan / world ticks so Runtime::gates() has landmarks.
    let _ = rt.api_command_submit("원목 1개 가져와");
    for _ in 0..80 {
        rt.tick();
        if rt.world().tick % 10 == 0 {
            let _ = rt.scan_now("test");
        }
    }
    let cat = rt.api_call("ts_series", &json!({})).expect("ts_series catalog");
    let lm_series = cat["series"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["id"].as_str().is_some_and(|s| s.starts_with("lm:")) && x["n"].as_u64().unwrap_or(0) >= 3);
    if let Some(item) = lm_series {
        let sid = item["id"].as_str().unwrap();
        let full = rt.api_call("ts_view", &json!({"series": sid, "horizon": 4})).expect("ts_view full");
        let pts = full["points"].as_array().unwrap();
        assert!(pts.len() >= 3);
        let cut_tick = pts[pts.len() / 2]["tick"].as_u64().unwrap();
        let past = rt
            .api_call("ts_view", &json!({"series": sid, "horizon": 4, "until_tick": cut_tick}))
            .expect("ts_view past");
        let past_pts = past["points"].as_array().unwrap();
        assert!(past_pts.len() < pts.len(), "until_tick must exclude later points");
        assert!(
            past_pts.iter().all(|p| p["tick"].as_u64().unwrap() <= cut_tick),
            "no point after until_tick={cut_tick} may appear"
        );
    } else {
        // Fallback: verify on `ts:` continuous channel which always has >= 10 samples after 80 ticks.
        let sid = "ts:w1:bal.margin";
        let full = rt.api_call("ts_view", &json!({"series": sid, "horizon": 4})).expect("ts_view full ts");
        let pts = full["points"].as_array().unwrap();
        assert!(pts.len() >= 5, "expected balance margin samples: {full}");
        let cut_tick = pts[pts.len() / 2]["tick"].as_u64().unwrap();
        let past = rt
            .api_call("ts_view", &json!({"series": sid, "horizon": 4, "until_tick": cut_tick}))
            .expect("ts_view past ts");
        let past_pts = past["points"].as_array().unwrap();
        assert!(past_pts.len() < pts.len());
        assert!(past_pts.iter().all(|p| p["tick"].as_u64().unwrap() <= cut_tick));
    }
    assert!(lm_id > 0);
}

#[test]
fn test_status_delta_appends_and_updates() {
    let mut rt = make_rt(2, BaseKind::HumanFoot);
    rt.test_start("창고로 가", 2, rt.cfg.body, Some(BaseKind::HumanFoot)).expect("test_start");
    let st0 = rt.api_test_status().expect("initial status");
    let id = st0["id"].as_str().unwrap().to_string();
    let rev0 = st0["test_rev"].as_u64().unwrap();

    // Advance 30 ticks => 3 new game keyframes (every 10 ticks).
    let _ = rt.test_advance(30).expect("advance game");
    let d1 = rt
        .api_call("test_status_delta", &json!({"id": id, "since_rev": rev0}))
        .expect("delta 1");
    let app1 = d1["game"]["appended"].as_array().unwrap();
    let upd1 = d1["game"]["updated"].as_array().unwrap();
    assert!(app1.len() >= 3, "expected >=3 newly appended keyframes, got {}", app1.len());
    assert!(upd1.is_empty(), "game capture only appends: {upd1:?}");
    let rev1 = d1["test_rev"].as_u64().unwrap();

    // Calling again with if_rev == rev1 must return unchanged.
    let unch = rt
        .api_call("test_status_delta", &json!({"id": id, "since_rev": rev1, "if_rev": rev1}))
        .expect("unchanged");
    assert_eq!(unch["unchanged"], true);

    // Finish Game pass and step once into Perceive -> existing keyframes get updated in place!
    for _ in 0..40 {
        let ph = rt.test_advance(30).expect("advance to perceive");
        if ph == noto_runtime::TestPhase::Perceive {
            break;
        }
    }
    let st_pre = rt.api_test_status().expect("pre perceive");
    let rev_pre = st_pre["test_rev"].as_u64().unwrap();
    if st_pre["phase"] == "perceive" {
        let _ = rt.test_advance(1).expect("perceive step");
        let d2 = rt
            .api_call("test_status_delta", &json!({"id": id, "since_rev": rev_pre}))
            .expect("delta perceive");
        let upd2 = d2["game"]["updated"].as_array().unwrap();
        let app2 = d2["game"]["appended"].as_array().unwrap();
        assert!(app2.is_empty(), "perceive must not append new game keyframes");
        assert!(!upd2.is_empty(), "perceive must report updated keyframe indices: {d2}");
    }
}

#[test]
fn audit_old_lines_still_verify() {
    // Mixed log containing lines without `tick` (omitted in JSON) and lines with `tick` must verify cleanly.
    let dir = std::env::temp_dir().join(format!("noto_rt_audit_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut log = noto_runtime::sds::AuditLog::open(&dir, "mixed", 32).expect("open audit");
    log.info("world", "legacy without tick");
    log.log_with_tick("quest", noto_runtime::sds::Level::Info, "with tick", None, Some(42));
    let path = log.path().unwrap().to_path_buf();
    let txt = std::fs::read_to_string(&path).unwrap();
    assert!(!txt.lines().next().unwrap().contains("\"tick\""), "None tick must be omitted from JSON");
    assert!(txt.lines().nth(1).unwrap().contains("\"tick\":42"), "Some tick must be serialized");
    let ok = noto_runtime::sds::verify_file(&path).expect("mixed audit file must verify");
    assert_eq!(ok.lines, 2);
    drop(log);
    let _ = std::fs::remove_dir_all(&dir);
}
