//! Stage 6 integration & cross-crate verification tests:
//! - Phase 6-P1 D2–D6 (offline_only_phases_never_match_live, a_pick_on_a_slope_scores_s1_lift_live)
//! - Phase 6-P2 D2–D6 (sequence_heat_equals_world_heat, sub_worker_static_cells_equal_first_worker, a_new_def_shows_up_everywhere)
//! - Phase 2-P2 D5 (phase-residual CUSUM axis, I² mixed flag, cumulative wear rate per hour)
//! - Phase 4-P2 D2/D4/D6 (heat_first_passage_matches_closed_form, t1_t2_adapter_respects_engine_rev_and_holdout)

use noto_core::balance::{Body, FeedForward};
use noto_core::bench::ENGINE_REV;
use noto_core::effects;
use noto_core::nn;
use noto_core::parts::{BodyPlan, Tier};
use noto_core::surface::Surface;
use noto_core::{BaseKind, FingerCount, NotoConfig};
use noto_runtime::config::RuntimeConfig;
use noto_runtime::Runtime;
use noto_runtime::sds::{Scope, Track};
use noto_runtime::ts::adapter::{evaluate_adoption, AffineAdapter, ResidualAdapter};
use noto_runtime::ts::physics::{heat_crossing_s, PhysDriven, State};
use noto_runtime::ts::registry::Group;
use serde_json::json;
use world_sim::crew::{Form as CrewForm, Job};
use world_sim::terrain::TerrainKind;
use world_sim::world::{Action, DropItem, World, WorldConfig};
use world_sim::ItemKind;

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json"))
        .expect("lexicon")
}

fn make_rt(seed: u64, base: BaseKind, terrain: TerrainKind) -> Runtime {
    let mut cfg = RuntimeConfig::default();
    cfg.seed = seed;
    cfg.noto = NotoConfig { base, fingers: FingerCount::Five, ..cfg.noto };
    cfg.body = BodyPlan::named("mid").unwrap_or_default();
    cfg.terrain = terrain;
    Runtime::new(cfg, &lexicon(), None).expect("runtime")
}

#[test]
fn heat_first_passage_matches_closed_form() {
    // Compare PhysDriven (noto-ts) against noto-core effects.rs closed-form endurance_s across efforts & actuators.
    let body = Body::new(&BodyPlan::uniform(Tier::Budget), BaseKind::Wheel, FingerCount::Five);
    let tau = body.arm.shoulder.thermal_tau_s as f64;
    for effort in [1.15f64, 1.35, 1.8] {
        let e2 = effort * effort;
        let closed_effects = -tau * (1.0 - 1.0 / e2).ln();
        let closed_ts = heat_crossing_s(0.0, e2, tau).expect("crosses");
        assert!((closed_effects - closed_ts).abs() < 1e-6, "{closed_effects} vs {closed_ts}");

        let dt = 0.5;
        let n_steps = ((closed_effects / dt).ceil() as usize + 20).min(4000);
        let phys = PhysDriven::new("heat:shoulder_r", State::Heat { tau_s: tau }, 1.0, dt);
        let drive = vec![e2; n_steps];
        let fp = phys.first_passage(0.0, &drive, &drive, &drive);
        let t50 = fp.t_s[1].expect("p50 crossing");
        assert!(
            (t50 - closed_effects).abs() <= dt * 2.0,
            "effort={effort}: PhysDriven t50={t50:.2}s vs closed={closed_effects:.2}s (extrap={})",
            fp.extrapolated
        );
    }
    // Also verify against an actual effects::evaluate mining scenario that produces an `endurance_s` cell.
    let fx_mine = effects::library().into_iter().find(|s| s.name_ko.contains("채취"));
    if let Some(scn) = fx_mine {
        let res = effects::evaluate(&body, &scn);
        if let Some(cell) = res.cells.iter().find(|c| c.axis == "endurance_s" && c.value.is_some_and(|v| v.is_finite())) {
            let t_fx = cell.value.unwrap() as f64;
            assert!(t_fx > 0.0 && t_fx.is_finite());
        }
    }
}

#[test]
fn sequence_heat_equals_world_heat() {
    // Run World::tick under a held standing posture with cargo and compare world.heat to wear::heat_step.
    let mut cfg = WorldConfig::default();
    cfg.seed = 7;
    cfg.terrain = TerrainKind::Open;
    cfg.body = BodyPlan::uniform(Tier::Budget);
    cfg.noto = NotoConfig { base: BaseKind::HumanFoot, fingers: FingerCount::Five, ..cfg.noto };
    let mut w = World::new(cfg).expect("world");
    w.agent.carried.push((ItemKind::Log, 2.0));
    let ticks = 30usize;
    let dt = 0.1f32;
    let mut manual_heat: std::collections::BTreeMap<String, f32> = std::collections::BTreeMap::new();
    for _ in 0..ticks {
        let (post, _, _) = w.posture_scored();
        let rep = w.body.solve(&post, &FeedForward::default(), 1e3);
        for j in &rep.joints {
            if let Some(act) = w.joint_actuator(&j.joint) {
                let h = manual_heat.entry(j.joint.clone()).or_insert(0.0);
                *h = noto_core::wear::heat_step(*h, j.effort, dt, act.thermal_tau_s);
            }
        }
        w.tick();
    }
    assert!(!w.heat.is_empty(), "world accumulated heat");
    for (joint, h_world) in &w.heat {
        let h_manual = manual_heat.get(joint).copied().unwrap_or(0.0);
        assert!(
            (h_world - h_manual).abs() < 1e-4,
            "joint {joint}: world={h_world:.6} vs heat_step={h_manual:.6}"
        );
    }
}

#[test]
fn offline_only_phases_never_match_live() {
    let mut rt = make_rt(42, BaseKind::HumanFoot, TerrainKind::Open);
    let _ = rt.api_command_submit("원목 1개 가져와");
    for _ in 0..200 {
        rt.tick();
    }
    let status = rt.api_call("nn_status", &json!({})).expect("nn_status");
    let phases = status["phases"].as_array().expect("phases array");
    for p in phases {
        let scn = p["scenario"].as_str().unwrap_or("");
        let ph = p["phase"].as_str().unwrap_or("");
        assert!(!(scn == "S3" && ph == "reach"), "OfflineOnly S3/reach matched live!");
        assert!(scn != "S5-A", "RecordOnly S5-A matched live!");
        assert!(!matches!(scn, "C1" | "C2" | "C3"), "OfflineOnly crew {scn} matched live!");
    }
}

#[test]
fn a_pick_on_a_slope_scores_s1_lift_live() {
    let mut cfg = RuntimeConfig::default();
    cfg.seed = 11;
    cfg.noto = NotoConfig { base: BaseKind::HumanFoot, fingers: FingerCount::Five, ..cfg.noto };
    cfg.body = BodyPlan::named("pro").unwrap_or_default();
    cfg.terrain = TerrainKind::Rugged;
    let mut rt = Runtime::new(cfg, &lexicon(), None).expect("runtime");
    // Find an uphill sloped cell (4..=12 deg) where the robot can cleanly reach and pick up an item.
    let drop_id = 9991u32;
    let mut picked_deg = None;
    'outer: for ix in -16..=16 {
        for iz in -16..=16 {
            let (x, z) = (ix as f32 * 0.75, iz as f32 * 0.75);
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                let (pitch, _) = rt.world().chassis_attitude(x, z, yaw);
                let deg = pitch.to_degrees();
                if !(4.0..=11.0).contains(&deg) {
                    continue;
                }
                for dist in [0.45f32, 0.55, 0.35] {
                    let w = rt.world_mut();
                    w.agent.x = x;
                    w.agent.z = z;
                    w.agent.yaw = yaw;
                    w.drops.retain(|d| d.id != drop_id);
                    let dx = x + dist * yaw.cos();
                    let dz = z + dist * yaw.sin();
                    w.drops.push(DropItem::whole(drop_id, ItemKind::IronOre, dx, dz));
                    if w.command(Action::pick(drop_id)).is_ok() {
                        picked_deg = Some(deg);
                        break 'outer;
                    }
                }
            }
        }
    }
    let deg = picked_deg.expect("rugged terrain must yield a reachable sloped pick");
    for _ in 0..15 {
        rt.tick();
    }
    let st = rt.api_call("nn_status", &json!({})).expect("nn_status");
    let phases = st["phases"].as_array().expect("phases array");
    let seen_s1_lift = phases
        .iter()
        .any(|p| p["scenario"] == "S1" && p["phase"] == "lift" && p["count"].as_u64().unwrap_or(0) > 0);
    assert!(
        seen_s1_lift,
        "Pick on a {deg:.1}° slope must score S1/lift live; seen={phases:?}"
    );
}

#[test]
fn sub_worker_static_cells_equal_first_worker() {
    let mut rt = make_rt(7, BaseKind::Wheel, TerrainKind::Open);
    let wid = rt.world_mut().add_crew(&[CrewForm::Forklift], 1)[0];
    {
        let w = rt.world_mut();
        let cw = w.crew.iter_mut().find(|c| c.id == wid).unwrap();
        cw.carried.push((ItemKind::Log, 2.0));
        cw.job = Job::Go { x: cw.x + 5.0, z: cw.z, reach_m: 0.5 };
        cw.speed_mps = 0.6;
    }
    let cw = rt.world().crew_worker(wid).unwrap().clone();
    let (cw_post, _) = cw.posture_now(0.0, Surface::Firm);
    let ff = FeedForward::default();
    let rep_cw = cw.body.solve(&cw_post, &ff, 1e3);
    let cells_cw = nn::cells(&cw.body, &cw_post, &rep_cw, None);
    let rep_first = rt.world().body.solve(&cw_post, &ff, 1e3);
    let cells_first = nn::cells(&rt.world().body, &cw_post, &rep_first, None);
    assert_eq!(cells_cw.len(), cells_first.len());
    for (a, b) in cells_cw.iter().zip(&cells_first) {
        assert_eq!((a.source, a.target, &a.axis), (b.source, b.target, &b.axis));
        assert!((a.value - b.value).abs() < 1e-4, "{}: {} vs {}", a.axis, a.value, b.value);
    }
    let before_w1 = rt.api_call("nn_status", &json!({})).unwrap()["phases"].as_array().unwrap().len();
    let crew_eval = rt.record_crew_static_nn();
    assert_eq!(crew_eval.len(), 1);
    assert_eq!(crew_eval[0]["static_only"], json!(true));
    assert!(!crew_eval[0]["phases"].as_array().unwrap().is_empty(), "sub-worker matched static phases: {crew_eval:?}");
    let after_w1 = rt.api_call("nn_status", &json!({})).unwrap()["phases"].as_array().unwrap().len();
    assert_eq!(after_w1, before_w1, "sub-worker static scoring must not pollute worker 1 dynamic PhaseStat");
}

#[test]
fn a_new_def_shows_up_everywhere() {
    let mut rt = make_rt(1, BaseKind::HumanFoot, TerrainKind::Open);
    let nn_st = rt.api_call("nn_status", &json!({})).expect("nn_status");
    let defs_arr = nn_st["defs"].as_array().expect("defs array");
    for id in ["S1", "S4", "S5-A", "S7", "T1", "T2", "C1", "Y1", "L1", "G1", "G2", "E1", "E2", "M1"] {
        assert!(defs_arr.iter().any(|d| d["id"] == id), "nn_status.defs missing {id}");
    }
    let cov = nn_st["coverage"].as_object().expect("coverage object");
    assert_eq!(cov.len(), 7, "7 morph::FORMS in coverage");
    assert_eq!(cov["truck"]["S6"], json!("partial"));
    assert_eq!(cov["legs_humanoid"]["T3"], json!("evaluated"));
    let stats = rt.api_test_stats().expect("test_stats");
    let axes = stats["axes"].as_array().expect("dynamic axes array");
    assert!(axes.iter().any(|a| a == "sway_gain"), "S5-A series axis must auto-appear in test_stats.axes");
    assert!(axes.iter().any(|a| a == "heat_max"), "T1/T2 series axis must auto-appear in test_stats.axes");
}

#[test]
fn phase_residual_i2_and_cumulative_wear_rate() {
    let mut rt = make_rt(3, BaseKind::HumanFoot, TerrainKind::Open);
    rt.world_mut().agent.carried.push((ItemKind::Log, 1.0));
    let _ = rt.api_command_submit("창고로 가");
    for _ in 0..80 {
        rt.tick();
    }
    // Check shift ledger & i2_mixed keys exist on nn_status.
    let st = rt.api_call("nn_status", &json!({})).expect("nn_status");
    assert!(st["shift"].is_object(), "shift ledger must be present");
    assert!(st["i2_mixed"].is_object(), "i2_mixed map must be present");
    assert!(st["unseen_expected"].is_array(), "unseen_expected must be present");
    // Verify cumulative wear scope & per-hour rate are recorded.
    let effort_keys = rt.sds().scope_keys_for_track(Track::Effort);
    assert!(effort_keys.iter().any(|k| k.starts_with("effort|wear")), "wear scope recorded: {effort_keys:?}");
    let wear_key = Scope::new(Track::Effort, "wear", "").key_primary();
    let wear_prim = rt.sds().merged_scope_by_key(Track::Effort.ledger(), &wear_key).expect("effort|wear primary");
    assert!(wear_prim.baseline.keys().any(|k| k.ends_with(":life_used")), "life_used recorded");
    assert!(wear_prim.baseline.keys().any(|k| k.ends_with(":life_rate_per_h")), "life_rate_per_h recorded: {:?}", wear_prim.baseline.keys().collect::<Vec<_>>());
    // Feed two distinct phase scores to confirm I² flags heterogeneity (`mixed: true`).
    {
        let sds = rt.sds_mut();
        let sc_a = Scope::new(Track::Balance, "nn:S1", "slope");
        let sc_b = Scope::new(Track::Balance, "nn:S1", "lift");
        for _ in 0..5 {
            sds.record_baseline(&sc_a, "score", 95.0);
            sds.record_baseline(&sc_b, "score", 55.0);
        }
    }
    let st2 = rt.api_call("nn_status", &json!({})).expect("nn_status");
    let s1_i2 = &st2["i2_mixed"]["S1"];
    assert_eq!(s1_i2["mixed"], json!(true), "divergent phases must trigger I² mixed=true: {s1_i2}");
    assert!(s1_i2["i2"].as_f64().unwrap() > 0.5);
}

#[test]
fn t1_t2_adapter_respects_engine_rev_and_holdout() {
    let base: Vec<f64> = (0..16).map(|i| 0.4 + 0.05 * i as f64).collect();
    let cov: Vec<Vec<f64>> = (0..16).map(|i| vec![1.0 + 0.2 * (i % 5) as f64]).collect();
    let truth: Vec<f64> = (0..16).map(|i| 1.15 * base[i] + 0.1 + 0.25 * cov[i][0]).collect();
    // Group B & Truth must be refused for control-bound adapters.
    assert!(AffineAdapter::fit("nn.score", ENGINE_REV, Group::B, true, &base, &truth).is_err());
    assert!(ResidualAdapter::fit("nav.belief_err_m", ENGINE_REV, Group::Truth, true, &base, &truth, &cov).is_err());
    let t1 = AffineAdapter::fit("heat:shoulder_r", ENGINE_REV, Group::A, true, &base[..10], &truth[..10]).unwrap();
    let t2 = ResidualAdapter::fit("locate.score", ENGINE_REV, Group::A, true, &base[..10], &truth[..10], &cov[..10]).unwrap();
    assert!(t1.is_valid_for(ENGINE_REV) && !t1.is_valid_for("noto-2.0.0-next"));
    assert!(t2.is_valid_for(ENGINE_REV) && !t2.is_valid_for("noto-2.0.0-next"));
    // Evaluate T2 on held-out tail 10..16.
    let pred_t2: Vec<f64> = (10..16).map(|i| base[i] + t2.correction(&cov[i])).collect();
    let b0: Vec<(f64, f64)> = base[10..16].iter().map(|&p| (p - 0.8, p + 0.8)).collect();
    let b2: Vec<(f64, f64)> = pred_t2.iter().map(|&p| (p - 0.8, p + 0.8)).collect();
    let ev = evaluate_adoption(&truth[10..16], &base[10..16], &b0, &pred_t2, &b2);
    assert!(ev.adopted && ev.rmse_adapted < ev.rmse_base, "T2 eval: {ev:?}");
}
