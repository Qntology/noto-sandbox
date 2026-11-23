//! Integration tests for Stage 2 — Phase 1 – Part 1 (D1–D5):
//! Time axis, BodySense, hysteresis phase spans, partitioning, and engine boundary.

use noto_core::{BaseKind, FingerCount, NotoConfig};
use noto_runtime::nnlive::{Clock, CloseWhy, NnLive};
use noto_runtime::{NoiseCfg, Runtime, RuntimeConfig};
use serde_json::Value;
use world_sim::{Action, ItemKind, TerrainKind, TimedEvent, WorldEvent};

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon")
}

fn make_rt(seed: u64, base: BaseKind, terrain: TerrainKind) -> Runtime {
    let mut cfg = RuntimeConfig::default();
    cfg.seed = seed;
    cfg.noto = NotoConfig::preset(FingerCount::Five);
    cfg.noto.base = base;
    cfg.terrain = terrain;
    cfg.noise = NoiseCfg::none();
    Runtime::new(cfg, &lexicon(), None).unwrap()
}

#[test]
fn nnlive_partitions_by_run_and_form() {
    let mut rt = make_rt(7, BaseKind::HumanFoot, TerrainKind::Open);
    rt.api_call("command_submit", &serde_json::json!({"text": "원목 1개를 창고로 가져와"})).unwrap();
    for _ in 0..800 {
        rt.tick();
        if !rt.busy() {
            break;
        }
    }
    let st1 = rt.api_call("nn_status", &Value::Null).unwrap();
    let run1 = st1["run"]["run_id"].as_str().unwrap().to_string();
    let form1 = st1["run"]["form"].as_str().unwrap().to_string();
    assert!(st1["seq"].as_u64().unwrap() > 0);

    // Run a different pass (integrated test creates game + ml passes) and then switch body form via reset_with.
    let mut hand = NotoConfig::preset(FingerCount::Five);
    hand.base = BaseKind::Wheel;
    rt.reset_with(Some(7), None, Some(hand)).unwrap();
    let st2 = rt.api_call("nn_status", &Value::Null).unwrap();
    let form2 = st2["run"]["form"].as_str().unwrap().to_string();
    assert_ne!(form1, form2, "changing base must switch partition form");
    assert!(st2["phases"].as_array().unwrap().is_empty(), "new form partition starts clean in summary()");
    assert!(st2["partitions"].as_u64().unwrap() >= 2, "previous partition kept alongside new one");

    // Simulate a third partition via explicit pass switch.
    let clk = Clock { seq: 9999, tick: 0, time_s: 0.0 };
    rt.api_call("test_start", &serde_json::json!({"command": "창고로 이동"})).unwrap();
    let _ = clk;
    let st3 = rt.api_call("nn_status", &Value::Null).unwrap();
    assert_ne!(st3["run"]["run_id"].as_str().unwrap(), run1);
    assert!(st3["partitions"].as_u64().unwrap() >= 3, "expected >= 3 partitions: {st3}");
}

#[test]
fn s1_lift_and_s6_strike_match_live_on_slope() {
    // Rugged terrain has natural slopes >= 3°. Find a sloped spot and verify Pick -> S1/lift and Mine -> S6/strike.
    let rt = make_rt(11, BaseKind::HumanFoot, TerrainKind::Rugged);
    let mut sloped_pose = None;
    let w = rt.world().terrain.w;
    let d = rt.world().terrain.d;
    'outer: for cz in 4..d - 4 {
        for cx in 4..w - 4 {
            if rt.world().terrain.is_water(cx, cz) {
                continue;
            }
            let (x, z) = world_sim::terrain::cell_center(cx, cz);
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                let (s, c) = yaw.sin_cos();
                let (pitch, _) = rt.world().chassis_attitude(x, z, yaw);
                let (pitch_fwd, _) = rt.world().chassis_attitude(x + 0.25 * c, z + 0.25 * s, yaw);
                if pitch.to_degrees().abs() >= 5.0 && pitch_fwd.to_degrees().abs() >= 4.0 {
                    sloped_pose = Some((x, z, yaw, pitch.to_degrees()));
                    break 'outer;
                }
            }
        }
    }
    let (sx, sz, syaw, deg) = sloped_pose.expect("rugged terrain must contain a sustained >= 5 deg slope");
    assert!(deg.abs() >= 5.0);

    // Place robot on slope facing pitch, place a tree/drop within reach, and verify posture_scored + NnLive.
    let mut world = rt.world().clone();
    world.agent.x = sx;
    world.agent.z = sz;
    world.agent.yaw = syaw;
    let (s, c) = syaw.sin_cos();
    let tx = sx + 0.30 * c;
    let tz = sz + 0.30 * s;
    world.drops.push(world_sim::world::DropItem::whole(9001, ItemKind::Parcel, tx, tz));
    world.nodes.push(world_sim::world::Node { id: 9002, kind: world_sim::NodeKind::Tree, x: sx + 0.55 * c, z: sz + 0.55 * s, hp: 2.0, max_hp: 2.0 });

    let mut live = NnLive::default();
    // 1) Pick on slope -> S1/lift
    world.command(Action::PickUp { drop: 9001, aim: None, side: false, memory_sigma_m: 0.0 }).unwrap();
    let mut seq = 0u64;
    for _ in 0..10 {
        let ev = world.tick();
        seq += 1;
        let (post, _, sense) = world.posture_scored();
        let (sm, pr) = world.latest_balance().map_or((0.0, 0.0), |r| (r.margin, r.worst().map_or(0.0, |w| w.peak_ratio)));
        live.observe(
            Clock { seq, tick: world.tick.saturating_sub(1), time_s: world.time_s },
            "slope-test",
            "game",
            &world.cfg.body,
            world.cfg.noto.base,
            world.cfg.noto.fingers,
            &world.body,
            &post,
            &sense,
            sm,
            pr,
            &ev,
        );
    }
    // 2) Mine on slope -> S6/strike
    world.cancel();
    world.agent.x = sx;
    world.agent.z = sz;
    world.agent.yaw = syaw;
    world.command(Action::Mine { node: 9002 }).unwrap();
    for _ in 0..10 {
        let ev = world.tick();
        seq += 1;
        let (post, _, sense) = world.posture_scored();
        let (sm, pr) = world.latest_balance().map_or((0.0, 0.0), |r| (r.margin, r.worst().map_or(0.0, |w| w.peak_ratio)));
        live.observe(
            Clock { seq, tick: world.tick.saturating_sub(1), time_s: world.time_s },
            "slope-test",
            "game",
            &world.cfg.body,
            world.cfg.noto.base,
            world.cfg.noto.fingers,
            &world.body,
            &post,
            &sense,
            sm,
            pr,
            &ev,
        );
    }
    live.close_all(CloseWhy::RunEnd, Clock { seq: seq + 1, tick: world.tick, time_s: world.time_s });
    let closed = &live.active_partition().closed_spans;
    assert!(closed.iter().any(|sp| sp.scenario == "S1" && sp.phase == "lift"), "expected S1/lift on slope: {closed:?}");
    assert!(closed.iter().any(|sp| sp.scenario == "S6" && sp.phase == "strike"), "expected S6/strike on slope: {closed:?}");
}

#[test]
fn s6_start_only_from_rest() {
    // Legged robot walks steadily on a slope, stops for a Hold, then restarts once.
    // Expected S6/start spans = 2 (one per acceleration from rest, NOT during steady walk).
    let plan = noto_core::BodyPlan::default();
    let body = noto_core::balance::Body::new(&plan, BaseKind::HumanFoot, FingerCount::Five);
    let mut live = NnLive::default();
    let mut seq = 0u64;
    let mut push = |tick: u64, post: noto_core::balance::Posture, sense: world_sim::BodySense, evs: &[TimedEvent]| {
        seq += 1;
        live.observe(
            Clock { seq, tick, time_s: tick as f64 * 0.1 },
            "s6-start",
            "game",
            &plan,
            BaseKind::HumanFoot,
            FingerCount::Five,
            &body,
            &post,
            &sense,
            0.4,
            0.3,
            evs,
        );
    };
    let start_post = noto_core::balance::Posture {
        source: noto_core::balance::Source::Accel,
        accel_mps2: 0.45,
        slope_deg: 7.0,
        moving: true,
        ..noto_core::balance::Posture::quiet(0.0)
    };
    let walk_post = noto_core::balance::Posture {
        source: noto_core::balance::Source::FootLift,
        single_support: true,
        slope_deg: 7.0,
        moving: true,
        ..noto_core::balance::Posture::quiet(0.0)
    };
    let hold_post = noto_core::balance::Posture {
        source: noto_core::balance::Source::Slope,
        slope_deg: 7.0,
        moving: false,
        ..noto_core::balance::Posture::quiet(0.0)
    };
    let mut t = 0u64;
    // Start #1 (4 ticks)
    for _ in 0..4 {
        push(t, start_post, world_sim::BodySense { accel_meas_mps2: 0.45, pitch_deg: 7.0, ..Default::default() }, &[]);
        t += 1;
    }
    // Steady walking on slope (20 ticks) -> must NOT open another S6/start
    for _ in 0..20 {
        push(t, walk_post, world_sim::BodySense { speed_mps: 0.8, accel_meas_mps2: 0.0, pitch_deg: 7.0, ..Default::default() }, &[]);
        t += 1;
    }
    // Hold stop (6 ticks)
    for k in 0..6 {
        let evs = if k == 0 {
            vec![TimedEvent { tick: t, time_s: t as f64 * 0.1, event: WorldEvent::Hold { worker: 1, other: 2, gap_m: 1.0, label: "hold".into() } }]
        } else {
            vec![]
        };
        push(t, hold_post, world_sim::BodySense { rest_s: 0.6, pitch_deg: 7.0, ..Default::default() }, &evs);
        t += 1;
    }
    // Start #2 after Hold (4 ticks)
    for _ in 0..4 {
        push(t, start_post, world_sim::BodySense { accel_meas_mps2: 0.45, pitch_deg: 7.0, ..Default::default() }, &[]);
        t += 1;
    }
    // Steady walking again (10 ticks)
    for _ in 0..10 {
        push(t, walk_post, world_sim::BodySense { speed_mps: 0.8, accel_meas_mps2: 0.0, pitch_deg: 7.0, ..Default::default() }, &[]);
        t += 1;
    }
    live.close_all(CloseWhy::RunEnd, Clock { seq: seq + 1, tick: t, time_s: t as f64 * 0.1 });
    let starts: Vec<_> = live.active_partition().closed_spans.iter().filter(|s| s.scenario == "S6" && s.phase == "start").collect();
    assert_eq!(starts.len(), 2, "S6/start should open once per start from rest: {starts:?}");
}

#[test]
fn brake_rung_follows_measured_decel() {
    // Wheeled robot carrying a log brakes smoothly at load_accel (< planned decel).
    let mut cfg = world_sim::WorldConfig::default();
    cfg.seed = 11;
    cfg.noto = NotoConfig::preset(FingerCount::Five);
    cfg.noto.base = BaseKind::Wheel;
    cfg.terrain = TerrainKind::Open;
    cfg.parcels = Some(0);
    let mut w = world_sim::World::new(cfg).unwrap();
    w.nodes.clear();
    w.drops.clear();
    w.agent.yaw = 0.0;
    w.agent.carried.push((ItemKind::Log, 1.0));
    let (x, z) = (w.agent.x, w.agent.z);
    w.command(Action::MoveTo { x: x + 12.0, z, reach_m: 3.0 }).unwrap();

    let mut live = NnLive::default();
    let mut seq = 0u64;
    let mut saw_brake_rung = false;
    let mut max_plan_gap = 0.0f32;
    for _ in 0..400 {
        let ev = w.tick();
        seq += 1;
        let (post, _, sense) = w.posture_scored();
        let (sm, pr) = w.latest_balance().map_or((0.0, 0.0), |r| (r.margin, r.worst().map_or(0.0, |w| w.peak_ratio)));
        live.observe(
            Clock { seq, tick: w.tick.saturating_sub(1), time_s: w.time_s },
            "brake-test",
            "game",
            &w.cfg.body,
            w.cfg.noto.base,
            w.cfg.noto.fingers,
            &w.body,
            &post,
            &sense,
            sm,
            pr,
            &ev,
        );
        if post.source == noto_core::balance::Source::Accel && post.accel_mps2 < 0.0 {
            let gap = (sense.accel_meas_mps2 - sense.accel_plan_mps2).abs();
            max_plan_gap = max_plan_gap.max(gap);
        }
        if !w.busy() {
            break;
        }
    }
    live.close_all(CloseWhy::RunEnd, Clock { seq: seq + 1, tick: w.tick, time_s: w.time_s });
    for sp in &live.active_partition().closed_spans {
        if sp.scenario == "S4" && sp.phase.starts_with("brake_") {
            saw_brake_rung = true;
            max_plan_gap = max_plan_gap.max(sp.plan_gap_max);
        }
    }
    assert!(saw_brake_rung, "expected S4 brake_* rung span: {:?}", live.active_partition().closed_spans);
    assert!(max_plan_gap > 1e-3, "expected non-zero plan_gap between measured and planned decel: {max_plan_gap}");
}

#[test]
fn world_events_mark_open_segments() {
    let plan = noto_core::BodyPlan::default();
    let body = noto_core::balance::Body::new(&plan, BaseKind::HumanFoot, FingerCount::Five);
    let mut live = NnLive::default();
    let step_post = noto_core::balance::Posture {
        source: noto_core::balance::Source::StepUp,
        step_m: 0.17,
        single_support: true,
        moving: true,
        ..noto_core::balance::Posture::quiet(6.0)
    };
    let sense = world_sim::BodySense { cargo_use_ratio: 0.72, ..Default::default() };
    // Open S2/step over 2 ticks, emit CargoSlip + Hold, then StepTaken closes the span.
    for t in 0..2u64 {
        live.observe(Clock { seq: t + 1, tick: t, time_s: t as f64 * 0.1 }, "ev", "game", &plan, BaseKind::HumanFoot, FingerCount::Five, &body, &step_post, &sense, 0.3, 0.4, &[]);
    }
    let evs = vec![
        TimedEvent { tick: 2, time_s: 0.2, event: WorldEvent::CargoSlip { worker: 1, item: ItemKind::Log, shift_m: 0.04, use_ratio: 0.88, label: "slip".into() } },
        TimedEvent { tick: 2, time_s: 0.2, event: WorldEvent::Hold { worker: 1, other: 2, gap_m: 1.2, label: "hold".into() } },
        TimedEvent { tick: 2, time_s: 0.2, event: WorldEvent::StepTaken { worker: 1, height_m: 0.17, crouch_m: 0.08, a_z: 1.4, label: "step".into() } },
    ];
    live.observe(Clock { seq: 3, tick: 2, time_s: 0.2 }, "ev", "game", &plan, BaseKind::HumanFoot, FingerCount::Five, &body, &step_post, &sense, 0.25, 0.5, &evs);
    let s2_step = live.active_partition().closed_spans.iter().find(|s| s.scenario == "S2" && s.phase == "step").expect("StepTaken must close S2/step");
    assert_eq!(s2_step.close, Some(CloseWhy::Event("step_taken")));
    assert!(s2_step.held, "Hold must mark held=true");
    assert!(s2_step.marks.iter().any(|m| m.kind == "cargo_slip"), "CargoSlip mark missing: {:?}", s2_step.marks);
    assert!(s2_step.marks.iter().any(|m| m.kind == "step_taken"), "StepTaken mark missing: {:?}", s2_step.marks);
}

#[test]
fn segments_are_deterministic() {
    let run_once = || {
        let mut rt = make_rt(7, BaseKind::HumanFoot, TerrainKind::Open);
        rt.api_call("command_submit", &serde_json::json!({"text": "원목 1개를 창고로 가져와"})).unwrap();
        for _ in 0..1200 {
            rt.tick();
            if !rt.busy() {
                break;
            }
        }
        rt.api_call("nn_status", &Value::Null).unwrap()["segments"].clone()
    };
    assert_eq!(run_once(), run_once(), "same seed and command must yield identical span sequence");
}

#[test]
fn engine_files_never_score() {
    let root = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    for rel in noto_core::bench::engine_files(&root) {
        let text = std::fs::read_to_string(root.join(&rel)).unwrap();
        for forbidden in ["noto_core::nn::", "crate::nn::", "score_phase(", "matching_phases(", "PhaseSpan", "SpanClosed"] {
            assert!(!text.contains(forbidden), "engine file {rel} must not contain `{forbidden}`");
        }
    }
}
