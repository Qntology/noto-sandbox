//! One lift recorded on the five-finger walker (the world's own build), played on other bodies and hands (`act::retarget`): the
//! record is the log's path, each palm's place and the demonstrating hand's contacts on the log; each embodiment
//! closes its own hand on the log, merges the contacts onto its fingers and reaches the palm targets frame by frame.
//!
//!   cargo test -p noto-runtime --test retarget -- --ignored --nocapture

use noto_core::act::retarget::{retarget, Demo, DemoContact, Embodiment, Rigid};
use noto_core::hand::FingerCount;
use noto_core::joint::Actuation;
use noto_core::parts::BodyPlan;
use noto_core::sim::{top_grasp_shaped, ShapeKind};
use noto_core::{BaseKind, NotoConfig};
use world_sim::world::DropItem;
use world_sim::{Action, ItemKind, World, WorldConfig};

fn walker_with_log() -> World {
    let mut noto = NotoConfig::preset(FingerCount::Five);
    noto.base = BaseKind::HumanFoot;
    let mut w = World::new(WorldConfig { seed: 7, parcels: Some(0), noto, ..WorldConfig::default() }).unwrap();
    w.nodes.clear();
    w.drops.clear();
    w.agent.yaw = 0.0;
    w.drops.push(DropItem::whole(77, ItemKind::Log, w.agent.x + 0.6, w.agent.z));
    w
}

/// The lift as a record, in the body frame (`reach`'s: x forward, y across, z up), and the walker's own body plan.
fn record() -> (Demo, BodyPlan) {
    let mut w = walker_with_log();
    let (dx, dz) = (w.drops[0].x, w.drops[0].z);
    w.command(Action::Look { x: dx, z: dz }).unwrap();
    while w.busy() {
        w.tick();
    }
    w.command(Action::pick(77)).unwrap();
    let (mut obj, mut wrist) = (Vec::new(), Vec::new());
    let mut item = None;
    while w.busy() {
        w.tick();
        let (Some((kind, size, c)), Some(pose)) = (w.hand_load().filter(|_| matches!(w.act_phase().map(|p| p.1), Some("lift" | "hold"))), w.body_pose()) else { continue };
        item.get_or_insert((kind, size));
        let boxes = w.holding_boxes(&pose);
        let palm = |name: &str| boxes.iter().find(|b| b.name == name).map(|b| Rigid { r: [0, 1, 2].map(|i| [0, 1, 2].map(|j| b.axes[j][i] as f64)), t: b.center.map(|v| v as f64) });
        obj.push(Rigid { r: Rigid::ID.r, t: c.map(|v| v as f64) });
        wrist.push([palm("palm_r"), palm("palm_l")]);
    }
    let (kind, size) = item.expect("the walker lifts the log");
    // The demonstrating hand's contacts on the item, item frame: where each phalanx presses and the force it can put
    // into the face (−normal × its squeeze), from the closing statics at the walker's size.
    let hand = w.body.working_hand(&w.hand).into_owned().scaled(w.body.dims.k);
    let ext = [size[2], size[0], size[1]];
    let tg = top_grasp_shaped(&hand, ext, ShapeKind::Box);
    let contacts: Vec<DemoContact> = tg
        .contacts
        .iter()
        .filter(|c| c.max_n > 1e-6)
        .map(|c| {
            let finger = if hand.fingers[c.finger].is_thumb { 0 } else { c.finger + 1 };
            DemoContact { finger, point: [(c.point.x - tg.center.x) as f64, (c.point.y - tg.center.y) as f64, (c.point.z - tg.center.z) as f64], force: [-(c.normal.x * c.max_n) as f64, -(c.normal.y * c.max_n) as f64, -(c.normal.z * c.max_n) as f64] }
        })
        .collect();
    let mass_kg = kind.spec().mass_kg;
    let mu = noto_core::act::material::pad_on(world_sim::world::material_of(kind)).mean;
    (Demo { dt_s: world_sim::TICK_S, obj, wrist, contacts: [contacts.clone(), contacts], size_m: ext, shape: ShapeKind::Box, mass_kg, mu, hand_k: w.body.dims.k }, w.cfg.body)
}

#[test]
#[ignore]
fn one_lift_on_other_bodies_and_hands() {
    let (demo, own) = record();
    for i in [0usize, demo.obj.len() / 2, demo.obj.len() - 1] {
        let w = demo.wrist[i];
        println!("[RETARGET] 프레임 {i}: 물건 {:?} · 오른 손바닥 {:?} · 왼 손바닥 {:?}", demo.obj[i].t, w[0].map(|r| r.t), w[1].map(|r| r.t));
    }
    println!("[RETARGET] 기록: 프레임 {} · 통나무 {:.1} kg · μ {:.2} · 시연 손 접촉 {}개 · 시연 몸 k {:.2}", demo.obj.len(), demo.mass_kg, demo.mu, demo.contacts[0].len(), demo.hand_k);
    let compact = BodyPlan::default();
    assert_ne!(own, compact, "the walker is not the compact build");
    let bodies: Vec<Embodiment> = vec![
        ("시연 몸 · 5 · coupled (시연)", own, FingerCount::Five, None),
        ("시연 몸 · 4 · coupled", own, FingerCount::Four, None),
        ("시연 몸 · 3 · coupled", own, FingerCount::Three, None),
        ("시연 몸 · 2 · direct", own, FingerCount::Two, None),
        ("시연 몸 · 5 · direct", own, FingerCount::Five, Some(Actuation::Direct)),
        ("compact · 5 · coupled", compact, FingerCount::Five, None),
        ("compact · 4 · coupled", compact, FingerCount::Four, None),
        ("compact · 3 · coupled", compact, FingerCount::Three, None),
        ("compact · 2 · direct", compact, FingerCount::Two, None),
        ("compact · 5 · direct", compact, FingerCount::Five, Some(Actuation::Direct)),
    ]
    .into_iter()
    .map(|(name, plan, fingers, actuation)| Embodiment { name: name.into(), plan, base: BaseKind::HumanFoot, fingers, actuation })
    .collect();
    println!("| 몸 · 손 | k | 쥐는 힘 최대 N | 한 손이 버티는 힘 N | 필요한 손 | 쥐기 여유 | 쥠 | 접촉 Δ힘 N | Δ모멘트 N·m | 손목 도달 | 최소 ZMP 여유 | 최대 Σ(τ/τmax)² |");
    let mut out = Vec::new();
    for e in &bodies {
        let r = retarget(&demo, e);
        println!(
            "| {} | {:.2} | {:.1} | {:.1} | {} | {:.2} | {} | {:.2} | {:.3} | {:.0} % | {:.2} | {:.3} |",
            r.name, r.k, r.squeeze_max_n, r.hold_n, r.hands, r.hold_margin, if r.hold_ok { "예" } else { "아니오" }, r.force_err_n, r.moment_err_nm, r.reach_share * 100.0, r.min_zmp_margin, r.max_effort
        );
        out.push(r);
    }
    assert!(out[0].reach_share > 0.6, "the demonstrating body plays most of its own lift: {}", out[0].reach_share);
    assert!(out[0].force_err_n < 1e-3, "its own hand keeps every contact");
}
