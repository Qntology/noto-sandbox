//! The hands as the eyes see them round what they hold. The physics' link boxes (`kin`) put a hand along its forearm and
//! curl the fingers by a closure share, knowing nothing of the item: on a log held from above the fingers curled up and
//! away from it, and in the first-person view the item floated with no hand on it. Here the palm turns to the face of
//! the item nearest the hand, every finger takes the angles it stopped at on that item (`sim::top_grasp_shaped`, on the
//! hand at the body's size), and a wrist link joins the forearm to the turned palm. Drawing only: collisions and falls
//! keep the physics' boxes.

use noto_core::dynamics::{BodyPose, LinkBox};
use noto_core::math::Vec3;
use noto_core::parts::Segment;
use noto_core::sim::{top_grasp_shaped, PHALANX_PAD_M};

use super::{material_of, World};

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f32 {
    dot(a, a).sqrt()
}

/// A link from `a` to `b` of half-width `w`, its z along the link (as `kin` builds its limbs).
fn link(name: &'static str, a: V3, b: V3, w: f32) -> LinkBox {
    let d = sub(b, a);
    let len = norm(d).max(1e-6);
    let z = scale(d, 1.0 / len);
    let x0 = if z[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let c = cross(z, x0);
    let y = scale(c, 1.0 / norm(c).max(1e-6));
    LinkBox { name, segment: Segment::Hand, center: scale(add(a, b), 0.5), axes: [cross(y, z), y, z], half: [w, w, 0.5 * len] }
}

/// How far a hand may be from the item and still be holding it (m, at the compact body's size).
const HOLD_REACH_M: f32 = 0.12;

/// One hand's frame round the item: x across the knuckles, y into the item (the palm's normal), z along the fingers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldFrame {
    pub side: usize,
    pub knuckles: V3,
    pub axes: [V3; 3],
}

impl World {
    /// The body pose's boxes with each hand that is on the item in it (`hand_load`) redrawn round that item.
    pub fn holding_boxes(&self, pose: &BodyPose) -> Vec<LinkBox> {
        let Some((kind, size, c)) = self.hand_load() else { return pose.boxes.clone() };
        let k = self.body.dims.k;
        let hand = self.body.working_hand(&self.hand).into_owned().scaled(k);
        let half = [0.5 * size[2], 0.5 * size[0], 0.5 * size[1]];
        let shape = material_of(kind).shape();
        let (mut out, mut held): (Vec<LinkBox>, [bool; 2]) = (Vec::with_capacity(pose.boxes.len() + 8), [false; 2]);
        let mut added = Vec::new();
        for side in 0..2 {
            let (palm_name, fore_name) = if side == 0 { ("palm_r", "forearm_r") } else { ("palm_l", "forearm_l") };
            let (Some(palm), Some(fore)) = (pose.boxes.iter().find(|b| b.name == palm_name), pose.boxes.iter().find(|b| b.name == fore_name)) else { continue };
            let tip = add(palm.center, scale(palm.axes[2], palm.half[2]));
            let gap: f32 = (0..3).map(|i| ((tip[i] - c[i]).abs() - half[i]).max(0.0).powi(2)).sum::<f32>().sqrt();
            if gap > HOLD_REACH_M * k {
                continue;
            }
            let wrist = add(fore.center, scale(fore.axes[2], fore.half[2]));
            let (frame, tg) = hold_frame(side, tip, c, half, |ext| top_grasp_shaped(&hand, ext, shape));
            let HoldFrame { knuckles, axes: [xh, yh, zh], .. } = frame;
            let to_body = |q: Vec3| add(knuckles, add(add(scale(xh, q.x), scale(yh, q.y)), scale(zh, q.z)));
            let palm_len = hand.palm.length;
            let thick = 0.62 * 0.035 * k;
            let heel = sub(knuckles, scale(zh, palm_len));
            added.push(LinkBox { name: palm_name, segment: Segment::Hand, center: add(sub(knuckles, scale(zh, 0.5 * palm_len)), scale(yh, -thick)), axes: [yh, xh, zh], half: [0.035 * k, 0.5 * hand.palm.width, 0.5 * palm_len] });
            added.push(link(if side == 0 { "wrist_r" } else { "wrist_l" }, wrist, add(heel, scale(yh, -thick)), 0.022 * k));
            for (fi, finger) in hand.fingers.iter().enumerate() {
                let pts: Vec<V3> = finger.points(&tg.angles[fi]).into_iter().map(to_body).collect();
                let name = match (finger.is_thumb, side) {
                    (true, 0) => "thumb_r",
                    (true, _) => "thumb_l",
                    (false, 0) => "finger_r",
                    (false, _) => "finger_l",
                };
                for w in pts.windows(2) {
                    added.push(link(name, w[0], w[1], PHALANX_PAD_M * k.sqrt()));
                }
            }
            held[side] = true;
        }
        for b in &pose.boxes {
            let side = if b.name.ends_with("_r") { 0 } else { 1 };
            let redrawn = matches!(b.name, "palm_r" | "palm_l" | "finger_r" | "finger_l") && held[side];
            if !redrawn {
                out.push(b.clone());
            }
        }
        out.extend(added);
        out
    }

    /// The forearms and hands (with what they hold redrawn round it), for the first-person view: the body's own arms
    /// come into the head camera's picture when it looks down at its hands.
    pub fn own_arm_boxes(&self, pose: &BodyPose) -> Vec<LinkBox> {
        self.holding_boxes(pose).into_iter().filter(|b| matches!(b.name, "forearm_r" | "forearm_l" | "wrist_r" | "wrist_l" | "palm_r" | "palm_l" | "finger_r" | "finger_l" | "thumb_r" | "thumb_l")).collect()
    }
}

/// How far the box of half-sizes `half` (body axes) reaches along `v`.
fn extent(v: V3, half: V3) -> f32 {
    v[0].abs() * half[0] + v[1].abs() * half[1] + v[2].abs() * half[2]
}

/// The palm on the face of the item nearest the hand point `tip` — its top or bottom, or its left or right side — with
/// the fingers forward over its far face (`top_grasp`'s placing: palm pad on the face, far face just behind the
/// knuckles), the hand kept where it is across the face.
pub fn hold_frame(side: usize, tip: V3, c: V3, half: V3, closed: impl Fn([f32; 3]) -> noto_core::sim::TopGrasp) -> (HoldFrame, noto_core::sim::TopGrasp) {
    let d = sub(tip, c);
    let (ry, rz) = (d[1].abs() / half[1].max(1e-3), d[2].abs() / half[2].max(1e-3));
    let yh = if rz >= ry { [0.0, 0.0, -d[2].signum()] } else { [0.0, -d[1].signum(), 0.0] };
    let zh = [1.0, 0.0, 0.0];
    // Right hand: x = y × z, so the thumb (at −x) is on the inside; the left hand is its mirror.
    let xr = cross(yh, zh);
    let xh = if side == 0 { xr } else { scale(xr, -1.0) };
    let axes = [xh, yh, zh];
    let ext = [0, 1, 2].map(|i| 2.0 * extent(axes[i], half));
    let tg = closed(ext);
    let room = (extent(xh, half) - 0.25 * ext[0].min(0.12)).max(0.0);
    let across = dot(d, xh).clamp(-room, room);
    let centre = [tg.center.x, tg.center.y, tg.center.z];
    let knuckles = add(sub(c, add(add(scale(xh, centre[0]), scale(yh, centre[1])), scale(zh, centre[2]))), scale(xh, across));
    (HoldFrame { side, knuckles, axes }, tg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::ItemKind;
    use crate::world::{Action, DropItem, WorldConfig};
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn walker_with(kind: ItemKind) -> World {
        let mut noto = NotoConfig::preset(FingerCount::Five);
        noto.base = BaseKind::HumanFoot;
        let mut w = World::new(WorldConfig { seed: 7, parcels: Some(0), noto, ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        w.agent.yaw = 0.0;
        w.drops.push(DropItem::whole(77, kind, w.agent.x + 0.6, w.agent.z));
        w
    }

    /// The signed distance from a body-frame point to the item box in the hands.
    fn into_item(p: V3, c: V3, half: V3) -> f32 {
        let q: Vec<f32> = (0..3).map(|i| (p[i] - c[i]).abs() - half[i]).collect();
        let out = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f32>().sqrt();
        out + q.iter().copied().fold(f32::NEG_INFINITY, f32::max).min(0.0)
    }

    #[test]
    fn a_held_log_has_the_palms_on_it_and_the_fingers_round_it_not_through_it() {
        let mut w = walker_with(ItemKind::Log);
        w.command(Action::pick(77)).unwrap();
        let (mut frames, mut touching_frames) = (0, 0);
        while w.busy() {
            w.tick();
            let (Some((_, size, c)), Some(pose)) = (w.hand_load().filter(|_| w.busy()), w.body_pose()) else { continue };
            let half = [0.5 * size[2], 0.5 * size[0], 0.5 * size[1]];
            let boxes = w.holding_boxes(&pose);
            let fingers: Vec<&LinkBox> = boxes.iter().filter(|b| b.name.starts_with("finger") || b.name.starts_with("thumb")).collect();
            assert!(boxes.iter().any(|b| b.name.starts_with("wrist")), "a held item turns the palm on a wrist");
            frames += 1;
            let k = w.body.dims.k;
            let mut touch = 0;
            for b in &fingers {
                for s in [-1.0f32, 0.0, 1.0] {
                    let p = add(b.center, scale(b.axes[2], s * b.half[2]));
                    let d = into_item(p, c, half) - b.half[0];
                    assert!(d > -0.004 * k, "a finger link goes {:.1} mm into the log", -d * 1000.0);
                    if d < 0.004 * k {
                        touch += 1;
                    }
                }
            }
            let palms: Vec<&LinkBox> = boxes.iter().filter(|b| b.name.starts_with("palm")).collect();
            assert!(!palms.is_empty());
            for p in palms {
                let inner = add(p.center, scale(p.axes[0], p.half[0] * 0.62));
                let d = into_item(inner, c, half);
                assert!(d.abs() < 0.03 * k, "the palm sits on the log: {:.3} m off", d);
            }
            if touch > 0 {
                touching_frames += 1;
            }
        }
        assert!(frames > 5 && touching_frames * 10 >= frames * 9, "fingers on the log in {touching_frames} of {frames} frames");
    }

    #[test]
    fn the_head_camera_sees_its_own_hands_on_the_log_it_lifts() {
        let mut w = walker_with(ItemKind::Log);
        let (dx, dz) = (w.drops[0].x, w.drops[0].z);
        w.command(Action::Look { x: dx, z: dz }).unwrap();
        while w.busy() {
            w.tick();
        }
        w.command(Action::pick(77)).unwrap();
        let cam = crate::Camera { width: 320, height: 213, ..crate::Camera::default() };
        let (mut frames, mut on_hand) = (0, 0);
        while w.busy() {
            w.tick();
            let (Some(_), Some(pose)) = (w.hand_load().filter(|_| matches!(w.act_phase().map(|p| p.1), Some("lift"))), w.body_pose()) else { continue };
            let f = w.render(&cam);
            let hv = w.head_view(&cam);
            let fingers: Vec<LinkBox> = w.own_arm_boxes(&pose).into_iter().filter(|b| b.name.starts_with("finger") || b.name.starts_with("thumb")).collect();
            frames += 1;
            // A finger in the picture is drawn there: the pixel its middle projects to is no item and no ground, and as
            // near as it. (The tips go round the log's far face, out of the head camera's sight behind it.)
            for b in &fingers {
                let at = w.body_to_world(b.center);
                let Some((x, y, z)) = cam.project_in(&hv.view, at) else { continue };
                let (i, j) = (x as usize, y as usize);
                if i >= f.width || j >= f.height {
                    continue;
                }
                let k = j * f.width + i;
                if f.ids[k] == 0 && (f.depth[k] - z).abs() < 0.05 {
                    on_hand += 1;
                    break;
                }
            }
        }
        assert!(frames > 5 && on_hand * 10 >= frames * 9, "the head camera sees a finger on the log in {on_hand} of {frames} lift frames");
    }

    /// PNGs of a log picked up: the head camera at each stage, the hand camera and a view from the side.
    ///   set HAND_FRAMES_DIR=...   cargo test -p world-sim --lib save_the_hands -- --ignored --nocapture
    #[test]
    #[ignore]
    fn save_the_hands_round_a_held_log() {
        let dir = std::path::PathBuf::from(std::env::var("HAND_FRAMES_DIR").unwrap_or_else(|_| ".".into()));
        std::fs::create_dir_all(&dir).unwrap();
        for kind in [ItemKind::Log, ItemKind::Stone, ItemKind::Parcel] {
            let mut w = walker_with(kind);
            let (dx, dz) = (w.drops[0].x, w.drops[0].z);
            w.command(Action::Look { x: dx, z: dz }).unwrap();
            while w.busy() {
                w.tick();
            }
            w.command(Action::pick(77)).unwrap();
            let (mut last, mut n) = (String::new(), 0);
            let cam = crate::Camera { width: 480, height: 320, ..crate::Camera::default() };
            let focus = crate::Camera { width: 480, height: 320, fov_deg: 50.0, max_dist_m: 12.0, ..crate::Camera::default() };
            let save = |f: &crate::Frame, name: String| image::save_buffer(dir.join(name), &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).unwrap();
            while w.busy() {
                w.tick();
                let phase = w.act_phase().map(|p| p.1).unwrap_or("-").to_string();
                if phase != last {
                    n += 1;
                    last = phase.clone();
                    let tag = format!("{kind:?}-{n:02}-{phase}").to_lowercase();
                    save(&w.render(&cam), format!("{tag}-eye.png"));
                    save(&crate::render::render_focus(&w, &focus, crate::render::FocusCam::Hand), format!("{tag}-hand.png"));
                    save(&crate::render::render_focus(&w, &focus, crate::render::FocusCam::Side), format!("{tag}-side.png"));
                    std::fs::write(dir.join(format!("{tag}-focus.json")), serde_json::to_string(&w.focus_frame()).unwrap()).unwrap();
                    println!("[HANDS] {tag}");
                }
            }
        }
    }
}
