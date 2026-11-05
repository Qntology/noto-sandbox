// The pose a look reads from SGBM's depth inside an item's box (`noto_vision::item_pose`), on the renderer's stereo
// pairs: a walker bends its head over a log (its long axis along the world's x) from headings that put it along, across
// and at angles to the view, and over a stone and a parcel. The box is where the renderer drew the item, so this
// scores the pose alone, not the scan that finds the box.
//   cargo test -p noto-vision --test item_pose_eval -- --nocapture
use noto_core::{BaseKind, NotoConfig};
use noto_vision::{item_pose, sgbm, Intrinsics, ItemPose, Pose, PyramidCfg, SgbmCfg, StereoRig};
use world_sim::render::{focal_px, pack_id};
use world_sim::world::{camera_of, Action, DropItem, WorldConfig};
use world_sim::{decode_webp, encode_webp, Camera, Frame, ItemKind, ObjectClass, TerrainKind, World};

fn walker() -> World {
    let mut noto = NotoConfig::preset(noto_core::FingerCount::Five);
    noto.base = BaseKind::HumanFoot;
    let mut w = World::new(WorldConfig { seed: 7, parcels: Some(0), terrain: TerrainKind::Open, noto, ..WorldConfig::default() }).expect("world");
    w.nodes.clear();
    w.drops.clear();
    w
}

fn webp(f: &Frame) -> noto_vision::Rgb {
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, 95.0).expect("webp");
    let (w, h, data) = decode_webp(&enc.bytes).expect("webp");
    noto_vision::Rgb::new(w, h, data)
}

fn fold_deg(a: f32) -> f32 {
    let mut a = a % 180.0;
    if a > 90.0 {
        a -= 180.0;
    } else if a <= -90.0 {
        a += 180.0;
    }
    a
}

/// The walker faces `heading_deg` with the item `dist_m` ahead, looks down at it (the Look act), and reads its pose
/// from SGBM on the pair it sees then. Returns the pose and the log's true heading from the camera's.
fn read(kind: ItemKind, heading_deg: f32, dist_m: f32) -> (ItemPose, f32) {
    let mut w = walker();
    let h = heading_deg.to_radians();
    w.agent.yaw = h;
    let (x, z) = (w.agent.x + dist_m * h.cos(), w.agent.z + dist_m * h.sin());
    w.drops.push(DropItem::whole(77, kind, x, z));
    w.command(Action::Look { x, z }).expect("look");
    while w.busy() {
        w.tick();
    }
    let cam = Camera::default();
    let (_, b, _) = camera_of(w.body.head.sensor);
    let left = w.render(&cam);
    let right = w.render_right(&cam, b);
    let rig = StereoRig { focal_px: focal_px(&cam), baseline_m: b, subpixel_px: 0.27 };
    let out = sgbm(&webp(&left), &webp(&right), &rig, &SgbmCfg::default(), Some(&PyramidCfg::default())).expect("sgbm");
    let id = pack_id(ObjectClass::Drop, 77);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (i, &v) in left.ids.iter().enumerate() {
        if v == id {
            let (px, py) = (i % left.width, i / left.width);
            (x0, y0, x1, y1) = (x0.min(px), y0.min(py), x1.max(px), y1.max(py));
        }
    }
    assert!(x1 > x0 && y1 > y0, "{kind:?} at {heading_deg}°: the item is in view");
    let (fw, fh) = (left.width as f32, left.height as f32);
    let pad = 0.02;
    let bbox = [x0 as f32 / fw - pad, y0 as f32 / fh - pad, (x1 + 1) as f32 / fw + pad, (y1 + 1) as f32 / fh + pad].map(|v| v.clamp(0.0, 1.0));
    let v = w.head_view(&cam).view;
    let intr = Intrinsics { width: cam.width as u32, height: cam.height as u32, fov_deg: cam.fov_deg, pitch_deg: v.pitch.to_degrees(), roll_deg: v.roll.to_degrees() };
    let pose = Pose { x: v.origin[0], z: v.origin[2], yaw: v.yaw, eye_m: v.origin[1] - w.agent_ground_m() };
    let p = item_pose(&out.depth, &intr, &pose, bbox, 0.0).unwrap_or_else(|| panic!("{kind:?} at {heading_deg}°: a pose"));
    if std::env::var("ITEM_DEBUG").is_ok() {
        // The same pose from the item's own pixels only (the renderer's mask) and from the renderer's depth there.
        let mut masked = out.depth.clone();
        let mut truth = out.depth.clone();
        let (gw, gh) = (out.depth.width as usize, out.depth.height as usize);
        for gy in 0..gh {
            for gx in 0..gw {
                let fx = (((gx as f32 + 0.5) * fw / gw as f32) as usize).min(left.width - 1);
                let fy = (((gy as f32 + 0.5) * fh / gh as f32) as usize).min(left.height - 1);
                let on = left.ids[fy * left.width + fx] == id;
                if !on {
                    masked.data[gy * gw + gx] = f32::NAN;
                }
                truth.data[gy * gw + gx] = left.depth[fy * left.width + fx];
            }
        }
        println!("[ITEM-DEBUG] {kind:?} {heading_deg}°: sgbm {:?}", p);
        println!("[ITEM-DEBUG]   mask-only {:?}", item_pose(&masked, &intr, &pose, bbox, 0.0));
        println!("[ITEM-DEBUG]   renderer depth {:?}", item_pose(&truth, &intr, &pose, bbox, 0.0));
    }
    (p, fold_deg(-v.yaw.to_degrees()))
}

#[test]
fn a_look_reads_which_way_a_log_lies_and_how_big_it_is() {
    let size = ItemKind::Log.spec().size_m;
    for heading in [0.0f32, 25.0, -40.0, 60.0, 90.0] {
        let (p, truth) = read(ItemKind::Log, heading, 0.75);
        let off = fold_deg(p.rel_yaw_deg - truth).abs();
        println!("[ITEM] log, heading {heading:+.0}°: read {:+.1}° (true {truth:+.1}°, off {off:.1}°) · {:.2} × {:.2} × {:.2} m · top {:.1}° · {} · {} pts", p.rel_yaw_deg, p.length_m, p.width_m, p.height_m, p.top_tilt_deg, p.form, p.points);
        assert!(off < 6.0, "heading {heading}°: {p:?} vs {truth}");
        assert_eq!(p.form, "bar", "heading {heading}°: {p:?}");
        assert!((p.length_m - size[0]).abs() < 0.12 && (p.width_m - size[2]).abs() < 0.06 && (p.height_m - size[1]).abs() < 0.04, "heading {heading}°: {p:?}");
        assert!(p.top_tilt_deg < 8.0, "a log on flat ground has a level top: {p:?}");
    }
}

#[test]
fn a_stone_and_a_parcel_read_as_blocks_of_their_height() {
    for kind in [ItemKind::Stone, ItemKind::Parcel] {
        let size = kind.spec().size_m;
        let (p, _) = read(kind, 15.0, 0.7);
        println!("[ITEM] {kind:?}: {:.2} × {:.2} × {:.2} m · top {:.1}° · {} · {} pts", p.length_m, p.width_m, p.height_m, p.top_tilt_deg, p.form, p.points);
        assert_eq!(p.form, "block", "{kind:?}: {p:?}");
        assert!((p.height_m - size[1]).abs() < 0.04, "{kind:?}: {p:?}");
    }
}
