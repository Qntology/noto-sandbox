use serde::Serialize;

use crate::sim::ShapeKind;
use crate::surface::Surface;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemMaterial {
    Wood,
    Stone,
    Coal,
    IronOre,
    Cardboard,
}

impl ItemMaterial {
    pub const ALL: [ItemMaterial; 5] = [ItemMaterial::Wood, ItemMaterial::Stone, ItemMaterial::Coal, ItemMaterial::IronOre, ItemMaterial::Cardboard];

    pub fn id(self) -> &'static str {
        match self {
            ItemMaterial::Wood => "wood",
            ItemMaterial::Stone => "stone",
            ItemMaterial::Coal => "coal",
            ItemMaterial::IronOre => "iron_ore",
            ItemMaterial::Cardboard => "cardboard",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            ItemMaterial::Wood => "나무",
            ItemMaterial::Stone => "돌",
            ItemMaterial::Coal => "석탄",
            ItemMaterial::IronOre => "철광석",
            ItemMaterial::Cardboard => "골판지",
        }
    }

    /// The shape the fingers meet: what the world is made of. Its renderer, its collision boxes, the tines and the
    /// stereo evaluation all hold every item as a box (a log too, 0.50 × 0.14 × 0.14 m), so the hand does as well —
    /// a round log only in the hand would show the eyes one thing and the fingers another. The world's sizes barely
    /// feel it anyway: five coupled fingers stop at 70 % closure on a 14 cm log as on the box, 38.9 N against 38.8 N
    /// (`sim::tests::print_top_grasps`); a round item changes the grip from about 6 cm down, where the fingertips go
    /// under it. When the world draws round logs, `Wood` becomes `ShapeKind::Cylinder { axis: 0 }` here.
    pub fn shape(self) -> ShapeKind {
        ShapeKind::Box
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Friction {
    pub mean: f32,
    pub cov: f32,
}

pub const PAD_MU_COV: f32 = 0.10;

pub fn pad_on(item: ItemMaterial) -> Friction {
    let mean = match item {
        ItemMaterial::Wood => 0.60,
        ItemMaterial::Stone => 0.83,
        ItemMaterial::Coal => 0.45,
        ItemMaterial::IronOre => 0.83,
        ItemMaterial::Cardboard => 0.60,
    };
    Friction { mean, cov: PAD_MU_COV }
}

/// An item on bare steel, the tines of a lift truck (static, kinetic) [추정: handbook ranges for dry contact].
pub fn on_steel(item: ItemMaterial) -> (Friction, Friction) {
    let s = match item {
        ItemMaterial::Wood => 0.45,
        ItemMaterial::Stone => 0.50,
        ItemMaterial::Coal => 0.45,
        ItemMaterial::IronOre => 0.55,
        ItemMaterial::Cardboard => 0.35,
    };
    let share = if item == ItemMaterial::Cardboard { CARDBOARD_KINETIC_SHARE } else { KINETIC_SHARE };
    (Friction { mean: s, cov: GROUND_SLIDE_COV }, Friction { mean: share * s, cov: GROUND_SLIDE_COV })
}

/// One piece of an item on another of the same kind (static, kinetic) [추정].
pub fn on_itself(item: ItemMaterial) -> (Friction, Friction) {
    let s = match item {
        ItemMaterial::Wood => 0.40,
        ItemMaterial::Stone => 0.60,
        ItemMaterial::Coal => 0.45,
        ItemMaterial::IronOre => 0.60,
        ItemMaterial::Cardboard => 0.40,
    };
    let share = if item == ItemMaterial::Cardboard { CARDBOARD_KINETIC_SHARE } else { KINETIC_SHARE };
    (Friction { mean: s, cov: GROUND_SLIDE_COV }, Friction { mean: share * s, cov: GROUND_SLIDE_COV })
}

pub const GROUND_SLIDE_COV: f32 = 0.10;
pub const KINETIC_SHARE: f32 = 0.9;
pub const CARDBOARD_KINETIC_SHARE: f32 = 0.75;

pub fn item_on(item: ItemMaterial, ground: Surface) -> (Friction, Friction) {
    let wood = item == ItemMaterial::Wood;
    if item == ItemMaterial::Cardboard {
        let s = match ground {
            Surface::Water => 0.0,
            Surface::WetSand => 0.5 * 0.74,
            _ => 0.50,
        };
        return (Friction { mean: s, cov: GROUND_SLIDE_COV }, Friction { mean: CARDBOARD_KINETIC_SHARE * s, cov: GROUND_SLIDE_COV });
    }
    let (s, k) = match ground {
        Surface::Firm => (if wood { 0.58 } else { 0.55 }, None),
        Surface::Rock => (0.62, None),
        Surface::Dirt => (if wood { 0.58 } else { 0.50 }, None),
        Surface::Grass => (0.50, Some(0.45)),
        Surface::Gravel => (0.57, None),
        Surface::Sand => (0.55, Some(0.50)),
        Surface::WetSand => (0.41, Some(0.37)),
        Surface::Water => (0.0, Some(0.0)),
    };
    let k = k.unwrap_or(KINETIC_SHARE * s);
    (Friction { mean: s, cov: GROUND_SLIDE_COV }, Friction { mean: k, cov: GROUND_SLIDE_COV })
}

pub fn restitution(item: ItemMaterial, ground: Surface) -> f32 {
    if item == ItemMaterial::Cardboard {
        let stone = restitution(ItemMaterial::Stone, ground);
        return 0.25 * stone / restitution(ItemMaterial::Stone, Surface::Rock);
    }
    match ground {
        Surface::Firm | Surface::Rock => 0.21,
        Surface::Sand | Surface::WetSand => 0.10,
        Surface::Gravel => 0.15,
        Surface::Dirt => 0.12,
        Surface::Grass => 0.10,
        Surface::Water => 0.0,
    }
}

pub fn ground_stiffness(ground: Surface) -> f32 {
    match ground {
        Surface::Firm | Surface::Rock => 2.0e6,
        Surface::WetSand => 4.5e5,
        Surface::Sand => 2.0e5,
        Surface::Gravel => 6.0e5,
        Surface::Dirt => 3.0e5,
        Surface::Grass => 1.5e5,
        Surface::Water => 1.0e4,
    }
}

pub const PAD_E_PA: f32 = 0.80e6;
pub const PAD_NU: f32 = 0.48;
pub const PAD_RADIUS_M: f32 = 0.010;
pub const FINGER_MASS_KG: f32 = 0.025;

pub fn contact_radius_m(n: f32) -> f32 {
    let e_star = PAD_E_PA / (1.0 - PAD_NU * PAD_NU);
    (3.0 * n.max(0.0) * PAD_RADIUS_M / (4.0 * e_star)).cbrt()
}

pub fn torsion_capacity_nm(mu: f32, n: f32) -> f32 {
    3.0 * std::f32::consts::PI / 16.0 * mu * n * contact_radius_m(n)
}

/// The silicone a phalanx pad can give before its core: half the 9 mm pad (`sim::PHALANX_PAD_M`) over a rigid
/// bone [추정].
pub const PAD_THICKNESS_M: f32 = 0.0045;

/// How far a pad pressed with `n` (N) sinks in (m). Hertz gives `δ = a²/R` for the patch `contact_radius_m` makes; a
/// layer of finite thickness on a rigid core follows it while it is thin against the layer and stiffens as it nears the
/// core, never passing it: `h·(1 − e^(−δ/h))` [추정: the saturating form; the patch radius itself stays Hertz, so the
/// contact model is unchanged].
pub fn pad_indentation_m(n: f32) -> f32 {
    let a = contact_radius_m(n);
    let d = a * a / PAD_RADIUS_M;
    PAD_THICKNESS_M * (1.0 - (-d / PAD_THICKNESS_M).exp())
}

/// The pressure in the middle of the patch (kPa): Hertz `3N / (2πa²)`, in the unit of `contact::SolePatch::peak_kpa`.
pub fn pad_peak_kpa(n: f32) -> f32 {
    let a = contact_radius_m(n).max(1e-6);
    3.0 * n.max(0.0) / (2.0 * std::f32::consts::PI * a * a) / 1000.0
}

pub fn impact(v: f32, m: f32, k: f32) -> (f32, f32) {
    (v.abs() * (k * m).sqrt(), std::f32::consts::PI * (m / k).sqrt())
}

pub fn pad_stiffness(n: f32) -> f32 {
    let e_star = PAD_E_PA / (1.0 - PAD_NU * PAD_NU);
    2.0 * e_star * contact_radius_m(n.max(0.5))
}

pub fn stereo_depth_sigma(z_m: f32, focal_px: f32, baseline_m: f32, subpixel: f32) -> f32 {
    z_m * z_m * subpixel / (focal_px * baseline_m).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_contacts_grow_with_force_and_twist_more() {
        let (a1, a2) = (contact_radius_m(10.0), contact_radius_m(80.0));
        assert!(a2 > a1 && (a2 / a1 - 2.0).abs() < 0.01, "a ∝ N^(1/3): {a1} {a2}");
        assert!(torsion_capacity_nm(0.6, 80.0) > 8.0 * torsion_capacity_nm(0.6, 10.0));
        let (f, dt) = impact(0.4, 0.025, pad_stiffness(5.0));
        assert!(f > 1.0 && f < 50.0 && dt > 1e-4 && dt < 0.05, "{f} N over {dt} s");
        assert!(restitution(ItemMaterial::Stone, Surface::Firm) > restitution(ItemMaterial::Stone, Surface::Sand));
        let (s, k) = item_on(ItemMaterial::Wood, Surface::Gravel);
        assert!(k.mean < s.mean);
        assert!((stereo_depth_sigma(1.0, 424.0, 0.05, 0.08) - 0.0038).abs() < 1e-4);
    }

    #[test]
    fn a_pad_sinks_like_hertz_when_light_and_never_through_its_layer() {
        let hertz = |n: f32| contact_radius_m(n).powi(2) / PAD_RADIUS_M;
        assert!((pad_indentation_m(0.2) / hertz(0.2) - 1.0).abs() < 0.03, "a light touch is the Hertz approach");
        let mut last = 0.0;
        for n in [0.5f32, 1.0, 5.0, 10.0, 30.0, 60.0, 100.0, 400.0] {
            let d = pad_indentation_m(n);
            assert!(d > last && d < PAD_THICKNESS_M && d <= hertz(n), "{n} N: {d}");
            last = d;
        }
        assert!(pad_indentation_m(100.0) > 0.8 * PAD_THICKNESS_M, "a hard squeeze nears the core");
        let (p1, p8) = (pad_peak_kpa(10.0), pad_peak_kpa(80.0));
        assert!(p1 > 100.0 && p1 < 500.0 && (p8 / p1 - 2.0).abs() < 0.01, "p ∝ N^(1/3): {p1} {p8}");
    }

    #[test]
    fn the_hand_meets_items_in_the_shape_the_world_draws_them() {
        assert!(ItemMaterial::ALL.iter().all(|m| m.shape() == ShapeKind::Box));
    }
}
