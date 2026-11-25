pub use lv_spatial::triangulate::{plan_step, Bearing, Fix, TriCfg, TriError, TriStep};

pub fn triangulate(rays: &[Bearing], min_parallax_deg: f32) -> Result<Fix, TriError> {
    let mut fix = lv_spatial::triangulate::triangulate(rays, min_parallax_deg)?;
    let (Some(first), Some(last)) = (rays.first(), rays.last()) else {
        return Ok(fix);
    };
    let range0 = (fix.x - first.x).hypot(fix.z - first.z);
    fix.sigma_m = (range0 * first.sigma).hypot(fix.range_m * last.sigma)
        / fix.parallax_deg.to_radians().sin().max(1e-3);
    Ok(fix)
}
