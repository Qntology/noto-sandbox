use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::scope::Track;

/// Source classification for every SDS axis (Phase 2 – Part 3 · D5).
/// Control paths may only read `Spec` and `Experience` axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Derived purely from robot ratings, kinematics, and geometry (Group A).
    Spec,
    /// Measured from the robot's own sensors and action outcomes.
    Experience,
    /// Derived from human reference benchmarks or N:N human-likeness scores (Group B).
    Reference,
    /// Derived from simulator ground-truth labels/pixels/ranges.
    Truth,
}

impl Provenance {
    pub fn control_allowed(self) -> bool {
        matches!(self, Provenance::Spec | Provenance::Experience)
    }
}

/// Declared consumer role for an SDS axis (Phase 2 – Part 3 · D3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "target", rename_all = "snake_case")]
pub enum Reader {
    Path(&'static str),
    Display,
    Report,
    Unread,
}

/// Static declaration of an SDS axis pattern, its provenance, and its readers.
#[derive(Clone, Copy, Debug)]
pub struct AxisSpec {
    pub track: Track,
    /// Empty matches any primary; otherwise matched with `starts_with`.
    pub primary_prefix: &'static str,
    /// Exact match, `*` wildcard, `prefix*`, `*suffix`, or `prefix*suffix`.
    pub axis: &'static str,
    pub provenance: Provenance,
    pub readers: &'static [Reader],
}

fn match_pattern(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some((pre, suf)) = pattern.split_once('*') {
        value.len() >= pre.len() + suf.len() && value.starts_with(pre) && value.ends_with(suf)
    } else {
        pattern == value
    }
}

/// Complete catalog of SDS axes written and read across `noto-runtime` (Phase 2 – Part 3 · D3).
pub const AXES: &[AxisSpec] = &[
    // ── Vision ───────────────────────────────────────────────────
    AxisSpec { track: Track::Vision, primary_prefix: "", axis: "score", provenance: Provenance::Experience, readers: &[Reader::Path("A"), Reader::Display] },
    AxisSpec { track: Track::Vision, primary_prefix: "", axis: "score_all", provenance: Provenance::Experience, readers: &[Reader::Path("B"), Reader::Display] },
    AxisSpec { track: Track::Vision, primary_prefix: "", axis: "margin", provenance: Provenance::Experience, readers: &[Reader::Display] },
    AxisSpec { track: Track::Vision, primary_prefix: "", axis: "touch", provenance: Provenance::Experience, readers: &[Reader::Path("A"), Reader::Display] },
    AxisSpec { track: Track::Vision, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Forecast ─────────────────────────────────────────────────
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "fusion_mse:*", provenance: Provenance::Experience, readers: &[Reader::Path("C"), Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:res2:*", provenance: Provenance::Experience, readers: &[Reader::Path("C"), Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:res:*", provenance: Provenance::Experience, readers: &[Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:cov*", provenance: Provenance::Experience, readers: &[Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:pit", provenance: Provenance::Experience, readers: &[Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:is80", provenance: Provenance::Experience, readers: &[Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:band_k", provenance: Provenance::Experience, readers: &[Reader::Path("B"), Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*:slope_dlogd", provenance: Provenance::Experience, readers: &[Reader::Path("B"), Reader::Display] },
    AxisSpec { track: Track::Forecast, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Locate ───────────────────────────────────────────────────
    AxisSpec { track: Track::Locate, primary_prefix: "", axis: "score", provenance: Provenance::Experience, readers: &[Reader::Path("D"), Reader::Display] },
    AxisSpec { track: Track::Locate, primary_prefix: "", axis: "margin", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Locate, primary_prefix: "", axis: "presence", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Locate, primary_prefix: "", axis: "err_px", provenance: Provenance::Truth, readers: &[Reader::Report] },
    // ── Depth ────────────────────────────────────────────────────
    AxisSpec { track: Track::Depth, primary_prefix: "", axis: "scale", provenance: Provenance::Experience, readers: &[Reader::Path("E"), Reader::Display] },
    AxisSpec { track: Track::Depth, primary_prefix: "", axis: "absrel", provenance: Provenance::Truth, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Depth, primary_prefix: "", axis: "delta1", provenance: Provenance::Truth, readers: &[Reader::Report] },
    AxisSpec { track: Track::Depth, primary_prefix: "", axis: "stereo_valid", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Depth, primary_prefix: "", axis: "align_*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Tracker ──────────────────────────────────────────────────
    AxisSpec { track: Track::Tracker, primary_prefix: "", axis: "iou_truth", provenance: Provenance::Truth, readers: &[Reader::Report] },
    AxisSpec { track: Track::Tracker, primary_prefix: "", axis: "iou_prev", provenance: Provenance::Experience, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Tracker, primary_prefix: "", axis: "alive", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Tracker, primary_prefix: "", axis: "area_frac", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Tracker, primary_prefix: "", axis: "score", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Fusion ───────────────────────────────────────────────────
    AxisSpec { track: Track::Fusion, primary_prefix: "", axis: "commit_ok", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Fusion, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Display, Reader::Report] },
    // ── Decide ───────────────────────────────────────────────────
    AxisSpec { track: Track::Decide, primary_prefix: "", axis: "residual_deg", provenance: Provenance::Experience, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Decide, primary_prefix: "", axis: "tri_sigma_frac", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Memory ───────────────────────────────────────────────────
    AxisSpec { track: Track::Memory, primary_prefix: "", axis: "drift_var_per_m2", provenance: Provenance::Experience, readers: &[Reader::Path("G"), Reader::Display] },
    AxisSpec { track: Track::Memory, primary_prefix: "", axis: "scatter_var_m2", provenance: Provenance::Experience, readers: &[Reader::Path("H"), Reader::Display] },
    AxisSpec { track: Track::Memory, primary_prefix: "", axis: "innovation_z", provenance: Provenance::Experience, readers: &[Reader::Display, Reader::Report] },
    // ── Search / Motion / Quest ──────────────────────────────────
    AxisSpec { track: Track::Search, primary_prefix: "", axis: "score", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Motion, primary_prefix: "", axis: "route_time_s", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Motion, primary_prefix: "", axis: "mine_time_s", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Motion, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Quest, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    // ── Balance (NN / FX / Couplings / FF) ───────────────────────
    AxisSpec { track: Track::Balance, primary_prefix: "nn:", axis: "*", provenance: Provenance::Reference, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "fx:", axis: "*", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "ff", provenance: Provenance::Experience, readers: &[Reader::Path("F"), Reader::Display] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:dev", provenance: Provenance::Reference, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:absdev", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:dev2", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:ref", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:tol", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*:in_band", provenance: Provenance::Reference, readers: &[Reader::Display, Reader::Report] },
    AxisSpec { track: Track::Balance, primary_prefix: "", axis: "*", provenance: Provenance::Spec, readers: &[Reader::Display, Reader::Report] },
    // ── Effort ───────────────────────────────────────────────────
    AxisSpec { track: Track::Effort, primary_prefix: "", axis: "*:life_used", provenance: Provenance::Spec, readers: &[Reader::Report] },
    AxisSpec { track: Track::Effort, primary_prefix: "", axis: "*", provenance: Provenance::Spec, readers: &[Reader::Report] },
    // ── Act ──────────────────────────────────────────────────────
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:dev", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:absdev", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:dev2", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:ref", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:tol", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*:in_band", provenance: Provenance::Reference, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "p_success*", provenance: Provenance::Experience, readers: &[Reader::Report] },
    AxisSpec { track: Track::Act, primary_prefix: "", axis: "*", provenance: Provenance::Experience, readers: &[Reader::Display, Reader::Report] },
];

pub fn lookup(track: Track, primary: &str, axis: &str) -> Option<&'static AxisSpec> {
    let p = primary.trim().to_lowercase();
    AXES.iter().find(|spec| {
        spec.track == track
            && (spec.primary_prefix.is_empty() || p.starts_with(spec.primary_prefix))
            && match_pattern(spec.axis, axis)
    })
}

/// Hard provenance guard independent of catalog order:
/// any reference-derived suffix or `nn:`/`fx:` scope is always `Reference`,
/// and ground-truth accuracy metrics are always `Truth`.
pub fn classify(track: Track, primary: &str, axis: &str) -> Provenance {
    let p = primary.trim().to_lowercase();
    if axis.ends_with(":dev")
        || axis.ends_with(":absdev")
        || axis.ends_with(":dev2")
        || axis.ends_with(":ref")
        || axis.ends_with(":tol")
        || axis.ends_with(":in_band")
    {
        return Provenance::Reference;
    }
    if track == Track::Balance && (p.starts_with("nn:") || p.starts_with("fx:")) {
        return Provenance::Reference;
    }
    if matches!(
        (track, axis),
        (Track::Depth, "absrel" | "delta1") | (Track::Locate, "err_px") | (Track::Tracker, "iou_truth")
    ) {
        return Provenance::Truth;
    }
    lookup(track, &p, axis).map_or(Provenance::Experience, |s| s.provenance)
}

pub fn is_control_allowed(track: Track, primary: &str, axis: &str) -> bool {
    classify(track, primary, axis).control_allowed()
}

/// Runtime / test contact table recording which `(track, primary_prefix, axis)` pairs were read or written.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ContactLog {
    pub writes: BTreeMap<(String, String, String), u64>,
    pub reads: BTreeMap<(String, String, String), u64>,
}

impl ContactLog {
    pub fn record_write(&mut self, track: Track, primary: &str, axis: &str) {
        *self
            .writes
            .entry((track.as_str().into(), primary.trim().to_lowercase(), axis.to_string()))
            .or_insert(0) += 1;
    }

    pub fn record_read(&mut self, track: Track, primary: &str, axis: &str) {
        *self
            .reads
            .entry((track.as_str().into(), primary.trim().to_lowercase(), axis.to_string()))
            .or_insert(0) += 1;
    }

    pub fn uncatalogued_writes(&self) -> Vec<(String, String, String)> {
        self.writes
            .keys()
            .filter(|(t, p, a)| Track::from_name(t).map_or(true, |tr| lookup(tr, p, a).is_none()))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_and_truth_axes_are_never_control_allowed() {
        assert!(!is_control_allowed(Track::Balance, "nn:s4", "score"));
        assert!(!is_control_allowed(Track::Balance, "fx:slope", "margin"));
        assert!(!is_control_allowed(Track::Balance, "accel>foot", "los_frac:dev"));
        assert!(!is_control_allowed(Track::Balance, "accel>foot", "los_frac:in_band"));
        assert!(!is_control_allowed(Track::Act, "grasp", "lift:absdev"));
        assert!(!is_control_allowed(Track::Depth, "tips", "absrel"));
        assert!(!is_control_allowed(Track::Depth, "tips", "delta1"));
        assert!(!is_control_allowed(Track::Locate, "log", "err_px"));
        assert!(!is_control_allowed(Track::Tracker, "log", "iou_truth"));
    }

    #[test]
    fn feedback_paths_a_through_h_are_control_allowed() {
        assert!(is_control_allowed(Track::Vision, "tree", "score"));
        assert!(is_control_allowed(Track::Vision, "tree", "touch"));
        assert!(is_control_allowed(Track::Vision, "tree", "score_all"));
        assert!(is_control_allowed(Track::Forecast, "tree", "fusion_mse:holt"));
        assert!(is_control_allowed(Track::Locate, "tree", "score"));
        assert!(is_control_allowed(Track::Depth, "tips", "scale"));
        assert!(is_control_allowed(Track::Balance, "accel>foot", "ff"));
        assert!(is_control_allowed(Track::Memory, "wheel", "drift_var_per_m2"));
        assert!(is_control_allowed(Track::Memory, "wheel", "scatter_var_m2"));
    }
}
