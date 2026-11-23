//! Faceted SDS scope schema registry, syntax validation, fallback chain generation,
//! and documentation generator for `docs/sds-scopes.md` (Phase 2 – Part 1 · D1 & D4).

use std::sync::OnceLock;

use crate::scope::{Facet, ScopeKey, Track};

/// Specification of a single facet dimension.
#[derive(Clone, Copy, Debug)]
pub struct FacetSpec {
    pub facet: Facet,
    pub syntax: &'static str,
    pub source: &'static str,
    pub description: &'static str,
    /// Cardinality cap (2x observed distinct values; overflow folds to `other`).
    pub max_values: usize,
}

pub const FACET_SPECS: [FacetSpec; 12] = [
    FacetSpec { facet: Facet::Ph, syntax: "[a-z0-9_.:+-]{1..48}", source: "PhaseSpec.id / act phase", description: "N:N or act phase identifier", max_values: 64 },
    FacetSpec { facet: Facet::Set, syntax: "dev | holdout", source: "ScenarioSpec.holdout", description: "Development vs holdout evaluation split", max_values: 4 },
    FacetSpec { facet: Facet::Stg, syntax: "[a-z0-9_]{1..24}", source: "Progression stage vocabulary", description: "Progression stage (approach/look/grasp/lift/stow/reacquire/move/step/brake)", max_values: 24 },
    FacetSpec { facet: Facet::Form, syntax: "<plan>:<base>:f<n>", source: "bench::form_id", description: "Robot body form identifier", max_values: 64 },
    FacetSpec { facet: Facet::Body, syntax: "[a-z0-9_@:+-]{1..48}", source: "BodyPlan::id", description: "Body plan identifier (including rig edit hash)", max_values: 64 },
    FacetSpec { facet: Facet::Gnd, syntax: "firm|rock|dirt|grass|gravel|sand|wet_sand|water", source: "Ground::id", description: "Terrain ground surface", max_values: 16 },
    FacetSpec { facet: Facet::Slope, syntax: "flat|step|up_lo|up_hi|down_lo|down_hi", source: "World::slope_bucket", description: "Terrain slope or curb bucket", max_values: 16 },
    FacetSpec { facet: Facet::Act, syntax: "[a-z0-9_]{1..24}", source: "World action / posture tag", description: "Activity label (move/brake/mine/pick/deliver/drop/idle/hold)", max_values: 32 },
    FacetSpec { facet: Facet::Src, syntax: "quiet|arm_raise|arm_lift|foot_lift|step_up|slope|carry_front|accel", source: "Source::id", description: "Dominant balance excitation source", max_values: 16 },
    FacetSpec { facet: Facet::Mode, syntax: "[a-z0-9_.:@+-]{1..48}", source: "Recognizer / ML / TTM mode", description: "Perception, depth, locate, tracker, or predictor mode", max_values: 32 },
    FacetSpec { facet: Facet::Wk, syntax: "lead | crew-<form>", source: "Worker role", description: "Worker role identifier (lead or sub-worker role)", max_values: 16 },
    FacetSpec { facet: Facet::Pass, syntax: "game | ml | holdout | eval", source: "Runtime::pass", description: "Execution pass partition", max_values: 8 },
];

/// Track-level facet schema and write/read fallback chain.
#[derive(Clone, Copy, Debug)]
pub struct TrackSchema {
    pub track: Track,
    pub subject_prefix: &'static str,
    pub facets: &'static [Facet],
    /// Write/read fallback rungs below subject (shortest to most specific).
    pub chain: &'static [&'static [Facet]],
    /// Chain indices that receive detailed reference axes (e.g. `+ph`, `+ph+form`).
    pub detail_rungs: &'static [usize],
}

pub const BALANCE_NN: TrackSchema = TrackSchema {
    track: Track::Balance,
    subject_prefix: "nn:",
    facets: &[Facet::Ph, Facet::Set, Facet::Stg, Facet::Form, Facet::Gnd, Facet::Slope, Facet::Wk],
    chain: &[
        &[],
        &[Facet::Ph],
        &[Facet::Ph, Facet::Form],
        &[Facet::Ph, Facet::Form, Facet::Gnd, Facet::Slope],
    ],
    detail_rungs: &[1, 2],
};

pub const BALANCE_FX: TrackSchema = TrackSchema {
    track: Track::Balance,
    subject_prefix: "fx:",
    facets: &[Facet::Gnd, Facet::Slope, Facet::Act, Facet::Form],
    chain: &[&[], &[Facet::Gnd], &[Facet::Gnd, Facet::Slope, Facet::Act]],
    detail_rungs: &[1, 2],
};

pub const BALANCE_COUPLING: TrackSchema = TrackSchema {
    track: Track::Balance,
    subject_prefix: "",
    facets: &[Facet::Slope, Facet::Act, Facet::Form, Facet::Gnd, Facet::Stg, Facet::Wk],
    chain: &[&[], &[Facet::Slope, Facet::Act], &[Facet::Form, Facet::Slope, Facet::Act]],
    detail_rungs: &[1],
};

pub const EFFORT_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Effort,
    subject_prefix: "",
    facets: &[Facet::Src, Facet::Slope, Facet::Act, Facet::Form, Facet::Gnd, Facet::Stg],
    chain: &[&[], &[Facet::Src], &[Facet::Slope, Facet::Act, Facet::Src]],
    detail_rungs: &[1, 2],
};

pub const VISION_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Vision,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Stg],
    chain: &[&[], &[Facet::Mode], &[Facet::Stg, Facet::Mode]],
    detail_rungs: &[1],
};

pub const DEPTH_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Depth,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Stg],
    chain: &[&[], &[Facet::Mode], &[Facet::Stg, Facet::Mode]],
    detail_rungs: &[1],
};

pub const LOCATE_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Locate,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Stg],
    chain: &[&[], &[Facet::Mode], &[Facet::Stg, Facet::Mode]],
    detail_rungs: &[1],
};

pub const TRACKER_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Tracker,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Stg],
    chain: &[&[], &[Facet::Mode], &[Facet::Stg, Facet::Mode]],
    detail_rungs: &[1],
};

pub const ACT_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Act,
    subject_prefix: "",
    facets: &[Facet::Ph, Facet::Stg, Facet::Form, Facet::Body, Facet::Gnd, Facet::Slope, Facet::Act, Facet::Wk],
    chain: &[&[], &[Facet::Form], &[Facet::Gnd], &[Facet::Ph], &[Facet::Stg, Facet::Form]],
    detail_rungs: &[1, 2, 3],
};

pub const FORECAST_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Forecast,
    subject_prefix: "",
    facets: &[Facet::Mode],
    chain: &[&[], &[Facet::Mode]],
    detail_rungs: &[1],
};

pub const MEMORY_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Memory,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Gnd],
    chain: &[&[], &[Facet::Mode], &[Facet::Gnd, Facet::Mode]],
    detail_rungs: &[1],
};

pub const QUEST_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Quest,
    subject_prefix: "",
    facets: &[Facet::Stg, Facet::Pass],
    chain: &[&[], &[Facet::Pass], &[Facet::Stg, Facet::Pass]],
    detail_rungs: &[1],
};

pub const GENERIC_MODE_SCHEMA: TrackSchema = TrackSchema {
    track: Track::Motion,
    subject_prefix: "",
    facets: &[Facet::Mode, Facet::Stg, Facet::Form, Facet::Gnd],
    chain: &[&[], &[Facet::Mode]],
    detail_rungs: &[1],
};

pub const ALL_SCHEMAS: &[&TrackSchema] = &[
    &BALANCE_NN,
    &BALANCE_FX,
    &BALANCE_COUPLING,
    &EFFORT_SCHEMA,
    &VISION_SCHEMA,
    &DEPTH_SCHEMA,
    &LOCATE_SCHEMA,
    &TRACKER_SCHEMA,
    &ACT_SCHEMA,
    &FORECAST_SCHEMA,
    &MEMORY_SCHEMA,
    &QUEST_SCHEMA,
    &GENERIC_MODE_SCHEMA,
];

pub fn schema_for(track: Track, subject: &str) -> &'static TrackSchema {
    match track {
        Track::Balance => {
            if subject.starts_with("nn:") {
                &BALANCE_NN
            } else if subject.starts_with("fx:") {
                &BALANCE_FX
            } else {
                &BALANCE_COUPLING
            }
        }
        Track::Effort => &EFFORT_SCHEMA,
        Track::Vision => &VISION_SCHEMA,
        Track::Depth => &DEPTH_SCHEMA,
        Track::Locate => &LOCATE_SCHEMA,
        Track::Tracker => &TRACKER_SCHEMA,
        Track::Act => &ACT_SCHEMA,
        Track::Forecast => &FORECAST_SCHEMA,
        Track::Memory => &MEMORY_SCHEMA,
        Track::Quest => &QUEST_SCHEMA,
        Track::Motion | Track::Search | Track::Decide | Track::Fusion => &GENERIC_MODE_SCHEMA,
    }
}

pub fn facet_spec(f: Facet) -> &'static FacetSpec {
    &FACET_SPECS[f as usize]
}

/// Validate a [`ScopeKey`] against syntax limits and the track schema.
pub fn validate_key(key: &ScopeKey) -> Result<(), String> {
    if key.subject.len() > 64 {
        return Err(format!("subject '{}' exceeds 64 chars", key.subject));
    }
    if key.subject.is_empty() && !key.facets.is_empty() {
        return Err("facets are not allowed on global track key".into());
    }
    let schema = schema_for(key.track, &key.subject);
    for (f, v) in &key.facets {
        if v.is_empty() || v.len() > 48 {
            return Err(format!("facet {} value '{}' must be 1..=48 chars", f.as_str(), v));
        }
        if !schema.facets.contains(f) {
            return Err(format!("facet '{}' is not permitted on track '{}'", f.as_str(), key.track.as_str()));
        }
    }
    Ok(())
}

/// Compute ordered write/fallback chain keys from global up to the key's concrete facets.
pub fn chain_keys(key: &ScopeKey) -> Vec<String> {
    let mut out = vec![ScopeKey::global(key.track).to_key_string()];
    if key.subject.is_empty() {
        return out;
    }
    let schema = schema_for(key.track, &key.subject);
    for rung in schema.chain {
        if rung.iter().all(|f| key.facet(*f).is_some()) {
            let k = key.project(rung).to_key_string();
            if !out.contains(&k) {
                out.push(k);
            }
        }
    }
    let full = key.to_key_string();
    if !out.contains(&full) {
        out.push(full);
    }
    out
}

static REGISTRY_REV: OnceLock<String> = OnceLock::new();

/// Stable 8-hex digest of the schema registry (for v2 ledger provenance).
pub fn registry_rev() -> &'static str {
    REGISTRY_REV
        .get_or_init(|| {
            let doc = generate_docs();
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in doc.as_bytes() {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            format!("{:08x}", (h ^ (h >> 32)) as u32)
        })
        .as_str()
}

/// Generate the canonical Markdown reference `docs/sds-scopes.md` directly from the registry.
pub fn generate_docs() -> String {
    let mut out = String::new();
    out.push_str("# SDS ScopeKey v2 Registry\n\n");
    out.push_str("Generated from `crates/noto-sds/src/schema.rs` (Phase 2 – Part 1 · D1 & D4).\n\n");
    out.push_str("## Syntax\n\n");
    out.push_str("- Key grammar: `<track>/<subject>;<facet>=<value>;...`\n");
    out.push_str("- Reserved characters in values (`/`, `;`, `=`, `%`) are percent-encoded.\n");
    out.push_str("- Facets appear in fixed canonical order listed below; empty values are omitted.\n\n");
    out.push_str("## Facets\n\n");
    out.push_str("| Facet | Syntax | Source | Cap | Description |\n");
    out.push_str("| --- | --- | --- | ---: | --- |\n");
    for f in FACET_SPECS {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | {} | {} |\n",
            f.facet.as_str(),
            f.syntax,
            f.source,
            f.max_values,
            f.description
        ));
    }
    out.push_str("\n## Track Fallback Chains\n\n");
    out.push_str("| Track | Prefix | Allowed Facets | Fallback Chain |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for s in ALL_SCHEMAS {
        let allowed = s.facets.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ");
        let chain = s
            .chain
            .iter()
            .map(|r| {
                if r.is_empty() {
                    "subject".to_string()
                } else {
                    format!("+{}", r.iter().map(|f| f.as_str()).collect::<Vec<_>>().join("+"))
                }
            })
            .collect::<Vec<_>>()
            .join(" -> ");
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` |\n",
            s.track.as_str(),
            if s.subject_prefix.is_empty() { "*" } else { s.subject_prefix },
            allowed,
            chain
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_nn_chain_builds_all_rungs() {
        let key = ScopeKey::new(
            Track::Balance,
            "nn:s4",
            vec![
                (Facet::Ph, "brake_1.5".into()),
                (Facet::Form, "mid:humanfoot:f5".into()),
                (Facet::Gnd, "dirt".into()),
                (Facet::Slope, "flat".into()),
            ],
        );
        assert!(validate_key(&key).is_ok());
        assert_eq!(
            chain_keys(&key),
            vec![
                "balance/",
                "balance/nn:s4",
                "balance/nn:s4;ph=brake_1.5",
                "balance/nn:s4;ph=brake_1.5;form=mid:humanfoot:f5",
                "balance/nn:s4;ph=brake_1.5;form=mid:humanfoot:f5;gnd=dirt;slope=flat",
            ]
        );
    }

    #[test]
    fn schema_rejects_disallowed_facet() {
        let bad = ScopeKey::new(Track::Depth, "tree", vec![(Facet::Slope, "up_lo".into())]);
        assert!(validate_key(&bad).is_err());
    }

    #[test]
    fn schema_doc_is_current() {
        let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/sds-scopes.md"));
        let expected = generate_docs();
        let actual = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing {}: {e}", path.display()))
            .replace("\r\n", "\n");
        assert_eq!(actual, expected, "docs/sds-scopes.md drifted from schema.rs");
    }
}
