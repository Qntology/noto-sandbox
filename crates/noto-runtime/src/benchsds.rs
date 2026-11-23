use noto_sds::{Scope, Track};
use world_sim::{TimedEvent, WorldEvent};

use crate::runtime::Runtime;

pub const CONTACT_KINDS: [(&str, &str); 7] = [("bump", "collide"), ("tip", "collide"), ("hit", "collide"), ("edge_fall", "collide"), ("incipient_slip", "tactile"), ("heel_off", "tactile"), ("toe_push", "tactile")];

pub const COOP_KINDS: [&str; 3] = ["yield", "co_drop", "overload_share"];

impl Runtime {
    pub(crate) fn record_bench_sds(&mut self, events: &[TimedEvent]) {
        self.record_contact_sds(events);
        self.record_coop_sds(events);
        self.record_world_event_sds(events);
    }

    /// WorldEvent physics adapter (Phase 2 – Part 3 · D3): Report-only Track::Act scopes.
    pub(crate) fn record_world_event_sds(&mut self, events: &[TimedEvent]) {
        let form = noto_core::bench::form_id(&self.world.body);
        let sds = self.active_sds_mut();
        for e in events {
            match &e.event {
                WorldEvent::Bump { peak_n, a_g, fell, .. } => {
                    let sc = Scope::new(Track::Act, "world:bump", &form);
                    sds.record_baseline(&sc, "peak_n", *peak_n as f64);
                    sds.record_baseline(&sc, "a_g", *a_g as f64);
                    sds.record_agreement(&sc, "ok", !*fell);
                }
                WorldEvent::Hold { gap_m, .. } => {
                    let sc = Scope::new(Track::Act, "world:hold", &form);
                    sds.record_baseline(&sc, "gap_m", *gap_m as f64);
                }
                WorldEvent::CargoSlip { shift_m, use_ratio, .. } => {
                    let sc = Scope::new(Track::Act, "world:cargo_slip", &form);
                    sds.record_baseline(&sc, "shift_m", *shift_m as f64);
                    sds.record_baseline(&sc, "use_ratio", *use_ratio as f64);
                }
                WorldEvent::CargoLost { .. } => {
                    let sc = Scope::new(Track::Act, "world:cargo_lost", &form);
                    sds.record_agreement(&sc, "ok", false);
                }
                WorldEvent::StepSeen { height_m, .. } => {
                    let sc = Scope::new(Track::Act, "world:step_seen", &form);
                    sds.record_baseline(&sc, "height_m", *height_m as f64);
                }
                WorldEvent::StepTaken { height_m, a_z, .. } => {
                    let sc = Scope::new(Track::Act, "world:step_taken", &form);
                    sds.record_baseline(&sc, "height_m", *height_m as f64);
                    sds.record_baseline(&sc, "a_z", *a_z as f64);
                    sds.record_agreement(&sc, "ok", true);
                }
                _ => {}
            }
        }
    }

    pub(crate) fn record_contact_sds(&mut self, events: &[TimedEvent]) {
        for e in events {
            let WorldEvent::Act { kind, ok, value, .. } = &e.event else { continue };
            let Some((_, area)) = CONTACT_KINDS.iter().find(|(k, _)| k == kind) else { continue };
            self.record_moment(&format!("{area}:{kind}"), *ok, *value);
        }
    }

    pub(crate) fn record_coop_sds(&mut self, events: &[TimedEvent]) {
        for e in events {
            let WorldEvent::Act { kind, ok, value, .. } = &e.event else { continue };
            if COOP_KINDS.contains(&kind.as_str()) {
                self.record_moment(&format!("coop:{kind}"), *ok, *value);
            }
        }
    }

    fn record_moment(&mut self, primary: &str, ok: bool, value: Option<f32>) {
        let form = noto_core::bench::form_id(&self.world.body);
        let ground = self.world.ground_here().id();
        let sds = self.active_sds_mut();
        sds.record_agreement(&Scope::new(Track::Act, primary, &form), "ok", ok);
        if let Some(v) = value {
            sds.record_baseline(&Scope::new(Track::Act, &format!("{primary}:ts"), ground), "value", v as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    fn rt() -> Runtime {
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        Runtime::new(RuntimeConfig::default(), &lex, None).unwrap()
    }

    fn act(kind: &str, ok: bool, value: Option<f32>) -> TimedEvent {
        TimedEvent { tick: 1, time_s: 0.1, event: WorldEvent::Act { act: "walk".into(), kind: kind.into(), label: String::new(), ok, value, parts: vec![] } }
    }

    #[test]
    fn contact_and_coop_moments_reach_their_scopes() {
        let mut rt = rt();
        let form = noto_core::bench::form_id(&rt.world().body);
        rt.record_bench_sds(&[act("bump", true, Some(310.0)), act("tip", false, None), act("toe_push", true, Some(0.4)), act("co_drop", false, Some(12.0)), act("contact", true, Some(1.0))]);
        let sds = rt.sds();
        assert_eq!(sds.agreement_rate(&Scope::new(Track::Act, "collide:bump", &form), "ok"), None, "one observation is not trusted yet");
        let st = sds.status(500);
        let has = |scope: &str, axis: &str| st.axes.iter().any(|a| a.scope == scope && a.axis == axis);
        assert!(has(&format!("act|collide:bump|{form}"), "ok"));
        assert!(has(&format!("act|collide:tip|{form}"), "ok"));
        assert!(has(&format!("act|tactile:toe_push|{form}"), "ok"));
        assert!(has(&format!("act|coop:co_drop|{form}"), "ok"));
        let ground = rt.world().ground_here().id();
        assert!(has(&format!("act|collide:bump:ts|{ground}"), "value"));
        assert!(!st.axes.iter().any(|a| a.scope.starts_with("act|collide:contact")), "a grasp contact is not a collision");
    }

    #[test]
    fn a_runtime_on_disk_stamps_its_ledgers_with_the_engine_revision() {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("noto-benchsds-{}-{nanos}", std::process::id()));
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        {
            let mut rt = Runtime::new(RuntimeConfig { parcels: Some(0), ..RuntimeConfig::default() }, &lex, Some(&dir)).unwrap();
            assert_eq!(rt.sds().engine_rev(), noto_core::bench::ENGINE_REV);
            rt.record_bench_sds(&[act("bump", true, Some(310.0))]);
            rt.sds_mut().flush();
        }
        let ledger: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sds").join(Track::Act.ledger().file_name())).unwrap()).unwrap();
        assert!(ledger["engine_revs"].as_array().unwrap().iter().any(|r| r == noto_core::bench::ENGINE_REV), "{}", ledger["engine_revs"]);
        let runs = dir.join("sds").join("runs").join(Track::Act.ledger().as_str());
        let run = std::fs::read_dir(&runs).unwrap().filter_map(|e| e.ok()).next().expect("a run file").path();
        let run: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run).unwrap()).unwrap();
        assert_eq!(run["engine_rev"], noto_core::bench::ENGINE_REV);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
