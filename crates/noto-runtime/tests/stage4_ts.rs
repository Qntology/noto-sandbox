use noto_runtime::{Runtime, RuntimeConfig};
use serde_json::json;

fn lexicon() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json"))
        .expect("lexicon file")
}

fn new_rt(seed: u64) -> Runtime {
    let cfg = RuntimeConfig { seed, ..RuntimeConfig::default() };
    Runtime::new(cfg, &lexicon(), None).expect("runtime")
}

#[test]
fn a_long_carry_keeps_nn_and_wear_samples() {
    let mut cfg = RuntimeConfig::default();
    cfg.seed = 11;
    cfg.terrain = world_sim::TerrainKind::Rugged;
    let mut rt = Runtime::new(cfg, &lexicon(), None).expect("runtime");

    // Place the robot on a natural >= 5 deg rugged slope while holding cargo for 620 ticks.
    let w = rt.world().terrain.w;
    let d = rt.world().terrain.d;
    let mut pose = None;
    'find: for cz in 4..d - 4 {
        for cx in 4..w - 4 {
            if rt.world().terrain.is_water(cx, cz) {
                continue;
            }
            let (x, z) = world_sim::terrain::cell_center(cx, cz);
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                let (pitch, _) = rt.world().chassis_attitude(x, z, yaw);
                if pitch.to_degrees().abs() >= 5.0 {
                    pose = Some((x, z, yaw));
                    break 'find;
                }
            }
        }
    }
    let (sx, sz, syaw) = pose.expect("rugged terrain has >= 5 deg slope");
    rt.world_mut().agent.x = sx;
    rt.world_mut().agent.z = sz;
    rt.world_mut().agent.yaw = syaw;
    rt.world_mut().agent.carried.push((world_sim::ItemKind::Log, 1.0));

    for _ in 0..620 {
        rt.tick();
    }

    let wear_keys = rt.ts().matching_keys(Some(1), Some("wear.life_used"));
    assert!(!wear_keys.is_empty(), "wear.life_used channel must be populated");
    let wear_max = wear_keys
        .iter()
        .map(|k| rt.ts().samples_vec(1, "wear.life_used", &k.subject).len())
        .max()
        .unwrap_or(0);
    assert!(wear_max >= 12, "600 ticks must yield >= 12 wear samples (50-tick cadence), got {wear_max}");

    let nn_keys = rt.ts().matching_keys(Some(1), Some("nn.score"));
    assert!(!nn_keys.is_empty(), "slope carry must match an N:N phase and emit nn.score samples");
    let nn_max = nn_keys
        .iter()
        .map(|k| rt.ts().samples_vec(1, "nn.score", &k.subject).len())
        .max()
        .unwrap_or(0);
    assert!(nn_max >= 10, "expected >= 10 nn.score dwell samples over 620 ticks, got {nn_max}");

    // Verify 50-tick dwell resamples did NOT inflate SDS transition score count.
    let s1_slope = noto_runtime::sds::Scope::new(noto_runtime::sds::Track::Balance, "nn:S1", "slope");
    if let Some((_, _, n)) = rt.sds().exact_baseline(&s1_slope, "score") {
        assert!(n <= 4, "50-tick dwell resamples must NOT inflate SDS transition score n (got {n})");
    }
}

#[test]
fn samples_are_monotone_in_world_tick() {
    let mut rt = new_rt(13);
    for _ in 0..120 {
        rt.tick();
    }
    rt.scan_now("test").unwrap();
    for _ in 0..40 {
        rt.tick();
    }
    assert_eq!(rt.ts().late_count(), 0, "no late samples allowed");
    let keys = rt.ts().keys_json();
    for k in keys["keys"].as_array().unwrap() {
        let key_str = k["key"].as_str().unwrap();
        let (_, samples) = rt.ts().samples_by_key(key_str).unwrap();
        for w in samples.windows(2) {
            let ok = w[0].dtick < w[1].dtick || (w[0].dtick == w[1].dtick && w[0].sub_ms <= w[1].sub_ms);
            assert!(ok, "non-monotone samples on {key_str}: {:?} -> {:?}", w[0], w[1]);
        }
    }
    // Reset world and tick again: monotonicity must still hold cleanly without late samples.
    rt.reset_world().unwrap();
    for _ in 0..40 {
        rt.tick();
    }
    assert_eq!(rt.ts().late_count(), 0, "late count must stay 0 after world reset");
}

#[test]
fn belief_series_exists_without_recording() {
    let mut rt = new_rt(17);
    assert_eq!(rt.api_call("test_status", &json!({})).unwrap()["active"], false);
    for _ in 0..55 {
        rt.tick();
    }
    let samples = rt.ts().samples_vec(1, "nav.belief_err_m", "");
    assert!(samples.len() >= 10, "expected >= 10 nav.belief_err_m samples over 55 ticks without recording, got {}", samples.len());
    for w in samples.windows(2) {
        assert_eq!(w[1].dtick - w[0].dtick, 5, "expected 5-tick cadence: {:?}", samples);
    }
}

#[test]
fn ts_keys_and_ts_series_apis_work() {
    let mut rt = new_rt(21);
    for _ in 0..60 {
        rt.tick();
        if rt.world().tick % 3 == 0 {
            let _ = rt.scan_now("test");
        }
    }
    let keys = rt.api_call("ts_keys", &json!({})).unwrap();
    assert_eq!(keys["contract"], "noto-ts-v1");
    assert_eq!(keys["record_bytes"], 16);
    assert!(keys["keys"].as_array().unwrap().len() >= 6);

    let raw = rt.api_call("ts_series", &json!({"key": "ts:1:bal.margin:", "period": 0})).unwrap();
    assert!(raw["values"].as_array().unwrap().len() >= 25);
    let reg = rt.api_call("ts_series", &json!({"key": "ts:1:bal.margin:", "period": 2, "n": 25})).unwrap();
    assert_eq!(reg["values"].as_array().unwrap().len(), 25);
    assert!(reg["imputed_share"].as_f64().unwrap() <= 0.1);

    // Verify enriched ts_forecast contract (bands, coverage, horizons, models).
    if let Some((label, _)) = rt.gates().series_labels().first().cloned() {
        let fc = rt.api_ts_forecast(&format!("vision|{label}|"), "score").unwrap();
        assert!(fc["bands"]["model"]["p10"].is_array());
        assert!(fc["bands"]["cal80"]["lo"].is_array());
        assert!(fc["bands"]["gate"]["lo"].is_array());
        assert!(fc["coverage"].is_object());
        assert_eq!(fc["horizons"].as_array().unwrap().len(), 5);
        assert!(fc["models"].as_array().unwrap().len() >= 3);
    }
}
