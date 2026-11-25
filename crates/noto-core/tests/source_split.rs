use std::path::PathBuf;

use noto_core::bench::{engine_files, engine_text, reference_reads, source_hits, BENCH_SOURCES, DECISION_FILES, NOT_ENGINE, SOURCE_EXCEPTIONS};

fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn the_registry_finds_the_engine_and_leaves_the_benchmarks_out() {
    let files = engine_files(&root());
    for f in [
        "crates/noto-core/src/kin.rs",
        "crates/noto-core/src/balance.rs",
        "crates/noto-core/src/act/grasp.rs",
        "crates/noto-core/src/act/gait.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/world/acts.rs",
        "crates/world-sim/src/world/eye.rs",
        "crates/world-sim/src/render.rs",
        "crates/noto-runtime/src/stereo.rs",
        "crates/noto-runtime/src/driver.rs",
        "crates/noto-runtime/src/vision.rs",
    ] {
        assert!(files.iter().any(|x| x == f), "{f} is engine: {files:?}");
    }
    for f in ["crates/noto-core/src/actbench.rs", "crates/noto-core/src/bench.rs", "crates/noto-core/src/effects.rs", "crates/noto-core/src/nn.rs", "crates/noto-core/src/lib.rs", "crates/noto-runtime/src/actsds.rs", "crates/noto-runtime/src/api.rs"] {
        assert!(!files.iter().any(|x| x == f), "{f} is not engine");
    }
    for (f, _) in NOT_ENGINE {
        assert!(root().join(f).is_file(), "{f}");
    }
    for (f, _, _) in SOURCE_EXCEPTIONS {
        assert!(root().join(f).is_file(), "{f}");
    }
    let balance = read("crates/noto-core/src/balance.rs");
    let engine = engine_text("crates/noto-core/src/balance.rs", &balance);
    assert!(engine.len() < balance.len() && engine.contains("impl Body") && !engine.contains("pub fn reference("));
    println!("[SRC] 엔진 파일 {}개 · 영역 {}개의 채점 출처 {}개", files.len(), BENCH_SOURCES.len(), BENCH_SOURCES.iter().map(|s| s.names.len()).sum::<usize>());
}

#[test]
fn the_engine_never_names_any_benchmark_source() {
    let mut hits = Vec::new();
    for f in engine_files(&root()) {
        let text = read(&f);
        for (area, name) in source_hits(&f, &text) {
            hits.push(format!("{f} names the {area} benchmark source `{name}`"));
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn decisions_never_read_reference_axes() {
    for f in DECISION_FILES {
        let reads = reference_reads(&read(f));
        assert!(reads.is_empty(), "{f} reads {reads:?}");
    }
}

#[test]
fn decisions_never_read_scenario_data() {
    for f in DECISION_FILES {
        let text = read(f);
        assert!(!text.contains("nn_scenarios"), "{f} must not reference nn_scenarios data");
    }
    for f in engine_files(&root()) {
        let text = read(&f);
        assert!(!text.contains("nn_scenarios.json"), "engine file {f} must not embed nn_scenarios.json");
    }
}

#[test]
fn decision_files_never_read_display_channels() {
    const DISPLAY_CHANNELS: &[&str] = &[
        "nn.score",
        "nn.min_margin",
        "nn.peak_ratio",
        "nn.sustained_effort",
        "nav.belief_err_m",
    ];
    for f in DECISION_FILES {
        let text = read(f);
        for ch in DISPLAY_CHANNELS {
            assert!(!text.contains(ch), "decision file {f} must not reference display-only channel `{ch}`");
        }
    }
}

#[test]
fn every_area_lists_distinct_sources() {
    let mut areas: Vec<&str> = BENCH_SOURCES.iter().map(|s| s.area).collect();
    areas.sort();
    areas.dedup();
    assert_eq!(areas.len(), BENCH_SOURCES.len(), "one row per area");
    for s in BENCH_SOURCES {
        assert!(!s.names.is_empty(), "{}", s.area);
        let mut n = s.names.to_vec();
        n.sort();
        n.dedup();
        assert_eq!(n.len(), s.names.len(), "{} lists a name twice", s.area);
    }
}

#[test]
fn decisions_never_read_live_nn() {
    for f in DECISION_FILES {
        let text = read(f);
        assert!(!text.contains("nn_live"), "{f} must not read nn_live");
        assert!(!text.contains("shift_ledger"), "{f} must not read shift_ledger");
    }
}

#[test]
fn holdout_seeds_live_in_one_place() {
    for f in [
        "crates/noto-core/tests/nn_scenarios.rs",
        "crates/noto-core/tests/action_bench.rs",
        "crates/noto-core/tests/effects_bench.rs",
        "crates/noto-core/tests/tactile_bench.rs",
    ] {
        let text = read(f);
        assert!(!text.contains("20261024"), "{f} hardcodes holdout seed 20261024; use bench::holdout_seed");
        assert!(!text.contains("20261027"), "{f} hardcodes holdout seed 20261027; use bench::holdout_seed");
        assert!(text.contains("holdout_seed("), "{f} should pull seed from bench::holdout_seed");
    }
    let body = noto_core::balance::Body::new(&noto_core::parts::BodyPlan::default(), noto_core::BaseKind::HumanFoot, noto_core::FingerCount::Five);
    let s = noto_core::bench::holdout_seed("NN");
    let a = noto_core::nn::holdout(&body, s, 8);
    let b = noto_core::nn::holdout(&body, s, 8);
    assert_eq!(a, b);
    assert!(a.iter().any(|x| x.phases.len() >= 2));
}
