use noto_data::test_tables::{write_mirror, write_ts_samples, TsSampleEntry, TABLE_BALANCE_EFFORT, TABLE_ML_KEYFRAMES, TABLE_NN_PHASES, TABLE_QUEST_RUNS, TABLE_QUEST_STATS, TABLE_QUEST_STEPS, TABLE_TS_SAMPLES};
use noto_data::LanceStore;
use noto_quest::records::{BalanceRow, KeyframeRow, NnPhaseRow, QuestRun, StepRecord, TestDb};

fn run(pass: &str, ok: bool) -> QuestRun {
    QuestRun {
        run_id: format!("t1-{pass}"),
        session: "s".into(),
        pass: pass.into(),
        quest_id: "q1".into(),
        kind: "fetch".into(),
        target: "log".into(),
        destination: Some("warehouse".into()),
        preset: "mid".into(),
        seed: 7,
        ok,
        failure: (!ok).then(|| "lift_overload".into()),
        ticks: 300,
        time_s: 30.0,
        energy_j: 4000.0,
        steps: vec![
            StepRecord { step: "locate".into(), attempts: 2, ok: true, fail_code: None, time_s: 0.0 },
            StepRecord { step: "collect".into(), attempts: 1, ok, fail_code: (!ok).then(|| "lift_overload".into()), time_s: 0.0 },
        ],
        created_ms: 1,
        generation: "noto-1.0.0-wave1+000000000000".into(),
        ledger_digest: "abcd1234".into(),
        sds_mode: "live".into(),
    }
}

#[tokio::test]
async fn test_records_round_trip_through_lance() {
    let dir = std::env::temp_dir().join(format!("noto-data-testtables-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = LanceStore::open(dir.to_str().unwrap()).await.unwrap();

    let mut db = TestDb::memory();
    db.record_run(run("game", true));
    db.record_run(run("ml", false));
    db.record_keyframe(KeyframeRow {
        run_id: "t1-ml".into(),
        pass: "ml".into(),
        index: 0,
        tick: 0,
        time_s: 0.0,
        target: "tree".into(),
        depth_mode: "stand-in".into(),
        depth_absrel: Some(0.2),
        depth_delta1: None,
        locate_mode: "stand-in".into(),
        locate_score: Some(0.8),
        locate_err_px: None,
        track_mode: "region-grow".into(),
        track_id: Some(3),
        track_iou_prev: None,
        track_iou_truth: Some(0.5),
        tri_range_m: Some(6.1),
        tri_parallax_deg: Some(8.0),
        depth_scale: Some(1.25),
        range_used_m: Some(6.1),
        range_true_m: Some(6.0),
        decision: "tri_step(angled, 0.9 m)".into(),
        ts_method: "last".into(),
        ts_residual: None,
        fusion_flags: String::new(),
        balance_margin: Some(0.7),
        effort_max: Some(0.3),
        generation: "noto-1.0.0-wave1+000000000000".into(),
        ledger_digest: "abcd1234".into(),
        sds_mode: "live".into(),
    });
    db.record_balance(BalanceRow {
        event_id: "t1-game#10".into(),
        run_id: "t1-game".into(),
        pass: "game".into(),
        tick: 10,
        source: "arm_lift".into(),
        source_seg: "arm".into(),
        target: "foot".into(),
        state: "flat|pick|com_shift_frac".into(),
        preset: "mid".into(),
        comp: 0.3,
        comp_unit: String::new(),
        reference: None,
        tolerance: None,
        torque_nm: 12.0,
        effort: 0.4,
        effort_noff: 0.5,
        margin: 0.6,
        margin_noff: 0.5,
        ff_gain: 1.0,
        generation: "noto-1.0.0-wave1+000000000000".into(),
        ledger_digest: "abcd1234".into(),
        sds_mode: "live".into(),
    });
    db.record_nn_phase(NnPhaseRow {
        event_id: "t1-game#10".into(),
        run_id: "t1-game".into(),
        pass: "game".into(),
        tick: 10,
        time_s: 1.0,
        instance_id: 1,
        span_id: 2,
        scenario: "S4".into(),
        phase: "brake_1.5".into(),
        holdout: false,
        form: "mid:wheel:f5".into(),
        gnd: "firm".into(),
        slope: "flat".into(),
        stg: "move".into(),
        wk: "main".into(),
        score: 82.5,
        item: 2,
        gate_codes: String::new(),
        beyond_human: false,
        static_margin: 0.6,
        min_margin: 0.45,
        peak_ratio: 0.52,
        sustained_effort: 0.31,
        worst_joint: Some("knee_r".into()),
        generation: "noto-1.0.0-wave1+000000000000".into(),
        ledger_digest: "abcd1234".into(),
        sds_mode: "live".into(),
    });
    let batch = db.take_mirror();
    let c = write_mirror(&store, &batch, &db.stats()).await.unwrap();
    assert_eq!((c.quest_runs, c.quest_steps, c.ml_keyframes, c.balance_effort, c.nn_phases), (2, 4, 1, 1, 1));
    assert_eq!(store.row_count(TABLE_QUEST_RUNS).await.unwrap(), 2);
    assert_eq!(store.row_count(TABLE_QUEST_STEPS).await.unwrap(), 4);
    assert_eq!(store.row_count(TABLE_ML_KEYFRAMES).await.unwrap(), 1);
    assert_eq!(store.row_count(TABLE_BALANCE_EFFORT).await.unwrap(), 1);
    assert_eq!(store.row_count(TABLE_NN_PHASES).await.unwrap(), 1);
    let stats = store.row_count(TABLE_QUEST_STATS).await.unwrap();
    assert_eq!(stats, db.stats().len());
    write_mirror(&store, &db.take_mirror(), &db.stats()).await.unwrap();
    assert_eq!(store.row_count(TABLE_QUEST_STATS).await.unwrap(), stats);
    let ts_n = write_ts_samples(
        &store,
        &[
            TsSampleEntry { run_id: "t1-game".into(), worker: 1, channel: "bal.margin".into(), subject: String::new(), dtick: 0, sub_ms: 0, value: 0.12, q: 1 },
            TsSampleEntry { run_id: "t1-game".into(), worker: 1, channel: "nn.score".into(), subject: "s4:brake_1.5".into(), dtick: 50, sub_ms: 0, value: 84.0, q: 1 },
        ],
    )
    .await
    .unwrap();
    assert_eq!(ts_n, 2);
    assert_eq!(store.row_count(TABLE_TS_SAMPLES).await.unwrap(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}
