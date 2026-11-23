//! Granite TTM-R3 (real weights, `TTM_DIR`) next to the simple predictors on series whose truth is known: its error
//! one and four steps ahead, how much of what comes next its own q10–q90 band holds, and the weight the fused
//! forecast gives it.

use std::path::PathBuf;

use candle_core::Device;
use noto_ts::fusion::{HoltPredictor, LastPredictor, Predictor};
use noto_ts::ttm::TtmForecaster;
use noto_ts::{forecast, TtmPredictor};

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn series(kind: &str, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut x = 0.6;
    (0..n)
        .map(|t| {
            let e = 0.05 * rng.normal();
            x = match kind {
                "iid" => 0.6 + e,
                "ar1" => 0.6 + 0.7 * (x - 0.6) + e,
                "walk" => x + e,
                _ => 0.2 + 0.01 * t as f64 + e,
            };
            x
        })
        .collect()
}

#[test]
#[ignore = "needs TTM_DIR (Granite TTM-R3 weights)"]
fn granite_ttm_next_to_the_simple_predictors() {
    let dir = PathBuf::from(std::env::var("TTM_DIR").expect("TTM_DIR"));
    let ttm = TtmForecaster::load(&dir, &Device::Cpu).expect("TTM loads");
    let (n, h, trials) = (52usize, 4usize, 200u32);
    for kind in ["iid", "ar1", "walk", "trend"] {
        let mut rng = Rng(0x7E57 ^ kind.len() as u64);
        let names = ["ttm", "holt", "last"];
        let mut sq = [[0.0f64; 2]; 3];
        let (mut inside, mut banded, mut weight, mut fused_in) = (0u32, 0u32, 0.0f64, 0u32);
        for _ in 0..trials {
            let y = series(kind, n + h, &mut rng);
            let (past, next) = (&y[..n], &y[n..]);
            let tp = TtmPredictor(&ttm);
            let preds: [&dyn Predictor; 3] = [&tp, &HoltPredictor, &LastPredictor];
            for (k, p) in preds.iter().enumerate() {
                let b = p.predict(past, h).expect("a forecast");
                sq[k][0] += (b.point[0] - next[0]).powi(2);
                sq[k][1] += (b.point[h - 1] - next[h - 1]).powi(2);
                if k == 0 {
                    if let (Some(lo), Some(hi)) = (&b.lo, &b.hi) {
                        banded += 1;
                        inside += (next[0] >= lo[0] && next[0] <= hi[0]) as u32;
                    }
                }
            }
            let f = forecast(past, 1, &[&tp], &[]);
            weight += f.weights.iter().find(|w| w.method == "ttm").map_or(0.0, |w| w.weight);
            fused_in += (next[0] >= f.p10[0] && next[0] <= f.p90[0]) as u32;
        }
        let rmse = |k: usize, s: usize| (sq[k][s] / trials as f64).sqrt();
        let line: Vec<String> = names.iter().enumerate().map(|(k, nm)| format!("{nm} {:.4}/{:.4}", rmse(k, 0), rmse(k, 1))).collect();
        let band = if banded > 0 { format!("{:.0}%", 100.0 * inside as f64 / banded as f64) } else { "없음".into() };
        println!(
            "[TTM] {kind:5}: RMSE h1/h4 {} · TTM q10–q90 holds {band} · fused weight on TTM {:.2} · fused p10–p90 holds {:.0}%",
            line.join(" · "),
            weight / trials as f64,
            100.0 * fused_in as f64 / trials as f64
        );
    }
}

fn pinball(y: f64, q_val: f64, tau: f64) -> f64 {
    let diff = y - q_val;
    if diff >= 0.0 { tau * diff } else { (tau - 1.0) * diff }
}

fn bootstrap_ci95(diffs: &[f64], rng: &mut Rng) -> (f64, f64) {
    if diffs.is_empty() {
        return (0.0, 0.0);
    }
    let n = diffs.len();
    let mut means = Vec::with_capacity(400);
    for _ in 0..400 {
        let mut s = 0.0;
        for _ in 0..n {
            let idx = (rng.uniform() * n as f64) as usize % n;
            s += diffs[idx];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    (means[10], means[389])
}

#[test]
#[ignore = "needs TTM_DIR (Granite TTM-R3 real weights evaluation harness)"]
fn ttm_eval_real_weights() {
    use noto_ts::ttm::{PadMode, TtmInputPolicy};
    let dir = PathBuf::from(std::env::var("TTM_DIR").expect("TTM_DIR"));
    let base_ttm = TtmForecaster::load(&dir, &Device::Cpu).expect("TTM loads");
    let mut rng = Rng(0x5EED_4101);
    for pad in [PadMode::Zero, PadMode::Mean, PadMode::Edge] {
        let ttm = base_ttm.clone().with_policy(TtmInputPolicy { pad, scale_floor: None, min_real: 8 });
        let tp = TtmPredictor(&ttm);
        for n_hist in [8usize, 11, 16, 24, 52] {
            for kind in ["iid", "ar1", "walk", "trend"] {
                let trials = 60usize;
                let mut diffs_pb = Vec::with_capacity(trials);
                let (mut mae_with, mut mae_without) = (0.0, 0.0);
                let (mut pb_with, mut pb_without) = (0.0, 0.0);
                let (mut cov_with, mut cov_without) = (0usize, 0usize);
                let (mut w_with, mut w_without, mut ttm_w) = (0.0, 0.0, 0.0);
                let (mut r_lo, mut r_hi) = (Vec::new(), Vec::new());

                for _ in 0..trials {
                    let y = series(kind, n_hist + 16, &mut rng);
                    let (past, next) = (&y[..n_hist], y[n_hist]);
                    let fw = forecast(past, 1, &[&tp], &[]);
                    let fo = forecast(past, 1, &[], &[]);
                    mae_with += (next - fw.point[0]).abs();
                    mae_without += (next - fo.point[0]).abs();
                    let p_w = (pinball(next, fw.p10[0], 0.1) + pinball(next, fw.point[0], 0.5) + pinball(next, fw.p90[0], 0.9)) / 3.0;
                    let p_o = (pinball(next, fo.p10[0], 0.1) + pinball(next, fo.point[0], 0.5) + pinball(next, fo.p90[0], 0.9)) / 3.0;
                    pb_with += p_w;
                    pb_without += p_o;
                    diffs_pb.push(p_o - p_w);
                    cov_with += (next >= fw.p10[0] && next <= fw.p90[0]) as usize;
                    cov_without += (next >= fo.p10[0] && next <= fo.p90[0]) as usize;
                    w_with += (fw.p90[0] - fw.p10[0]).max(0.0);
                    w_without += (fo.p90[0] - fo.p10[0]).max(0.0);
                    ttm_w += fw.weights.iter().find(|w| w.method == "ttm").map_or(0.0, |w| w.weight);
                    if let Ok(b) = tp.predict(past, 1) {
                        if let (Some(lo), Some(hi)) = (&b.lo, &b.hi) {
                            let dlo = (b.point[0] - lo[0]).max(1e-6);
                            let dhi = (hi[0] - b.point[0]).max(1e-6);
                            r_lo.push(((b.point[0] - next) / dlo).max(0.0));
                            r_hi.push(((next - b.point[0]) / dhi).max(0.0));
                        }
                    }
                }
                let (ci_lo, ci_hi) = bootstrap_ci95(&diffs_pb, &mut rng);
                let c_lo = if r_lo.len() >= 20 {
                    r_lo.sort_by(|a, b| a.total_cmp(b));
                    r_lo[((r_lo.len() as f64) * 0.9).min((r_lo.len() - 1) as f64) as usize].clamp(0.5, 3.0)
                } else {
                    1.0
                };
                let c_hi = if r_hi.len() >= 20 {
                    r_hi.sort_by(|a, b| a.total_cmp(b));
                    r_hi[((r_hi.len() as f64) * 0.9).min((r_hi.len() - 1) as f64) as usize].clamp(0.5, 3.0)
                } else {
                    1.0
                };
                let status = if r_lo.len() >= 20 { "calibrated" } else { "uncalibrated" };
                let row = serde_json::json!({
                    "report": "ttm_eval",
                    "fp8": ttm.fp8(),
                    "device": ttm.call_stats().device,
                    "pad": pad.as_str(),
                    "series": kind,
                    "seed_set": "dev",
                    "n_hist": n_hist,
                    "h": 1,
                    "with_ttm": {"mae": mae_with / trials as f64, "pinball": pb_with / trials as f64, "cover_80": cov_with as f64 / trials as f64, "width_80": w_with / trials as f64},
                    "without_ttm": {"mae": mae_without / trials as f64, "pinball": pb_without / trials as f64, "cover_80": cov_without as f64 / trials as f64, "width_80": w_without / trials as f64},
                    "diff_ci95": {"pinball": [ci_lo, ci_hi]},
                    "ttm_weight_mean": ttm_w / trials as f64,
                    "calib": {"c_lo": c_lo, "c_hi": c_hi, "n": r_lo.len(), "status": status},
                });
                println!("{row}");
            }
        }
    }
}
