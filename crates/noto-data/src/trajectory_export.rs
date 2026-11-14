//! A robot's motion as a training episode: per sample the time, the observation it pairs with and the joint command
//! that makes the robot's servo follow the recorded motion, written as JSON lines in a format of this repository's own.
//!
//! **Latency.** A command reaches the joint after the link's delay Δ, and the servo then follows it by its own law. The
//! label that reproduces the motion inverts both: evaluate the target at t + Δ and undo the servo —
//! - a computed-torque servo (`noto_core::dynamics`: the reference's acceleration and velocity fed forward, PD on the
//!   error) follows its reference without lag, so the label is the target itself at t + Δ;
//! - a PD servo with no feedforward, critically damped at ω_n, follows q̈ = ω_n²(u − q) − 2ω_n q̇, so
//!   u = q + (2/ω_n)·q̇ + q̈/ω_n²;
//! - a first-order servo with time constant τ follows τ·q̇ + q = u, so u = q + τ·q̇ (the phase-lead label).
//!
//! **Augmentation.** One rigid transform (a turn about the vertical and a shift) applied on the left of the item's and
//! the wrists' poses keeps every wrist's pose in the item's frame; a time scale resamples the motion (its velocities by
//! 1/α, its accelerations by 1/α²: a faster copy must be checked against the actuators again).

use std::io::Write;

use anyhow::Result;
use noto_core::act::retarget::{Demo, Rigid};
use serde::{Deserialize, Serialize};

/// How the target robot's joint follows its command.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServoModel {
    ComputedTorque { bandwidth_hz: f32 },
    Pd { bandwidth_hz: f32 },
    FirstOrder { tau_s: f32 },
}

/// The rule that turns the target motion into labels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LeadRule {
    /// The target as it is (no delay, no servo inverse).
    None,
    /// q(t + Δ) + τ·q̇(t + Δ), whatever the servo.
    PhaseLead { tau_s: f32 },
    /// The inverse of the servo model at t + Δ.
    Exact(ServoModel),
}

/// q, q̇, q̈ of a sampled signal at time `t` (linear between samples, central differences, ends held).
fn at(q: &[f32], dt: f32, t: f32) -> (f32, f32, f32) {
    let n = q.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let val = |k: i64| q[k.clamp(0, n as i64 - 1) as usize];
    let x = (t / dt).clamp(0.0, (n - 1) as f32);
    let k = x.floor() as i64;
    let u = x - k as f32;
    let lerp = |f: &dyn Fn(i64) -> f32| f(k) * (1.0 - u) + f(k + 1) * u;
    let v = lerp(&|i| val(i));
    let d = lerp(&|i| (val(i + 1) - val(i - 1)) / (2.0 * dt));
    let a = lerp(&|i| (val(i + 1) - 2.0 * val(i) + val(i - 1)) / (dt * dt));
    (v, d, a)
}

/// The label sequence of one joint's target `q` (sampled every `dt`), for a link delay `delay_s`.
pub fn latency_labels(q: &[f32], dt: f32, delay_s: f32, rule: LeadRule) -> Vec<f32> {
    (0..q.len())
        .map(|i| {
            let t = i as f32 * dt;
            match rule {
                LeadRule::None => q[i],
                LeadRule::PhaseLead { tau_s } => {
                    let (v, d, _) = at(q, dt, t + delay_s);
                    v + tau_s * d
                }
                LeadRule::Exact(ServoModel::ComputedTorque { .. }) => at(q, dt, t + delay_s).0,
                LeadRule::Exact(ServoModel::Pd { bandwidth_hz }) => {
                    let wn = std::f32::consts::TAU * bandwidth_hz;
                    let (v, d, a) = at(q, dt, t + delay_s);
                    v + 2.0 / wn * d + a / (wn * wn)
                }
                LeadRule::Exact(ServoModel::FirstOrder { tau_s }) => {
                    let (v, d, _) = at(q, dt, t + delay_s);
                    v + tau_s * d
                }
            }
        })
        .collect()
}

/// One joint of `servo` driven by the labels `u` (sampled every `dt`) through a link delay `delay_s`, from rest at `u[0]`;
/// the joint angle at each sample. A computed-torque servo feeds the delayed command's velocity and acceleration forward.
pub fn simulate_servo(u: &[f32], dt: f32, delay_s: f32, servo: ServoModel) -> Vec<f32> {
    let h = 1e-4f32;
    let steps = (dt / h).round().max(1.0) as usize;
    let (mut q, mut qd) = (u.first().copied().unwrap_or(0.0), 0.0f32);
    let mut out = Vec::with_capacity(u.len());
    for i in 0..u.len() {
        out.push(q);
        for s in 0..steps {
            let t = i as f32 * dt + s as f32 * h - delay_s;
            let (c, cd, cdd) = at(u, dt, t.max(0.0));
            let qdd = match servo {
                ServoModel::ComputedTorque { bandwidth_hz } => {
                    let wn = std::f32::consts::TAU * bandwidth_hz;
                    cdd + wn * wn * (c - q) + 2.0 * wn * (cd - qd)
                }
                ServoModel::Pd { bandwidth_hz } => {
                    let wn = std::f32::consts::TAU * bandwidth_hz;
                    wn * wn * (c - q) - 2.0 * wn * qd
                }
                ServoModel::FirstOrder { tau_s } => {
                    // τ·q̇ + q = c, integrated on the velocity directly.
                    qd = (c - q) / tau_s.max(1e-6);
                    0.0
                }
            };
            qd += qdd * h;
            q += qd * h;
        }
    }
    out
}

/// A rigid transform on the left of every pose and a time scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Augment {
    /// Turn about the vertical (the demonstration frame's z), rad.
    pub yaw_rad: f64,
    pub shift_m: [f64; 3],
    /// α > 1 plays the motion slower: the copy's sample k is the original at k/α.
    pub time_scale: f64,
}

impl Augment {
    pub fn transform(&self) -> Rigid {
        let (c, s) = (self.yaw_rad.cos(), self.yaw_rad.sin());
        Rigid { r: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], t: self.shift_m }
    }
}

/// `demo` moved by `aug`: every item and wrist pose left-multiplied by the transform, the frames resampled at the
/// same spacing on the scaled clock (the nearest earlier original frame's rotation, the translation interpolated).
/// The contacts are in the item's frame and do not change.
pub fn augment_demo(demo: &Demo, aug: &Augment) -> Demo {
    let tf = aug.transform();
    let n = demo.obj.len();
    let frames = ((n.max(1) - 1) as f64 * aug.time_scale.max(1e-6)).round() as usize + 1;
    let pick = |poses: &dyn Fn(usize) -> Option<Rigid>, k: usize| -> Option<Rigid> {
        let x = (k as f64 / aug.time_scale.max(1e-6)).min((n.max(1) - 1) as f64);
        let (i, u) = (x.floor() as usize, x - x.floor());
        let a = poses(i)?;
        let b = poses((i + 1).min(n - 1)).unwrap_or(a);
        Some(Rigid { r: a.r, t: [0, 1, 2].map(|j| a.t[j] * (1.0 - u) + b.t[j] * u) })
    };
    let obj = (0..frames).map(|k| tf.then(&pick(&|i| Some(demo.obj[i]), k).unwrap_or(Rigid::ID))).collect();
    let wrist = (0..frames).map(|k| [0, 1].map(|s| pick(&|i| demo.wrist[i][s], k).map(|w| tf.then(&w)))).collect();
    Demo { obj, wrist, ..demo.clone() }
}

/// One sample of an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlignedStep {
    pub t_s: f32,
    /// The observation this sample pairs with (a frame file, a keyframe id).
    pub obs: String,
    /// The command per joint (`AlignedEpisode::joints` order).
    pub action: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlignedEpisode {
    pub task: String,
    pub embodiment: String,
    pub joints: Vec<String>,
    pub dt_s: f32,
    pub delay_s: f32,
    pub rule: LeadRule,
    pub steps: Vec<AlignedStep>,
}

impl AlignedEpisode {
    /// Pairs each observation with the labels of the joint targets `q[joint][sample]` under `rule`.
    pub fn build(task: &str, embodiment: &str, joints: Vec<String>, q: &[Vec<f32>], obs: &[String], dt_s: f32, delay_s: f32, rule: LeadRule) -> AlignedEpisode {
        let labels: Vec<Vec<f32>> = q.iter().map(|j| latency_labels(j, dt_s, delay_s, rule)).collect();
        let n = obs.len().min(labels.iter().map(Vec::len).min().unwrap_or(0));
        let steps = (0..n).map(|i| AlignedStep { t_s: i as f32 * dt_s, obs: obs[i].clone(), action: labels.iter().map(|l| l[i]).collect() }).collect();
        AlignedEpisode { task: task.into(), embodiment: embodiment.into(), joints, dt_s, delay_s, rule, steps }
    }

    /// A header line (everything but the steps), then one line per step.
    pub fn write_jsonl(&self, out: &mut dyn Write) -> Result<()> {
        let head = serde_json::json!({"task": self.task, "embodiment": self.embodiment, "joints": self.joints, "dt_s": self.dt_s, "delay_s": self.delay_s, "rule": self.rule, "steps": self.steps.len()});
        writeln!(out, "{head}")?;
        for s in &self.steps {
            writeln!(out, "{}", serde_json::to_string(s)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::act::retarget::{object_relative, DemoContact};
    use noto_core::sim::ShapeKind;

    /// A minimum-jerk reach of 1 rad in 0.6 s, held 0.3 s, at 50 Hz.
    fn reach() -> Vec<f32> {
        (0..46)
            .map(|i| {
                let u = (i as f32 * 0.02 / 0.6).min(1.0);
                10.0 * u.powi(3) - 15.0 * u.powi(4) + 6.0 * u.powi(5)
            })
            .collect()
    }

    fn rms(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
    }

    /// Tracking error of each servo under each label rule (printed with --nocapture): the phase-lead label is the exact
    /// inverse of a first-order servo only.
    #[test]
    fn the_phase_lead_label_fits_a_first_order_servo_and_not_the_engines() {
        let q = reach();
        let (dt, delay) = (0.02f32, 0.02f32);
        let servos = [ServoModel::ComputedTorque { bandwidth_hz: 4.0 }, ServoModel::Pd { bandwidth_hz: 4.0 }, ServoModel::FirstOrder { tau_s: 0.08 }];
        let mut table = Vec::new();
        for servo in servos {
            let err = |rule| rms(&simulate_servo(&latency_labels(&q, dt, delay, rule), dt, delay, servo), &q);
            let row = (err(LeadRule::None), err(LeadRule::PhaseLead { tau_s: 0.08 }), err(LeadRule::Exact(servo)));
            println!("{servo:?}: none {:.4} · phase lead τ=0.08 {:.4} · exact {:.4} rad", row.0, row.1, row.2);
            table.push(row);
        }
        let (ct, pd, fo) = (table[0], table[1], table[2]);
        assert!(fo.2 < 0.25 * fo.0 && (fo.1 - fo.2).abs() < 1e-3, "for a first-order servo the phase lead is the exact label: {fo:?}");
        assert!(pd.2 < pd.1 && pd.2 < 0.25 * pd.0, "a PD servo wants its second-order inverse: {pd:?}");
        assert!(ct.2 < ct.1, "the engine's computed-torque servo needs no lead, only the delay: {ct:?}");
    }

    #[test]
    fn an_augmented_demo_keeps_each_wrist_on_the_item_and_its_length_scales() {
        let obj: Vec<Rigid> = (0..10).map(|i| Rigid { r: Rigid::ID.r, t: [0.5, 0.0, 0.2 + 0.05 * i as f64] }).collect();
        let wrist: Vec<[Option<Rigid>; 2]> = obj.iter().map(|o| [Some(Rigid { r: Rigid::ID.r, t: [o.t[0] - 0.1, -0.15, o.t[2]] }), None]).collect();
        let demo = Demo { dt_s: 0.1, obj, wrist, contacts: [vec![DemoContact { finger: 0, point: [0.0; 3], force: [0.0, 1.0, 0.0] }], vec![]], size_m: [0.3, 0.2, 0.2], shape: ShapeKind::Box, mass_kg: 1.0, mu: 0.5, hand_k: 1.0 };
        let aug = Augment { yaw_rad: 0.7, shift_m: [0.3, -0.2, 0.0], time_scale: 1.5 };
        let out = augment_demo(&demo, &aug);
        assert_eq!(out.obj.len(), 15);
        for (o, w) in out.obj.iter().zip(&out.wrist) {
            let rel = object_relative(o, &w[0].unwrap());
            assert!((rel.t[0] + 0.1).abs() < 1e-9 && (rel.t[1] + 0.15).abs() < 1e-9 && rel.t[2].abs() < 1e-9, "{rel:?}");
            assert!(w[1].is_none());
        }
        // The episode file: a header and a line per step.
        let ep = AlignedEpisode::build("lift", "compact", vec!["shoulder".into()], &[reach()], &(0..46).map(|i| format!("kf{i}")).collect::<Vec<_>>(), 0.02, 0.02, LeadRule::Exact(ServoModel::ComputedTorque { bandwidth_hz: 4.0 }));
        let mut buf = Vec::new();
        ep.write_jsonl(&mut buf).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap().lines().count(), 47);
    }
}
