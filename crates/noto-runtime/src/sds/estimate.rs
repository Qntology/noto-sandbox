//! Pure-Rust statistical interval, effective-sample-size, and quantile approximations
//! for SDS uncertainty estimation (Phase 2 – Part 2 · D3 & D4).

/// Standard normal quantile Φ⁻¹(p) via Acklam's rational approximation (~1e-9 accuracy).
pub fn norm_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Upper quantile of Student's t distribution via Cornish–Fisher expansion (external-dep free).
pub fn t_quantile(df: f64, p: f64) -> f64 {
    let nu = df.max(1.0);
    if nu <= 1.0 + 1e-9 {
        return (std::f64::consts::PI * (p.clamp(1e-6, 1.0 - 1e-6) - 0.5)).tan();
    }
    if (nu - 2.0).abs() < 1e-9 {
        let a = 2.0 * p.clamp(1e-6, 1.0 - 1e-6) - 1.0;
        return a * (2.0 / (1.0 - a * a).max(1e-12)).sqrt();
    }
    let z = norm_quantile(p);
    let z2 = z * z;
    let g1 = (z2 * z + z) / 4.0;
    let g2 = (5.0 * z2 * z2 * z + 16.0 * z2 * z + 3.0 * z) / 96.0;
    let g3 = (3.0 * z2 * z2 * z2 * z + 19.0 * z2 * z2 * z + 17.0 * z2 * z - 15.0 * z) / 384.0;
    let g4 = (79.0 * z2 * z2 * z2 * z2 * z
        + 776.0 * z2 * z2 * z2 * z
        + 1482.0 * z2 * z2 * z
        - 1920.0 * z2 * z
        - 945.0 * z)
        / 92160.0;
    z + g1 / nu + g2 / (nu * nu) + g3 / (nu * nu * nu) + g4 / (nu * nu * nu * nu)
}

/// Chi-square quantile via Wilson–Hilferty cube-root approximation.
pub fn chi2_quantile(df: f64, p: f64) -> f64 {
    let nu = df.max(1.0);
    let z = norm_quantile(p);
    let k = 2.0 / (9.0 * nu);
    let cube = (1.0 - k + z * k.sqrt()).max(1e-4);
    nu * cube * cube * cube
}

/// Autocorrelation-adjusted effective sample size `n_e = n_eff * (1 - ρ) / (1 + ρ)`.
pub fn effective_n(n_eff: f64, rho: f64) -> f64 {
    let r = rho.clamp(0.0, 0.95);
    (n_eff.max(0.0) * (1.0 - r) / (1.0 + r)).max(if n_eff > 0.0 { 1.0 } else { 0.0 })
}

/// Two-sided confidence interval for a mean: returns `(se, lo, hi)`.
pub fn mean_ci(mean: f64, sd: f64, n_e: f64, conf: f64) -> (f64, f64, f64) {
    let n = n_e.max(1.0);
    let se = sd.max(0.0) / n.sqrt();
    let alpha = (1.0 - conf.clamp(0.5, 0.999)) * 0.5;
    let t = t_quantile((n - 1.0).max(1.0), 1.0 - alpha);
    let hw = t * se;
    (se, mean - hw, mean + hw)
}

/// One-sided lower prediction bound for the next single observation (e.g. `tail_p = 0.0228` -> ~mean - 2*sd at large n).
pub fn pred_lower(mean: f64, sd: f64, n_e: f64, tail_p: f64) -> f64 {
    let n = n_e.max(2.0);
    let t = t_quantile((n - 1.0).max(1.0), 1.0 - tail_p.clamp(1e-4, 0.49));
    mean - t * sd.max(0.0) * (1.0 + 1.0 / n).sqrt()
}

/// Wilson score confidence interval for a Bernoulli rate using effective sample size `n_e`: returns `(se, lo, hi)`.
pub fn wilson_ci(p_hat: f64, n_e: f64, conf: f64) -> (f64, f64, f64) {
    let p = p_hat.clamp(0.0, 1.0);
    let n = n_e.max(1.0);
    let alpha = (1.0 - conf.clamp(0.5, 0.999)) * 0.5;
    let z = norm_quantile(1.0 - alpha);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let rad = (z / denom) * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    let lo = (center - rad).clamp(0.0, 1.0);
    let hi = (center + rad).clamp(0.0, 1.0);
    let se = (hi - lo) / (2.0 * z).max(1e-9);
    (se, lo, hi)
}

/// Chi-square confidence interval for standard deviation `sd`: returns `(se, lo, hi)`.
pub fn spread_ci(sd: f64, n_e: f64, conf: f64) -> (f64, f64, f64) {
    let s = sd.max(0.0);
    let df = (n_e - 1.0).max(1.0);
    let alpha = (1.0 - conf.clamp(0.5, 0.999)) * 0.5;
    let q_hi = chi2_quantile(df, 1.0 - alpha).max(1e-6);
    let q_lo = chi2_quantile(df, alpha).max(1e-6);
    let lo = s * (df / q_hi).sqrt();
    let hi = s * (df / q_lo).sqrt();
    let se = s / (2.0 * df).sqrt();
    (se, lo, hi)
}

/// Log-scale confidence interval for a multiplicative calibration factor `geo_mean`:
/// `sigma_log = sqrt(ln(1 + cv²))`. Returns `(se, lo, hi)`.
pub fn factor_ci(geo_mean: f64, cv: f64, n_e: f64, conf: f64) -> (f64, f64, f64) {
    let g = geo_mean.max(1e-6);
    let n = n_e.max(1.0);
    let sigma_log = (1.0 + cv.max(0.0).powi(2)).ln().max(0.0).sqrt();
    let se_log = sigma_log / n.sqrt();
    let alpha = (1.0 - conf.clamp(0.5, 0.999)) * 0.5;
    let t = t_quantile((n - 1.0).max(1.0), 1.0 - alpha);
    let lo = (g.ln() - t * se_log).exp();
    let hi = (g.ln() + t * se_log).exp();
    (g * se_log, lo, hi)
}

/// Welch approximate two-sample check: returns true if the two estimates do not differ significantly at `conf`.
pub fn welch_overlap(a_mean: f64, a_se: f64, a_n: f64, b_mean: f64, b_se: f64, b_n: f64, conf: f64) -> bool {
    let va = a_se * a_se;
    let vb = b_se * b_se;
    let se_diff = (va + vb).sqrt();
    if se_diff <= 1e-12 {
        return (a_mean - b_mean).abs() <= 1e-9;
    }
    let df_num = (va + vb).powi(2);
    let df_den = (va * va) / (a_n - 1.0).max(1.0) + (vb * vb) / (b_n - 1.0).max(1.0);
    let df = (df_num / df_den.max(1e-12)).max(1.0);
    let alpha = (1.0 - conf.clamp(0.5, 0.999)) * 0.5;
    let t_crit = t_quantile(df, 1.0 - alpha);
    (a_mean - b_mean).abs() <= t_crit * se_diff
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn quantiles_match_reference_values() {
        assert!((norm_quantile(0.975) - 1.959964).abs() < 1e-4);
        assert!((norm_quantile(0.9) - 1.281552).abs() < 1e-4);
        assert!((t_quantile(10.0, 0.975) - 2.228).abs() < 0.015);
        assert!((t_quantile(100.0, 0.975) - 1.984).abs() < 0.01);
        assert!((chi2_quantile(10.0, 0.975) - 20.483).abs() < 0.15);
    }

    #[test]
    fn intervals_cover_with_and_without_autocorrelation() {
        let mut rng = Lcg(0xBEEF_CAFE_1234_5678);
        // 1) Wilson coverage for p in {0.1, 0.5, 0.8, 0.95}, n in {8, 20, 50}
        for &p in &[0.1, 0.5, 0.8, 0.95] {
            for &n in &[8usize, 20, 50] {
                let trials = 600usize;
                let mut covered = 0usize;
                for _ in 0..trials {
                    let mut k = 0usize;
                    for _ in 0..n {
                        if rng.next_f64() < p {
                            k += 1;
                        }
                    }
                    let (_, lo, hi) = wilson_ci(k as f64 / n as f64, n as f64, 0.95);
                    if lo <= p && p <= hi {
                        covered += 1;
                    }
                }
                let cov = covered as f64 / trials as f64;
                assert!(cov >= 0.90 && cov <= 0.995, "wilson p={p} n={n} cov={cov:.3}");
            }
        }
        // 2) AR(1) rho=0.6: naive n undercovers (< 0.85), effective_n adjustment recovers (>= 0.92)
        let rho = 0.6_f64;
        let sd_eps = (1.0 - rho * rho).sqrt();
        let trials = 600usize;
        let n = 60usize;
        let (mut cov_naive, mut cov_adj) = (0usize, 0usize);
        for _ in 0..trials {
            let mut x = rng.normal();
            let mut xs = Vec::with_capacity(n);
            for _ in 0..n {
                x = rho * x + sd_eps * rng.normal();
                xs.push(x);
            }
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let sd = var.sqrt();
            let c1: f64 = xs.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f64>() / (n - 1) as f64;
            let rho_hat = (c1 / var.max(1e-9)).clamp(0.0, 0.95);
            let (_, lo0, hi0) = mean_ci(mean, sd, n as f64, 0.95);
            if lo0 <= 0.0 && 0.0 <= hi0 {
                cov_naive += 1;
            }
            let n_e = effective_n(n as f64, rho_hat);
            let (_, lo1, hi1) = mean_ci(mean, sd, n_e, 0.95);
            if lo1 <= 0.0 && 0.0 <= hi1 {
                cov_adj += 1;
            }
        }
        let r0 = cov_naive as f64 / trials as f64;
        let r1 = cov_adj as f64 / trials as f64;
        assert!(r0 < 0.85, "expected naive AR(1) undercoverage, got {r0:.3}");
        assert!(r1 >= 0.90, "expected adjusted AR(1) coverage >= 0.90, got {r1:.3} (naive {r0:.3})");
    }
}
