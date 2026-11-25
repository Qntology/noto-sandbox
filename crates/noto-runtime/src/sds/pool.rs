//! Hierarchical empirical-Bayes partial pooling and heterogeneity diagnostics
//! over faceted SDS v2 fallback chains (Phase 2 – Part 1 · D5, Phase 2 – Part 2 · D4/D5).

use serde::{Deserialize, Serialize};

use super::schema::chain_keys;
use super::scope::ScopeKey;
use super::store::SdsStore;

/// Result of hierarchical partial pooling along a [`ScopeKey`] fallback chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pooled {
    pub value: f64,
    pub n: u64,
    pub n_eff: f64,
    pub rungs: Vec<String>,
}

/// Compute Cochran's `I² = max(0, (Q - (J - 1)) / Q)` across `J` sibling groups
/// `(mean, n, variance)` to flag structurally heterogeneous (`Mixed`) parent scopes.
pub fn heterogeneity_i2(groups: &[(f64, f64, f64)]) -> f64 {
    let valid: Vec<(f64, f64, f64)> = groups.iter().copied().filter(|(_, n, _)| *n >= 2.0).collect();
    let j = valid.len();
    if j < 2 {
        return 0.0;
    }
    let n_tot: f64 = valid.iter().map(|(_, n, _)| *n).sum();
    let grand = valid.iter().map(|(m, n, _)| m * n).sum::<f64>() / n_tot.max(1e-12);
    let df_within: f64 = valid.iter().map(|(_, n, _)| (n - 1.0).max(0.0)).sum();
    let s2_within = valid.iter().map(|(_, n, v)| (n - 1.0).max(0.0) * v.max(0.0)).sum::<f64>() / df_within.max(1.0);
    if s2_within <= 1e-12 {
        let spread = valid.iter().map(|(m, _, _)| (m - grand).abs()).fold(0.0f64, f64::max);
        return if spread > 1e-6 { 1.0 } else { 0.0 };
    }
    let q: f64 = valid.iter().map(|(m, n, _)| n * (m - grand).powi(2) / s2_within).sum();
    if q <= 1e-12 {
        0.0
    } else {
        ((q - (j as f64 - 1.0)) / q).clamp(0.0, 1.0)
    }
}

/// Estimate prior strength `κ = s_within² / τ²` from sibling scopes sharing `parent_prefix`,
/// falling back to `default_kappa` (`track.min_obs().1`) when fewer than 3 siblings exist.
fn sibling_kappa(store: &SdsStore, key: &ScopeKey, parent_key: &str, rung_depth: usize, axis: &str, agreement: bool, default_kappa: f64) -> f64 {
    let siblings = store.sibling_stats(key.track, parent_key, rung_depth, axis, agreement);
    if siblings.len() < 3 {
        return default_kappa.max(1.0);
    }
    let j = siblings.len() as f64;
    let grand = siblings.iter().map(|(m, _, _)| *m).sum::<f64>() / j;
    let var_between = siblings.iter().map(|(m, _, _)| (m - grand).powi(2)).sum::<f64>() / (j - 1.0);
    let mean_within_share = siblings.iter().map(|(_, n, v)| v.max(0.0) / n.max(1.0)).sum::<f64>() / j;
    let mean_s2 = siblings.iter().map(|(_, _, v)| v.max(0.0)).sum::<f64>() / j;
    let tau2 = (var_between - mean_within_share).max(0.0);
    if tau2 <= 1e-12 {
        // Siblings agree closely -> heavy shrinkage toward parent.
        return (default_kappa * 4.0).max(16.0);
    }
    (mean_s2 / tau2).clamp(1.0, default_kappa * 8.0)
}

/// Top-down empirical Bayes partial pooling for a baseline axis along `key`'s schema chain.
/// Display / `nn_status` / forecast-input only (never called by A–H control paths).
pub fn pooled_baseline(store: &SdsStore, key: &ScopeKey, axis: &str) -> Option<Pooled> {
    let chain = chain_keys(key);
    let default_k = key.track.min_obs().1.max(2) as f64;
    let mut est: Option<f64> = None;
    let mut last_n = 0u64;
    let mut n_eff = 0.0f64;
    let mut rungs = Vec::new();
    let mut prev_key = String::new();

    for (idx, k) in chain.iter().enumerate() {
        let Some(st) = store.merged_scope_by_key(key.track.ledger(), k) else {
            prev_key = k.clone();
            continue;
        };
        let Some(w) = st.baseline.get(axis).filter(|w| w.n > 0) else {
            prev_key = k.clone();
            continue;
        };
        let n = w.n as f64;
        match est {
            None => {
                est = Some(w.mean);
                n_eff = n;
            }
            Some(parent_mu) => {
                let kappa = sibling_kappa(store, key, &prev_key, idx, axis, false, default_k);
                let pooled = (n * w.mean + kappa * parent_mu) / (n + kappa);
                est = Some(pooled);
                n_eff = n + kappa;
            }
        }
        last_n = w.n;
        rungs.push(k.clone());
        prev_key = k.clone();
    }

    est.map(|value| Pooled { value, n: last_n, n_eff, rungs })
}

/// Top-down Beta-Binomial partial pooling for an agreement rate axis along `key`'s schema chain.
pub fn pooled_agreement(store: &SdsStore, key: &ScopeKey, axis: &str) -> Option<Pooled> {
    let chain = chain_keys(key);
    let default_k = key.track.min_obs().1.max(2) as f64;
    let mut est: Option<f64> = None;
    let mut last_n = 0u64;
    let mut n_eff = 0.0f64;
    let mut rungs = Vec::new();
    let mut prev_key = String::new();

    for (idx, k) in chain.iter().enumerate() {
        let Some(st) = store.merged_scope_by_key(key.track.ledger(), k) else {
            prev_key = k.clone();
            continue;
        };
        let Some(w) = st.agreement.get(axis).filter(|w| w.n > 0) else {
            prev_key = k.clone();
            continue;
        };
        let n = w.n as f64;
        let k_succ = w.mean * n;
        match est {
            None => {
                est = Some(w.mean.clamp(0.0, 1.0));
                n_eff = n;
            }
            Some(parent_p) => {
                let kappa = sibling_kappa(store, key, &prev_key, idx, axis, true, default_k);
                let p = parent_p.clamp(0.01, 0.99);
                let alpha = kappa * p;
                let beta = kappa * (1.0 - p);
                est = Some(((k_succ + alpha) / (n + alpha + beta)).clamp(0.0, 1.0));
                n_eff = n + kappa;
            }
        }
        last_n = w.n;
        rungs.push(k.clone());
        prev_key = k.clone();
    }

    est.map(|value| Pooled { value, n: last_n, n_eff, rungs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sds::scope::{Facet, Track};

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
    fn pooling_beats_cutoff_on_sparse_cells() {
        let mut rng = Lcg(0xA11CE_5EED);
        let parent_mu = 80.0;
        let tau = 4.0;
        let sigma = 10.0;
        let mut mse_pooled = 0.0;
        let mut mse_cutoff = 0.0;
        let trials = 60usize;
        for _ in 0..trials {
            let mut store = SdsStore::memory();
            // 5 sibling forms under balance/nn:s4;ph=brake_1.0; 4 rich siblings + 1 sparse cell (n=3 < min_obs=8)
            let mut target_true = parent_mu;
            for f in 0..5 {
                let mu_f = parent_mu + tau * rng.normal();
                let n = if f == 0 { 3 } else { 25 };
                if f == 0 {
                    target_true = mu_f;
                }
                let key = ScopeKey::new(
                    Track::Balance,
                    "nn:s4",
                    vec![(Facet::Ph, "brake_1.0".into()), (Facet::Form, format!("f{f}:wheel:f5"))],
                );
                for _ in 0..n {
                    store.record_v2_baseline(&key, "score", mu_f + sigma * rng.normal());
                }
            }
            let target_key = ScopeKey::new(
                Track::Balance,
                "nn:s4",
                vec![(Facet::Ph, "brake_1.0".into()), (Facet::Form, "f0:wheel:f5".into())],
            );
            let p = pooled_baseline(&store, &target_key, "score").unwrap();
            // Step cutoff drops straight to parent mean (balance/nn:s4;ph=brake_1.0) because n=3 < 8.
            let parent_key = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "brake_1.0".into())]);
            let parent_val = store.merged_scope_by_key(Track::Balance.ledger(), &parent_key.to_key_string()).unwrap().baseline["score"].mean;
            mse_pooled += (p.value - target_true).powi(2);
            mse_cutoff += (parent_val - target_true).powi(2);
        }
        mse_pooled /= trials as f64;
        mse_cutoff /= trials as f64;
        assert!(mse_pooled <= mse_cutoff + 1.5, "pooled MSE {mse_pooled:.2} vs cutoff MSE {mse_cutoff:.2}");

        // Large cell: pooled estimate stays within 1 SE of the cell mean.
        let mut store = SdsStore::memory();
        let big = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "brake_1.0".into()), (Facet::Form, "big:wheel:f5".into())]);
        let other = ScopeKey::new(Track::Balance, "nn:s4", vec![(Facet::Ph, "brake_1.0".into()), (Facet::Form, "other:wheel:f5".into())]);
        for _ in 0..200 {
            store.record_v2_baseline(&big, "score", 90.0 + 3.0 * rng.normal());
            store.record_v2_baseline(&other, "score", 70.0 + 3.0 * rng.normal());
        }
        let p = pooled_baseline(&store, &big, "score").unwrap();
        let cell = store.merged_scope_by_key(Track::Balance.ledger(), &big.to_key_string()).unwrap().baseline["score"].clone();
        let se = cell.sd() / (cell.n as f64).sqrt();
        assert!((p.value - cell.mean).abs() <= se * 2.5, "large cell drifted: pooled {} vs mean {} (se {})", p.value, cell.mean, se);
    }

    #[test]
    fn heterogeneity_i2_detects_mixed_phases() {
        let homogeneous = [(80.0, 20.0, 9.0), (80.5, 20.0, 9.0), (79.5, 20.0, 9.0)];
        assert!(heterogeneity_i2(&homogeneous) < 0.25);
        let heterogeneous = [(95.0, 20.0, 9.0), (60.0, 20.0, 9.0), (40.0, 20.0, 9.0)];
        assert!(heterogeneity_i2(&heterogeneous) > 0.85);
    }
}
