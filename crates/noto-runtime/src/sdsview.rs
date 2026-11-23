//! Server-side SDS explorer (`sds_tree`, `sds_query`, `sds_watch`) and drift-watch ring
//! (Phase 5 – Part 2 · D1).
//!
//! Strictly OUTSIDE `ENGINE_FILES` and `DECISION_FILES`: read-only inspection over all ledgers
//! with prefix filtering BEFORE scope merging, maturity calculation, reference folding, and feedback badges.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use noto_sds::catalog;
use noto_sds::{Scope, SdsStore, Track};
use serde::Serialize;
use serde_json::{json, Value};

const WATCH_EVERY_DEFAULT: u64 = 10;
const WATCH_CAP_DEFAULT: usize = 600;
const WATCH_MAX_PINNED: usize = 16;
const QUERY_MAX_LIMIT: usize = 500;

#[derive(Clone, Debug, Serialize)]
pub struct WatchPoint {
    pub tick: u64,
    pub n: u64,
    pub mean: f64,
    pub recent: f64,
    pub drift_z: f64,
}

#[derive(Clone, Debug)]
pub struct SdsWatch {
    pub every_ticks: u64,
    pub cap: usize,
    pub axes: BTreeMap<(String, String), VecDeque<WatchPoint>>,
    pub pinned: BTreeSet<(String, String)>,
    last_sample_tick: u64,
}

impl Default for SdsWatch {
    fn default() -> Self {
        Self {
            every_ticks: WATCH_EVERY_DEFAULT,
            cap: WATCH_CAP_DEFAULT,
            axes: BTreeMap::new(),
            pinned: BTreeSet::new(),
            last_sample_tick: 0,
        }
    }
}

impl SdsWatch {
    pub fn pin(&mut self, scope: String, axis: String) -> bool {
        if self.pinned.contains(&(scope.clone(), axis.clone())) {
            return true;
        }
        if self.pinned.len() >= WATCH_MAX_PINNED {
            return false;
        }
        self.pinned.insert((scope, axis));
        true
    }

    pub fn unpin(&mut self, scope: &str, axis: &str) {
        self.pinned.remove(&(scope.to_string(), axis.to_string()));
    }

    pub fn sample_if_due(&mut self, tick: u64, sds: &SdsStore) {
        if tick == 0 || tick.saturating_sub(self.last_sample_tick) < self.every_ticks {
            return;
        }
        self.last_sample_tick = tick;
        let targets: Vec<(String, String)> = self.pinned.iter().cloned().collect();
        for (sc_key, axis) in targets {
            let Some(tr) = track_of_scope_key(&sc_key) else { continue };
            let Some(st) = sds.merged_scope_by_key(tr.ledger(), &sc_key) else { continue };
            let Some(m) = st.baseline.get(&axis) else { continue };
            let pt = WatchPoint {
                tick,
                n: m.n,
                mean: r4(m.mean),
                recent: r4(m.recent_mean()),
                drift_z: r4(m.drift_z()),
            };
            let q = self.axes.entry((sc_key, axis)).or_default();
            q.push_back(pt);
            while q.len() > self.cap {
                q.pop_front();
            }
        }
    }

    /// Snapshot of watched axes nearest to `tick` (for `test_context` time-cursor sync).
    pub fn at_tick(&self, tick: u64) -> Vec<Value> {
        self.axes
            .iter()
            .filter_map(|((sc, ax), pts)| {
                let nearest = pts.iter().min_by_key(|p| p.tick.abs_diff(tick))?;
                Some(json!({
                    "scope": sc, "axis": ax,
                    "tick": nearest.tick, "n": nearest.n, "mean": nearest.mean,
                    "recent": nearest.recent, "drift_z": nearest.drift_z,
                }))
            })
            .collect()
    }
}

fn r4(x: f64) -> f64 {
    if x.is_finite() { (x * 1e4).round() / 1e4 } else { 0.0 }
}

fn parse_track(name: &str) -> Option<Track> {
    Track::ALL.into_iter().find(|t| t.as_str() == name)
}

fn track_of_scope_key(key: &str) -> Option<Track> {
    let head = key.split('|').next()?;
    parse_track(head)
}

fn scope_level_and_need(track: Track, key: &str) -> (&'static str, u64) {
    let parts: Vec<&str> = key.split('|').collect();
    let (sec, pri, glo) = track.min_obs();
    let primary = parts.get(1).copied().unwrap_or("");
    let secondary = parts.get(2).copied().unwrap_or("");
    if !secondary.is_empty() {
        ("secondary", sec)
    } else if !primary.is_empty() {
        ("primary", pri)
    } else {
        ("global", glo)
    }
}

fn scope_from_key(track: Track, key: &str) -> Scope {
    let mut parts = key.splitn(3, '|');
    let _ = parts.next();
    let p = parts.next().unwrap_or("");
    let s = parts.next().unwrap_or("");
    Scope::new(track, p, s)
}

fn feedback_paths_for(track: Track, key: &str, axis: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    let primary = key.split('|').nth(1).unwrap_or("");
    let prov = catalog::classify(track, primary, axis);
    if !prov.control_allowed() {
        return out;
    }
    match (track, axis) {
        (Track::Vision, "score" | "touch") => out.push("A"),
        (Track::Vision, "score_all") => out.push("B"),
        (Track::Forecast, a) if a.starts_with("fusion_mse:") || a.contains(":res2:") => out.push("C"),
        (Track::Locate, "score") => out.push("D"),
        (Track::Depth, "scale") => out.push("E"),
        (Track::Balance, "ff" | "ff_ratio") => out.push("F"),
        (Track::Memory, "drift_var_per_m2") => out.push("G"),
        (Track::Memory, "scatter_var_m2") => out.push("H"),
        _ => {}
    }
    out
}

fn is_ref_aux_axis(axis: &str) -> bool {
    axis.ends_with(":dev")
        || axis.ends_with(":absdev")
        || axis.ends_with(":dev2")
        || axis.ends_with(":ref")
        || axis.ends_with(":tol")
        || axis.ends_with(":in_band")
}

/// Tree browser: one level at a time (`None` -> tracks + virtual `nn_scenarios`; `"vision"` -> primary; `"vision|tree"` -> secondary).
pub fn sds_tree(sds: &SdsStore, parent: Option<&str>, if_rev: Option<u64>) -> Value {
    let rev = sds.rev();
    if if_rev == Some(rev) {
        return json!({"unchanged": true, "rev": rev});
    }
    let p = parent.unwrap_or("").trim();
    if p.is_empty() {
        let mut nodes = Vec::new();
        for track in Track::ALL {
            let keys = sds.scope_keys_for_track(track);
            let mut n_max = 0u64;
            let mut immature = 0usize;
            let mut max_dz = 0.0f64;
            let mut fb_set: BTreeSet<&'static str> = BTreeSet::new();
            for k in &keys {
                let (_, need) = scope_level_and_need(track, k);
                if let Some(st) = sds.merged_scope_by_key(track.ledger(), k) {
                    for (ax, m) in &st.baseline {
                        if is_ref_aux_axis(ax) {
                            continue;
                        }
                        n_max = n_max.max(m.n);
                        if m.n < need {
                            immature += 1;
                        }
                        max_dz = max_dz.max(m.drift_z().abs());
                        for pth in feedback_paths_for(track, k, ax) {
                            fb_set.insert(pth);
                        }
                    }
                }
            }
            nodes.push(json!({
                "id": track.as_str(),
                "label": track.as_str(),
                "kind": "track",
                "scopes": keys.len(),
                "n_max": n_max,
                "immature": immature,
                "max_drift_z": r4(max_dz),
                "feedback": fb_set.into_iter().collect::<Vec<_>>(),
                "engine_revs": sds.engine_revs(track.ledger()),
                "has_children": !keys.is_empty(),
            }));
        }
        // Virtual node grouping `balance|nn:*` scenarios.
        let nn_keys: Vec<String> = sds
            .scope_keys_for_track(Track::Balance)
            .into_iter()
            .filter(|k| k.starts_with("balance|nn:"))
            .collect();
        nodes.push(json!({
            "id": "nn_scenarios",
            "label": "시나리오 (N:N)",
            "kind": "virtual",
            "scopes": nn_keys.len(),
            "n_max": nn_keys.len() as u64,
            "immature": 0,
            "max_drift_z": 0.0,
            "feedback": [],
            "engine_revs": sds.engine_revs(Track::Balance.ledger()),
            "has_children": !nn_keys.is_empty(),
        }));
        return json!({"rev": rev, "parent": "", "nodes": nodes});
    }

    if p == "nn_scenarios" {
        let mut by_scn: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for k in sds.scope_keys_for_track(Track::Balance) {
            if let Some(rest) = k.strip_prefix("balance|nn:") {
                let scn = rest.split('|').next().unwrap_or("").to_string();
                if !scn.is_empty() {
                    by_scn.entry(scn).or_default().push(k);
                }
            }
        }
        let nodes: Vec<Value> = by_scn
            .into_iter()
            .map(|(scn, ks)| {
                json!({
                    "id": format!("balance|nn:{scn}"),
                    "label": format!("nn:{scn}"),
                    "kind": "primary",
                    "scopes": ks.len(),
                    "has_children": true,
                })
            })
            .collect();
        return json!({"rev": rev, "parent": p, "nodes": nodes});
    }

    // Track level or primary level.
    let parts: Vec<&str> = p.split('|').collect();
    let Some(track) = parse_track(parts[0]) else {
        return json!({"rev": rev, "parent": p, "nodes": []});
    };
    let keys = sds.scope_keys_for_track(track);
    if parts.len() == 1 {
        let mut primaries: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for k in keys {
            let pri = k.split('|').nth(1).unwrap_or("").to_string();
            primaries.entry(pri).or_default().push(k);
        }
        let nodes: Vec<Value> = primaries
            .into_iter()
            .map(|(pri, ks)| {
                let mut n_max = 0u64;
                let mut max_dz = 0.0f64;
                for k in &ks {
                    if let Some(st) = sds.merged_scope_by_key(track.ledger(), k) {
                        for m in st.baseline.values() {
                            n_max = n_max.max(m.n);
                            max_dz = max_dz.max(m.drift_z().abs());
                        }
                    }
                }
                let node_id = format!("{}|{}", track.as_str(), pri);
                json!({
                    "id": node_id,
                    "label": if pri.is_empty() { "(전역)" } else { &pri },
                    "kind": "primary",
                    "scopes": ks.len(),
                    "n_max": n_max,
                    "max_drift_z": r4(max_dz),
                    "has_children": ks.iter().any(|k| !k.split('|').nth(2).unwrap_or("").is_empty()),
                })
            })
            .collect();
        return json!({"rev": rev, "parent": p, "nodes": nodes});
    }

    let prefix = format!("{}|{}|", parts[0], parts[1]);
    let nodes: Vec<Value> = keys
        .into_iter()
        .filter(|k| k.starts_with(&prefix))
        .map(|k| {
            let sec = k.split('|').nth(2).unwrap_or("").to_string();
            let (_, need) = scope_level_and_need(track, &k);
            let (mut n_max, mut max_dz) = (0u64, 0.0f64);
            if let Some(st) = sds.merged_scope_by_key(track.ledger(), &k) {
                for m in st.baseline.values() {
                    n_max = n_max.max(m.n);
                    max_dz = max_dz.max(m.drift_z().abs());
                }
            }
            json!({
                "id": k,
                "label": if sec.is_empty() { "(기본)" } else { &sec },
                "kind": "secondary",
                "n_max": n_max,
                "need": need,
                "max_drift_z": r4(max_dz),
                "has_children": false,
            })
        })
        .collect();
    json!({"rev": rev, "parent": p, "nodes": nodes})
}

/// Filtered, paginated server-side SDS query: filters scope keys by `track`/`prefix` BEFORE merging!
pub fn sds_query(sds: &SdsStore, args: &Value) -> Value {
    let rev = sds.rev();
    if args.get("if_rev").and_then(Value::as_u64) == Some(rev) {
        return json!({"unchanged": true, "rev": rev});
    }
    let track_filter = args.get("track").and_then(Value::as_str).filter(|s| !s.is_empty());
    let prefix = args.get("prefix").and_then(Value::as_str).unwrap_or("");
    let text = args.get("text").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
    let min_n = args.get("min_n").and_then(Value::as_u64).unwrap_or(0);
    let max_mat = args.get("max_maturity").and_then(Value::as_f64);
    let drift_min = args.get("drift_min").and_then(Value::as_f64);
    let fb_only = args.get("feedback_only").and_then(Value::as_bool).unwrap_or(false);
    let kinds: Vec<String> = args
        .get("kinds")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();
    let sort = args.get("sort").and_then(Value::as_str).unwrap_or("n_desc");
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(100).clamp(1, QUERY_MAX_LIMIT as u64) as usize;
    let cursor = args.get("cursor").and_then(Value::as_u64).unwrap_or(0) as usize;

    let (track_filter, prefix) = if prefix == "nn_scenarios" {
        (Some("balance"), "balance|nn:")
    } else {
        (track_filter, prefix)
    };
    let tracks: Vec<Track> = match track_filter {
        Some(t) => parse_track(t).into_iter().collect(),
        None => Track::ALL.to_vec(),
    };

    let mut merged_scopes = 0usize;
    let mut rows: Vec<Value> = Vec::new();

    for track in tracks {
        let revs = sds.engine_revs(track.ledger());
        // Filter keys BEFORE merging!
        let keys: Vec<String> = sds
            .scope_keys_for_track(track)
            .into_iter()
            .filter(|k| prefix.is_empty() || k.starts_with(prefix))
            .collect();
        for key in keys {
            let Some(st) = sds.merged_scope_by_key(track.ledger(), &key) else { continue };
            merged_scopes += 1;
            let (level, need) = scope_level_and_need(track, &key);
            let sc = scope_from_key(track, &key);

            // Detect reference base axes (`axis` that has `axis:ref` or `axis:dev`).
            let ref_bases: BTreeSet<String> = st
                .baseline
                .keys()
                .filter_map(|ax| ax.strip_suffix(":ref").or_else(|| ax.strip_suffix(":dev")).map(String::from))
                .collect();

            // Detect reliability bin groups (`p_success:bin*`).
            let mut rel_bins: Vec<(String, f64, u64)> = Vec::new();
            for (ax, a) in &st.agreement {
                if ax.starts_with("p_success:bin") {
                    rel_bins.push((ax.clone(), a.mean, a.n));
                }
            }
            if !rel_bins.is_empty() {
                let total_n: u64 = rel_bins.iter().map(|b| b.2).sum();
                let row_kind = "reliability";
                let mat = (total_n as f64 / need.max(1) as f64).min(1.0);
                if (kinds.is_empty() || kinds.iter().any(|k| k == row_kind))
                    && total_n >= min_n
                    && max_mat.map_or(true, |mm| mat <= mm)
                    && drift_min.is_none()
                    && !fb_only
                    && (text.is_empty() || key.to_ascii_lowercase().contains(&text) || "p_success".contains(&text))
                {
                    rows.push(json!({
                        "scope": key, "axis": "p_success", "kind": row_kind, "level": level,
                        "n": total_n, "need": need, "maturity": r4(mat),
                        "bins": rel_bins.iter().map(|(b, r, n)| json!({"bin": b, "rate": r4(*r), "n": n})).collect::<Vec<_>>(),
                        "feedback": feedback_paths_for(track, &key, "p_success"),
                        "engine_revs": revs,
                    }));
                }
            }

            for (ax, m) in &st.baseline {
                if is_ref_aux_axis(ax) {
                    continue;
                }
                let is_ref = ref_bases.contains(ax);
                let row_kind = if is_ref { "reference" } else { "baseline" };
                if !kinds.is_empty() && !kinds.iter().any(|k| k == row_kind) {
                    continue;
                }
                if m.n < min_n {
                    continue;
                }
                let mat = m.n as f64 / need.max(1) as f64;
                if let Some(mm) = max_mat {
                    if mat > mm {
                        continue;
                    }
                }
                let dz = m.drift_z();
                if let Some(dm) = drift_min {
                    if dz.abs() < dm {
                        continue;
                    }
                }
                let fb = feedback_paths_for(track, &key, ax);
                if fb_only && fb.is_empty() {
                    continue;
                }
                if !text.is_empty() && !key.to_ascii_lowercase().contains(&text) && !ax.to_ascii_lowercase().contains(&text) {
                    continue;
                }
                let resolves_to = sds
                    .adaptive_baseline_scoped(&sc, ax)
                    .map(|(_, rk)| rk)
                    .unwrap_or_else(|| key.clone());
                let mut row = json!({
                    "scope": key, "axis": ax, "kind": row_kind, "level": level,
                    "n": m.n, "need": need, "maturity": r4(mat),
                    "resolves_to": resolves_to,
                    "mean": r4(m.mean), "sd": r4(m.sd()), "recent": r4(m.recent_mean()),
                    "drift_z": r4(dz),
                    "feedback": fb, "engine_revs": revs,
                });
                if is_ref {
                    if let Some(rr) = sds.exact_reference(&sc, ax) {
                        row["reference"] = json!(r4(rr.mean_ref));
                        row["tolerance"] = json!(r4(rr.mean_tol));
                        row["dev"] = json!(r4(rr.mean_dev));
                        row["in_band"] = json!(r4(rr.in_band_rate));
                    }
                }
                rows.push(row);
            }

            for (ax, a) in &st.agreement {
                if ax.starts_with("p_success:bin") || ax.ends_with(":in_band") {
                    continue;
                }
                let row_kind = "agreement";
                if !kinds.is_empty() && !kinds.iter().any(|k| k == row_kind) {
                    continue;
                }
                if a.n < min_n {
                    continue;
                }
                let mat = a.n as f64 / need.max(1) as f64;
                if let Some(mm) = max_mat {
                    if mat > mm {
                        continue;
                    }
                }
                if drift_min.is_some() {
                    continue;
                }
                let fb = feedback_paths_for(track, &key, ax);
                if fb_only && fb.is_empty() {
                    continue;
                }
                if !text.is_empty() && !key.to_ascii_lowercase().contains(&text) && !ax.to_ascii_lowercase().contains(&text) {
                    continue;
                }
                let resolves_to = sds
                    .agreement_rate_scoped(&sc, ax)
                    .map(|(_, rk)| rk)
                    .unwrap_or_else(|| key.clone());
                rows.push(json!({
                    "scope": key, "axis": ax, "kind": row_kind, "level": level,
                    "n": a.n, "need": need, "maturity": r4(mat),
                    "resolves_to": resolves_to,
                    "rate": r4(a.mean), "drift_z": 0.0,
                    "feedback": fb, "engine_revs": revs,
                }));
            }

            for (gname, g) in &st.gates {
                let row_kind = "gate";
                if !kinds.is_empty() && !kinds.iter().any(|k| k == row_kind) {
                    continue;
                }
                if g.evaluated < min_n {
                    continue;
                }
                let mat = g.evaluated as f64 / need.max(1) as f64;
                if let Some(mm) = max_mat {
                    if mat > mm {
                        continue;
                    }
                }
                if drift_min.is_some() || fb_only {
                    continue;
                }
                if !text.is_empty() && !key.to_ascii_lowercase().contains(&text) && !gname.to_ascii_lowercase().contains(&text) {
                    continue;
                }
                let blocked_rate = if g.evaluated > 0 { g.blocked as f64 / g.evaluated as f64 } else { 0.0 };
                let sole_rate = if g.evaluated > 0 { g.sole as f64 / g.evaluated as f64 } else { 0.0 };
                rows.push(json!({
                    "scope": key, "axis": format!("gate:{gname}"), "kind": row_kind, "level": level,
                    "n": g.evaluated, "need": need, "maturity": r4(mat),
                    "evaluated": g.evaluated, "blocked_rate": r4(blocked_rate), "sole_rate": r4(sole_rate),
                    "drift_z": 0.0, "feedback": [], "engine_revs": revs,
                }));
            }
        }
    }

    match sort {
        "drift_desc" => rows.sort_by(|a, b| {
            let da = a["drift_z"].as_f64().unwrap_or(0.0).abs();
            let db = b["drift_z"].as_f64().unwrap_or(0.0).abs();
            db.total_cmp(&da).then_with(|| a["scope"].as_str().cmp(&b["scope"].as_str()))
        }),
        "maturity_asc" => rows.sort_by(|a, b| {
            let ma = a["maturity"].as_f64().unwrap_or(0.0);
            let mb = b["maturity"].as_f64().unwrap_or(0.0);
            ma.total_cmp(&mb).then_with(|| a["scope"].as_str().cmp(&b["scope"].as_str()))
        }),
        _ => rows.sort_by(|a, b| {
            let na = a["n"].as_u64().unwrap_or(0);
            let nb = b["n"].as_u64().unwrap_or(0);
            nb.cmp(&na)
                .then_with(|| a["scope"].as_str().cmp(&b["scope"].as_str()))
                .then_with(|| a["axis"].as_str().cmp(&b["axis"].as_str()))
        }),
    }

    let total = rows.len();
    let page: Vec<Value> = rows.into_iter().skip(cursor).take(limit).collect();
    let next_cursor = (cursor + page.len() < total).then_some(cursor + page.len());

    json!({
        "rev": rev,
        "total": total,
        "merged_scopes": merged_scopes,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "rows": page,
    })
}

pub fn sds_watch_api(watch: &mut SdsWatch, sds: &SdsStore, args: &Value) -> Value {
    if let Some(p) = args.get("pin").and_then(Value::as_object) {
        if let (Some(sc), Some(ax)) = (p.get("scope").and_then(Value::as_str), p.get("axis").and_then(Value::as_str)) {
            watch.pin(sc.to_string(), ax.to_string());
        }
    }
    if let Some(u) = args.get("unpin").and_then(Value::as_object) {
        if let (Some(sc), Some(ax)) = (u.get("scope").and_then(Value::as_str), u.get("axis").and_then(Value::as_str)) {
            watch.unpin(sc, ax);
        }
    }
    let rev = sds.rev();
    if args.get("if_rev").and_then(Value::as_u64) == Some(rev) && args.get("pin").is_none() && args.get("unpin").is_none() {
        return json!({"unchanged": true, "rev": rev});
    }
    let pinned: Vec<Value> = watch.pinned.iter().map(|(s, a)| json!({"scope": s, "axis": a})).collect();
    let series: Vec<Value> = watch
        .axes
        .iter()
        .map(|((s, a), pts)| json!({"scope": s, "axis": a, "points": pts}))
        .collect();
    json!({"rev": rev, "every_ticks": watch.every_ticks, "pinned": pinned, "series": series})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sds_query_filters_before_merging() {
        let mut sds = SdsStore::memory();
        for i in 0..10 {
            sds.record_baseline(&Scope::new(Track::Vision, "tree", "siglip"), "score", 0.75 + 0.01 * i as f64);
            sds.record_baseline(&Scope::new(Track::Vision, "boulder", "siglip"), "score", 0.65);
            sds.record_baseline(&Scope::new(Track::Balance, "nn:S4", "brake_1.5"), "score", 82.0);
        }
        let filtered = sds_query(&sds, &json!({"track": "vision", "prefix": "vision|tree|"}));
        // Only tree primary + secondary keys (2 scopes) should be merged, NOT boulder or balance!
        assert!(filtered["merged_scopes"].as_u64().unwrap() <= 2, "{filtered}");
        // Paginated sum equals full query total.
        let all = sds_query(&sds, &json!({"limit": 500}));
        let total = all["total"].as_u64().unwrap() as usize;
        let mut seen = 0usize;
        let mut cursor = 0usize;
        loop {
            let p = sds_query(&sds, &json!({"limit": 2, "cursor": cursor}));
            seen += p["rows"].as_array().unwrap().len();
            match p["next_cursor"].as_u64() {
                Some(nc) => cursor = nc as usize,
                None => break,
            }
        }
        assert_eq!(seen, total);
    }

    #[test]
    fn sds_query_need_follows_min_obs() {
        let mut sds = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "log", "embedding");
        for _ in 0..14 {
            sds.record_baseline(&sc, "score", 0.62);
        }
        let res = sds_query(&sds, &json!({"track": "vision", "prefix": "vision|log|embedding"}));
        let row = res["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["scope"] == "vision|log|embedding" && r["axis"] == "score")
            .expect("secondary row present");
        let (need_sec, _, _) = Track::Vision.min_obs();
        assert_eq!(row["need"].as_u64().unwrap(), need_sec);
        assert!((row["maturity"].as_f64().unwrap() - 14.0 / need_sec as f64).abs() < 1e-3);
        // Since n=14 < 20 (secondary min_obs) while primary n=14 >= 8, resolves_to falls back to primary!
        assert_eq!(row["resolves_to"], "vision|log|");
    }

    #[test]
    fn if_rev_returns_unchanged() {
        let mut sds = SdsStore::memory();
        sds.record_baseline(&Scope::new(Track::Vision, "tree", ""), "score", 0.8);
        let first = sds_query(&sds, &json!({}));
        let rev = first["rev"].as_u64().unwrap();
        let second = sds_query(&sds, &json!({"if_rev": rev}));
        assert_eq!(second["unchanged"], true);
        assert_eq!(second["rev"].as_u64().unwrap(), rev);
        let t2 = sds_tree(&sds, None, Some(rev));
        assert_eq!(t2["unchanged"], true);
    }
}
