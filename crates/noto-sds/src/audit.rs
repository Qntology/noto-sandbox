use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::store::write_atomic;

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const ANCHORS_FILE: &str = "anchors.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogLine {
    pub seq: u64,
    pub ts: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick: Option<u64>,
    pub cat: String,
    pub level: Level,
    pub msg: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileLine {
    #[serde(flatten)]
    line: LogLine,
    prev: String,
    hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorEntry {
    pub session: String,
    pub file: String,
    pub seq: u64,
    pub hash: String,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AnchorFile {
    #[serde(default)]
    last_session: Option<String>,
    #[serde(default)]
    sessions: BTreeMap<String, AnchorEntry>,
}

fn load_anchors(dir: &Path) -> AnchorFile {
    let p = dir.join(ANCHORS_FILE);
    std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<AnchorFile>(&t).ok())
        .unwrap_or_default()
}

fn save_anchor(dir: &Path, entry: AnchorEntry) -> std::io::Result<()> {
    let mut af = load_anchors(dir);
    af.last_session = Some(entry.session.clone());
    af.sessions.insert(entry.session.clone(), entry);
    let txt = serde_json::to_string_pretty(&af).unwrap_or_default();
    write_atomic(&dir.join(ANCHORS_FILE), txt.as_bytes())
}

fn chain_hash(prev: &str, line: &LogLine) -> String {
    let body = serde_json::to_string(line).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(prev.as_bytes());
    h.update(b"|");
    h.update(body.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOk {
    pub lines: usize,
    pub last_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyError {
    pub line_no: usize,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionVerifyReport {
    pub session: String,
    pub file: String,
    pub lines: usize,
    pub chain_ok: bool,
    pub broken_at: Option<usize>,
    pub reason: Option<String>,
    pub anchor_ok: Option<bool>,
    pub seal_ok: Option<bool>,
    pub linked_from: Option<String>,
}

pub fn verify_file(path: &Path) -> Result<VerifyOk, VerifyError> {
    let f = File::open(path).map_err(|e| VerifyError { line_no: 0, reason: format!("open: {e}") })?;
    let mut prev = GENESIS.to_string();
    let mut count = 0usize;
    let mut last_seq = 0u64;
    for (i, raw) in BufReader::new(f).lines().enumerate() {
        let line_no = i + 1;
        let raw = raw.map_err(|e| VerifyError { line_no, reason: format!("read: {e}") })?;
        if raw.trim().is_empty() {
            continue;
        }
        let fl: FileLine = serde_json::from_str(&raw).map_err(|e| VerifyError { line_no, reason: format!("parse: {e}") })?;
        if fl.prev != prev {
            return Err(VerifyError { line_no, reason: "previous-hash mismatch (a line was removed, inserted or reordered)".into() });
        }
        if fl.line.seq != last_seq + 1 {
            return Err(VerifyError { line_no, reason: format!("sequence jumps from {} to {}", last_seq, fl.line.seq) });
        }
        let expect = chain_hash(&prev, &fl.line);
        if fl.hash != expect {
            return Err(VerifyError { line_no, reason: "hash mismatch (the line content was edited)".into() });
        }
        prev = fl.hash;
        last_seq = fl.line.seq;
        count += 1;
    }
    Ok(VerifyOk { lines: count, last_hash: prev })
}

/// Verify all session audit files in `log_dir` against their hash chains, `anchors.json`
/// (detecting tail truncation), and optional live `sds_digests` (detecting external ledger edits)
/// (Phase 2 – Part 3 · D4).
pub fn verify_dir(log_dir: &Path, sds_digests: Option<&BTreeMap<String, String>>) -> Vec<SessionVerifyReport> {
    let anchors = load_anchors(log_dir);
    let mut files: Vec<PathBuf> = std::fs::read_dir(log_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .collect();
    files.sort();

    let mut out = Vec::with_capacity(files.len());
    let total = files.len();
    for (idx, path) in files.iter().enumerate() {
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let session = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let mut linked_from = None;
        let mut last_seal: Option<BTreeMap<String, String>> = None;
        if let Ok(txt) = std::fs::read_to_string(path) {
            for raw in txt.lines() {
                if let Ok(fl) = serde_json::from_str::<FileLine>(raw) {
                    if fl.line.cat == "session_link" && linked_from.is_none() {
                        linked_from = fl.line.data.as_ref().and_then(|d| d.get("prev_file")).and_then(|v| v.as_str()).map(String::from);
                    }
                    if fl.line.cat == "sds_seal" {
                        if let Some(map) = fl.line.data.as_ref().and_then(|d| d.get("digests")).and_then(|v| v.as_object()) {
                            let mut parsed = BTreeMap::new();
                            for (k, v) in map {
                                if let Some(s) = v.as_str() {
                                    parsed.insert(k.clone(), s.to_string());
                                }
                            }
                            last_seal = Some(parsed);
                        }
                    }
                }
            }
        }
        match verify_file(path) {
            Ok(ok) => {
                let anchor_ok = anchors.sessions.get(&session).map(|a| ok.lines as u64 == a.seq && ok.last_hash == a.hash);
                let seal_ok = if idx + 1 == total {
                    match (last_seal.as_ref(), sds_digests) {
                        (Some(sealed), Some(cur)) => Some(sealed.iter().all(|(k, v)| cur.get(k) == Some(v))),
                        _ => None,
                    }
                } else {
                    last_seal.as_ref().map(|_| true)
                };
                out.push(SessionVerifyReport {
                    session,
                    file: fname,
                    lines: ok.lines,
                    chain_ok: true,
                    broken_at: None,
                    reason: None,
                    anchor_ok,
                    seal_ok,
                    linked_from,
                });
            }
            Err(err) => {
                out.push(SessionVerifyReport {
                    session,
                    file: fname,
                    lines: err.line_no.saturating_sub(1),
                    chain_ok: false,
                    broken_at: Some(err.line_no),
                    reason: Some(err.reason),
                    anchor_ok: Some(false),
                    seal_ok: Some(false),
                    linked_from,
                });
            }
        }
    }
    out
}

pub struct AuditLog {
    ring: VecDeque<LogLine>,
    cap: usize,
    next_seq: u64,
    prev_hash: String,
    session: String,
    file: Option<(PathBuf, BufWriter<File>)>,
    pub write_error: Option<String>,
    anchor_dirty: bool,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

impl AuditLog {
    pub fn memory(cap: usize) -> Self {
        AuditLog {
            ring: VecDeque::new(),
            cap: cap.max(16),
            next_seq: 1,
            prev_hash: GENESIS.into(),
            session: "memory".into(),
            file: None,
            write_error: None,
            anchor_dirty: false,
        }
    }

    pub fn open(dir: &Path, session: &str, cap: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let safe: String = session.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let path = dir.join(format!("{safe}.jsonl"));
        let prev_anchor = {
            let af = load_anchors(dir);
            af.last_session.as_deref().and_then(|s| af.sessions.get(s).cloned())
        };
        let f = OpenOptions::new().create_new(true).append(true).open(&path)?;
        let mut log = AuditLog::memory(cap);
        log.session = safe;
        log.file = Some((path, BufWriter::new(f)));
        if let Some(prev) = prev_anchor {
            log.log(
                "session_link",
                Level::Info,
                format!("이전 세션 {} (seq {})와 체인 연결", prev.file, prev.seq),
                Some(serde_json::json!({
                    "prev_session": prev.session,
                    "prev_file": prev.file,
                    "prev_seq": prev.seq,
                    "prev_hash": prev.hash,
                })),
            );
        }
        Ok(log)
    }

    pub fn path(&self) -> Option<&Path> {
        self.file.as_ref().map(|(p, _)| p.as_path())
    }

    pub fn dir(&self) -> Option<&Path> {
        self.file.as_ref().and_then(|(p, _)| p.parent())
    }

    /// Atomically persist the current session's latest `(seq, hash)` into `anchors.json`.
    pub fn sync_anchor(&mut self) {
        if self.last_seq() == 0 {
            return;
        }
        let Some((path, _)) = &self.file else { return };
        let Some(dir) = path.parent() else { return };
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let entry = AnchorEntry {
            session: self.session.clone(),
            file: fname,
            seq: self.last_seq(),
            hash: self.prev_hash.clone(),
            updated_ms: now_ms(),
        };
        if let Err(e) = save_anchor(dir, entry) {
            self.write_error = Some(format!("anchor: {e}"));
        } else {
            self.anchor_dirty = false;
        }
    }

    pub fn log_with_tick(&mut self, cat: &str, level: Level, msg: impl Into<String>, data: Option<serde_json::Value>, tick: Option<u64>) -> u64 {
        let line = LogLine { seq: self.next_seq, ts: now_ms(), tick, cat: cat.to_string(), level, msg: msg.into(), data };
        let hash = chain_hash(&self.prev_hash, &line);
        if let Some((_, w)) = self.file.as_mut() {
            let fl = FileLine { line: line.clone(), prev: self.prev_hash.clone(), hash: hash.clone() };
            let res = serde_json::to_string(&fl).map_err(|e| e.to_string()).and_then(|s| {
                writeln!(w, "{s}").and_then(|_| w.flush()).map_err(|e| e.to_string())
            });
            if let Err(e) = res {
                let err_msg = e.clone();
                self.write_error = Some(e);
                self.ring.push_back(LogLine {
                    seq: self.next_seq,
                    ts: now_ms(),
                    tick,
                    cat: "audit".into(),
                    level: Level::Warn,
                    msg: format!("감사 로그 쓰기 오류: {err_msg}"),
                    data: None,
                });
            }
        }
        self.prev_hash = hash;
        self.next_seq += 1;
        let seq = line.seq;
        self.ring.push_back(line);
        while self.ring.len() > self.cap {
            self.ring.pop_front();
        }
        self.anchor_dirty = true;
        if self.file.is_some()
            && (seq == 1 || seq % 16 == 0 || matches!(cat, "session_link" | "sds_seal" | "sds_purge" | "sds_prune" | "replay"))
        {
            self.sync_anchor();
        }
        seq
    }

    pub fn log(&mut self, cat: &str, level: Level, msg: impl Into<String>, data: Option<serde_json::Value>) -> u64 {
        self.log_with_tick(cat, level, msg, data, None)
    }

    pub fn info(&mut self, cat: &str, msg: impl Into<String>) -> u64 {
        self.log(cat, Level::Info, msg, None)
    }

    pub fn warn(&mut self, cat: &str, msg: impl Into<String>) -> u64 {
        self.log(cat, Level::Warn, msg, None)
    }

    pub fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    pub fn last_hash(&self) -> &str {
        &self.prev_hash
    }

    pub fn tail(&self, after_seq: u64, limit: usize, cat: Option<&str>) -> (Vec<LogLine>, u64) {
        let limit = limit.max(1);
        let mut out = Vec::new();
        let mut next = after_seq;
        for l in self.ring.iter().filter(|l| l.seq > after_seq) {
            if out.len() >= limit {
                break;
            }
            next = l.seq;
            if cat.map_or(true, |c| c == l.cat) {
                out.push(l.clone());
            }
        }
        (out, next)
    }
}

impl Drop for AuditLog {
    fn drop(&mut self) {
        if self.anchor_dirty && self.file.is_some() {
            self.sync_anchor();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("noto_audit_{}_{}_{}", tag, std::process::id(), now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample(dir: &Path) -> (PathBuf, AuditLog) {
        let mut log = AuditLog::open(dir, "s1", 100).unwrap();
        log.info("world", "world created");
        log.log("vision", Level::Debug, "tile scores", Some(serde_json::json!({"tree": 0.71, "boulder": 0.2})));
        log.warn("quest", "step failed: no_path");
        log.info("sds", "flush");
        (log.path().unwrap().to_path_buf(), log)
    }

    #[test]
    fn floats_that_came_from_f32_survive_the_round_trip_of_verification() {
        let dir = tmpdir("floats");
        let mut log = AuditLog::open(&dir, "f", 100).unwrap();
        for i in 0..200 {
            let x = 0.98f32 - i as f32 * 0.00371;
            log.log("vision", Level::Info, "scores", Some(serde_json::json!({"score": x, "margin": x * 0.37, "nested": {"v": [x, x / 3.0]}})));
        }
        let path = log.path().unwrap().to_path_buf();
        drop(log);
        assert_eq!(verify_file(&path).unwrap().lines, 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fresh_chain_verifies_and_matches_the_in_memory_hash() {
        let dir = tmpdir("ok");
        let (path, log) = sample(&dir);
        let ok = verify_file(&path).unwrap();
        assert_eq!(ok.lines, 4);
        assert_eq!(ok.last_hash, log.last_hash());
        assert_eq!(log.last_seq(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_a_line_is_detected_at_that_line() {
        let dir = tmpdir("edit");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let edited = txt.replace("step failed: no_path", "step failed: none");
        assert_ne!(txt, edited);
        std::fs::write(&path, edited).unwrap();
        let err = verify_file(&path).unwrap_err();
        assert_eq!(err.line_no, 3);
        assert!(err.reason.contains("edited"), "{}", err.reason);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn removing_or_reordering_lines_is_detected() {
        let dir = tmpdir("remove");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = txt.lines().collect();

        let removed = [lines[0], lines[2], lines[3]].join("\n") + "\n";
        std::fs::write(&path, removed).unwrap();
        let err = verify_file(&path).unwrap_err();
        assert_eq!(err.line_no, 2);

        let swapped = [lines[0], lines[2], lines[1], lines[3]].join("\n") + "\n";
        std::fs::write(&path, swapped).unwrap();
        assert!(verify_file(&path).is_err());

        let truncated_tail = [lines[0], lines[1]].join("\n") + "\n";
        std::fs::write(&path, truncated_tail).unwrap();
        assert_eq!(verify_file(&path).unwrap().lines, 2);

        std::fs::write(&path, "{broken").unwrap();
        assert!(verify_file(&path).unwrap_err().reason.starts_with("parse"));
        assert!(verify_file(&dir.join("nope.jsonl")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_audit_chain_links_sessions_and_seals_ledgers() {
        let dir = tmpdir("seal_link");
        let mut digests = BTreeMap::new();
        digests.insert("body".to_string(), "aaa111".to_string());
        {
            let mut s1 = AuditLog::open(&dir, "s01", 64).unwrap();
            s1.info("world", "first session");
            s1.info("world", "second line");
        }
        let s2_path = {
            let mut s2 = AuditLog::open(&dir, "s02", 64).unwrap();
            s2.log("sds_seal", Level::Info, "seal", Some(serde_json::json!({"generation": "g1", "digests": digests})));
            s2.path().unwrap().to_path_buf()
        };
        let rep = verify_dir(&dir, Some(&digests));
        assert_eq!(rep.len(), 2);
        assert!(rep[0].chain_ok && rep[0].anchor_ok == Some(true));
        assert!(rep[1].chain_ok && rep[1].anchor_ok == Some(true) && rep[1].seal_ok == Some(true));
        assert_eq!(rep[1].linked_from.as_deref(), Some("s01.jsonl"));

        // External ledger edit -> seal_ok becomes false
        let mut altered = digests.clone();
        altered.insert("body".to_string(), "tampered".to_string());
        let rep_edit = verify_dir(&dir, Some(&altered));
        assert_eq!(rep_edit[1].seal_ok, Some(false), "external ledger edit must fail seal check");

        // Tail truncation -> anchor_ok becomes false
        let txt = std::fs::read_to_string(&s2_path).unwrap();
        let first_line = txt.lines().next().unwrap().to_string() + "\n";
        std::fs::write(&s2_path, first_line).unwrap();
        let rep_trunc = verify_dir(&dir, Some(&digests));
        assert_eq!(rep_trunc[1].anchor_ok, Some(false), "tail truncation must fail anchor check");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tail_is_incremental_and_filters_by_category() {
        let mut log = AuditLog::memory(100);
        for i in 0..10 {
            log.info(if i % 2 == 0 { "world" } else { "quest" }, format!("m{i}"));
        }
        let (a, next) = log.tail(0, 4, None);
        assert_eq!(a.iter().map(|l| l.seq).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(next, 4);
        let (b, next) = log.tail(next, 100, Some("quest"));
        assert_eq!(b.iter().map(|l| l.msg.clone()).collect::<Vec<_>>(), vec!["m5", "m7", "m9"]);
        assert_eq!(next, 10, "advances past lines the filter skipped");
        let (c, next2) = log.tail(next, 10, None);
        assert!(c.is_empty());
        assert_eq!(next2, 10);
    }

    #[test]
    fn the_ring_is_bounded_but_the_file_keeps_everything() {
        let dir = tmpdir("ring");
        let mut log = AuditLog::open(&dir, "s2", 16).unwrap();
        for i in 0..50 {
            log.info("world", format!("line {i}"));
        }
        let (lines, _) = log.tail(0, 1000, None);
        assert_eq!(lines.len(), 16);
        assert_eq!(lines[0].seq, 35);
        assert_eq!(verify_file(log.path().unwrap()).unwrap().lines, 50);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_file_is_never_overwritten() {
        let dir = tmpdir("exists");
        let _a = AuditLog::open(&dir, "same", 16).unwrap();
        assert!(AuditLog::open(&dir, "same", 16).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_round_trips_through_the_file_format() {
        let dir = tmpdir("data");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let second: FileLine = serde_json::from_str(txt.lines().nth(1).unwrap()).unwrap();
        assert_eq!(second.line.cat, "vision");
        assert_eq!(second.line.data.unwrap()["tree"], 0.71);
        assert_eq!(second.line.level, Level::Debug);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audit_old_lines_still_verify() {
        let dir = tmpdir("legacy");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("legacy.jsonl");
        let body1 = r#"{"seq":1,"ts":1700000000000,"cat":"world","level":"info","msg":"legacy line 1"}"#;
        let h1: String = Sha256::digest(format!("{GENESIS}|{body1}").as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let line1 = format!(r#"{{"seq":1,"ts":1700000000000,"cat":"world","level":"info","msg":"legacy line 1","prev":"{GENESIS}","hash":"{h1}"}}"#);
        let body2 = r#"{"seq":2,"ts":1700000001000,"cat":"quest","level":"info","msg":"legacy line 2"}"#;
        let h2: String = Sha256::digest(format!("{h1}|{body2}").as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let line2 = format!(r#"{{"seq":2,"ts":1700000001000,"cat":"quest","level":"info","msg":"legacy line 2","prev":"{h1}","hash":"{h2}"}}"#);
        std::fs::write(&path, format!("{line1}\n{line2}\n")).unwrap();
        let ok = verify_file(&path).expect("pre-tick audit file must verify");
        assert_eq!(ok.lines, 2);
        assert_eq!(ok.last_hash, h2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
