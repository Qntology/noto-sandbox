use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Track {
    Vision,
    Search,
    Motion,
    Forecast,
    Depth,
    Locate,
    Tracker,
    Decide,
    Fusion,
    Balance,
    Effort,
    Quest,
    Act,
    Memory,
}

/// Statistical interpretation of an axis (Phase 2 – Part 2 · D4/D5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisKind {
    #[default]
    Level,
    Rate,
    Cumulative,
}

/// Which temporal window to read from an axis (Phase 2 – Part 2 · D1/D3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Window {
    #[default]
    Lifetime,
    Long,
    Short,
    Span,
    Auto,
}

/// Precision-based minimum-observation rule for a track/axis kind (Phase 2 – Part 2 · D4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MinObsRule {
    pub floor_n: u64,
    pub kind: AxisKind,
    /// Optional half-width target for the confidence interval; `None` means `floor_n` only.
    pub target_hw: Option<f64>,
}

/// Exponential-decay window, CUSUM detector, and read-window policy per track (Phase 2 – Part 2 · D1–D4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalPolicy {
    pub m_short: f64,
    pub m_long: f64,
    pub cusum_k: f64,
    pub cusum_h: f64,
    pub read: Window,
    pub min_obs: MinObsRule,
}

impl Track {
    pub const ALL: [Track; 14] = [
        Track::Vision,
        Track::Search,
        Track::Motion,
        Track::Forecast,
        Track::Depth,
        Track::Locate,
        Track::Tracker,
        Track::Decide,
        Track::Fusion,
        Track::Balance,
        Track::Effort,
        Track::Quest,
        Track::Act,
        Track::Memory,
    ];

    pub const ML: [Track; 4] = [Track::Depth, Track::Locate, Track::Tracker, Track::Decide];

    pub fn as_str(&self) -> &'static str {
        match self {
            Track::Vision => "vision",
            Track::Search => "search",
            Track::Motion => "motion",
            Track::Forecast => "forecast",
            Track::Depth => "depth",
            Track::Locate => "locate",
            Track::Tracker => "tracker",
            Track::Decide => "decide",
            Track::Fusion => "fusion",
            Track::Balance => "balance",
            Track::Effort => "effort",
            Track::Quest => "quest",
            Track::Act => "act",
            Track::Memory => "memory",
        }
    }

    pub fn from_name(name: &str) -> Option<Track> {
        Track::ALL.into_iter().find(|t| t.as_str() == name)
    }

    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Search => (20, 8, 30),
            Track::Motion => (6, 4, 8),
            Track::Forecast => (4, 3, 6),
            Track::Depth | Track::Locate | Track::Tracker | Track::Fusion => (12, 6, 20),
            Track::Decide => (8, 4, 12),
            Track::Balance | Track::Effort => (8, 4, 12),
            Track::Quest => (4, 3, 6),
            Track::Act => (6, 4, 8),
            Track::Memory => (6, 4, 8),
        }
    }

    /// Default temporal policy: `m_s = 2 * primary_min_obs`, `m_l = 8 * m_s`, standard 1σ CUSUM `(k=0.5, h=5.0)`.
    pub fn temporal_policy(&self) -> TemporalPolicy {
        let (_, prim, _) = self.min_obs();
        let m_short = (prim * 2).max(6) as f64;
        let m_long = m_short * 8.0;
        let target_hw = match self {
            Track::Vision | Track::Locate => Some(0.15),
            Track::Depth => Some(0.25),
            Track::Memory => Some(0.05),
            _ => None,
        };
        TemporalPolicy {
            m_short,
            m_long,
            cusum_k: 0.5,
            cusum_h: 5.0,
            read: Window::Lifetime,
            min_obs: MinObsRule { floor_n: prim, kind: AxisKind::Level, target_hw },
        }
    }

    pub fn ring_len(&self) -> usize {
        (self.min_obs().1 as usize).max(4)
    }

    pub fn ledger(&self) -> Ledger {
        match self {
            Track::Vision => Ledger::Perception,
            Track::Search => Ledger::Command,
            Track::Motion | Track::Forecast | Track::Quest => Ledger::Task,
            Track::Depth | Track::Locate | Track::Tracker | Track::Decide | Track::Fusion => Ledger::Ml,
            Track::Balance | Track::Effort => Ledger::Body,
            Track::Act => Ledger::Act,
            Track::Memory => Ledger::Perception,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ledger {
    Perception,
    Command,
    Task,
    Ml,
    Body,
    Act,
}

impl Ledger {
    pub const ALL: [Ledger; 6] = [Ledger::Perception, Ledger::Command, Ledger::Task, Ledger::Ml, Ledger::Body, Ledger::Act];

    pub fn as_str(&self) -> &'static str {
        match self {
            Ledger::Perception => "perception",
            Ledger::Command => "command",
            Ledger::Task => "task",
            Ledger::Ml => "ml",
            Ledger::Body => "body",
            Ledger::Act => "act",
        }
    }

    pub fn file_name(&self) -> String {
        format!("{}.json", self.as_str())
    }
}

/// Legacy v1 two-slot scope (`track|primary|secondary`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub track: Track,
    pub primary: String,
    pub secondary: String,
}

impl Scope {
    pub fn new(track: Track, primary: &str, secondary: &str) -> Self {
        Scope { track, primary: primary.trim().to_lowercase(), secondary: secondary.trim().to_lowercase() }
    }

    pub fn key_secondary(&self) -> String {
        format!("{}|{}|{}", self.track.as_str(), self.primary, self.secondary)
    }

    pub fn key_primary(&self) -> String {
        format!("{}|{}|", self.track.as_str(), self.primary)
    }

    pub fn key_global(&self) -> String {
        format!("{}||", self.track.as_str())
    }

    pub fn write_keys(&self) -> Vec<String> {
        let mut keys = vec![self.key_global()];
        if !self.primary.is_empty() {
            keys.push(self.key_primary());
            if !self.secondary.is_empty() {
                keys.push(self.key_secondary());
            }
        }
        keys
    }
}

/// Faceted dimension names for ScopeKey v2 in canonical registry order (Phase 2 – Part 1 · D1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Facet {
    Ph,
    Set,
    Stg,
    Form,
    Body,
    Gnd,
    Slope,
    Act,
    Src,
    Mode,
    Wk,
    Pass,
}

impl Facet {
    pub const ALL: [Facet; 12] = [
        Facet::Ph,
        Facet::Set,
        Facet::Stg,
        Facet::Form,
        Facet::Body,
        Facet::Gnd,
        Facet::Slope,
        Facet::Act,
        Facet::Src,
        Facet::Mode,
        Facet::Wk,
        Facet::Pass,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Facet::Ph => "ph",
            Facet::Set => "set",
            Facet::Stg => "stg",
            Facet::Form => "form",
            Facet::Body => "body",
            Facet::Gnd => "gnd",
            Facet::Slope => "slope",
            Facet::Act => "act",
            Facet::Src => "src",
            Facet::Mode => "mode",
            Facet::Wk => "wk",
            Facet::Pass => "pass",
        }
    }

    pub fn from_name(s: &str) -> Option<Facet> {
        Facet::ALL.into_iter().find(|f| f.as_str() == s)
    }
}

fn encode_part(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for b in raw.trim().to_lowercase().bytes() {
        match b {
            b'/' => out.push_str("%2F"),
            b';' => out.push_str("%3B"),
            b'=' => out.push_str("%3D"),
            b'%' => out.push_str("%25"),
            _ => out.push(b as char),
        }
    }
    out
}

fn decode_part(raw: &str) -> Result<String, String> {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(format!("truncated percent escape in '{raw}'"));
            }
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).map_err(|_| format!("invalid utf8 escape in '{raw}'"))?;
            let val = u8::from_str_radix(hex, 16).map_err(|_| format!("invalid hex escape '%{hex}' in '{raw}'"))?;
            out.push(val);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| format!("invalid utf8 after decoding '{raw}'"))
}

/// Faceted SDS v2 scope key: `<track>/<subject>;<facet>=<val>;...` (Phase 2 – Part 1 · D1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopeKey {
    pub track: Track,
    pub subject: String,
    pub facets: Vec<(Facet, String)>,
}

impl ScopeKey {
    pub fn new(track: Track, subject: &str, facets: Vec<(Facet, String)>) -> Self {
        let mut clean: Vec<(Facet, String)> = facets
            .into_iter()
            .filter_map(|(f, v)| {
                let norm = v.trim().to_lowercase();
                (!norm.is_empty()).then_some((f, norm))
            })
            .collect();
        clean.sort_by_key(|(f, _)| *f);
        clean.dedup_by_key(|(f, _)| *f);
        ScopeKey {
            track,
            subject: subject.trim().to_lowercase(),
            facets: clean,
        }
    }

    pub fn global(track: Track) -> Self {
        ScopeKey { track, subject: String::new(), facets: Vec::new() }
    }

    pub fn with_facet(mut self, facet: Facet, val: &str) -> Self {
        let norm = val.trim().to_lowercase();
        if !norm.is_empty() {
            self.facets.retain(|(f, _)| *f != facet);
            self.facets.push((facet, norm));
            self.facets.sort_by_key(|(f, _)| *f);
        }
        self
    }

    pub fn facet(&self, facet: Facet) -> Option<&str> {
        self.facets.iter().find(|(f, _)| *f == facet).map(|(_, v)| v.as_str())
    }

    /// Key hierarchy depth: 0 = global (`track/`), 1 = subject (`track/subj`), 1 + k = k facets.
    pub fn depth(&self) -> usize {
        if self.subject.is_empty() {
            0
        } else {
            1 + self.facets.len()
        }
    }

    /// Retain only the specified facet subset (preserving canonical order).
    pub fn project(&self, keep: &[Facet]) -> ScopeKey {
        if self.subject.is_empty() {
            return ScopeKey::global(self.track);
        }
        let facets = self.facets.iter().filter(|(f, _)| keep.contains(f)).cloned().collect();
        ScopeKey { track: self.track, subject: self.subject.clone(), facets }
    }

    pub fn to_key_string(&self) -> String {
        let mut s = format!("{}/{}", self.track.as_str(), encode_part(&self.subject));
        for (f, v) in &self.facets {
            if !v.is_empty() {
                s.push(';');
                s.push_str(f.as_str());
                s.push('=');
                s.push_str(&encode_part(v));
            }
        }
        s
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        let (tr_s, rest) = raw.split_once('/').ok_or_else(|| format!("missing '/' in scope key '{raw}'"))?;
        let track = Track::from_name(tr_s).ok_or_else(|| format!("unknown track '{tr_s}'"))?;
        let mut parts = rest.split(';');
        let subj_raw = parts.next().unwrap_or("");
        let subject = decode_part(subj_raw)?;
        let mut facets = Vec::new();
        for seg in parts {
            if seg.is_empty() {
                continue;
            }
            let (k, v) = seg.split_once('=').ok_or_else(|| format!("invalid facet '{seg}' in '{raw}'"))?;
            let facet = Facet::from_name(k).ok_or_else(|| format!("unknown facet '{k}' in '{raw}'"))?;
            let val = decode_part(v)?;
            if !val.is_empty() {
                facets.push((facet, val));
            }
        }
        Ok(ScopeKey::new(track, &subject, facets))
    }

    /// Convert a legacy v1 pipe-delimited scope string (`track|primary|secondary...`) to v2 `ScopeKey`
    /// using the per-track migration table (Phase 2 – Part 1 · §5).
    pub fn from_v1(v1: &str) -> Result<Self, String> {
        if v1.contains('/') && !v1.contains('|') {
            return ScopeKey::parse(v1);
        }
        let mut segs = v1.splitn(3, '|');
        let tr_s = segs.next().unwrap_or("");
        let track = Track::from_name(tr_s).ok_or_else(|| format!("unknown v1 track '{tr_s}'"))?;
        let primary = segs.next().unwrap_or("").trim().to_lowercase();
        let secondary = segs.next().unwrap_or("").trim().to_lowercase();
        if primary.is_empty() {
            return Ok(ScopeKey::global(track));
        }
        if secondary.is_empty() {
            return Ok(ScopeKey::new(track, &primary, vec![]));
        }
        let key = match track {
            Track::Balance => {
                if primary.starts_with("nn:") {
                    ScopeKey::new(track, &primary, vec![(Facet::Ph, secondary)])
                } else if primary.starts_with("fx:") {
                    let parts: Vec<&str> = secondary.split('|').collect();
                    match parts.as_slice() {
                        [gnd, slope, act] => ScopeKey::new(track, &primary, vec![(Facet::Gnd, (*gnd).into()), (Facet::Slope, (*slope).into()), (Facet::Act, (*act).into())]),
                        [gnd, act] => ScopeKey::new(track, &primary, vec![(Facet::Gnd, (*gnd).into()), (Facet::Act, (*act).into())]),
                        _ => ScopeKey::new(track, &primary, vec![(Facet::Act, secondary)]),
                    }
                } else if primary.contains('>') {
                    let parts: Vec<&str> = secondary.split('|').collect();
                    match parts.as_slice() {
                        [slope, act] => ScopeKey::new(track, &primary, vec![(Facet::Slope, (*slope).into()), (Facet::Act, (*act).into())]),
                        _ => ScopeKey::new(track, &primary, vec![(Facet::Act, secondary)]),
                    }
                } else {
                    ScopeKey::new(track, &primary, vec![(Facet::Act, secondary)])
                }
            }
            Track::Effort => {
                let parts: Vec<&str> = secondary.split('|').collect();
                match parts.as_slice() {
                    [src, slope, act] => ScopeKey::new(track, &primary, vec![(Facet::Src, (*src).into()), (Facet::Slope, (*slope).into()), (Facet::Act, (*act).into())]),
                    [slope, act] => ScopeKey::new(track, &primary, vec![(Facet::Slope, (*slope).into()), (Facet::Act, (*act).into())]),
                    _ => ScopeKey::new(track, &primary, vec![(Facet::Act, secondary)]),
                }
            }
            Track::Depth => {
                // v1 was depth|<mode>|<label>; v2 normalizes subject=label, mode=<mode>
                ScopeKey::new(track, &secondary, vec![(Facet::Mode, primary)])
            }
            Track::Vision | Track::Locate | Track::Tracker | Track::Decide | Track::Fusion | Track::Search => {
                ScopeKey::new(track, &primary, vec![(Facet::Mode, secondary)])
            }
            Track::Forecast => ScopeKey::new(track, &primary, vec![(Facet::Mode, secondary)]),
            Track::Act => {
                if primary == "walk" {
                    let is_form = secondary.contains(":f") && secondary.matches(':').count() >= 2;
                    let f = if is_form { Facet::Form } else { Facet::Gnd };
                    ScopeKey::new(track, &primary, vec![(f, secondary)])
                } else if primary.ends_with(":ts") {
                    ScopeKey::new(track, &primary, vec![(Facet::Gnd, secondary)])
                } else if primary.contains(">part") {
                    ScopeKey::new(track, &primary, vec![(Facet::Ph, secondary)])
                } else if secondary.contains(":f") && secondary.matches(':').count() >= 2 {
                    ScopeKey::new(track, &primary, vec![(Facet::Form, secondary)])
                } else {
                    ScopeKey::new(track, &primary, vec![(Facet::Act, secondary)])
                }
            }
            Track::Quest => {
                let parts: Vec<&str> = primary.split('|').collect();
                if parts.len() >= 2 {
                    ScopeKey::new(track, parts[1], vec![(Facet::Pass, parts[0].into()), (Facet::Stg, secondary)])
                } else {
                    ScopeKey::new(track, &primary, vec![(Facet::Stg, secondary)])
                }
            }
            Track::Motion | Track::Memory => ScopeKey::new(track, &primary, vec![(Facet::Mode, secondary)]),
        };
        Ok(key)
    }
}

impl std::fmt::Display for ScopeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_key_string())
    }
}

/// Depth of either a v1 (`track|p|s`) or v2 (`track/subj;f=v`) scope key string.
pub fn key_depth(key: &str) -> usize {
    if let Ok(sk) = ScopeKey::parse(key) {
        return sk.depth();
    }
    let mut parts = key.splitn(3, '|');
    let _t = parts.next();
    let p = parts.next().unwrap_or("");
    let s = parts.next().unwrap_or("");
    if p.is_empty() {
        0
    } else if s.is_empty() {
        1
    } else {
        2
    }
}

pub fn ring_for_key(key: &str) -> usize {
    let head = key.split(|c| c == '|' || c == '/').next().unwrap_or("");
    Track::from_name(head).map_or(6, |t| t.ring_len())
}

pub fn track_of_key(key: &str) -> Option<Track> {
    let head = key.split(|c| c == '|' || c == '/').next()?;
    Track::from_name(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_normalised_and_hierarchical() {
        let s = Scope::new(Track::Vision, "  Tree ", "NEAR");
        assert_eq!(s.key_secondary(), "vision|tree|near");
        assert_eq!(s.key_primary(), "vision|tree|");
        assert_eq!(s.key_global(), "vision||");
        assert_eq!(s.write_keys(), vec!["vision||", "vision|tree|", "vision|tree|near"]);
        assert_eq!(Scope::new(Track::Search, "", "x").write_keys(), vec!["search||"], "no primary, no deeper keys");
        assert_eq!(Scope::new(Track::Motion, "track", "").write_keys().len(), 2);
    }

    #[test]
    fn every_track_has_a_ledger_and_a_sane_ring() {
        for t in Track::ALL {
            assert_eq!(Track::from_name(t.as_str()), Some(t));
            assert!(t.ring_len() >= 4);
            let (s, p, g) = t.min_obs();
            assert!(s >= p && g >= p, "secondary and global need at least the primary count");
            assert!(Ledger::ALL.contains(&t.ledger()));
        }
        assert_eq!(ring_for_key("vision|tree|"), Track::Vision.ring_len());
        assert_eq!(ring_for_key("balance/nn:s4;ph=brake_1.5"), Track::Balance.ring_len());
        assert_eq!(ring_for_key("unknown|x|"), 6);
    }

    #[test]
    fn scope_v2_roundtrip() {
        // Order-independent facet canonicalization + percent-encoding of '/', ';', '='
        let a = ScopeKey::new(
            Track::Balance,
            "nn:s4",
            vec![
                (Facet::Gnd, "wet/sand".into()),
                (Facet::Ph, "brake=1.5;x".into()),
                (Facet::Form, "mid:humanfoot:f5".into()),
            ],
        );
        let b = ScopeKey::new(
            Track::Balance,
            "nn:s4",
            vec![
                (Facet::Form, "mid:humanfoot:f5".into()),
                (Facet::Ph, "brake=1.5;x".into()),
                (Facet::Gnd, "wet/sand".into()),
            ],
        );
        assert_eq!(a, b);
        let s = a.to_key_string();
        assert_eq!(s, "balance/nn:s4;ph=brake%3D1.5%3Bx;form=mid:humanfoot:f5;gnd=wet%2Fsand");
        assert_eq!(ScopeKey::parse(&s).unwrap(), a);

        // Migration checks
        assert_eq!(ScopeKey::from_v1("balance|nn:s4|brake_1.0").unwrap().to_key_string(), "balance/nn:s4;ph=brake_1.0");
        assert_eq!(ScopeKey::from_v1("depth|tips|tree").unwrap().to_key_string(), "depth/tree;mode=tips");
        assert_eq!(ScopeKey::from_v1("effort|arm|arm_lift|up_lo|pick").unwrap().to_key_string(), "effort/arm;slope=up_lo;act=pick;src=arm_lift");
        assert_eq!(ScopeKey::from_v1("act|walk|mid:humanfoot:f5").unwrap().to_key_string(), "act/walk;form=mid:humanfoot:f5");
        assert_eq!(ScopeKey::from_v1("act|walk|sand").unwrap().to_key_string(), "act/walk;gnd=sand");
    }
}
