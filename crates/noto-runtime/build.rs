//! Computes `NOTO_ENGINE_FP` (12-hex SHA-256 prefix over sorted engine source files,
//! matching `noto_core::bench::ENGINE_FILES` minus `NOT_ENGINE` and `*bench.rs`).

use std::path::{Path, PathBuf};

const ENGINE_FILES: [&str; 12] = [
    "crates/noto-core/src/**/*.rs",
    "crates/world-sim/src/**/*.rs",
    "crates/noto-runtime/src/stereo.rs",
    "crates/noto-runtime/src/driver.rs",
    "crates/noto-runtime/src/mldrive.rs",
    "crates/noto-runtime/src/vision.rs",
    "crates/noto-runtime/src/precision.rs",
    "crates/noto-runtime/src/runtime.rs",
    "crates/noto-runtime/src/record.rs",
    "crates/noto-runtime/src/cargo.rs",
    "crates/noto-runtime/src/chase.rs",
    "crates/noto-runtime/src/perception.rs",
];

const NOT_ENGINE: [&str; 3] = [
    "crates/noto-core/src/lib.rs",
    "crates/noto-core/src/effects.rs",
    "crates/noto-core/src/nn.rs",
];

const REFERENCE_SECTIONS: [(&str, &str); 1] = [
    ("crates/noto-core/src/balance.rs", "pub struct RefCtx {"),
];

fn is_bench_module(rel: &str) -> bool {
    rel.starts_with("crates/noto-core/src/") && rel.ends_with("bench.rs")
}

fn excluded(rel: &str) -> bool {
    is_bench_module(rel) || NOT_ENGINE.contains(&rel)
}

fn walk(root: &Path, rel_dir: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(root.join(rel_dir)) else { return };
    let mut entries: Vec<(String, bool)> = rd
        .filter_map(|e| e.ok())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.file_type().map_or(false, |t| t.is_dir())))
        .collect();
    entries.sort();
    for (name, dir) in entries {
        let rel = format!("{rel_dir}/{name}");
        if dir {
            walk(root, &rel, out);
        } else if name.ends_with(".rs") {
            out.push(rel);
        }
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir);
    let mut files = Vec::new();
    for p in ENGINE_FILES {
        match p.strip_suffix("/**/*.rs") {
            Some(dir) => walk(&root, dir, &mut files),
            None => {
                if root.join(p).is_file() {
                    files.push(p.to_string());
                }
            }
        }
    }
    files.retain(|f| !excluded(f));
    files.sort();
    files.dedup();

    let mut payload = Vec::new();
    for rel in &files {
        let full = root.join(rel);
        println!("cargo:rerun-if-changed={}", full.display());
        payload.extend_from_slice(rel.as_bytes());
        payload.push(0);
        if let Ok(text) = std::fs::read_to_string(&full) {
            let trimmed = match REFERENCE_SECTIONS.iter().find(|(f, _)| *f == rel.as_str()) {
                Some((_, marker)) => text.split(*marker).next().unwrap_or(&text),
                None => &text,
            };
            // Normalize CRLF -> LF so Windows and Unix produce the exact same fingerprint.
            for line in trimmed.split('\n') {
                let l = line.strip_suffix('\r').unwrap_or(line);
                payload.extend_from_slice(l.as_bytes());
                payload.push(b'\n');
            }
        }
        payload.push(0xff);
    }
    let digest = sha256(&payload);
    let hex: String = digest[..6].iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo:rustc-env=NOTO_ENGINE_FP={hex}");
}
