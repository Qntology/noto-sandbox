use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use premium_discount_manifest::{hash_prefix, hex, part_path};
use sha2::{Digest, Sha256};
use tauri::State;

use crate::sim::models_json;
use crate::state::{bank_dir, model_dir, tips_dir, ttm_dir, ttm_download_dir, AppState};

type App<'a> = State<'a, Arc<AppState>>;

pub struct FileSpec {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub role_ko: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
    pub files: &'static [FileSpec],
}

pub const MODELS: [ModelSpec; 3] = [
    ModelSpec {
        id: "siglip2",
        label: "SigLIP2 base (patch16, NaFlex)",
        role_ko: "비전 인식 · 의미 위치 · 명령 임베딩",
        repo: "google/siglip2-base-patch16-naflex",
        revision: "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
        license: "Apache-2.0",
        files: &[
            FileSpec { name: "config.json", bytes: 329, sha256: "c0b8c2e7f0527b0bea1b1d9abe0381c0f294352df92439c54590b8420e539118" },
            FileSpec { name: "tokenizer.json", bytes: 34_356_304, sha256: "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0" },
            FileSpec { name: "model.safetensors", bytes: 1_500_985_224, sha256: "ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec" },
        ],
    },
    ModelSpec {
        id: "ttm",
        label: "Granite TTM-R3 (52-16)",
        role_ko: "시계열 예측 (정밀도 게이트)",
        repo: "ibm-granite/granite-timeseries-ttm-r3",
        revision: "7d99b7e35b3dc18f9db3b2b73cc9699dea6c9508",
        license: "Apache-2.0",
        files: &[
            FileSpec { name: "config.json", bytes: 2_669, sha256: "2c69af1a0ba3b3582b5a84adb309c3eb0ed24dec204576fcf23d7894c3a10f07" },
            FileSpec { name: "model.safetensors", bytes: 5_885_560, sha256: "710403c80572c022702ae6a2fad069c309c22b04ea96f17838f0ce7c8e4bba9c" },
        ],
    },
    ModelSpec {
        id: "tips",
        label: "TIPSv2-B/14 DPT",
        role_ko: "깊이 추정 (키프레임)",
        repo: "google/tipsv2-b14-dpt",
        revision: "da81b552538c453432e7a1efea39737c89580424",
        license: "Apache-2.0",
        files: &[
            FileSpec { name: "config.json", bytes: 7_622, sha256: "4727780d704da533f4caa36405d96af9fb8a602154ac248dcfe4afc8a8756883" },
            FileSpec { name: "model.safetensors", bytes: 632_066_716, sha256: "718757c7c816630bbe2a9ac9bdce4ee3ec22f2e1ddbb32c838ff569dc15f88a9" },
        ],
    },
];

pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn folder(id: &str) -> PathBuf {
    match id {
        "siglip2" => model_dir().unwrap_or_default(),
        "ttm" => ttm_download_dir(),
        _ => tips_dir(),
    }
}

fn hash_verified(path: &Path, expected_sha256: &str, mtime_ns: u128, len: u64) -> bool {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, (u128, u64, String)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some((mt, sz, digest)) = guard.get(path) {
            if *mt == mtime_ns && *sz == len {
                return digest.eq_ignore_ascii_case(expected_sha256);
            }
        }
    }
    let Ok(mut f) = File::open(path) else { return false };
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(_) => return false,
        }
    }
    let digest = format!("{:x}", h.finalize());
    if let Ok(mut guard) = cache.lock() {
        guard.insert(path.to_path_buf(), (mtime_ns, len, digest.clone()));
    }
    digest.eq_ignore_ascii_case(expected_sha256)
}

fn file_state(dir: &Path, f: &FileSpec) -> (u64, bool) {
    let p = dir.join(f.name);
    if let Ok(m) = fs::metadata(&p) {
        if m.len() == f.bytes {
            let mtime_ns = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            // Re-verify SHA-256 on small/fast weights (e.g., TTM <= 64 MiB) with mtime+size cache.
            let ok = if !f.sha256.is_empty() && f.bytes <= 64 * 1024 * 1024 {
                hash_verified(&p, f.sha256, mtime_ns, m.len())
            } else {
                true
            };
            if ok {
                return (f.bytes, true);
            }
            return (0, false);
        }
    }
    (fs::metadata(part_path(&dir.join(f.name))).map_or(0, |m| m.len().min(f.bytes)), false)
}

pub fn present(m: &ModelSpec) -> bool {
    m.files.iter().all(|f| file_state(&folder(m.id), f).1)
}

/// Whether the eye needs this model: a stereo pair measures depth with SGBM and has no use for TIPSv2.
pub fn needed(m: &ModelSpec, stereo: bool) -> bool {
    !(stereo && m.id == "tips")
}

pub fn missing(stereo: bool) -> Vec<&'static ModelSpec> {
    MODELS.iter().filter(|m| needed(m, stereo) && !present(m)).collect()
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub running: bool,
    pub model: Option<String>,
    pub file: Option<String>,
    pub phase: String,
    pub done: u64,
    pub total: u64,
    pub rate_bps: f64,
    pub eta_s: Option<f64>,
    pub error: Option<String>,
    pub queue: Vec<String>,
}

#[derive(Default)]
pub struct Downloads {
    progress: Mutex<Progress>,
    cancel: AtomicBool,
}

impl Downloads {
    pub fn snapshot(&self) -> Progress {
        self.progress.lock().map(|p| p.clone()).unwrap_or_default()
    }

    pub fn running(&self) -> bool {
        self.snapshot().running
    }

    fn update(&self, f: impl FnOnce(&mut Progress)) {
        if let Ok(mut p) = self.progress.lock() {
            f(&mut p);
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

const CANCELLED: &str = "취소됨";

async fn run(dl: Arc<Downloads>, ids: Vec<&'static ModelSpec>) {
    let client = match reqwest::Client::builder().user_agent(concat!("noto-sim-app/", env!("CARGO_PKG_VERSION"))).build() {
        Ok(c) => c,
        Err(e) => {
            dl.update(|p| {
                p.running = false;
                p.phase = "error".into();
                p.error = Some(format!("HTTP 클라이언트를 만들 수 없습니다: {e}"));
            });
            return;
        }
    };
    let jobs: Vec<(&ModelSpec, &FileSpec, PathBuf)> =
        ids.iter().flat_map(|m| m.files.iter().filter(|f| !file_state(&folder(m.id), f).1).map(move |f| (*m, f, folder(m.id)))).collect();
    let total: u64 = jobs.iter().map(|(_, f, _)| f.bytes).sum();
    let mut before: u64 = 0;
    let mut meter = Meter::new(jobs.iter().map(|(_, f, d)| file_state(d, f).0).sum());
    for (m, f, dir) in &jobs {
        dl.update(|p| {
            p.model = Some(m.id.into());
            p.file = Some(f.name.into());
            p.phase = "download".into();
            p.total = total;
        });
        let r = fetch(&client, &dl, m, f, dir, |have| {
            let done = before + have;
            let (rate, eta) = meter.tick(done, total);
            dl.update(|p| {
                p.done = done;
                p.rate_bps = rate;
                p.eta_s = eta;
            });
        })
        .await;
        if let Err(e) = r {
            let msg = format!("{e:#}");
            dl.update(|p| {
                p.running = false;
                if msg == CANCELLED {
                    p.phase = "cancelled".into();
                } else {
                    p.phase = "error".into();
                    p.error = Some(format!("{} · {}: {msg}", m.label, f.name));
                }
            });
            return;
        }
        before += f.bytes;
    }
    dl.update(|p| {
        p.running = false;
        p.phase = "done".into();
        p.done = total;
        p.eta_s = Some(0.0);
    });
}

struct Meter {
    t: Instant,
    last: u64,
    rate: f64,
}

impl Meter {
    fn new(start: u64) -> Meter {
        Meter { t: Instant::now(), last: start, rate: 0.0 }
    }

    fn tick(&mut self, done: u64, total: u64) -> (f64, Option<f64>) {
        let dt = self.t.elapsed().as_secs_f64();
        if dt >= 0.25 {
            let now = done.saturating_sub(self.last) as f64 / dt;
            self.rate = if self.rate == 0.0 { now } else { self.rate + (now - self.rate) * (dt / 2.0).min(1.0) };
            self.t = Instant::now();
            self.last = done;
        }
        let eta = (self.rate > 1.0).then(|| total.saturating_sub(done) as f64 / self.rate);
        (self.rate, eta)
    }
}

async fn fetch(client: &reqwest::Client, dl: &Downloads, m: &ModelSpec, f: &FileSpec, dir: &Path, mut on_bytes: impl FnMut(u64)) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("폴더 {}를 만들 수 없습니다", dir.display()))?;
    let part = part_path(&dir.join(f.name));
    let mut have = fs::metadata(&part).map_or(0, |m| m.len());
    if have > f.bytes {
        fs::remove_file(&part)?;
        have = 0;
    }
    let mut hasher = Sha256::new();
    if have > 0 {
        hash_prefix(&part, have, &mut hasher)?;
    }
    on_bytes(have);
    if have < f.bytes {
        let url = format!("https://huggingface.co/{}/resolve/{}/{}", m.repo, m.revision, f.name);
        let mut req = client.get(&url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = req.send().await.with_context(|| format!("{url} 에 연결할 수 없습니다"))?.error_for_status()?;
        let mut out = if have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            OpenOptions::new().append(true).open(&part)?
        } else {
            have = 0;
            hasher = Sha256::new();
            File::create(&part)?
        };
        while let Some(chunk) = resp.chunk().await? {
            if dl.cancelled() {
                out.flush()?;
                bail!(CANCELLED);
            }
            out.write_all(&chunk)?;
            hasher.update(&chunk);
            have += chunk.len() as u64;
            if have > f.bytes {
                drop(out);
                let _ = fs::remove_file(&part);
                bail!("받은 크기가 {} 바이트를 넘었습니다", f.bytes);
            }
            on_bytes(have);
        }
        out.sync_all()?;
    }
    if have != f.bytes {
        bail!("연결이 끊겼습니다 ({have}/{} 바이트). 다시 받으면 이어서 받습니다", f.bytes);
    }
    dl.update(|p| p.phase = "verify".into());
    let got = hex(&hasher.finalize());
    if got != f.sha256 {
        let _ = fs::remove_file(&part);
        bail!("SHA-256 이 다릅니다 (받음 {got}, 기대 {})", f.sha256);
    }
    let dest = dir.join(f.name);
    if dest.exists() {
        fs::remove_file(&dest)?;
    }
    fs::rename(&part, &dest).with_context(|| format!("{}의 이름을 바꿀 수 없습니다", part.display()))?;
    dl.update(|p| p.phase = "download".into());
    Ok(())
}

fn catalog(state: &AppState) -> Value {
    let models: Vec<Value> = MODELS
        .iter()
        .map(|m| {
            let dir = folder(m.id);
            let files: Vec<Value> = m
                .files
                .iter()
                .map(|f| {
                    let (have, done) = file_state(&dir, f);
                    json!({"name": f.name, "bytes": f.bytes, "have": have, "complete": done})
                })
                .collect();
            let total: u64 = m.files.iter().map(|f| f.bytes).sum();
            let have: u64 = m.files.iter().map(|f| file_state(&dir, f).0).sum();
            json!({
                "id": m.id, "label": m.label, "role_ko": m.role_ko, "repo": m.repo, "revision": m.revision, "license": m.license,
                "dir": dir.display().to_string(), "files": files, "bytes": total, "have": have, "present": present(m),
                "needed": needed(m, state.stereo()),
            })
        })
        .collect();
    let (backend, ordinal) = state.selected();
    json!({
        "required": true,
        "live_devices": state.live_devices(),
        "gpu_hint": gpu_hint(state),
        "models": models,
        "all_present": missing(state.stereo()).is_empty(),
        "cameras": if state.stereo() { 2 } else { 1 },
        "ready": state.models_ready(),
        "loading": state.loading(),
        "load_error": state.load_error(),
        "download": state.downloads.snapshot(),
        "device": {"backend": backend.name(), "ordinal": ordinal},
    })
}

#[tauri::command]
pub async fn models_catalog(state: App<'_>) -> Result<Value, String> {
    Ok(catalog(&state))
}

#[tauri::command]
pub async fn models_download(state: App<'_>, ids: Option<Vec<String>>) -> Result<Value, String> {
    let dl = Arc::clone(&state.downloads);
    let picked: Vec<&'static ModelSpec> = match ids.filter(|v| !v.is_empty()) {
        None => MODELS.iter().collect(),
        Some(v) => v.iter().map(|id| spec(id).ok_or_else(|| format!("모르는 모델 {id}"))).collect::<Result<_, _>>()?,
    };
    let todo: Vec<&'static ModelSpec> = picked.into_iter().filter(|m| m.files.iter().any(|f| !file_state(&folder(m.id), f).1)).collect();
    {
        let mut p = dl.progress.lock().map_err(|_| "다운로드 상태 잠금이 깨졌습니다")?;
        if p.running {
            return Err("이미 받는 중입니다".into());
        }
        *p = Progress { running: !todo.is_empty(), phase: if todo.is_empty() { "done".into() } else { "download".into() }, queue: todo.iter().map(|m| m.id.to_string()).collect(), ..Progress::default() };
    }
    dl.cancel.store(false, Ordering::Relaxed);
    if !todo.is_empty() {
        tauri::async_runtime::spawn(run(Arc::clone(&dl), todo));
    }
    Ok(catalog(&state))
}

#[tauri::command]
pub async fn models_cancel(state: App<'_>) -> Result<Value, String> {
    state.downloads.cancel.store(true, Ordering::Relaxed);
    Ok(catalog(&state))
}

#[tauri::command]
pub async fn models_delete(state: App<'_>, id: String) -> Result<Value, String> {
    let m = spec(&id).ok_or_else(|| format!("모르는 모델 {id}"))?;
    if state.downloads.running() {
        return Err("받는 중에는 지울 수 없습니다. 먼저 취소하세요".into());
    }
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        shared.set_models_ready(false);
        shared.with_world(|rt| {
            match m.id {
                "siglip2" => rt.use_oracle(),
                "ttm" => rt.attach_ttm(None),
                "tips" => rt.use_depth_stand_in(),
                _ => rt.use_region_tracker_clamped(),
            }
            Ok(())
        })?;
        if m.id == "siglip2" {
            shared.drop_vision();
        }
        let dir = folder(m.id);
        let mut removed = Vec::new();
        for f in m.files {
            for p in [dir.join(f.name), part_path(&dir.join(f.name))] {
                if p.exists() {
                    fs::remove_file(&p).map_err(|e| format!("{}를 지울 수 없습니다: {e}", p.display()))?;
                    removed.push(p.display().to_string());
                }
            }
        }
        if fs::read_dir(&dir).is_ok_and(|mut d| d.next().is_none()) {
            let _ = fs::remove_dir(&dir);
        }
        let mut v = catalog(&shared);
        v["removed"] = json!(removed);
        Ok(v)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn models_activate(state: App<'_>) -> Result<Value, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        let r = activate(&shared);
        shared.set_loading(None);
        shared.set_load_error(r.as_ref().err().cloned());
        r?;
        let mut v = catalog(&shared);
        v["status"] = shared.with_world(|rt| models_json(&shared, rt))?;
        Ok(v)
    })
    .await
    .map_err(|e| e.to_string())?
}

fn gpu_hint(state: &AppState) -> Option<String> {
    let (backend, _) = state.selected();
    if backend != noto_runtime::ml_bridge::Backend::Cpu {
        return None;
    }
    if noto_runtime::ml_bridge::device::GPU_ORDER.iter().any(|b| b.compiled()) {
        return Some("GPU 백엔드가 이 빌드에 있지만 열리지 않아 CPU에서 돕니다 (장치 칩의 설명 참고)".into());
    }
    if cfg!(target_os = "macos") {
        return Some("이 Mac에는 Metal GPU가 있지만 이 빌드는 CPU 전용입니다. ./dev_metal.sh(또는 ./build_metal.sh)로 실행하면 네 모델이 모두 GPU에서 돕니다".into());
    }
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
    let sys32 = Path::new(&windir).join("System32");
    if sys32.join("nvcuda.dll").is_file() {
        return Some("이 PC에는 NVIDIA GPU가 있지만 이 빌드는 CPU 전용입니다. dev_cuda.bat(또는 build_cuda.bat)로 실행하면 모델이 GPU에서 돕니다".into());
    }
    let rocm_present = sys32.join("amdhip64.dll").is_file()
        || Path::new(r"C:\Program Files\AMD\ROCm").is_dir();
    if rocm_present {
        return Some("이 PC에는 AMD ROCm 환경이 있지만 이 빌드는 CPU 전용입니다. dev_rocm.bat(또는 build_rocm.bat)로 실행하면 모델이 GPU에서 돕니다".into());
    }
    if sys32.join("vulkan-1.dll").is_file() {
        return Some("이 PC에는 Vulkan GPU가 있지만 이 빌드는 CPU 전용입니다. dev_vulkan.bat(또는 build_vulkan.bat)로 실행하면 모델이 GPU에서 돕니다".into());
    }
    None
}

fn activate(state: &Arc<AppState>) -> Result<(), String> {
    let stereo = state.with_world(|rt| Ok(rt.stereo()))?;
    state.set_stereo(stereo);
    let gone = missing(stereo);
    if !gone.is_empty() {
        return Err(format!("모델 파일이 없습니다: {}. 먼저 받으세요", gone.iter().map(|m| m.label).collect::<Vec<_>>().join(", ")));
    }
    state.prefer_gpu();
    let step = |id: &str, f: &mut dyn FnMut() -> Result<()>| -> Result<(), String> {
        state.set_loading(Some(id));
        f().map_err(|e| format!("{} 를 불러오지 못했습니다: {e:#}", spec(id).map_or(id, |m| m.label)))
    };
    let mut engine = None;
    step("siglip2", &mut || {
        engine = Some(state.vision_engine()?);
        Ok(())
    })?;
    let engine = engine.expect("loaded above");
    let device = state.device().map_err(|e| format!("연산 장치를 열 수 없습니다: {e:#}"))?;
    let live = state.with_world(|rt| {
        step("siglip2", &mut || rt.use_siglip(Arc::clone(&engine.model), Some(&bank_dir())))?;
        step("ttm", &mut || rt.load_ttm(&ttm_dir(), &device))?;
        // Two cameras: depth is SGBM on the stereo pair (CPU, no weights); TIPSv2 is not loaded at all.
        if stereo {
            rt.use_stereo();
        } else {
            step("tips", &mut || rt.use_tips(&tips_dir(), &device))?;
        }
        rt.use_stereo_fast_tick();
        let v = rt.api_models_status()?;
        let dev = |k: &str| v["devices"][k].as_str().unwrap_or("-").to_string();
        let depth = if stereo { "sgbm" } else { "tips" };
        Ok([("siglip2", noto_runtime::ml_bridge::device_label(engine.model.device())), ("ttm", dev("ttm")), (depth, dev("depth")), ("track", dev("track"))]
            .into_iter()
            .map(|(k, d)| (k.to_string(), d))
            .collect::<std::collections::BTreeMap<_, _>>())
    })?;
    state.set_live_devices(live);
    state.set_models_ready(true);
    Ok(())
}

pub fn not_ready_message(stereo: bool) -> String {
    let gone = missing(stereo);
    if gone.is_empty() {
        "필수 모델을 아직 불러오지 않았습니다. ④ 사전 · 모델에서 불러오기를 누르세요".into()
    } else {
        format!("필수 모델이 없어 시뮬레이터를 실행할 수 없습니다: {}. ④ 사전 · 모델에서 받으세요", gone.iter().map(|m| m.label).collect::<Vec<_>>().join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_three_pinned_models() {
        assert_eq!(MODELS.len(), 3);
        assert!(!MODELS.iter().any(|m| m.id == "sam"));
        for m in &MODELS {
            assert_eq!(m.revision.len(), 40, "{} must be pinned to a commit", m.id);
            for f in m.files {
                assert_eq!(f.sha256.len(), 64, "{} {}", m.id, f.name);
                assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            }
        }
    }

    #[test]
    fn file_state_reads_parts_and_finished_files() {
        let dir = std::env::temp_dir().join(format!("noto-models-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let f = FileSpec { name: "w.bin", bytes: 10, sha256: "" };
        assert_eq!(file_state(&dir, &f), (0, false));
        fs::write(part_path(&dir.join(f.name)), [0u8; 4]).unwrap();
        assert_eq!(file_state(&dir, &f), (4, false));
        fs::write(dir.join("w.bin"), [0u8; 10]).unwrap();
        assert_eq!(file_state(&dir, &f), (10, true));
        fs::write(dir.join("w.bin"), [0u8; 9]).unwrap();
        assert_eq!(file_state(&dir, &f), (4, false));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn file_state_rejects_mismatched_sha256() {
        let dir = std::env::temp_dir().join(format!("noto-models-sha-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let data = [0u8; 10];
        let p = dir.join("w.bin");
        fs::write(&p, data).unwrap();
        let expected = hex(&Sha256::digest(data));
        let meta = fs::metadata(&p).unwrap();
        let mt = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        assert!(hash_verified(&p, &expected, mt, meta.len()));
        let bad = FileSpec {
            name: "w.bin",
            bytes: 10,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        };
        assert_eq!(file_state(&dir, &bad), (0, false));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore]
    fn all_models_download_load_and_run() {
        let state = Arc::new(AppState::new());
        let todo: Vec<&'static ModelSpec> = missing(false);
        println!("[MODELS] missing before: {:?}", todo.iter().map(|m| m.id).collect::<Vec<_>>());
        if !todo.is_empty() {
            let dl = Arc::clone(&state.downloads);
            let rt = tokio::runtime::Runtime::new().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let watch = Arc::clone(&dl);
            let poll = std::thread::spawn(move || loop {
                let p = watch.snapshot();
                println!("[MODELS] {} {}/{} bytes · {:.1} MB/s · {:?}", p.phase, p.done, p.total, p.rate_bps / 1e6, p.file);
                if rx.recv_timeout(std::time::Duration::from_secs(2)).is_ok() {
                    break;
                }
            });
            dl.update(|p| p.running = true);
            rt.block_on(run(Arc::clone(&dl), todo));
            tx.send(()).unwrap();
            poll.join().unwrap();
            let p = dl.snapshot();
            assert_eq!(p.phase, "done", "{:?}", p.error);
        }
        assert!(missing(false).is_empty(), "still missing: {:?}", missing(false).iter().map(|m| m.id).collect::<Vec<_>>());
        let t0 = Instant::now();
        activate(&state).expect("all four models load");
        let live = state.live_devices();
        println!("[MODELS] loaded on {:?} in {:.1} s · {live:?}", state.selected(), t0.elapsed().as_secs_f32());
        assert!(state.models_ready());
        if noto_runtime::ml_bridge::device::select(None).backend != noto_runtime::ml_bridge::Backend::Cpu && std::env::var(noto_runtime::ml_bridge::device::DEVICE_ENV).map_or(true, |v| !v.eq_ignore_ascii_case("cpu")) {
            assert!(
                live.iter()
                    .filter(|(k, _)| k.as_str() != "track")
                    .all(|(_, d)| !d.starts_with("CPU") && d != "cpu"),
                "a model is not on the GPU: {live:?}"
            );
        }
        state
            .with_world(|rt| {
                assert!(rt.stand_ins().is_empty(), "stand-ins left: {:?}", rt.stand_ins());
                let d = rt.depth_view(80.0).map_err(|e| format!("{e:#}"))?;
                println!("[MODELS] depth {} on {} · {:.0} ms · {:?}", d.mode, d.device, d.infer_ms, d.metrics);
                assert!(d.mode.starts_with(lv_spatial::tips::MODE), "{}", d.mode);
                let s = rt.scan_now("test").map_err(|e| format!("{e:#}"))?;
                println!("[MODELS] scan {} on {} · {} tiles · {} judged", s.mode, s.device, s.tiles, s.judged.len());
                assert_ne!(s.mode, "oracle");
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn hash_prefix_matches_a_known_digest() {
        let dir = std::env::temp_dir().join(format!("noto-models-hash-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc");
        fs::write(&p, b"abcdef").unwrap();
        let mut h = Sha256::new();
        hash_prefix(&p, 3, &mut h).unwrap();
        assert_eq!(hex(&h.finalize()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        fs::remove_dir_all(&dir).unwrap();
    }
}
