use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

extern "C" {
    fn cl_system_ram() -> u64;
    fn cl_system_cores() -> i32;
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo { os: String, arch: String, ram_bytes: u64, cores: u32, models_dir: String }

#[derive(Serialize, Clone)]
pub struct ModelInfo { path: String, name: String, bytes: u64 }

/// One entry of the download table shared with the command line (catalog.json is generated from packages/cli/src/models.ts).
#[derive(Deserialize)]
struct Entry { id: String, name: String, repo: String, file: String, bytes: u64, sha256: String, description: String }

#[derive(Serialize)]
pub struct CatalogItem { id: String, name: String, bytes: u64, description: String }

#[derive(Serialize, Clone)]
struct Progress { id: String, done: u64, total: u64, state: &'static str, #[serde(skip_serializing_if = "Option::is_none")] error: Option<String> }

fn catalog() -> Vec<Entry> {
    serde_json::from_str(include_str!("../catalog.json")).unwrap_or_default()
}

/// Where the app keeps models: $CLUTOR_MODELS if set, else a `models` folder in the app's data directory (private to the app on a phone).
fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = match std::env::var_os("CLUTOR_MODELS") {
        Some(d) => PathBuf::from(d),
        None => app.path().app_data_dir().map_err(|e| e.to_string())?.join("models"),
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir)
}

#[tauri::command]
pub fn system_info(app: AppHandle) -> Result<SystemInfo, String> {
    Ok(SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        ram_bytes: unsafe { cl_system_ram() },
        cores: unsafe { cl_system_cores() }.max(1) as u32,
        models_dir: models_dir(&app)?.display().to_string(),
    })
}

fn is_model_file(name: &str) -> bool {
    let n = name.to_lowercase();
    if !n.ends_with(".gguf") || n.starts_with("mmproj") { return false; }
    // later parts of a split model are not models you can open on their own
    match n.rsplit_once("-of-") {
        Some((head, _)) => head.rsplit('-').next().and_then(|p| p.parse::<u32>().ok()).map_or(true, |p| p <= 1),
        None => true,
    }
}

fn walk(dir: &Path, depth: u32, out: &mut Vec<ModelInfo>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') { continue; }
        let Ok(meta) = e.metadata() else { continue };
        if meta.is_dir() { if depth > 0 { walk(&p, depth - 1, out); } }
        else if meta.is_file() && is_model_file(&name) {
            out.push(ModelInfo { path: p.display().to_string(), name: name.trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string(), bytes: meta.len() });
        }
    }
}

#[tauri::command]
pub fn list_models(app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    let mut dirs = vec![models_dir(&app)?];
    if let Ok(home) = app.path().home_dir() { dirs.push(home.join(".clutor").join("models")); }   // the command line's folder, on a computer
    let mut found = Vec::new();
    for d in dirs { walk(&d, 4, &mut found); }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found.dedup_by(|a, b| a.path == b.path);
    Ok(found)
}

#[tauri::command]
pub fn model_catalog() -> Vec<CatalogItem> {
    catalog().into_iter().map(|e| CatalogItem { id: e.id, name: e.name, bytes: e.bytes, description: e.description }).collect()
}

#[tauri::command]
pub fn reveal_models(app: AppHandle) -> Result<(), String> {
    let dir = models_dir(&app)?;
    #[cfg(target_os = "macos")] let cmd = "open";
    #[cfg(target_os = "windows")] let cmd = "explorer";
    #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))] let cmd = "xdg-open";
    #[cfg(target_os = "android")] { let _ = dir; return Err("The models folder is private to the app on this device: use Get a model to add one.".to_string()); }
    #[cfg(not(target_os = "android"))]
    { std::process::Command::new(cmd).arg(&dir).spawn().map(|_| ()).map_err(|e| e.to_string()) }
}

fn emit(app: &AppHandle, p: Progress) { let _ = app.emit("download-progress", p); }

/// Downloads a catalog model into the models folder through a `.part` file (an interrupted download resumes), then checks the whole file against the published SHA-256.
#[tauri::command]
pub async fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    let entry = catalog().into_iter().find(|e| e.id == id).ok_or_else(|| "unknown model".to_string())?;
    let dir = models_dir(&app)?;
    let dest = dir.join(&entry.file);
    let part = dir.join(format!("{}.part", entry.file));
    let fail = |msg: String| -> String { emit(&app, Progress { id: id.clone(), done: 0, total: entry.bytes, state: "error", error: Some(msg.clone()) }); msg };
    if let Ok(m) = std::fs::metadata(&dest) { if m.len() == entry.bytes { emit(&app, Progress { id: id.clone(), done: entry.bytes, total: entry.bytes, state: "done", error: None }); return Ok(()); } }

    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > entry.bytes { let _ = std::fs::remove_file(&part); have = 0; }
    let url = format!("https://huggingface.co/{}/resolve/main/{}", entry.repo, entry.file);
    let client = reqwest::Client::builder().user_agent("clutor-app").build().map_err(|e| fail(e.to_string()))?;
    let mut req = client.get(&url);
    if have > 0 { req = req.header("Range", format!("bytes={have}-")); }
    let resp = req.send().await.map_err(|e| fail(format!("could not reach the download server: {e}")))?;
    if !resp.status().is_success() { return Err(fail(format!("the download server answered {}", resp.status()))); }
    if have > 0 && resp.status().as_u16() != 206 { have = 0; }   // the server ignored the range: start again
    let mut file = tokio::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(&part).await.map_err(|e| fail(e.to_string()))?;
    let mut done = have;
    let mut last = std::time::Instant::now();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| fail(format!("the download was interrupted: {e}")))?;
        file.write_all(&chunk).await.map_err(|e| fail(e.to_string()))?;
        done += chunk.len() as u64;
        if last.elapsed().as_millis() > 250 { last = std::time::Instant::now(); emit(&app, Progress { id: id.clone(), done, total: entry.bytes, state: "downloading", error: None }); }
    }
    file.flush().await.map_err(|e| fail(e.to_string()))?;
    drop(file);
    if done != entry.bytes { return Err(fail(format!("the download ended at {done} of {} bytes; try again to resume", entry.bytes))); }

    emit(&app, Progress { id: id.clone(), done, total: entry.bytes, state: "verifying", error: None });
    let mut f = tokio::fs::File::open(&part).await.map_err(|e| fail(e.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await.map_err(|e| fail(e.to_string()))?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let sum: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if sum != entry.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(fail("the downloaded file does not match the publisher's checksum, so it was deleted".to_string()));
    }
    std::fs::rename(&part, &dest).map_err(|e| fail(e.to_string()))?;
    emit(&app, Progress { id, done, total: entry.bytes, state: "done", error: None });
    Ok(())
}
