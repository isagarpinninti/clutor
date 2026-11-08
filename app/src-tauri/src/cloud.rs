//! Cloud models: the app sends a chat request to a provider the user set up and streams the answer back to the interface. The request is made here, not in the web view, for two reasons:
//! the API key never reaches the page (it is kept in a file only this app can read and added to the request here), and a web view would block most providers' addresses (CORS).
//! Nothing is sent unless the user chose a cloud model; the local engine never uses this.

use futures_util::StreamExt;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::task::AbortHandle;

#[derive(Default)]
pub struct CloudState { tasks: Mutex<HashMap<String, AbortHandle>> }

#[derive(Serialize, Clone)]
struct Chunk { id: String, text: String }
#[derive(Serialize, Clone)]
struct End { id: String, status: u16, error: Option<String> }

fn keys_path(app: &AppHandle) -> Result<PathBuf, String> { Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("cloud-keys.json")) }

fn read_keys(app: &AppHandle) -> HashMap<String, String> {
    keys_path(app).ok().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn write_keys(app: &AppHandle, keys: &HashMap<String, String>) -> Result<(), String> {
    let path = keys_path(app)?;
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
    let text = serde_json::to_string(keys).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path).map_err(|e| e.to_string())?;
        f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    { std::fs::write(&path, text).map_err(|e| e.to_string())?; }
    Ok(())
}

fn valid_provider(p: &str) -> bool { !p.is_empty() && p.len() <= 40 && p.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') }

/// Saves the key for a provider. It is never sent back to the page.
#[tauri::command]
pub fn cloud_key_set(app: AppHandle, provider: String, key: String) -> Result<(), String> {
    if !valid_provider(&provider) { return Err("that is not a valid provider name".into()); }
    let key = key.trim().to_string();
    if key.is_empty() || key.len() > 500 || key.chars().any(|c| c.is_control()) { return Err("that does not look like an API key".into()); }
    let mut keys = read_keys(&app);
    keys.insert(provider, key);
    write_keys(&app, &keys)
}

#[tauri::command]
pub fn cloud_key_has(app: AppHandle, provider: String) -> bool { read_keys(&app).contains_key(&provider) }

#[tauri::command]
pub fn cloud_key_clear(app: AppHandle, provider: String) -> Result<(), String> {
    let mut keys = read_keys(&app);
    keys.remove(&provider);
    write_keys(&app, &keys)
}

async fn run(app: &AppHandle, id: &str, url: url::Url, auth: &str, key: Option<String>, body: String) -> Result<u16, String> {
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).connect_timeout(Duration::from_secs(30)).build().map_err(|e| e.to_string())?;
    let mut req = client.post(url).header("content-type", "application/json").header("accept", "text/event-stream").body(body);
    if let Some(k) = key {
        req = match auth {
            "anthropic" => req.header("x-api-key", k).header("anthropic-version", "2023-06-01"),
            _ => req.header("authorization", format!("Bearer {k}")),
        };
    }
    let res = req.send().await.map_err(|e| format!("could not reach the provider: {e}"))?;
    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        let text: String = text.chars().take(1500).collect();
        return Err(format!("{}: {}", status.as_u16(), text));
    }
    let mut stream = res.bytes_stream();
    let mut carry: Vec<u8> = Vec::new();
    loop {
        // A long silence means the connection is dead: reasoning models can think for a while, so the wait is generous.
        let next = tokio::time::timeout(Duration::from_secs(300), stream.next()).await.map_err(|_| "the provider stopped answering".to_string())?;
        let Some(piece) = next else { break };
        let bytes = piece.map_err(|e| format!("the connection broke: {e}"))?;
        carry.extend_from_slice(&bytes);
        // Send whole characters only: a chunk can end in the middle of one.
        let valid = match std::str::from_utf8(&carry) { Ok(_) => carry.len(), Err(e) => e.valid_up_to() };
        if valid > 0 {
            let text = String::from_utf8_lossy(&carry[..valid]).into_owned();
            carry.drain(..valid);
            let _ = app.emit("cloud-chunk", Chunk { id: id.to_string(), text });
        }
    }
    Ok(status.as_u16())
}

/// Starts a streaming request. The answer arrives as `cloud-chunk` events (the raw text of the response) and ends with one `cloud-end`.
#[tauri::command]
pub async fn cloud_request(app: AppHandle, state: State<'_, CloudState>, id: String, provider: String, url: String, auth: String, body: String) -> Result<(), String> {
    let parsed = url::Url::parse(&url).map_err(|_| "that is not a valid address".to_string())?;
    if !matches!(parsed.scheme(), "http" | "https") { return Err("only http and https addresses are used".into()); }
    if !valid_provider(&provider) { return Err("that is not a valid provider name".into()); }
    if body.len() > 20 * 1024 * 1024 { return Err("the request is too large".into()); }
    let key = if auth == "none" { None } else { Some(read_keys(&app).get(&provider).cloned().ok_or("no API key is saved for this provider")?) };
    let (app2, id2) = (app.clone(), id.clone());
    let task = tokio::spawn(async move {
        let result = run(&app2, &id2, parsed, &auth, key, body).await;
        let end = match result { Ok(status) => End { id: id2.clone(), status, error: None }, Err(e) => End { id: id2.clone(), status: 0, error: Some(e) } };
        let _ = app2.emit("cloud-end", end);
        if let Ok(mut t) = app2.state::<CloudState>().tasks.lock() { t.remove(&id2); }
    });
    if let Ok(mut t) = state.tasks.lock() { t.insert(id, task.abort_handle()); }
    Ok(())
}

#[tauri::command]
pub fn cloud_cancel(app: AppHandle, state: State<'_, CloudState>, id: String) {
    let handle = state.tasks.lock().ok().and_then(|mut t| t.remove(&id));
    if let Some(h) = handle { h.abort(); }
    let _ = app.emit("cloud-end", End { id, status: 0, error: Some("cancelled".into()) });
}
