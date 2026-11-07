//! The commands the app's interface uses to run remote control: turn it on and off, show a pairing code, list and revoke phones, and pass messages to and from phones.

use super::host::{Events, Host};
use super::store::{Device, Store};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};

const PORT: u16 = 8765;
const MAX_PHONES: usize = 4;
const OFFER_SECONDS: u64 = 120;
const LOG_LIMIT: u64 = 1_000_000;

#[derive(Default)]
pub struct RemoteState { store: Mutex<Option<Arc<Store>>>, host: tokio::sync::Mutex<Option<Arc<Host>>>, awake: Mutex<Option<keepawake::KeepAwake>> }

struct TauriEvents(AppHandle);
impl Events for TauriEvents {
    fn opened(&self, client: u64, device: &Device) { let _ = self.0.emit("remote-open", serde_json::json!({ "client": client, "device": device })); }
    fn message(&self, client: u64, text: String) { let _ = self.0.emit("remote-msg", serde_json::json!({ "client": client, "text": text })); }
    fn closed(&self, client: u64) { let _ = self.0.emit("remote-close", serde_json::json!({ "client": client })); }
}

fn store(app: &AppHandle, state: &RemoteState) -> Result<Arc<Store>, String> {
    let mut g = state.store.lock().unwrap();
    if let Some(s) = g.as_ref() { return Ok(s.clone()); }
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let s = Arc::new(Store::open(dir.join("remote.json"))?);
    *g = Some(s.clone());
    Ok(s)
}

/// This computer's addresses on the local network (not loopback, not link-local), as `ip:port`.
fn lan_addresses(port: u16) -> Vec<String> {
    if_addrs::get_if_addrs().unwrap_or_default().into_iter()
        .filter(|i| !i.is_loopback())
        .filter_map(|i| match i.ip() { std::net::IpAddr::V4(v) if !v.is_link_local() => Some(format!("{v}:{port}")), _ => None })
        .collect()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status { on: bool, port: Option<u16>, addresses: Vec<String>, host_id: String, devices: Vec<Device>, clients: usize, offer_open: bool, awake: bool }

async fn status_of(app: &AppHandle, state: &RemoteState) -> Result<Status, String> {
    let store = store(app, state)?;
    let host = state.host.lock().await.clone();
    let port = host.as_ref().map(|h| h.addr().port());
    Ok(Status { on: host.is_some(), port, addresses: port.map(lan_addresses).unwrap_or_default(), host_id: store.host_id(), devices: store.devices(), clients: host.as_ref().map_or(0, |h| h.client_count()), offer_open: store.offer_open(), awake: state.awake.lock().unwrap().is_some() })
}

/// Turns remote control on. It listens on this computer's network, and only phones that were paired (or hold the pairing code on screen) get past the handshake.
#[tauri::command]
pub async fn remote_start(app: AppHandle, state: State<'_, RemoteState>, keep_awake: bool) -> Result<Status, String> {
    let st = store(&app, &state)?;
    {
        let mut g = state.host.lock().await;
        if g.is_none() {
            let events: Arc<dyn Events> = Arc::new(TauriEvents(app.clone()));
            let host = match Host::start(SocketAddr::from(([0, 0, 0, 0], PORT)), st.clone(), events.clone(), MAX_PHONES).await {
                Ok(h) => h,
                Err(_) => Host::start(SocketAddr::from(([0, 0, 0, 0], 0)), st.clone(), events, MAX_PHONES).await?,
            };
            *g = Some(Arc::new(host));
        }
    }
    set_awake(&state, keep_awake);
    status_of(&app, &state).await
}

fn set_awake(state: &RemoteState, on: bool) {
    let mut g = state.awake.lock().unwrap();
    if !on { *g = None; return; }
    if g.is_none() { *g = keepawake::Builder::default().idle(true).sleep(true).reason("Clutor remote control is on").app_name("Clutor").app_reverse_domain("app.clutor.desktop").create().ok(); }
}

#[tauri::command]
pub async fn remote_stop(app: AppHandle, state: State<'_, RemoteState>) -> Result<Status, String> {
    if let Some(h) = state.host.lock().await.take() { h.stop(); }
    store(&app, &state)?.cancel_offer();
    set_awake(&state, false);
    status_of(&app, &state).await
}

#[tauri::command]
pub async fn remote_set_awake(app: AppHandle, state: State<'_, RemoteState>, on: bool) -> Result<Status, String> {
    set_awake(&state, on && state.host.lock().await.is_some());
    status_of(&app, &state).await
}

#[tauri::command]
pub async fn remote_status(app: AppHandle, state: State<'_, RemoteState>) -> Result<Status, String> { status_of(&app, &state).await }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Offer { /** The text to turn into a QR code. */ qr: String, expires_in: u64, addresses: Vec<String> }

/// Shows a pairing code: a one-time secret, this computer's public key and its addresses. It works once and expires after two minutes.
#[tauri::command]
pub async fn remote_pair_start(app: AppHandle, state: State<'_, RemoteState>) -> Result<Offer, String> {
    let port = state.host.lock().await.as_ref().map(|h| h.addr().port()).ok_or("turn remote access on first")?;
    let st = store(&app, &state)?;
    let secret = st.start_offer(Duration::from_secs(OFFER_SECONDS));
    let addresses = lan_addresses(port);
    let name = hostname::get().ok().and_then(|h| h.into_string().ok()).unwrap_or_else(|| "This computer".into());
    let qr = serde_json::json!({ "v": 1, "name": name, "key": B64.encode(st.public_key()), "id": st.host_id(), "addrs": addresses, "secret": B64.encode(secret) }).to_string();
    Ok(Offer { qr, expires_in: OFFER_SECONDS, addresses })
}

#[tauri::command]
pub fn remote_pair_cancel(app: AppHandle, state: State<'_, RemoteState>) -> Result<(), String> { store(&app, &state)?.cancel_offer(); Ok(()) }

/// Removes a phone: it can no longer connect, and is disconnected now if it is connected.
#[tauri::command]
pub async fn remote_revoke(app: AppHandle, state: State<'_, RemoteState>, id: String) -> Result<Status, String> {
    store(&app, &state)?.revoke(&id);
    if let Some(h) = state.host.lock().await.as_ref() { h.kick_device(&id); }
    status_of(&app, &state).await
}

#[tauri::command]
pub async fn remote_send(state: State<'_, RemoteState>, client: u64, text: String) -> Result<bool, String> { Ok(state.host.lock().await.as_ref().is_some_and(|h| h.send(client, text))) }

#[tauri::command]
pub async fn remote_broadcast(state: State<'_, RemoteState>, text: String) -> Result<(), String> { if let Some(h) = state.host.lock().await.as_ref() { h.broadcast(&text); } Ok(()) }

#[tauri::command]
pub async fn remote_kick(state: State<'_, RemoteState>, client: u64) -> Result<(), String> { if let Some(h) = state.host.lock().await.as_ref() { h.kick(client); } Ok(()) }

fn log_path(app: &AppHandle) -> Result<std::path::PathBuf, String> { Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("remote-log.jsonl")) }

/// Adds one line to the remote activity log (kept to about 1 MB: the older half is dropped when it fills).
#[tauri::command]
pub fn remote_log(app: AppHandle, line: String) -> Result<(), String> {
    use std::io::Write;
    let path = log_path(&app)?;
    if let Some(d) = path.parent() { std::fs::create_dir_all(d).map_err(|e| e.to_string())?; }
    if std::fs::metadata(&path).map(|m| m.len() > LOG_LIMIT).unwrap_or(false) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let keep: Vec<&str> = text.lines().skip(text.lines().count() / 2).collect();
        std::fs::write(&path, keep.join("\n") + "\n").map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| e.to_string())?;
    writeln!(f, "{}", line.replace('\n', " ")).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn remote_log_read(app: AppHandle, max: usize) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(log_path(&app)?).unwrap_or_default();
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    Ok(lines[lines.len().saturating_sub(max.min(1000))..].to_vec())
}
