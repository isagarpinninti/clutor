//! The desktop side of remote control: a WebSocket server that runs the secure handshake with each phone and passes its messages to the app (and the app's messages to it).
//! It decides nothing about what a phone may do; that is the app's job. It only decides who gets in (paired phones, or one holding the open pairing offer), and it limits how
//! many connect at once and how fast a stranger can keep trying.

use super::secure::{respond, Link, Secure};
use super::store::{Device, Store};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio_tungstenite::tungstenite::{protocol::WebSocketConfig, Message};
use tokio_tungstenite::WebSocketStream;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// A WebSocket as a [`Link`]: binary frames in, binary frames out.
pub struct WsLink<S>(pub WebSocketStream<S>);

impl<S: AsyncRead + AsyncWrite + Unpin + Send> Link for WsLink<S> {
    async fn send(&mut self, frame: Vec<u8>) -> Result<(), String> { self.0.send(Message::Binary(frame)).await.map_err(|e| e.to_string()) }
    async fn recv(&mut self) -> Result<Vec<u8>, String> {
        loop {
            match self.0.next().await {
                Some(Ok(Message::Binary(b))) => return Ok(b),
                Some(Ok(Message::Close(_))) | None => return Err("closed".into()),
                Some(Ok(_)) => continue,
                Some(Err(e)) => return Err(e.to_string()),
            }
        }
    }
}

/// What the app wants to know about.
pub trait Events: Send + Sync + 'static {
    fn opened(&self, client: u64, device: &Device);
    fn message(&self, client: u64, text: String);
    fn closed(&self, client: u64);
}

enum Out { Text(String), Kick }
struct ClientEntry { out: mpsc::UnboundedSender<Out>, device_id: String }

#[derive(Default)]
struct Shared { clients: Mutex<HashMap<u64, ClientEntry>>, fails: Mutex<HashMap<IpAddr, (u32, Instant)>>, next: Mutex<u64> }

pub struct Host { addr: SocketAddr, shared: Arc<Shared>, stop: Arc<Notify> }

impl Host {
    pub async fn start(bind: SocketAddr, store: Arc<Store>, events: Arc<dyn Events>, max_clients: usize) -> Result<Host, String> {
        let listener = TcpListener::bind(bind).await.map_err(|e| format!("cannot listen on {bind}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;
        let shared = Arc::new(Shared::default());
        let stop = Arc::new(Notify::new());
        let (sh, st) = (shared.clone(), stop.clone());
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = st.notified() => break,
                    accepted = listener.accept() => {
                        let Ok((stream, peer)) = accepted else { continue };
                        if sh.clients.lock().unwrap().len() >= max_clients || locked_out(&sh, peer.ip()) { drop(stream); continue; }
                        let (sh, store, events, st) = (sh.clone(), store.clone(), events.clone(), st.clone());
                        tokio::spawn(async move { serve(stream, peer.ip(), sh, store, events, st).await });
                    }
                }
            }
        });
        Ok(Host { addr, shared, stop })
    }

    pub fn addr(&self) -> SocketAddr { self.addr }
    pub fn client_count(&self) -> usize { self.shared.clients.lock().unwrap().len() }
    pub fn send(&self, client: u64, text: String) -> bool { self.shared.clients.lock().unwrap().get(&client).is_some_and(|c| c.out.send(Out::Text(text)).is_ok()) }
    pub fn broadcast(&self, text: &str) { for c in self.shared.clients.lock().unwrap().values() { let _ = c.out.send(Out::Text(text.to_string())); } }
    pub fn kick(&self, client: u64) { if let Some(c) = self.shared.clients.lock().unwrap().get(&client) { let _ = c.out.send(Out::Kick); } }
    /// Drops every connection of one paired phone (after it was revoked).
    pub fn kick_device(&self, device_id: &str) { for c in self.shared.clients.lock().unwrap().values().filter(|c| c.device_id == device_id) { let _ = c.out.send(Out::Kick); } }
    pub fn kick_all(&self) { for c in self.shared.clients.lock().unwrap().values() { let _ = c.out.send(Out::Kick); } }
    pub fn stop(&self) { self.stop.notify_waiters(); self.kick_all(); }
}

/// Five failed handshakes from one address in a minute lock it out for the rest of that minute.
fn locked_out(sh: &Shared, ip: IpAddr) -> bool {
    let mut f = sh.fails.lock().unwrap();
    f.retain(|_, (_, until)| Instant::now() < *until);
    f.get(&ip).is_some_and(|(n, _)| *n >= 5)
}
fn record_failure(sh: &Shared, ip: IpAddr) {
    let mut f = sh.fails.lock().unwrap();
    let e = f.entry(ip).or_insert((0, Instant::now() + Duration::from_secs(60)));
    e.0 += 1;
}

async fn serve(stream: TcpStream, ip: IpAddr, sh: Arc<Shared>, store: Arc<Store>, events: Arc<dyn Events>, stop: Arc<Notify>) {
    let config = WebSocketConfig { max_message_size: Some(1 << 20), max_frame_size: Some(1 << 17), ..Default::default() };
    let Ok(Ok(ws)) = tokio::time::timeout(HANDSHAKE_TIMEOUT, tokio_tungstenite::accept_async_with_config(stream, Some(config))).await else { return record_failure(&sh, ip) };
    let mut link = WsLink(ws);
    let identity = store.identity();
    let mut secure = match tokio::time::timeout(HANDSHAKE_TIMEOUT, respond(&mut link, &identity, &*store)).await {
        Ok(Ok(s)) => s,
        _ => return record_failure(&sh, ip),
    };
    // The first message is the phone introducing itself; a phone being paired is only remembered once it has.
    let hello = match tokio::time::timeout(HELLO_TIMEOUT, next_message(&mut link, &mut secure)).await {
        Ok(Ok(m)) => m,
        _ => return,
    };
    let name = serde_json::from_slice::<serde_json::Value>(&hello).ok().filter(|v| v["t"] == "hello").and_then(|v| v["name"].as_str().map(str::to_string));
    let Some(name) = name else { return };
    let device = if secure.paired_now { store.add_device(&secure.remote, &name) } else { store.touch(&secure.remote, Some(&name)).ok_or_else(|| "no longer paired".to_string()) };
    let Ok(device) = device else { return };

    let (tx, mut rx) = mpsc::unbounded_channel();
    let id = { let mut n = sh.next.lock().unwrap(); *n += 1; *n };
    sh.clients.lock().unwrap().insert(id, ClientEntry { out: tx, device_id: device.id.clone() });
    events.opened(id, &device);

    loop {
        tokio::select! {
            _ = stop.notified() => break,
            out = rx.recv() => match out {
                Some(Out::Text(t)) => {
                    let Ok(frames) = secure.seal(t.as_bytes()) else { break };
                    let mut ok = true;
                    for f in frames { if link.send(f).await.is_err() { ok = false; break; } }
                    if !ok { break; }
                }
                Some(Out::Kick) | None => break,
            },
            frame = tokio::time::timeout(IDLE_TIMEOUT, link.recv()) => match frame {
                Ok(Ok(f)) => match secure.open(&f) {
                    Ok(Some(m)) => { if let Ok(text) = String::from_utf8(m) { events.message(id, text); } }
                    Ok(None) => {}
                    Err(_) => break,
                },
                _ => break,
            },
        }
    }
    sh.clients.lock().unwrap().remove(&id);
    events.closed(id);
    let _ = link.0.close(None).await;
}

async fn next_message<L: Link>(link: &mut L, secure: &mut Secure) -> Result<Vec<u8>, String> {
    loop { let f = link.recv().await?; if let Some(m) = secure.open(&f)? { return Ok(m); } }
}

#[cfg(test)]
mod tests {
    use super::super::client::connect;
    use super::super::secure::{Auth, Identity};
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Rec { opened: StdMutex<Vec<(u64, Device)>>, messages: StdMutex<Vec<(u64, String)>>, closed: StdMutex<Vec<u64>> }
    impl Events for Rec {
        fn opened(&self, c: u64, d: &Device) { self.opened.lock().unwrap().push((c, d.clone())); }
        fn message(&self, c: u64, t: String) { self.messages.lock().unwrap().push((c, t)); }
        fn closed(&self, c: u64) { self.closed.lock().unwrap().push(c); }
    }
    async fn until(f: impl Fn() -> bool) { for _ in 0..400 { if f() { return; } tokio::time::sleep(Duration::from_millis(10)).await; } panic!("timed out"); }
    async fn start(max: usize) -> (Host, Arc<Store>, Arc<Rec>) {
        let store = Arc::new(Store::memory());
        let rec = Arc::new(Rec::default());
        let host = Host::start("127.0.0.1:0".parse().unwrap(), store.clone(), rec.clone(), max).await.unwrap();
        (host, store, rec)
    }

    #[tokio::test]
    async fn a_phone_pairs_chats_reconnects_and_is_cut_off_when_revoked() {
        let (host, store, rec) = start(4).await;
        let url = format!("ws://{}", host.addr());
        let phone = Identity::generate().unwrap();
        let secret = store.start_offer(Duration::from_secs(60));

        let mut c = connect(&url, &phone, Auth::Pair { secret }, &store.public_key(), "Pixel 8").await.unwrap();
        until(|| rec.opened.lock().unwrap().len() == 1).await;
        let (id, device) = rec.opened.lock().unwrap()[0].clone();
        assert_eq!(device.name, "Pixel 8");
        assert_eq!(store.devices().len(), 1);

        c.send(r#"{"t":"ping"}"#).await.unwrap();
        until(|| rec.messages.lock().unwrap().iter().any(|(_, t)| t.contains("ping"))).await;
        assert!(host.send(id, r#"{"t":"pong"}"#.into()));
        assert_eq!(c.recv().await.unwrap(), r#"{"t":"pong"}"#);
        drop(c);
        until(|| rec.closed.lock().unwrap().len() == 1).await;

        // Later: no pairing code needed, because the phone is known.
        let mut again = connect(&url, &phone, Auth::Known, &store.public_key(), "Pixel 8").await.unwrap();
        until(|| rec.opened.lock().unwrap().len() == 2).await;
        store.revoke(&device.id);
        host.kick_device(&device.id);
        assert!(again.recv().await.is_err());
        assert!(connect(&url, &phone, Auth::Known, &store.public_key(), "Pixel 8").await.is_err());
        host.stop();
    }

    #[tokio::test]
    async fn a_stranger_without_the_code_gets_nowhere_and_is_locked_out_after_repeated_tries() {
        let (host, store, rec) = start(4).await;
        let url = format!("ws://{}", host.addr());
        let stranger = Identity::generate().unwrap();
        store.start_offer(Duration::from_secs(60));
        for _ in 0..5 { assert!(connect(&url, &stranger, Auth::Pair { secret: [0u8; 32] }, &store.public_key(), "x").await.is_err()); }
        assert!(rec.opened.lock().unwrap().is_empty());
        assert!(store.devices().is_empty());
        // The sixth try is not even answered.
        assert!(connect(&url, &stranger, Auth::Known, &store.public_key(), "x").await.is_err());
        host.stop();
    }

    #[tokio::test]
    async fn only_a_limited_number_of_phones_can_be_connected_at_once() {
        let (host, store, rec) = start(1).await;
        let url = format!("ws://{}", host.addr());
        let (a, b) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let s1 = store.start_offer(Duration::from_secs(60));
        let _first = connect(&url, &a, Auth::Pair { secret: s1 }, &store.public_key(), "A").await.unwrap();
        until(|| rec.opened.lock().unwrap().len() == 1).await;
        let s2 = store.start_offer(Duration::from_secs(60));
        assert!(connect(&url, &b, Auth::Pair { secret: s2 }, &store.public_key(), "B").await.is_err());
        host.stop();
    }
}
