//! The phone's side of the connection (and the test client): connect to the desktop, run the secure handshake, introduce itself.

use super::host::WsLink;
use super::secure::{initiate, Auth, Identity, Link, Secure};
use futures_util::StreamExt;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

pub struct Connection { link: WsLink<MaybeTlsStream<TcpStream>>, secure: Secure }

/// Connects to `ws://host:port`, authenticates, and says hello with this phone's name. `expect` is the desktop's public key from the QR code or from when it was paired.
pub async fn connect(url: &str, id: &Identity, auth: Auth, expect: &[u8], name: &str) -> Result<Connection, String> {
    let (ws, _): (WebSocketStream<MaybeTlsStream<TcpStream>>, _) = tokio::time::timeout(Duration::from_secs(10), tokio_tungstenite::connect_async(url)).await.map_err(|_| "could not reach the computer".to_string())?.map_err(|e| e.to_string())?;
    let mut link = WsLink(ws);
    let secure = tokio::time::timeout(Duration::from_secs(10), initiate(&mut link, id, auth, expect)).await.map_err(|_| "the computer did not answer".to_string())??;
    let mut c = Connection { link, secure };
    c.send(&serde_json::json!({ "t": "hello", "name": name }).to_string()).await?;
    Ok(c)
}

impl Connection {
    pub async fn send(&mut self, text: &str) -> Result<(), String> { for f in self.secure.seal(text.as_bytes())? { self.link.send(f).await?; } Ok(()) }
    pub async fn recv(&mut self) -> Result<String, String> {
        loop {
            let f = self.link.recv().await?;
            if let Some(m) = self.secure.open(&f)? { return String::from_utf8(m).map_err(|e| e.to_string()); }
        }
    }
    pub fn remote_key(&self) -> &[u8] { &self.secure.remote }
}

#[allow(dead_code)]
fn _assert_stream<S: StreamExt>() {}
