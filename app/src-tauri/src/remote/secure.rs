//! The encrypted channel between a phone and the desktop app: a Noise handshake (X25519, ChaCha20-Poly1305, BLAKE2s) over any link that moves binary frames, then
//! authenticated, encrypted messages. Nothing here trusts the link: a relay or a network attacker sees only ciphertext, and cannot alter, replay or reorder it unnoticed.
//!
//! Two ways in:
//! * **Pairing** (first time): the desktop shows a one-time secret and its public key in a QR code. The phone proves it saw the code by mixing the secret into the handshake
//!   (`XXpsk3`), and checks that the desktop's key is the one in the code. Both sides then remember each other's public key.
//! * **Known device** (every later time): a plain `XX` handshake; the desktop accepts only keys it has paired, the phone only the key it paired with.

use snow::{Builder, TransportState};

const PARAMS_KNOWN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
const PARAMS_PAIR: &str = "Noise_XXpsk3_25519_ChaChaPoly_BLAKE2s";
const PROLOGUE: &[u8] = b"clutor-remote-v1";
const MODE_KNOWN: u8 = 0;
const MODE_PAIR: u8 = 1;
const CHUNK: usize = 60_000;
const MAX_MESSAGE: usize = 4 * 1024 * 1024;
const READY: &[u8] = b"clutor-ready";

pub type Key = Vec<u8>;

#[derive(Clone)]
pub struct Identity { pub private: Key, pub public: Key }

impl Identity {
    pub fn generate() -> Result<Self, String> {
        let kp = Builder::new(PARAMS_KNOWN.parse().map_err(|e| format!("{e}"))?).generate_keypair().map_err(|e| e.to_string())?;
        Ok(Identity { private: kp.private, public: kp.public })
    }
}

/// Something that moves binary frames in order: a WebSocket to the desktop, a connection through a relay, or an in-memory pipe in tests.
pub trait Link {
    fn send(&mut self, frame: Vec<u8>) -> impl std::future::Future<Output = Result<(), String>> + Send;
    fn recv(&mut self) -> impl std::future::Future<Output = Result<Vec<u8>, String>> + Send;
}

/// How the phone introduces itself.
pub enum Auth { Pair { secret: [u8; 32] }, Known }

/// What the desktop knows: who is paired, and the one-time secret of the pairing offer on screen (if any).
pub trait Policy: Sync {
    fn is_paired(&self, public: &[u8]) -> bool;
    /// Takes the secret of the open pairing offer. It can be taken once: a second attempt, or an expired offer, gets nothing.
    fn take_pairing_secret(&self) -> Option<[u8; 32]>;
}

impl std::fmt::Debug for Secure { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("Secure").field("paired_now", &self.paired_now).finish_non_exhaustive() } }

pub struct Secure { t: TransportState, pub remote: Key, pub paired_now: bool, partial: Vec<u8> }

fn builder<'a>(params: &str, id: &'a Identity, psk: Option<&'a [u8; 32]>) -> Result<Builder<'a>, String> {
    let mut b = Builder::new(params.parse().map_err(|e| format!("{e}"))?).local_private_key(&id.private).prologue(PROLOGUE);
    if let Some(p) = psk { b = b.psk(3, p); }
    Ok(b)
}

/// The phone's side. `expect` is the desktop's public key, from the QR code (pairing) or from when it was paired (known), and the handshake fails if the desktop is anyone else.
pub async fn initiate<L: Link>(link: &mut L, id: &Identity, auth: Auth, expect: &[u8]) -> Result<Secure, String> {
    let (mode, params, psk) = match &auth { Auth::Pair { secret } => (MODE_PAIR, PARAMS_PAIR, Some(*secret)), Auth::Known => (MODE_KNOWN, PARAMS_KNOWN, None) };
    let mut hs = builder(params, id, psk.as_ref())?.build_initiator().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 65535];
    let n = hs.write_message(&[], &mut buf).map_err(|e| e.to_string())?;
    let mut first = vec![mode];
    first.extend_from_slice(&buf[..n]);
    link.send(first).await?;
    let m2 = link.recv().await?;
    hs.read_message(&m2, &mut buf).map_err(|e| format!("the other side did not complete the handshake: {e}"))?;
    let n = hs.write_message(&[], &mut buf).map_err(|e| e.to_string())?;
    link.send(buf[..n].to_vec()).await?;
    let remote = hs.get_remote_static().ok_or("the other side sent no key")?.to_vec();
    if !constant_eq(&remote, expect) { return Err("the computer's key is not the one this phone was paired with".into()); }
    let mut s = Secure { t: hs.into_transport_mode().map_err(|e| e.to_string())?, remote, paired_now: matches!(auth, Auth::Pair { .. }), partial: Vec::new() };
    // The desktop confirms it accepted us. Without this the phone would think it was connected when the desktop had refused it.
    let reply = link.recv().await.map_err(|_| "the computer refused this phone (is it paired, and is the pairing code still valid?)".to_string())?;
    match s.open(&reply)? { Some(m) if m == READY => Ok(s), _ => Err("unexpected reply from the computer".into()) }
}

/// The desktop's side. A device is let in only if it is already paired, or if it proves it holds the secret of the open pairing offer. The caller adds a newly paired key to its list.
pub async fn respond<L: Link>(link: &mut L, id: &Identity, policy: &dyn Policy) -> Result<Secure, String> {
    let first = link.recv().await?;
    let (mode, m1) = first.split_first().ok_or("empty first message")?;
    let (params, psk) = match *mode {
        MODE_KNOWN => (PARAMS_KNOWN, None),
        MODE_PAIR => (PARAMS_PAIR, Some(policy.take_pairing_secret().ok_or("there is no open pairing offer")?)),
        _ => return Err("unknown handshake".into()),
    };
    let mut hs = builder(params, id, psk.as_ref())?.build_responder().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 65535];
    hs.read_message(m1, &mut buf).map_err(|e| e.to_string())?;
    let n = hs.write_message(&[], &mut buf).map_err(|e| e.to_string())?;
    link.send(buf[..n].to_vec()).await?;
    let m3 = link.recv().await?;
    hs.read_message(&m3, &mut buf).map_err(|e| format!("the device did not pass the handshake: {e}"))?;
    let remote = hs.get_remote_static().ok_or("the device sent no key")?.to_vec();
    if *mode == MODE_KNOWN && !policy.is_paired(&remote) { return Err("this device is not paired".into()); }
    let mut s = Secure { t: hs.into_transport_mode().map_err(|e| e.to_string())?, remote, paired_now: *mode == MODE_PAIR, partial: Vec::new() };
    for f in s.seal(READY)? { link.send(f).await?; }
    Ok(s)
}

fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl Secure {
    /// Encrypts a message into frames to send, in order. Long messages are split.
    pub fn seal(&mut self, plain: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let mut frames = Vec::new();
        let mut parts: Vec<&[u8]> = plain.chunks(CHUNK).collect();
        if parts.is_empty() { parts.push(&[]); }
        let last = parts.len() - 1;
        for (i, part) in parts.iter().enumerate() {
            let mut body = Vec::with_capacity(part.len() + 1);
            body.push(if i == last { 0 } else { 1 });
            body.extend_from_slice(part);
            let mut out = vec![0u8; body.len() + 16];
            let n = self.t.write_message(&body, &mut out).map_err(|e| e.to_string())?;
            out.truncate(n);
            frames.push(out);
        }
        Ok(frames)
    }

    /// Decrypts one frame. Returns the whole message when its last frame arrives. A frame that was changed, replayed or out of order is an error.
    pub fn open(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let mut out = vec![0u8; frame.len()];
        let n = self.t.read_message(frame, &mut out).map_err(|_| "a message failed to verify and was dropped".to_string())?;
        let (flag, body) = out[..n].split_first().ok_or("empty message")?;
        if self.partial.len() + body.len() > MAX_MESSAGE { self.partial.clear(); return Err("message too large".into()); }
        self.partial.extend_from_slice(body);
        if *flag == 0 { Ok(Some(std::mem::take(&mut self.partial))) } else { Ok(None) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::sync::mpsc;

    struct Mem { tx: mpsc::UnboundedSender<Vec<u8>>, rx: mpsc::UnboundedReceiver<Vec<u8>>, tamper: bool }
    impl Link for Mem {
        async fn send(&mut self, mut f: Vec<u8>) -> Result<(), String> { if self.tamper && f.len() > 20 { let i = f.len() - 3; f[i] ^= 1; } self.tx.send(f).map_err(|e| e.to_string()) }
        async fn recv(&mut self) -> Result<Vec<u8>, String> { self.rx.recv().await.ok_or_else(|| "closed".to_string()) }
    }
    fn pipe() -> (Mem, Mem) {
        let (a_tx, b_rx) = mpsc::unbounded_channel();
        let (b_tx, a_rx) = mpsc::unbounded_channel();
        (Mem { tx: a_tx, rx: a_rx, tamper: false }, Mem { tx: b_tx, rx: b_rx, tamper: false })
    }

    struct Desk { paired: Mutex<Vec<Key>>, offer: Mutex<Option<[u8; 32]>> }
    impl Policy for Desk {
        fn is_paired(&self, p: &[u8]) -> bool { self.paired.lock().unwrap().iter().any(|k| k == p) }
        fn take_pairing_secret(&self) -> Option<[u8; 32]> { self.offer.lock().unwrap().take() }
    }
    fn desk(offer: Option<[u8; 32]>, paired: Vec<Key>) -> Desk { Desk { paired: Mutex::new(paired), offer: Mutex::new(offer) } }

    async fn run(phone: &Identity, auth: Auth, expect: &[u8], host: &Identity, policy: &Desk, tamper: bool) -> (Result<Secure, String>, Result<Secure, String>) {
        let (mut a, mut b) = pipe();
        a.tamper = tamper;
        // Each side owns its end of the pipe, so a side that gives up closes it and the other side stops waiting, as a real connection would.
        tokio::join!(async move { initiate(&mut a, phone, auth, expect).await }, async move { respond(&mut b, host, policy).await })
    }

    #[tokio::test]
    async fn pairing_then_messages_both_ways() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let secret = [7u8; 32];
        let (p, h) = run(&phone, Auth::Pair { secret }, &host.public, &host, &desk(Some(secret), vec![]), false).await;
        let (mut p, mut h) = (p.unwrap(), h.unwrap());
        assert!(p.paired_now && h.paired_now);
        assert_eq!(h.remote, phone.public);
        assert_eq!(p.remote, host.public);
        let f = p.seal(b"hello desktop").unwrap();
        assert_eq!(h.open(&f[0]).unwrap().unwrap(), b"hello desktop");
        let f = h.seal(b"hello phone").unwrap();
        assert_eq!(p.open(&f[0]).unwrap().unwrap(), b"hello phone");
    }

    #[tokio::test]
    async fn a_wrong_pairing_secret_is_refused_and_pairs_nobody() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(Some([1u8; 32]), vec![]);
        let (p, h) = run(&phone, Auth::Pair { secret: [2u8; 32] }, &host.public, &host, &d, false).await;
        assert!(h.is_err());
        assert!(p.is_err());
        assert!(d.paired.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_pairing_offer_works_once() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let s = [9u8; 32];
        let d = desk(Some(s), vec![]);
        assert!(run(&phone, Auth::Pair { secret: s }, &host.public, &host, &d, false).await.1.is_ok());
        let other = Identity::generate().unwrap();
        let (_, h) = run(&other, Auth::Pair { secret: s }, &host.public, &host, &d, false).await;
        assert!(h.unwrap_err().contains("no open pairing offer"));
    }

    #[tokio::test]
    async fn only_paired_devices_reconnect() {
        let (host, phone, stranger) = (Identity::generate().unwrap(), Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(None, vec![phone.public.clone()]);
        assert!(run(&phone, Auth::Known, &host.public, &host, &d, false).await.1.is_ok());
        let (_, h) = run(&stranger, Auth::Known, &host.public, &host, &d, false).await;
        assert!(h.unwrap_err().contains("not paired"));
    }

    #[tokio::test]
    async fn the_phone_refuses_an_impostor_desktop() {
        let (host, phone, impostor) = (Identity::generate().unwrap(), Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(None, vec![phone.public.clone()]);
        let (p, _) = run(&phone, Auth::Known, &host.public, &impostor, &d, false).await;
        assert!(p.unwrap_err().contains("not the one this phone was paired with"));
    }

    #[tokio::test]
    async fn long_messages_are_split_and_rejoined() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(None, vec![phone.public.clone()]);
        let (p, h) = run(&phone, Auth::Known, &host.public, &host, &d, false).await;
        let (mut p, mut h) = (p.unwrap(), h.unwrap());
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let frames = p.seal(&big).unwrap();
        assert!(frames.len() >= 4);
        let mut got = None;
        for f in &frames { got = h.open(f).unwrap(); }
        assert_eq!(got.unwrap(), big);
        assert_eq!(h.open(&p.seal(b"").unwrap()[0]).unwrap().unwrap(), Vec::<u8>::new());
    }

    #[tokio::test]
    async fn changed_replayed_and_reordered_frames_are_rejected() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(None, vec![phone.public.clone()]);
        let (p, h) = run(&phone, Auth::Known, &host.public, &host, &d, false).await;
        let (mut p, mut h) = (p.unwrap(), h.unwrap());
        let one = p.seal(b"one").unwrap().remove(0);
        let two = p.seal(b"two").unwrap().remove(0);
        let mut bad = one.clone(); let last = bad.len() - 1; bad[last] ^= 1;
        assert!(h.open(&bad).is_err());
        assert!(h.open(&two).is_err());           // out of order
        assert_eq!(h.open(&one).unwrap().unwrap(), b"one");
        assert!(h.open(&one).is_err());           // replay
    }

    #[tokio::test]
    async fn a_handshake_changed_in_transit_fails() {
        let (host, phone) = (Identity::generate().unwrap(), Identity::generate().unwrap());
        let d = desk(None, vec![phone.public.clone()]);
        let (p, h) = run(&phone, Auth::Known, &host.public, &host, &d, true).await;
        assert!(p.is_err() || h.is_err());
    }
}
