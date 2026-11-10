//! What the desktop remembers about remote control: its own key pair, the phones it has paired, and the pairing offer currently on screen.
//! The private key is kept in a file in the app's data folder that only your user can read; keeping it in the system keychain is a later improvement.

use super::secure::{Identity, Key, Policy};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Device { pub id: String, pub name: String, pub public: String, pub created: u64, pub last_seen: u64 }

#[derive(Serialize, Deserialize)]
struct Disk { private: String, public: String, devices: Vec<Device> }

struct Inner { identity: Identity, devices: Vec<Device>, offer: Option<([u8; 32], Instant)> }

pub struct Store { path: Option<PathBuf>, inner: Mutex<Inner> }

fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }
pub fn device_id(public: &[u8]) -> String { hex::encode(&Sha256::digest(public)[..8]) }

impl Store {
    /// Loads the saved state, or creates a new key pair on first use.
    pub fn open(path: PathBuf) -> Result<Store, String> {
        let inner = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let d: Disk = serde_json::from_str(&text).map_err(|e| format!("the remote settings file is damaged: {e}"))?;
                Inner { identity: Identity { private: B64.decode(d.private).map_err(|e| e.to_string())?, public: B64.decode(d.public).map_err(|e| e.to_string())? }, devices: d.devices, offer: None }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Inner { identity: Identity::generate()?, devices: vec![], offer: None },
            Err(e) => return Err(e.to_string()),
        };
        let s = Store { path: Some(path), inner: Mutex::new(inner) };
        s.save()?;
        Ok(s)
    }

    pub fn memory() -> Store { Store { path: None, inner: Mutex::new(Inner { identity: Identity::generate().expect("keys"), devices: vec![], offer: None }) } }

    fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else { return Ok(()) };
        let g = self.inner.lock().unwrap();
        let text = serde_json::to_string_pretty(&Disk { private: B64.encode(&g.identity.private), public: B64.encode(&g.identity.public), devices: g.devices.clone() }).map_err(|e| e.to_string())?;
        drop(g);
        if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
        let tmp = path.with_extension("tmp");
        {
            use std::io::Write;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            #[cfg(unix)]
            { use std::os::unix::fs::OpenOptionsExt; opts.mode(0o600); }
            let mut f = opts.open(&tmp).map_err(|e| e.to_string())?;
            f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    pub fn identity(&self) -> Identity { self.inner.lock().unwrap().identity.clone() }
    pub fn public_key(&self) -> Key { self.inner.lock().unwrap().identity.public.clone() }
    /// The computer's name for a relay: 64 hex characters derived from its public key.
    pub fn host_id(&self) -> String { hex::encode(Sha256::digest(&self.public_key())) }
    pub fn devices(&self) -> Vec<Device> { self.inner.lock().unwrap().devices.clone() }

    /// Opens a pairing offer: a fresh one-time secret, valid for `ttl`. Starting another replaces it.
    pub fn start_offer(&self, ttl: Duration) -> [u8; 32] {
        let mut secret = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut secret);
        self.inner.lock().unwrap().offer = Some((secret, Instant::now() + ttl));
        secret
    }
    pub fn cancel_offer(&self) { self.inner.lock().unwrap().offer = None; }
    pub fn offer_open(&self) -> bool { self.inner.lock().unwrap().offer.is_some_and(|(_, t)| Instant::now() < t) }

    pub fn add_device(&self, public: &[u8], name: &str) -> Result<Device, String> {
        let d = Device { id: device_id(public), name: clean(name), public: hex::encode(public), created: now(), last_seen: now() };
        {
            let mut g = self.inner.lock().unwrap();
            g.devices.retain(|x| x.id != d.id);
            g.devices.push(d.clone());
        }
        self.save()?;
        Ok(d)
    }
    /// Records that a paired phone connected (and what it calls itself).
    pub fn touch(&self, public: &[u8], name: Option<&str>) -> Option<Device> {
        let id = device_id(public);
        let out = {
            let mut g = self.inner.lock().unwrap();
            let d = g.devices.iter_mut().find(|x| x.id == id)?;
            d.last_seen = now();
            if let Some(n) = name { d.name = clean(n); }
            d.clone()
        };
        let _ = self.save();
        Some(out)
    }
    pub fn revoke(&self, id: &str) -> Option<Device> {
        let removed = { let mut g = self.inner.lock().unwrap(); let i = g.devices.iter().position(|d| d.id == id)?; Some(g.devices.remove(i)) };
        let _ = self.save();
        removed
    }
}

fn clean(name: &str) -> String {
    let n: String = name.chars().filter(|c| !c.is_control()).take(40).collect();
    if n.trim().is_empty() { "Phone".into() } else { n.trim().to_string() }
}

impl Policy for Store {
    fn is_paired(&self, public: &[u8]) -> bool { let h = hex::encode(public); self.inner.lock().unwrap().devices.iter().any(|d| d.public == h) }
    fn take_pairing_secret(&self) -> Option<[u8; 32]> {
        let (secret, until) = self.inner.lock().unwrap().offer.take()?;
        (Instant::now() < until).then_some(secret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_devices_survive_a_restart_and_the_file_is_private() {
        let dir = std::env::temp_dir().join(format!("clutor-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("remote.json");
        let a = Store::open(path.clone()).unwrap();
        let phone = Identity::generate().unwrap();
        let d = a.add_device(&phone.public, "Pixel \u{7}8").unwrap();
        assert_eq!(d.name, "Pixel 8");
        let public = a.public_key();
        let b = Store::open(path.clone()).unwrap();
        assert_eq!(b.public_key(), public);
        assert!(b.is_paired(&phone.public));
        #[cfg(unix)]
        { use std::os::unix::fs::PermissionsExt; assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600); }
        assert!(b.revoke(&d.id).is_some());
        assert!(!Store::open(path).unwrap().is_paired(&phone.public));
    }

    #[test]
    fn an_offer_is_single_use_and_expires() {
        let s = Store::memory();
        let secret = s.start_offer(Duration::from_secs(60));
        assert!(s.offer_open());
        assert_eq!(s.take_pairing_secret(), Some(secret));
        assert_eq!(s.take_pairing_secret(), None);
        s.start_offer(Duration::from_millis(0));
        assert_eq!(s.take_pairing_secret(), None);
    }

    #[test]
    fn the_host_id_is_a_64_character_hex_name() {
        let id = Store::memory().host_id();
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
