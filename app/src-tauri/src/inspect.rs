//! Looking at a model file without loading it: read pieces of its header so the planner can say whether it will run, and time the disk it sits on.

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::Serialize;
use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::{c_char, c_double};

extern "C" {
    fn cl_disk_bench(path: *const c_char, seconds: c_double, seq: *mut c_double, rnd: *mut c_double, err: *mut c_char, errlen: usize) -> i32;
}

const MAX_READ: u64 = 32 * 1024 * 1024;

fn check(path: &str) -> Result<(), String> {
    if !path.to_lowercase().ends_with(".gguf") { return Err("only .gguf model files can be inspected".into()); }
    Ok(())
}

#[tauri::command]
pub fn model_size(path: String) -> Result<u64, String> { check(&path)?; std::fs::metadata(&path).map(|m| m.len()).map_err(|e| e.to_string()) }

/// `length` bytes of the file from `offset`, base64 (at most 32 MB per call; fewer at the end of the file).
#[tauri::command]
pub fn model_read(path: String, offset: u64, length: u64) -> Result<String, String> {
    check(&path)?;
    if length > MAX_READ { return Err("that is too much to read at once".into()); }
    let mut f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(length as usize);
    f.take(length).read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(B64.encode(buf))
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DiskSpeed { sequential_bytes_per_sec: f64, scattered_bytes_per_sec: f64, seconds: f64 }

/// Times reading the model's own file (about `seconds` in all): sequential and scattered reads, in bytes per second. Where the system cannot be told to skip its cache the
/// figure can be too high for a file that was read recently; the caller says so when it is implausibly fast.
#[tauri::command]
pub async fn disk_bench(path: String, seconds: f64) -> Result<DiskSpeed, String> {
    check(&path)?;
    let secs = seconds.clamp(1.0, 20.0);
    tauri::async_runtime::spawn_blocking(move || {
        let c = CString::new(path).map_err(|_| "bad path".to_string())?;
        let (mut seq, mut rnd) = (0f64, 0f64);
        let mut err = vec![0 as c_char; 256];
        let rc = unsafe { cl_disk_bench(c.as_ptr(), secs, &mut seq, &mut rnd, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            let msg: Vec<u8> = err.iter().take_while(|&&b| b != 0).map(|&b| b as u8).collect();
            return Err(String::from_utf8_lossy(&msg).to_string());
        }
        Ok(DiskSpeed { sequential_bytes_per_sec: seq, scattered_bytes_per_sec: rnd, seconds: secs })
    }).await.map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_gguf_files_and_bounded_reads() {
        assert!(model_size("/etc/passwd".into()).is_err());
        assert!(model_read("a.gguf".into(), 0, MAX_READ + 1).is_err());
    }

    #[test]
    fn reads_the_requested_piece_and_stops_at_the_end() {
        let p = std::env::temp_dir().join(format!("clutor-inspect-{}.gguf", std::process::id()));
        std::fs::write(&p, b"0123456789").unwrap();
        let path = p.to_string_lossy().to_string();
        assert_eq!(model_size(path.clone()).unwrap(), 10);
        assert_eq!(B64.decode(model_read(path.clone(), 2, 4).unwrap()).unwrap(), b"2345");
        assert_eq!(B64.decode(model_read(path, 8, 100).unwrap()).unwrap(), b"89");
        let _ = std::fs::remove_file(p);
    }

    #[tokio::test]
    async fn times_a_file_and_refuses_one_too_small() {
        let p = std::env::temp_dir().join(format!("clutor-bench-{}.gguf", std::process::id()));
        std::fs::write(&p, vec![7u8; 32 * 1024 * 1024]).unwrap();
        let r = disk_bench(p.to_string_lossy().to_string(), 1.0).await.unwrap();
        assert!(r.sequential_bytes_per_sec > 1e6 && r.scattered_bytes_per_sec > 1e6, "{} {}", r.sequential_bytes_per_sec, r.scattered_bytes_per_sec);
        std::fs::write(&p, b"tiny").unwrap();
        assert!(disk_bench(p.to_string_lossy().to_string(), 1.0).await.unwrap_err().contains("too small"));
        let _ = std::fs::remove_file(p);
    }
}
