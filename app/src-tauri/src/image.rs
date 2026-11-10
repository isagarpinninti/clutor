//! Image generation inside Clutor's own process (see ../image/clutor_image.c). A picture is one call: the model is loaded, the picture made and saved, everything freed. The chat model
//! is stopped by the interface before a picture starts, so the two take turns instead of both sitting in memory.
//!
//! This is only built when the image engine's libraries are available (see build.rs); other builds report that the engine is not included.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use tauri::{AppHandle, Emitter, Manager};

static CANCEL: AtomicI32 = AtomicI32::new(0);
static BUSY: AtomicBool = AtomicBool::new(false);

#[derive(Deserialize)]
pub struct ImageRequest {
    diffusion: String,
    text_encoder: String,
    vae: String,
    prompt: String,
    negative: Option<String>,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f32,
    seed: i64,
    turbo: bool,
    low_memory: bool,
}

#[derive(Serialize)]
pub struct ImageResult { path: String, seconds: f64 }

#[derive(Serialize, Clone)]
struct Progress { step: i32, steps: i32, seconds: f32 }

#[cfg(has_image)]
mod ffi {
    use std::os::raw::{c_char, c_float, c_int, c_void};
    #[repr(C)]
    pub struct Job {
        pub diffusion: *const c_char, pub text_encoder: *const c_char, pub vae: *const c_char, pub prompt: *const c_char, pub negative: *const c_char,
        pub width: c_int, pub height: c_int, pub steps: c_int, pub turbo: c_int,
        pub cfg: c_float, pub seed: i64, pub low_memory: c_int, pub out_png: *const c_char,
    }
    pub type Progress = Option<extern "C" fn(c_int, c_int, c_float, *mut c_void)>;
    extern "C" {
        pub fn clutor_image_generate(job: *const Job, progress: Progress, user: *mut c_void, cancel: *mut c_int, err: *mut c_char, errlen: usize) -> c_int;
    }
}

#[cfg(has_image)]
extern "C" fn on_progress(step: i32, steps: i32, seconds: f32, user: *mut std::os::raw::c_void) {
    let app = unsafe { &*(user as *const AppHandle) };
    let _ = app.emit("image-progress", Progress { step, steps, seconds });
}

fn images_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let d = app.path().app_data_dir().map_err(|e| e.to_string())?.join("images");
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}

#[tauri::command]
pub fn image_available() -> bool { cfg!(has_image) }

#[tauri::command]
pub fn image_cancel() { CANCEL.store(1, Ordering::SeqCst); }

#[tauri::command]
pub async fn image_generate(app: AppHandle, req: ImageRequest) -> Result<ImageResult, String> {
    #[cfg(not(has_image))]
    { let _ = (app, req); return Err("This build of Clutor does not include the image engine.".into()); }
    #[cfg(has_image)]
    {
        use std::ffi::CString;
        if BUSY.swap(true, Ordering::SeqCst) { return Err("a picture is already being made".into()); }
        CANCEL.store(0, Ordering::SeqCst);
        let out = images_dir(&app)?.join(format!("clutor-{}.png", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)));
        let out_s = out.display().to_string();
        let handle = app.clone();
        let res = tauri::async_runtime::spawn_blocking(move || {
            let c = |s: &str| CString::new(s).map_err(|_| "a file name or prompt contains a NUL character".to_string());
            let (d, t, v, p, o) = (c(&req.diffusion)?, c(&req.text_encoder)?, c(&req.vae)?, c(&req.prompt)?, c(&out_s)?);
            let neg = c(req.negative.as_deref().unwrap_or(""))?;
            let job = ffi::Job { diffusion: d.as_ptr(), text_encoder: t.as_ptr(), vae: v.as_ptr(), prompt: p.as_ptr(), negative: neg.as_ptr(), width: req.width as i32, height: req.height as i32,
                steps: req.steps as i32, turbo: req.turbo as i32, cfg: req.cfg, seed: req.seed, low_memory: req.low_memory as i32, out_png: o.as_ptr() };
            let mut err = vec![0 as std::os::raw::c_char; 512];
            let t0 = std::time::Instant::now();
            let rc = unsafe { ffi::clutor_image_generate(&job, Some(on_progress), &handle as *const AppHandle as *mut std::os::raw::c_void, CANCEL.as_ptr(), err.as_mut_ptr(), err.len()) };
            if rc == 0 { Ok(t0.elapsed().as_secs_f64()) } else { Err(unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()) }
        }).await.map_err(|e| e.to_string());
        BUSY.store(false, Ordering::SeqCst);
        match res? { Ok(seconds) => Ok(ImageResult { path: out.display().to_string(), seconds }), Err(e) => { let _ = std::fs::remove_file(&out); Err(e) } }
    }
}

/// A picture the app made, as base64 (the interface shows it from this). Only files in the app's own pictures folder can be read.
#[tauri::command]
pub fn image_read(app: AppHandle, path: String) -> Result<String, String> {
    use base64::Engine;
    let dir = images_dir(&app)?.canonicalize().map_err(|e| e.to_string())?;
    let p = std::path::Path::new(&path).canonicalize().map_err(|_| "that picture no longer exists".to_string())?;
    if !p.starts_with(&dir) { return Err("that is not one of Clutor's pictures".into()); }
    let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// Copies a picture to a place the person chose.
#[tauri::command]
pub fn image_save_as(app: AppHandle, path: String, dest: String) -> Result<(), String> {
    let dir = images_dir(&app)?.canonicalize().map_err(|e| e.to_string())?;
    let p = std::path::Path::new(&path).canonicalize().map_err(|_| "that picture no longer exists".to_string())?;
    if !p.starts_with(&dir) { return Err("that is not one of Clutor's pictures".into()); }
    std::fs::copy(&p, &dest).map(|_| ()).map_err(|e| e.to_string())
}
