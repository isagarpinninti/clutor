use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;
use tauri::{AppHandle, Emitter, State};

// The engine's C API (c/src/chat.h).
#[repr(C)]
pub struct ClChat {
    _private: [u8; 0],
}
type Emit = unsafe extern "C" fn(*mut c_void, *const c_char);
extern "C" {
    fn cl_chat_open(argc: c_int, argv: *const *const c_char, emit: Emit, user: *mut c_void, err: *mut c_char, errlen: usize) -> *mut ClChat;
    fn cl_chat_run(c: *mut ClChat, op: *const c_char) -> c_int;
    fn cl_chat_cancel(c: *mut ClChat);
    fn cl_chat_close(c: *mut ClChat);
}

/// What the engine's callback needs: where to send events. The first event (ready) goes back to the command that opened the engine.
struct Ctx {
    app: AppHandle,
    ready: Mutex<Option<Sender<Result<String, String>>>>,
}

unsafe extern "C" fn on_event(user: *mut c_void, line: *const c_char) {
    let ctx = &*(user as *const Ctx);
    let text = CStr::from_ptr(line).to_string_lossy().into_owned();
    if let Some(tx) = ctx.ready.lock().ok().and_then(|mut g| g.take()) {
        let _ = tx.send(Ok(text));
    } else {
        let _ = ctx.app.emit("engine-event", text);
    }
}

struct Running {
    ops: Sender<String>,
    /// The chat handle as an address, so cancel can be called while the worker thread is busy (cl_chat_cancel is thread-safe).
    chat: usize,
    worker: JoinHandle<()>,
}

#[derive(Default)]
pub struct EngineState(Mutex<Option<Running>>);

fn stop(running: Running) {
    let _ = running.ops.send("{\"op\":\"quit\"}".to_string());
    let _ = running.worker.join();
}

/// Loads a model into the engine and returns its first ("ready") event as JSON. Later events arrive as `engine-event` events.
#[tauri::command]
pub async fn engine_open(app: AppHandle, state: State<'_, EngineState>, path: String, ctx: u32, lock: bool) -> Result<String, String> {
    if let Some(old) = state.0.lock().map_err(|e| e.to_string())?.take() {
        tauri::async_runtime::spawn_blocking(move || stop(old)).await.map_err(|e| e.to_string())?;
    }
    let (ready_tx, ready_rx): (Sender<Result<String, String>>, Receiver<Result<String, String>>) = channel();
    let (ops_tx, ops_rx) = channel::<String>();
    let (chat_tx, chat_rx) = channel::<usize>();
    let mut args = vec![path, "--ctx".to_string(), ctx.to_string()];
    if lock { args.push("--lock".to_string()); }
    let ctx_box = Box::new(Ctx { app, ready: Mutex::new(Some(ready_tx.clone())) });
    let worker = std::thread::spawn(move || {
        let ctx_ptr = Box::into_raw(ctx_box);
        let cargs: Vec<CString> = args.iter().filter_map(|a| CString::new(a.as_str()).ok()).collect();
        let argv: Vec<*const c_char> = cargs.iter().map(|c| c.as_ptr()).collect();
        let mut err = vec![0 as c_char; 512];
        let chat = unsafe { cl_chat_open(argv.len() as c_int, argv.as_ptr(), on_event, ctx_ptr as *mut c_void, err.as_mut_ptr(), err.len()) };
        if chat.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            let _ = ready_tx.send(Err(if msg.is_empty() { "the engine could not open this model".to_string() } else { msg }));
            unsafe { drop(Box::from_raw(ctx_ptr)) };
            return;
        }
        let _ = chat_tx.send(chat as usize);
        while let Ok(op) = ops_rx.recv() {
            let Ok(c) = CString::new(op) else { continue };
            if unsafe { cl_chat_run(chat, c.as_ptr()) } == 1 { break; }
        }
        unsafe { cl_chat_close(chat); drop(Box::from_raw(ctx_ptr)); }
    });
    let first = tauri::async_runtime::spawn_blocking(move || ready_rx.recv()).await.map_err(|e| e.to_string())?;
    match first {
        Ok(Ok(ready)) => {
            let chat = chat_rx.recv().map_err(|e| e.to_string())?;
            *state.0.lock().map_err(|e| e.to_string())? = Some(Running { ops: ops_tx, chat, worker });
            Ok(ready)
        }
        Ok(Err(e)) => { let _ = worker.join(); Err(e) }
        Err(e) => Err(e.to_string()),
    }
}

/// Runs one operation (a JSON line, as `clutor-engine chat` takes). "cancel" stops the running operation at once; the rest are queued in order.
#[tauri::command]
pub fn engine_op(state: State<'_, EngineState>, op: String) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    let running = guard.as_ref().ok_or_else(|| "no model is loaded".to_string())?;
    if op.contains("\"cancel\"") {
        unsafe { cl_chat_cancel(running.chat as *mut ClChat) };
        return Ok(());
    }
    running.ops.send(op).map_err(|_| "the engine has stopped".to_string())
}

#[tauri::command]
pub async fn engine_close(state: State<'_, EngineState>) -> Result<(), String> {
    let old = state.0.lock().map_err(|e| e.to_string())?.take();
    if let Some(old) = old {
        tauri::async_runtime::spawn_blocking(move || stop(old)).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}
