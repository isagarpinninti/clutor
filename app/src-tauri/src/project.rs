//! What the coding agent needs from the machine: read and write files inside one project folder, run an approved command in a scratch copy of it, and talk to MCP
//! servers over their standard input and output. Every path the web view sends is checked again here, so a confused or hostile model cannot reach outside the folder.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Mutex;

const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "dist", "build", ".clutor", ".venv", "venv", "__pycache__", ".next", ".gradle", ".idea", ".DS_Store"];
const MAX_FILES: usize = 5000;
const MAX_FILE_BYTES: u64 = 1024 * 1024;

fn root_of(root: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(root).canonicalize().map_err(|e| format!("cannot open the project folder: {e}"))?;
    if !p.is_dir() { return Err("the project path is not a folder".into()); }
    Ok(p)
}

/// A relative path that stays inside `root`: no absolute paths, no `..`, and no symbolic link that leads out.
pub fn jail(root: &Path, rel: &str) -> Result<PathBuf, String> {
    if rel.is_empty() || rel.contains('\0') { return Err("empty or invalid path".into()); }
    let r = Path::new(rel);
    if r.is_absolute() { return Err("absolute paths are not allowed".into()); }
    for c in r.components() {
        match c {
            Component::Normal(_) | Component::CurDir => {}
            _ => return Err("the path leaves the project".into()),
        }
    }
    let full = root.join(r);
    // The nearest existing ancestor must still be inside the root once symbolic links are resolved.
    let mut probe = full.as_path();
    loop {
        if probe.exists() { break; }
        probe = probe.parent().ok_or("invalid path")?;
    }
    let real = probe.canonicalize().map_err(|e| e.to_string())?;
    if !real.starts_with(root) { return Err("the path leads outside the project".into()); }
    Ok(full)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        if out.len() >= MAX_FILES { return; }
        let name = e.file_name().to_string_lossy().to_string();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_symlink() { continue; }
        let path = e.path();
        if ft.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) { continue; }
            walk(root, &path, out);
        } else if ft.is_file() {
            if SKIP_DIRS.contains(&name.as_str()) { continue; }
            if e.metadata().map(|m| m.len() > MAX_FILE_BYTES * 4).unwrap_or(true) { continue; }
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect::<Vec<_>>().join("/"));
            }
        }
    }
}

#[tauri::command]
pub fn project_list(root: String) -> Result<Vec<String>, String> {
    let root = root_of(&root)?;
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    Ok(out)
}

/// The file's text, or None when it does not exist. Binary and very large files are an error the agent can read.
#[tauri::command]
pub fn project_read(root: String, path: String) -> Result<Option<String>, String> {
    let root = root_of(&root)?;
    let full = jail(&root, &path)?;
    let meta = match std::fs::metadata(&full) { Ok(m) => m, Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None), Err(e) => return Err(e.to_string()) };
    if !meta.is_file() { return Err("not a file".into()); }
    if meta.len() > MAX_FILE_BYTES { return Err(format!("{path} is larger than {} KB", MAX_FILE_BYTES / 1024)); }
    let bytes = std::fs::read(&full).map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map(Some).map_err(|_| format!("{path} is not a text file"))
}

/// Writes the file atomically (a temporary file in the same folder, then a rename), or deletes it when `content` is None.
#[tauri::command]
pub fn project_write(root: String, path: String, content: Option<String>) -> Result<(), String> {
    let root = root_of(&root)?;
    let full = jail(&root, &path)?;
    match content {
        None => match std::fs::remove_file(&full) { Ok(()) => Ok(()), Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), Err(e) => Err(e.to_string()) },
        Some(text) => {
            if let Some(dir) = full.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
            let tmp = full.with_extension(format!("clutor-{}.tmp", std::process::id()));
            std::fs::write(&tmp, text.as_bytes()).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &full).map_err(|e| { let _ = std::fs::remove_file(&tmp); e.to_string() })
        }
    }
}

#[derive(Serialize)]
pub struct RunResult { exit_code: Option<i32>, output: String, timed_out: bool, duration_ms: u64 }

#[derive(Deserialize)]
pub struct Overlay { path: String, content: Option<String> }

fn copy_project(root: &Path, dest: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    walk(root, root, &mut files);
    let mut total = 0u64;
    for rel in files {
        let src = root.join(&rel);
        total += std::fs::metadata(&src).map(|m| m.len()).unwrap_or(0);
        if total > 200_000_000 { return Err("the project is larger than 200 MB, too big to copy for a test run".into()); }
        let to = dest.join(&rel);
        if let Some(d) = to.parent() { std::fs::create_dir_all(d).map_err(|e| e.to_string())?; }
        std::fs::copy(&src, &to).map_err(|e| e.to_string())?;
    }
    // Dependencies are linked, not copied.
    let nm = root.join("node_modules");
    if nm.is_dir() {
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(&nm, dest.join("node_modules"));
        #[cfg(windows)]
        let _ = std::os::windows::fs::symlink_dir(&nm, dest.join("node_modules"));
    }
    Ok(())
}

fn scratch_dir() -> Result<PathBuf, String> {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let d = std::env::temp_dir().join(format!("clutor-run-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(d.join(".home")).map_err(|e| e.to_string())?;
    d.canonicalize().map_err(|e| e.to_string())
}

#[cfg(unix)]
fn kill_group(child: &Child) {
    if let Some(pid) = child.id() { unsafe { libc::kill(-(pid as i32), libc::SIGKILL); } }
}
/// On Windows a command runs under cmd.exe, so the whole process tree is ended, not only cmd.
#[cfg(not(unix))]
fn kill_group(child: &Child) {
    if let Some(pid) = child.id() { let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status(); }
}

/// Runs the steps (already checked by the command policy in the web view) in a scratch copy of the project that includes the agent's unapplied changes, with a stripped
/// environment and a time limit. The copy is deleted afterwards. This isolates the project and the environment, not the machine.
#[tauri::command]
pub async fn project_run(root: String, overlay: Vec<Overlay>, steps: Vec<Vec<String>>, timeout_ms: u64) -> Result<RunResult, String> {
    let root = root_of(&root)?;
    let dir = scratch_dir()?;
    let started = Instant::now();
    let result = run_in(&root, &dir, overlay, steps, timeout_ms).await;
    let _ = std::fs::remove_dir_all(&dir);
    result.map(|(code, output, timed_out)| RunResult { exit_code: code, output, timed_out, duration_ms: started.elapsed().as_millis() as u64 })
}

async fn run_in(root: &Path, dir: &Path, overlay: Vec<Overlay>, steps: Vec<Vec<String>>, timeout_ms: u64) -> Result<(Option<i32>, String, bool), String> {
    copy_project(root, dir)?;
    for o in overlay {
        let dest = jail(dir, &o.path)?;
        match o.content {
            None => { let _ = std::fs::remove_file(&dest); }
            Some(t) => { if let Some(d) = dest.parent() { std::fs::create_dir_all(d).map_err(|e| e.to_string())?; } std::fs::write(&dest, t).map_err(|e| e.to_string())?; }
        }
    }
    let mut left = Duration::from_millis(timeout_ms.max(1000));
    let mut output = String::new();
    for step in steps {
        let Some(program) = step.first() else { continue };
        #[cfg(windows)]
        if step.iter().any(|a| a.chars().any(|c| "&|<>^%\"!`".contains(c))) { return Ok((None, "an argument contains a character that is not allowed on Windows".into(), false)); }
        // Windows programs such as npm are .cmd files, which only cmd.exe can start; the arguments were checked above for anything cmd.exe would interpret.
        #[cfg(windows)]
        let mut cmd = { let mut c = Command::new("cmd"); c.arg("/C").arg(program); c };
        #[cfg(not(windows))]
        let mut cmd = Command::new(program);
        cmd.args(&step[1..]).current_dir(dir).env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", dir.join(".home")).env("TMPDIR", dir).env("CI", "1").env("NO_COLOR", "1").env("FORCE_COLOR", "0").env("LANG", "C.UTF-8").env("NODE_ENV", "test")
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        #[cfg(windows)]
        for k in ["SystemRoot", "ComSpec", "PATHEXT", "TEMP", "TMP"] { if let Ok(v) = std::env::var(k) { cmd.env(k, v); } }
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);   // no console window
        let mut child = match cmd.spawn() { Ok(c) => c, Err(e) => return Ok((None, format!("{output}could not start {program}: {e}"), false)) };
        let mut so = child.stdout.take().ok_or("no stdout")?;
        let mut se = child.stderr.take().ok_or("no stderr")?;
        let t0 = Instant::now();
        let read = async {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let _ = tokio::join!(so.read_to_end(&mut a), se.read_to_end(&mut b));
            a.extend_from_slice(&b);
            a
        };
        let waited = tokio::time::timeout(left, async { let bytes = read.await; let st = child.wait().await; (bytes, st) }).await;
        left = left.saturating_sub(t0.elapsed());
        match waited {
            Ok((bytes, st)) => {
                let mut text = String::from_utf8_lossy(&bytes).to_string();
                if text.len() > 200_000 { text.truncate(text.floor_char_boundary(200_000)); text.push_str("\n[output cut off]"); }
                if !output.is_empty() { output.push('\n'); }
                output.push_str(&text);
                let code = st.map_err(|e| e.to_string())?.code();
                if code != Some(0) { return Ok((code, output, false)); }
            }
            Err(_) => { kill_group(&child); let _ = child.kill().await; return Ok((None, output, true)); }
        }
    }
    Ok((Some(0), output, false))
}

// ---- MCP servers over stdio -----------------------------------------------------------------------------------------------------------------------------------

struct McpProc { child: Child, stdin: ChildStdin }

#[derive(Default)]
pub struct McpState(Arc<Mutex<HashMap<String, McpProc>>>);

#[derive(Serialize, Clone)]
struct McpLine { id: String, line: String }
#[derive(Serialize, Clone)]
struct McpExit { id: String, code: Option<i32> }

/// Starts an MCP server the user configured. Its environment is inherited (npx and uvx need it) plus what the configuration adds; its messages arrive as `mcp-line` events.
#[tauri::command]
pub async fn mcp_start(app: AppHandle, state: State<'_, McpState>, id: String, command: String, args: Vec<String>, env: HashMap<String, String>, cwd: Option<String>) -> Result<(), String> {
    let mut procs = state.0.lock().await;
    if procs.contains_key(&id) { return Err("that server is already running".into()); }
    let mut cmd = Command::new(&command);
    cmd.args(&args).envs(&env).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    if let Some(c) = cwd { cmd.current_dir(c); }
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let mut child = cmd.spawn().map_err(|e| format!("could not start {command}: {e}"))?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;
    let (a1, i1) = (app.clone(), id.clone());
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await { let _ = a1.emit("mcp-line", McpLine { id: i1.clone(), line }); }
        let _ = a1.emit("mcp-exit", McpExit { id: i1, code: None });
    });
    let (a2, i2) = (app.clone(), id.clone());
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await { let _ = a2.emit("mcp-log", McpLine { id: i2.clone(), line }); }
    });
    procs.insert(id, McpProc { child, stdin });
    Ok(())
}

#[tauri::command]
pub async fn mcp_send(state: State<'_, McpState>, id: String, line: String) -> Result<(), String> {
    let mut procs = state.0.lock().await;
    let p = procs.get_mut(&id).ok_or("that server is not running")?;
    p.stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
    p.stdin.write_all(b"\n").await.map_err(|e| e.to_string())?;
    p.stdin.flush().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn mcp_stop(state: State<'_, McpState>, id: String) -> Result<(), String> {
    let removed = state.0.lock().await.remove(&id);
    if let Some(mut p) = removed { drop(p.stdin); let _ = p.child.kill().await; }
    Ok(())
}

// ---- A terminal for the project folder ---------------------------------------------------------------------------------------------------------------------------
// This is the user's own command line: it runs what they type in the project folder with their permissions and environment, with no sandbox. (The agent never uses it.)

#[derive(Default)]
pub struct TermState(Arc<Mutex<HashMap<String, Child>>>);

#[derive(Serialize, Clone)]
struct TermOut { id: String, text: String }
#[derive(Serialize, Clone)]
struct TermExit { id: String, code: Option<i32> }

#[tauri::command]
pub async fn term_run(app: AppHandle, state: State<'_, TermState>, id: String, root: String, command: String) -> Result<(), String> {
    let root = root_of(&root)?;
    #[cfg(not(windows))]
    let mut cmd = { let mut c = Command::new("sh"); c.arg("-c").arg(&command); c };
    #[cfg(windows)]
    let mut cmd = { let mut c = Command::new("cmd"); c.arg("/C").arg(&command); c };
    cmd.current_dir(&root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let mut child = cmd.spawn().map_err(|e| format!("could not start: {e}"))?;
    let out = child.stdout.take().ok_or("no stdout")?;
    let err = child.stderr.take().ok_or("no stderr")?;
    state.0.lock().await.insert(id.clone(), child);
    let map = state.0.clone();
    tokio::spawn(async move {
        let pump = |stream: Box<dyn tokio::io::AsyncRead + Unpin + Send>, app: AppHandle, id: String| async move {
            let mut r = BufReader::new(stream);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match r.read_until(b'\n', &mut buf).await { Ok(0) | Err(_) => break, Ok(_) => { let _ = app.emit("term-out", TermOut { id: id.clone(), text: String::from_utf8_lossy(&buf).to_string() }); } }
            }
        };
        let a = tokio::spawn(pump(Box::new(out), app.clone(), id.clone()));
        let b = tokio::spawn(pump(Box::new(err), app.clone(), id.clone()));
        let _ = tokio::join!(a, b);
        let code = match map.lock().await.remove(&id) { Some(mut c) => c.wait().await.ok().and_then(|s| s.code()), None => None };
        let _ = app.emit("term-exit", TermExit { id, code });
    });
    Ok(())
}

#[tauri::command]
pub async fn term_kill(state: State<'_, TermState>, id: String) -> Result<(), String> {
    if let Some(c) = state.0.lock().await.get_mut(&id) {
        kill_group(c);
        let _ = c.start_kill();
    }
    Ok(())
}

// ---- Folders a phone may use ---------------------------------------------------------------------------------------------------------------------------------------
// These run on the real, resolved paths (symbolic links followed), so a link inside an allowed folder that points elsewhere does not get a phone outside it.

/// The folder's real path if it lies inside one of the allowed folders; otherwise an error.
#[tauri::command]
pub fn remote_check_root(root: String, allowed: Vec<String>) -> Result<String, String> {
    let real = PathBuf::from(&root).canonicalize().map_err(|e| format!("cannot open that folder: {e}"))?;
    if !real.is_dir() { return Err("that is not a folder".into()); }
    for a in allowed {
        if let Ok(base) = PathBuf::from(a).canonicalize() { if real.starts_with(&base) { return Ok(real.to_string_lossy().to_string()); } }
    }
    Err("that folder is not inside a folder you allowed for remote control".into())
}

/// Creates a new, empty project folder named `name` inside `parent`. The name must be a plain folder name: no separators, no leading dot, and not already taken.
#[tauri::command]
pub fn remote_new_project(parent: String, name: String) -> Result<String, String> {
    let name = name.trim();
    let ok_char = |c: char| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.');
    if name.is_empty() || name.chars().count() > 64 || name.starts_with('.') || !name.chars().all(ok_char) { return Err("a project name can use letters, numbers, spaces, - _ and . (up to 64 characters, not starting with a dot)".into()); }
    let base = PathBuf::from(&parent).canonicalize().map_err(|e| format!("the projects folder is not available: {e}"))?;
    let path = base.join(name);
    if path.exists() { return Err(format!("{name} already exists in the projects folder")); }
    std::fs::create_dir(&path).map_err(|e| e.to_string())?;
    let real = path.canonicalize().map_err(|e| e.to_string())?;
    if !real.starts_with(&base) { let _ = std::fs::remove_dir(&path); return Err("the new folder would be outside the projects folder".into()); }
    Ok(real.to_string_lossy().to_string())
}

/// The folders directly inside `root` (not hidden ones, not dependency folders), for choosing a project.
#[tauri::command]
pub fn project_dirs(root: String) -> Result<Vec<String>, String> {
    let root = root_of(&root)?;
    let mut out: Vec<String> = std::fs::read_dir(&root).map_err(|e| e.to_string())?.flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !n.starts_with('.') && !SKIP_DIRS.contains(&n.as_str()))
        .collect();
    out.sort();
    out.truncate(200);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("clutor-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.canonicalize().unwrap()
    }

    #[test]
    fn jail_refuses_escapes() {
        let root = temp("jail");
        assert!(jail(&root, "a/b.txt").is_ok());
        assert!(jail(&root, "../x").is_err());
        assert!(jail(&root, "a/../../x").is_err());
        assert!(jail(&root, "/etc/passwd").is_err());
        assert!(jail(&root, "").is_err());
        #[cfg(unix)]
        {
            let outside = temp("outside");
            std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
            assert!(jail(&root, "link/secret").is_err());
        }
    }

    #[test]
    fn write_read_delete_roundtrip() {
        let root = temp("rw");
        let r = root.to_string_lossy().to_string();
        assert_eq!(project_read(r.clone(), "src/a.txt".into()).unwrap(), None);
        project_write(r.clone(), "src/a.txt".into(), Some("hello".into())).unwrap();
        assert_eq!(project_read(r.clone(), "src/a.txt".into()).unwrap().as_deref(), Some("hello"));
        assert!(project_list(r.clone()).unwrap().contains(&"src/a.txt".to_string()));
        project_write(r.clone(), "src/a.txt".into(), None).unwrap();
        assert_eq!(project_read(r, "src/a.txt".into()).unwrap(), None);
    }

    #[test]
    fn list_skips_dependency_folders() {
        let root = temp("list");
        std::fs::create_dir_all(root.join("node_modules/x")).unwrap();
        std::fs::write(root.join("node_modules/x/i.js"), "x").unwrap();
        std::fs::write(root.join("main.rs"), "fn main(){}").unwrap();
        assert_eq!(project_list(root.to_string_lossy().to_string()).unwrap(), vec!["main.rs".to_string()]);
    }

    #[test]
    fn binary_files_are_reported_not_returned() {
        let root = temp("bin");
        std::fs::write(root.join("b.bin"), [0xff, 0xfe, 0x00]).unwrap();
        assert!(project_read(root.to_string_lossy().to_string(), "b.bin".into()).unwrap_err().contains("not a text file"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_sees_unapplied_changes_and_leaves_the_project_alone() {
        let root = temp("run");
        std::fs::write(root.join("a.txt"), "old").unwrap();
        let r = project_run(root.to_string_lossy().to_string(), vec![Overlay { path: "a.txt".into(), content: Some("new".into()) }], vec![vec!["cat".into(), "a.txt".into()]], 10_000).await.unwrap();
        assert_eq!(r.exit_code, Some(0));
        assert_eq!(r.output.trim(), "new");
        assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "old");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn run_sees_unapplied_changes_and_leaves_the_project_alone_on_windows() {
        let root = temp("run-win");
        std::fs::write(root.join("a.txt"), "old").unwrap();
        let r = project_run(root.to_string_lossy().to_string(), vec![Overlay { path: "a.txt".into(), content: Some("new".into()) }], vec![vec!["type".into(), "a.txt".into()]], 20_000).await.unwrap();
        assert_eq!(r.exit_code, Some(0), "{}", r.output);
        assert_eq!(r.output.trim(), "new");
        assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "old");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn run_stops_at_the_time_limit_on_windows() {
        let root = temp("limit-win");
        let r = project_run(root.to_string_lossy().to_string(), vec![], vec![vec!["ping".into(), "-n".into(), "30".into(), "127.0.0.1".into()]], 1500).await.unwrap();
        assert!(r.timed_out);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_stops_at_the_time_limit_and_strips_the_environment() {
        let root = temp("limit");
        std::env::set_var("CLUTOR_TEST_SECRET", "s3cret");
        let r = project_run(root.to_string_lossy().to_string(), vec![], vec![vec!["sleep".into(), "30".into()]], 1000).await.unwrap();
        assert!(r.timed_out);
        let e = project_run(root.to_string_lossy().to_string(), vec![], vec![vec!["env".into()]], 5000).await.unwrap();
        assert!(!e.output.contains("s3cret"));
    }
    #[test]
    fn a_folder_must_really_be_inside_an_allowed_one() {
        let base = temp("allow-base");
        let inside = base.join("proj");
        std::fs::create_dir_all(&inside).unwrap();
        let other = temp("allow-other");
        let allowed = vec![base.to_string_lossy().to_string()];
        assert!(remote_check_root(inside.to_string_lossy().to_string(), allowed.clone()).is_ok());
        assert!(remote_check_root(base.to_string_lossy().to_string(), allowed.clone()).is_ok());
        assert!(remote_check_root(other.to_string_lossy().to_string(), allowed.clone()).is_err());
        assert!(remote_check_root(format!("{}/../{}", inside.display(), other.file_name().unwrap().to_string_lossy()), allowed.clone()).is_err());
        assert!(remote_check_root(base.join("missing").to_string_lossy().to_string(), allowed.clone()).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&other, base.join("escape")).unwrap();
            assert!(remote_check_root(base.join("escape").to_string_lossy().to_string(), allowed).is_err(), "a link pointing outside must not pass");
        }
    }

    #[test]
    fn new_projects_get_plain_names_and_never_overwrite() {
        let base = temp("newproj");
        let b = base.to_string_lossy().to_string();
        let made = remote_new_project(b.clone(), "My App".into()).unwrap();
        assert!(std::path::Path::new(&made).is_dir());
        assert!(remote_new_project(b.clone(), "My App".into()).unwrap_err().contains("already exists"));
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "x\u{0}y", &"n".repeat(65)] { assert!(remote_new_project(b.clone(), bad.into()).is_err(), "{bad:?}"); }
    }

    #[test]
    fn project_dirs_lists_visible_folders_only() {
        let base = temp("dirs");
        for d in ["beta", "alpha", ".git", "node_modules"] { std::fs::create_dir_all(base.join(d)).unwrap(); }
        std::fs::write(base.join("file.txt"), "x").unwrap();
        assert_eq!(project_dirs(base.to_string_lossy().to_string()).unwrap(), vec!["alpha".to_string(), "beta".to_string()]);
    }
}
