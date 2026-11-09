import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
import { EventQueue, NativeError, type Ev, type NativeReady, type Transport } from "@engine/session";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import type { AgentHost, Bridge, CloudApi, FileInfo, ImageApi, ImageProgress, DiskSpeed, InspectApi, RemoteApi, RemoteDevice, WebApi, CatalogItem, DownloadProgress, ModelInfo, OpenEngine, SystemInfo } from "./types";

/** The engine runs inside this process (src-tauri links it); its events arrive as Tauri events and its operations go out as commands. */
class TauriTransport implements Transport {
  readonly queue = new EventQueue();
  private unlisten: UnlistenFn | undefined;
  private lastError = "";
  async attach(): Promise<void> {
    this.unlisten = await listen<string>("engine-event", (e) => {
      try { this.queue.push(JSON.parse(e.payload) as Ev); } catch { /* not an event */ }
    });
  }
  send(op: Record<string, unknown>): void {
    invoke("engine_op", { op: JSON.stringify(op) }).catch((err: unknown) => {
      this.lastError = String(err);
      this.queue.push({ ev: "error", msg: this.lastError });
    });
  }
  kill(): void { void invoke("engine_close").catch(() => undefined); this.unlisten?.(); this.queue.push(null); }
  async close(): Promise<void> { this.unlisten?.(); await invoke("engine_close").catch(() => undefined); this.queue.push(null); }
  failure(): string { return this.lastError; }
}

const host: AgentHost = {
  async pickFolder() { const r = await open({ directory: true, multiple: false, title: "Choose a project folder" }); return typeof r === "string" ? r : null; },
  list: (root) => invoke<string[]>("project_list", { root }),
  read: (root, path) => invoke<string | null>("project_read", { root, path }),
  write: (root, path, content) => invoke("project_write", { root, path, content }),
  run: (root, overlay, steps, timeoutMs) => invoke("project_run", { root, overlay, steps, timeoutMs }),
  async mcpStart(id, l, onLine, onExit, onLog) {
    const un = [
      await listen<{ id: string; line: string }>("mcp-line", (e) => { if (e.payload.id === id) onLine(e.payload.line); }),
      await listen<{ id: string; line: string }>("mcp-log", (e) => { if (e.payload.id === id) onLog(e.payload.line); }),
      await listen<{ id: string }>("mcp-exit", (e) => { if (e.payload.id === id) { un.forEach((f) => f()); onExit(); } }),
    ];
    try { await invoke("mcp_start", { id, command: l.command, args: l.args, env: l.env, cwd: l.cwd ?? null }); } catch (e) { un.forEach((f) => f()); throw e; }
  },
  async termRun(id, root, command, onOut, onExit) {
    const un = [
      await listen<{ id: string; text: string }>("term-out", (e) => { if (e.payload.id === id) onOut(e.payload.text); }),
      await listen<{ id: string; code: number | null }>("term-exit", (e) => { if (e.payload.id === id) { un.forEach((f) => f()); onExit(e.payload.code); } }),
    ];
    try { await invoke("term_run", { id, root, command }); } catch (e) { un.forEach((f) => f()); throw e; }
  },
  termKill: (id) => invoke("term_kill", { id }),
  checkRoot: (root, allowed) => invoke("remote_check_root", { root, allowed }),
  newProject: (parent, name) => invoke("remote_new_project", { parent, name }),
  dirs: (root) => invoke("project_dirs", { root }),
  mcpSend: (id, line) => invoke("mcp_send", { id, line }),
  mcpStop: (id) => invoke("mcp_stop", { id }),
};

const web: WebApi = {
  search: (query) => invoke("web_search", { query }),
  fetch: (url) => invoke("web_fetch", { url }),
  openExternal: (url) => (/^https?:\/\//.test(url) ? openUrl(url) : Promise.reject(new Error("only web addresses can be opened"))),
};

const remote: RemoteApi = {
  status: () => invoke("remote_status"),
  start: (keepAwake) => invoke("remote_start", { keepAwake }),
  stop: () => invoke("remote_stop"),
  setAwake: (on) => invoke("remote_set_awake", { on }),
  pairStart: () => invoke("remote_pair_start"),
  pairCancel: () => invoke("remote_pair_cancel"),
  revoke: (id) => invoke("remote_revoke", { id }),
  send: (client, text) => invoke("remote_send", { client, text }),
  broadcast: (text) => invoke("remote_broadcast", { text }),
  kick: (client) => invoke("remote_kick", { client }),
  log: (line) => invoke("remote_log", { line }),
  readLog: (max) => invoke("remote_log_read", { max }),
  onOpen: (cb) => listen<{ client: number; device: RemoteDevice }>("remote-open", (e) => cb(e.payload.client, e.payload.device)),
  onMessage: (cb) => listen<{ client: number; text: string }>("remote-msg", (e) => cb(e.payload.client, e.payload.text)),
  onClose: (cb) => listen<{ client: number }>("remote-close", (e) => cb(e.payload.client)),
};

const cloudWaiters = new Map<string, { onChunk: (t: string) => void; done: (r: { status: number; error?: string }) => void }>();
let cloudListening: Promise<void> | undefined;
function listenCloud(): Promise<void> {
  cloudListening ??= (async () => {
    await listen<{ id: string; text: string }>("cloud-chunk", (e) => cloudWaiters.get(e.payload.id)?.onChunk(e.payload.text));
    await listen<{ id: string; status: number; error?: string | null }>("cloud-end", (e) => { const w = cloudWaiters.get(e.payload.id); if (w) { cloudWaiters.delete(e.payload.id); w.done({ status: e.payload.status, ...(e.payload.error ? { error: e.payload.error } : {}) }); } });
  })();
  return cloudListening;
}
const cloud: CloudApi = {
  keySet: (provider, key) => invoke("cloud_key_set", { provider, key }),
  keyHas: (provider) => invoke<boolean>("cloud_key_has", { provider }),
  keyClear: (provider) => invoke("cloud_key_clear", { provider }),
  async request(o, onChunk) {
    await listenCloud();
    return new Promise((resolve) => {
      cloudWaiters.set(o.id, { onChunk, done: resolve });
      invoke("cloud_request", { id: o.id, provider: o.provider, url: o.url, auth: o.auth, body: o.body }).catch((e: unknown) => { cloudWaiters.delete(o.id); resolve({ status: 0, error: String(e) }); });
    });
  },
  cancel: (id) => invoke("cloud_cancel", { id }),
};

const image: ImageApi = {
  available: () => invoke<boolean>("image_available").catch(() => false),
  async generate(req, onProgress) {
    const un = await listen<ImageProgress>("image-progress", (e) => onProgress(e.payload));
    try { return await invoke<{ path: string; seconds: number }>("image_generate", { req }); }
    catch (e) { throw new Error(String(e), { cause: e }); }
    finally { un(); }
  },
  cancel: () => invoke("image_cancel"),
  read: (path) => invoke<string>("image_read", { path }),
  async saveAs(path, suggestedName) {
    const dest = await save({ defaultPath: suggestedName, filters: [{ name: "PNG picture", extensions: ["png"] }] });
    if (!dest) return false;
    await invoke("image_save_as", { path, dest });
    return true;
  },
};

const inspect: InspectApi = {
  size: (path) => invoke("model_size", { path }),
  async read(path, offset, length) {
    const b64 = await invoke<string>("model_read", { path, offset, length });
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  },
  diskBench: (path, seconds) => invoke<DiskSpeed>("disk_bench", { path, seconds }),
};

export const tauriBridge: Bridge = {
  native: true,
  host,
  web,
  cloud,
  image,
  remote,
  inspect,
  system: () => invoke<SystemInfo>("system_info"),
  listModels: (folders) => invoke<ModelInfo[]>("list_models", { folders: folders ?? [] }),
  scanFiles: (folders) => invoke<FileInfo[]>("scan_files", { folders: folders ?? [] }),
  catalog: () => invoke<CatalogItem[]>("model_catalog"),
  revealModels: () => invoke("reveal_models"),
  async notify(title, body) {
    try {
      let ok = await isPermissionGranted();
      if (!ok) ok = (await requestPermission()) === "granted";
      if (ok) sendNotification({ title, body });
    } catch { /* no notifications on this system */ }
  },
  async download(id, onProgress) {
    const un = await listen<DownloadProgress>("download-progress", (e) => { if (e.payload.id === id) onProgress(e.payload); });
    try { await invoke("download_model", { id }); } finally { un(); }
  },
  async openEngine(path, ctx, lock): Promise<OpenEngine> {
    const transport = new TauriTransport();
    await transport.attach();
    try {
      const ready = await invoke<string>("engine_open", { path, ctx, lock });
      return { transport, ready: JSON.parse(ready) as NativeReady };
    } catch (e) {
      await transport.close();
      throw new NativeError(String(e));
    }
  },
};
