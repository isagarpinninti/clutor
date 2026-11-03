import { EventQueue, NativeError, type Ev, type NativeReady, type Transport } from "@engine/session";
import type { AgentHost, Bridge, RemoteApi, RemoteDevice, RemoteStatus, WebApi, CatalogItem, DownloadProgress, ModelInfo, OpenEngine, SystemInfo } from "./types";

/** A browser preview with no engine: models, downloads and replies are simulated so the interface can be developed and tested without the native shell. */

const REPLY = [
  "Here is a short answer, written by the **preview engine** (no model is running).\n\n",
  "- Lists, **bold**, `inline code` and tables render as they will in the app.\n",
  "- Code blocks get syntax colours and a copy button.\n\n",
  "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n",
  "| Model | Size | Speed |\n|---|---|---|\n| Llama 3.2 1B | 0.8 GB | 30 tok/s |\n| Qwen3.5 9B | 5.6 GB | 8 tok/s |\n\n",
  "Open the app on your computer or phone to chat with a real model on Clutor's engine.",
].join("");

const SCRIPT = [
  '{"tool":"read_file","args":{"path":"src/greet.js"}}',
  '{"tool":"edit_file","args":{"path":"src/greet.js","old_text":"Hello, ${name}","new_text":"Hello, ${name}!"}}',
  '{"tool":"run_command","args":{"command":"npm test"}}',
  '{"tool":"finish","args":{"summary":"Added the exclamation mark to the greeting. The tests pass."}}',
];

class MockTransport implements Transport {
  readonly queue = new EventQueue();
  private cancelled = false;
  private fed = "";
  send(op: Record<string, unknown>): void {
    const emit = (e: Ev): void => this.queue.push(e);
    if (op["op"] === "cancel") { this.cancelled = true; return; }
    if (op["op"] === "reset") { this.fed = ""; setTimeout(() => emit({ ev: "reset" }), 5); return; }
    if (op["op"] === "append") {
      this.fed += String(op["text"] ?? "");
      const n = Math.max(1, Math.round(String(op["text"] ?? "").length / 4));
      setTimeout(() => { emit({ ev: "progress", done: n, total: n }); emit({ ev: "appended", tokens: n, requested: n, pos: n, seconds: 0.05, cancelled: false }); }, 80);
      return;
    }
    if (op["op"] === "generate") {
      this.cancelled = false;
      // The agent talks in JSON; the preview plays a fixed script so the agent screens can be developed without a model.
      const agent = this.fed.includes("Reply with exactly one JSON object");
      const reply = agent ? (SCRIPT[Math.min(this.fed.split("Result of ").length - 1, SCRIPT.length - 1)] as string) : REPLY;
      const pieces = reply.match(/\S+\s*|\s+/g) ?? [];
      let i = 0;
      const tick = (): void => {
        if (this.cancelled || i >= pieces.length) {
          emit({ ev: "done", finish: this.cancelled ? "cancel" : "stop", tokens: i, seconds: i * 0.03, tokensPerSecond: 33.3, pos: 40 + i, desync: false, ids: [] });
          return;
        }
        emit({ ev: "t", s: pieces[i++] });
        setTimeout(tick, Number(localStorage.getItem("clutor.mock.delay") ?? 28));
      };
      setTimeout(tick, Number(localStorage.getItem("clutor.mock.first") ?? 120));
    }
  }
  kill(): void { this.queue.push(null); }
  async close(): Promise<void> { this.queue.push(null); }
  failure(): string { return ""; }
}

const MODELS: ModelInfo[] = [
  { path: "/preview/Llama-3.2-1B-Instruct-Q4_K_M.gguf", name: "Llama-3.2-1B-Instruct-Q4_K_M", bytes: 770_000_000 },
  { path: "/preview/Qwen3.5-9B-Q4_K_M.gguf", name: "Qwen3.5-9B-Q4_K_M", bytes: 5_600_000_000 },
];

const CATALOG: CatalogItem[] = [
  { id: "llama-3.2-1b-q4_k_m", name: "Llama 3.2 1B Instruct", bytes: 807_000_000, description: "Small and fast; good for a phone or an old laptop." },
  { id: "qwen2.5-1.5b-q4_k_m", name: "Qwen2.5 1.5B Instruct", bytes: 1_117_000_000, description: "A little larger; multilingual." },
  { id: "granite-3.0-3b-a800m-q4_k_m", name: "Granite 3.0 3B-A800M", bytes: 2_060_000_000, description: "Mixture of experts: 3B parameters, 0.8B active per token." },
];

const DEMO: Record<string, string> = {
  "README.md": "# Demo project\n\nA tiny project for trying the Code tab in the browser preview.\n",
  "package.json": '{ "name": "demo", "scripts": { "test": "node --test" } }\n',
  "src/greet.js": "export function greet(name) {\n  return `Hello, ${name}`;\n}\n",
};
const demoFiles = new Map(Object.entries(DEMO));

const mockHost: AgentHost = {
  async pickFolder() { return "/preview/demo-project"; },
  async list() { return [...demoFiles.keys()].filter((k) => !k.startsWith(".clutor")).sort(); },
  async read(_r, p) { return demoFiles.get(p) ?? null; },
  async write(_r, p, c) { if (c === null) demoFiles.delete(p); else demoFiles.set(p, c); },
  async run(_r, overlay) { await new Promise((r) => setTimeout(r, 400)); return { exit_code: 0, output: `preview: pretended to run the tests with ${overlay.length} changed file(s)\n# pass 1`, timed_out: false, duration_ms: 400 }; },
  async mcpStart() { throw new Error("MCP servers run in the desktop app, not in the browser preview"); },
  async mcpSend() { /* none */ },
  async mcpStop() { /* none */ },
  async termRun(_id, _root, command, onOut, onExit) {
    setTimeout(() => onOut(`preview terminal: pretending to run "${command}"\n`), 150);
    setTimeout(() => { onOut("ok\n"); onExit(0); }, 500);
  },
  async termKill() { /* nothing runs */ },
};

const mockWeb: WebApi = {
  async search(q) {
    await new Promise((r) => setTimeout(r, 300));
    return [
      { title: `About ${q}`, url: "https://example.org/about", snippet: "A short description from the preview search." },
      { title: `${q}: a guide`, url: "https://example.com/guide", snippet: "Another preview result." },
    ];
  },
  async fetch(url) {
    await new Promise((r) => setTimeout(r, 300));
    return { url, title: "Preview page", text: "This is a pretend page, shown by the browser preview. In the app, real pages are read as text.", truncated: false, status: 200 };
  },
  async openExternal(url) { window.open(url, "_blank", "noopener"); },
};

let mockRemote: RemoteStatus = { on: false, addresses: [], hostId: "ab".repeat(32), devices: [{ id: "1a2b3c4d5e6f7a8b", name: "Pixel 8", public: "00", created: 1_700_000_000, lastSeen: 1_700_000_500 }], clients: 0, offerOpen: false, awake: false };
const mockLog: string[] = [];
const remote: RemoteApi = {
  async status() { return mockRemote; },
  async start(keepAwake) { mockRemote = { ...mockRemote, on: true, port: 8765, addresses: ["192.168.1.20:8765"], awake: keepAwake }; return mockRemote; },
  async stop() { mockRemote = { ...mockRemote, on: false, addresses: [], clients: 0, awake: false, offerOpen: false }; return mockRemote; },
  async setAwake(on) { mockRemote = { ...mockRemote, awake: on && mockRemote.on }; return mockRemote; },
  async pairStart() { mockRemote = { ...mockRemote, offerOpen: true }; return { qr: JSON.stringify({ v: 1, name: "preview", key: "preview", id: mockRemote.hostId, addrs: mockRemote.addresses, secret: "preview" }), expiresIn: 120, addresses: mockRemote.addresses }; },
  async pairCancel() { mockRemote = { ...mockRemote, offerOpen: false }; },
  async revoke(id) { mockRemote = { ...mockRemote, devices: mockRemote.devices.filter((d: RemoteDevice) => d.id !== id) }; return mockRemote; },
  async send() { return false; }, async broadcast() { /* no phones */ }, async kick() { /* none */ },
  async log(line) { mockLog.push(line); }, async readLog(max) { return mockLog.slice(-max); },
  async onOpen() { return () => undefined; }, async onMessage() { return () => undefined; }, async onClose() { return () => undefined; },
};

export const mockBridge: Bridge = {
  native: false,
  host: mockHost,
  web: mockWeb,
  remote,
  async system(): Promise<SystemInfo> { return { os: "preview", arch: "browser", ramBytes: 16 * 2 ** 30, cores: 8, modelsDir: "~/.clutor/models" }; },
  async listModels(folders) { return folders?.length ? [...MODELS, { path: `${folders[0]}/Custom-Model-7B-Q4_K_M.gguf`, name: "Custom-Model-7B-Q4_K_M", bytes: 4_400_000_000 }] : MODELS; },
  async catalog() { return CATALOG; },
  async revealModels() { /* nothing to open in a browser */ },
  async notify(title, body) {
    try { if (typeof Notification !== "undefined" && Notification.permission !== "denied") { if (Notification.permission !== "granted") await Notification.requestPermission(); if (Notification.permission === "granted") new Notification(title, { body }); } } catch { /* blocked */ }
  },
  async download(id, onProgress: (p: DownloadProgress) => void) {
    const item = CATALOG.find((c) => c.id === id);
    if (!item) throw new Error("unknown download");
    for (let i = 1; i <= 10; i++) { await new Promise((r) => setTimeout(r, 120)); onProgress({ id, done: (item.bytes * i) / 10, total: item.bytes, state: "downloading" }); }
    onProgress({ id, done: item.bytes, total: item.bytes, state: "done" });
  },
  async openEngine(path): Promise<OpenEngine> {
    await new Promise((r) => setTimeout(r, Number(localStorage.getItem("clutor.mock.load") ?? 400)));
    if (path.includes("fail")) throw new NativeError("preview: this model cannot be opened");
    const ready: NativeReady = { arch: "llama", name: path.split("/").pop() ?? path, nCtx: 4096, nVocab: 128256, nLayer: 16, threads: 8, streaming: false, locked: true, addBos: true, template: "<|im_start|>" };
    return { transport: new MockTransport(), ready };
  },
};
