import { buildGgufHeader, type SyntheticModelSpec } from "@core/gguf/write";
import { EventQueue, NativeError, type Ev, type NativeReady, type Transport } from "@engine/session";
import type { AgentHost, Bridge, InspectApi, RemoteApi, RemoteDevice, RemoteStatus, WebApi, CatalogItem, DownloadProgress, ModelInfo, OpenEngine, SystemInfo } from "./types";

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
  '{"tool":"update_plan","args":{"steps":"[>] Read the greeting code\\n[ ] Add the exclamation mark\\n[ ] Run the tests"}}',
  '{"tool":"read_file","args":{"path":"src/greet.js"}}',
  '{"tool":"update_plan","args":{"steps":"[x] Read the greeting code\\n[>] Add the exclamation mark\\n[ ] Run the tests"}}',
  '{"tool":"edit_file","args":{"path":"src/greet.js","old_text":"Hello, ${name}","new_text":"Hello, ${name}!"}}',
  '{"tool":"update_plan","args":{"steps":"[x] Read the greeting code\\n[x] Add the exclamation mark\\n[>] Run the tests"}}',
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
  async checkRoot(root, allowed) { if (allowed.some((a) => root === a || root.startsWith(`${a}/`)) && !root.includes("..")) return root; throw new Error("that folder is not inside a folder you allowed for remote control"); },
  async newProject(parent, name) { if (!/^[\w .-]{1,64}$/.test(name) || name.startsWith(".")) throw new Error("a project name can use letters, numbers, spaces, - _ and ."); return `${parent}/${name}`; },
  async dirs() { return ["demo-project", "notes"]; },
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
  async send(client, text) { phoneInbox.push({ client, text }); return true; }, async broadcast() { /* no phones */ }, async kick() { /* none */ },
  async log(line) { mockLog.push(line); }, async readLog(max) { return mockLog.slice(-max); },
  async onOpen(cb) { phoneHooks.open = cb; return () => { if (phoneHooks.open === cb) phoneHooks.open = undefined; }; },
  async onMessage(cb) { phoneHooks.message = cb; return () => { if (phoneHooks.message === cb) phoneHooks.message = undefined; }; },
  async onClose(cb) { phoneHooks.close = cb; return () => { if (phoneHooks.close === cb) phoneHooks.close = undefined; }; },
};
// A stand-in phone for developing and testing in the browser preview: window.__mockPhone.open(), .say({...}), .inbox.
const phoneInbox: Array<{ client: number; text: string }> = [];
const phoneHooks: { open?: (c: number, d: RemoteDevice) => void; message?: (c: number, t: string) => void; close?: (c: number) => void } = {};
if (typeof window !== "undefined") {
  (window as unknown as { __mockPhone: unknown }).__mockPhone = {
    inbox: phoneInbox,
    open: () => phoneHooks.open?.(1, { id: "mock", name: "Mock phone", public: "00", created: 1, lastSeen: 1 }),
    say: (v: unknown) => phoneHooks.message?.(1, JSON.stringify(v)),
    close: () => phoneHooks.close?.(1),
  };
}

/** The preview has no files: a pretend model file whose header says it is a large mixture-of-experts model, and a pretend disk. */
const SPEC_SMALL: SyntheticModelSpec = { arch: "llama", name: "Llama 3.2 1B (preview)", layers: 16, embedding: 2048, heads: 32, kvHeads: 8, contextLength: 131072, expertCount: 0, expertsUsed: 0, expertFfn: 0, denseFfn: 8192, vocab: 128256 };
const SPEC_BIG: SyntheticModelSpec = { arch: "qwen3moe", name: "Qwen3 235B-A22B (preview)", layers: 94, embedding: 4096, heads: 64, kvHeads: 4, contextLength: 262144, expertCount: 128, expertsUsed: 8, expertFfn: 1536, denseFfn: 0, vocab: 151936, sharedExperts: 0, quantType: 12 };
const SPEC_VL: SyntheticModelSpec = { arch: "qwen3vlmoe", name: "Qwen3-VL-30B-A3B (preview)", layers: 48, embedding: 2048, heads: 32, kvHeads: 4, contextLength: 262144, expertCount: 128, expertsUsed: 8, expertFfn: 768, denseFfn: 0, vocab: 151936, quantType: 10 };
const specFor = (path: string): SyntheticModelSpec => (/qwen3-vl/i.test(path) ? SPEC_VL : /qwen/i.test(path) ? SPEC_BIG : SPEC_SMALL);

/** The preview's pretend models folder: a chat model or two, a picture-reading model with its projector, and an image generator with its text encoder and VAE. */
const GB = 1_000_000_000;
const PREVIEW_FILES = [
  { path: "/preview/Llama-3.2-1B-Instruct-Q4_K_M.gguf", bytes: buildGgufHeader(SPEC_SMALL).dataOffset + buildGgufHeader(SPEC_SMALL).dataSize },
  { path: "/preview/Qwen3-VL-30B-A3B-Instruct.i1-Q2_K.gguf", bytes: Math.round(10.5 * GB) },
  { path: "/preview/Qwen3-VL-30B-A3B-Instruct.mmproj-f16.gguf", bytes: GB },
  { path: "/preview/qwen-image-2.1-turbo-Q4_K_M.gguf", bytes: 4 * GB },
  { path: "/preview/qwen3vl_8b_int8_convrot.safetensors", bytes: Math.round(9.3 * GB) },
  { path: "/preview/qwen_image_2.1_vae_bf16.safetensors", bytes: 680_000_000 },
];
const previewHeader = (path: string): { head: Uint8Array; size: number } | undefined => {
  const n = path.toLowerCase();
  const file = PREVIEW_FILES.find((f) => f.path === path);
  if (!file) return undefined;
  const enc = new TextEncoder();
  const gguf = (kv: Array<[string, string]>, tensors: Array<{ name: string; dims: number[]; type: number }>): Uint8Array => {
    const out: number[] = [];
    const u32 = (v: number): void => { out.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255); };
    const u64 = (v: number): void => { u32(v); u32(Math.floor(v / 2 ** 32)); };
    const str = (s: string): void => { const b = enc.encode(s); u64(b.length); out.push(...b); };
    out.push(0x47, 0x47, 0x55, 0x46); u32(3); u64(tensors.length); u64(kv.length);
    for (const [k, v] of kv) { str(k); u32(8); str(v); }
    let off = 0;
    for (const t of tensors) { str(t.name); u32(t.dims.length); for (const d of t.dims) u64(d); u32(t.type); u64(off); off += 1 << 12; }
    while (out.length % 32) out.push(0);
    return Uint8Array.from(out);
  };
  if (n.endsWith(".safetensors")) {
    const names = n.includes("vae") ? ["encoder.conv1.weight", "decoder.conv1.weight", "decoder.head.2.weight"] : ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"];
    const json = enc.encode(JSON.stringify(Object.fromEntries(names.map((k) => [k, { dtype: "BF16", shape: [1], data_offsets: [0, 2] }]))));
    const head = new Uint8Array(8 + json.length);
    new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
    head.set(json, 8);
    return { head, size: file.bytes };
  }
  if (n.includes("mmproj")) return { head: gguf([["general.architecture", "clip"], ["general.type", "mmproj"], ["clip.projector_type", "qwen3vl_merger"]], []), size: file.bytes };
  if (n.includes("qwen-image")) {
    const head = gguf([], [{ name: "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight", dims: [64, 64], type: 12 }, { name: "model.diffusion_model.transformer_blocks.0.img_mlp.out.weight", dims: [64, 64], type: 12 }]);
    return { head, size: head.length + 4 * (1 << 12) + 4096 };
  }
  const b = buildGgufHeader(specFor(path));
  return { head: b.header, size: b.dataOffset + b.dataSize };
};
const mockInspect: InspectApi = {
  async size(path) { const p = previewHeader(path); if (p) return p.size; const b = buildGgufHeader(specFor(path)); return b.dataOffset + b.dataSize; },
  async read(path, offset, length) { const p = previewHeader(path); const head = p ? p.head : buildGgufHeader(specFor(path)).header; return head.subarray(offset, Math.min(head.length, offset + length)); },
  async diskBench() { await new Promise((r) => setTimeout(r, 600)); return { sequentialBytesPerSec: 2.4e9, scatteredBytesPerSec: 1.1e9, seconds: 4 }; },
};

export const mockBridge: Bridge = {
  native: false,
  host: mockHost,
  web: mockWeb,
  remote,
  inspect: mockInspect,
  image: {
    async available() { return true; },
    async generate(req, onProgress) {
      for (let s = 1; s <= req.steps; s++) { await new Promise((r) => setTimeout(r, 250)); onProgress({ step: s, steps: req.steps, seconds: 0.25 }); }
      return { path: "/preview/picture.png", seconds: req.steps * 0.25 };
    },
    async cancel() { /* the preview picture is quick */ },
    async read() {
      const c = document.createElement("canvas"); c.width = c.height = 256;
      const x = c.getContext("2d");
      if (x) { const g = x.createLinearGradient(0, 0, 256, 256); g.addColorStop(0, "#3150e0"); g.addColorStop(1, "#c0392b"); x.fillStyle = g; x.fillRect(0, 0, 256, 256); x.fillStyle = "#fff"; x.font = "16px sans-serif"; x.fillText("preview picture", 70, 130); }
      return c.toDataURL("image/png").split(",")[1] as string;
    },
    async saveAs() { return true; },
  },
  cloud: {
    async keySet() { /* nothing is stored in a browser preview */ },
    async keyHas() { return true; },
    async keyClear() { /* nothing to clear */ },
    async request(o, onChunk) {
      const anthropic = o.auth === "anthropic";
      const words = "This is a canned answer from the preview cloud model. In the app it comes from your provider.".split(" ");
      for (const w of words) { await new Promise((r) => setTimeout(r, 25)); onChunk(anthropic ? `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", delta: { type: "text_delta", text: `${w} ` } })}\n\n` : `data: ${JSON.stringify({ choices: [{ delta: { content: `${w} ` } }] })}\n\n`); }
      onChunk(anthropic ? `event: message_stop\ndata: {"type":"message_stop"}\n\n` : `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 20, completion_tokens: words.length } })}\n\ndata: [DONE]\n\n`);
      return { status: 200 };
    },
    async cancel() { /* the canned answer is short */ },
  },
  async system(): Promise<SystemInfo> { return { os: "preview", arch: "browser", ramBytes: 16 * 2 ** 30, cores: 8, modelsDir: "~/.clutor/models" }; },
  async scanFiles() { return PREVIEW_FILES.map((f) => ({ path: f.path, name: f.path.split("/").pop() as string, bytes: f.bytes })); },
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
