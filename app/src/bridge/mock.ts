import { EventQueue, NativeError, type Ev, type NativeReady, type Transport } from "@engine/session";
import type { Bridge, CatalogItem, DownloadProgress, ModelInfo, OpenEngine, SystemInfo } from "./types";

/** A browser preview with no engine: models, downloads and replies are simulated so the interface can be developed and tested without the native shell. */

const REPLY = [
  "Here is a short answer, written by the **preview engine** (no model is running).\n\n",
  "- Lists, **bold**, `inline code` and tables render as they will in the app.\n",
  "- Code blocks get syntax colours and a copy button.\n\n",
  "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n",
  "| Model | Size | Speed |\n|---|---|---|\n| Llama 3.2 1B | 0.8 GB | 30 tok/s |\n| Qwen3.5 9B | 5.6 GB | 8 tok/s |\n\n",
  "Open the app on your computer or phone to chat with a real model on Clutor's engine.",
].join("");

class MockTransport implements Transport {
  readonly queue = new EventQueue();
  private cancelled = false;
  send(op: Record<string, unknown>): void {
    const emit = (e: Ev): void => this.queue.push(e);
    if (op["op"] === "cancel") { this.cancelled = true; return; }
    if (op["op"] === "reset") { setTimeout(() => emit({ ev: "reset" }), 5); return; }
    if (op["op"] === "append") {
      const n = Math.max(1, Math.round(String(op["text"] ?? "").length / 4));
      setTimeout(() => { emit({ ev: "progress", done: n, total: n }); emit({ ev: "appended", tokens: n, requested: n, pos: n, seconds: 0.05, cancelled: false }); }, 80);
      return;
    }
    if (op["op"] === "generate") {
      this.cancelled = false;
      const pieces = REPLY.match(/\S+\s*|\s+/g) ?? [];
      let i = 0;
      const tick = (): void => {
        if (this.cancelled || i >= pieces.length) {
          emit({ ev: "done", finish: this.cancelled ? "cancel" : "stop", tokens: i, seconds: i * 0.03, tokensPerSecond: 33.3, pos: 40 + i, desync: false, ids: [] });
          return;
        }
        emit({ ev: "t", s: pieces[i++] });
        setTimeout(tick, 28);
      };
      setTimeout(tick, 120);
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

export const mockBridge: Bridge = {
  native: false,
  async system(): Promise<SystemInfo> { return { os: "preview", arch: "browser", ramBytes: 16 * 2 ** 30, cores: 8, modelsDir: "~/.clutor/models" }; },
  async listModels() { return MODELS; },
  async catalog() { return CATALOG; },
  async revealModels() { /* nothing to open in a browser */ },
  async download(id, onProgress: (p: DownloadProgress) => void) {
    const item = CATALOG.find((c) => c.id === id);
    if (!item) throw new Error("unknown download");
    for (let i = 1; i <= 10; i++) { await new Promise((r) => setTimeout(r, 120)); onProgress({ id, done: (item.bytes * i) / 10, total: item.bytes, state: "downloading" }); }
    onProgress({ id, done: item.bytes, total: item.bytes, state: "done" });
  },
  async openEngine(path): Promise<OpenEngine> {
    await new Promise((r) => setTimeout(r, 400));
    if (path.includes("fail")) throw new NativeError("preview: this model cannot be opened");
    const ready: NativeReady = { arch: "llama", name: path.split("/").pop() ?? path, nCtx: 4096, nVocab: 128256, nLayer: 16, threads: 8, streaming: false, locked: true, addBos: true, template: "<|im_start|>" };
    return { transport: new MockTransport(), ready };
  },
};
