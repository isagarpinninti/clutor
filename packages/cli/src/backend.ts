import { spawn, type ChildProcess } from "node:child_process";
import { accessSync, constants } from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { createHash } from "node:crypto";
import { TASK_MARKER, type LlmClient, type MemoryPlan } from "@clutor/core";

/**
 * Clutor does not contain a transformer. Running a model is llama.cpp's job (its mmap loading is what actually lets a model larger
 * than RAM run from an SSD); this module finds `llama-server`, works out which flags that particular build supports, starts it with
 * settings derived from the memory plan, and supervises it.
 */

const exe = process.platform === "win32" ? "llama-server.exe" : "llama-server";

export function findLlamaServer(env: NodeJS.ProcessEnv = process.env, extraDirs: string[] = []): string | null {
  const dirs = [...extraDirs, ...(env["PATH"] ?? "").split(delimiter), "/opt/homebrew/bin", "/usr/local/bin", join(homedir(), ".clutor", "bin")];
  for (const d of dirs) {
    if (!d) continue;
    const p = join(d, exe);
    try { accessSync(p, constants.X_OK); return p; } catch { /* keep looking */ }
  }
  return null;
}

export interface LlamaCapabilities {
  flags: Set<string>;
  /** `--flash-attn on|off|auto` (newer builds) rather than a bare `--flash-attn` switch. */
  flashAttnTakesValue: boolean;
  helpText: string;
}

/** Runs `llama-server --help` and records which options this build knows, so unsupported flags are never passed. */
export function probeLlamaServer(binary: string, timeoutMs = 15_000): Promise<LlamaCapabilities> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, ["--help"], { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new Error(`${binary} --help did not finish in ${timeoutMs / 1000}s`)); }, timeoutMs);
    child.stdout?.on("data", (b: Buffer) => (out += b.toString()));
    child.stderr?.on("data", (b: Buffer) => (out += b.toString()));
    child.on("error", (e) => { clearTimeout(timer); reject(e); });
    child.on("close", () => {
      clearTimeout(timer);
      resolve({ flags: new Set(out.match(/--[a-z0-9][a-z0-9-]*/g) ?? []), flashAttnTakesValue: /--flash-attn[^\n]*\[on\|off\|auto\]|--flash-attn\s+\[?on/.test(out), helpText: out });
    });
  });
}

export interface LlamaArgsInput {
  modelPath: string;
  port: number;
  contextTokens: number;
  kvBits: 16 | 8 | 4;
  threads?: number;
  plan?: Pick<MemoryPlan, "strategy">;
  caps: LlamaCapabilities;
  extraArgs?: string[];
  /** Where llama-server may save and restore KV-cache snapshots. */
  slotSavePath?: string;
}

export function buildLlamaArgs(i: LlamaArgsInput): { args: string[]; notes: string[] } {
  const notes: string[] = [];
  const args = ["-m", i.modelPath, "--host", "127.0.0.1", "--port", String(i.port), "-c", String(i.contextTokens)];
  if (i.threads) args.push("-t", String(i.threads));
  if (i.caps.flags.has("--jinja")) args.push("--jinja");
  else notes.push("this llama-server has no --jinja, so native tool calling may not work; use agent.toolMode \"json\"");
  if (i.kvBits < 16) {
    const t = i.kvBits === 8 ? "q8_0" : "q4_0";
    if (i.caps.flags.has("--cache-type-k") && i.caps.flags.has("--cache-type-v")) {
      args.push("--cache-type-k", t, "--cache-type-v", t);
      // A quantised V cache needs flash attention in llama.cpp.
      if (i.caps.flags.has("--flash-attn")) args.push(...(i.caps.flashAttnTakesValue ? ["--flash-attn", "on"] : ["--flash-attn"]));
      else notes.push("quantised KV cache needs flash attention, which this build lacks; the server may refuse to start");
    } else notes.push("this llama-server cannot quantise the KV cache; the plan's KV size is too small");
  }
  // When the model fits, pinning it avoids page-ins. When it does not, the operating system's page cache IS the expert cache: leave mmap alone.
  if (i.plan?.strategy === "fully-resident" && i.caps.flags.has("--mlock")) args.push("--mlock");
  if (i.plan && i.plan.strategy !== "fully-resident") notes.push("the model is larger than the memory budget, so it will be paged in from disk by the operating system (mmap): speed depends on your SSD and on how often the same experts are reused");
  if (i.slotSavePath) {
    if (i.caps.flags.has("--slot-save-path")) args.push("--slot-save-path", i.slotSavePath);
    else notes.push("this llama-server has no --slot-save-path, so the project prefix cannot be saved between runs");
  }
  args.push(...(i.extraArgs ?? []));
  return { args, notes };
}

export type BackendState = "stopped" | "starting" | "ready" | "failed";

export class BackendManager {
  state: BackendState = "stopped";
  error: string | null = null;
  private child: ChildProcess | null = null;
  private readonly log: string[] = [];
  private readonly listeners = new Set<(s: BackendState) => void>();

  constructor(readonly port: number, private readonly fetchFn: typeof globalThis.fetch = globalThis.fetch) {}

  get baseUrl(): string { return `http://127.0.0.1:${this.port}/v1`; }
  get recentLog(): string[] { return this.log.slice(-200); }
  onState(f: (s: BackendState) => void): () => void { this.listeners.add(f); return () => this.listeners.delete(f); }
  private set(s: BackendState, err?: string): void { this.state = s; this.error = err ?? null; for (const l of this.listeners) l(s); }

  async start(binary: string, args: string[], opts: { readyTimeoutMs?: number; env?: NodeJS.ProcessEnv } = {}): Promise<void> {
    if (this.child) throw new Error("the backend is already running");
    this.log.length = 0;
    this.set("starting");
    const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"], env: opts.env ?? process.env });
    this.child = child;
    const take = (b: Buffer): void => { for (const l of b.toString().split("\n")) if (l.trim()) { this.log.push(l); if (this.log.length > 1000) this.log.shift(); } };
    child.stdout?.on("data", take);
    child.stderr?.on("data", take);
    child.on("error", (e) => { this.child = null; this.set("failed", `could not start ${binary}: ${e.message}`); });
    child.on("exit", (code, sig) => {
      this.child = null;
      if (this.state !== "stopped") this.set("failed", `llama-server exited (${sig ?? `code ${code}`}); last output: ${this.log.slice(-3).join(" | ")}`);
    });
    const deadline = Date.now() + (opts.readyTimeoutMs ?? 600_000);
    while (Date.now() < deadline) {
      if (this.state === "failed") throw new Error(this.error ?? "the backend failed to start");
      try {
        const r = await this.fetchFn(`http://127.0.0.1:${this.port}/health`);
        if (r.ok) { this.set("ready"); return; }
      } catch { /* not listening yet */ }
      await new Promise((r) => setTimeout(r, 250));
    }
    await this.stop();
    this.set("failed", "the model did not finish loading in time");
    throw new Error(this.error ?? "timeout");
  }

  async stop(): Promise<void> {
    const c = this.child;
    this.set("stopped");
    if (!c) return;
    await new Promise<void>((resolve) => {
      const kill = setTimeout(() => c.kill("SIGKILL"), 5000);
      c.once("exit", () => { clearTimeout(kill); resolve(); });
      c.kill("SIGTERM");
    });
    this.child = null;
  }
}


/**
 * Saves and restores a llama-server slot's KV cache (POST /slots/<id>?action=save|restore). Used so the project map and instructions,
 * which every agent step starts with, are not recomputed on the next run. Every call is best effort: a refusal just means a slower first step.
 */
export class SlotSnapshots {
  readonly stats = { restored: 0, saved: 0, restoreMisses: 0, failures: 0 };
  constructor(private readonly baseUrl: string, private readonly slot = 0, private readonly fetchFn: typeof globalThis.fetch = globalThis.fetch) {}
  private url(action: string): string { return `${this.baseUrl.replace(/\/v1\/?$/, "")}/slots/${this.slot}?action=${action}`; }
  private async post(action: "save" | "restore", filename: string): Promise<Response | undefined> {
    try {
      return await this.fetchFn(this.url(action), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ filename }), signal: AbortSignal.timeout(60_000) });
    } catch { this.stats.failures++; return undefined; }
  }
  async restore(name: string): Promise<boolean> {
    const r = await this.post("restore", `${name}.bin`);
    if (r?.ok) { this.stats.restored++; return true; }
    if (r) { r.body?.cancel().catch(() => undefined); this.stats.restoreMisses++; }
    return false;
  }
  async save(name: string): Promise<boolean> {
    const r = await this.post("save", `${name}.bin`);
    if (r?.ok) { this.stats.saved++; return true; }
    if (r) { this.stats.failures++; r.body?.cancel().catch(() => undefined); }
    return false;
  }
}

/** Wraps a model client so the shared prefix (instructions + project map) is restored before the first call and saved after it. */
export function withPrefixSnapshots(inner: LlmClient, snaps: SlotSnapshots): LlmClient & { snapshots: SlotSnapshots } {
  let done = false;
  const wrapped: LlmClient & { snapshots: SlotSnapshots } = {
    snapshots: snaps,
    model: inner.model,
    toolMode: inner.toolMode,
    async chat(messages, opts) {
      if (done) return inner.chat(messages, opts);
      done = true;
      const head = messages.map((m) => m.content ?? "").join("\n\n");
      const cut = head.lastIndexOf(TASK_MARKER);
      const prefix = cut >= 0 ? head.slice(0, cut) : head;
      const key = `clt-${createHash("sha256").update(inner.model).update("\0").update(prefix).digest("hex").slice(0, 32)}`;
      const had = await snaps.restore(key);
      const out = await inner.chat(messages, opts);
      if (!had) await snaps.save(key);
      return out;
    },
  };
  return wrapped;
}
