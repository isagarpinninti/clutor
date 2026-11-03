import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import type { ChatResult, LlmClient } from "@clutor/core";
import { detectTemplate, renderChat, ThinkSplitter, type TemplateInfo } from "./templates";
import type { ChatMessage, StreamEvent } from "./stream";

/**
 * A conversation with Clutor's own engine (`clutor-engine chat`). The engine loads the model once and keeps the KV cache; this side
 * renders the chat template and sends only the text the engine has not seen, so a turn costs its new tokens, not the whole history.
 */

export interface NativeReady { arch: string; name: string; nCtx: number; nVocab: number; nLayer: number; threads: number; streaming: boolean; locked?: boolean; addBos: boolean; template: string }

interface Ev { ev: string; [k: string]: unknown }

export interface NativeOptions { bin: string; model: string; ctx: number; extraArgs?: string[]; readyTimeoutMs?: number }

export interface SampleOptions { thinking?: boolean; temp?: number; topP?: number; topK?: number; repeat?: number; seed?: number; max?: number; signal?: AbortSignal }

export class NativeError extends Error { constructor(message: string, readonly code?: string) { super(message); this.name = "NativeError"; } }

export class NativeEngine {
  readonly template: TemplateInfo;
  /** The text the engine's KV cache holds, or null when that is unknown (after a cancelled prompt, a refused one, or a failure): the next turn starts over. */
  private fed: string | null = "";
  private tail: Promise<void> = Promise.resolve();
  private readonly queue: Ev[] = [];
  private waiter: ((e: Ev | null) => void) | undefined;
  private closed = false;
  private stderr = "";
  /** What the last turn had to send: how many tokens, and whether the cache was started over. Shown by the UI and used by tests. */
  lastAppend: { tokens: number; reset: boolean } = { tokens: 0, reset: false };

  private constructor(private readonly child: ChildProcess, readonly ready: NativeReady) {
    this.template = detectTemplate(ready.template);
  }

  static async start(o: NativeOptions): Promise<NativeEngine> {
    const child = spawn(o.bin, ["chat", o.model, "--ctx", String(o.ctx), ...(o.extraArgs ?? [])], { stdio: ["pipe", "pipe", "pipe"] });
    let stderr = "";
    child.stderr?.on("data", (b: Buffer) => { stderr = (stderr + b.toString()).slice(-4000); });
    let onLine: ((e: Ev | null) => void) | undefined;
    const rl = createInterface({ input: child.stdout as NodeJS.ReadableStream });
    const lines: Array<Ev | null> = [];
    rl.on("line", (l) => { let e: Ev; try { e = JSON.parse(l) as Ev; } catch { return; } if (onLine) onLine(e); else lines.push(e); });
    rl.on("close", () => { if (onLine) onLine(null); else lines.push(null); });
    child.on("error", () => undefined);
    const first = await new Promise<Ev | null>((resolve) => {
      const timer = setTimeout(() => { child.kill("SIGKILL"); resolve(null); }, o.readyTimeoutMs ?? 600_000);
      const take = (e: Ev | null): void => { clearTimeout(timer); onLine = undefined; resolve(e); };
      if (lines.length) take(lines.shift() as Ev | null); else onLine = take;
    });
    if (!first || first.ev !== "ready") {
      child.kill("SIGKILL");
      throw new NativeError(first?.ev === "error" ? String(first["msg"]) : `the engine did not start${stderr.trim() ? `: ${stderr.trim().split("\n").slice(-2).join(" ")}` : ""}`);
    }
    const eng = new NativeEngine(child, first as unknown as NativeReady);
    for (const e of lines) eng.push(e);
    onLine = (e) => eng.push(e);
    child.on("exit", () => eng.push(null));
    child.stderr?.on("data", (b: Buffer) => { eng.stderr = (eng.stderr + b.toString()).slice(-4000); });
    return eng;
  }

  private push(e: Ev | null): void {
    if (e === null) this.closed = true;
    if (this.waiter && (e === null || this.queue.length === 0)) { const w = this.waiter; this.waiter = undefined; w(e); return; }
    if (e !== null) this.queue.push(e);
  }
  private next(): Promise<Ev | null> {
    if (this.queue.length) return Promise.resolve(this.queue.shift() as Ev | null);
    if (this.closed) return Promise.resolve(null);
    return new Promise((resolve) => { this.waiter = resolve; });
  }
  private send(o: Record<string, unknown>): void {
    if (this.closed || !this.child.stdin?.writable) return;
    this.child.stdin.write(`${JSON.stringify(o)}\n`);
  }
  private async lock(): Promise<() => void> {
    let release!: () => void;
    const prev = this.tail;
    this.tail = new Promise<void>((r) => { release = r; });
    await prev;
    return release;
  }
  private dead(): NativeError { return new NativeError(`the engine stopped${this.stderr.trim() ? `: ${this.stderr.trim().split("\n").slice(-2).join(" ")}` : ""}`); }

  /** Forget the conversation (the next turn re-reads everything it needs). */
  async reset(): Promise<void> {
    const release = await this.lock();
    try { this.send({ op: "reset" }); for (;;) { const e = await this.next(); if (!e) throw this.dead(); if (e.ev === "reset") break; } this.fed = ""; } finally { release(); }
  }

  /** Streams the reply to `messages`. Only text the engine has not seen is sent; a conversation that no longer extends the cache starts over. */
  async *stream(messages: ChatMessage[], o: SampleOptions = {}): AsyncGenerator<StreamEvent> {
    const release = await this.lock();
    let aborted = false;
    const onAbort = (): void => { aborted = true; this.send({ op: "cancel" }); };
    o.signal?.addEventListener("abort", onAbort, { once: true });
    try {
      if (o.signal?.aborted) onAbort();
      const full = renderChat(this.template, messages, { thinking: o.thinking ?? false });
      let toSend: string, restarted = false;
      if (this.fed !== null && full.length > this.fed.length && full.startsWith(this.fed)) toSend = full.slice(this.fed.length);
      else {
        this.send({ op: "reset" });
        for (;;) { const e = await this.next(); if (!e) throw this.dead(); if (e.ev === "reset") break; }
        this.fed = "";
        toSend = full;
        restarted = true;
      }
      this.send({ op: "append", text: toSend });
      let newTokens = 0, prefillSecs = 0;
      for (;;) {
        const e = await this.next();
        if (!e) throw this.dead();
        if (e.ev === "progress") { yield { type: "progress", processed: Number(e["done"]), total: Number(e["total"]) }; continue; }
        if (e.ev === "error") {
          if (e["code"] === "context") throw new NativeError("the conversation no longer fits in the context window: /compact summarises it, /clear starts over", "context");
          this.fed = null;
          throw new NativeError(String(e["msg"]));
        }
        if (e.ev === "appended") {
          newTokens = Number(e["tokens"]); prefillSecs = Number(e["seconds"]);
          this.lastAppend = { tokens: newTokens, reset: restarted };
          if (e["cancelled"]) { this.fed = null; yield { type: "done", finish: "cancel" }; return; }
          this.fed = full;
          break;
        }
      }
      if (aborted) { this.send({ op: "cancel" }); }
      const startsInThink = this.template.thinkTags && o.thinking === true && /<think>\s*$/.test(full);
      const split = new ThinkSplitter(this.template.thinkTags && o.thinking === true, startsInThink);
      this.send({ op: "generate", max: o.max ?? 1024, temp: o.temp ?? 0.7, topP: o.topP ?? 0.95, topK: o.topK ?? 40, repeat: o.repeat ?? 1.0, ...(o.seed !== undefined ? { seed: o.seed } : {}) });
      let raw = "";
      for (;;) {
        const e = await this.next();
        if (!e) throw this.dead();
        if (e.ev === "t") { const s = String(e["s"]); raw += s; for (const p of split.push(s)) yield { type: p.kind, text: p.text }; continue; }
        if (e.ev === "error") { this.fed = null; throw new NativeError(String(e["msg"])); }
        if (e.ev === "done") {
          for (const p of split.end()) yield { type: p.kind, text: p.text };
          this.fed = e["desync"] ? null : full + raw;
          const finish = String(e["finish"]);
          yield {
            type: "done", finish: finish === "stop" ? "stop" : finish,
            promptTokens: Number(e["pos"]) - Number(e["tokens"]), completionTokens: Number(e["tokens"]), tokensPerSecond: Number(e["tokensPerSecond"]),
            ...(prefillSecs > 0 && newTokens > 0 ? { promptPerSecond: newTokens / prefillSecs } : {}),
            ...(e["cache"] ? { cache: e["cache"] as { hitRate: number; gbRead: number } } : {}),
          };
          return;
        }
      }
    } finally {
      o.signal?.removeEventListener("abort", onAbort);
      release();
    }
  }

  /** The whole reply as one string, for callers that do not stream (the coding agent). */
  async complete(messages: ChatMessage[], o: SampleOptions = {}): Promise<{ content: string; promptTokens: number; completionTokens: number; finish: string }> {
    let content = "", promptTokens = 0, completionTokens = 0, finish = "stop";
    for await (const e of this.stream(messages, o)) {
      if (e.type === "content") content += e.text;
      else if (e.type === "done") { promptTokens = e.promptTokens ?? 0; completionTokens = e.completionTokens ?? 0; finish = e.finish ?? "stop"; }
    }
    return { content, promptTokens, completionTokens, finish };
  }

  /** The agent's view of the engine. Replies are still validated by the agent loop; there is no grammar-constrained decoding here. */
  asLlm(name: string, o: SampleOptions = {}): LlmClient {
    return {
      model: name,
      toolMode: "json",
      chat: async (messages, opts): Promise<ChatResult> => {
        const msgs: ChatMessage[] = messages.map((m) => ({ role: m.role === "system" ? "system" : m.role === "assistant" ? "assistant" : "user", content: m.content ?? "" }));
        const r = await this.complete(msgs, { thinking: false, temp: 0.2, max: 1024, ...o, ...(opts?.signal ? { signal: opts.signal } : {}) });
        return { content: r.content, toolCalls: [], finishReason: r.finish, usage: { promptTokens: r.promptTokens, completionTokens: r.completionTokens } };
      },
    };
  }

  /** Ends the engine at once without waiting; for a process that is exiting. */
  killNow(): void { try { this.child.kill("SIGKILL"); } catch { /* already gone */ } }

  async close(): Promise<void> {
    if (this.closed) return;
    try { this.child.stdin?.end(); } catch { /* closed */ }
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => { this.child.kill("SIGKILL"); resolve(); }, 3000);
      this.child.once("exit", () => { clearTimeout(t); resolve(); });
    });
  }
}
