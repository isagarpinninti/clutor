import { detectTemplate, renderChat, ThinkSplitter, type TemplateInfo } from "./templates";
import type { ChatMessage, StreamEvent } from "./stream";

/**
 * A conversation with Clutor's engine, independent of how the engine is reached (a child process in the terminal, an in-process library in the app). The engine
 * keeps the KV cache; this side renders the chat template and sends only the text the engine has not seen, so a turn costs its new tokens, not the whole history.
 * Nothing in this file touches the operating system, so the graphical app runs the same code.
 */

export interface NativeReady { arch: string; name: string; nCtx: number; nVocab: number; nLayer: number; threads: number; streaming: boolean; /** True when the model did not fit in memory and the engine chose to stream its experts from disk by itself. */ autoStream?: boolean; locked?: boolean; addBos: boolean; template: string;
  /** The longest context the model was trained for, and the memory the attention cache takes per token (both from the engine). */
  nCtxTrain?: number; kvBytesPerToken?: number;
  /** The model was opened with a vision projector and takes pictures. */
  vision?: boolean }

export interface Ev { ev: string; [k: string]: unknown }

export interface SampleOptions {
  /** Text the reply has already begun with (a paused background job resumes from it): the engine continues after it instead of starting the reply again. */
  prefill?: string;
  thinking?: boolean; temp?: number; topP?: number; topK?: number; repeat?: number; seed?: number; max?: number; signal?: AbortSignal;
  /** Drop tokens much less likely than the best one (min-p, 0 = off); gentler than top-p for small models. */
  minP?: number;
  /** Constrained decoding: the reply can only be one valid JSON object (the engine refuses any token that would break it) and ends when the object does. For the coding agent's tool calls. */
  json?: boolean }

/** The part of an already-written reply that is JSON: from its first brace (anything before it is prose the engine need not check). */
export function jsonPrefix(prefill: string | undefined): string { const i = (prefill ?? "").indexOf("{"); return i < 0 ? "" : (prefill as string).slice(i); }

export class NativeError extends Error { constructor(message: string, readonly code?: string) { super(message); this.name = "NativeError"; } }

/** Events from the engine, in order; `null` means the engine is gone. */
export class EventQueue {
  private readonly items: Ev[] = [];
  private waiter: ((e: Ev | null) => void) | undefined;
  closed = false;
  push(e: Ev | null): void {
    if (e === null) this.closed = true;
    if (this.waiter && (e === null || this.items.length === 0)) { const w = this.waiter; this.waiter = undefined; w(e); return; }
    if (e !== null) this.items.push(e);
  }
  next(): Promise<Ev | null> {
    if (this.items.length) return Promise.resolve(this.items.shift() as Ev);
    if (this.closed) return Promise.resolve(null);
    return new Promise((resolve) => { this.waiter = resolve; });
  }
}

/** How a session talks to the engine. */
export interface Transport {
  readonly queue: EventQueue;
  send(op: Record<string, unknown>): void;
  /** Ends the engine at once; for a program that is exiting. */
  kill(): void;
  close(): Promise<void>;
  /** Why the engine stopped, for an error message ("" when unknown). */
  failure(): string;
}

export class EngineSession {
  readonly template: TemplateInfo;
  /** The text the engine's KV cache holds, or null when that is unknown (after a cancelled prompt, a refused one, or a failure): the next turn starts over. */
  private fed: string | null = "";
  private tail: Promise<void> = Promise.resolve();
  /** What the last turn had to send: how many tokens, and whether the cache was started over. Shown by the UI and used by tests. */
  lastAppend: { tokens: number; reset: boolean } = { tokens: 0, reset: false };

  constructor(protected readonly t: Transport, readonly ready: NativeReady) {
    this.template = detectTemplate(ready.template);
  }

  private async lock(): Promise<() => void> {
    let release!: () => void;
    const prev = this.tail;
    this.tail = new Promise<void>((r) => { release = r; });
    await prev;
    return release;
  }
  private dead(): NativeError { const f = this.t.failure().trim(); return new NativeError(`the engine stopped${f ? `: ${f.split("\n").slice(-2).join(" ")}` : ""}`); }

  /** Forget the conversation (the next turn re-reads everything it needs). */
  async reset(): Promise<void> {
    const release = await this.lock();
    try { this.t.send({ op: "reset" }); for (;;) { const e = await this.t.queue.next(); if (!e) throw this.dead(); if (e.ev === "reset") break; } this.fed = ""; } finally { release(); }
  }

  /** Streams the reply to `messages`. Only text the engine has not seen is sent; a conversation that no longer extends the cache starts over. */
  async *stream(messages: ChatMessage[], o: SampleOptions = {}): AsyncGenerator<StreamEvent> {
    const release = await this.lock();
    let aborted = false;
    const onAbort = (): void => { aborted = true; this.t.send({ op: "cancel" }); };
    o.signal?.addEventListener("abort", onAbort, { once: true });
    try {
      if (o.signal?.aborted) onAbort();
      const full = renderChat(this.template, messages, { thinking: o.thinking ?? false }) + (o.prefill ?? "");
      let toSend: string, restarted = false;
      const same = this.fed !== null && full === this.fed;   // continuing a reply: the engine already holds everything
      if (this.fed !== null && full.length > this.fed.length && full.startsWith(this.fed)) toSend = full.slice(this.fed.length);
      else if (same) toSend = "";
      else {
        this.t.send({ op: "reset" });
        for (;;) { const e = await this.t.queue.next(); if (!e) throw this.dead(); if (e.ev === "reset") break; }
        this.fed = "";
        toSend = full;
        restarted = true;
      }
      if (toSend) this.t.send({ op: "append", text: toSend });
      let newTokens = 0, prefillSecs = 0;
      if (toSend) for (;;) {
        const e = await this.t.queue.next();
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
      if (aborted) { this.t.send({ op: "cancel" }); }
      const startsInThink = this.template.thinkTags && o.thinking === true && /<think>\s*$/.test(full);
      const split = new ThinkSplitter(this.template.thinkTags && o.thinking === true, startsInThink);
      this.t.send({ op: "generate", max: o.max ?? 1024, temp: o.temp ?? 0.7, topP: o.topP ?? 0.95, topK: o.topK ?? 40, repeat: o.repeat ?? 1.0, ...(o.seed !== undefined ? { seed: o.seed } : {}), ...(o.minP ? { minP: o.minP } : {}), ...(o.json ? { json: true, jsonPrefix: jsonPrefix(o.prefill) } : {}) });
      let raw = "";
      for (;;) {
        const e = await this.t.queue.next();
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
            ...(e["cache"] ? { cache: e["cache"] as { hitRate: number; gbRead: number; waitSeconds?: number } } : {}),
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

  killNow(): void { this.t.kill(); }
  close(): Promise<void> { return this.t.close(); }
}
