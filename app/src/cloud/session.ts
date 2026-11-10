import { EngineSession, EventQueue, NativeError, type NativeReady, type SampleOptions, type Transport } from "@engine/session";
import type { ChatMessage, StreamEvent } from "@engine/stream";
import { bridge } from "../bridge";
import { endpoint, explainError, type CloudProvider } from "./providers";
import { SseParser } from "./sse";

type Msg = { role: "system" | "user" | "assistant"; content: string };

/** The request for one reply: the address and the JSON body, in the form the provider's API takes. */
export function buildRequest(p: Pick<CloudProvider, "kind" | "baseUrl">, model: string, messages: Msg[], o: { temp?: number; max?: number; prefill?: string }): { url: string; body: string } {
  const prefill = o.prefill ?? "";
  if (p.kind === "anthropic") {
    const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
    const turns: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (const m of messages) {
      if (m.role === "system") continue;
      const last = turns[turns.length - 1];
      if (last && last.role === m.role) last.content += `\n\n${m.content}`; else turns.push({ role: m.role, content: m.content });
    }
    if (!turns.length || turns[0]!.role !== "user") turns.unshift({ role: "user", content: "Hello." });
    if (prefill.trim()) turns.push({ role: "assistant", content: prefill.trimEnd() });   // the API continues a reply that ends the list
    return { url: endpoint(p), body: JSON.stringify({ model, max_tokens: o.max ?? 4096, stream: true, ...(system ? { system } : {}), messages: turns, ...(o.temp !== undefined ? { temperature: Math.min(1, o.temp) } : {}) }) };
  }
  const list: Msg[] = messages.map((m) => ({ role: m.role, content: m.content }));
  // Chat-completions APIs cannot continue a reply, so a reply that was cut off is shown to the model and it is asked for the rest.
  if (prefill) list.push({ role: "assistant", content: prefill }, { role: "user", content: "Your reply was cut off. Continue it exactly from where it stopped, with only the rest of it, no repeating." });
  return { url: endpoint(p), body: JSON.stringify({ model, stream: true, stream_options: { include_usage: true }, messages: list, ...(o.max ? { max_tokens: o.max } : {}), ...(o.temp !== undefined ? { temperature: o.temp } : {}) }) };
}

const at = (v: unknown, ...path: Array<string | number>): unknown => path.reduce<unknown>((o, k) => (o && typeof o === "object" ? (o as Record<string | number, unknown>)[k] : undefined), v);
const num = (v: unknown): number | undefined => (typeof v === "number" ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" && v ? v : undefined);

/** Turns the text of a streamed response into the pieces of the reply, the token counts and the reason it ended. */
export class ReplyParser {
  private readonly sse = new SseParser();
  promptTokens = 0; completionTokens = 0; finish: string | null = null; done = false;
  constructor(private readonly kind: "openai" | "anthropic") {}
  push(text: string): StreamEvent[] {
    const out: StreamEvent[] = [];
    for (const { event, data } of this.sse.push(text)) {
      if (data.trim() === "[DONE]") { this.done = true; continue; }
      let j: unknown;
      try { j = JSON.parse(data); } catch { continue; }
      const err = at(j, "error");
      if (err) throw new NativeError(str(at(err, "message")) ?? JSON.stringify(err));
      if (this.kind === "anthropic") {
        const t = str(at(j, "type")) ?? event;
        if (t === "message_start") this.promptTokens = num(at(j, "message", "usage", "input_tokens")) ?? this.promptTokens;
        else if (t === "content_block_delta") {
          const kind = at(j, "delta", "type");
          const text = str(at(j, "delta", "text")), think = str(at(j, "delta", "thinking"));
          if (kind === "text_delta" && text) out.push({ type: "content", text });
          else if (kind === "thinking_delta" && think) out.push({ type: "reasoning", text: think });
        } else if (t === "message_delta") { this.finish = str(at(j, "delta", "stop_reason")) ?? this.finish; this.completionTokens = num(at(j, "usage", "output_tokens")) ?? this.completionTokens; }
        else if (t === "message_stop") this.done = true;
      } else {
        const reasoning = str(at(j, "choices", 0, "delta", "reasoning_content")), content = str(at(j, "choices", 0, "delta", "content"));
        if (reasoning) out.push({ type: "reasoning", text: reasoning });
        if (content) out.push({ type: "content", text: content });
        this.finish = str(at(j, "choices", 0, "finish_reason")) ?? this.finish;
        this.promptTokens = num(at(j, "usage", "prompt_tokens")) ?? this.promptTokens;
        this.completionTokens = num(at(j, "usage", "completion_tokens")) ?? this.completionTokens;
      }
    }
    return out;
  }
}

const noTransport: Transport = { queue: new EventQueue(), send() { /* a cloud model has no engine to talk to */ }, kill() { /* nothing to stop */ }, async close() { /* nothing to close */ }, failure: () => "" };

/**
 * A model that runs at a provider, used where the app uses the local engine: it has the same `stream` and `complete`, so chat, background jobs and the coding agent work unchanged.
 * The conversation is sent in full each time (the provider keeps no state for us), and the key is added by the app's native side.
 */
export class CloudSession extends EngineSession {
  private seq = 0;
  private active = new Set<string>();
  constructor(readonly provider: CloudProvider, readonly model: string) {
    const ready: NativeReady = { arch: "cloud", name: model, nCtx: provider.ctx, nVocab: 0, nLayer: 0, threads: 0, streaming: false, locked: false, addBos: false, template: "" } as NativeReady;
    super(noTransport, ready);
  }

  override async *stream(messages: ChatMessage[], o: SampleOptions = {}): AsyncGenerator<StreamEvent> {
    const id = `c${Date.now().toString(36)}${(this.seq++).toString(36)}`;
    const req = buildRequest(this.provider, this.model, messages as Msg[], { ...(o.temp !== undefined ? { temp: o.temp } : {}), max: o.max ?? 4096, ...(o.prefill ? { prefill: o.prefill } : {}) });
    const parser = new ReplyParser(this.provider.kind);
    const queue: Array<StreamEvent | { type: "end"; status: number; error?: string }> = [];
    let wake: (() => void) | undefined;
    const put = (e: StreamEvent | { type: "end"; status: number; error?: string }): void => { queue.push(e); wake?.(); };
    let parseError: unknown;
    const t0 = performance.now();
    let aborted = false;
    const onAbort = (): void => { aborted = true; void bridge.cloud.cancel(id); };
    o.signal?.addEventListener("abort", onAbort, { once: true });
    this.active.add(id);
    void bridge.cloud.request({ id, provider: this.provider.id, url: req.url, auth: this.provider.auth, body: req.body }, (text) => { try { for (const e of parser.push(text)) put(e); } catch (e) { parseError = e; void bridge.cloud.cancel(id); } })
      .then((r) => put({ type: "end", status: r.status, ...(r.error ? { error: r.error } : {}) }), (e: unknown) => put({ type: "end", status: 0, error: String(e) }));
    let chars = 0;
    try {
      for (;;) {
        while (!queue.length) await new Promise<void>((r) => { wake = r; });
        const e = queue.shift() as StreamEvent | { type: "end"; status: number; error?: string };
        if (e.type === "end") {
          if (parseError) throw parseError;
          if (aborted || e.error === "cancelled") { yield { type: "done", finish: "cancel" }; return; }
          if (e.error) throw new NativeError(explainError(e.status, e.error, this.provider.name));
          const secs = Math.max(0.001, (performance.now() - t0) / 1000);
          const completion = parser.completionTokens || Math.round(chars / 4);
          yield { type: "done", finish: parser.finish === "length" || parser.finish === "max_tokens" ? "length" : "stop", promptTokens: parser.promptTokens, completionTokens: completion, tokensPerSecond: completion / secs };
          return;
        }
        if (e.type === "content") chars += e.text.length;
        yield e;
      }
    } finally {
      o.signal?.removeEventListener("abort", onAbort);
      this.active.delete(id);
    }
  }

  /** Nothing is kept on this side to forget: each request carries the whole conversation. */
  override async reset(): Promise<void> { /* stateless */ }
  override killNow(): void { for (const id of this.active) void bridge.cloud.cancel(id); }
  override close(): Promise<void> { this.killNow(); return Promise.resolve(); }
}
