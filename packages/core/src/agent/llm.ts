import { TOOLS, toolSchemas } from "./tools";

/** The model side of the agent: anything that speaks the OpenAI chat-completions protocol (llama-server, Ollama, LM Studio, a remote API). */

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

export interface ChatResult {
  content: string | null;
  toolCalls: Array<{ id: string; name: string; arguments: string }>;
  finishReason?: string;
  usage?: { promptTokens?: number; completionTokens?: number };
}

export interface LlmClient {
  readonly model: string;
  /** `native` sends the tool schemas to the API; `json` asks the model to answer with a JSON object and parses it. */
  readonly toolMode: "native" | "json";
  chat(messages: ChatMessage[], opts?: { signal?: AbortSignal }): Promise<ChatResult>;
}

export interface OpenAiClientOptions {
  baseUrl: string;
  model: string;
  apiKey?: string;
  toolMode?: "native" | "json";
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
  /** In "json" tool mode, ask the server to constrain decoding to this JSON Schema (response_format). If the server refuses, it is turned off and the request retried. */
  jsonSchema?: Record<string, unknown>;
  /** Ask a llama.cpp server to keep this conversation in one slot and reuse its cached prompt prefix. */
  slot?: number;
}

export class LlmError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "LlmError";
  }
}

export class OpenAiCompatibleClient implements LlmClient {
  readonly model: string;
  readonly toolMode: "native" | "json";
  /** True once the server has been seen to refuse schema-constrained decoding. */
  schemaRefused = false;
  constructor(private readonly o: OpenAiClientOptions) {
    this.model = o.model;
    this.toolMode = o.toolMode ?? "json";
  }

  async chat(messages: ChatMessage[], opts: { signal?: AbortSignal } = {}): Promise<ChatResult> {
    const f = this.o.fetch ?? globalThis.fetch;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(new LlmError(`the model did not answer within ${(this.o.timeoutMs ?? 300_000) / 1000} seconds`)), this.o.timeoutMs ?? 300_000);
    const onAbort = (): void => ctl.abort(opts.signal?.reason);
    opts.signal?.addEventListener("abort", onAbort, { once: true });
    try {
      const send = (constrained: boolean): Promise<Response> => f(`${this.o.baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", ...(this.o.apiKey ? { authorization: `Bearer ${this.o.apiKey}` } : {}) },
        body: JSON.stringify({
          model: this.o.model, messages, stream: false, temperature: this.o.temperature ?? 0.2, ...(this.o.maxTokens ? { max_tokens: this.o.maxTokens } : {}),
          ...(this.toolMode === "native" ? { tools: toolSchemas(), tool_choice: "auto" } : {}),
          ...(constrained ? { response_format: { type: "json_schema", json_schema: { name: "tool_call", strict: true, schema: this.o.jsonSchema } } } : {}),
          ...(this.o.slot !== undefined ? { id_slot: this.o.slot, cache_prompt: true } : {}),
        }),
        signal: ctl.signal,
      });
      const wantSchema = this.toolMode === "json" && this.o.jsonSchema !== undefined && !this.schemaRefused;
      let res = await send(wantSchema);
      if (!res.ok && wantSchema && (res.status === 400 || res.status === 422 || res.status === 501)) {
        // This server cannot constrain decoding to a schema. Remember that and carry on without it: the validator still checks every reply.
        this.schemaRefused = true;
        res = await send(false);
      }
      if (!res.ok) throw new LlmError(`the model server answered ${res.status}: ${(await res.text().catch(() => "")).slice(0, 300)}`, res.status);
      const body = (await res.json()) as { choices?: Array<{ message?: { content?: string | null; tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }> }; finish_reason?: string }>; usage?: { prompt_tokens?: number; completion_tokens?: number } };
      const choice = body.choices?.[0];
      if (!choice?.message) throw new LlmError("the model server returned no choices");
      return {
        content: choice.message.content ?? null,
        toolCalls: (choice.message.tool_calls ?? []).map((t, i) => ({ id: t.id ?? `call_${i}`, name: t.function?.name ?? "", arguments: t.function?.arguments ?? "{}" })),
        ...(choice.finish_reason ? { finishReason: choice.finish_reason } : {}),
        ...(body.usage ? { usage: { ...(body.usage.prompt_tokens !== undefined ? { promptTokens: body.usage.prompt_tokens } : {}), ...(body.usage.completion_tokens !== undefined ? { completionTokens: body.usage.completion_tokens } : {}) } } : {}),
      };
    } catch (e) {
      if (e instanceof LlmError) throw e;
      if (ctl.signal.aborted && ctl.signal.reason instanceof Error) throw ctl.signal.reason;
      throw new LlmError(`could not reach the model at ${this.o.baseUrl}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onAbort);
    }
  }
}

/** Escapes the raw line breaks and tabs that models put inside a JSON string when they write a whole file into a tool call: strictly invalid, but unambiguous. */
function escapeControlChars(json: string): string {
  let out = "", inStr = false, esc = false;
  for (const c of json) {
    if (inStr) {
      if (esc) { esc = false; out += c; continue; }
      if (c === "\\") { esc = true; out += c; continue; }
      if (c === '"') { inStr = false; out += c; continue; }
      out += c === "\n" ? "\\n" : c === "\r" ? "\\r" : c === "\t" ? "\\t" : c;
    } else { if (c === '"') inStr = true; out += c; }
  }
  return out;
}

/** The first balanced JSON object in `text`, tolerating prose and ``` fences around it and raw line breaks inside its strings. Returns undefined if there is none. */
export function extractJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  for (let s = start; s >= 0 && s < text.length; s = text.indexOf("{", s + 1)) {
    let depth = 0, inStr = false, esc = false;
    for (let i = s; i < text.length; i++) {
      const c = text[i] as string;
      if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true;
      else if (c === "{") depth++;
      else if (c === "}" && --depth === 0) {
        const raw = text.slice(s, i + 1);
        try { return JSON.parse(raw); } catch { /* try again with the line breaks escaped */ }
        try { return JSON.parse(escapeControlChars(raw)); } catch { break; }
      }
    }
  }
  return undefined;
}

/** True when the text starts a tool call as JSON but ends before it is closed: the reply was cut off (the model ran out of room), usually in the middle of a long file. */
export function looksCutOff(text: string): boolean {
  const s = text.indexOf("{");
  if (s < 0 || !/"(tool|name)"\s*:/.test(text.slice(s, s + 200))) return false;
  let depth = 0, inStr = false, esc = false;
  for (let i = s; i < text.length; i++) {
    const c = text[i] as string;
    if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true; else if (c === "{") depth++; else if (c === "}" && --depth === 0) return false;
  }
  return true;
}

/** What the model is told in `json` mode, appended to the system prompt. */
export function jsonProtocolPrompt(): string {
  return [
    "Reply with exactly one JSON object and nothing else: {\"tool\": \"<name>\", \"args\": {...}}.",
    "Call one tool per reply. You will see the result and then reply again. The tools:",
    ...TOOLS.map((t) => `- ${t.name}(${Object.entries(t.fields).map(([k, f]) => `${k}${f.required ? "" : "?"}: ${f.type}`).join(", ")}): ${t.description}`),
    "To create a file you must call write_file with the whole file in args.content (escape newlines as \\n). Never paste code into your reply as text: text is not saved anywhere. Example of a first reply for \"make a page\":",
    '{"tool": "write_file", "args": {"path": "index.html", "content": "<!doctype html>\\n<html>...</html>\\n"}}',
    "Then write the next file the same way. Call finish only after every file you need exists, and say plainly what you made.",
  ].join("\n");
}
