import { parseSse } from "../sse";

/** Streams one chat completion from an OpenAI-compatible server (llama-server reports more: prompt progress and timings). */

/** A picture sent with a message: `rgb` is the base64 of w * h * 3 bytes (8-bit RGB, row by row). The engine resizes it if its sides are not multiples of 32. */
export interface ChatImage { w: number; h: number; rgb: string }
export interface ChatMessage { role: "system" | "user" | "assistant"; content: string; /** Pictures that come before the text (user messages, for a model that takes pictures). */ images?: ChatImage[] }

export type StreamEvent =
  | { type: "content"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "progress"; processed: number; total: number }
  | { type: "done"; finish: string | null; promptTokens?: number; completionTokens?: number; promptPerSecond?: number; tokensPerSecond?: number; cache?: { hitRate: number; gbRead: number; waitSeconds?: number } };

export interface StreamRequest {
  baseUrl: string;
  model: string;
  apiKey?: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  /** Ask a thinking model to think (true) or answer straight away (false). Servers that do not know the option ignore it. */
  thinking?: boolean;
  signal?: AbortSignal;
  fetchFn?: typeof globalThis.fetch;
}

export async function* streamChat(r: StreamRequest): AsyncGenerator<StreamEvent> {
  const f = r.fetchFn ?? globalThis.fetch;
  const res = await f(`${r.baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(r.apiKey ? { authorization: `Bearer ${r.apiKey}` } : {}) },
    body: JSON.stringify({
      model: r.model, messages: r.messages, stream: true, stream_options: { include_usage: true }, return_progress: true,
      ...(r.temperature !== undefined ? { temperature: r.temperature } : {}),
      ...(r.maxTokens ? { max_tokens: r.maxTokens } : {}),
      ...(r.thinking !== undefined ? { chat_template_kwargs: { enable_thinking: r.thinking } } : {}),
    }),
    ...(r.signal ? { signal: r.signal } : {}),
  });
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => "");
    let msg = body.slice(0, 300);
    try { const j = JSON.parse(body) as { error?: { message?: string } | string }; msg = typeof j.error === "string" ? j.error : j.error?.message ?? msg; } catch { /* not JSON */ }
    throw new Error(`the model server answered ${res.status}${msg ? `: ${msg}` : ""}`);
  }
  let finish: string | null = null;
  let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
  let timings: { prompt_per_second?: number; predicted_per_second?: number } | undefined;
  for await (const data of parseSse(res.body)) {
    if (data === "[DONE]") break;
    let j: {
      choices?: Array<{ delta?: { content?: string | null; reasoning_content?: string | null }; finish_reason?: string | null }>;
      usage?: typeof usage; timings?: typeof timings; prompt_progress?: { total?: number; processed?: number };
    };
    try { j = JSON.parse(data) as typeof j; } catch { continue; }
    if (j.prompt_progress?.total) yield { type: "progress", processed: j.prompt_progress.processed ?? 0, total: j.prompt_progress.total };
    const c = j.choices?.[0];
    if (c?.delta?.reasoning_content) yield { type: "reasoning", text: c.delta.reasoning_content };
    if (c?.delta?.content) yield { type: "content", text: c.delta.content };
    if (c?.finish_reason) finish = c.finish_reason;
    if (j.usage) usage = j.usage;
    if (j.timings) timings = j.timings;
  }
  yield {
    type: "done", finish,
    ...(usage?.prompt_tokens !== undefined ? { promptTokens: usage.prompt_tokens } : {}),
    ...(usage?.completion_tokens !== undefined ? { completionTokens: usage.completion_tokens } : {}),
    ...(timings?.prompt_per_second ? { promptPerSecond: timings.prompt_per_second } : {}),
    ...(timings?.predicted_per_second ? { tokensPerSecond: timings.predicted_per_second } : {}),
  };
}
