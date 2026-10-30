import { parseSse } from "./sse";

/** Translation between the Ollama HTTP API and an OpenAI-compatible backend, so tools that speak Ollama can use whatever Clutor fronts. */

export interface OllamaMessage { role: string; content: string }

export interface OllamaChatRequest { model?: string; messages?: OllamaMessage[]; prompt?: string; system?: string; stream?: boolean; options?: { temperature?: number; num_predict?: number; top_p?: number; stop?: string[] } }

export function toOpenAiBody(req: OllamaChatRequest, model: string, kind: "chat" | "generate"): Record<string, unknown> {
  const messages: OllamaMessage[] = kind === "generate" ? [...(req.system ? [{ role: "system", content: req.system }] : []), { role: "user", content: req.prompt ?? "" }] : (req.messages ?? []);
  const o = req.options ?? {};
  return {
    model, messages, stream: req.stream !== false,
    ...(o.temperature !== undefined ? { temperature: o.temperature } : {}), ...(o.top_p !== undefined ? { top_p: o.top_p } : {}),
    ...(o.num_predict !== undefined && o.num_predict > 0 ? { max_tokens: o.num_predict } : {}), ...(o.stop ? { stop: o.stop } : {}),
  };
}

const now = (): string => new Date().toISOString();
const piece = (kind: "chat" | "generate", model: string, text: string, done: boolean, extra: Record<string, unknown> = {}): Record<string, unknown> =>
  kind === "chat" ? { model, created_at: now(), message: { role: "assistant", content: text }, done, ...extra } : { model, created_at: now(), response: text, done, ...extra };

/** Converts one non-streaming OpenAI reply into one Ollama reply. */
export function fromOpenAiReply(kind: "chat" | "generate", model: string, body: { choices?: Array<{ message?: { content?: string | null }; finish_reason?: string }>; usage?: { prompt_tokens?: number; completion_tokens?: number } }): Record<string, unknown> {
  const text = body.choices?.[0]?.message?.content ?? "";
  return piece(kind, model, text, true, { done_reason: body.choices?.[0]?.finish_reason ?? "stop", prompt_eval_count: body.usage?.prompt_tokens ?? 0, eval_count: body.usage?.completion_tokens ?? 0 });
}

/** Converts an OpenAI event stream into Ollama's newline-delimited JSON. */
export async function* streamToOllama(kind: "chat" | "generate", model: string, body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
  let reason = "stop";
  for await (const data of parseSse(body)) {
    if (data.trim() === "[DONE]") break;
    let j: { choices?: Array<{ delta?: { content?: string | null }; finish_reason?: string | null }>; usage?: typeof usage };
    try { j = JSON.parse(data); } catch { continue; }
    if (j.usage) usage = j.usage;
    const c = j.choices?.[0];
    if (c?.finish_reason) reason = c.finish_reason;
    const text = c?.delta?.content;
    if (text) yield `${JSON.stringify(piece(kind, model, text, false))}\n`;
  }
  yield `${JSON.stringify(piece(kind, model, "", true, { done_reason: reason, prompt_eval_count: usage?.prompt_tokens ?? 0, eval_count: usage?.completion_tokens ?? 0 }))}\n`;
}
