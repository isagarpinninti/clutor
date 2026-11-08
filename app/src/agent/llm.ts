import type { ChatMessage as CoreMessage, ChatResult, LlmClient } from "@core/agent/llm";
import type { SampleOptions } from "@engine/session";
import type { StreamEvent, ChatMessage } from "@engine/stream";

/** The part of the engine session the agent needs; the real one is EngineSession. */
export interface AgentSession { stream(messages: ChatMessage[], o?: SampleOptions): AsyncGenerator<StreamEvent> }

/**
 * The loaded model as the agent's language model. The native engine has no tool-calling API, so the agent talks in its JSON protocol (one JSON object per reply) and
 * every reply is validated before anything runs.
 */
export function engineLlm(session: AgentSession, name: string, o: { maxTokens?: number; temperature?: number; onContext?: (tokens: number) => void; onSpeed?: (tokensPerSecond: number, tokens: number) => void } = {}): LlmClient {
  return {
    model: name,
    toolMode: "json",
    canContinue: true,
    async chat(messages: CoreMessage[], opts: { signal?: AbortSignal; onPartial?: (text: string) => void; prefill?: string } = {}): Promise<ChatResult> {
      const msgs: ChatMessage[] = messages.map((m) => ({ role: m.role === "tool" ? "user" : m.role, content: m.content ?? "" }));
      let content = opts.prefill ?? "", prompt = 0, completion = 0, lastPartial = 0;
      for await (const ev of session.stream(msgs, { thinking: false, temp: o.temperature ?? 0.2, max: o.maxTokens ?? 1024, ...(opts.prefill ? { prefill: opts.prefill } : {}), ...(opts.signal ? { signal: opts.signal } : {}) })) {
        if (ev.type === "content") { content += ev.text; if (opts.onPartial) { const t = Date.now(); if (t - lastPartial >= 250) { lastPartial = t; opts.onPartial(content); } } }
        else if (ev.type === "done") { prompt = ev.promptTokens ?? 0; completion = ev.completionTokens ?? 0; if (ev.tokensPerSecond) o.onSpeed?.(ev.tokensPerSecond, completion); }
      }
      opts.onPartial?.(content);   // the last piece, so what is shown is the whole reply
      o.onContext?.(prompt + completion);
      return { content, toolCalls: [], usage: { promptTokens: prompt, completionTokens: completion } };
    },
  };
}
