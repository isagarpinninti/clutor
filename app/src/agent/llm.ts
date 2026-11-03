import type { ChatMessage as CoreMessage, ChatResult, LlmClient } from "@core/agent/llm";
import type { SampleOptions } from "@engine/session";
import type { StreamEvent, ChatMessage } from "@engine/stream";

/** The part of the engine session the agent needs; the real one is EngineSession. */
export interface AgentSession { stream(messages: ChatMessage[], o?: SampleOptions): AsyncGenerator<StreamEvent> }

/**
 * The loaded model as the agent's language model. The native engine has no tool-calling API, so the agent talks in its JSON protocol (one JSON object per reply) and
 * every reply is validated before anything runs.
 */
export function engineLlm(session: AgentSession, name: string, o: { maxTokens?: number; temperature?: number; onContext?: (tokens: number) => void } = {}): LlmClient {
  return {
    model: name,
    toolMode: "json",
    async chat(messages: CoreMessage[], opts: { signal?: AbortSignal } = {}): Promise<ChatResult> {
      const msgs: ChatMessage[] = messages.map((m) => ({ role: m.role === "tool" ? "user" : m.role, content: m.content ?? "" }));
      let content = "", prompt = 0, completion = 0;
      for await (const ev of session.stream(msgs, { thinking: false, temp: o.temperature ?? 0.2, max: o.maxTokens ?? 1024, ...(opts.signal ? { signal: opts.signal } : {}) })) {
        if (ev.type === "content") content += ev.text;
        else if (ev.type === "done") { prompt = ev.promptTokens ?? 0; completion = ev.completionTokens ?? 0; }
      }
      o.onContext?.(prompt + completion);
      return { content, toolCalls: [], usage: { promptTokens: prompt, completionTokens: completion } };
    },
  };
}
