import type { Msg } from "./store";

/** The smallest part of the engine session that compacting needs. */
export interface Summariser { stream(messages: Array<{ role: "system" | "user" | "assistant"; content: string }>, o?: { thinking?: boolean; temp?: number; max?: number; signal?: AbortSignal }): AsyncGenerator<{ type: string; text?: string }> }

/** True when the next message should be preceded by a summary: the context is at least `at` full and there is enough conversation to summarise. */
export function shouldCompact(contextUsed: number, contextMax: number, messages: Msg[], at = 0.8): boolean {
  return contextMax > 0 && contextUsed >= contextMax * at && messages.length >= 4;
}

/** Replaces the conversation so far with one summary written by the model. Returns undefined when the model gave no usable summary (the conversation is then left as it is). */
export async function compactMessages(s: Summariser, messages: Msg[], signal?: AbortSignal): Promise<Msg[] | undefined> {
  const asked = [
    { role: "system" as const, content: "Summarise the conversation so far in under 250 words, keeping every fact, name, number, decision and open question the user may refer to later. Write the summary only." },
    { role: "user" as const, content: messages.map((m) => `${m.role}: ${m.content.slice(0, 3000)}`).join("\n\n") },
  ];
  let summary = "";
  for await (const ev of s.stream(asked, { thinking: false, temp: 0.2, max: 400, ...(signal ? { signal } : {}) })) if (ev.type === "content" && ev.text) summary += ev.text;
  if (!summary.trim() || signal?.aborted) return undefined;
  const id = (): string => Math.random().toString(36).slice(2);
  return [{ id: id(), role: "user", content: `Summary of our conversation so far:\n${summary.trim()}` }, { id: id(), role: "assistant", content: "Understood. I'll continue from that summary." }];
}
