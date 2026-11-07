import type { Msg } from "./store";

export interface ChatUsage {
  replies: number;
  /** Tokens the model wrote in this conversation. */
  tokens: number;
  /** Generation speed averaged over the replies that measured one, weighted by length. */
  tps?: number;
  /** Tokens in the model's context right now (the last prompt plus its reply), and the most it can hold. */
  contextUsed: number;
  contextMax: number;
}

/** What a conversation has used, from the numbers the engine measured for each reply. Nothing is estimated except where the field is absent, which counts as zero. */
export function chatUsage(messages: Msg[], contextMax: number): ChatUsage {
  let tokens = 0, replies = 0, weighted = 0, timed = 0, contextUsed = 0;
  for (const m of messages) {
    if (m.role !== "assistant" || !m.stats) continue;
    replies++;
    tokens += m.stats.tokens;
    if (m.stats.tps) { weighted += m.stats.tps * m.stats.tokens; timed += m.stats.tokens; }
    contextUsed = (m.stats.promptTokens ?? 0) + m.stats.tokens;
  }
  return { replies, tokens, ...(timed > 0 ? { tps: weighted / timed } : {}), contextUsed, contextMax };
}

export const contextPercent = (u: Pick<ChatUsage, "contextUsed" | "contextMax">): number => (u.contextMax > 0 ? Math.min(100, Math.round((u.contextUsed / u.contextMax) * 100)) : 0);
