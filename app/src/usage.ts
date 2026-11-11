import type { Msg } from "./store";

export interface ChatUsage {
  replies: number;
  /** Tokens the model wrote in this conversation. */
  tokens: number;
  /** Generation speed averaged over the replies that measured one, weighted by length. */
  tps?: number;
  /** The slowest, median and fastest generation speed among the replies that measured one (tokens per second). */
  speed?: { min: number; median: number; max: number; replies: number };
  /** Expert-cache behaviour over the replies that streamed experts from disk: the mean hit rate and the total read. Absent when no reply streamed. */
  cache?: { replies: number; hitRate: number; gbRead: number };
  /** Tokens in the model's context right now (the last prompt plus its reply), and the most it can hold. */
  contextUsed: number;
  contextMax: number;
}

/** What a conversation has used, from the numbers the engine measured for each reply. Nothing is estimated except where the field is absent, which counts as zero. */
export function chatUsage(messages: Msg[], contextMax: number): ChatUsage {
  let tokens = 0, replies = 0, weighted = 0, timed = 0, contextUsed = 0, hitSum = 0, gb = 0, cached = 0;
  const speeds: number[] = [];
  for (const m of messages) {
    if (m.role !== "assistant" || !m.stats) continue;
    replies++;
    tokens += m.stats.tokens;
    if (m.stats.tps) { weighted += m.stats.tps * m.stats.tokens; timed += m.stats.tokens; speeds.push(m.stats.tps); }
    if (m.stats.cache) { cached++; hitSum += m.stats.cache.hitRate; gb += m.stats.cache.gbRead; }
    contextUsed = (m.stats.promptTokens ?? 0) + m.stats.tokens;
  }
  speeds.sort((a, b) => a - b);
  const mid = speeds.length ? (speeds.length % 2 ? (speeds[(speeds.length - 1) / 2] as number) : ((speeds[speeds.length / 2 - 1] as number) + (speeds[speeds.length / 2] as number)) / 2) : 0;
  return {
    replies, tokens, ...(timed > 0 ? { tps: weighted / timed } : {}),
    ...(speeds.length ? { speed: { min: speeds[0] as number, median: mid, max: speeds[speeds.length - 1] as number, replies: speeds.length } } : {}),
    ...(cached ? { cache: { replies: cached, hitRate: hitSum / cached, gbRead: gb } } : {}),
    contextUsed, contextMax,
  };
}

export const contextPercent = (u: Pick<ChatUsage, "contextUsed" | "contextMax">): number => (u.contextMax > 0 ? Math.min(100, Math.round((u.contextUsed / u.contextMax) * 100)) : 0);
