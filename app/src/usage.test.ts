import { describe, expect, it } from "vitest";
import type { Msg } from "./store";
import { chatUsage, contextPercent } from "./usage";

const reply = (tokens: number, tps: number | undefined, promptTokens?: number): Msg => ({ id: String(Math.random()), role: "assistant", content: "x", stats: { tokens, ms: 1000, ...(tps ? { tps } : {}), ...(promptTokens ? { promptTokens } : {}) } });

describe("chatUsage", () => {
  it("sums tokens, weights speed by length, and takes the context from the last reply", () => {
    const u = chatUsage([reply(100, 10, 50), reply(300, 20, 200)], 4096);
    expect(u.tokens).toBe(400);
    expect(u.tps).toBeCloseTo(17.5);
    expect(u.contextUsed).toBe(500);
    expect(contextPercent(u)).toBe(12);
  });
  it("is empty for a new conversation", () => {
    const u = chatUsage([], 4096);
    expect(u).toEqual({ replies: 0, tokens: 0, contextUsed: 0, contextMax: 4096 });
    expect(contextPercent(u)).toBe(0);
  });
});

describe("measured performance", () => {
  const reply = (tps: number, cache?: { hitRate: number; gbRead: number }): Msg => ({ id: String(tps), role: "assistant", content: "x", stats: { tokens: 50, ms: 1000, tps, ...(cache ? { cache } : {}) } });
  it("reports the speed range, median and cache totals, and nothing when nothing was measured", () => {
    const u = chatUsage([reply(10), reply(4, { hitRate: 0.5, gbRead: 2 }), reply(7, { hitRate: 0.7, gbRead: 1 })], 4096);
    expect(u.speed).toEqual({ min: 4, median: 7, max: 10, replies: 3 });
    expect(u.cache?.replies).toBe(2);
    expect(u.cache?.hitRate).toBeCloseTo(0.6);
    expect(u.cache?.gbRead).toBe(3);
    const w = chatUsage([{ id: "a", role: "assistant", content: "x", stats: { tokens: 10, ms: 4000, tps: 3, cache: { hitRate: 0.5, gbRead: 1, waitSeconds: 3 } } }, { id: "b", role: "assistant", content: "x", stats: { tokens: 10, ms: 1000, tps: 9, cache: { hitRate: 0.9, gbRead: 0, waitSeconds: 0.2 } } }, reply(5, { hitRate: 0.5, gbRead: 1 })], 4096);
    expect(w.cache?.wait?.seconds).toBeCloseTo(3.2);
    expect(w.cache?.wait?.of).toBeCloseTo(5);   // a reply from an older engine, with no wait figure, is left out of both
    expect(chatUsage([reply(5, { hitRate: 0.5, gbRead: 1 })], 4096).cache?.wait).toBeUndefined();
    const none = chatUsage([{ id: "u", role: "user", content: "hi" }], 4096);
    expect(none.speed).toBeUndefined();
    expect(none.cache).toBeUndefined();
  });
});
