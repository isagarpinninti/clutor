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
