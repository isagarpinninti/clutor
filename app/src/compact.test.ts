import { describe, expect, it } from "vitest";
import { compactMessages, shouldCompact } from "./compact";
import type { Msg } from "./store";

const m = (role: "user" | "assistant", content: string): Msg => ({ id: content, role, content });
const four = [m("user", "a"), m("assistant", "b"), m("user", "c"), m("assistant", "d")];

describe("auto-compact", () => {
  it("triggers at 80% of the context and only with enough conversation", () => {
    expect(shouldCompact(3300, 4096, four)).toBe(true);
    expect(shouldCompact(3000, 4096, four)).toBe(false);
    expect(shouldCompact(4000, 4096, four.slice(0, 2))).toBe(false);
    expect(shouldCompact(100, 0, four)).toBe(false);
  });
  it("replaces the conversation with the model's summary", async () => {
    const s = { async *stream() { yield { type: "content", text: "We talked " }; yield { type: "content", text: "about a, c." }; } };
    const out = await compactMessages(s, four);
    expect(out).toHaveLength(2);
    expect(out?.[0]?.content).toContain("We talked about a, c.");
  });
  it("leaves the conversation alone when no summary comes back", async () => {
    const s = { async *stream() { yield { type: "content", text: "  " }; } };
    expect(await compactMessages(s, four)).toBeUndefined();
  });
});
