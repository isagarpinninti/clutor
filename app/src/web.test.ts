import { describe, expect, it } from "vitest";
import type { WebApi } from "./bridge";
import { agentWeb, gather, needsWeb, withWebContext } from "./web";

const api = (over: Partial<WebApi> = {}): WebApi => ({
  search: async () => [{ title: "A", url: "https://a.org/x", snippet: "sa" }, { title: "B", url: "https://b.org/y", snippet: "sb" }, { title: "C", url: "https://c.org/z", snippet: "sc" }],
  fetch: async (url) => ({ url, title: `T ${url}`, text: `body of ${url} `.repeat(300), truncated: false, status: 200 }),
  openExternal: async () => undefined,
  ...over,
});

describe("agentWeb", () => {
  it("asks once per run and remembers a refusal", async () => {
    let asked = 0;
    const w = agentWeb(api(), async () => { asked++; return false; });
    expect((await w.search("x")).ok).toBe(false);
    expect((await w.fetch("https://a.org/")).output).toMatch(/did not allow/);
    expect(asked).toBe(1);
  });
  it("marks web text as untrusted and reports errors as text", async () => {
    const w = agentWeb(api({ fetch: async () => { throw new Error("that address is on this computer"); } }), async () => true);
    expect((await w.search("x")).output).toMatch(/untrusted/);
    expect(await w.fetch("http://127.0.0.1/")).toEqual({ ok: false, output: "that address is on this computer" });
  });
});

describe("gather", () => {
  it("reads the top pages within a size bound and lists the sources", async () => {
    const g = await gather(api(), "what is a moe", { pages: 2, charsPerPage: 500 });
    expect(g.sources.map((s) => s.url)).toEqual(["https://a.org/x", "https://b.org/y"]);
    expect(g.context.length).toBeLessThan(1400);
    expect(g.context).toContain("[1]");
  });
  it("skips pages that fail and falls back to snippets when none can be read", async () => {
    const one = await gather(api({ fetch: async (u) => { if (u.includes("a.org")) throw new Error("x"); return { url: u, title: "T", text: "ok", truncated: false, status: 200 }; } }), "q");
    expect(one.sources.map((s) => s.url)).toEqual(["https://b.org/y", "https://c.org/z"]);
    const none = await gather(api({ fetch: async () => { throw new Error("x"); } }), "q");
    expect(none.context).toContain("sa");
  });
});

describe("withWebContext", () => {
  it("adds the pages to the system message", () => {
    const out = withWebContext([{ role: "system", content: "base" }, { role: "user", content: "hi" }], { context: "[1] page", sources: [] });
    expect(out[0]?.content).toMatch(/^base[\s\S]*untrusted[\s\S]*\[1\] page/);
    expect(out).toHaveLength(2);
  });
  it("leaves the messages alone when there is nothing", () => {
    const m = [{ role: "user" as const, content: "hi" }];
    expect(withWebContext(m, { context: "", sources: [] })).toBe(m);
  });
});

describe("needsWeb", () => {
  it("skips greetings and one-word replies, and looks up real questions", () => {
    for (const s of ["hi", "thanks!", "ok", "Hello there", "yes"]) expect(needsWeb(s)).toBe(false);
    for (const s of ["who won the match last night", "latest rust release notes", "what is a MoE model"]) expect(needsWeb(s)).toBe(true);
  });
});
