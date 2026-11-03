import { describe, expect, it } from "vitest";
import { executeTool, MemoryProject, ShadowWorkspace, validateToolCall } from "../src";

const ws = (): ShadowWorkspace => new ShadowWorkspace(new MemoryProject());

describe("use_tool (external servers)", () => {
  it("validates its arguments like any other tool", () => {
    expect(validateToolCall("use_tool", { server: "s", tool: "t" }).ok).toBe(true);
    expect(validateToolCall("use_tool", { server: "s" }).ok).toBe(false);
  });

  it("reports when no server is connected", async () => {
    const r = await executeTool({ ws: ws() }, "use_tool", { server: "s", tool: "t" });
    expect(r.ok).toBe(false);
    expect(r.output).toMatch(/no external servers/);
  });

  it("runs a read-only tool without asking, and passes the parsed arguments", async () => {
    let got: unknown;
    const r = await executeTool({ ws: ws(), external: { readOnly: () => true, call: async (s, t, a) => { got = [s, t, a]; return { ok: true, output: "hi" }; } } }, "use_tool", { server: "s", tool: "t", arguments: '{"q":1}' });
    expect(r).toMatchObject({ ok: true, output: "hi" });
    expect(got).toEqual(["s", "t", { q: 1 }]);
  });

  it("asks before any tool that may change something, and does not call it when refused", async () => {
    let called = false;
    const external = { readOnly: () => false, call: async () => { called = true; return { ok: true, output: "x" }; } };
    const none = await executeTool({ ws: ws(), external }, "use_tool", { server: "s", tool: "t" });
    expect(none.ok).toBe(false);
    const no = await executeTool({ ws: ws(), external, approve: async () => false }, "use_tool", { server: "s", tool: "t" });
    expect(no.output).toMatch(/did not approve/);
    expect(called).toBe(false);
    const yes = await executeTool({ ws: ws(), external, approve: async () => true }, "use_tool", { server: "s", tool: "t" });
    expect(yes.ok).toBe(true);
  });

  it("rejects arguments that are not a JSON object", async () => {
    const external = { readOnly: () => true, call: async () => ({ ok: true, output: "" }) };
    expect((await executeTool({ ws: ws(), external }, "use_tool", { server: "s", tool: "t", arguments: "[1]" })).ok).toBe(false);
    expect((await executeTool({ ws: ws(), external }, "use_tool", { server: "s", tool: "t", arguments: "{oops" })).ok).toBe(false);
  });
});

describe("web tools", () => {
  it("report when the web is not available", async () => {
    expect((await executeTool({ ws: ws() }, "web_search", { query: "x" })).output).toMatch(/not available/);
    expect(validateToolCall("fetch_url", {}).ok).toBe(false);
  });
  it("call the web adapter", async () => {
    const web = { search: async (q: string) => ({ ok: true, output: `hits for ${q}` }), fetch: async (u: string) => ({ ok: true, output: `page ${u}` }) };
    expect((await executeTool({ ws: ws(), web }, "web_search", { query: "moe" })).output).toBe("hits for moe");
    expect((await executeTool({ ws: ws(), web }, "fetch_url", { url: "https://a.b/" })).output).toBe("page https://a.b/");
  });
});
