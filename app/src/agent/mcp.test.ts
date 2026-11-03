import { describe, expect, it } from "vitest";
import type { AgentHost } from "../bridge";
import { McpClient, McpManager, serverName, splitCommandLine } from "./mcp";

/** A host whose "server" answers like a small MCP server. */
function fakeHost(opts: { crash?: boolean; tools?: unknown[] } = {}): { host: AgentHost; sent: string[] } {
  const sent: string[] = [];
  let onLine: (l: string) => void = () => undefined;
  let onExit: () => void = () => undefined;
  const reply = (id: number, result: unknown): void => queueMicrotask(() => onLine(JSON.stringify({ jsonrpc: "2.0", id, result })));
  const host = {
    async mcpStart(_id: string, _l: unknown, line: (l: string) => void, exit: () => void) { onLine = line; onExit = exit; if (opts.crash) throw new Error("could not start x"); },
    async mcpSend(_id: string, line: string) {
      sent.push(line);
      const m = JSON.parse(line) as { id?: number; method: string; params?: { name?: string; arguments?: { q?: string } } };
      if (m.id === undefined) return;
      if (m.method === "initialize") reply(m.id, { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: { name: "fake" } });
      else if (m.method === "tools/list") reply(m.id, { tools: opts.tools ?? [{ name: "echo", description: "Echo text", inputSchema: { properties: { q: { type: "string" } }, required: ["q"] }, annotations: { readOnlyHint: true } }, { name: "wipe", description: "Dangerous" }] });
      else if (m.method === "tools/call") reply(m.id, m.params?.name === "echo" ? { content: [{ type: "text", text: `echo:${m.params.arguments?.q}` }] } : { isError: true, content: [{ type: "text", text: "nope" }] });
    },
    async mcpStop() { onExit(); },
  } as unknown as AgentHost;
  return { host, sent };
}

describe("McpClient", () => {
  it("does the handshake, lists tools and calls them", async () => {
    const { host, sent } = fakeHost();
    const c = new McpClient({ id: "fake", command: "x", args: [], env: {}, enabled: true }, host);
    await c.start();
    expect(c.status).toBe("ready");
    expect(c.tools.map((t) => [t.name, t.readOnly])).toEqual([["echo", true], ["wipe", false]]);
    expect(sent.some((l) => l.includes("notifications/initialized"))).toBe(true);
    expect(await c.call("echo", { q: "hi" })).toEqual({ ok: true, output: "echo:hi" });
    expect((await c.call("wipe", {})).ok).toBe(false);
  });

  it("reports a server that cannot start", async () => {
    const c = new McpClient({ id: "bad", command: "x", args: [], env: {}, enabled: true }, fakeHost({ crash: true }).host);
    await c.start();
    expect(c.status).toBe("error");
    expect(c.error).toMatch(/could not start/);
    expect((await c.call("echo", {})).ok).toBe(false);
  });
});

describe("McpManager", () => {
  it("exposes read-only knowledge and describes tools for the agent", async () => {
    const m = new McpManager(fakeHost().host);
    m.sync([{ id: "fake", command: "x", args: [], env: {}, enabled: true }]);
    await new Promise((r) => setTimeout(r, 20));
    const ext = m.external();
    expect(ext.available()).toBe(true);
    expect(ext.readOnly("fake", "echo")).toBe(true);
    expect(ext.readOnly("fake", "wipe")).toBe(false);
    expect(ext.readOnly("fake", "missing")).toBe(false);
    expect(ext.describe()).toContain('server "fake", tool "echo"(q: string)');
    expect((await ext.call("fake", "missing", {})).ok).toBe(false);
  });
});

describe("helpers", () => {
  it("splits a command line with quotes", () => {
    expect(splitCommandLine('npx -y srv "/my path" \'a b\'')).toEqual(["npx", "-y", "srv", "/my path", "a b"]);
  });
  it("makes safe server names", () => { expect(serverName("My Server!")).toBe("my-server"); });
});
