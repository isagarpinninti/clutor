import type { AgentHost } from "../bridge";

/**
 * A client for MCP servers (Model Context Protocol): JSON-RPC 2.0, one message per line, over the server's standard input and output. It performs the handshake, lists
 * the server's tools and calls them. Servers are programs the user configured and chose to run; their tool calls go through the agent's approval unless the server marks
 * the tool read-only.
 */

export interface McpServerConfig { id: string; command: string; args: string[]; env: Record<string, string>; enabled: boolean }
export interface McpTool { name: string; description: string; inputSchema: { properties?: Record<string, { type?: string; description?: string }>; required?: string[] }; readOnly: boolean }
export type McpStatus = "stopped" | "starting" | "ready" | "error";

interface Pending { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }

export class McpClient {
  status: McpStatus = "stopped";
  error?: string;
  tools: McpTool[] = [];
  log: string[] = [];
  private seq = 0;
  private pending = new Map<number, Pending>();

  constructor(readonly config: McpServerConfig, private readonly host: AgentHost, private readonly onChange: () => void = () => undefined) {}

  private set(status: McpStatus, error?: string): void { this.status = status; if (error !== undefined) this.error = error; else delete this.error; this.onChange(); }

  async start(cwd?: string): Promise<void> {
    if (this.status === "starting" || this.status === "ready") return;
    this.set("starting");
    try {
      await this.host.mcpStart(this.config.id, { command: this.config.command, args: this.config.args, env: this.config.env, ...(cwd ? { cwd } : {}) },
        (line) => this.onLine(line), () => this.onExit(), (l) => { this.log = [...this.log.slice(-49), l]; });
      await this.request("initialize", { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "clutor", version: "0.4.0" } }, 30_000);
      await this.host.mcpSend(this.config.id, JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }));
      this.tools = await this.listTools();
      this.set("ready");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      await this.host.mcpStop(this.config.id).catch(() => undefined);
      this.failAll(new Error(msg));
      this.set("error", msg);
    }
  }

  async stop(): Promise<void> {
    this.failAll(new Error("the server was stopped"));
    await this.host.mcpStop(this.config.id).catch(() => undefined);
    this.tools = [];
    this.set("stopped");
  }

  private async listTools(): Promise<McpTool[]> {
    const out: McpTool[] = [];
    let cursor: string | undefined;
    for (let page = 0; page < 10; page++) {
      const r = (await this.request("tools/list", cursor ? { cursor } : {}, 30_000)) as { tools?: Array<{ name: string; description?: string; inputSchema?: McpTool["inputSchema"]; annotations?: { readOnlyHint?: boolean } }>; nextCursor?: string };
      for (const t of r.tools ?? []) out.push({ name: t.name, description: t.description ?? "", inputSchema: t.inputSchema ?? {}, readOnly: t.annotations?.readOnlyHint === true });
      if (!r.nextCursor) break;
      cursor = r.nextCursor;
    }
    return out;
  }

  async call(tool: string, args: Record<string, unknown>, timeoutMs = 120_000): Promise<{ ok: boolean; output: string }> {
    if (this.status !== "ready") return { ok: false, output: `the server ${this.config.id} is not running` };
    try {
      const r = (await this.request("tools/call", { name: tool, arguments: args }, timeoutMs)) as { content?: Array<{ type: string; text?: string }>; isError?: boolean };
      const text = (r.content ?? []).map((c) => (c.type === "text" ? c.text ?? "" : `[${c.type} content not shown]`)).join("\n");
      return { ok: r.isError !== true, output: text || "(no output)" };
    } catch (e) {
      return { ok: false, output: e instanceof Error ? e.message : String(e) };
    }
  }

  private request(method: string, params: unknown, timeoutMs: number): Promise<unknown> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`${method} timed out after ${Math.round(timeoutMs / 1000)} s`)); }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.host.mcpSend(this.config.id, JSON.stringify({ jsonrpc: "2.0", id, method, params })).catch((e: unknown) => {
        clearTimeout(timer); this.pending.delete(id); reject(e instanceof Error ? e : new Error(String(e)));
      });
    });
  }

  private onLine(line: string): void {
    let msg: { id?: number; result?: unknown; error?: { message?: string }; method?: string };
    try { msg = JSON.parse(line); } catch { return; }   // servers may print other text; only JSON-RPC lines matter
    if (typeof msg.id === "number" && this.pending.has(msg.id)) {
      const p = this.pending.get(msg.id) as Pending;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(msg.error.message ?? "the server returned an error")); else p.resolve(msg.result);
    } else if (msg.id !== undefined && msg.method) {
      // A request from the server (sampling, roots...) is not supported: say so instead of leaving it waiting.
      void this.host.mcpSend(this.config.id, JSON.stringify({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "not supported by this client" } })).catch(() => undefined);
    }
  }

  private onExit(): void {
    this.failAll(new Error("the server exited"));
    if (this.status !== "stopped") this.set("error", this.error ?? "the server exited");
  }

  private failAll(e: Error): void { for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(e); } this.pending.clear(); }
}

const KEY = "clutor.mcp.v1";

export const loadServers = (): McpServerConfig[] => { try { const r = localStorage.getItem(KEY); return r ? (JSON.parse(r) as McpServerConfig[]) : []; } catch { return []; } };
export const saveServers = (s: McpServerConfig[]): void => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ } };

/** `npx -y @modelcontextprotocol/server-filesystem /tmp` -> command and arguments (quotes group words; there is no shell). */
export function splitCommandLine(line: string): string[] {
  const out: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) out.push(m[1] ?? m[2] ?? m[3] ?? "");
  return out;
}

export const serverName = (s: string): string => s.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "server";

/** The servers the user configured, and the one view of their tools the agent gets. */
export class McpManager {
  private clients = new Map<string, McpClient>();
  constructor(private readonly host: AgentHost, private readonly onChange: () => void = () => undefined) {}

  list(): McpClient[] { return [...this.clients.values()]; }

  sync(configs: McpServerConfig[], cwd?: string): void {
    for (const c of configs) {
      if (!this.clients.has(c.id)) this.clients.set(c.id, new McpClient(c, this.host, this.onChange));
      const cl = this.clients.get(c.id) as McpClient;
      if (c.enabled && (cl.status === "stopped")) void cl.start(cwd);
      if (!c.enabled && cl.status !== "stopped") void cl.stop();
    }
    for (const [id, cl] of this.clients) if (!configs.some((c) => c.id === id)) { void cl.stop(); this.clients.delete(id); }
    this.onChange();
  }

  async restart(id: string, cwd?: string): Promise<void> { const c = this.clients.get(id); if (!c) return; await c.stop(); await c.start(cwd); }

  /** What the agent needs: the tools it may call (through `use_tool`), and a description to put in its instructions. */
  external() {
    const ready = () => this.list().filter((c) => c.status === "ready");
    const find = (server: string, tool: string): { c: McpClient; t: McpTool } | undefined => { const c = ready().find((x) => x.config.id === server); const t = c?.tools.find((x) => x.name === tool); return c && t ? { c, t } : undefined; };
    return {
      available: () => ready().some((c) => c.tools.length > 0),
      readOnly: (server: string, tool: string) => find(server, tool)?.t.readOnly === true,
      call: async (server: string, tool: string, args: Record<string, unknown>) => {
        const f = find(server, tool);
        if (!f) return { ok: false, output: `there is no tool ${server}/${tool}; the connected tools are listed in the instructions` };
        return f.c.call(tool, args);
      },
      describe: (): string => {
        const lines: string[] = [];
        for (const c of ready()) for (const t of c.tools.slice(0, 40)) {
          const props = Object.entries(t.inputSchema.properties ?? {}).map(([k, v]) => `${k}${t.inputSchema.required?.includes(k) ? "" : "?"}: ${v.type ?? "any"}`).join(", ");
          lines.push(`- server "${c.config.id}", tool "${t.name}"(${props})${t.readOnly ? " [read-only]" : ""}: ${t.description.replace(/\s+/g, " ").slice(0, 160)}`);
        }
        return lines.length ? `External tools you can call with use_tool (server, tool, arguments as a JSON string):\n${lines.join("\n")}` : "";
      },
    };
  }
}
