import { CATALOG, GiB, QUANTS, ShadowWorkspace, buildRepoMap, fmtGiB, planMemory, TOOLS, executeTool, validateToolCall } from "@clutor/core";
import { NodeProject, hostInfo } from "@clutor/core/node";
import { resolveModel } from "./models";

/**
 * A Model Context Protocol server (JSON-RPC 2.0 over HTTP) with read-only tools, so an editor or agent that speaks MCP can ask Clutor
 * about a project and about whether a model fits. It cannot change files or run commands.
 */

interface Rpc { jsonrpc?: string; id?: string | number | null; method?: string; params?: Record<string, unknown> }

const READ_TOOLS = ["list_files", "read_file", "search", "repo_map"] as const;

function tools(): unknown[] {
  const fromAgent = TOOLS.filter((t) => (READ_TOOLS as readonly string[]).includes(t.name)).map((t) => ({
    name: t.name, description: t.description,
    inputSchema: {
      type: "object", additionalProperties: false,
      properties: { dir: { type: "string", description: "Project folder (must be one Clutor has been given access to)." }, ...Object.fromEntries(Object.entries(t.fields).filter(([k]) => !(t.name === "list_files" && k === "dir") && !(t.name === "search" && k === "dir")).map(([k, f]) => [k, { type: f.type, description: f.description }])), ...(t.name === "list_files" || t.name === "search" ? { subdir: { type: "string", description: "Only look under this folder." } } : {}) },
      required: ["dir", ...Object.entries(t.fields).filter(([, f]) => f.required).map(([k]) => k)],
    },
  }));
  return [
    ...fromAgent,
    { name: "plan_memory", description: "Would a model fit in this much RAM, and how fast could it possibly run from disk? Models: " + CATALOG.map((c) => c.id).join(", "), inputSchema: { type: "object", additionalProperties: false, properties: { model: { type: "string" }, ram_gib: { type: "number" }, quant: { type: "string", enum: Object.keys(QUANTS) }, context_tokens: { type: "integer" } }, required: ["model", "ram_gib"] } },
  ];
}

export async function handleMcp(req: Rpc, ctx: { isRoot(dir: string): Promise<string | null> }): Promise<{ status: number; body: unknown }> {
  const id = req.id ?? null;
  const ok = (result: unknown): { status: number; body: unknown } => ({ status: 200, body: { jsonrpc: "2.0", id, result } });
  const err = (code: number, message: string): { status: number; body: unknown } => ({ status: 200, body: { jsonrpc: "2.0", id, error: { code, message } } });
  if (req.jsonrpc !== "2.0" || typeof req.method !== "string") return err(-32600, "not a JSON-RPC 2.0 request");
  switch (req.method) {
    case "initialize": return ok({ protocolVersion: "2025-03-26", capabilities: { tools: {} }, serverInfo: { name: "clutor", version: "0.1.0" } });
    case "notifications/initialized": return { status: 202, body: "" };
    case "ping": return ok({});
    case "tools/list": return ok({ tools: tools() });
    case "tools/call": {
      const name = req.params?.["name"], args = (req.params?.["arguments"] ?? {}) as Record<string, unknown>;
      const text = (t: string, isError = false): { status: number; body: unknown } => ok({ content: [{ type: "text", text: t }], isError });
      if (name === "plan_memory") {
        try {
          const m = await resolveModel(`${String(args["model"])}${args["quant"] ? `@${String(args["quant"])}` : ""}`, { allowFiles: false });
          const h = hostInfo();
          const p = planMemory(m.layout, { hostRamBytes: Math.max(1, Number(args["ram_gib"])) * GiB, ...(args["context_tokens"] ? { contextTokens: Number(args["context_tokens"]) } : {}) });
          return text([`${m.name} (${m.quant}) on ${fmtGiB(p.ceilingBytes)}: ${p.strategy}`, ...p.components.map((c) => `- ${c.label}: ${fmtGiB(c.bytes)}`), ...p.speed.map((s) => `at ${(s.hitRate * 100).toFixed(0)}% cache hits: at most ${s.maxTokensPerSec === Infinity ? "unlimited" : s.maxTokensPerSec.toFixed(2)} tokens/s (disk bound)`), ...p.warnings.map((w) => `! ${w}`), `(this machine has ${fmtGiB(h.totalRamBytes)})`].join("\n"));
        } catch (e) { return text(e instanceof Error ? e.message : String(e), true); }
      }
      if (typeof name === "string" && (READ_TOOLS as readonly string[]).includes(name)) {
        const dir = typeof args["dir"] === "string" ? await ctx.isRoot(args["dir"]) : null;
        if (!dir) return text("that folder has not been opened in Clutor; open it in the app or run `clutor serve --dir <folder>`", true);
        const { dir: _d, subdir, ...rest } = args;
        void _d;
        const call = name === "list_files" || name === "search" ? { ...rest, ...(typeof subdir === "string" ? { dir: subdir } : {}) } : rest;
        const v = validateToolCall(name, call);
        if (!v.ok) return text(v.error, true);
        const r = await executeTool({ ws: new ShadowWorkspace(new NodeProject(dir)) }, v.call.name, v.call.args);
        return text(r.output, !r.ok);
      }
      return err(-32602, `unknown tool ${JSON.stringify(name)}`);
    }
    default: return err(-32601, `method not found: ${req.method}`);
  }
}
export { buildRepoMap };
