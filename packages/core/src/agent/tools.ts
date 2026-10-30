import { planCommand, type CommandPlan } from "./commands";
import { buildRepoMap } from "./repomap";
import { type ShadowWorkspace, WorkspaceError } from "./workspace";

/**
 * The agent's tools: a strict schema for each (which is also what the model is shown), a validator that refuses anything that does not
 * match it exactly, and an executor that runs the call against the shadow workspace. A failure is returned to the model as text it can
 * act on; it never throws out of the loop.
 */

export type ToolName = "list_files" | "read_file" | "search" | "write_file" | "edit_file" | "delete_file" | "repo_map" | "run_command" | "finish";

interface Field { type: "string" | "integer" | "boolean"; description: string; required?: boolean; min?: number; max?: number; maxLength?: number }
interface ToolSpec { name: ToolName; description: string; fields: Record<string, Field> }

const S = (description: string, o: Partial<Field> = {}): Field => ({ type: "string", description, maxLength: 200_000, ...o });
const I = (description: string, o: Partial<Field> = {}): Field => ({ type: "integer", description, ...o });
const B = (description: string): Field => ({ type: "boolean", description });

export const TOOLS: ToolSpec[] = [
  { name: "list_files", description: "List project files, optionally only under a directory.", fields: { dir: S("Directory to list, relative to the project (default: everything).", { maxLength: 1024 }) } },
  { name: "read_file", description: "Read a text file. Use start_line and end_line (1-based, inclusive) for a part of a large file.", fields: { path: S("File path relative to the project.", { required: true, maxLength: 1024 }), start_line: I("First line to return.", { min: 1 }), end_line: I("Last line to return.", { min: 1 }) } },
  { name: "search", description: "Search file contents for a text or regular expression. Returns file:line: text for each match.", fields: { pattern: S("What to look for.", { required: true, maxLength: 500 }), dir: S("Only search under this directory.", { maxLength: 1024 }), regex: B("Treat the pattern as a regular expression (default: plain text)."), max_results: I("Most matches to return (default 50).", { min: 1, max: 200 }) } },
  { name: "write_file", description: "Create a file or replace it entirely with the given content. Prefer edit_file for small changes to an existing file.", fields: { path: S("File path relative to the project.", { required: true, maxLength: 1024 }), content: S("The complete new contents of the file.", { required: true }) } },
  { name: "edit_file", description: "Replace exact text in an existing file. old_text must match the file exactly (including indentation) and occur once, unless replace_all is true.", fields: { path: S("File path relative to the project.", { required: true, maxLength: 1024 }), old_text: S("Text to replace, copied exactly from the file.", { required: true }), new_text: S("Replacement text.", { required: true }), replace_all: B("Replace every occurrence.") } },
  { name: "delete_file", description: "Delete a file.", fields: { path: S("File path relative to the project.", { required: true, maxLength: 1024 }) } },
  { name: "repo_map", description: "Show the project tree and the main symbols of each source file.", fields: {} },
  { name: "run_command", description: "Run a command in a scratch copy of the project that includes your changes (for example the tests). Only a short list of programs is allowed, there is no shell, and the user may be asked to approve it.", fields: { command: S("The command line, for example `npm test`.", { required: true, maxLength: 1000 }) } },
  { name: "finish", description: "Call this when the task is done, or when you cannot continue. Summarise what you changed and anything the user must know.", fields: { summary: S("What was done, and what was not.", { required: true, maxLength: 8000 }) } },
];

export interface ToolCall { name: string; args: Record<string, unknown> }
export type Validation = { ok: true; call: { name: ToolName; args: Record<string, unknown> } } | { ok: false; error: string };

export function validateToolCall(name: unknown, args: unknown): Validation {
  const spec = TOOLS.find((t) => t.name === name);
  if (!spec) return { ok: false, error: `unknown tool ${JSON.stringify(name)}; the tools are ${TOOLS.map((t) => t.name).join(", ")}` };
  if (args === null || typeof args !== "object" || Array.isArray(args)) return { ok: false, error: `${spec.name}: arguments must be an object` };
  const a = args as Record<string, unknown>;
  for (const k of Object.keys(a)) if (!(k in spec.fields)) return { ok: false, error: `${spec.name}: unknown argument "${k}"; allowed: ${Object.keys(spec.fields).join(", ") || "none"}` };
  for (const [k, f] of Object.entries(spec.fields)) {
    const v = a[k];
    if (v === undefined || v === null) { if (f.required) return { ok: false, error: `${spec.name}: "${k}" is required` }; delete a[k]; continue; }
    if (f.type === "string") {
      if (typeof v !== "string") return { ok: false, error: `${spec.name}: "${k}" must be a string` };
      if (f.maxLength !== undefined && v.length > f.maxLength) return { ok: false, error: `${spec.name}: "${k}" is longer than ${f.maxLength} characters` };
    } else if (f.type === "integer") {
      if (typeof v !== "number" || !Number.isInteger(v)) return { ok: false, error: `${spec.name}: "${k}" must be an integer` };
      if ((f.min !== undefined && v < f.min) || (f.max !== undefined && v > f.max)) return { ok: false, error: `${spec.name}: "${k}" must be between ${f.min ?? "-inf"} and ${f.max ?? "inf"}` };
    } else if (typeof v !== "boolean") return { ok: false, error: `${spec.name}: "${k}" must be true or false` };
  }
  return { ok: true, call: { name: spec.name, args: a } };
}

/** The tool list in the OpenAI function-calling format. */
export function toolSchemas(): Array<{ type: "function"; function: { name: string; description: string; parameters: unknown } }> {
  return TOOLS.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name, description: t.description,
      parameters: {
        type: "object", additionalProperties: false,
        properties: Object.fromEntries(Object.entries(t.fields).map(([k, f]) => [k, { type: f.type, description: f.description, ...(f.min !== undefined ? { minimum: f.min } : {}), ...(f.max !== undefined ? { maximum: f.max } : {}) }])),
        required: Object.entries(t.fields).filter(([, f]) => f.required).map(([k]) => k),
      },
    },
  }));
}

/**
 * One JSON Schema that accepts exactly the replies the "json" tool protocol allows: `{"tool": <name>, "args": {...}}` for one of the tools,
 * with each tool's own argument types. A server that supports schema-constrained decoding (llama.cpp turns this into a grammar) can then
 * only produce a reply that parses and passes the validator's shape checks; the validator still runs, because limits like path rules are not in the schema.
 */
export function toolCallJsonSchema(): Record<string, unknown> {
  return {
    oneOf: TOOLS.map((t) => ({
      type: "object", additionalProperties: false, required: ["tool", "args"],
      properties: {
        tool: { const: t.name },
        args: {
          type: "object", additionalProperties: false,
          properties: Object.fromEntries(Object.entries(t.fields).map(([k, f]) => [k, { type: f.type, ...(f.min !== undefined ? { minimum: f.min } : {}), ...(f.max !== undefined ? { maximum: f.max } : {}), ...(f.type === "string" && f.maxLength ? { maxLength: Math.min(f.maxLength, 100_000) } : {}) }])),
          required: Object.entries(t.fields).filter(([, f]) => f.required).map(([k]) => k),
        },
      },
    })),
  };
}

export interface CommandResult { exitCode: number | null; output: string; timedOut: boolean; durationMs: number }

export interface ToolContext {
  ws: ShadowWorkspace;
  /** Runs an approved plan in a scratch copy including the shadow's changes. Absent: commands are refused. */
  runCommand?: (plan: Extract<CommandPlan, { ok: true }>, command: string) => Promise<CommandResult>;
  /** Asked before any command that runs project code or uses the network. Absent: such commands are refused. */
  approve?: (info: { command: string; network: boolean }) => Promise<boolean>;
  maxOutputChars?: number;
}

export interface ToolOutcome {
  ok: boolean;
  output: string;
  /** Set by run_command, for the receipt. */
  command?: { command: string; exitCode: number | null; approved: boolean };
  finished?: string;
}

const clip = (s: string, max: number): string => (s.length <= max ? s : `${s.slice(0, max)}\n... [${s.length - max} more characters not shown]`);
const underDir = (path: string, dir: string | undefined): boolean => {
  if (!dir) return true;
  const d = dir.replace(/\\/g, "/").replace(/^\.\//, "").replace(/\/+$/, "");
  return d === "" || d === "." || path === d || path.startsWith(`${d}/`);
};

export async function executeTool(ctx: ToolContext, name: ToolName, a: Record<string, unknown>): Promise<ToolOutcome> {
  const max = ctx.maxOutputChars ?? 12_000;
  try {
    switch (name) {
      case "list_files": {
        const files = (await ctx.ws.list()).filter((f) => underDir(f, a["dir"] as string | undefined));
        return { ok: true, output: files.length ? clip(files.slice(0, 500).join("\n") + (files.length > 500 ? `\n... ${files.length - 500} more` : ""), max) : "(no files)" };
      }
      case "read_file": {
        const text = await ctx.ws.read(a["path"] as string);
        if (text === null) return { ok: false, output: `${a["path"]} does not exist` };
        const lines = text.split("\n");
        const start = (a["start_line"] as number | undefined) ?? 1, end = Math.min(lines.length, (a["end_line"] as number | undefined) ?? lines.length);
        if (start > end) return { ok: false, output: `start_line ${start} is after end_line ${end}` };
        const body = lines.slice(start - 1, end).map((l, i) => `${String(start + i).padStart(5)}| ${l}`).join("\n");
        return { ok: true, output: clip(body, max) + (end < lines.length ? `\n[showing lines ${start}-${end} of ${lines.length}]` : "") };
      }
      case "search": {
        let re: RegExp;
        try { re = new RegExp(a["regex"] === true ? (a["pattern"] as string) : (a["pattern"] as string).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); } catch { return { ok: false, output: "the pattern is not a valid regular expression" }; }
        const limit = (a["max_results"] as number | undefined) ?? 50;
        const out: string[] = [];
        for (const f of await ctx.ws.list()) {
          if (!underDir(f, a["dir"] as string | undefined)) continue;
          const text = await ctx.ws.read(f).catch(() => null);
          if (text === null) continue;
          const ls = text.split("\n");
          for (let i = 0; i < ls.length && out.length < limit; i++) if (re.test(ls[i] as string)) out.push(`${f}:${i + 1}: ${(ls[i] as string).trim().slice(0, 200)}`);
          if (out.length >= limit) break;
        }
        return { ok: true, output: out.length ? out.join("\n") : "no matches" };
      }
      case "write_file": {
        const r = await ctx.ws.write(a["path"] as string, a["content"] as string);
        return { ok: true, output: `${r.created ? "created" : "replaced"} ${a["path"]} (${(a["content"] as string).split("\n").length} lines)` };
      }
      case "edit_file": {
        const r = await ctx.ws.edit(a["path"] as string, a["old_text"] as string, a["new_text"] as string, a["replace_all"] === true);
        return { ok: true, output: `edited ${a["path"]} (${r.replacements} replacement${r.replacements === 1 ? "" : "s"})` };
      }
      case "delete_file": {
        await ctx.ws.delete(a["path"] as string);
        return { ok: true, output: `deleted ${a["path"]}` };
      }
      case "repo_map":
        return { ok: true, output: await buildRepoMap(ctx.ws, max) };
      case "run_command": {
        const command = a["command"] as string;
        const plan = planCommand(command);
        if (!plan.ok) { ctx.ws.violations++; return { ok: false, output: `not allowed: ${plan.reason}`, command: { command, exitCode: null, approved: false } }; }
        if (!ctx.runCommand) return { ok: false, output: "commands cannot be run in this session", command: { command, exitCode: null, approved: false } };
        let approved = !plan.requiresApproval;
        if (!approved) approved = (await ctx.approve?.({ command, network: plan.steps.some((s) => s.network) })) === true;
        if (!approved) return { ok: false, output: "the user did not approve this command; continue without running it, or explain what you need", command: { command, exitCode: null, approved: false } };
        const r = await ctx.runCommand(plan, command);
        const header = r.timedOut ? "timed out" : `exit code ${r.exitCode}`;
        return { ok: r.exitCode === 0 && !r.timedOut, output: `${header}\n${clip(r.output, max)}`, command: { command, exitCode: r.exitCode, approved: true } };
      }
      case "finish":
        return { ok: true, output: "done", finished: a["summary"] as string };
    }
  } catch (e) {
    if (e instanceof WorkspaceError) return { ok: false, output: e.message };
    return { ok: false, output: `error: ${e instanceof Error ? e.message : String(e)}` };
  }
}
