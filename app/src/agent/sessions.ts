import type { AgentEvent } from "@core/agent/loop";
import type { FileDiff } from "@core/agent/diff";
import type { Change } from "./apply";

/**
 * Coding sessions: a project folder and the tasks run on it, each with the steps the agent took, shown as blocks. Sessions are saved, so a closed app comes back with its
 * transcripts, and changes that were waiting for review can still be applied.
 */

export type BlockStatus = "running" | "ok" | "error";
export type Block =
  | { id: string; kind: "text"; text: string }
  | { id: string; kind: "note"; text: string; bad?: boolean; /** How many times in a row this happened. */ count?: number }
  | { id: string; kind: "tool"; name: string; label: string; path?: string; status: BlockStatus; output?: string; patch?: string };

/** One step of the plan the agent wrote for itself (the update_plan tool). */
export interface PlanStep { text: string; state: "done" | "active" | "todo" }

/** Reads the plan text the model writes: one step a line, "[x]" done, "[>]" in progress, "[ ]" still to do. A line with no marker is a step still to do. */
export function parsePlan(text: string): PlanStep[] {
  const steps: PlanStep[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim().replace(/^[-*]\s+/, "");
    if (!line) continue;
    const m = /^\[( |x|X|>|-|~)\]\s*(.*)$/.exec(line);
    const body = (m ? m[2] : line.replace(/^\d+[.)]\s+/, "")) as string;
    if (!body) continue;
    const mark = m ? (m[1] as string) : " ";
    steps.push({ text: body.slice(0, 200), state: mark === "x" || mark === "X" ? "done" : mark === ">" || mark === "~" ? "active" : "todo" });
  }
  return steps.slice(0, 12);
}

export type TurnStatus = "running" | "done" | "stopped" | "failed" | "interrupted";

export interface Turn {
  id: string;
  task: string;
  created: number;
  status: TurnStatus;
  blocks: Block[];
  summary?: string;
  /** When the task ended. */
  ended?: number;
  /** The agent's own plan, kept up to date as it works. */
  plan?: PlanStep[];
  /** The task was run in "plan only" mode, so no changes are expected. */
  planOnly?: boolean;
  /** Proposed changes, kept so they can be applied after a restart. */
  changes?: Change[];
  diffs?: FileDiff[];
  /** The checkpoint of an apply (to undo it). */
  applied?: string;
  discarded?: boolean;
  /** Changes existed but were too large to keep across a restart. */
  lost?: boolean;
  usage?: { steps: number; tokens: number };
  /** Tokens the model's context held at the last step. */
  context?: number;
  error?: string;
}

export interface CodeSession { id: string; title: string; root: string; created: number; updated: number; turns: Turn[] }

export type Dot = "running" | "review" | "done" | "failed" | "idle";

const WRITES = new Set(["write_file", "edit_file", "delete_file"]);
let seq = 0;
const bid = (): string => `b${Date.now().toString(36)}${(seq++).toString(36)}`;

/** The one line a tool call gets in the transcript. */
export function toolLabel(name: string, args: Record<string, unknown>): { label: string; path?: string } {
  const str = (k: string): string => (typeof args[k] === "string" ? (args[k] as string) : "");
  const clip = (s: string, n = 70): string => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
  switch (name) {
    case "read_file": return { label: `Read ${str("path")}`, path: str("path") };
    case "list_files": return { label: str("dir") ? `Listed ${str("dir")}` : "Listed files" };
    case "search": return { label: `Searched for “${clip(str("pattern"), 40)}”` };
    case "write_file": return { label: `Wrote ${str("path")}`, path: str("path") };
    case "edit_file": return { label: `Edited ${str("path")}`, path: str("path") };
    case "delete_file": return { label: `Deleted ${str("path")}`, path: str("path") };
    case "repo_map": return { label: "Mapped the project" };
    case "run_command": return { label: `Ran ${clip(str("command"))}` };
    case "web_search": return { label: `Searched the web for “${clip(str("query"), 40)}”` };
    case "fetch_url": return { label: `Opened ${clip(str("url"))}` };
    case "use_tool": return { label: `Used ${str("server")}/${str("tool")}` };
    default: return { label: name };
  }
}

/** Tool steps that follow each other, shown as one line ("Read 2 files, ran 3 commands") that opens to the steps. */
export type Item = Block | { id: string; kind: "group"; tools: Array<Extract<Block, { kind: "tool" }>> };
export function groupBlocks(blocks: Block[]): Item[] {
  const out: Item[] = [];
  for (const b of blocks) {
    if (b.kind !== "tool") { out.push(b); continue; }
    const last = out[out.length - 1];
    if (last && last.kind === "group") last.tools.push(b);
    else out.push({ id: `g${b.id}`, kind: "group", tools: [b] });
  }
  return out;
}

const plural = (n: number, w: string): string => `${n} ${w}${n === 1 ? "" : "s"}`;
/** What a run of steps did, in words: "Read 2 files, wrote 1 file, ran 3 commands". */
export function summarizeTools(tools: Array<Extract<Block, { kind: "tool" }>>): string {
  const n = { read: 0, search: 0, write: 0, edit: 0, del: 0, run: 0, web: 0, other: 0 };
  for (const t of tools) {
    if (t.name === "read_file" || t.name === "list_files" || t.name === "repo_map") n.read++;
    else if (t.name === "search") n.search++;
    else if (t.name === "write_file") n.write++;
    else if (t.name === "edit_file") n.edit++;
    else if (t.name === "delete_file") n.del++;
    else if (t.name === "run_command") n.run++;
    else if (t.name === "web_search" || t.name === "fetch_url") n.web++;
    else if (t.name !== "update_plan") n.other++;
  }
  const parts: string[] = [];
  if (n.read) parts.push(`read ${plural(n.read, "file")}`);
  if (n.search) parts.push(`searched ${plural(n.search, "time")}`);
  if (n.write) parts.push(`wrote ${plural(n.write, "file")}`);
  if (n.edit) parts.push(`edited ${plural(n.edit, "file")}`);
  if (n.del) parts.push(`deleted ${plural(n.del, "file")}`);
  if (n.run) parts.push(`ran ${plural(n.run, "command")}`);
  if (n.web) parts.push(`browsed ${plural(n.web, "page")}`);
  if (n.other) parts.push(`used ${plural(n.other, "tool")}`);
  const s = parts.join(", ");
  return s ? s[0]!.toUpperCase() + s.slice(1) : "Planned";
}

/** Folds one agent event into a turn's blocks. */
export function applyEvent(blocks: Block[], e: AgentEvent): Block[] {
  switch (e.type) {
    case "thought": return [...blocks, { id: bid(), kind: "text", text: e.text }];
    case "tool_call": {
      if (e.name === "finish") return blocks;
      const { label, path } = toolLabel(e.name, e.args);
      return [...blocks, { id: bid(), kind: "tool", name: e.name, label, ...(path ? { path } : {}), status: "running" }];
    }
    case "tool_result": {
      if (e.name === "finish") return blocks;
      const i = [...blocks].reverse().findIndex((b) => b.kind === "tool" && b.name === e.name && b.status === "running");
      if (i < 0) return blocks;
      const at = blocks.length - 1 - i;
      return blocks.map((b, j) => (j === at && b.kind === "tool" ? { ...b, status: e.ok ? "ok" : "error", output: e.output.slice(0, 4000) } : b));
    }
    case "diff": return blocks.map((b) => {
      if (b.kind !== "tool" || !WRITES.has(b.name) || !b.path) return b;
      const d = e.diffs.find((x) => x.path === b.path);
      return d?.patch ? { ...b, patch: d.patch } : b;
    });
    case "continue": return blocks;   // a long file is being finished in a second go; nothing to show
    case "compact": {
      const last = blocks[blocks.length - 1];
      const text = "The conversation was compacted: older steps were replaced by a short record so the task could continue.";
      if (last && last.kind === "note" && last.text === text) return blocks;
      return [...blocks, { id: bid(), kind: "note", text }];
    }
    case "invalid": {
      const text = e.error === "the reply was not a tool call" ? "The model answered in plain text instead of using a tool, so it was asked again."
        : e.error === "the reply was cut off" ? "The model's reply was cut off, so it was asked to write smaller files."
        : `The model's tool call was not valid (${e.error}), so it was asked to fix it.`;
      const last = blocks[blocks.length - 1];
      if (last && last.kind === "note" && !last.bad && last.text === text) return [...blocks.slice(0, -1), { ...last, count: (last.count ?? 1) + 1 }];
      return [...blocks, { id: bid(), kind: "note", text }];
    }
    case "verify": return [...blocks, { id: bid(), kind: "note", text: `Check “${e.command}” ${e.ran ? (e.ok ? "passed" : "failed") : "was not run"}`, bad: e.ran && !e.ok }];
    default: return blocks;
  }
}

/** Folds one agent event into a turn: the blocks, and the plan when the agent writes or updates it. */
export function applyTurnEvent(t: Turn, e: AgentEvent): Turn {
  if (e.type === "tool_call" && e.name === "update_plan" && typeof e.args["steps"] === "string") {
    const plan = parsePlan(e.args["steps"]);
    if (plan.length) return { ...t, plan };
  }
  const next: Turn = { ...t, blocks: applyEvent(t.blocks, e) };
  // A weak model often forgets to tick its own plan: a step that names a file is done when that file has been written.
  if (e.type === "tool_result" && e.ok && WRITES.has(e.name) && next.plan?.length) {
    const b = [...next.blocks].reverse().find((x): x is Extract<Block, { kind: "tool" }> => x.kind === "tool" && x.name === e.name && x.status === "ok" && !!x.path);
    if (b?.path) return { ...next, plan: tickPlan(next.plan, b.path) };
  }
  return next;
}

/** Marks the steps that name `path` (or its file name) as done, and makes the first step still to do the active one. */
export function tickPlan(plan: PlanStep[], path: string): PlanStep[] {
  const full = path.toLowerCase(), base = (full.split("/").pop() ?? full);
  let out = plan.map((s) => (s.state !== "done" && (s.text.toLowerCase().includes(full) || s.text.toLowerCase().includes(base)) ? { ...s, state: "done" as const } : s));
  if (!out.some((s) => s.state === "active")) { const i = out.findIndex((s) => s.state === "todo"); if (i >= 0) out = out.map((s, j) => (j === i ? { ...s, state: "active" as const } : s)); }
  return out;
}

/** The task finished: every step of its plan is done. */
export const completePlan = (plan: PlanStep[] | undefined): PlanStep[] | undefined => plan?.map((s) => ({ ...s, state: "done" as const }));

export type Stage = "planning" | "exploring" | "writing" | "checking" | "researching";
const STAGE_OF: Record<string, Stage> = { read_file: "exploring", list_files: "exploring", search: "exploring", repo_map: "exploring", write_file: "writing", edit_file: "writing", delete_file: "writing", run_command: "checking", web_search: "researching", fetch_url: "researching", use_tool: "researching" };

/** What the agent is doing now, from what it is actually doing, for a run that gave no plan: the stage and the line to show. */
export function currentStage(t: Turn): { stage: Stage; label: string; seen: Stage[] } {
  const tools = t.blocks.filter((b): b is Extract<Block, { kind: "tool" }> => b.kind === "tool");
  const seen: Stage[] = [];
  for (const b of tools) { const st = STAGE_OF[b.name]; if (st && !seen.includes(st)) seen.push(st); }
  const running = [...tools].reverse().find((b) => b.status === "running");
  if (running) return { stage: STAGE_OF[running.name] ?? "exploring", label: running.label.replace(/^(Read|Listed|Searched|Wrote|Edited|Deleted|Mapped|Ran|Opened|Used)\b/, (w) => ({ Read: "Reading", Listed: "Listing", Searched: "Searching", Wrote: "Writing", Edited: "Editing", Deleted: "Deleting", Mapped: "Mapping", Ran: "Running", Opened: "Opening", Used: "Using" } as Record<string, string>)[w] ?? w), seen };
  const lastTool = tools[tools.length - 1];
  if (!lastTool) return { stage: "planning", label: "Planning the change", seen };
  return { stage: STAGE_OF[lastTool.name] ?? "exploring", label: "Deciding the next step", seen };
}

export const lastTurn = (s: CodeSession): Turn | undefined => s.turns[s.turns.length - 1];
export const reviewable = (t: Turn): boolean => !!t.changes?.length && !t.applied && !t.discarded;

export function sessionDot(s: CodeSession): Dot {
  const t = lastTurn(s);
  if (!t) return "idle";
  if (t.status === "running") return "running";
  if (t.status === "failed") return "failed";
  if (reviewable(t)) return "review";
  return t.status === "done" ? "done" : "idle";
}

export const folderName = (root: string): string => root.split(/[\\/]/).filter(Boolean).pop() ?? root;

/** Sessions grouped by project folder, most recently used first. */
export function groupByProject(sessions: CodeSession[]): Array<{ root: string; name: string; sessions: CodeSession[] }> {
  const map = new Map<string, CodeSession[]>();
  for (const s of [...sessions].sort((a, b) => b.updated - a.updated)) map.set(s.root, [...(map.get(s.root) ?? []), s]);
  return [...map.entries()].map(([root, list]) => ({ root, name: folderName(root), sessions: list }));
}

const KEY = "clutor.sessions.v1";
const KEEP_CHANGES_BYTES = 600_000;

/** What gets saved: a turn that was running comes back "interrupted", and changes too large to keep are dropped (and said to be). */
export function toSaved(sessions: CodeSession[]): CodeSession[] {
  return sessions.slice(0, 60).map((s) => ({
    ...s,
    turns: s.turns.slice(-20).map((t): Turn => {
      const size = (t.changes ?? []).reduce((n, c) => n + (c.before?.length ?? 0) + (c.after?.length ?? 0), 0);
      const base: Turn = t.status === "running" ? { ...t, status: "interrupted" } : t;
      if (size > KEEP_CHANGES_BYTES) { const { changes: _c, ...rest } = base; void _c; return { ...rest, lost: true }; }
      return base;
    }),
  }));
}

export const loadSessions = (): CodeSession[] => { try { const r = localStorage.getItem(KEY); return r ? toSaved(JSON.parse(r) as CodeSession[]) : []; } catch { return []; } };
export const saveSessions = (s: CodeSession[]): void => { try { localStorage.setItem(KEY, JSON.stringify(toSaved(s))); } catch { /* storage full or unavailable */ } };

export const newSession = (root: string, now = Date.now()): CodeSession => ({ id: `s${now.toString(36)}${(seq++).toString(36)}`, title: "New session", root, created: now, updated: now, turns: [] });
