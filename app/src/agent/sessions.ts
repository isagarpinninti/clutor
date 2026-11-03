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
  | { id: string; kind: "note"; text: string; bad?: boolean }
  | { id: string; kind: "tool"; name: string; label: string; path?: string; status: BlockStatus; output?: string; patch?: string };

export type TurnStatus = "running" | "done" | "stopped" | "failed" | "interrupted";

export interface Turn {
  id: string;
  task: string;
  created: number;
  status: TurnStatus;
  blocks: Block[];
  summary?: string;
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
    case "invalid": return [...blocks, { id: bid(), kind: "note", text: `The model's reply could not be used: ${e.error}`, bad: true }];
    case "verify": return [...blocks, { id: bid(), kind: "note", text: `Check “${e.command}” ${e.ran ? (e.ok ? "passed" : "failed") : "was not run"}`, bad: e.ran && !e.ok }];
    default: return blocks;
  }
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
