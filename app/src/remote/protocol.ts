import type { CodeSession, Turn } from "../agent/sessions";

/**
 * What a phone may say to the computer, and what the computer says back. Everything a phone sends is untrusted input: it is parsed strictly here, and the handler checks
 * each request again before acting. A phone can only ask for "ask first" or "plan only"; "auto-run" exists only for a task started on the computer itself.
 */

export type PhoneMode = "ask" | "plan";
export type PhoneMsg =
  | { t: "state" }
  | { t: "projects" }
  | { t: "open"; session: string }
  | { t: "task"; text: string; mode?: string; session?: string; project?: { dir?: string; newName?: string }; ref?: string }
  | { t: "approve"; id: string; yes: boolean }
  | { t: "stop"; session: string }
  | { t: "apply" | "discard" | "undo"; session: string; turn: string; ref?: string }
  | { t: "ping" };

export const MAX_TASK_CHARS = 8000;
const str = (v: unknown, max = 200): v is string => typeof v === "string" && v.length > 0 && v.length <= max;

/** Parses one message from a phone, or says why it is not acceptable. */
export function parsePhoneMsg(text: string): { ok: true; msg: PhoneMsg } | { ok: false; error: string } {
  if (text.length > 20_000) return { ok: false, error: "message too long" };
  let v: Record<string, unknown>;
  try { const j = JSON.parse(text) as unknown; if (!j || typeof j !== "object" || Array.isArray(j)) throw new Error("x"); v = j as Record<string, unknown>; } catch { return { ok: false, error: "not a valid message" }; }
  const ref = typeof v["ref"] === "string" && v["ref"].length <= 40 ? { ref: v["ref"] } : {};
  switch (v["t"]) {
    case "state": case "projects": case "ping": return { ok: true, msg: { t: v["t"] } };
    case "open": case "stop": return str(v["session"], 80) ? { ok: true, msg: { t: v["t"], session: v["session"] } } : { ok: false, error: "missing session" };
    case "approve": return str(v["id"], 80) && typeof v["yes"] === "boolean" ? { ok: true, msg: { t: "approve", id: v["id"], yes: v["yes"] } } : { ok: false, error: "bad approval" };
    case "apply": case "discard": case "undo": return str(v["session"], 80) && str(v["turn"], 80) ? { ok: true, msg: { t: v["t"], session: v["session"], turn: v["turn"], ...ref } } : { ok: false, error: "missing session or task" };
    case "task": {
      if (typeof v["text"] !== "string" || !v["text"].trim()) return { ok: false, error: "the task is empty" };
      if (v["text"].length > MAX_TASK_CHARS) return { ok: false, error: `the task is longer than ${MAX_TASK_CHARS} characters` };
      const p = v["project"] as Record<string, unknown> | undefined;
      const project = p && typeof p === "object" ? { ...(typeof p["dir"] === "string" ? { dir: p["dir"] } : {}), ...(typeof p["newName"] === "string" ? { newName: p["newName"] } : {}) } : undefined;
      return { ok: true, msg: { t: "task", text: v["text"], ...(typeof v["mode"] === "string" ? { mode: v["mode"] } : {}), ...(str(v["session"], 80) ? { session: v["session"] } : {}), ...(project ? { project } : {}), ...ref } };
    }
    default: return { ok: false, error: "unknown message" };
  }
}

/** The mode for a phone's task. Missing means plan only; "auto" (or anything else) is refused. */
export function phoneMode(mode: string | undefined): PhoneMode | undefined {
  if (mode === undefined) return "plan";
  return mode === "ask" || mode === "plan" ? mode : undefined;
}

/** A folder name inside the projects folder: no separators, no dots at the start, nothing that walks up. */
export function plainFolderName(name: string): boolean {
  return /^[\p{L}\p{N}][\p{L}\p{N} ._-]{0,63}$/u.test(name) && !name.includes("..");
}

export interface SessionSummary { id: string; title: string; root: string; project: string; updated: number; status: string; review: boolean }

const lastOf = (s: CodeSession): Turn | undefined => s.turns[s.turns.length - 1];
export function summary(s: CodeSession): SessionSummary {
  const t = lastOf(s);
  return { id: s.id, title: s.title, root: s.root, project: s.root.split(/[\\/]/).filter(Boolean).pop() ?? s.root, updated: s.updated, status: t?.status ?? "idle", review: !!t?.changes?.length && !t.applied && !t.discarded };
}

/** A session as sent to a phone: the steps and the changes' diffs, without the files' full contents, and with long output cut. */
export function wireSession(s: CodeSession): unknown {
  const cut = (x: string | undefined, n: number): string | undefined => (x && x.length > n ? `${x.slice(0, n)}\n[cut off]` : x);
  return {
    id: s.id, title: s.title, project: summary(s).project, updated: s.updated,
    turns: s.turns.slice(-10).map((t) => ({
      id: t.id, task: t.task, status: t.status, summary: t.summary, error: t.error, applied: !!t.applied, discarded: !!t.discarded, context: t.context,
      blocks: t.blocks.slice(-80).map((b) => (b.kind === "tool" ? { ...b, output: cut(b.output, 600), patch: cut(b.patch, 6000) } : b)),
      diffs: t.diffs?.map((d) => ({ path: d.path, status: d.status, additions: d.additions, deletions: d.deletions, patch: cut(d.patch, 20_000) })),
    })),
  };
}

/** At most `max` requests in any `windowMs`. */
export class RateLimiter {
  private stamps: number[] = [];
  constructor(private readonly max: number, private readonly windowMs: number, private readonly now: () => number = Date.now) {}
  allow(): boolean {
    const t = this.now();
    this.stamps = this.stamps.filter((x) => t - x < this.windowMs);
    if (this.stamps.length >= this.max) return false;
    this.stamps.push(t);
    return true;
  }
}
