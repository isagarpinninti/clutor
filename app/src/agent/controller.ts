import type { AgentHost, WebApi } from "../bridge";
import { applyChanges, ConflictError, undoCheckpoint } from "./apply";
import type { AgentSession } from "./llm";
import type { McpManager } from "./mcp";
import { runCodingTask } from "./runner";
import { liveStore } from "./live";
import { applyTurnEvent, completePlan, reviewable, type CodeSession, type Turn } from "./sessions";

/**
 * Runs coding tasks, asks for approvals, and applies or undoes changes. It is the one place that does these things, so the desktop screen and a paired phone behave the same:
 * both start tasks here, both answer the same pending approvals, and both see the same sessions. Where a request came from is recorded with the task, never trusted for anything.
 */

export type Permission = "ask" | "auto" | "plan";
export type Source = "local" | "remote";
export interface Approval { id: string; sessionId: string; command: string; network: boolean; source: Source; device?: string; at: number }
export interface Outcome { ok: boolean; error?: string }

export interface Deps {
  host: AgentHost;
  web: WebApi;
  getSession(id: string): CodeSession | undefined;
  change(id: string, f: (s: CodeSession) => CodeSession): void;
  /** The engine, loading the model first if needed. */
  engine(): Promise<AgentSession | null>;
  /** Takes the engine from background jobs for a run; the result gives it back. */
  hold(): Promise<() => void>;
  mcp(): McpManager | undefined;
  config(): { maxSteps: number; contextTokens: number; modelName: string };
  onUsage(u: { steps: number; tokens: number }): void;
  /** A task ended (for a notification). */
  onFinish?(session: CodeSession, turn: Turn): void;
  /** The list of pending approvals changed. */
  onApprovals(list: Approval[]): void;
  /** Writes to the remote activity log. */
  log?(entry: { device?: string; what: string }): void;
  id(): string;
  titleFrom(text: string): string;
}

export class AgentController {
  private aborts = new Map<string, AbortController>();
  private pending = new Map<string, { approval: Approval; resolve: (yes: boolean) => void }>();

  constructor(private readonly d: Deps) {}

  isRunning(sessionId: string): boolean { return this.aborts.has(sessionId); }
  approvals(): Approval[] { return [...this.pending.values()].map((p) => p.approval); }

  private emit(): void { this.d.onApprovals(this.approvals()); }
  private patch(sid: string, tid: string, f: (t: Turn) => Turn): void { this.d.change(sid, (s) => ({ ...s, updated: Date.now(), turns: s.turns.map((t) => (t.id === tid ? f(t) : t)) })); }

  /** Starts a task on a session and resolves when it ends. */
  async run(sessionId: string, text: string, o: { perm: Permission; source: Source; device?: string }): Promise<void> {
    const session = this.d.getSession(sessionId);
    if (!session) throw new Error("that session no longer exists");
    if (this.isRunning(sessionId)) throw new Error("a task is already running in this session");
    const task = text.trim();
    if (!task) throw new Error("the task is empty");
    const turn: Turn = { id: this.d.id(), task, created: Date.now(), status: "running", blocks: [], ...(o.perm === "plan" ? { planOnly: true } : {}) };
    const ac = new AbortController();
    this.aborts.set(sessionId, ac);
    this.d.change(sessionId, (x) => ({ ...x, title: x.turns.length ? x.title : this.d.titleFrom(task), updated: Date.now(), turns: [...x.turns, turn] }));
    this.d.log?.({ ...(o.device ? { device: o.device } : {}), what: `task started (${o.perm}) in ${session.root}: ${task.slice(0, 120)}` });

    let release: (() => void) | undefined;
    try {
      const engine = await this.d.engine();
      if (!engine) { this.patch(sessionId, turn.id, (t) => ({ ...t, status: "failed", error: "The model could not be loaded." })); return; }
      release = await this.d.hold();
      const cfg = this.d.config();
      const out = await runCodingTask({
        host: this.d.host, web: this.d.web, root: session.root, session: engine, modelName: cfg.modelName, task, signal: ac.signal, planOnly: o.perm === "plan",
        maxSteps: cfg.maxSteps, contextTokens: cfg.contextTokens, ...(this.d.mcp() ? { mcp: this.d.mcp() as McpManager } : {}),
        approve: (info) => this.approve(sessionId, info, o),
        onContext: (n) => this.patch(sessionId, turn.id, (t) => ({ ...t, context: n })),
        onEvent: (e) => {
          if (e.type === "partial") { liveStore.set(turn.id, e.text); return; }   // the reply as it is written: kept outside the saved state
          if (e.type === "tool_call" || e.type === "thought" || e.type === "invalid") liveStore.clear(turn.id);
          this.patch(sessionId, turn.id, (t) => applyTurnEvent(t, e));
        },
      });
      const changes = o.perm === "plan" ? [] : await out.ws.changeSet();
      const diffs = o.perm === "plan" ? [] : out.result.diffs.filter((x) => x.patch || x.status !== "modified");
      this.d.onUsage({ steps: out.result.steps, tokens: out.result.usage.completionTokens });
      this.patch(sessionId, turn.id, (t) => ({
        ...t, ...(out.result.status === "finished" && t.plan ? { plan: completePlan(t.plan) as NonNullable<typeof t.plan> } : {}), ended: Date.now(), status: out.result.status === "aborted" ? "stopped" : out.result.status === "error" || out.result.status === "stuck" ? "failed" : "done", summary: out.result.summary,
        usage: { steps: out.result.steps, tokens: out.result.usage.completionTokens }, ...(out.result.status === "stuck" ? { error: "The model kept answering in text and did not use the tools, so nothing was changed. Its last answer is shown above." } : out.result.error ? { error: out.result.error } : {}), ...(diffs.length ? { changes, diffs } : {}),
      }));
    } catch (e) {
      this.patch(sessionId, turn.id, (t) => ({ ...t, ended: Date.now(), status: "failed", error: e instanceof Error ? e.message : String(e) }));
    } finally {
      liveStore.clear(turn.id);
      release?.();
      this.aborts.delete(sessionId);
      for (const [id, p] of this.pending) if (p.approval.sessionId === sessionId) { this.pending.delete(id); p.resolve(false); }
      this.emit();
      const done = this.d.getSession(sessionId);
      const last = done?.turns.find((t) => t.id === turn.id);
      if (done && last) this.d.onFinish?.(done, last);
      this.d.log?.({ ...(o.device ? { device: o.device } : {}), what: `task ${last?.status ?? "ended"}${last?.diffs?.length ? `, ${last.diffs.length} file(s) changed, waiting for review` : ""}` });
    }
  }

  private approve(sessionId: string, info: { command: string; network: boolean }, o: { perm: Permission; source: Source; device?: string }): Promise<boolean> {
    if (o.perm === "plan") return Promise.resolve(false);
    // "Auto-run checks" is only ever given to a task started on this computer, and still asks for anything that uses the network.
    if (o.perm === "auto" && o.source === "local" && !info.network) return Promise.resolve(true);
    return new Promise<boolean>((resolve) => {
      const approval: Approval = { id: this.d.id(), sessionId, command: info.command, network: info.network, source: o.source, ...(o.device ? { device: o.device } : {}), at: Date.now() };
      this.pending.set(approval.id, { approval, resolve });
      this.emit();
    });
  }

  /** Answers a pending approval. Returns false if it was already answered or the task ended. */
  answer(approvalId: string, yes: boolean, by?: { source: Source; device?: string }): boolean {
    const p = this.pending.get(approvalId);
    if (!p) return false;
    this.pending.delete(approvalId);
    this.emit();
    this.d.log?.({ ...(by?.device ? { device: by.device } : {}), what: `${yes ? "allowed" : "refused"}: ${p.approval.command.slice(0, 120)}` });
    p.resolve(yes);
    return true;
  }

  stop(sessionId: string): void { this.aborts.get(sessionId)?.abort(); }

  private turn(sessionId: string, turnId: string): { session: CodeSession; turn: Turn } | { error: string } {
    const session = this.d.getSession(sessionId);
    const turn = session?.turns.find((t) => t.id === turnId);
    return session && turn ? { session, turn } : { error: "that task no longer exists" };
  }

  /** Writes a task's changes into the folder, if nothing changed there meanwhile. */
  async apply(sessionId: string, turnId: string, by?: { device?: string }): Promise<Outcome> {
    const f = this.turn(sessionId, turnId);
    if ("error" in f) return { ok: false, error: f.error };
    if (!reviewable(f.turn) || !f.turn.changes) return { ok: false, error: "there is nothing to apply" };
    try {
      const id = await applyChanges(this.d.host, f.session.root, f.turn.changes, f.turn.task.slice(0, 80));
      this.patch(sessionId, turnId, (t) => ({ ...t, applied: id }));
      this.d.log?.({ ...(by?.device ? { device: by.device } : {}), what: `applied ${f.turn.changes.length} file(s) to ${f.session.root}` });
      return { ok: true };
    } catch (e) {
      return { ok: false, error: e instanceof ConflictError ? `Nothing was written. ${e.message}. Run the task again to start from the current files.` : e instanceof Error ? e.message : String(e) };
    }
  }

  async undo(sessionId: string, turnId: string, by?: { device?: string }): Promise<Outcome> {
    const f = this.turn(sessionId, turnId);
    if ("error" in f) return { ok: false, error: f.error };
    if (!f.turn.applied) return { ok: false, error: "that task was not applied" };
    try {
      await undoCheckpoint(this.d.host, f.session.root, f.turn.applied);
      this.patch(sessionId, turnId, (t) => { const { applied: _a, ...rest } = t; void _a; return { ...rest, discarded: true }; });
      this.d.log?.({ ...(by?.device ? { device: by.device } : {}), what: `undid the changes in ${f.session.root}` });
      return { ok: true };
    } catch (e) {
      return { ok: false, error: e instanceof ConflictError ? `Not undone, because ${e.paths.join(", ")} changed after the apply.` : e instanceof Error ? e.message : String(e) };
    }
  }

  discard(sessionId: string, turnId: string, by?: { device?: string }): Outcome {
    const f = this.turn(sessionId, turnId);
    if ("error" in f) return { ok: false, error: f.error };
    this.patch(sessionId, turnId, (t) => ({ ...t, discarded: true }));
    this.d.log?.({ ...(by?.device ? { device: by.device } : {}), what: "discarded the proposed changes" });
    return { ok: true };
  }
}
