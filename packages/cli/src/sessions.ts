import { randomBytes } from "node:crypto";
import { realpath, stat } from "node:fs/promises";
import { join } from "node:path";
import { type ShadowWorkspace, evaluateGates, runAgent, type AgentEvent, type AgentResult, type FileDiff, type GateResult, type LlmClient } from "@clutor/core";
import { ConflictError, type NodeProject, appendReceipt, applyChanges, lastReceiptTag, loadOrCreateReceiptSigner, rollbackCheckpoint, scratchRunner, verifyReceiptLog } from "@clutor/core/node";
import type { ClutorConfig } from "./config";

/**
 * One agent task, held in memory so a UI can watch it (events are kept and replayed to late subscribers), approve commands it wants to
 * run, and then apply or discard what it did. Nothing reaches the project folder until `apply`.
 */

export type SessionEvent = AgentEvent | { type: "approval"; requestId: string; command: string; network: boolean } | { type: "approval_resolved"; requestId: string; allowed: boolean } | { type: "applied"; checkpointId: string; files: number } | { type: "rolled_back"; checkpointId: string };

export interface SessionInfo {
  id: string;
  dir: string;
  task: string;
  status: "running" | "finished" | "max_steps" | "aborted" | "stuck" | "error";
  startedAt: string;
  result?: AgentResult;
  diffs: FileDiff[];
  pendingApprovals: Array<{ requestId: string; command: string; network: boolean }>;
  checkpointId?: string;
  gates: GateResult[];
}

export class AgentSession {
  readonly id = `task_${randomBytes(6).toString("hex")}`;
  readonly events: SessionEvent[] = [];
  readonly startedAt = new Date().toISOString();
  status: SessionInfo["status"] = "running";
  result?: AgentResult;
  checkpointId?: string;
  private readonly subs = new Set<(e: SessionEvent | null) => void>();
  private readonly pending = new Map<string, { command: string; network: boolean; resolve: (b: boolean) => void; timer: NodeJS.Timeout }>();
  private readonly abort = new AbortController();
  private lastDiffs: FileDiff[] = [];
  private done: Promise<void> = Promise.resolve();

  constructor(
    readonly dir: string,
    readonly task: string,
    readonly project: NodeProject,
    readonly ws: ShadowWorkspace,
    private readonly cfg: ClutorConfig,
    private readonly hostRamBytes: number,
  ) {}

  private push(e: SessionEvent): void {
    this.events.push(e);
    if (e.type === "diff") this.lastDiffs = e.diffs;
    for (const s of this.subs) s(e);
  }

  /** Replays everything so far, then streams new events until the task ends. Returns a function that stops listening. */
  subscribe(from: number, send: (e: SessionEvent) => void, onEnd: () => void): () => void {
    for (const e of this.events.slice(from)) send(e);
    if (this.status !== "running" && this.pending.size === 0) { onEnd(); return () => undefined; }
    const f = (e: SessionEvent | null): void => { if (e === null) onEnd(); else send(e); };
    this.subs.add(f);
    return () => this.subs.delete(f);
  }

  start(llm: LlmClient): void {
    const root = this.dir;
    this.done = (async () => {
      const key = await loadOrCreateReceiptSigner(root);
      const previous = await lastReceiptTag(root);
      const result = await runAgent({
        task: this.task, llm, ws: this.ws, signal: this.abort.signal,
        maxSteps: this.cfg.agent.maxSteps, contextChars: Math.floor(this.cfg.memory.contextTokens * 2.5), ...(this.cfg.agent.verifyCommand ? { verifyCommand: this.cfg.agent.verifyCommand } : {}),
        runCommand: scratchRunner(this.project, this.ws, { timeoutMs: this.cfg.agent.commandTimeoutSeconds * 1000 }),
        approve: (info) => this.askApproval(info.command, info.network),
        onEvent: (e) => { if (e.type !== "done") this.push(e); },
        receipt: { signer: key, previous },
      });
      this.result = result;
      this.status = result.status;
      if (result.receipt) await appendReceipt(root, result.receipt);
      this.push({ type: "done", result });
      for (const s of this.subs) s(null);
    })().catch((e: unknown) => {
      this.status = "error";
      this.push({ type: "done", result: { status: "error", summary: e instanceof Error ? e.message : String(e), error: e instanceof Error ? e.message : String(e), steps: 0, toolCalls: 0, invalidReplies: 0, violations: 0, diffs: [], commands: [], usage: { promptTokens: 0, completionTokens: 0 } } });
      for (const s of this.subs) s(null);
    });
  }

  wait(): Promise<void> { return this.done; }

  private askApproval(command: string, network: boolean): Promise<boolean> {
    if (this.cfg.agent.autoApproveProjectCommands) return Promise.resolve(true);
    const requestId = `appr_${randomBytes(4).toString("hex")}`;
    return new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => this.resolveApproval(requestId, false), 10 * 60_000);
      this.pending.set(requestId, { command, network, resolve, timer });
      this.push({ type: "approval", requestId, command, network });
    });
  }

  resolveApproval(requestId: string, allow: boolean): boolean {
    const p = this.pending.get(requestId);
    if (!p) return false;
    clearTimeout(p.timer);
    this.pending.delete(requestId);
    this.push({ type: "approval_resolved", requestId, allowed: allow });
    p.resolve(allow);
    return true;
  }

  stop(): void {
    this.abort.abort();
    for (const id of [...this.pending.keys()]) this.resolveApproval(id, false);
  }

  /** Writes the agent's changes to the project (after checking nothing changed under it), with a checkpoint to roll back to. */
  async apply(force = false): Promise<{ checkpointId: string; files: number }> {
    if (this.status === "running") throw new Error("the task is still running");
    const changes = await this.ws.changeSet();
    if (changes.length === 0) throw new Error("there is nothing to apply");
    const r = await applyChanges(this.project, changes, { force });
    this.checkpointId = r.id;
    this.push({ type: "applied", checkpointId: r.id, files: r.applied.length });
    return { checkpointId: r.id, files: r.applied.length };
  }

  async rollback(force = false): Promise<void> {
    if (!this.checkpointId) throw new Error("nothing has been applied by this task");
    await rollbackCheckpoint(this.project, this.checkpointId, { force });
    this.push({ type: "rolled_back", checkpointId: this.checkpointId });
    delete this.checkpointId;
  }

  async info(): Promise<SessionInfo> {
    const receipts = await verifyReceiptLog(this.dir);
    const gates = evaluateGates({
      hostRamBytes: this.hostRamBytes,
      agent: this.result ? { toolCalls: this.result.toolCalls, invalidReplies: this.result.invalidReplies, violations: this.result.violations, commandApproval: this.cfg.agent.autoApproveProjectCommands ? "never-asked" : "project-code" } : { toolCalls: 0, invalidReplies: 0, violations: this.ws.violations, commandApproval: "project-code" },
      receiptsOk: receipts.ok ? { ok: true } : { ok: false, detail: `entry ${receipts.index}: ${receipts.reason}` },
    });
    return {
      id: this.id, dir: this.dir, task: this.task, status: this.status, startedAt: this.startedAt, diffs: this.lastDiffs,
      pendingApprovals: [...this.pending].map(([requestId, p]) => ({ requestId, command: p.command, network: p.network })),
      ...(this.result ? { result: this.result } : {}), ...(this.checkpointId ? { checkpointId: this.checkpointId } : {}), gates,
    };
  }
}

export { ConflictError };

/** A project directory the server may work in: it must exist, be a directory, and is stored by its real path (links resolved). */
export async function normaliseRoot(dir: string): Promise<string> {
  const real = await realpath(dir);
  if (!(await stat(real)).isDirectory()) throw new Error(`${dir} is not a directory`);
  if (real === "/" || /^[A-Za-z]:\\?$/.test(real)) throw new Error("the root of a drive cannot be used as a project");
  return real;
}
export const isInside = (root: string, p: string): boolean => p === root || p.startsWith(join(root, "/"));
