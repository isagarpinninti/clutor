import type { ChatMessage, StreamEvent } from "@engine/stream";
import type { SampleOptions } from "@engine/session";

/**
 * Background jobs: prompts sent to the loaded model that run without blocking the chat. A very large model streamed from disk answers in minutes, not seconds, so the
 * app lets you hand it a question and come back to the answer. Jobs persist (a closed app loses nothing but the time it was closed), run one at a time on the single
 * engine, and a paused job resumes from the text it had already written.
 */

export type JobStatus = "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  title: string;
  conversationId: string;
  /** The assistant message this job writes into. */
  messageId: string;
  /** What to send the model: the system prompt and the conversation up to and including your question. */
  messages: ChatMessage[];
  thinking: boolean;
  temperature: number;
  maxTokens: number;
  status: JobStatus;
  /** The visible reply so far. */
  content: string;
  reasoning: string;
  tokens: number;
  /** Time spent generating, summed over runs (a paused job does not count its pause). */
  elapsedMs: number;
  /** Measured generation speed of the latest run. */
  tps?: number;
  created: number;
  started?: number;
  finished?: number;
  error?: string;
  /** Why a paused job is paused: you asked, or the app was closed while it ran, or a chat message took the engine. */
  pausedBy?: "user" | "interrupted" | "chat";
}

/** The part of the engine session the queue needs; the real one is EngineSession. */
export interface JobSession { stream(messages: ChatMessage[], o?: SampleOptions): AsyncGenerator<StreamEvent> }

export interface JobStore { load(): Job[]; save(jobs: Job[]): void }

export const ACTIVE: ReadonlyArray<JobStatus> = ["queued", "running", "paused"];
export const isActive = (j: Job): boolean => ACTIVE.includes(j.status);

export function uid(): string { return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`; }

/** The jobs as they were saved: a job that was running when the app closed comes back paused ("interrupted") with its partial reply, never silently lost or restarted. */
export function recover(saved: Job[]): Job[] {
  return saved.map((j) => (j.status === "running" ? { ...j, status: "paused" as const, pausedBy: "interrupted" as const } : j));
}

/** Time left, as an upper bound: the tokens the job may still write at the speed measured so far. Undefined until a speed is known. */
export function etaMs(j: Job): number | undefined {
  if (!j.tps || j.tps <= 0 || j.status !== "running") return undefined;
  return Math.max(0, ((j.maxTokens - j.tokens) / j.tps) * 1000);
}

type Listener = () => void;

export class JobQueue {
  private jobs: Job[];
  private session: JobSession | null = null;
  private current: { job: Job; ac: AbortController; reason: "user" | "chat" | "cancel" | null } | null = null;
  private listeners = new Set<Listener>();
  private lastSave = 0;
  private holds = 0;
  private pumping = false;

  constructor(private readonly store: JobStore, private readonly hooks: { onUpdate?: (j: Job) => void; onFinish?: (j: Job) => void } = {}) {
    this.jobs = recover(store.load());
    this.persist(true);
  }

  subscribe(f: Listener): () => void { this.listeners.add(f); return () => { this.listeners.delete(f); }; }
  list(): Job[] { return this.jobs; }
  get(id: string): Job | undefined { return this.jobs.find((j) => j.id === id); }
  get running(): Job | undefined { return this.current?.job; }
  /** Position in line (1 = next), or 0 when the job is not waiting. */
  position(id: string): number {
    const waiting = this.jobs.filter((j) => j.status === "queued");
    const i = waiting.findIndex((j) => j.id === id);
    return i < 0 ? 0 : i + 1 + (this.current ? 1 : 0);
  }

  /** The engine the jobs run on: set when a model is loaded, null when it is closed. */
  attach(session: JobSession | null): void {
    if (this.session !== session && this.current) this.current.ac.abort();   // the model went away under a running job
    this.session = session;
    this.emit();
    void this.pump();
  }

  enqueue(j: Omit<Job, "status" | "content" | "reasoning" | "tokens" | "elapsedMs" | "created"> & Partial<Pick<Job, "created">>): Job {
    const job: Job = { ...j, status: "queued", content: "", reasoning: "", tokens: 0, elapsedMs: 0, created: j.created ?? Date.now() };
    this.jobs = [...this.jobs, job];
    this.changed(job, true);
    void this.pump();
    return job;
  }

  cancel(id: string): void {
    const job = this.get(id);
    if (!job || !isActive(job)) return;
    if (this.current?.job.id === id) { this.current.reason = "cancel"; this.current.ac.abort(); return; }
    this.update(job, { status: "cancelled", finished: Date.now() }, true);
    this.hooks.onFinish?.(job);
  }

  pause(id: string): void {
    const job = this.get(id);
    if (!job) return;
    if (this.current?.job.id === id) { this.current.reason = "user"; this.current.ac.abort(); return; }
    if (job.status === "queued") this.update(job, { status: "paused", pausedBy: "user" }, true);
  }

  resume(id: string): void {
    const job = this.get(id);
    if (!job || job.status !== "paused") return;
    const { pausedBy: _pausedBy, ...rest } = job;
    void _pausedBy;
    this.replace({ ...rest, status: "queued" });
    void this.pump();
  }

  /** Runs a failed or cancelled job again from the start. */
  retry(id: string): void {
    const job = this.get(id);
    if (!job || (job.status !== "failed" && job.status !== "cancelled")) return;
    const { error: _e, finished: _f, pausedBy: _p, ...rest } = job;
    void _e; void _f; void _p;
    this.replace({ ...rest, status: "queued", content: "", reasoning: "", tokens: 0, elapsedMs: 0 });
    void this.pump();
  }

  remove(id: string): void {
    const job = this.get(id);
    if (!job) return;
    if (this.current?.job.id === id) { this.cancel(id); return; }
    this.jobs = this.jobs.filter((j) => j.id !== id);
    this.persist(true);
    this.emit();
  }

  /**
   * A chat message needs the engine now. Pauses the running job (it keeps what it wrote), and holds the queue until the returned function is called, which lets the
   * paused job and the rest continue. Resolves once the engine is free.
   */
  async holdForChat(): Promise<() => void> {
    this.holds++;
    const cur = this.current;
    if (cur) {
      cur.reason = "chat";
      cur.ac.abort();
      await this.idle();
    }
    let released = false;
    return () => { if (released) return; released = true; this.holds--; void this.pump(); };
  }

  private idle(): Promise<void> {
    return new Promise((resolve) => { const check = (): void => { if (!this.current) resolve(); else setTimeout(check, 20); }; check(); });
  }

  private emit(): void { for (const l of this.listeners) l(); }
  private changed(job: Job, save: boolean): void { this.persist(save); this.hooks.onUpdate?.(job); this.emit(); }
  private update(job: Job, patch: Partial<Job>, save: boolean): void { Object.assign(job, patch); this.jobs = [...this.jobs]; this.changed(job, save); }
  private replace(next: Job): void { this.jobs = this.jobs.map((j) => (j.id === next.id ? next : j)); this.changed(next, true); }

  private persist(force: boolean): void {
    const now = Date.now();
    if (!force && now - this.lastSave < 1500) return;
    this.lastSave = now;
    try { this.store.save(this.jobs); } catch { /* a full disk loses progress, not the app */ }
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (this.session && this.holds === 0 && !this.current) {
        const next = this.jobs.find((j) => j.status === "queued");
        if (!next) break;
        await this.run(next);
      }
    } finally { this.pumping = false; }
  }

  private async run(job: Job): Promise<void> {
    const session = this.session;
    if (!session) return;
    const ac = new AbortController();
    this.current = { job, ac, reason: null };
    this.update(job, { status: "running", started: job.started ?? Date.now() }, true);
    const t0 = Date.now(), base = job.elapsedMs;
    let firstAt = 0, produced = 0;
    try {
      const budget = Math.max(1, job.maxTokens - job.tokens);
      // A job paused with visible text continues after it. With thinking on, the reasoning is not saved, so it starts the reply again.
      const resumeFrom = job.thinking ? "" : job.content;
      if (job.thinking) { job.content = ""; job.reasoning = ""; job.tokens = 0; }
      const baseTokens = job.tokens;
      for await (const ev of session.stream(job.messages, { thinking: job.thinking, temp: job.temperature, max: budget, signal: ac.signal, ...(resumeFrom ? { prefill: resumeFrom } : {}) })) {
        if (ev.type === "content" || ev.type === "reasoning") {
          if (!firstAt) firstAt = Date.now();
          produced++;
          if (ev.type === "content") job.content += ev.text; else job.reasoning += ev.text;
          job.tokens = baseTokens + produced;
          const secs = (Date.now() - firstAt) / 1000;
          this.update(job, { elapsedMs: base + (Date.now() - t0), ...(secs > 0.5 ? { tps: produced / secs } : {}) }, false);
        } else if (ev.type === "done") {
          if (ev.tokensPerSecond) job.tps = ev.tokensPerSecond;
          if (ev.completionTokens !== undefined) job.tokens = baseTokens + ev.completionTokens;
          if (ev.finish === "cancel") throw new DOMException("stopped", "AbortError");
        }
      }
      this.update(job, { status: "done", finished: Date.now(), elapsedMs: base + (Date.now() - t0) }, true);
      this.hooks.onFinish?.(job);
    } catch (e) {
      const reason = this.current?.reason ?? null;
      const spent = base + (Date.now() - t0);
      if (reason === "cancel") { this.update(job, { status: "cancelled", finished: Date.now(), elapsedMs: spent }, true); this.hooks.onFinish?.(job); }
      else if (reason === "user" || reason === "chat") this.update(job, { status: "paused", pausedBy: reason, elapsedMs: spent }, true);
      else if (ac.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) this.update(job, { status: "paused", pausedBy: "interrupted", elapsedMs: spent }, true);   // the model was closed under it
      else { this.update(job, { status: "failed", finished: Date.now(), elapsedMs: spent, error: e instanceof Error ? e.message : String(e) }, true); this.hooks.onFinish?.(job); }
    } finally {
      this.current = null;
    }
    // A paused-for-chat job goes back in line once the chat is done.
    if (job.status === "paused" && job.pausedBy === "chat") {
      const { pausedBy: _p, ...rest } = job;
      void _p;
      this.replace({ ...rest, status: "queued" });
    }
  }
}

const KEY = "clutor.jobs.v1";

/** Jobs kept in the app's own storage (the web view's local storage, which the app keeps between launches). */
export const localJobStore: JobStore = {
  load() { try { const raw = localStorage.getItem(KEY); return raw ? (JSON.parse(raw) as Job[]) : []; } catch { return []; } },
  save(jobs) { try { localStorage.setItem(KEY, JSON.stringify(jobs.slice(-100))); } catch { /* storage full */ } },
};
