import type { AgentController, Approval } from "../agent/controller";
import type { CodeSession } from "../agent/sessions";
import type { AgentHost, RemoteDevice } from "../bridge";
import { parsePhoneMsg, phoneMode, plainFolderName, RateLimiter, summary, wireSession } from "./protocol";

/**
 * Receives what paired phones send and does it, after checking each request against the rules in docs/REMOTE.md: a phone works only inside the projects folder and the folders
 * you allowed (checked on the real path, links followed), can ask for "ask first" or "plan only" and never "auto-run", and every request is logged. It also keeps each phone
 * up to date with the sessions it is watching.
 */

export interface HandlerDeps {
  ctl: AgentController;
  host: Pick<AgentHost, "checkRoot" | "newProject" | "dirs">;
  sessions(): CodeSession[];
  createSession(root: string): CodeSession;
  settings(): { projectsRoot?: string | undefined; allowedRoots: string[] };
  model(): { name: string; loaded: boolean } | undefined;
  send(client: number, text: string): void;
  log(device: string, what: string): void;
  now?: () => number;
}

interface Client { device: RemoteDevice; watching?: string; lastSession?: number; lastList?: string; limiter: RateLimiter }

export class RemoteHandler {
  private clients = new Map<number, Client>();
  constructor(private readonly d: HandlerDeps) {}

  private reply(client: number, v: unknown): void { this.d.send(client, JSON.stringify(v)); }
  private fail(client: number, error: string, ref?: string): void { this.reply(client, { t: "error", error, ...(ref ? { ref } : {}) }); }
  private allowed(): string[] { const s = this.d.settings(); return [...(s.projectsRoot ? [s.projectsRoot] : []), ...s.allowedRoots]; }

  onOpen(client: number, device: RemoteDevice): void {
    this.clients.set(client, { device, limiter: new RateLimiter(20, 10_000, this.d.now) });
    this.d.log(device.name, "connected");
    this.snapshot(client);
  }
  onClose(client: number): void { const c = this.clients.get(client); if (c) this.d.log(c.device.name, "disconnected"); this.clients.delete(client); }

  private snapshot(client: number): void {
    const c = this.clients.get(client);
    if (!c) return;
    const sessions = [...this.d.sessions()].sort((a, b) => b.updated - a.updated).slice(0, 50).map(summary);
    c.lastList = JSON.stringify(sessions);
    const s = this.d.settings();
    this.reply(client, { t: "snapshot", sessions, approvals: this.d.ctl.approvals().map(wireApproval), model: this.d.model() ?? null, projectsFolder: s.projectsRoot ? s.projectsRoot.split(/[\\/]/).filter(Boolean).pop() : null, canCreate: !!s.projectsRoot });
  }

  /** Call when sessions or approvals changed: each phone gets the update for what it is watching. */
  onChange(): void {
    for (const [id, c] of this.clients) {
      const list = JSON.stringify([...this.d.sessions()].sort((a, b) => b.updated - a.updated).slice(0, 50).map(summary));
      if (list !== c.lastList) this.snapshot(id);
      const s = c.watching ? this.d.sessions().find((x) => x.id === c.watching) : undefined;
      if (s && s.updated !== c.lastSession) { c.lastSession = s.updated; this.reply(id, { t: "session", session: wireSession(s) }); }
      this.reply(id, { t: "approvals", approvals: this.d.ctl.approvals().map(wireApproval) });
    }
  }

  /** The session if its folder is, as really resolved, inside an allowed folder. */
  private async guarded(client: number, sessionId: string, ref?: string): Promise<CodeSession | undefined> {
    const s = this.d.sessions().find((x) => x.id === sessionId);
    if (!s) { this.fail(client, "that session does not exist", ref); return undefined; }
    try { await this.d.host.checkRoot(s.root, this.allowed()); return s; } catch (e) { this.fail(client, e instanceof Error ? e.message : String(e), ref); return undefined; }
  }

  async onMessage(client: number, text: string): Promise<void> {
    const c = this.clients.get(client);
    if (!c) return;
    const who = c.device.name;
    if (!c.limiter.allow()) { this.fail(client, "too many requests: slow down"); return; }
    const parsed = parsePhoneMsg(text);
    if (!parsed.ok) { this.fail(client, parsed.error); return; }
    const m = parsed.msg;
    const ref = "ref" in m ? m.ref : undefined;
    try {
      switch (m.t) {
        case "ping": this.reply(client, { t: "pong" }); return;
        case "state": this.snapshot(client); return;
        case "projects": {
          const root = this.d.settings().projectsRoot;
          if (!root) { this.reply(client, { t: "projects", dirs: [], canCreate: false }); return; }
          this.reply(client, { t: "projects", dirs: await this.d.host.dirs(root), canCreate: true });
          return;
        }
        case "open": {
          const s = await this.guarded(client, m.session);
          if (!s) return;
          c.watching = s.id; c.lastSession = s.updated;
          this.reply(client, { t: "session", session: wireSession(s) });
          return;
        }
        case "task": {
          const mode = phoneMode(m.mode);
          if (!mode) { this.d.log(who, `refused a task in mode "${m.mode}"`); this.fail(client, "a phone can only start tasks in \"ask first\" or \"plan only\" mode", ref); return; }
          let session: CodeSession | undefined;
          if (m.session) session = await this.guarded(client, m.session, ref);
          else {
            const projects = this.d.settings().projectsRoot;
            if (!projects) { this.fail(client, "choose a projects folder on the computer first (Settings > Remote)", ref); return; }
            let root: string;
            if (m.project?.newName !== undefined) {
              if (!plainFolderName(m.project.newName.trim())) { this.fail(client, "a project name can use letters, numbers, spaces, - _ and . (up to 64 characters)", ref); return; }
              root = await this.d.host.newProject(projects, m.project.newName.trim());
              this.d.log(who, `created project ${m.project.newName.trim()}`);
            } else if (m.project?.dir !== undefined) {
              if (!plainFolderName(m.project.dir)) { this.fail(client, "that is not a project folder name", ref); return; }
              root = await this.d.host.checkRoot(`${projects.replace(/[\\/]+$/, "")}/${m.project.dir}`, this.allowed());
            } else { this.fail(client, "say which session or project the task is for", ref); return; }
            session = this.d.createSession(root);
          }
          if (!session) return;
          c.watching = session.id; c.lastSession = undefined;
          this.reply(client, { t: "started", session: session.id, mode, ...(ref ? { ref } : {}) });
          void this.d.ctl.run(session.id, m.text, { perm: mode, source: "remote", device: who }).catch((e: unknown) => this.fail(client, e instanceof Error ? e.message : String(e), ref));
          this.onChange();
          return;
        }
        case "approve": {
          const ok = this.d.ctl.answer(m.id, m.yes, { source: "remote", device: who });
          if (!ok) this.fail(client, "that request was already answered or the task ended");
          return;
        }
        case "stop": {
          const s = await this.guarded(client, m.session);
          if (s) { this.d.ctl.stop(s.id); this.d.log(who, "stopped a task"); }
          return;
        }
        case "apply": case "discard": case "undo": {
          const s = await this.guarded(client, m.session, ref);
          if (!s) return;
          const by = { device: who };
          const r = m.t === "apply" ? await this.d.ctl.apply(s.id, m.turn, by) : m.t === "undo" ? await this.d.ctl.undo(s.id, m.turn, by) : this.d.ctl.discard(s.id, m.turn, by);
          this.reply(client, { t: "result", action: m.t, ok: r.ok, ...(r.error ? { error: r.error } : {}), ...(ref ? { ref } : {}) });
          return;
        }
      }
    } catch (e) {
      this.fail(client, e instanceof Error ? e.message : String(e), ref);
    }
  }
}

const wireApproval = (a: Approval): unknown => ({ id: a.id, session: a.sessionId, command: a.command, network: a.network, source: a.source, device: a.device, at: a.at });
