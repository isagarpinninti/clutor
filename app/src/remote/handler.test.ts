import { describe, expect, it } from "vitest";
import type { AgentController } from "../agent/controller";
import { newSession, type CodeSession } from "../agent/sessions";
import type { RemoteDevice } from "../bridge";
import { RemoteHandler } from "./handler";
import { parsePhoneMsg, phoneMode, plainFolderName, RateLimiter, wireSession } from "./protocol";

const DEVICE: RemoteDevice = { id: "d1", name: "Pixel", public: "00", created: 1, lastSeen: 1 };
const PROJECTS = "/home/me/Projects";

function setup(opts: { allowed?: string[]; projects?: string | undefined } = {}) {
  type Reply = { t: string; error?: string; ok?: boolean; ref?: string; action?: string; canCreate?: boolean; projectsFolder?: string; model?: { name: string }; sessions: Array<{ id: string }> };
  const sent: Array<{ client: number; v: Reply }> = [];
  const logs: string[] = [];
  const runs: Array<{ id: string; text: string; o: { perm: string; source: string; device?: string } }> = [];
  const answers: Array<[string, boolean, unknown]> = [];
  const calls: string[] = [];
  let sessions: CodeSession[] = [{ ...newSession(`${PROJECTS}/app`, 5), id: "s1", title: "App" }, { ...newSession("/etc/secret", 5), id: "bad", title: "Outside" }];
  const allowedRoots = opts.allowed ?? [];
  const projects = "projects" in opts ? opts.projects : PROJECTS;
  const inside = (root: string): boolean => [...(projects ? [projects] : []), ...allowedRoots].some((a) => root === a || root.startsWith(`${a}/`)) && !root.includes("..");
  const ctl = {
    approvals: () => [], answer: (id: string, yes: boolean, by: unknown) => { answers.push([id, yes, by]); return id !== "gone"; },
    run: async (id: string, text: string, o: { perm: string; source: string; device?: string }) => { runs.push({ id, text, o }); },
    stop: (id: string) => calls.push(`stop ${id}`),
    apply: async (id: string, t: string) => { calls.push(`apply ${id} ${t}`); return { ok: true }; },
    undo: async () => ({ ok: true }), discard: () => ({ ok: true }),
  } as unknown as AgentController;
  let t = 0;
  const h = new RemoteHandler({
    ctl, sessions: () => sessions,
    host: { checkRoot: async (r: string, a: string[]) => { if (a.some((x) => r === x || r.startsWith(`${x}/`)) && !r.includes("..")) return r; throw new Error("that folder is not inside a folder you allowed for remote control"); }, newProject: async (p: string, n: string) => { calls.push(`new ${n}`); return `${p}/${n}`; }, dirs: async () => ["app", "notes"] },
    createSession: (root) => { const s = { ...newSession(root, 9), id: `n${sessions.length}` }; sessions = [...sessions, s]; return s; },
    settings: () => ({ projectsRoot: projects, allowedRoots }), model: () => ({ name: "Llama", loaded: true }),
    send: (client, text) => sent.push({ client, v: JSON.parse(text) }), log: (_d, w) => logs.push(w), now: () => (t += 1),
  });
  h.onOpen(1, DEVICE);
  const msg = (v: unknown) => h.onMessage(1, JSON.stringify(v));
  return { h, msg, sent, logs, runs, answers, calls, inside, setSessions: (s: CodeSession[]) => { sessions = s; } };
}
const last = (s: { sent: Array<{ v: { t: string; error?: string; ok?: boolean; ref?: string; action?: string } }> }) => s.sent[s.sent.length - 1]!.v;

describe("a phone's messages", () => {
  it("gets a snapshot when it connects", () => {
    const s = setup();
    expect(s.sent[0]!.v).toMatchObject({ t: "snapshot", canCreate: true, projectsFolder: "Projects", model: { name: "Llama" } });
    expect(s.sent[0]!.v.sessions.map((x) => x.id)).toContain("s1");
  });

  it("starts a task in an existing session inside an allowed folder, in plan mode unless it asks for ask-first", async () => {
    const s = setup();
    await s.msg({ t: "task", text: "add tests", session: "s1" });
    expect(s.runs[0]).toMatchObject({ id: "s1", text: "add tests", o: { perm: "plan", source: "remote", device: "Pixel" } });
    await s.msg({ t: "task", text: "again", session: "s1", mode: "ask" });
    expect(s.runs[1]!.o.perm).toBe("ask");
  });

  it("refuses auto mode and any unknown mode, and starts nothing", async () => {
    const s = setup();
    for (const mode of ["auto", "AUTO", "yolo", ""]) await s.msg({ t: "task", text: "x", session: "s1", mode });
    expect(s.runs).toEqual([]);
    expect(last(s)).toMatchObject({ t: "error" });
    expect(s.logs.some((l) => l.includes("refused a task"))).toBe(true);
  });

  it("refuses a session whose folder is outside the allowed folders", async () => {
    const s = setup();
    await s.msg({ t: "task", text: "x", session: "bad" });
    await s.msg({ t: "open", session: "bad" });
    await s.msg({ t: "stop", session: "bad" });
    await s.msg({ t: "apply", session: "bad", turn: "t" });
    expect(s.runs).toEqual([]);
    expect(s.calls).toEqual([]);
    expect(s.sent.filter((x) => x.v.t === "error").length).toBe(4);
  });

  it("allows an extra folder you added", async () => {
    const s = setup({ allowed: ["/etc"] });
    await s.msg({ t: "task", text: "x", session: "bad" });
    expect(s.runs.length).toBe(1);
  });

  it("creates a new project inside the projects folder, with a plain name", async () => {
    const s = setup();
    await s.msg({ t: "task", text: "build a todo app", project: { newName: "Todo App" } });
    expect(s.calls).toContain("new Todo App");
    expect(s.runs[0]!.o).toMatchObject({ source: "remote" });
    for (const bad of ["../escape", "a/b", ".hidden", "", "x".repeat(65), "..", "a\\b"]) await s.msg({ t: "task", text: "x", project: { newName: bad } });
    expect(s.calls.filter((c) => c.startsWith("new")).length).toBe(1);
    expect(s.runs.length).toBe(1);
  });

  it("opens an existing project folder by name, and refuses names that walk out", async () => {
    const s = setup();
    await s.msg({ t: "task", text: "x", project: { dir: "app" } });
    expect(s.runs.length).toBe(1);
    await s.msg({ t: "task", text: "x", project: { dir: "../../etc" } });
    await s.msg({ t: "task", text: "x", project: { dir: "a/b" } });
    expect(s.runs.length).toBe(1);
  });

  it("cannot create or open projects when no projects folder was chosen", async () => {
    const s = setup({ projects: undefined });
    await s.msg({ t: "task", text: "x", project: { newName: "Thing" } });
    expect(s.runs).toEqual([]);
    expect(last(s).error).toMatch(/projects folder/);
  });

  it("passes approvals on, recording who answered, and says when it was already answered", async () => {
    const s = setup();
    await s.msg({ t: "approve", id: "a1", yes: true });
    expect(s.answers[0]).toEqual(["a1", true, { source: "remote", device: "Pixel" }]);
    await s.msg({ t: "approve", id: "gone", yes: true });
    expect(last(s)).toMatchObject({ t: "error" });
  });

  it("applies and reports the result for a session inside the folders", async () => {
    const s = setup();
    await s.msg({ t: "apply", session: "s1", turn: "t1", ref: "r1" });
    expect(s.calls).toContain("apply s1 t1");
    expect(last(s)).toMatchObject({ t: "result", action: "apply", ok: true, ref: "r1" });
  });

  it("slows down a flood of requests", async () => {
    const s = setup();
    for (let i = 0; i < 40; i++) await s.msg({ t: "ping" });
    expect(s.sent.filter((x) => x.v.t === "pong").length).toBe(20);
    expect(s.sent.some((x) => x.v.error?.includes("slow down"))).toBe(true);
  });

  it("rejects garbage, oversized tasks and unknown messages, and ignores unknown clients", async () => {
    const s = setup();
    await s.h.onMessage(1, "not json");
    await s.msg({ t: "task", text: "x".repeat(9000), session: "s1" });
    await s.msg({ t: "rm -rf" });
    expect(s.runs).toEqual([]);
    const n = s.sent.length;
    await s.h.onMessage(99, JSON.stringify({ t: "ping" }));
    expect(s.sent.length).toBe(n);
  });

  it("logs the connection", () => {
    expect(setup().logs).toContain("connected");
  });
});

describe("protocol helpers", () => {
  it("parses strictly", () => {
    expect(parsePhoneMsg('{"t":"approve","id":"a","yes":"true"}').ok).toBe(false);
    expect(parsePhoneMsg("[]").ok).toBe(false);
    expect(parsePhoneMsg('{"t":"open"}').ok).toBe(false);
    expect(parsePhoneMsg('{"t":"state"}')).toEqual({ ok: true, msg: { t: "state" } });
  });
  it("only grants ask and plan", () => {
    expect([phoneMode(undefined), phoneMode("ask"), phoneMode("plan"), phoneMode("auto")]).toEqual(["plan", "ask", "plan", undefined]);
  });
  it("judges folder names", () => {
    expect(["My App", "app-1", "a.b"].every(plainFolderName)).toBe(true);
    expect(["", ".x", "a/b", "../x", "a..b", " x"].some(plainFolderName)).toBe(false);
  });
  it("limits the rate", () => {
    let t = 0;
    const r = new RateLimiter(2, 100, () => t);
    expect([r.allow(), r.allow(), r.allow()]).toEqual([true, true, false]);
    t = 200;
    expect(r.allow()).toBe(true);
  });
  it("sends sessions without the files' contents and with long output cut", () => {
    const s = newSession("/p", 1);
    s.turns.push({ id: "t", task: "x", created: 1, status: "done", blocks: [{ id: "b", kind: "tool", name: "read_file", label: "Read a", status: "ok", output: "y".repeat(2000) }], changes: [{ path: "a", before: "secret", after: "new", baseHash: null }], diffs: [{ path: "a", status: "modified", additions: 1, deletions: 0, patch: "@@" }] });
    const w = JSON.stringify(wireSession(s));
    expect(w).not.toContain("secret");
    expect(w).toContain("[cut off]");
    expect(w).toContain('"patch":"@@"');
  });
});
