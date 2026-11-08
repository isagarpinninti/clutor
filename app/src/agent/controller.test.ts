import { describe, expect, it } from "vitest";
import type { StreamEvent } from "@engine/stream";
import type { AgentHost } from "../bridge";
import { sha256Hex } from "@core/agent/workspace";
import { AgentController, type Approval, type Deps } from "./controller";
import { newSession, type CodeSession } from "./sessions";

function scripted(replies: string[], gate?: Promise<void>) {
  let i = 0;
  return { async *stream(): AsyncGenerator<StreamEvent> { if (gate) await gate; yield { type: "content", text: replies[i++] ?? '{"tool":"finish","args":{"summary":"done"}}' }; yield { type: "done", finish: "stop" }; } };
}

function setup(replies: string[], files: Record<string, string> = { "a.txt": "old\n" }, gate?: Promise<void>) {
  const fs = new Map(Object.entries(files));
  const host = {
    list: async () => [...fs.keys()].filter((k) => !k.startsWith(".clutor")).sort(),
    read: async (_r: string, p: string) => fs.get(p) ?? null,
    write: async (_r: string, p: string, c: string | null) => { if (c === null) fs.delete(p); else fs.set(p, c); },
    run: async () => ({ exit_code: 0, output: "ok", timed_out: false, duration_ms: 1 }),
  } as unknown as AgentHost;
  let session: CodeSession = newSession("/p");
  let approvals: Approval[] = [];
  const log: string[] = [];
  let n = 0;
  const deps: Deps = {
    host, web: {} as never, getSession: (id) => (id === session.id ? session : undefined),
    change: (_id, f) => { session = f(session); },
    engine: async () => scripted(replies, gate), hold: async () => () => undefined, mcp: () => undefined,
    config: () => ({ maxSteps: 10, contextTokens: 4096, modelName: "scripted" }), onUsage: () => undefined,
    onApprovals: (l) => { approvals = l; }, log: (e) => log.push(e.what), id: () => `id${++n}`, titleFrom: (t) => t.slice(0, 20),
  };
  return { ctl: new AgentController(deps), sess: () => session, approvals: () => approvals, fs, log };
}

const EDIT = ['{"tool":"edit_file","args":{"path":"a.txt","old_text":"old","new_text":"new"}}', '{"tool":"finish","args":{"summary":"changed"}}'];
const wait = async (f: () => boolean) => { for (let i = 0; i < 300; i++) { if (f()) return; await new Promise((r) => setTimeout(r, 5)); } throw new Error("timed out"); };

describe("AgentController", () => {
  it("writes a local task's files to the folder when it ends, and undo restores them", async () => {
    const t = setup(EDIT);
    await t.ctl.run(t.sess().id, "make it new", { perm: "ask", source: "local" });
    const turn = t.sess().turns[0]!;
    expect(turn.status).toBe("done");
    expect(turn.diffs?.map((d) => d.path)).toEqual(["a.txt"]);
    expect(turn.applied).toBeTruthy();
    expect(t.fs.get("a.txt")).toBe("new\n");
    expect(await t.ctl.undo(t.sess().id, turn.id)).toEqual({ ok: true });
    expect(t.fs.get("a.txt")).toBe("old\n");
  });

  it("holds a task started from another device for review until it is applied here", async () => {
    const t = setup(EDIT);
    await t.ctl.run(t.sess().id, "make it new", { perm: "ask", source: "remote", device: "Pixel" });
    const turn = t.sess().turns[0]!;
    expect(turn.applied).toBeUndefined();
    expect(t.fs.get("a.txt")).toBe("old\n");
    expect(await t.ctl.apply(t.sess().id, turn.id)).toEqual({ ok: true });
    expect(t.fs.get("a.txt")).toBe("new\n");
  });

  it("writes nothing when the file changed meanwhile, and says so", async () => {
    const t = setup(EDIT);
    const changed = t.fs;
    const p = t.ctl.run(t.sess().id, "x", { perm: "ask", source: "remote" });
    await p;
    changed.set("a.txt", "edited by you\n");
    const r = await t.ctl.apply(t.sess().id, t.sess().turns[0]!.id);
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/changed since/);
    expect(t.fs.get("a.txt")).toBe("edited by you\n");
    expect(await sha256Hex("x")).toBeTruthy();
  });

  it("waits for approval before running a command, and the answer decides", async () => {
    const t = setup(['{"tool":"run_command","args":{"command":"npm test"}}', '{"tool":"finish","args":{"summary":"ran"}}'], { "package.json": '{"scripts":{"test":"echo"}}' });
    const done = t.ctl.run(t.sess().id, "run tests", { perm: "ask", source: "remote", device: "Pixel" });
    await wait(() => t.approvals().length === 1);
    expect(t.approvals()[0]).toMatchObject({ command: "npm test", source: "remote", device: "Pixel" });
    expect(t.ctl.answer(t.approvals()[0]!.id, true, { source: "remote", device: "Pixel" })).toBe(true);
    await done;
    expect(t.sess().turns[0]!.blocks.some((b) => b.kind === "tool" && b.name === "run_command" && b.status === "ok")).toBe(true);
    expect(t.approvals()).toEqual([]);
    expect(t.ctl.answer("gone", true)).toBe(false);
  });

  it("never auto-approves for a task started remotely, even in auto mode, but does for one started here", async () => {
    const cmd = ['{"tool":"run_command","args":{"command":"npm test"}}', '{"tool":"finish","args":{"summary":"ran"}}'];
    const files = { "package.json": '{"scripts":{"test":"echo"}}' };
    const remote = setup(cmd, files);
    const p = remote.ctl.run(remote.sess().id, "t", { perm: "auto", source: "remote" });
    await wait(() => remote.approvals().length === 1);
    remote.ctl.answer(remote.approvals()[0]!.id, false);
    await p;
    const local = setup(cmd, files);
    await local.ctl.run(local.sess().id, "t", { perm: "auto", source: "local" });
    expect(local.approvals()).toEqual([]);
    expect(local.sess().turns[0]!.blocks.some((b) => b.kind === "tool" && b.name === "run_command" && b.status === "ok")).toBe(true);
  });

  it("plan mode refuses commands and keeps no changes", async () => {
    const t = setup(['{"tool":"edit_file","args":{"path":"a.txt","old_text":"old","new_text":"new"}}', '{"tool":"run_command","args":{"command":"npm test"}}', '{"tool":"finish","args":{"summary":"plan"}}'], { "a.txt": "old\n", "package.json": "{}" });
    await t.ctl.run(t.sess().id, "plan it", { perm: "plan", source: "remote" });
    const turn = t.sess().turns[0]!;
    expect(turn.changes).toBeUndefined();
    expect(turn.diffs).toBeUndefined();
    expect(t.approvals()).toEqual([]);
  });

  it("stops a running task, releases its approvals, and allows only one task per session", async () => {
    let open!: () => void;
    const gate = new Promise<void>((r) => { open = r; });
    const t = setup(EDIT, undefined, gate);
    const p = t.ctl.run(t.sess().id, "slow", { perm: "ask", source: "local" });
    await wait(() => t.ctl.isRunning(t.sess().id));
    await expect(t.ctl.run(t.sess().id, "second", { perm: "ask", source: "local" })).rejects.toThrow(/already running/);
    t.ctl.stop(t.sess().id);
    open();
    await p;
    expect(t.sess().turns[0]!.status).toBe("stopped");
    expect(t.ctl.isRunning(t.sess().id)).toBe(false);
  });

  it("logs what a phone did", async () => {
    const t = setup(EDIT);
    await t.ctl.run(t.sess().id, "log me", { perm: "plan", source: "remote", device: "Pixel" });
    expect(t.log.some((l) => l.startsWith("task started (plan)"))).toBe(true);
    expect(t.log.some((l) => l.startsWith("task done"))).toBe(true);
  });

  it("rejects an empty task and an unknown session", async () => {
    const t = setup(EDIT);
    await expect(t.ctl.run(t.sess().id, "   ", { perm: "ask", source: "local" })).rejects.toThrow(/empty/);
    await expect(t.ctl.run("nope", "x", { perm: "ask", source: "local" })).rejects.toThrow(/no longer exists/);
  });
});

describe("allow for this session", () => {
  it("runs later commands in the session without asking, but never one that uses the network", async () => {
    const t = setup(['{"tool":"run_command","args":{"command":"npm test"}}', '{"tool":"finish","args":{"summary":"ran"}}'], { "package.json": '{"scripts":{"test":"echo"}}' });
    t.ctl.allowSession(t.sess().id);
    await t.ctl.run(t.sess().id, "run tests", { perm: "ask", source: "local" });
    expect(t.approvals().length).toBe(0);
    expect(t.sess().turns[0]!.blocks.some((b) => b.kind === "tool" && b.name === "run_command" && b.status === "ok")).toBe(true);
  });
});
