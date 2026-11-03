import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@core/agent/loop";
import { applyEvent, groupByProject, newSession, reviewable, sessionDot, toolLabel, toSaved, type CodeSession, type Turn } from "./sessions";

const turn = (over: Partial<Turn> = {}): Turn => ({ id: "t", task: "x", created: 1, status: "done", blocks: [], ...over });
const sess = (over: Partial<CodeSession> = {}): CodeSession => ({ ...newSession("/p/app", 1), ...over });
const run = (events: AgentEvent[]) => events.reduce((b, e) => applyEvent(b, e), [] as ReturnType<typeof applyEvent>);

describe("applyEvent", () => {
  it("turns a call and its result into one block, and attaches the diff to an edit", () => {
    const blocks = run([
      { type: "tool_call", step: 1, name: "read_file", args: { path: "a.js" } },
      { type: "tool_result", step: 1, name: "read_file", ok: true, output: "1| x" },
      { type: "tool_call", step: 2, name: "edit_file", args: { path: "a.js", old_text: "x", new_text: "y" } },
      { type: "tool_result", step: 2, name: "edit_file", ok: true, output: "edited" },
      { type: "diff", diffs: [{ path: "a.js", status: "modified", additions: 1, deletions: 1, patch: "@@\n-x\n+y" }] },
      { type: "tool_call", step: 3, name: "finish", args: { summary: "done" } },
    ]);
    expect(blocks.map((b) => (b.kind === "tool" ? [b.label, b.status] : b.kind))).toEqual([["Read a.js", "ok"], ["Edited a.js", "ok"]]);
    expect(blocks[1]).toMatchObject({ patch: "@@\n-x\n+y" });
  });
  it("marks a failed call and keeps prose and invalid replies as their own blocks", () => {
    const blocks = run([
      { type: "thought", step: 1, text: "Let me look" },
      { type: "tool_call", step: 1, name: "run_command", args: { command: "npm test" } },
      { type: "tool_result", step: 1, name: "run_command", ok: false, output: "exit code 1" },
      { type: "invalid", step: 2, error: "unknown tool" },
    ]);
    expect(blocks.map((b) => b.kind)).toEqual(["text", "tool", "note"]);
    expect(blocks[1]).toMatchObject({ status: "error", label: "Ran npm test" });
  });
});

describe("labels and state", () => {
  it("labels tools in plain words", () => {
    expect(toolLabel("search", { pattern: "foo" }).label).toBe("Searched for “foo”");
    expect(toolLabel("use_tool", { server: "s", tool: "t" }).label).toBe("Used s/t");
  });
  it("derives the sidebar dot", () => {
    expect(sessionDot(sess())).toBe("idle");
    expect(sessionDot(sess({ turns: [turn({ status: "running" })] }))).toBe("running");
    expect(sessionDot(sess({ turns: [turn({ changes: [{ path: "a", before: null, after: "x", baseHash: null }] })] }))).toBe("review");
    expect(sessionDot(sess({ turns: [turn({ changes: [{ path: "a", before: null, after: "x", baseHash: null }], applied: "c1" })] }))).toBe("done");
    expect(sessionDot(sess({ turns: [turn({ status: "failed" })] }))).toBe("failed");
    expect(reviewable(turn({ changes: [], }))).toBe(false);
  });
  it("groups sessions by project, newest first", () => {
    const g = groupByProject([sess({ id: "1", root: "/a/one", updated: 1 }), sess({ id: "2", root: "/b/two", updated: 5 }), sess({ id: "3", root: "/a/one", updated: 9 })]);
    expect(g.map((x) => [x.name, x.sessions.map((s) => s.id)])).toEqual([["one", ["3", "1"]], ["two", ["2"]]]);
  });
});

describe("saving", () => {
  it("brings a running turn back as interrupted and drops changes that are too large", () => {
    const big = "x".repeat(700_000);
    const saved = toSaved([sess({ turns: [turn({ status: "running" }), turn({ changes: [{ path: "a", before: null, after: big, baseHash: null }] })] })]);
    expect(saved[0]?.turns[0]?.status).toBe("interrupted");
    expect(saved[0]?.turns[1]?.changes).toBeUndefined();
    expect(saved[0]?.turns[1]?.lost).toBe(true);
  });
});
