import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@core/agent/loop";
import { applyEvent, applyTurnEvent, sessionMemory, completePlan, tickPlan, groupBlocks, summarizeTools, currentStage, groupByProject, parsePlan, newSession, reviewable, sessionDot, toolLabel, toSaved, type CodeSession, type Turn } from "./sessions";

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

describe("the plan and the stages of a task", () => {
  it("reads a plan with done, active and to-do steps, and lines with no marker", () => {
    expect(parsePlan("[x] Read the code\n[>] Write index.html\n[ ] Style it\n- [ ] Check it\n2. Ship it\n\n")).toEqual([
      { text: "Read the code", state: "done" }, { text: "Write index.html", state: "active" }, { text: "Style it", state: "todo" }, { text: "Check it", state: "todo" }, { text: "Ship it", state: "todo" },
    ]);
  });
  it("keeps the plan on the turn when the agent calls update_plan, and shows no block for it", () => {
    const e = { type: "tool_call", step: 1, name: "update_plan", args: { steps: "[>] One\n[ ] Two" } } as AgentEvent;
    const t = applyTurnEvent(turn({ status: "running" }), e);
    expect(t.plan?.map((s) => s.state)).toEqual(["active", "todo"]);
    expect(t.blocks).toEqual([]);
  });
  it("works out the stage from what the agent is doing when there is no plan", () => {
    const run = (name: string, args: Record<string, unknown>, status: "running" | "ok" = "running"): Turn => turn({ status: "running", blocks: [{ id: "b", kind: "tool", name, label: `${name === "read_file" ? "Read" : "Wrote"} ${String(args["path"])}`, status }] });
    expect(currentStage(turn({ status: "running" })).stage).toBe("planning");
    expect(currentStage(run("read_file", { path: "a.ts" }))).toMatchObject({ stage: "exploring", label: "Reading a.ts" });
    expect(currentStage(run("write_file", { path: "b.ts" }))).toMatchObject({ stage: "writing", label: "Writing b.ts" });
    expect(currentStage(run("write_file", { path: "b.ts" }, "ok")).seen).toEqual(["writing"]);
  });
});

describe("plan ticking", () => {
  const plan = parsePlan("[>] Create index.html\n[ ] Create style.css\n[ ] Check it works");
  it("marks the step that names a written file as done and moves on to the next", () => {
    const out = tickPlan(plan, "style.css");
    expect(out.map((s) => s.state)).toEqual(["active", "done", "todo"]);
    const both = tickPlan(out, "index.html");
    expect(both.map((s) => s.state)).toEqual(["done", "done", "active"]);
  });
  it("marks every step done when the task finishes", () => {
    expect(completePlan(plan)?.every((s) => s.state === "done")).toBe(true);
  });
});

describe("grouping steps", () => {
  const tool = (id: string, name: string, status: "ok" | "running" = "ok") => ({ id, kind: "tool" as const, name, label: name, status });
  it("joins steps that follow each other and keeps text between them apart", () => {
    const items = groupBlocks([tool("1", "read_file"), tool("2", "read_file"), { id: "t", kind: "text", text: "hi" }, tool("3", "run_command")]);
    expect(items.map((i) => i.kind)).toEqual(["group", "text", "group"]);
    const first = items[0];
    expect(first && first.kind === "group" ? first.tools.length : 0).toBe(2);
  });
  it("says what a run of steps did", () => {
    expect(summarizeTools([tool("1", "read_file"), tool("2", "read_file"), tool("3", "write_file"), tool("4", "run_command"), tool("5", "run_command"), tool("6", "run_command")])).toBe("Read 2 files, wrote 1 file, ran 3 commands");
  });
});

describe("session memory", () => {
  const turn = (task: string, summary: string, files: string[], status: "done" | "running" = "done"): Turn => ({ id: task, task, created: 0, status, blocks: [], summary, diffs: files.map((path) => ({ path, status: "added" as const, additions: 1, deletions: 0, patch: "" })) });
  it("tells the next task what earlier tasks did and which files they made", () => {
    const s: CodeSession = { id: "s", title: "t", root: "/p", created: 0, updated: 0, turns: [turn("make a game", "Made a tic tac toe.", ["index.html", "style.css"]), turn("now", "", [], "running")] };
    const m = sessionMemory(s);
    expect(m).toContain("make a game");
    expect(m).toContain("Files changed: index.html, style.css");
    expect(m).not.toContain("now");
  });
  it("forgets turns before the /compact point", () => {
    const s: CodeSession = { id: "s", title: "t", root: "/p", created: 0, updated: 0, forget: 1, turns: [turn("old", "x", []), turn("new", "y", [])] };
    expect(sessionMemory(s)).not.toContain("old");
    expect(sessionMemory(s)).toContain("new");
  });
});
