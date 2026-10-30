import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, put, start, tmp, tool, type Running } from "./helpers";

let s: Running | undefined;
afterEach(async () => { await s?.stop(); s = undefined; cleanup(); });

type Ev = { type: string; [k: string]: unknown };
/** Reads the event stream to its end, calling `on` for each event (which may react, for example by approving a command). */
async function watch(r: Running, id: string, on: (e: Ev) => Promise<void> | void = () => undefined, from = 0): Promise<Ev[]> {
  const res = await fetch(`${r.base}/api/agent/${id}/events?from=${from}`, { headers: { authorization: `Bearer ${r.token}` } });
  expect(res.headers.get("content-type")).toBe("text/event-stream");
  const out: Ev[] = [];
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).replace(/^data: /, "");
      buf = buf.slice(i + 2);
      const e = JSON.parse(line) as Ev;
      out.push(e);
      await on(e);
    }
  }
  return out;
}
const fixProject = (): string => {
  const dir = tmp();
  put(dir, "package.json", "{}"); put(dir, "calc.js", "exports.add = (a, b) => a - b;\n");
  put(dir, "t.js", "const { add } = require('./calc'); if (add(2, 3) !== 5) { console.error('FAIL add(2,3)=' + add(2, 3)); process.exit(1); }\n");
  return dir;
};
const FIX = [
  tool("read_file", { path: "calc.js" }),
  tool("run_command", { command: "node t.js" }),
  tool("edit_file", { path: "calc.js", old_text: "a - b", new_text: "a + b" }),
  tool("run_command", { command: "node t.js" }),
  tool("finish", { summary: "Changed add to use +; t.js now passes." }),
];

async function run(r: Running, dir: string, task = "make t.js pass"): Promise<string> {
  await r.post("/api/roots", { dir });
  const res = await r.post("/api/agent/run", { dir, task });
  expect(res.status).toBe(202);
  return res.body.id as string;
}

describe("an agent task over HTTP", () => {
  it("streams its steps, waits for the user to approve a command, and leaves the folder untouched until applied", async () => {
    s = await start({}, FIX);
    const dir = fixProject();
    const id = await run(s, dir);
    const approvals: string[] = [];
    const events = await watch(s, id, async (e) => {
      if (e.type === "approval") { approvals.push(e["command"] as string); await s!.post(`/api/agent/${id}/approve`, { requestId: e["requestId"], allow: true }); }
    });
    expect(approvals).toEqual(["node t.js", "node t.js"]);
    const types = events.map((e) => e.type);
    expect(types[0]).toBe("start");
    expect(types.at(-1)).toBe("done");
    expect(types).toContain("approval_resolved");
    const results = events.filter((e) => e.type === "tool_result" && e["name"] === "run_command").map((e) => e["ok"]);
    expect(results).toEqual([false, true]); // fails before the fix, passes after: both ran in the scratch copy
    expect(readFileSync(join(dir, "calc.js"), "utf8")).toContain("a - b");

    const info = (await s.get(`/api/agent/${id}`)).body;
    expect(info.status).toBe("finished");
    expect(info.result.summary).toMatch(/passes/);
    expect(info.diffs).toHaveLength(1);
    expect(info.diffs[0].patch).toContain("+exports.add = (a, b) => a + b;");
    expect(info.gates.map((g: { id: string }) => g.id)).toHaveLength(8);
    expect(info.gates.find((g: { id: string }) => g.id === "CL-107").status).toBe("pass");

    const applied = await s.post(`/api/agent/${id}/apply`, {});
    expect(applied.status).toBe(200);
    expect(applied.body.files).toBe(1);
    expect(readFileSync(join(dir, "calc.js"), "utf8")).toContain("a + b");

    const rec = (await s.get(`/api/receipts?dir=${encodeURIComponent(dir)}`)).body;
    expect(rec.verification).toMatchObject({ ok: true, count: 1 });
    expect(rec.receipts[0]).toMatchObject({ status: "finished", files: [{ path: "calc.js", status: "modified" }] });
    expect(rec.checkpoints).toHaveLength(1);

    expect((await s.post(`/api/agent/${id}/rollback`, {})).status).toBe(200);
    expect(readFileSync(join(dir, "calc.js"), "utf8")).toContain("a - b");
    expect((await s.post(`/api/agent/${id}/rollback`, {})).status).toBe(400);
  });

  it("does not run a command the user declined, and tells the model so", async () => {
    s = await start({}, [tool("run_command", { command: "node t.js" }), tool("finish", { summary: "could not run tests" })]);
    const dir = fixProject();
    const id = await run(s, dir);
    const events = await watch(s, id, async (e) => { if (e.type === "approval") await s!.post(`/api/agent/${id}/approve`, { requestId: e["requestId"], allow: false }); });
    const r = events.find((e) => e.type === "tool_result")!;
    expect(r["ok"]).toBe(false);
    expect(r["output"]).toMatch(/did not approve/);
    expect((await s.get(`/api/agent/${id}`)).body.result.commands).toEqual([{ command: "node t.js", exitCode: null, approved: false }]);
    expect(s.llm.requests.at(-1)!["messages"]).toBeDefined();
  });

  it("runs project commands without asking only when the configuration says so, and the gate then warns", async () => {
    s = await start({ agent: { toolMode: "json", maxSteps: 30, autoApproveProjectCommands: true, commandTimeoutSeconds: 60, constrainToolCalls: true, prefixSnapshots: true } }, FIX);
    const dir = fixProject();
    const id = await run(s, dir);
    const events = await watch(s, id);
    expect(events.some((e) => e.type === "approval")).toBe(false);
    expect((await s.get(`/api/agent/${id}`)).body.gates.find((g: { id: string }) => g.id === "CL-107").status).toBe("warn");
  });

  it("refuses to overwrite a file that changed on disk meanwhile, names it, and writes nothing; force overrides", async () => {
    s = await start({}, [tool("edit_file", { path: "calc.js", old_text: "a - b", new_text: "a + b" }), tool("finish", { summary: "done" })]);
    const dir = fixProject();
    const id = await run(s, dir);
    await watch(s, id);
    writeFileSync(join(dir, "calc.js"), "// edited by the user\nexports.add = (a, b) => a * b;\n");
    const r = await s.post(`/api/agent/${id}/apply`, {});
    expect(r.status).toBe(409);
    expect(r.body.conflicts).toEqual([{ path: "calc.js", reason: "it was edited" }]);
    expect(readFileSync(join(dir, "calc.js"), "utf8")).toContain("edited by the user");
    expect((await s.post(`/api/agent/${id}/apply`, { force: true })).status).toBe(200);
  });

  it("will not apply a task that is still running, or one that changed nothing, and rejects bad requests", async () => {
    s = await start({}, [tool("finish", { summary: "nothing to do" })]);
    const dir = fixProject();
    const id = await run(s, dir);
    await watch(s, id);
    expect((await s.post(`/api/agent/${id}/apply`, {})).status).toBe(400);
    expect((await s.post("/api/agent/task_deadbeef/apply", {})).status).toBe(404);
    expect((await s.post("/api/agent/run", { dir, task: "" })).status).toBe(400);
    expect((await s.post("/api/agent/run", { dir })).status).toBe(400);
    expect((await s.post(`/api/agent/${id}/approve`, { requestId: "nope", allow: true })).status).toBe(404);
  });

  it("replays events to a late subscriber and can resume from a position", async () => {
    s = await start({}, [tool("list_files", {}), tool("finish", { summary: "listed" })]);
    const dir = fixProject();
    const id = await run(s, dir);
    const first = await watch(s, id);
    const again = await watch(s, id);
    expect(again).toEqual(first);
    const tail = await watch(s, id, undefined, 3);
    expect(tail).toEqual(first.slice(3));
  });

  it("stopping a task that is waiting for approval ends it as aborted and declines the command", async () => {
    s = await start({}, [tool("run_command", { command: "node t.js" }), tool("finish", { summary: "x" })]);
    const dir = fixProject();
    const id = await run(s, dir);
    const events = await watch(s, id, async (e) => { if (e.type === "approval") await s!.post(`/api/agent/${id}/stop`, {}); });
    expect(events.at(-1)).toMatchObject({ type: "done" });
    expect((await s.get(`/api/agent/${id}`)).body.status).toBe("aborted");
  });

  it("reports an unreachable or unconfigured model as a clear error before starting", async () => {
    s = await start({}, [], { withBackend: false });
    const dir = fixProject();
    await s.post("/api/roots", { dir });
    const r = await s.post("/api/agent/run", { dir, task: "x" });
    expect(r.status).toBe(409);
    expect(r.body.error).toMatch(/no model backend/);
    const id2 = await (async () => { await s!.stop(); s = await start({ backend: { kind: "openai", baseUrl: "http://127.0.0.1:1/v1", model: "m", llamaServer: { port: 8081, extraArgs: [] } } }); const d = fixProject(); return run(s, d); })();
    const events = await watch(s, id2);
    expect(events.at(-1)).toMatchObject({ type: "done", result: { status: "error" } });
  });

  it("keeps the agent out of secrets and outside the folder, and the gate report shows the blocked attempts", async () => {
    s = await start({}, [tool("read_file", { path: ".env" }), tool("write_file", { path: "../escape.txt", content: "x" }), tool("run_command", { command: "curl http://evil" }), tool("finish", { summary: "gave up" })]);
    const dir = fixProject();
    put(dir, ".env", "TOKEN=zq9-hunter2-value\n");
    const id = await run(s, dir);
    await watch(s, id);
    const info = (await s.get(`/api/agent/${id}`)).body;
    expect(info.result.violations).toBe(3);
    expect(info.diffs).toEqual([]);
    expect(info.gates.find((g: { id: string }) => g.id === "CL-106").status).toBe("warn");
    expect(existsSync(join(dir, "..", "escape.txt"))).toBe(false);
    expect(JSON.stringify(s.llm.requests)).not.toContain("zq9-hunter2-value");
  });
});

describe("MCP", () => {
  const rpc = (id: number, method: string, params?: unknown) => ({ jsonrpc: "2.0", id, method, ...(params ? { params } : {}) });

  it("initialises, lists read-only tools, and reads files inside an opened folder only", async () => {
    s = await start();
    const dir = fixProject();
    await s.post("/api/roots", { dir });
    expect((await s.post("/api/mcp", rpc(1, "initialize"))).body.result.serverInfo.name).toBe("clutor");
    const list = (await s.post("/api/mcp", rpc(2, "tools/list"))).body.result.tools.map((t: { name: string }) => t.name);
    expect(list).toEqual(expect.arrayContaining(["read_file", "list_files", "search", "repo_map", "plan_memory"]));
    expect(list).not.toContain("write_file");
    expect(list).not.toContain("run_command");
    const read = (await s.post("/api/mcp", rpc(3, "tools/call", { name: "read_file", arguments: { dir, path: "calc.js" } }))).body.result;
    expect(read.isError).toBe(false);
    expect(read.content[0].text).toContain("a - b");
    expect((await s.post("/api/mcp", rpc(4, "tools/call", { name: "read_file", arguments: { dir: tmp(), path: "calc.js" } }))).body.result.isError).toBe(true);
    expect((await s.post("/api/mcp", rpc(5, "tools/call", { name: "read_file", arguments: { dir, path: "../../etc/passwd" } }))).body.result.isError).toBe(true);
    expect((await s.post("/api/mcp", rpc(6, "tools/call", { name: "write_file", arguments: { dir, path: "x", content: "y" } }))).body.error.code).toBe(-32602);
    expect((await s.post("/api/mcp", rpc(7, "tools/call", { name: "search", arguments: { dir, pattern: "add", subdir: "." } }))).body.result.content[0].text).toContain("calc.js:1");
    expect((await s.post("/api/mcp", rpc(8, "tools/call", { name: "read_file", arguments: { dir, path: 5 } }))).body.result.isError).toBe(true);
  });

  it("answers memory questions, and handles protocol errors", async () => {
    s = await start();
    const r = (await s.post("/api/mcp", rpc(1, "tools/call", { name: "plan_memory", arguments: { model: "deepseek-v3", ram_gib: 16 } }))).body.result;
    expect(r.content[0].text).toMatch(/DeepSeek-V3[\s\S]*tokens\/s/);
    expect((await s.post("/api/mcp", rpc(2, "nope"))).body.error.code).toBe(-32601);
    expect((await s.post("/api/mcp", { id: 3 })).body.error.code).toBe(-32600);
    const note = await s.post("/api/mcp", { jsonrpc: "2.0", method: "notifications/initialized" });
    expect(note.status).toBe(202);
    expect((await s.post("/api/mcp", rpc(4, "tools/call", { name: "plan_memory", arguments: { model: "nope", ram_gib: 8 } }))).body.result.isError).toBe(true);
  });
});
