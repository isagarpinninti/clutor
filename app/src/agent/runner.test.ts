import { describe, expect, it } from "vitest";
import type { StreamEvent } from "@engine/stream";
import type { AgentHost } from "../bridge";
import { applyChanges } from "./apply";
import { runCodingTask } from "./runner";

/** A model that follows a script, one JSON tool call per reply. */
function scripted(replies: string[]) {
  let i = 0;
  return { async *stream(): AsyncGenerator<StreamEvent> { yield { type: "content", text: replies[i++] ?? '{"tool":"finish","args":{"summary":"out of script"}}' }; yield { type: "done", finish: "stop" }; } };
}

function host(files: Record<string, string>): AgentHost {
  const m = new Map(Object.entries(files));
  return {
    list: async () => [...m.keys()].filter((k) => !k.startsWith(".clutor")).sort(),
    read: async (_r: string, p: string) => m.get(p) ?? null,
    write: async (_r: string, p: string, c: string | null) => { if (c === null) m.delete(p); else m.set(p, c); },
    run: async (_r: string, overlay: Array<{ path: string; content: string | null }>) => ({ exit_code: 0, output: `ran with ${overlay.length} changed file(s)`, timed_out: false, duration_ms: 1 }),
  } as unknown as AgentHost;
}

describe("runCodingTask", () => {
  it("reads, edits in the shadow workspace, leaves the folder alone, and the changes can then be applied", async () => {
    const h = host({ "app.js": "const x = 1;\n" });
    const events: string[] = [];
    const { result, ws } = await runCodingTask({
      host: h, root: "/p", modelName: "scripted", task: "set x to 2", approve: async () => true, onEvent: (e) => events.push(e.type),
      session: scripted([
        '{"tool":"read_file","args":{"path":"app.js"}}',
        '{"tool":"edit_file","args":{"path":"app.js","old_text":"x = 1","new_text":"x = 2"}}',
        '{"tool":"finish","args":{"summary":"changed x"}}',
      ]),
    });
    expect(result.status).toBe("finished");
    expect(result.diffs.map((d) => d.path)).toEqual(["app.js"]);
    expect(await h.read("/p", "app.js")).toBe("const x = 1;\n");   // not applied yet
    expect(events).toContain("tool_call");
    await applyChanges(h, "/p", await ws.changeSet(), "set x");
    expect(await h.read("/p", "app.js")).toBe("const x = 2;\n");
  });

  it("asks before running a command and does not run it when refused", async () => {
    const asked: string[] = [];
    const { result } = await runCodingTask({
      host: host({ "package.json": '{"scripts":{"test":"echo"}}' }), root: "/p", modelName: "s", task: "run tests",
      approve: async (i) => { asked.push(i.command); return false; }, onEvent: () => undefined,
      session: scripted(['{"tool":"run_command","args":{"command":"npm test"}}', '{"tool":"finish","args":{"summary":"not run"}}']),
    });
    expect(asked).toEqual(["npm test"]);
    expect(result.commands[0]?.approved).toBe(false);
  });

  it("stops when aborted", async () => {
    const ac = new AbortController();
    ac.abort();
    const { result } = await runCodingTask({ host: host({}), root: "/p", modelName: "s", task: "x", approve: async () => true, onEvent: () => undefined, signal: ac.signal, session: scripted([]) });
    expect(result.status).toBe("aborted");
  });

  it("lets the agent search the web only after one approval for the run", async () => {
    const asked: string[] = [];
    const web = { search: async () => [{ title: "T", url: "https://a.org/", snippet: "s" }], fetch: async () => { throw new Error("no"); }, openExternal: async () => undefined };
    const { result } = await runCodingTask({
      host: host({}), root: "/p", modelName: "s", task: "look up x", web, approve: async (i) => { asked.push(i.command); return true; }, onEvent: () => undefined,
      session: scripted(['{"tool":"web_search","args":{"query":"x"}}', '{"tool":"web_search","args":{"query":"y"}}', '{"tool":"finish","args":{"summary":"done"}}']),
    });
    expect(result.status).toBe("finished");
    expect(asked).toHaveLength(1);
    expect(asked[0]).toMatch(/web/i);
  });
});
