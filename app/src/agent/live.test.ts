import { describe, expect, it } from "vitest";
import { liveStore, parseLive } from "./live";

describe("reading a reply while it is still being written", () => {
  it("knows nothing yet from an empty or prose-only start", () => {
    expect(parseLive("")).toMatchObject({ label: "Thinking", chars: 0 });
    expect(parseLive("I will first look at the project.")).toMatchObject({ label: "Thinking it through", prose: "I will first look at the project." });
  });
  it("shows the file and its content as they arrive, unescaped", () => {
    const l = parseLive('{"tool": "write_file", "args": {"path": "index.html", "content": "<!doctype html>\\n<p>he');
    expect(l).toMatchObject({ tool: "write_file", path: "index.html", label: "Writing index.html", text: "<!doctype html>\n<p>he" });
  });
  it("handles an escape cut off at the end, and escaped quotes", () => {
    expect(parseLive('{"tool":"write_file","args":{"path":"a.js","content":"say \\"hi\\"\\').text).toBe('say "hi"');
  });
  it("reads a plan as it is written", () => {
    const l = parseLive('{"tool":"update_plan","args":{"steps":"[x] Read the code\\n[>] Write index.html\\n[ ] Sty');
    expect(l.label).toBe("Writing the plan");
    expect(l.plan?.map((s) => s.state)).toEqual(["done", "active", "todo"]);
  });
  it("labels other tools, ignores a thinking block, and stops a string at its closing quote", () => {
    expect(parseLive('<think>hmm {"tool":"x"}</think>{"tool":"read_file","args":{"path":"src/a.ts"}}').label).toBe("Reading src/a.ts");
    expect(parseLive('{"tool":"run_command","args":{"command":"npm test"}} trailing').label).toBe("Running npm test");
    expect(parseLive('{"tool":"finish","args":{"summary":"All done."}}').text).toBe("All done.");
  });
});

describe("the live store", () => {
  it("tells listeners when a task's text changes, and forgets it when the task is done", () => {
    let n = 0;
    const off = liveStore.subscribe(() => n++);
    liveStore.set("t1", "abc"); liveStore.set("t1", "abcd");
    expect(liveStore.get("t1")).toBe("abcd");
    liveStore.clear("t1");
    expect(liveStore.get("t1")).toBeUndefined();
    expect(n).toBe(3);
    off();
    liveStore.set("t2", "x"); liveStore.clear("t2");
    expect(n).toBe(3);
  });
});
