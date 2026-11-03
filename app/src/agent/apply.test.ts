import { describe, expect, it } from "vitest";
import { sha256Hex } from "@core/agent/workspace";
import type { AgentHost } from "../bridge";
import { applyChanges, ConflictError, listCheckpoints, undoCheckpoint } from "./apply";

function memHost(files: Record<string, string>): { host: AgentHost; files: Map<string, string> } {
  const m = new Map(Object.entries(files));
  const host = {
    read: async (_r: string, p: string) => m.get(p) ?? null,
    write: async (_r: string, p: string, c: string | null) => { if (c === null) m.delete(p); else m.set(p, c); },
  } as unknown as AgentHost;
  return { host, files: m };
}

describe("applyChanges", () => {
  it("applies, keeps a checkpoint, and undoes", async () => {
    const { host, files } = memHost({ "a.txt": "old" });
    const id = await applyChanges(host, "/p", [
      { path: "a.txt", before: "old", after: "new", baseHash: await sha256Hex("old") },
      { path: "b.txt", before: null, after: "made", baseHash: null },
    ], "task");
    expect(files.get("a.txt")).toBe("new");
    expect(files.get("b.txt")).toBe("made");
    expect((await listCheckpoints(host, "/p"))[0]?.id).toBe(id);
    await undoCheckpoint(host, "/p", id);
    expect(files.get("a.txt")).toBe("old");
    expect(files.has("b.txt")).toBe(false);
    expect((await listCheckpoints(host, "/p"))[0]?.undone).toBe(true);
  });

  it("refuses when a file changed since the agent read it, and writes nothing", async () => {
    const { host, files } = memHost({ "a.txt": "edited by you" });
    await expect(applyChanges(host, "/p", [{ path: "a.txt", before: "old", after: "new", baseHash: await sha256Hex("old") }], "t")).rejects.toBeInstanceOf(ConflictError);
    expect(files.get("a.txt")).toBe("edited by you");
    expect([...files.keys()].some((k) => k.startsWith(".clutor"))).toBe(false);
  });

  it("refuses to undo over later edits unless forced", async () => {
    const { host, files } = memHost({ "a.txt": "old" });
    const id = await applyChanges(host, "/p", [{ path: "a.txt", before: "old", after: "new", baseHash: await sha256Hex("old") }], "t");
    files.set("a.txt", "newer");
    await expect(undoCheckpoint(host, "/p", id)).rejects.toBeInstanceOf(ConflictError);
    await undoCheckpoint(host, "/p", id, true);
    expect(files.get("a.txt")).toBe("old");
  });
});
