import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { GENESIS_TAG, ShadowWorkspace, buildModelLayout, parseGguf, planCommand, runAgent, sealReceipt, type CommandPlan } from "../src";
import {
  ConflictError, NodeProject, applyChanges, appendReceipt, diskModelFrom, fileReader, hostInfo, lastReceiptTag, listCheckpoints, loadOrCreateReceiptSigner, measureExpertReads, parseIgnore,
  readReceipts, rollbackCheckpoint, scratchRunner, verifyReceiptLog, writeSyntheticGguf,
} from "../src/node";

const dirs: string[] = [];
const tmp = (): string => { const d = mkdtempSync(join(tmpdir(), "clutor-test-")); dirs.push(d); return d; };
afterEach(() => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });
const put = (root: string, rel: string, text: string): void => { mkdirSync(join(root, rel, ".."), { recursive: true }); writeFileSync(join(root, rel), text); };
const plan = (c: string) => { const p = planCommand(c); if (!p.ok) throw new Error(p.reason); return p as Extract<CommandPlan, { ok: true }>; };

describe("NodeProject", () => {
  it("lists source files, skipping generated folders, .gitignore entries and links, and reads text", async () => {
    const r = tmp();
    put(r, "src/a.ts", "export const a = 1;\n"); put(r, "node_modules/x/i.js", "x"); put(r, ".git/config", "x"); put(r, "dist/o.js", "x"); put(r, "gen/out.txt", "x"); put(r, "keep.log", "x"); put(r, "src/b.log", "x");
    put(r, ".gitignore", "gen/\n*.log\n# comment\n");
    symlinkSync("/etc", join(r, "etc-link"));
    const p = new NodeProject(r);
    expect(await p.list()).toEqual([".gitignore", "src/a.ts"]);
    expect(await p.read("src/a.ts")).toBe("export const a = 1;\n");
    expect(await p.read("missing.txt")).toBeNull();
  });
  it("refuses a symbolic link out of the project, even through a directory, and never follows it", async () => {
    const r = tmp(), outside = tmp();
    put(outside, "secret.txt", "TOP SECRET");
    symlinkSync(join(outside, "secret.txt"), join(r, "link.txt"));
    symlinkSync(outside, join(r, "dir-link"));
    const p = new NodeProject(r);
    await expect(p.read("link.txt")).rejects.toThrow(/outside the project|symbolic link/);
    await expect(p.read("dir-link/secret.txt")).rejects.toThrow(/outside the project/);
    await expect(p.safePath("dir-link/new.txt")).rejects.toThrow(/outside the project/);
    await expect(p.safePath("../escape")).rejects.toThrow(/outside the project/);
    expect(await p.list()).toEqual([]);
  });
  it("will not read binary or oversized files", async () => {
    const r = tmp();
    writeFileSync(join(r, "img.png"), Buffer.from([137, 80, 78, 71, 0, 1, 2]));
    put(r, "big.txt", "x".repeat(5000));
    const p = new NodeProject(r, { maxFileBytes: 1000 });
    await expect(p.read("img.png")).rejects.toThrow(/binary|bytes/);
    await expect(p.read("big.txt")).rejects.toThrow(/over 1000 bytes/);
  });
  it("stops at a file limit instead of walking a huge folder", async () => {
    const r = tmp();
    for (let i = 0; i < 30; i++) put(r, `f${i}.txt`, "x");
    await expect(new NodeProject(r, { maxFiles: 10 }).list()).rejects.toThrow(/more than 10 files/);
  });
  it("understands the simple .gitignore forms", () => {
    const ig = parseIgnore("build/\n*.tmp\n/root-only.txt\ndocs/gen\n");
    expect([ig("build/x.js", false), ig("a/build/y", false), ig("x.tmp", false), ig("a/b.tmp", false), ig("root-only.txt", false), ig("sub/root-only.txt", false), ig("docs/gen/a", false), ig("src/ok.ts", false)]).toEqual([true, true, true, true, true, false, true, false]);
  });
});

describe("applying changes", () => {
  async function setup() {
    const r = tmp();
    put(r, "src/a.ts", "const a = 1;\n"); put(r, "src/keep.ts", "keep\n"); put(r, "old.txt", "old\n");
    const project = new NodeProject(r);
    const ws = new ShadowWorkspace(project);
    return { r, project, ws };
  }

  it("writes edits, creations and deletions, and leaves no temp files", async () => {
    const { r, project, ws } = await setup();
    await ws.edit("src/a.ts", "1", "2"); await ws.write("deep/new/file.ts", "n\n"); await ws.delete("old.txt");
    const res = await applyChanges(project, await ws.changeSet());
    expect(res.applied.map((a) => `${a.status}:${a.path}`).sort()).toEqual(["added:deep/new/file.ts", "deleted:old.txt", "modified:src/a.ts"]);
    expect(readFileSync(join(r, "src/a.ts"), "utf8")).toBe("const a = 2;\n");
    expect(readFileSync(join(r, "deep/new/file.ts"), "utf8")).toBe("n\n");
    expect(existsSync(join(r, "old.txt"))).toBe(false);
    expect(readFileSync(join(r, "src/keep.ts"), "utf8")).toBe("keep\n");
    expect(readdirSync(join(r, "src")).filter((f) => f.endsWith(".clutor-tmp"))).toEqual([]);
  });

  it("keeps a file's permissions", async () => {
    const { r, project, ws } = await setup();
    put(r, "run.sh", "#!/bin/sh\n"); chmodSync(join(r, "run.sh"), 0o755);
    await ws.edit("run.sh", "sh", "bash");
    await applyChanges(project, await ws.changeSet());
    expect(statSync(join(r, "run.sh")).mode & 0o777).toBe(0o755);
  });

  it("refuses to overwrite a file you edited meanwhile, naming it, and changes nothing", async () => {
    const { r, project, ws } = await setup();
    await ws.edit("src/a.ts", "1", "2"); await ws.edit("src/keep.ts", "keep", "kept");
    put(r, "src/a.ts", "const a = 99; // my edit\n");
    const err = await applyChanges(project, await ws.changeSet()).catch((e) => e);
    expect(err).toBeInstanceOf(ConflictError);
    expect(err.conflicts).toEqual([{ path: "src/a.ts", reason: "it was edited" }]);
    expect(readFileSync(join(r, "src/keep.ts"), "utf8")).toBe("keep\n");
    expect(readFileSync(join(r, "src/a.ts"), "utf8")).toContain("my edit");
    expect(await listCheckpoints(project)).toEqual([]);
    const forced = await applyChanges(project, await ws.changeSet(), { force: true });
    expect(forced.applied).toHaveLength(2);
  });

  it("notices a file created or deleted behind its back", async () => {
    const { r, project, ws } = await setup();
    await ws.write("new.txt", "mine\n");
    put(r, "new.txt", "theirs\n");
    await expect(applyChanges(project, await ws.changeSet())).rejects.toMatchObject({ conflicts: [{ path: "new.txt", reason: "it now exists" }] });
    const ws2 = new ShadowWorkspace(project);
    await ws2.edit("old.txt", "old", "newer");
    rmSync(join(r, "old.txt"));
    await expect(applyChanges(project, await ws2.changeSet())).rejects.toMatchObject({ conflicts: [{ path: "old.txt", reason: "it was deleted" }] });
  });

  it("rolls back to exactly the original, including deleting what was created and restoring what was deleted", async () => {
    const { r, project, ws } = await setup();
    await ws.edit("src/a.ts", "1", "2"); await ws.write("created.ts", "c\n"); await ws.delete("old.txt");
    const { id } = await applyChanges(project, await ws.changeSet());
    expect((await listCheckpoints(project))[0]).toMatchObject({ id, files: 3 });
    await rollbackCheckpoint(project, id);
    expect(readFileSync(join(r, "src/a.ts"), "utf8")).toBe("const a = 1;\n");
    expect(existsSync(join(r, "created.ts"))).toBe(false);
    expect(readFileSync(join(r, "old.txt"), "utf8")).toBe("old\n");
  });

  it("will not roll back over edits made after the apply, unless forced", async () => {
    const { r, project, ws } = await setup();
    await ws.edit("src/a.ts", "1", "2");
    const { id } = await applyChanges(project, await ws.changeSet());
    put(r, "src/a.ts", "const a = 3; // after\n");
    await expect(rollbackCheckpoint(project, id)).rejects.toBeInstanceOf(ConflictError);
    await rollbackCheckpoint(project, id, { force: true });
    expect(readFileSync(join(r, "src/a.ts"), "utf8")).toBe("const a = 1;\n");
    await expect(rollbackCheckpoint(project, "../../etc")).rejects.toThrow(/not a checkpoint id/);
  });

  it("will not write through a link that leaves the project", async () => {
    const { r, project } = await setup();
    const outside = tmp();
    symlinkSync(outside, join(r, "ln"));
    await expect(applyChanges(project, [{ path: "ln/pwned.txt", before: null, after: "x", baseHash: null }])).rejects.toThrow(/outside the project/);
    expect(existsSync(join(outside, "pwned.txt"))).toBe(false);
  });
});

describe("running commands in a scratch copy", () => {
  async function proj(files: Record<string, string>) {
    const r = tmp();
    for (const [k, v] of Object.entries(files)) put(r, k, v);
    const project = new NodeProject(r);
    return { r, project, ws: new ShadowWorkspace(project) };
  }

  it("runs against the agent's unapplied changes and leaves the real folder alone", async () => {
    const { r, project, ws } = await proj({ "t.js": "process.exit(require('./v.js') === 2 ? 0 : 1)", "v.js": "module.exports = 1;" });
    const run = scratchRunner(project, ws);
    expect((await run(plan("node t.js"), "node t.js")).exitCode).toBe(1);
    await ws.write("v.js", "module.exports = 2;");
    expect((await run(plan("node t.js"), "node t.js")).exitCode).toBe(0);
    expect(readFileSync(join(r, "v.js"), "utf8")).toBe("module.exports = 1;");
  });

  it("returns output and a failing exit code, and stops chained commands at the first failure", async () => {
    const { project, ws } = await proj({ "ok.js": "console.log('first')", "bad.js": "console.error('boom'); process.exit(3)", "never.js": "console.log('NEVER')" });
    const r = await scratchRunner(project, ws)(plan("node ok.js && node bad.js && node never.js"), "x");
    expect(r.exitCode).toBe(3);
    expect(r.output).toContain("first");
    expect(r.output).toContain("boom");
    expect(r.output).not.toContain("NEVER");
  });

  it("kills a program that runs too long", async () => {
    const { project, ws } = await proj({ "hang.js": "setInterval(() => {}, 1000)" });
    const t0 = Date.now();
    const r = await scratchRunner(project, ws, { timeoutMs: 1500 })(plan("node hang.js"), "x");
    expect(r.timedOut).toBe(true);
    expect(Date.now() - t0).toBeLessThan(6000);
  });

  it("gives the program a clean environment: no inherited secrets, a throwaway HOME, and a working folder that is removed afterwards", async () => {
    const { r: root, project, ws } = await proj({ "env.js": "console.log(JSON.stringify({ secret: process.env.CLUTOR_TEST_SECRET ?? null, home: process.env.HOME, cwd: process.cwd(), ci: process.env.CI }))" });
    const out = await scratchRunner(project, ws, { env: { PATH: process.env["PATH"], CLUTOR_TEST_SECRET: "hunter2", HOME: "/Users/real" } })(plan("node env.js"), "x");
    const o = JSON.parse(out.output.trim().split("\n").at(-1) as string) as { secret: string | null; home: string; cwd: string; ci: string };
    expect(o.secret).toBeNull();
    expect(o.home).not.toBe("/Users/real");
    expect(o.ci).toBe("1");
    expect(o.cwd.startsWith(root)).toBe(false);
    expect(existsSync(o.cwd)).toBe(false);
    const passed = await scratchRunner(project, ws, { passEnv: ["CLUTOR_TEST_SECRET"], env: { PATH: process.env["PATH"], CLUTOR_TEST_SECRET: "visible" } })(plan("node env.js"), "x");
    expect(passed.output).toContain("visible");
  });

  it("links node_modules instead of copying it, and reports a missing program clearly", async () => {
    const { r, project, ws } = await proj({ "use.js": "console.log(require('dep'))" });
    put(r, "node_modules/dep/index.js", "module.exports = 'from-dep';");
    expect((await scratchRunner(project, ws)(plan("node use.js"), "x")).output).toContain("from-dep");
    const missing = await scratchRunner(project, ws, { env: { PATH: "/nonexistent" } })(plan("node use.js"), "x");
    expect(missing.exitCode).not.toBe(0);
  });

  it("caps output and the size of project it will copy", async () => {
    const { project, ws } = await proj({ "spam.js": "console.log('x'.repeat(100000))" });
    const r = await scratchRunner(project, ws, { maxOutputBytes: 5000 })(plan("node spam.js"), "x");
    expect(r.output.length).toBeLessThan(5200);
    expect(r.output).toContain("cut off");
    const big = await scratchRunner(project, ws, { maxCopyBytes: 10 })(plan("node spam.js"), "x");
    expect(big.output).toMatch(/too big to copy/);
  });

  it("works end to end with the agent: a failing test is fixed and verified in the scratch copy, and applied for real afterwards", async () => {
    const { r, project, ws } = await proj({ "package.json": "{}", "calc.js": "exports.add = (a, b) => a - b;", "t.js": "const { add } = require('./calc'); if (add(2, 3) !== 5) { console.error('add(2,3) =', add(2, 3)); process.exit(1); }" });
    const script = [
      { tool: "read_file", args: { path: "calc.js" } },
      { tool: "run_command", args: { command: "node t.js" } },
      { tool: "edit_file", args: { path: "calc.js", old_text: "a - b", new_text: "a + b" } },
      { tool: "finish", args: { summary: "fixed add" } },
    ];
    let i = 0;
    const llm = { model: "scripted", toolMode: "json" as const, async chat() { return { content: JSON.stringify(script[i++]), toolCalls: [] }; } };
    const asked: string[] = [];
    const res = await runAgent({ task: "make t.js pass", llm, ws, verifyCommand: "node t.js", runCommand: scratchRunner(project, ws), approve: async ({ command }) => (asked.push(command), true) });
    expect(res.status).toBe("finished");
    expect(res.commands.map((c) => c.exitCode)).toEqual([1, 0]);
    expect(res.verification).toEqual({ command: "node t.js", ok: true, ran: true, retries: 0 });
    expect(readFileSync(join(r, "calc.js"), "utf8")).toContain("a - b");
    await applyChanges(project, await ws.changeSet());
    expect(readFileSync(join(r, "calc.js"), "utf8")).toContain("a + b");
  });
});

describe("model files on disk", () => {
  const spec = { arch: "qwen2moe", name: "disk-moe", layers: 4, embedding: 64, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 16, expertsUsed: 2, expertFfn: 64, denseFfn: 0, vocab: 128, sharedExperts: 0 };
  it("writes a real GGUF that parses back, with every tensor inside the file, and reads experts from it", async () => {
    const r = tmp(), path = join(r, "m.gguf");
    const { bytes } = await writeSyntheticGguf(path, spec);
    expect(statSync(path).size).toBe(bytes);
    const reader = await fileReader(path);
    const f = await parseGguf(reader);
    const layout = buildModelLayout(f);
    expect(layout).toMatchObject({ layerCount: 4, expertCount: 16, expertsUsed: 2 });
    const m = await measureExpertReads(path, layout, 16 * 2 ** 30, 20);
    expect(m.samples).toBe(20);
    expect(m.bytes).toBeGreaterThan(0);
    expect(m.bytesPerSec).toBeGreaterThan(0);
    expect(m.likelyCold).toBe(false); // a tiny file in a big-RAM machine is served from memory
    const dm = diskModelFrom(m, layout.layers[0]!.expertBytes, 3);
    expect(dm.bytesPerSec).toBeGreaterThan(0);
    expect(dm.requestLatencySec).toBeGreaterThanOrEqual(0);
    await reader.close();
  });
  it("is reproducible from a seed", async () => {
    const r = tmp();
    await writeSyntheticGguf(join(r, "a.gguf"), spec, 7); await writeSyntheticGguf(join(r, "b.gguf"), spec, 7); await writeSyntheticGguf(join(r, "c.gguf"), spec, 8);
    expect(readFileSync(join(r, "a.gguf")).equals(readFileSync(join(r, "b.gguf")))).toBe(true);
    expect(readFileSync(join(r, "a.gguf")).equals(readFileSync(join(r, "c.gguf")))).toBe(false);
  });
  it("reports this machine", () => {
    const h = hostInfo();
    expect(h.totalRamBytes).toBeGreaterThan(2 ** 28);
    expect(h.logicalCores).toBeGreaterThan(0);
    expect(["macOS", "Windows", "Linux"]).toContain(h.platformName);
  });
});

describe("receipt log", () => {
  it("keeps a private key, chains receipts across runs, and notices tampering with the file", async () => {
    const r = tmp();
    const key = await loadOrCreateReceiptSigner(r);
    expect(key.publicJwk).toMatchObject({ kty: "EC", crv: "P-256" });
    expect(statSync(join(r, ".clutor/keys/receipt-p256.jwk.json")).mode & 0o777).toBe(0o600);
    expect(JSON.stringify(await loadOrCreateReceiptSigner(r))).toBe(JSON.stringify(key));
    expect(await lastReceiptTag(r)).toBe(GENESIS_TAG);
    for (let i = 0; i < 3; i++) {
      const rec = await sealReceipt(key, { v: 2, id: `r${i}`, task: "t", model: "m", startedAt: "a", finishedAt: "b", status: "finished", files: [], commands: [], steps: i, previous: await lastReceiptTag(r) });
      await appendReceipt(r, rec);
    }
    expect(await verifyReceiptLog(r)).toMatchObject({ ok: true, count: 3 });
    const path = join(r, ".clutor/receipts.jsonl");
    const lines = readFileSync(path, "utf8").trim().split("\n");
    writeFileSync(path, [lines[0], lines[1]!.replace('"steps":1', '"steps":9'), lines[2]].join("\n") + "\n");
    expect(await verifyReceiptLog(r)).toMatchObject({ ok: false, index: 1 });
    expect((await readReceipts(r))).toHaveLength(3);
  });
});
