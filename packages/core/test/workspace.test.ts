import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { GENESIS_TAG, MemoryProject, ShadowWorkspace, WorkspaceError, applyUnifiedDiff, buildRepoMap, extractSymbols, generateReceiptSigner, keyThumbprint, sealReceipt, verifyReceiptChain, type ReceiptDraft, type ReceiptSigner } from "../src";

const proj = () => MemoryProject.from({
  "package.json": '{"name":"demo","scripts":{"test":"vitest run"}}',
  "src/math.ts": "export function add(a: number, b: number) {\n  return a + b;\n}\n\nexport const PI = 3.14;\n",
  "src/util.ts": "export class Counter {\n  n = 0;\n}\n",
  "README.md": "# demo\n",
  ".env": "SECRET=1\n",
});

describe("shadow workspace", () => {
  it("reads through to the project and shows the agent its own changes, without touching the project", async () => {
    const p = proj(), ws = new ShadowWorkspace(p);
    expect(await ws.read("src/math.ts")).toContain("add");
    await ws.write("src/new.ts", "export {};\n");
    await ws.edit("src/math.ts", "a + b", "a + b + 0");
    expect(await ws.read("src/new.ts")).toBe("export {};\n");
    expect(await ws.read("src/math.ts")).toContain("a + b + 0");
    expect(p.files.has("src/new.ts")).toBe(false);
    expect(p.files.get("src/math.ts")).toContain("return a + b;");
    expect(await ws.list()).toContain("src/new.ts");
  });

  it("edits only an unambiguous match, and says what to do otherwise", async () => {
    const ws = new ShadowWorkspace(MemoryProject.from({ "a.txt": "x\nfoo\nx\n" }));
    await expect(ws.edit("a.txt", "x", "y")).rejects.toMatchObject({ code: "ambiguous" });
    await expect(ws.edit("a.txt", "zzz", "y")).rejects.toMatchObject({ code: "no_match" });
    await expect(ws.edit("a.txt", "", "y")).rejects.toMatchObject({ code: "no_match" });
    await expect(ws.edit("missing.txt", "a", "b")).rejects.toMatchObject({ code: "not_found" });
    expect(await ws.edit("a.txt", "x", "y", true)).toEqual({ replacements: 2 });
    expect(await ws.read("a.txt")).toBe("y\nfoo\ny\n");
    // a replacement string containing $ patterns must be taken literally
    await ws.edit("a.txt", "foo", "$& $1 $$");
    expect(await ws.read("a.txt")).toContain("$& $1 $$");
  });

  it("deletes, recreates, and reports diffs against the real files", async () => {
    const ws = new ShadowWorkspace(proj());
    await ws.delete("README.md");
    await ws.write("src/new.ts", "a\nb\n");
    await ws.edit("src/util.ts", "n = 0", "n = 1");
    const d = await ws.diffs();
    expect(d.map((x) => [x.path, x.status])).toEqual([["README.md", "deleted"], ["src/new.ts", "added"], ["src/util.ts", "modified"]]);
    await expect(ws.delete("nope.txt")).rejects.toMatchObject({ code: "not_found" });
    expect(await ws.exists("README.md")).toBe(false);
    expect(await ws.list()).not.toContain("README.md");
  });

  it("drops a write that leaves the file as it was", async () => {
    const ws = new ShadowWorkspace(proj());
    await ws.write("README.md", "# demo\n");
    expect(await ws.diffs()).toEqual([]);
    expect(await ws.changeSet()).toEqual([]);
  });

  it("applies the reported patch to the original to get the edited file", async () => {
    const p = proj(), ws = new ShadowWorkspace(p);
    await ws.edit("src/math.ts", "return a + b;", "const s = a + b;\n  return s;");
    const [d] = await ws.diffs();
    expect(applyUnifiedDiff(p.files.get("src/math.ts") as string, d!.patch)).toBe(await ws.read("src/math.ts"));
  });

  it("keeps the agent out of secrets and protected files, and counts the attempts", async () => {
    const ws = new ShadowWorkspace(proj());
    await expect(ws.read(".env")).rejects.toMatchObject({ code: "jail" });
    await expect(ws.read("../outside.txt")).rejects.toMatchObject({ code: "jail" });
    await expect(ws.write(".git/hooks/pre-commit", "x")).rejects.toMatchObject({ code: "jail" });
    await expect(ws.write("/etc/passwd", "x")).rejects.toMatchObject({ code: "jail" });
    await expect(ws.delete(".env")).rejects.toMatchObject({ code: "jail" });
    expect(ws.violations).toBe(5);
    expect(ws.size).toBe(0);
  });

  it("enforces size and count limits", async () => {
    const ws = new ShadowWorkspace(proj(), { maxFileBytes: 100, maxChangedFiles: 3, maxTotalChangeBytes: 150 });
    await expect(ws.write("big.txt", "x".repeat(101))).rejects.toMatchObject({ code: "too_large" });
    await ws.write("a", "x".repeat(60)); await ws.write("b", "x".repeat(60));
    await expect(ws.write("c", "x".repeat(60))).rejects.toMatchObject({ code: "too_large" });
    await ws.write("c", "y");
    await expect(ws.write("d", "y")).rejects.toMatchObject({ code: "too_many" });
    await ws.write("a", "z"); // rewriting a file already changed is fine
  });

  it("remembers the hash each file had when it was first read, to detect edits made meanwhile", async () => {
    const ws = new ShadowWorkspace(proj());
    await ws.edit("src/util.ts", "n = 0", "n = 2");
    await ws.write("fresh.txt", "x");
    const cs = await ws.changeSet();
    expect(cs.find((c) => c.path === "src/util.ts")?.baseHash).toMatch(/^[0-9a-f]{64}$/);
    expect(cs.find((c) => c.path === "fresh.txt")?.baseHash).toBeNull();
  });

  it("any sequence of writes and deletes leaves a workspace whose listing matches its reads", async () => {
    await fc.assert(fc.asyncProperty(fc.array(fc.tuple(fc.constantFrom("w", "d"), fc.constantFrom("a.txt", "b.txt", "src/c.ts", "README.md"), fc.string({ maxLength: 8 })), { maxLength: 25 }), async (ops) => {
      const ws = new ShadowWorkspace(proj());
      for (const [op, path, text] of ops) {
        try { if (op === "w") await ws.write(path, text); else await ws.delete(path); } catch (e) { expect(e).toBeInstanceOf(WorkspaceError); }
      }
      for (const f of await ws.list()) expect(await ws.read(f)).not.toBeNull();
    }), { numRuns: 100 });
  });
});

describe("repo map", () => {
  it("finds exported symbols in several languages", () => {
    expect(extractSymbols("a.ts", "export function foo() {}\nexport const bar = 1;\nclass Baz {}\nexport default class Qux {}\n")).toEqual(["foo", "bar", "Baz", "Qux"]);
    expect(extractSymbols("a.py", "def f():\n  pass\nclass K:\n  def m(self): pass\nasync def g(): pass\n")).toEqual(["f", "K", "g"]);
    expect(extractSymbols("a.go", "func Run() {}\nfunc (s *S) Do() {}\ntype T struct{}\n")).toEqual(["Run", "Do", "T"]);
    expect(extractSymbols("a.rs", "pub fn go() {}\npub struct S;\nfn private() {}\n")).toEqual(["go", "S", "private"]);
    expect(extractSymbols("notes.txt", "export function nope() {}")).toEqual([]);
  });
  it("summarises the project within a budget and never reads secrets", async () => {
    const m = await buildRepoMap(new ShadowWorkspace(proj()));
    expect(m).toContain("src/math.ts: add, PI");
    expect(m).toContain("Key files: README.md, package.json");
    expect(m).toContain("4 files"); // .env is not counted or shown
    expect(m).not.toContain("SECRET");
    const files: Record<string, string> = {};
    for (let i = 0; i < 300; i++) files[`src/m${i}.ts`] = `export function f${i}() {}\n`;
    const big = await buildRepoMap(new ShadowWorkspace(MemoryProject.from(files)), 1500);
    expect(big.length).toBeLessThan(1800);
    expect(big).toContain("more source files not shown");
  });
});

describe("execution receipts", () => {
  const draft = (n: number, previous: string): ReceiptDraft => ({ v: 2, id: `r${n}`, task: "t", model: "m", startedAt: "2026-01-01T00:00:00Z", finishedAt: "2026-01-01T00:00:01Z", status: "finished", files: [], commands: [], steps: n, previous });
  async function chain(signer: ReceiptSigner, n: number) {
    const out = [];
    let prev = GENESIS_TAG;
    for (let i = 0; i < n; i++) { const r = await sealReceipt(signer, draft(i, prev)); out.push(r); prev = r.tag; }
    return out;
  }
  it("form a chain that verifies with no secret, and any change, removal or reordering breaks it", async () => {
    const signer = await generateReceiptSigner();
    const c = await chain(signer, 4);
    expect(await verifyReceiptChain(c)).toEqual({ ok: true, keyIds: [await keyThumbprint(signer.publicJwk)] });
    expect(await verifyReceiptChain([{ ...c[1]!, task: "edited" }, ...c.slice(2)])).toMatchObject({ ok: false });
    expect(await verifyReceiptChain([c[0]!, c[2]!, c[3]!])).toMatchObject({ ok: false, index: 1 });
    expect(await verifyReceiptChain([c[1]!, c[0]!, c[2]!])).toMatchObject({ ok: false });
    expect(await verifyReceiptChain([c[0]!, { ...c[1]!, steps: 99 }, c[2]!])).toMatchObject({ ok: false, index: 1 });
  });
  it("cannot be re-signed by someone else: swapping in another key changes the key id, and pinning a key id refuses it", async () => {
    const mine = await generateReceiptSigner();
    const theirs = await generateReceiptSigner();
    const c = await chain(mine, 2);
    const forged = { ...c[0]!, publicKey: theirs.publicJwk };
    expect(await verifyReceiptChain([forged])).toMatchObject({ ok: false, index: 0 });
    const other = await chain(theirs, 2);
    expect(await verifyReceiptChain(other)).toMatchObject({ ok: true });
    expect(await verifyReceiptChain(other, { trustedKeyIds: [await keyThumbprint(mine.publicJwk)] })).toMatchObject({ ok: false, index: 0 });
    expect(await verifyReceiptChain(c, { trustedKeyIds: [await keyThumbprint(mine.publicJwk)] })).toMatchObject({ ok: true });
  });
  it("is independent of key order in the JSON and has the documented tag shape and thumbprint", async () => {
    const signer = await generateReceiptSigner();
    const a = await sealReceipt(signer, draft(1, GENESIS_TAG));
    const reordered = Object.fromEntries(Object.entries(draft(1, GENESIS_TAG)).reverse()) as unknown as ReceiptDraft;
    const b = await sealReceipt(signer, reordered);
    expect(await verifyReceiptChain([b])).toMatchObject({ ok: true });
    expect(a.tag).toMatch(/^clt_[A-Za-z0-9_-]{86}$/);
    expect(a.keyId).toMatch(/^[A-Za-z0-9_-]{43}$/);
  });
});
