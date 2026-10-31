import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { writeSyntheticGguf } from "@clutor/core/node";
import { main, type Io } from "../src";
import { cleanup, fakeLlm, put, tmp, tool, writeFakeLlamaServer, type FakeLlm } from "./helpers";

let llm: FakeLlm | undefined;
afterEach(async () => { await llm?.close(); llm = undefined; cleanup(); });

function harness(cwd: string, answers: string[] = []) {
  const out: string[] = [], err: string[] = [], asked: string[] = [];
  const io: Io = { cwd, out: (l) => out.push(l), err: (l) => err.push(l), prompt: async (q) => (asked.push(q), answers.shift() ?? "n") };
  return { io, out, err, asked, run: (...a: string[]) => main(a, io), text: () => out.join("\n"), errText: () => err.join("\n") };
}
const SPEC = { arch: "qwen2moe", name: "tiny-moe", layers: 3, embedding: 64, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 8, expertsUsed: 2, expertFfn: 64, denseFfn: 0, vocab: 128 };

describe("basics", () => {
  it("prints help and the version, and exits 2 for an unknown command or a bad flag", async () => {
    const h = harness(tmp());
    expect(await h.run("--help")).toBe(0);
    expect(h.text()).toMatch(/Usage: clutor <command>[\s\S]*code "<task>"/);
    expect(await h.run("--version")).toBe(0);
    expect(await h.run("nonsense")).toBe(2);
    expect(h.errText()).toMatch(/unknown command/);
    expect(await harness(tmp()).run("plan")).toBe(2);
    expect(await harness(tmp()).run("plan", "deepseek-v3", "--ram", "-4")).toBe(2);
    expect(await harness(tmp()).run("plan", "deepseek-v3", "--kv", "7")).toBe(2);
  });
  it("rejects a configuration with an unknown key", async () => {
    const d = tmp();
    put(d, "clutor.config.json", '{"surprise": true}');
    const h = harness(d);
    expect(await h.run("doctor")).toBe(1);
    expect(h.errText() + h.text()).toMatch(/surprise|Unrecognized/);
  });
});

describe("init, models, inspect", () => {
  it("creates a configuration and a models folder, and will not overwrite", async () => {
    const d = tmp(), h = harness(d);
    expect(await h.run("init")).toBe(0);
    expect(existsSync(join(d, "clutor.config.json"))).toBe(true);
    expect(existsSync(join(d, "models"))).toBe(true);
    expect(h.text()).toMatch(/This machine: /);
    const h2 = harness(d);
    expect(await h2.run("init")).toBe(1);
    expect(h2.errText()).toMatch(/already exists/);
  });
  it("lists the catalog and local model files, as text or JSON", async () => {
    const d = tmp();
    await harness(d).run("init");
    await writeSyntheticGguf(join(d, "models", "tiny.gguf"), SPEC);
    const h = harness(d);
    await h.run("models");
    expect(h.text()).toMatch(/deepseek-v3[\s\S]*tiny\.gguf/);
    const j = harness(d);
    await j.run("models", "--json");
    expect(JSON.parse(j.text()).local[0].file).toBe("tiny.gguf");
  });
  it("reads a model file's header, and says so clearly when the file is not a model", async () => {
    const d = tmp();
    await writeSyntheticGguf(join(d, "t.gguf"), SPEC);
    const h = harness(d);
    expect(await h.run("inspect", "t.gguf")).toBe(0);
    expect(h.text()).toMatch(/tiny-moe \(qwen2moe\), GGUF v3, \d+ tensors[\s\S]*3 layers, 8 experts per layer \(2 used per token\)[\s\S]*All tensors lie inside the file/);
    const j = harness(d);
    await j.run("inspect", "t.gguf", "--json");
    expect(JSON.parse(j.text())).toMatchObject({ layers: 3, experts: 8, expertsUsed: 2 });
    writeFileSync(join(d, "bad.gguf"), "not a model, just a long enough bit of plain text");
    const b = harness(d);
    expect(await b.run("inspect", "bad.gguf")).toBe(1);
    expect(b.errText()).toMatch(/GGUF|magic/);
    expect(await harness(d).run("inspect")).toBe(2);
    expect(await harness(d).run("inspect", "missing.gguf")).toBe(1);
  });
});

describe("plan and simulate", () => {
  it("prints where the memory goes and the disk-bound speed ceiling, and exits 1 when the model cannot fit", async () => {
    const h = harness(tmp());
    expect(await h.run("plan", "deepseek-v3", "--ram", "8", "--disk", "3")).toBe(0);
    expect(h.text()).toMatch(/Expert cache: \d+ of 14848 experts/);
    expect(h.text()).toMatch(/at most 0\.\d+ tokens\/s/);
    expect(h.text()).toMatch(/CL-101[\s\S]*CL-108/);
    const nofit = harness(tmp());
    expect(await nofit.run("plan", "llama-3.3-70b", "--quant", "F16", "--ram", "2")).toBe(1);
    expect(nofit.text()).toMatch(/DOES NOT FIT/);
  });
  it("plans a real file and says the figures come from it, not from the catalog", async () => {
    const d = tmp();
    await writeSyntheticGguf(join(d, "t.gguf"), SPEC);
    const j = harness(d);
    expect(await j.run("plan", "t.gguf", "--json", "--ram", "4")).toBe(0);
    const r = JSON.parse(j.text());
    expect(r.model.approximate).toBe(false);
    expect(r.plan.strategy).toBe("fully-resident");
    expect(r.gates[7].status).toBe("pass");
  });
  it("replays the synthetic trace and labels it as synthetic", async () => {
    const h = harness(tmp());
    expect(await h.run("simulate", "qwen3-30b-a3b", "--cache", "1", "--tokens", "60")).toBe(0);
    expect(h.text()).toMatch(/baseline \(LRU\)\s+clutor[\s\S]*cache hit rate[\s\S]*SYNTHETIC/);
    const j = harness(tmp());
    await j.run("simulate", "qwen3-30b-a3b", "--cache", "1", "--tokens", "60", "--json");
    const r = JSON.parse(j.text());
    expect(r.synthetic).toBe(true);
    expect(r.clutor.expertUses).toBe(60 * 48 * 8);
    expect(await harness(tmp()).run("simulate", "llama-3.3-70b")).toBe(2);
  });
  it("measures a real file", async () => {
    const d = tmp();
    await writeSyntheticGguf(join(d, "t.gguf"), SPEC);
    const j = harness(d);
    expect(await j.run("bench", "--file", "t.gguf", "--json")).toBe(0);
    const r = JSON.parse(j.text());
    expect(r.measurement.bytesPerSec).toBeGreaterThan(0);
    expect(r.measurement.likelyCold).toBe(false);
    const h = harness(d);
    await h.run("bench", "--file", "t.gguf");
    expect(h.text()).toMatch(/operating system's cache probably served/);
  });
});

describe("clutor code", () => {
  async function project(script: string[]) {
    llm = await fakeLlm(script);
    const d = tmp();
    put(d, "clutor.config.json", JSON.stringify({ backend: { kind: "openai", baseUrl: llm.url, model: "m" } }));
    put(d, "package.json", "{}"); put(d, "calc.js", "exports.add = (a, b) => a - b;\n");
    put(d, "t.js", "const { add } = require('./calc'); process.exit(add(2, 3) === 5 ? 0 : 1);\n");
    return d;
  }
  const FIX = [tool("edit_file", { path: "calc.js", old_text: "a - b", new_text: "a + b" }), tool("finish", { summary: "fixed" })];

  it("shows the diff and writes nothing unless told to", async () => {
    const d = await project(FIX);
    const h = harness(d);
    expect(await h.run("code", "fix add", "--dir", ".")).toBe(0);
    expect(h.text()).toMatch(/finished: fixed[\s\S]*-exports.add = \(a, b\) => a - b;[\s\S]*\+exports.add = \(a, b\) => a \+ b;/);
    expect(h.text()).toMatch(/Nothing was written/);
    expect(readFileSync(join(d, "calc.js"), "utf8")).toContain("a - b");
  });
  it("writes the changes with --apply, records a receipt, and can be rolled back", async () => {
    const d = await project(FIX);
    const h = harness(d);
    expect(await h.run("code", "fix add", "--apply")).toBe(0);
    expect(readFileSync(join(d, "calc.js"), "utf8")).toContain("a + b");
    const id = /clutor rollback (ckpt_[a-z0-9]+)/.exec(h.text())?.[1];
    expect(id).toBeDefined();
    const r = harness(d);
    expect(await r.run("receipts")).toBe(0);
    expect(r.text()).toMatch(/Receipt chain intact: 1 receipt[\s\S]*fix add/);
    const rb = harness(d);
    expect(await rb.run("rollback", id!)).toBe(0);
    expect(readFileSync(join(d, "calc.js"), "utf8")).toContain("a - b");
  });
  it("verifies receipts offline from a copy of the log, and only for a key you trust", async () => {
    const d = await project(FIX);
    await harness(d).run("code", "fix add", "--apply");
    const log = readFileSync(join(d, ".clutor/receipts.jsonl"), "utf8");
    const token = JSON.parse(log.trim().split("\n")[0]!).tag as string;
    const k = harness(d);
    expect(await k.run("receipt", "key")).toBe(0);
    const keyId = /Key id\s+(\S+)/.exec(k.text())?.[1];
    expect(keyId).toBeDefined();
    // A third party has only the copied log, never the private key.
    const elsewhere = await project(FIX);
    writeFileSync(join(elsewhere, "copy.jsonl"), log);
    const ok = harness(elsewhere);
    expect(await ok.run("receipt", "verify", "copy.jsonl", "--key-id", keyId!)).toBe(0);
    expect(ok.text()).toMatch(/Valid: 1 receipt[\s\S]*one you trust/);
    const wrong = harness(elsewhere);
    expect(await wrong.run("receipt", "verify", "copy.jsonl", "--key-id", "AAAA")).toBe(1);
    expect(wrong.errText()).toMatch(/not one you trust/);
    writeFileSync(join(elsewhere, "bad.jsonl"), log.replace('"steps":', '"steps":9'));
    const bad = harness(elsewhere);
    expect(await bad.run("receipt", "verify", "bad.jsonl")).toBe(1);
    const byToken = harness(d);
    expect(await byToken.run("receipt", "verify", token)).toBe(0);
  });
  it("asks before applying when it has a terminal, and before running project code", async () => {
    const d = await project([tool("run_command", { command: "node t.js" }), tool("edit_file", { path: "calc.js", old_text: "a - b", new_text: "a + b" }), tool("run_command", { command: "node t.js" }), tool("finish", { summary: "fixed and verified" })]);
    const h = harness(d, ["y", "y", "y"]);
    expect(await h.run("code", "fix add")).toBe(0);
    expect(h.asked).toHaveLength(3);
    expect(h.asked[0]).toMatch(/Run `node t.js`/);
    expect(h.asked[2]).toMatch(/Apply these changes/);
    expect(readFileSync(join(d, "calc.js"), "utf8")).toContain("a + b");
    const no = await project([tool("run_command", { command: "node t.js" }), tool("finish", { summary: "d" })]);
    const h2 = harness(no, ["n"]);
    await h2.run("code", "x");
    expect(h2.text()).toMatch(/did not approve/);
  });
  it("refuses to overwrite a file edited while the agent worked, and says nothing was written", async () => {
    const d = await project([tool("edit_file", { path: "calc.js", old_text: "a - b", new_text: "a + b" }), (() => { throw new Error("unused"); }) as never]);
    llm!.script[1] = () => { writeFileSync(join(d, "calc.js"), "// user edit\n"); return tool("finish", { summary: "done" }); };
    const h = harness(d);
    expect(await h.run("code", "fix add", "--apply")).toBe(1);
    expect(h.errText()).toMatch(/changed on disk[\s\S]*Nothing was written/);
    expect(readFileSync(join(d, "calc.js"), "utf8")).toBe("// user edit\n");
  });
  it("exits 2 with advice when no model is configured, and needs a task", async () => {
    const d = tmp();
    const h = harness(d);
    expect(await h.run("code", "do it")).toBe(2);
    expect(h.errText()).toMatch(/no model is configured/);
    expect(await harness(d).run("code")).toBe(2);
  });
  it("exits 1 and applies nothing when the agent gives up", async () => {
    const d = await project(["no tool here"]);
    const h = harness(d);
    expect(await h.run("code", "x", "--apply")).toBe(1);
    expect(h.text()).toMatch(/stuck/);
  });
});

describe("backend and doctor", () => {
  it("shows what the installed llama-server supports and the exact command it would run", async () => {
    const d = tmp(), bin = tmp();
    const fake = writeFakeLlamaServer(bin);
    await writeSyntheticGguf(join(d, "m.gguf"), SPEC);
    put(d, "clutor.config.json", JSON.stringify({ backend: { kind: "llama-server", llamaServer: { binary: fake, model: join(d, "m.gguf") } }, memory: { kvBits: 8 } }));
    const h = harness(d);
    expect(await h.run("backend")).toBe(0);
    expect(h.text()).toContain(fake);
    expect(h.text()).toMatch(/--jinja yes, --cache-type-k yes/);
    expect(h.text()).toMatch(/-m .*m\.gguf --host 127\.0\.0\.1 --port 8081 -c 8192 --jinja --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn --mlock/);
  });
  it("says what is missing when llama-server cannot be found", async () => {
    const d = tmp();
    const old = process.env["PATH"];
    process.env["PATH"] = tmp();
    try {
      const h = harness(d);
      if (!existsSync("/opt/homebrew/bin/llama-server") && !existsSync("/usr/local/bin/llama-server")) { expect(await h.run("backend")).toBe(1); expect(h.errText()).toMatch(/not found/); }
    } finally { process.env["PATH"] = old; }
  });
  it("checks the machine and configuration, failing only on real problems", async () => {
    const d = tmp();
    const j = harness(d);
    expect(await j.run("doctor", "--json")).toBe(0);
    const checks = JSON.parse(j.text()) as Array<{ level: string; text: string }>;
    expect(checks.some((c) => /RAM/.test(c.text))).toBe(true);
    put(d, "clutor.config.json", JSON.stringify({ backend: { kind: "openai" } }));
    const bad = harness(d);
    expect(await bad.run("doctor")).toBe(1);
    expect(bad.errText()).toMatch(/baseUrl is missing/);
  });
});

describe("downloads", () => {
  it("lists only models whose catalog entry exists, and downloads through a part file with a size check", async () => {
    const { DOWNLOADS, downloadFile } = await import("../src/models");
    const { CATALOG } = await import("@clutor/core");
    for (const d of DOWNLOADS) expect(CATALOG.some((c) => c.id === d.catalogId), d.id).toBe(true);
    const dir = tmp();
    const data = new Uint8Array(1000).fill(7);
    const ok = (async () => new Response(data, { headers: { "content-length": "1000" } })) as unknown as typeof fetch;
    expect(await downloadFile("http://x/y", join(dir, "a.gguf"), { fetch: ok, expectedBytes: 1000 })).toBe(1000);
    expect(readFileSync(join(dir, "a.gguf")).length).toBe(1000);
    expect(existsSync(join(dir, "a.gguf.part"))).toBe(false);
    await expect(downloadFile("http://x/y", join(dir, "b.gguf"), { fetch: ok, expectedBytes: 5000 })).rejects.toThrow(/not downloading/);
    const noLength = (async () => new Response(data)) as unknown as typeof fetch;
    await expect(downloadFile("http://x/y", join(dir, "c.gguf"), { fetch: noLength })).rejects.toThrow(/how large/);
    expect(existsSync(join(dir, "b.gguf"))).toBe(false);
  });
  it("resumes an interrupted download and verifies the SHA-256 of the whole file", async () => {
    const { downloadFile } = await import("../src/models");
    const { createHash } = await import("node:crypto");
    const dir = tmp();
    const data = new Uint8Array(5000).map((_, i) => (i * 31) % 251);
    const sha = createHash("sha256").update(data).digest("hex");
    const ranges: string[] = [];
    let calls = 0;
    const flaky = (async (_url: string, init?: RequestInit) => {
      calls++;
      const range = (init?.headers as Record<string, string> | undefined)?.["Range"];
      ranges.push(range ?? "none");
      if (!range) {   // first attempt: 2000 bytes, then the connection dies
        let n = 0;
        const body = new ReadableStream({ async pull(c) { if (n++ === 0) c.enqueue(data.slice(0, 2000)); else { await new Promise((r) => setTimeout(r, 30)); c.error(new Error("connection reset")); } } });
        return new Response(body, { headers: { "content-length": "5000" } });
      }
      const from = Number(/bytes=(\d+)-/.exec(range)?.[1]);
      return new Response(data.slice(from), { status: 206, headers: { "content-range": `bytes ${from}-4999/5000`, "content-length": String(5000 - from) } });
    }) as unknown as typeof fetch;
    const dest = join(dir, "r.gguf");
    await expect(downloadFile("http://x/y", dest, { fetch: flaky, expectedSha256: sha })).rejects.toThrow(/run the same command again to resume/);
    expect(existsSync(`${dest}.part`)).toBe(true);
    expect(await downloadFile("http://x/y", dest, { fetch: flaky, expectedSha256: sha })).toBe(5000);
    expect(ranges.length).toBe(2);
    expect(ranges[1]).toMatch(/^bytes=\d+-$/);
    expect(Buffer.compare(readFileSync(dest), Buffer.from(data))).toBe(0);
    expect(existsSync(`${dest}.part`)).toBe(false);
    expect(calls).toBe(2);
    // a wrong hash deletes the file instead of keeping a corrupt model
    const whole = (async () => new Response(data, { headers: { "content-length": "5000" } })) as unknown as typeof fetch;
    await expect(downloadFile("http://x/y", join(dir, "bad.gguf"), { fetch: whole, expectedSha256: "0".repeat(64) })).rejects.toThrow(/SHA-256/);
    expect(existsSync(join(dir, "bad.gguf"))).toBe(false);
    expect(existsSync(join(dir, "bad.gguf.part"))).toBe(false);
  });
  it("asks before downloading, and lists the choices", async () => {
    const h = harness(tmp(), ["n"]);
    expect(await h.run("pull")).toBe(0);
    expect(h.text()).toMatch(/olmoe-1b-7b-q4_k_m/);
    const h2 = harness(tmp(), ["n"]);
    expect(await h2.run("pull", "olmoe-1b-7b-q4_k_m")).toBe(0);
    expect(h2.asked[0]).toMatch(/Download .* from huggingface.co/);
    expect(h2.text()).toMatch(/Nothing downloaded/);
  });
});

describe("interactive session (clutor with no command)", () => {
  it("lists its commands, forwards ordinary commands, rejects unknown ones and leaves on /exit", async () => {
    const d = tmp();
    const h = harness(d, ["", "/help", "/tools", "/nope", "/status", "/verify npm test", "/verify", "/verify off", "/yes", "/cost", "/undo", "/history", "/exit"]);
    expect(await main([], h.io)).toBe(0);
    const t = h.text();
    expect(t).toMatch(/clutor interactive session/);
    expect(t).toMatch(/\/verify \[command\|off\][\s\S]*\/undo[\s\S]*\/plan <model>/);
    expect(t).toMatch(/read_file[\s\S]*run_command/);
    expect(h.errText()).toMatch(/unknown command \/nope/);
    expect(t).toMatch(/Host/); // /status ran the real status command
    expect(t).toMatch(/verify: npm test/);
    expect(t).toMatch(/verify: off/);
    expect(t).toMatch(/auto-approve ON/);
    expect(t).toMatch(/0 prompt \+ 0 generated tokens/);
    expect(t).toMatch(/Nothing to undo/);
    expect(t).toMatch(/No tasks yet/);
  });
  it("sends plain text to the coding agent and keeps going after an error", async () => {
    const d = tmp();
    const h = harness(d, ["add a readme", "/exit"]);
    expect(await main([], h.io)).toBe(0);
    expect(h.errText()).toMatch(/no model is configured/);
  });
  it("runs a task against a model, applies it on approval, remembers it, and /undo restores the files", async () => {
    const d = tmp();
    put(d, "a.txt", "old\n");
    llm = await fakeLlm([tool("write_file", { path: "a.txt", content: "new\n" }), tool("finish", { summary: "rewrote a.txt" })]);
    put(d, "clutor.config.json", JSON.stringify({ backend: { kind: "openai", baseUrl: llm.url, model: "fake" } }));
    const h = harness(d, ["rewrite a.txt", "y", "/history", "/undo", "/exit"]);
    expect(await main([], h.io)).toBe(0);
    expect(h.text()).toMatch(/Applied 1 files\. Undo with: \/undo/);
    expect(h.text()).toMatch(/1\. "rewrite a\.txt" -> finished: rewrote a\.txt/);
    expect(h.text()).toMatch(/Restored 1 files/);
    expect(readFileSync(join(d, "a.txt"), "utf8")).toBe("old\n");
  });
});

describe("bench --receipt", () => {
  it("signs a receipt whose start and finish are the real times around the run, and it verifies and shows its payload", async () => {
    const d = tmp();
    const moe = join(d, "m.gguf");
    await writeSyntheticGguf(moe, SPEC);
    const h = harness(d);
    const before = Date.now();
    expect(await h.run("bench", "--file", moe, "--cache-mb", "1", "--tokens", "40", "--receipt")).toBe(0);
    const after = Date.now();
    const log = readFileSync(join(d, ".clutor", "receipts.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l) as { tag: string; startedAt: string; finishedAt: string; task: string; previous: string; steps: number });
    expect(log).toHaveLength(1);
    const r = log[0]!;
    const a = Date.parse(r.startedAt), b = Date.parse(r.finishedAt);
    expect(a).toBeGreaterThanOrEqual(before);
    expect(b).toBeLessThanOrEqual(after);
    expect(b - a).toBeGreaterThan(0);
    expect(r.steps).toBe(3);
    expect(r.task).toMatch(/LRU, no prefetch: hit .*adaptive cache \+ prefetch: hit .*smart-route/);
    const v = harness(d);
    expect(await v.run("receipt", "verify", r.tag, d)).toBe(0);
    expect(v.text()).toMatch(/Decoded payload[\s\S]*"previous": "clt_genesis"[\s\S]*Signature tag: clt_/);
  }, 120_000);
  it("does not write a receipt unless asked", async () => {
    const d = tmp();
    const moe = join(d, "m.gguf");
    await writeSyntheticGguf(moe, SPEC);
    expect(await harness(d).run("bench", "--file", moe, "--cache-mb", "1", "--tokens", "40")).toBe(0);
    expect(existsSync(join(d, ".clutor", "receipts.jsonl"))).toBe(false);
  }, 120_000);
});
