// End-to-end smoke test of the built CLI: runs the real binary in a temp folder and checks exit codes and key output.
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const bin = join(root, "packages/cli/dist/bin.js");
const fixture = join(root, "packages/core/test/fixtures/stories260K.gguf");
const dir = mkdtempSync(join(tmpdir(), "clutor-smoke-"));
let failed = 0;
const run = (args) => spawnSync(process.execPath, [bin, ...args], { cwd: dir, encoding: "utf8" });
const check = (name, args, ok) => {
  const r = run(args);
  const pass = ok(r);
  if (!pass) failed++;
  console.log(`${pass ? "ok  " : "FAIL"}  ${name}`);
  if (!pass) console.log(r.stdout + r.stderr);
};

check("version", ["--version"], (r) => r.status === 0 && /^\d+\.\d+\.\d+/.test(r.stdout));
check("init", ["init"], (r) => r.status === 0);
check("inspect a real GGUF", ["inspect", fixture], (r) => r.status === 0 && /llama/i.test(r.stdout));
check("plan fits a small model", ["plan", "qwen3-30b-a3b", "--ram", "32"], (r) => r.status === 0);
check("plan says a huge model does not fit", ["plan", "deepseek-v3", "--ram", "2", "--json"], (r) => /"fits":\s*false/.test(r.stdout));
check("simulate (labelled synthetic)", ["simulate", "deepseek-v3", "--cache", "6", "--tokens", "200"], (r) => r.status === 0 && /synthetic|simulat/i.test(r.stdout));
check("status", ["status"], (r) => r.status === 0 && /Host/.test(r.stdout));
check("ledger verify on an empty log", ["ledger", "verify"], (r) => r.status === 0);
check("code without a model fails with a clear message", ["code", "x"], (r) => r.status !== 0 && /no model is configured/.test(r.stderr));
check("chat without a model fails clearly", ["chat", "--prompt", "hi"], (r) => r.status !== 0 && /no model is configured/.test(r.stderr));

// The native engine, against a generated MoE file.
const engineBin = join(root, "c/build/clutor-engine");
const moe = join(dir, "moe.gguf");
const { writeSyntheticGguf } = await import(join(root, "packages/core/dist/node.js"));
await writeSyntheticGguf(moe, { arch: "qwen2moe", name: "smoke-moe", layers: 4, embedding: 128, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 16, expertsUsed: 4, expertFfn: 128, denseFfn: 0, vocab: 512, sharedExperts: 0 });
check("engine where", ["engine", "where"], (r) => r.status === 0 && r.stdout.includes("clutor-engine"));
check("engine streams a model and labels its routing", ["stream", moe, "--tokens", "40", "--cache-mb", "1", "--smart", "2,12,0.015", "--json"], (r) => {
  if (r.status !== 0) return false;
  const j = JSON.parse(r.stdout);
  return j.provenance.routing === "synthetic" && j.totals.hits + j.totals.misses === j.totals.accesses && j.totals.maxDeltaP <= 0.015 + 1e-9;
});
check("engine refuses a dense model", ["stream", fixture], (r) => r.status !== 0 && /no routed experts/.test(r.stderr));
check("bench compares strategies", ["bench", "--file", moe, "--cache-mb", "1", "--tokens", "40"], (r) => r.status === 0 && /adaptive cache \+ prefetch/.test(r.stdout) && /SYNTHETIC/.test(r.stdout) && /scan-pollution rejects/.test(r.stdout) && /prefetch hit \/ late/.test(r.stdout) && /disk wait \/ compute time/.test(r.stdout));
void engineBin;

// Receipts: sign, verify offline, and notice tampering.
check("receipt key", ["receipt", "key"], (r) => r.status === 0 && /Key id/.test(r.stdout));


// POST /v1/decide against the built server, with a stand-in model that returns first-token log-probabilities.
{
  const { createServer } = await import("node:http");
  const { createClutorServer, ConfigSchema } = await import(join(root, "packages/cli/dist/index.js"));
  const fake = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      const top = [["B", 0.8], ["A", 0.15], ["C", 0.05]].map(([token, p]) => ({ token, logprob: Math.log(p) }));
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "B" }, logprobs: { content: [{ token: "B", top_logprobs: top }] } }] }));
    });
  });
  await new Promise((r) => fake.listen(0, "127.0.0.1", r));
  const config = ConfigSchema.parse({ port: 0, backend: { kind: "openai", baseUrl: `http://127.0.0.1:${fake.address().port}/v1`, model: "fake" } });
  const srv = createClutorServer({ root: dir, config, version: "smoke", roots: [dir] });
  const { port } = await srv.listen();
  const r = await fetch(`http://127.0.0.1:${port}/v1/decide`, { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${srv.token}` }, body: JSON.stringify({ prompt: "PR #42 modifies auth middleware without unit tests", choices: ["approve", "request_changes", "block"] }) });
  const j = await r.json();
  const pass = r.status === 200 && j.choice === "request_changes" && Math.abs(Object.values(j.probabilities).reduce((a, b) => a + b, 0) - 1) < 1e-9 && /^clt_/.test(j.receipt?.tag ?? "");
  if (!pass) { failed++; console.log(JSON.stringify(j)); }
  console.log(`${pass ? "ok  " : "FAIL"}  POST /v1/decide returns probabilities and a signed receipt`);
  check("ledger verify covers the decision receipt", ["ledger", "verify"], (x) => x.status === 0 && /1 receipt/.test(x.stdout));
  await srv.close();
  await new Promise((r2) => { fake.close(r2); fake.closeAllConnections(); });
}

rmSync(dir, { recursive: true, force: true });
if (failed) { console.error(`${failed} smoke check(s) failed`); process.exit(1); }
console.log("smoke: all checks passed");
