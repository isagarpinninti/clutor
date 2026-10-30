import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { writeSyntheticGguf } from "@clutor/core/node";
import { verifyReceiptChain } from "@clutor/core";
import { readReceipts, verifyReceiptLog } from "@clutor/core/node";
import { BackendManager } from "../src";
import { cleanup, start, tmp, tool, writeFakeLlamaServer, type Running } from "./helpers";

let s: Running | undefined;
afterEach(async () => { await s?.stop(); s = undefined; cleanup(); });
const noTok = { token: false } as const;

describe("who may talk to the server", () => {
  it("only answers on localhost host names, which is the defence against DNS rebinding", async () => {
    s = await start();
    const { request } = await import("node:http");
    const status = await new Promise<number>((resolve) => { const req = request(`${s!.base}/api/health`, { headers: { host: "evil.example.com" } }, (res) => resolve(res.statusCode ?? 0)); req.end(); });
    expect(status).toBe(421);
    expect((await s.get("/api/health", noTok)).status).toBe(200);
  });

  it("refuses browser requests from other sites, allows the app and listed origins, and answers preflights", async () => {
    s = await start({ corsOrigins: ["https://notes.example.com"] });
    expect((await s.get("/api/health", { ...noTok, headers: { origin: "https://evil.example.com" } })).status).toBe(403);
    const app = await s.get("/api/health", { ...noTok, headers: { origin: "app://clutor" } });
    expect(app.status).toBe(200);
    expect(app.headers.get("access-control-allow-origin")).toBe("app://clutor");
    const listed = await s.get("/api/health", { ...noTok, headers: { origin: "https://notes.example.com" } });
    expect(listed.headers.get("access-control-allow-origin")).toBe("https://notes.example.com");
    const pre = await fetch(`${s.base}/api/agent/run`, { method: "OPTIONS", headers: { origin: "app://clutor" } });
    expect(pre.status).toBe(204);
    expect(pre.headers.get("access-control-allow-headers")).toMatch(/authorization/);
    expect((await fetch(`${s.base}/api/agent/run`, { method: "OPTIONS", headers: { origin: "https://evil.example.com" } })).status).toBe(403);
  });

  it("needs the token for everything that touches files, processes or the model's lifecycle", async () => {
    s = await start();
    for (const [m, p] of [["GET", "/api/models"], ["GET", "/api/roots"], ["POST", "/api/roots"], ["GET", "/api/backend"], ["POST", "/api/backend/stop"], ["POST", "/api/agent/run"], ["POST", "/api/mcp"], ["GET", "/api/receipts?dir=/"], ["GET", "/api/agent/task_abc"], ["POST", "/api/agent/task_abc/apply"]] as const) {
      const r = m === "GET" ? await s.get(p, noTok) : await s.post(p, {}, noTok);
      expect(r.status, `${m} ${p}`).toBe(401);
    }
    expect((await s.get("/api/roots", { token: "wrong-token" })).status).toBe(401);
    expect((await s.get("/api/roots", { headers: { "x-clutor-token": s.token }, token: false })).status).toBe(200);
    expect((await s.get("/api/roots")).status).toBe(200);
  });

  it("leaves the planning and model API open to local tools without a token, and returns JSON errors for the rest", async () => {
    s = await start();
    expect((await s.get("/api/hardware", noTok)).body.totalRamBytes).toBeGreaterThan(0);
    expect((await s.get("/api/catalog", noTok)).body.models.map((m: { id: string }) => m.id)).toContain("deepseek-v3");
    expect((await s.post("/api/nope", {}, noTok)).status).toBe(401);
    expect((await s.post("/api/nope", {})).status).toBe(404);
    expect((await s.post("/api/plan", "{not json", noTok)).status).toBe(400);
    const big = await fetch(`${s.base}/api/plan`, { method: "POST", headers: { "content-type": "application/json" }, body: "x".repeat(9 * 1024 * 1024) });
    expect(big.status).toBe(413);
  });
});

describe("planning and simulation over HTTP", () => {
  it("plans a catalog model without a token, and refuses to read a file path without one", async () => {
    s = await start();
    const r = await s.post("/api/plan", { model: "deepseek-v3", hostRamGiB: 16, ramCeilingGiB: 8 }, noTok);
    expect(r.status).toBe(200);
    expect(r.body.model).toMatchObject({ id: "deepseek-v3@Q4_K_M", approximate: true });
    expect(r.body.plan.fits).toBe(true);
    expect(r.body.plan.speed[0].maxTokensPerSec).toBeLessThan(0.5);
    expect(r.body.gates.map((g: { id: string }) => g.id)).toHaveLength(8);
    expect((await s.post("/api/plan", { model: "/etc/passwd.gguf" }, noTok)).status).toBe(401);
    expect((await s.post("/api/plan", { model: "nope" }, noTok)).status).toBe(400);
    expect((await s.post("/api/plan", { model: "deepseek-v3@Q9" }, noTok)).body.error).toMatch(/unknown quantisation/);
    expect((await s.post("/api/plan", { model: "deepseek-v3", hostRamGiB: -1 }, noTok)).status).toBe(400);
    expect((await s.post("/api/plan", { model: "deepseek-v3", extra: 1 }, noTok)).status).toBe(400);
  });

  it("plans a real .gguf file when given the token, and a missing file is a clean error", async () => {
    s = await start();
    const path = join(tmp(), "m.gguf");
    await writeSyntheticGguf(path, { arch: "qwen2moe", name: "tiny", layers: 3, embedding: 64, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 8, expertsUsed: 2, expertFfn: 64, denseFfn: 0, vocab: 128 });
    const r = await s.post("/api/plan", { model: path, hostRamGiB: 16 });
    expect(r.status).toBe(200);
    expect(r.body.model).toMatchObject({ source: "file", approximate: false, layers: 3, experts: 8 });
    expect(r.body.plan.strategy).toBe("fully-resident");
    expect(r.body.gates.find((g: { id: string }) => g.id === "CL-108").status).toBe("pass");
    expect((await s.post("/api/plan", { model: join(tmp(), "gone.gguf") })).status).toBe(400);
    const bad = join(tmp(), "bad.gguf");
    writeFileSync(bad, "this is not a gguf file at all, just some text that is long enough");
    const r2 = await s.post("/api/plan", { model: bad });
    expect(r2.status).toBe(400);
    expect(r2.body.error).toMatch(/GGUF/i);
  });

  it("simulates baseline against clutor on a synthetic trace, with limits on how much work a request can ask for", async () => {
    s = await start();
    const r = await s.post("/api/simulate", { model: "qwen3-30b-a3b", cacheGiB: 1, tokens: 120 }, noTok);
    expect(r.status).toBe(200);
    expect(r.body.synthetic).toBe(true);
    expect(r.body.clutor.expertUses).toBe(120 * 48 * 8);
    expect(r.body.clutor.hitRate).toBeGreaterThan(r.body.baseline.hitRate);
    expect((await s.post("/api/simulate", { model: "qwen3-30b-a3b", cacheGiB: 1, tokens: 100000 }, noTok)).status).toBe(400);
    expect((await s.post("/api/simulate", { model: "llama-3.3-70b", cacheGiB: 1 }, noTok)).status).toBe(400);
  });
});

describe("OpenAI and Ollama compatible model API", () => {
  it("proxies chat completions to the backend, including streams, and uses the configured model name", async () => {
    s = await start({}, ["Hello there friend"]);
    const r = await s.post("/v1/chat/completions", { model: "anything", messages: [{ role: "user", content: "hi" }] }, noTok);
    expect(r.body.choices[0].message.content).toBe("Hello there friend");
    expect(s.llm.requests[0]!["model"]).toBe("fake-model");
    const st = await s.post("/v1/chat/completions", { messages: [{ role: "user", content: "hi" }], stream: true }, noTok);
    expect(st.headers.get("content-type")).toMatch(/event-stream/);
    expect(st.text).toContain('"content":"Hello "');
    expect(st.text.trim().endsWith("data: [DONE]")).toBe(true);
    expect((await s.get("/v1/models", noTok)).body.data[0].id).toBe("fake-model");
  });

  it("says plainly when no model is configured", async () => {
    s = await start({}, [], { withBackend: false });
    const r = await s.post("/v1/chat/completions", { messages: [] }, noTok);
    expect(r.status).toBe(503);
    expect(r.body.error.message).toMatch(/no model backend/);
  });

  it("speaks Ollama: lists models, and answers /api/chat and /api/generate as one reply or as newline-delimited JSON", async () => {
    s = await start({}, ["one two three"]);
    expect((await s.get("/api/tags", noTok)).body.models[0].name).toBe("fake-model");
    const chat = await s.post("/api/chat", { model: "m", stream: false, messages: [{ role: "user", content: "hi" }] }, noTok);
    expect(chat.body).toMatchObject({ model: "m", message: { role: "assistant", content: "one two three" }, done: true, prompt_eval_count: 7, eval_count: 3 });
    const gen = await s.post("/api/generate", { prompt: "hi", system: "be brief", stream: false, options: { temperature: 0.1, num_predict: 5 } }, noTok);
    expect(gen.body.response).toBe("one two three");
    const sent = s.llm.requests.at(-1)!;
    expect(sent["messages"]).toEqual([{ role: "system", content: "be brief" }, { role: "user", content: "hi" }]);
    expect(sent["temperature"]).toBe(0.1);
    expect(sent["max_tokens"]).toBe(5);
    const stream = await s.post("/api/chat", { messages: [{ role: "user", content: "hi" }] }, noTok);
    expect(stream.headers.get("content-type")).toBe("application/x-ndjson");
    const lines = stream.text.trim().split("\n").map((l) => JSON.parse(l));
    expect(lines.slice(0, -1).map((l) => l.message.content).join("")).toBe("one two three");
    expect(lines.at(-1)).toMatchObject({ done: true, done_reason: "stop", prompt_eval_count: 7, eval_count: 3 });
  });

  it("forwards an API key to the backend from the environment, never from the file", async () => {
    process.env["CLUTOR_TEST_KEY"] = "sk-test";
    try {
      const llm = await (await import("./helpers")).fakeLlm(["hi"]);
      s = await start({ backend: { kind: "openai", model: "m", apiKeyEnv: "CLUTOR_TEST_KEY", baseUrl: llm.url, llamaServer: { port: 8081, extraArgs: [] } } });
      await s.post("/v1/chat/completions", { messages: [] }, noTok);
      expect(llm.headers[0]!["authorization"]).toBe("Bearer sk-test");
      await llm.close();
    } finally { delete process.env["CLUTOR_TEST_KEY"]; }
  });
});

describe("folders the server may work in", () => {
  it("refuses a folder nobody opened, accepts one that was registered, and resolves links before deciding", async () => {
    s = await start({}, [tool("finish", { summary: "d" })]);
    const proj = tmp();
    expect((await s.post("/api/agent/run", { dir: proj, task: "x" })).status).toBe(403);
    expect((await s.post("/api/roots", { dir: proj })).status).toBe(200);
    expect((await s.get("/api/roots")).body.roots).toHaveLength(1);
    const run = await s.post("/api/agent/run", { dir: proj, task: "x" });
    expect(run.status).toBe(202);
    expect(run.body.id).toMatch(/^task_[a-f0-9]+$/);
    expect((await s.post("/api/agent/run", { dir: "/", task: "x" })).status).toBe(400);
    expect((await s.post("/api/roots", { dir: "/" })).status).toBe(400);
    expect((await s.post("/api/roots", { dir: join(proj, "missing") })).status).toBe(400);
    expect((await s.post("/api/agent/run", { dir: join(proj, "..", "elsewhere"), task: "x" })).status).toBeGreaterThanOrEqual(400);
  });
});

describe("backend lifecycle and disk benchmark", () => {
  it("starts the model on request without blocking, reports its state, serves chat through it, and stops it", async () => {
    const bin = writeFakeLlamaServer(tmp(), { loadMs: 500 });
    const port = 19000 + Math.floor(Math.random() * 500);
    const backend = new BackendManager(port);
    s = await start({ backend: { kind: "llama-server", llamaServer: { port, extraArgs: [] } } }, [], { extra: { backend, startBackend: async () => backend.start(bin, ["--port", String(port)]) } });
    expect((await s.get("/api/backend")).body.state).toBe("stopped");
    expect((await s.post("/v1/chat/completions", { messages: [] }, noTok)).status).toBe(503);
    const t0 = Date.now();
    const started = await s.post("/api/backend/start");
    expect(started.status).toBe(202);
    expect(Date.now() - t0).toBeLessThan(400); // it answers immediately while the model loads
    expect((await s.get("/api/backend")).body.state).toBe("starting");
    await new Promise<void>((resolve, reject) => { const t = setInterval(async () => { if ((await s!.get("/api/backend")).body.state === "ready") { clearInterval(t); resolve(); } }, 100); setTimeout(() => reject(new Error("never ready")), 8000); });
    const chat = await s.post("/v1/chat/completions", { messages: [{ role: "user", content: "hi" }] }, noTok);
    expect(chat.body.choices[0].message.content).toContain("from fake llama-server");
    expect((await s.get("/api/health", noTok)).body.backend).toEqual({ kind: "llama-server", state: "ready" });
    expect((await s.post("/api/backend/stop")).body.state).toBe("stopped");
  }, 20000);

  it("explains that it cannot start a model when none is configured", async () => {
    s = await start();
    const r = await s.post("/api/backend/start");
    expect(r.status).toBe(409);
    expect(r.body.error).toMatch(/llama-server/);
  });

  it("measures real reads of a given model file, and refuses without the token", async () => {
    s = await start();
    const path = join(tmp(), "m.gguf");
    await writeSyntheticGguf(path, { arch: "qwen2moe", name: "t", layers: 3, embedding: 64, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 8, expertsUsed: 2, expertFfn: 64, denseFfn: 0, vocab: 128 });
    expect((await s.post("/api/bench", { model: path }, noTok)).status).toBe(401);
    const r = await s.post("/api/bench", { model: path });
    expect(r.status).toBe(200);
    expect(r.body.synthetic).toBe(false);
    expect(r.body.measurement.bytesPerSec).toBeGreaterThan(0);
    expect(r.body.measurement.likelyCold).toBe(false);
    expect((await s.post("/api/bench", { model: join(tmp(), "none.gguf") })).status).toBe(400);
  });
});


describe("POST /v1/decide", () => {
  const firstToken = (probs: Record<string, number>) => ({ content: [{ token: Object.keys(probs)[0], top_logprobs: Object.entries(probs).map(([token, p]) => ({ token, logprob: Math.log(p) })) }] });

  it("returns the first-token probabilities of each choice and a signed receipt that chains in the project's log", async () => {
    const root = tmp();
    s = await start({}, [{ content: "B", logprobs: firstToken({ B: 0.7, A: 0.2, C: 0.05, D: 0.05 }) }], { roots: [root] });
    const body = { prompt: "PR #42 modifies auth middleware without unit tests", choices: ["approve", "request_changes", "block"] };
    const r = await s.post("/v1/decide", body);
    expect(r.status).toBe(200);
    expect(r.body.choice).toBe("request_changes");
    expect(r.body.probabilities.request_changes).toBeCloseTo(0.7 / 0.95, 6);
    expect(Object.values(r.body.probabilities as Record<string, number>).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(r.body.latencyMs).toBeGreaterThanOrEqual(0);
    expect(r.body.receipt.tag).toMatch(/^clt_/);
    expect(r.body.receipt.task).toContain("request_changes");
    expect(r.body.receipt.task).not.toContain("auth middleware");
    const asked = (s.llm.requests[0]!["messages"] as Array<{ content: string }>)[0]!.content;
    expect(asked).toContain("A) approve");
    expect(s.llm.requests[0]).toMatchObject({ max_tokens: 1, logprobs: true, model: "fake-model" });
    const again = await s.post("/v1/decide", body);
    expect(again.body.receipt.previous).toBe(r.body.receipt.tag);
    const log = await readReceipts(root);
    expect(log).toHaveLength(2);
    expect((await verifyReceiptChain(log)).ok).toBe(true);
    expect((await verifyReceiptLog(root)).ok).toBe(true);
  });

  it("needs the token, checks the body, and says so when the model gives no log-probabilities", async () => {
    s = await start({}, ["B"], { roots: [tmp()] });
    expect((await s.post("/v1/decide", { prompt: "x", choices: ["a", "b"] }, noTok)).status).toBe(401);
    expect((await s.post("/v1/decide", { prompt: "x", choices: ["only"] })).status).toBe(400);
    expect((await s.post("/v1/decide", { prompt: "x", choices: ["a", "a"] })).status).toBe(400);
    expect((await s.post("/v1/decide", { prompt: "x", choices: ["a", "b"], extra: 1 })).status).toBe(400);
    const r = await s.post("/v1/decide", { prompt: "x", choices: ["a", "b"] });
    expect(r.status).toBe(502);
    expect(r.body.error ?? r.text).toMatch(/log-probabilities/);
  });

  it("answers 503 when no model is configured", async () => {
    s = await start({}, [], { withBackend: false });
    expect((await s.post("/v1/decide", { prompt: "x", choices: ["a", "b"] })).status).toBe(503);
  });
});
