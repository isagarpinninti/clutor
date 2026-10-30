import { afterEach, describe, expect, it } from "vitest";
import { BackendManager, buildLlamaArgs, findLlamaServer, probeLlamaServer, type LlamaCapabilities } from "../src";
import { cleanup, tmp, writeFakeLlamaServer } from "./helpers";

afterEach(cleanup);
const caps = (flags: string[], flashAttnTakesValue = false): LlamaCapabilities => ({ flags: new Set(flags), flashAttnTakesValue, helpText: "" });
const ALL = ["--jinja", "--cache-type-k", "--cache-type-v", "--flash-attn", "--mlock"];
let port = 18080 + Math.floor(Math.random() * 1000);
const nextPort = (): number => port++;

describe("finding and probing llama-server", () => {
  it("finds the binary on PATH and returns null when it is absent", () => {
    const d = tmp();
    const bin = writeFakeLlamaServer(d);
    expect(findLlamaServer({ PATH: d })).toBe(bin);
    expect(findLlamaServer({ PATH: tmp() }, [])).toBe(findLlamaServer({ PATH: "" }, []));
    expect(findLlamaServer({ PATH: d }, [tmp()])).toBe(bin);
  });
  it("reads which options this build has, and whether --flash-attn takes a value", async () => {
    const d1 = tmp(), d2 = tmp();
    const old = await probeLlamaServer(writeFakeLlamaServer(d1, { flashValue: false }));
    expect([...old.flags].sort()).toEqual(["--cache-type-k", "--cache-type-v", "--flash-attn", "--jinja", "--mlock", "--no-mmap"]);
    expect(old.flashAttnTakesValue).toBe(false);
    const modern = await probeLlamaServer(writeFakeLlamaServer(d2, { flashValue: true }));
    expect(modern.flashAttnTakesValue).toBe(true);
    await expect(probeLlamaServer("/nonexistent/llama-server")).rejects.toThrow();
  });
});

describe("start-up arguments", () => {
  const base = { modelPath: "/m/model.gguf", port: 8081, contextTokens: 8192, kvBits: 16 as const };
  it("binds to localhost only and sets the model, port and context", () => {
    const { args } = buildLlamaArgs({ ...base, caps: caps(ALL) });
    expect(args.slice(0, 8)).toEqual(["-m", "/m/model.gguf", "--host", "127.0.0.1", "--port", "8081", "-c", "8192"]);
    expect(args).toContain("--jinja");
  });
  it("quantises the KV cache only if the build can, and adds flash attention in the form the build wants", () => {
    const a = buildLlamaArgs({ ...base, kvBits: 8, caps: caps(ALL, true) }).args.join(" ");
    expect(a).toContain("--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on");
    const b = buildLlamaArgs({ ...base, kvBits: 4, caps: caps(ALL, false) }).args.join(" ");
    expect(b).toContain("--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn");
    expect(b).not.toContain("--flash-attn on");
    const none = buildLlamaArgs({ ...base, kvBits: 8, caps: caps(["--jinja"]) });
    expect(none.args).not.toContain("--cache-type-k");
    expect(none.notes.join(" ")).toMatch(/cannot quantise the KV cache/);
    expect(buildLlamaArgs({ ...base, kvBits: 8, caps: caps(["--cache-type-k", "--cache-type-v"]) }).notes.join(" ")).toMatch(/flash attention/);
  });
  it("pins memory only when the whole model fits, and explains paging when it does not", () => {
    expect(buildLlamaArgs({ ...base, plan: { strategy: "fully-resident" }, caps: caps(ALL) }).args).toContain("--mlock");
    const big = buildLlamaArgs({ ...base, plan: { strategy: "dense-resident-experts-streamed" }, caps: caps(ALL) });
    expect(big.args).not.toContain("--mlock");
    expect(big.notes.join(" ")).toMatch(/paged in from disk/);
  });
  it("warns about missing tool-call support, passes threads and extra arguments through, and never invents a flag", () => {
    const r = buildLlamaArgs({ ...base, threads: 6, extraArgs: ["--temp", "0.1"], caps: caps([]) });
    expect(r.args).toEqual(expect.arrayContaining(["-t", "6", "--temp", "0.1"]));
    expect(r.args).not.toContain("--jinja");
    expect(r.notes.join(" ")).toMatch(/--jinja/);
  });
});

describe("supervising llama-server", () => {
  it("starts it, waits while the model loads, serves requests, and stops it", async () => {
    const bin = writeFakeLlamaServer(tmp(), { loadMs: 600 });
    const p = nextPort();
    const b = new BackendManager(p);
    const states: string[] = [];
    b.onState((s) => states.push(s));
    const t0 = Date.now();
    await b.start(bin, ["--port", String(p)]);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(550);
    expect(b.state).toBe("ready");
    const r = await fetch(`${b.baseUrl}/chat/completions`, { method: "POST", body: "{}" });
    expect(((await r.json()) as { choices: Array<{ message: { content: string } }> }).choices[0]!.message.content).toContain("from fake llama-server");
    await b.stop();
    expect(b.state).toBe("stopped");
    await expect(fetch(`http://127.0.0.1:${p}/health`)).rejects.toThrow();
    expect(states).toEqual(["starting", "ready", "stopped"]);
  });
  it("reports a crash with the last output instead of waiting forever", async () => {
    const b = new BackendManager(nextPort());
    await expect(b.start(writeFakeLlamaServer(tmp(), { crash: true }), ["--port", "1"], { readyTimeoutMs: 5000 })).rejects.toThrow(/failed to load model|exited/);
    expect(b.state).toBe("failed");
    expect(b.error).toMatch(/failed to load model/);
  });
  it("gives up when the model never finishes loading, and does not leave the process running", async () => {
    const p = nextPort();
    const b = new BackendManager(p);
    await expect(b.start(writeFakeLlamaServer(tmp(), { neverReady: true }), ["--port", String(p)], { readyTimeoutMs: 1200 })).rejects.toThrow(/did not finish loading/);
    await new Promise((r) => setTimeout(r, 300));
    await expect(fetch(`http://127.0.0.1:${p}/health`)).rejects.toThrow();
  });
  it("refuses to start twice, and reports a missing binary", async () => {
    const p = nextPort();
    const b = new BackendManager(p);
    await b.start(writeFakeLlamaServer(tmp()), ["--port", String(p)]);
    await expect(b.start("x", [])).rejects.toThrow(/already running/);
    await b.stop();
    const c = new BackendManager(nextPort());
    await expect(c.start("/nonexistent/llama-server", [], { readyTimeoutMs: 3000 })).rejects.toThrow(/could not start/);
  });
});
