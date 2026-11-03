import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { buildEngine, findEngine } from "../src/engine";
import { PassThrough } from "node:stream";
import { stripAnsi } from "../src/tui/ansi";
import { runTui } from "../src/tui/app";
import { NativeEngine } from "../src/tui/native";
import type { StreamEvent } from "../src/tui/stream";
import { ThinkSplitter, detectTemplate, renderChat } from "../src/tui/templates";

const WRITER = join(__dirname, "../../../c/test/tiny_olmoe.py");
let dir: string, bin: string;
const models: Record<string, string> = {};

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), "clutor-native-"));
  for (const arch of ["llama", "olmoe"]) { models[arch] = join(dir, `${arch}.gguf`); execFileSync("python3", [WRITER, "write", models[arch] as string, ...(arch === "olmoe" ? [] : ["--arch", arch])]); }
  bin = findEngine() ?? (await buildEngine());
}, 240_000);
afterAll(() => rmSync(dir, { recursive: true, force: true }));

async function collect(g: AsyncGenerator<StreamEvent>): Promise<StreamEvent[]> { const out: StreamEvent[] = []; for await (const e of g) out.push(e); return out; }

describe("chat templates", () => {
  const msgs = [{ role: "system" as const, content: "Be brief." }, { role: "user" as const, content: "Hi" }];
  it("recognises each family by the markers in the model's own template", () => {
    expect(detectTemplate("{{'<|im_start|>' + role}}")).toEqual({ family: "chatml", thinkTags: false });
    expect(detectTemplate("<|im_start|> ... <think>")).toEqual({ family: "chatml", thinkTags: true });
    expect(detectTemplate("<|start_header_id|>").family).toBe("llama3");
    expect(detectTemplate("<|start_of_role|>").family).toBe("granite");
    expect(detectTemplate("<|user|> <|assistant|>").family).toBe("olmo");
    expect(detectTemplate(undefined).family).toBe("plain");
  });
  it("renders ChatML, Llama 3, Granite and OLMo layouts exactly", () => {
    expect(renderChat({ family: "chatml", thinkTags: false }, msgs)).toBe("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    expect(renderChat({ family: "llama3", thinkTags: false }, msgs)).toBe("<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    expect(renderChat({ family: "granite", thinkTags: false }, msgs)).toBe("<|start_of_role|>system<|end_of_role|>Be brief.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
    expect(renderChat({ family: "olmo", thinkTags: false }, msgs)).toBe("<|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\n");
  });
  it("adds Qwen3's empty think block when thinking is off, and not when it is on", () => {
    const q3 = { family: "chatml" as const, thinkTags: true };
    expect(renderChat(q3, msgs, { thinking: false }).endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n")).toBe(true);
    expect(renderChat(q3, msgs, { thinking: true }).endsWith("<|im_start|>assistant\n")).toBe(true);
  });
  it("tells Qwen3.5's template (which opens the think block itself) from Qwen3's (which does not)", () => {
    const q35 = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}<|im_start|>";
    expect(detectTemplate(q35)).toEqual({ family: "chatml", thinkTags: true, opensThink: true });
    expect(detectTemplate("<|im_start|> <think> without an opening in the prompt")).toEqual({ family: "chatml", thinkTags: true });
    const info = detectTemplate(q35);
    const msgs2 = [{ role: "user" as const, content: "Hi" }];
    expect(renderChat(info, msgs2, { thinking: true }).endsWith("<|im_start|>assistant\n<think>\n")).toBe(true);
    expect(renderChat(info, msgs2, { thinking: false }).endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n")).toBe(true);
  });
  it("renders earlier replies the way they were generated, so the engine's cache stays valid across turns", () => {
    const info = { family: "chatml" as const, thinkTags: true, opensThink: true };
    const first = renderChat(info, [{ role: "user", content: "Hi" }], { thinking: false });
    const next = renderChat(info, [{ role: "user", content: "Hi" }, { role: "assistant", content: "Hello!" }, { role: "user", content: "Again" }], { thinking: false });
    expect(next.startsWith(`${first}Hello!<|im_end|>`)).toBe(true);
  });
  it("closes earlier assistant turns, so a longer conversation extends the shorter one", () => {
    const info = { family: "chatml" as const, thinkTags: false };
    const first = renderChat(info, msgs);
    const second = renderChat(info, [...msgs, { role: "assistant", content: "Hello!" }, { role: "user", content: "Again" }]);
    expect(second.startsWith(`${first}Hello!`)).toBe(true);
    expect(second).toContain("Hello!<|im_end|>\n<|im_start|>user\nAgain");
  });
  it("splits reasoning from the answer however the tags are cut up", () => {
    const text = "<think>let me see</think>The answer.";
    for (const size of [1, 2, 3, 7, 100]) {
      const s = new ThinkSplitter(true), got: Array<{ kind: string; text: string }> = [];
      for (let i = 0; i < text.length; i += size) got.push(...s.push(text.slice(i, i + size)));
      got.push(...s.end());
      expect(got.filter((g) => g.kind === "reasoning").map((g) => g.text).join("")).toBe("let me see");
      expect(got.filter((g) => g.kind === "content").map((g) => g.text).join("")).toBe("The answer.");
    }
    const plain = new ThinkSplitter(false);
    expect(plain.push("<think>x</think>")).toEqual([{ kind: "content", text: "<think>x</think>" }]);
  });
});

describe("the native engine as a chat backend", () => {
  it("streams a reply, then only sends what is new on the next turn", async () => {
    const eng = await NativeEngine.start({ bin, model: models["llama"] as string, ctx: 512 });
    try {
      expect(eng.ready.arch).toBe("llama");
      const m1 = [{ role: "user" as const, content: "the cat is in" }];
      const e1 = await collect(eng.stream(m1, { temp: 0, max: 8 }));
      const done1 = e1[e1.length - 1] as Extract<StreamEvent, { type: "done" }>;
      expect(done1.type).toBe("done");
      expect(done1.completionTokens).toBeGreaterThan(0);
      expect(eng.lastAppend.reset).toBe(false);
      const firstTokens = eng.lastAppend.tokens;
      const reply = e1.filter((e) => e.type === "content").map((e) => (e as { text: string }).text).join("");

      const m2 = [...m1, { role: "assistant" as const, content: reply }, { role: "user" as const, content: "and the rat" }];
      await collect(eng.stream(m2, { temp: 0, max: 4 }));
      expect(eng.lastAppend.reset).toBe(false);
      expect(eng.lastAppend.tokens).toBeLessThan(firstTokens + 10);

      // A conversation that does not extend the cache starts over, and says so.
      await collect(eng.stream([{ role: "user" as const, content: "something else entirely" }], { temp: 0, max: 2 }));
      expect(eng.lastAppend.reset).toBe(true);
    } finally { await eng.close(); }
  }, 60_000);

  it("gives the same tokens for the same conversation whether or not the cache was warm", async () => {
    const eng = await NativeEngine.start({ bin, model: models["llama"] as string, ctx: 512 });
    try {
      const m = [{ role: "user" as const, content: "the dog and the cat" }];
      const text = async () => (await collect(eng.stream(m, { temp: 0, max: 10 }))).filter((e) => e.type === "content").map((e) => (e as { text: string }).text).join("");
      const a = await text();
      await eng.reset();
      expect(await text()).toBe(a);
    } finally { await eng.close(); }
  }, 60_000);

  it("stops when asked and stays usable", async () => {
    const eng = await NativeEngine.start({ bin, model: models["llama"] as string, ctx: 512 });
    try {
      const ac = new AbortController();
      const events: StreamEvent[] = [];
      for await (const e of eng.stream([{ role: "user", content: "go on" }], { temp: 0, max: 40, signal: ac.signal })) { events.push(e); if (events.length === 3) ac.abort(); }
      const done = events[events.length - 1] as Extract<StreamEvent, { type: "done" }>;
      expect(done.type).toBe("done");
      // The tiny model can finish before the cancel arrives; the protocol test (c/test/chat_protocol.py) checks the early stop deterministically.
      expect(done.completionTokens ?? 0).toBeLessThanOrEqual(40);
      const again = await eng.complete([{ role: "user", content: "hello" }], { temp: 0, max: 3 });
      expect(again.completionTokens).toBeGreaterThan(0);
    } finally { await eng.close(); }
  }, 60_000);

  it("refuses a conversation that cannot fit, with advice, and survives", async () => {
    const eng = await NativeEngine.start({ bin, model: models["llama"] as string, ctx: 128 });
    try {
      await expect(collect(eng.stream([{ role: "user", content: "word ".repeat(400) }], { max: 2 }))).rejects.toThrow(/does not fit|no longer fits/);
      const ok = await eng.complete([{ role: "user", content: "hi" }], { temp: 0, max: 2 });
      expect(ok.completionTokens).toBeGreaterThan(0);
    } finally { await eng.close(); }
  }, 60_000);

  it("works as the coding agent's model client", async () => {
    const eng = await NativeEngine.start({ bin, model: models["olmoe"] as string, ctx: 512 });
    try {
      const llm = eng.asLlm("tiny");
      expect(llm.toolMode).toBe("json");
      const r = await llm.chat([{ role: "system", content: "x" }, { role: "user", content: "hello" }]);
      expect(typeof r.content).toBe("string");
      expect(r.usage?.completionTokens).toBeGreaterThan(0);
    } finally { await eng.close(); }
  }, 60_000);

  it("reports a clear error when the engine cannot load the model", async () => {
    await expect(NativeEngine.start({ bin, model: join(dir, "missing.gguf"), ctx: 128, readyTimeoutMs: 10_000 })).rejects.toThrow();
  }, 30_000);
});

describe("the interactive session on Clutor's own engine", () => {
  it("loads the model in the native engine, chats, shows the engine, and exits", async () => {
    const stdin = new PassThrough() as unknown as NodeJS.ReadStream;
    const stdout = new PassThrough() as unknown as NodeJS.WriteStream;
    (stdout as unknown as { columns: number }).columns = 110;
    let screen = "";
    stdout.on("data", (c) => (screen += String(c)));
    const home = mkdtempSync(join(tmpdir(), "clt-native-home-"));
    const done = runTui({ io: { out: () => undefined, err: () => undefined, cwd: dir }, version: "9.9.9", argv: ["--model", models["llama"] as string, "--ctx", "512"], stdin, stdout, color: false, home, run: async () => 0 });
    const send = async (s: string, ms = 600) => { stdin.emit("data", s); await new Promise((r) => setTimeout(r, ms)); };
    await new Promise((r) => setTimeout(r, 2500));
    await send("/system hi\r", 300);
    await send("the cat is\r", 1500);
    await send("/engine\r");
    await send("/exit\r", 1000);
    expect(await done).toBe(0);
    const text = stripAnsi(screen);
    expect(text).toContain("Clutor engine");
    expect(text).not.toContain("llama-server");
    expect(text).toMatch(/\d+ tokens · [\d.]+ tok\/s/);
    expect(text).toContain("engine: Clutor engine");
    expect(text).toMatch(/llama · \d+ layers · \d+ threads · context 64 · prompt format plain/);
    expect(text).toMatch(/last turn sent \d+ new tokens/);
    rmSync(home, { recursive: true, force: true });
  }, 60_000);
});
