import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { main, type Io } from "../src";
import { buildEngine, findEngine } from "../src/engine";

// A tiny runnable OLMoE (random K-quant weights) written by the same script the C golden test uses, so these tests need no model download.
const WRITER = join(__dirname, "../../../c/test/tiny_olmoe.py");
let dir: string;
let model: string;

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), "clutor-generate-"));
  model = join(dir, "tiny.gguf");
  execFileSync("python3", [WRITER, "write", model]);
  writeFileSync(join(dir, "text.txt"), "the cat is in the room and a rat is on the mat. we are in the car on the road. ".repeat(3));
  if (!findEngine()) await buildEngine();
}, 180_000);
afterAll(() => rmSync(dir, { recursive: true, force: true }));

function harness() {
  const out: string[] = [], err: string[] = [], written: string[] = [];
  const io: Io = { cwd: dir, out: (l) => out.push(l), err: (l) => err.push(l), write: (s) => written.push(s) };
  return { run: (...a: string[]) => main(a, io), out: () => out.join("\n"), err: () => err.join("\n"), written: () => written.join("") };
}

describe("clutor generate and clutor quality", () => {
  it("generates text with the real forward pass and echoes the prompt", async () => {
    const h = harness();
    expect(await h.run("generate", model, "the cat", "--n", "4")).toBe(0);
    expect(h.written()).toContain("the cat");
    expect(h.err()).toMatch(/tokens fed/);
  }, 60_000);

  it("gives the same text when the experts are streamed through a small cache", async () => {
    const a = harness(), b = harness();
    expect(await a.run("generate", model, "the cat is in", "--n", "6")).toBe(0);
    expect(await b.run("generate", model, "the cat is in", "--n", "6", "--cache-mb", "1", "--io-threads", "2")).toBe(0);
    expect(b.written()).toBe(a.written());
    expect(b.err()).toMatch(/streaming: cache/);
  }, 60_000);

  it("refuses --smart without a cache, and rejects bad flags before opening the model", async () => {
    const h = harness();
    expect(await h.run("generate", model, "the cat", "--n", "2", "--smart", "2,4,0.1")).toBe(1);
    expect(h.err()).toMatch(/--smart needs --cache-mb/);
    expect(await harness().run("generate", model, "x", "--smart", "banana")).toBe(2);
    expect(await harness().run("generate", model, "x", "--policy", "random")).toBe(2);
    expect(await harness().run("generate", model)).toBe(2);
    expect(await harness().run("generate", join(dir, "missing.gguf"), "hi")).toBe(2);
  }, 60_000);

  it("quality: a streamed run without smart-route equals exact routing; with it, the report says how far it moved", async () => {
    const plain = harness();
    expect(await plain.run("quality", model, "text.txt", "--cache-mb", "1", "--max-tokens", "24", "--json")).toBe(0);
    const p = JSON.parse(plain.out()) as { perplexityRatio: number; meanKL: number; top1Agreement: number; substitutions: number };
    expect(p.perplexityRatio).toBeCloseTo(1, 6);
    expect(p.meanKL).toBeCloseTo(0, 6);
    expect(p.top1Agreement).toBe(1);
    expect(p.substitutions).toBe(0);
    const smart = harness();
    expect(await smart.run("quality", model, "text.txt", "--cache-mb", "1", "--max-tokens", "24", "--smart", "0,8,1.0")).toBe(0);
    expect(smart.out()).toMatch(/MEASURED {2}perplexity[\s\S]*smart-route used a resident expert/);
    expect(await harness().run("quality", model, "text.txt")).toBe(2);
  }, 120_000);
});
