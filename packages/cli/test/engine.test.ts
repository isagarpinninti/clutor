import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { buildModelLayout, expertSizeBytes, parseGguf } from "@clutor/core";
import { fileReader, writeSyntheticGguf } from "@clutor/core/node";
import { buildEngine, engineInfo, engineStream, findEngine } from "../src/engine";

const SPEC = { arch: "qwen2moe", name: "engine-test", layers: 5, embedding: 128, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 16, expertsUsed: 4, expertFfn: 128, denseFfn: 0, vocab: 512, sharedExperts: 0 };
const FIXTURE = join(__dirname, "../../core/test/fixtures/stories260K.gguf");

let dir: string;
let bin: string | undefined;
let model: string;

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), "clutor-engine-"));
  model = join(dir, "moe.gguf");
  await writeSyntheticGguf(model, SPEC);
  bin = findEngine();
  if (!bin) { try { bin = await buildEngine(); } catch { bin = undefined; } }
}, 180_000);
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// These need the native engine and fail loudly without it: a missing C compiler must not look like a pass.
describe("native engine", () => {
  const need = (): string => { if (!bin) throw new Error("the native engine could not be built here (no C compiler?)"); return bin; };

  it("reads a GGUF file the same way the TypeScript reader does", async () => {
    const r = await fileReader(model);
    const layout = buildModelLayout(await parseGguf(r));
    await r.close();
    const info = await engineInfo(need(), model);
    expect(info).toMatchObject({ arch: "qwen2moe", layers: layout.layerCount, experts: layout.expertCount, expertsUsed: layout.expertsUsed, totalBytes: layout.totalBytes, residentBytes: layout.residentBytes, maxExpertBytes: expertSizeBytes(layout), unsizedTensors: layout.unsizedTensors });
    expect(info.layerTable).toHaveLength(layout.layerCount);
    layout.layers.forEach((l, i) => expect(info.layerTable[i]).toMatchObject({ denseBytes: l.denseBytes, expertBytes: l.expertBytes, routerBytes: l.routerBytes }));
  });

  it("also agrees on a real llama.cpp-made file", async () => {
    const r = await fileReader(FIXTURE);
    const layout = buildModelLayout(await parseGguf(r));
    await r.close();
    const info = await engineInfo(need(), FIXTURE);
    expect(info).toMatchObject({ arch: "llama", layers: layout.layerCount, moeLayers: 0, totalBytes: layout.totalBytes, residentBytes: layout.residentBytes });
  });

  it("counts every access, and reads exactly what it missed when prefetch is off", async () => {
    const r = await engineStream(need(), model, { tokens: 60, warmup: 0, cacheMb: 1, policy: "lru", prefetch: false, ioThreads: 1 });
    const t = r.totals;
    expect(t.accesses).toBe(60 * SPEC.layers * SPEC.expertsUsed);
    expect(t.hits + t.misses).toBe(t.accesses);
    expect(t.demandBytes).toBe(t.misses * r.model.expertBytes);
    expect(t.prefetchIssued).toBe(0);
    expect(r.layers.reduce((n, l) => n + l.accesses, 0)).toBe(t.accesses);
    expect(r.provenance.routing).toBe("synthetic");
  });

  it("is deterministic for a seed when nothing runs concurrently, and differs between seeds", async () => {
    const o = { tokens: 80, warmup: 10, cacheMb: 1, policy: "adaptive" as const, prefetch: false, ioThreads: 1, smart: "2,12,0.02" };
    const a = await engineStream(need(), model, { ...o, seed: 7 });
    const b = await engineStream(need(), model, { ...o, seed: 7 });
    const c = await engineStream(need(), model, { ...o, seed: 8 });
    expect([a.totals.hits, a.totals.misses, a.totals.substituted]).toEqual([b.totals.hits, b.totals.misses, b.totals.substituted]);
    expect([a.totals.hits, a.totals.misses]).not.toEqual([c.totals.hits, c.totals.misses]);
  });

  it("never moves more router probability than tau, and substituting raises the hit rate", async () => {
    const base = { tokens: 120, warmup: 20, cacheMb: 1, policy: "lru" as const, prefetch: false, ioThreads: 1, seed: 3 };
    const plain = await engineStream(need(), model, base);
    const bounded = await engineStream(need(), model, { ...base, smart: "2,12,0.015" });
    const loose = await engineStream(need(), model, { ...base, smart: "2,12,1000" });
    expect(bounded.totals.maxDeltaP).toBeLessThanOrEqual(0.015 + 1e-9);
    expect(plain.totals.substituted).toBe(0);
    expect(bounded.totals.hitRate).toBeGreaterThanOrEqual(plain.totals.hitRate);
    expect(loose.totals.hitRate).toBeGreaterThanOrEqual(bounded.totals.hitRate);
    expect(loose.totals.maxDeltaP).toBeGreaterThan(bounded.totals.maxDeltaP);
  });

  it("keeps dense weights in RAM when asked, reading them once instead of every token", async () => {
    const o = { tokens: 40, warmup: 0, cacheMb: 1, policy: "lru" as const, prefetch: false, ioThreads: 1, seed: 2 };
    const ring = await engineStream(need(), model, { ...o, dense: "ring" });
    const res = await engineStream(need(), model, { ...o, dense: "resident" });
    expect(ring.config.denseMode).toBe("ring");
    expect(ring.totals.denseBytes).toBeGreaterThan(0);
    expect(res.config.denseMode).toBe("resident");
    expect(res.totals.denseBytes).toBe(0);
    expect(res.config.densePreloadBytes).toBeGreaterThan(0);
    expect([res.totals.hits, res.totals.misses]).toEqual([ring.totals.hits, ring.totals.misses]);
    expect((await engineStream(need(), model, { ...o, dense: "auto" })).config.denseMode).toBe("resident");
  });

  it("runs a Q4_0 dot-product kernel over the loaded bytes by default, labels it, and leaves the cache behaviour alone", async () => {
    const o = { tokens: 40, warmup: 0, cacheMb: 1, policy: "lru" as const, prefetch: false, ioThreads: 1, seed: 2 };
    const kernel = await engineStream(need(), model, o);
    const hash = await engineStream(need(), model, { ...o, compute: "checksum" });
    expect(kernel.config.computeMode).toBe("kernel");
    expect(["neon+dotprod", "neon", "avx2", "scalar"]).toContain(kernel.config.computeKernel);
    expect(kernel.provenance.compute).toMatch(/Q4_0 x Q8_0.*not this model's maths/);
    expect(hash.config).toMatchObject({ computeMode: "checksum", computeKernel: "none" });
    expect(hash.provenance.compute).toMatch(/checksum/);
    expect([kernel.totals.hits, kernel.totals.misses]).toEqual([hash.totals.hits, hash.totals.misses]);
    expect(kernel.totals.checksum).not.toBe(hash.totals.checksum);
    expect(kernel.totals.checksum).toBe((await engineStream(need(), model, o)).totals.checksum);
    expect(kernel.totals.computeSeconds).toBeGreaterThan(0);
  });

  it("prefetches into staging without evicting what the cache chose to keep, and accounts for every prefetch", async () => {
    const r = await engineStream(need(), model, { tokens: 80, warmup: 10, cacheMb: 2, policy: "adaptive", prefetch: true, ioThreads: 2, seed: 5 });
    const t = r.totals;
    expect(t.prefetchIssued).toBeGreaterThan(0);
    expect(t.stageHits).toBeLessThanOrEqual(t.prefetchIssued);
    expect(t.hits + t.misses).toBe(t.accesses);
    expect(t.prefetchBytes).toBeGreaterThan(0);
    expect(t.overlap).toBeGreaterThanOrEqual(0);
    expect(t.overlap).toBeLessThanOrEqual(1);
  });

  it("refuses a dense model, a missing file and a file that is not GGUF, with a message", async () => {
    await expect(engineStream(need(), FIXTURE, { tokens: 5 })).rejects.toThrow(/no routed experts/);
    await expect(engineStream(need(), join(dir, "nope.gguf"), { tokens: 5 })).rejects.toThrow(/cannot open/);
    const junk = join(dir, "junk.gguf");
    writeFileSync(junk, Buffer.alloc(4096, 0x41));
    await expect(engineInfo(need(), junk)).rejects.toThrow(/not a GGUF/);
    const trunc = join(dir, "trunc.gguf");
    const full = Buffer.from(await (await import("node:fs/promises")).readFile(FIXTURE));
    writeFileSync(trunc, full.subarray(0, 3000));
    await expect(engineInfo(need(), trunc)).rejects.toThrow();
  });

  describe("scan-noise scenario, windows and memory", () => {
    const WIDE = { arch: "qwen2moe", name: "wide-moe", layers: 6, embedding: 128, heads: 8, kvHeads: 4, contextLength: 2048, expertCount: 128, expertsUsed: 8, expertFfn: 128, denseFfn: 0, vocab: 512, sharedExperts: 0, quantType: 2 };
    let wide: string;
    beforeAll(async () => { wide = join(dir, "wide.gguf"); await writeSyntheticGguf(wide, WIDE, 5); }, 60_000);
    const base = { tokens: 120, warmup: 0, cacheMb: 2, seed: 3, noOsCache: true, window: 10, shift: { at: 40, len: 10 }, compute: "checksum" } as const;
    const rate = (w: { hits: number; accesses: number }): number => w.hits / w.accesses;

    it("--window reports one entry per window, and --shift makes exactly those tokens a cache miss storm for LRU", async () => {
      const r = await engineStream(need(), wide, { ...base, policy: "lru", prefetch: false });
      expect(r.windows.map((w) => w.tokEnd)).toEqual(Array.from({ length: 12 }, (_, i) => (i + 1) * 10));
      expect(r.windows.every((w) => w.accesses === 10 * 6 * 8)).toBe(true);
      expect(r.scenario).toEqual({ shiftAt: 40, shiftLen: 10 });
      const steady = r.windows.filter((_, i) => i >= 2 && i !== 4).map(rate);
      expect(rate(r.windows[4] as { hits: number; accesses: number })).toBeLessThan(Math.min(...steady) - 0.1);
    }, 120_000);

    it("W-TinyLFU rejects the noise and recovers at least as well as LRU in the window after it", async () => {
      const lru = await engineStream(need(), wide, { ...base, policy: "lru", prefetch: false });
      const tiny = await engineStream(need(), wide, { ...base, policy: "wtinylfu", prefetch: false });
      expect(tiny.totals.admitRejects).toBeGreaterThan(0);
      expect(lru.totals.admitRejects).toBe(0);
      expect(rate(tiny.windows[5] as { hits: number; accesses: number })).toBeGreaterThanOrEqual(rate(lru.windows[5] as { hits: number; accesses: number }));
    }, 120_000);

    it("the lookahead predictor cannot see the noise: with prefetch on, the noise window is not lifted above the cache's share", async () => {
      const r = await engineStream(need(), wide, { ...base, policy: "adaptive", prefetch: true, predictCorr: 1 });
      const noise = rate(r.windows[4] as { hits: number; accesses: number });
      const steady = r.windows.filter((_, i) => i >= 2 && i !== 4).map(rate);
      expect(noise).toBeLessThan(Math.min(...steady) - 0.15);
      expect(noise).toBeLessThan(0.45);
    }, 120_000);

    it("peak RSS is flat across tokens and matches the computed figure within 3 MiB once the cache is full", async () => {
      const r = await engineStream(need(), wide, { ...base, policy: "adaptive", prefetch: true, tokens: 200, window: 20, shift: { at: 40, len: 10 }, cacheMb: 1 });
      const first = (r.windows[0] as { peakRssBytes: number }).peakRssBytes, last = (r.windows[r.windows.length - 1] as { peakRssBytes: number }).peakRssBytes;
      expect(r.windows.length).toBe(10);
      expect(last).toBeLessThanOrEqual(first * 1.02);
      expect(r.rss.peakBytes).toBeGreaterThan(r.rss.startBytes);
      // 3 MiB is what was measured on macOS; other platforms (glibc, 4 KiB pages, thread stacks) get a wider band until they have been measured.
      expect(Math.abs(r.rss.peakBytes - r.rss.predictedBytes)).toBeLessThanOrEqual((process.platform === "darwin" ? 3 : 8) * 2 ** 20);
    }, 120_000);

    it("the cache never takes more memory than --cache-mb", async () => {
      const r = await engineStream(need(), wide, { tokens: 20, warmup: 2, cacheMb: 1, compute: "checksum", noOsCache: true });
      const stride = Math.ceil(r.model.expertBytes / 4096) * 4096;
      expect(r.config.cacheSlots * stride).toBeLessThanOrEqual(2 ** 20);
      expect((r.config.cacheSlots + 1) * stride).toBeGreaterThan(2 ** 20);
    }, 60_000);

    it("rejects a malformed --shift", async () => {
      await expect(engineStream(need(), wide, { tokens: 10, shift: { at: 1, len: 0 } })).rejects.toThrow(/--shift/);
    });
  });
});
