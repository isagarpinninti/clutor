import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { CATALOG, GiB, QUANTS, approximateLayout, planMemory } from "../src";

const model = (id: string, q = "Q4_K_M") => approximateLayout(CATALOG.find((s) => s.id === id)!, QUANTS[q] as number);

describe("memory planner", () => {
  it("runs DeepSeek-V3 out of core on a 16 GiB machine: dense weights resident, experts streamed, and speed depends entirely on the hit rate", () => {
    const p = planMemory(model("deepseek-v3"), { hostRamBytes: 16 * GiB });
    expect(p.strategy).toBe("dense-resident-experts-streamed");
    expect(p.fits).toBe(true);
    expect(p.usedBytes).toBeLessThanOrEqual(p.ceilingBytes);
    expect(p.expertCache.fractionOfExperts).toBeLessThan(0.05); // a few percent of the experts fit
    expect(p.perToken.denseStreamBytes).toBe(0);
    // 58 layers x 8 experts x ~27 MB: about 12 GB of experts are active per token, so with no cache help the disk caps speed hard.
    expect(p.perToken.activeExpertBytes / GiB).toBeGreaterThan(10);
    expect(p.speed[0]!.hitRate).toBe(0);
    expect(p.speed[0]!.maxTokensPerSec).toBeLessThan(0.4);
    expect(p.speed.find((s) => s.hitRate === 0.5)!.maxTokensPerSec).toBeLessThan(0.7);
  });

  it("says a 671B model on an 8 GiB machine is possible only at well under one token per second", () => {
    const p = planMemory(model("deepseek-v3"), { hostRamBytes: 8 * GiB });
    expect(p.fits).toBe(true);
    expect(p.strategy).toBe("dense-ring-experts-streamed"); // 8 GiB of dense weights alone do not fit
    // the dense weights are re-read for every token, so even a perfect cache cannot beat the disk
    for (const s of p.speed) expect(s.maxTokensPerSec).toBeLessThan(0.5);
  });

  it("keeps a small mixture-of-experts model usable on a 16 GiB machine: dense weights stay resident, experts stream", () => {
    const p = planMemory(model("qwen3-30b-a3b"), { hostRamBytes: 16 * GiB });
    expect(p.strategy).toBe("dense-resident-experts-streamed");
    expect(p.expertCache.slots).toBeGreaterThan(p.expertCache.slotsPerToken * 10);
    const at95 = p.speed.find((s) => s.hitRate === 0.95)!;
    expect(at95.maxTokensPerSec).toBeGreaterThan(5);
  });

  it("loads everything when the model fits, and then no disk traffic is needed", () => {
    const p = planMemory(model("qwen3-30b-a3b", "Q2_K"), { hostRamBytes: 64 * GiB });
    expect(p.strategy).toBe("fully-resident");
    expect(p.speed.every((s) => s.maxTokensPerSec === Infinity)).toBe(true);
  });

  it("refuses when even the always-resident parts do not fit, and says by how much", () => {
    const p = planMemory(model("llama-3.3-70b", "F16"), { hostRamBytes: 4 * GiB });
    expect(p.fits).toBe(false);
    expect(p.strategy).toBe("does-not-fit");
    expect(p.shortfallBytes).toBeGreaterThan(0);
    expect(p.expertCache.slots).toBe(0);
  });

  it("treats a dense 70B model as pure streaming: every token re-reads the weights", () => {
    const p = planMemory(model("llama-3.3-70b"), { hostRamBytes: 16 * GiB });
    expect(p.strategy).toBe("dense-ring-experts-streamed");
    expect(p.perToken.activeExpertBytes).toBe(0);
    expect(p.speed[0]!.maxTokensPerSec).toBeLessThan(0.15);
  });

  it("uses the KV-cache geometry: a longer context needs more memory, a smaller element type needs less", () => {
    const l = model("qwen3-235b-a22b");
    const kv = (ctx: number, bits: 16 | 8 | 4) => planMemory(l, { hostRamBytes: 32 * GiB, contextTokens: ctx, kvBits: bits }).components.find((c) => c.id === "kv")!.bytes;
    expect(kv(32768, 16)).toBe(4 * kv(8192, 16));
    expect(kv(8192, 8)).toBe(kv(8192, 16) / 2);
    expect(kv(8192, 4)).toBe(kv(8192, 16) / 4);
  });

  it("an MLA model caches far less per token than a grouped-query model of similar size", () => {
    const kvPerToken = (id: string) => planMemory(model(id), { hostRamBytes: 64 * GiB, contextTokens: 1024 }).components.find((c) => c.id === "kv")!.bytes / 1024;
    expect(kvPerToken("deepseek-v3")).toBeLessThan(kvPerToken("llama-3.3-70b") / 3);
  });

  it("flags a cache too small to hold one token's experts", () => {
    const p = planMemory(model("deepseek-v3"), { hostRamBytes: 16 * GiB, ramCeilingBytes: 2.9 * GiB, osHeadroomBytes: 1 * GiB });
    if (p.fits) expect(p.warnings.join(" ")).toMatch(/cache holds/);
  });

  it("is labelled approximate for catalog models, and says the disk speed is an assumption until it is measured", () => {
    const p = planMemory(model("qwen3-30b-a3b"), { hostRamBytes: 16 * GiB });
    expect(p.approximate).toBe(true);
    expect(p.warnings.join(" ")).toMatch(/architecture/);
    expect(p.warnings.join(" ")).toMatch(/assumption/);
    expect(planMemory(model("qwen3-30b-a3b"), { hostRamBytes: 16 * GiB, diskBytesPerSec: 5 * GiB, diskMeasured: true }).warnings.join(" ")).not.toMatch(/assumption/);
  });

  it("never plans to use more than the ceiling, for any model, size, context or element type", () => {
    fc.assert(
      fc.property(fc.constantFrom(...CATALOG.map((c) => c.id)), fc.constantFrom(...Object.keys(QUANTS)), fc.integer({ min: 2, max: 128 }), fc.integer({ min: 256, max: 131072 }), fc.constantFrom<16 | 8 | 4>(16, 8, 4), (id, q, gb, ctx, bits) => {
        const p = planMemory(model(id, q), { hostRamBytes: gb * GiB, contextTokens: ctx, kvBits: bits });
        if (p.fits) expect(p.usedBytes).toBeLessThanOrEqual(p.ceilingBytes + 1);
        expect(p.components.every((c) => c.bytes >= 0)).toBe(true);
        for (let i = 1; i < p.speed.length; i++) expect(p.speed[i]!.maxTokensPerSec).toBeGreaterThanOrEqual(p.speed[i - 1]!.maxTokensPerSec);
      }),
      { numRuns: 300 },
    );
  });

  it("gives a model more cache when it is given more memory", () => {
    const l = model("qwen3-235b-a22b");
    let last = -1;
    for (const gb of [16, 24, 32, 48, 64, 96, 128]) {
      const p = planMemory(l, { hostRamBytes: gb * GiB });
      expect(p.expertCache.slots).toBeGreaterThanOrEqual(last);
      last = p.expertCache.slots;
    }
  });
});
