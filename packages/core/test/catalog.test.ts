import { describe, expect, it } from "vitest";
import { CATALOG, QUANTS, approximateLayout, totalParams } from "../src/planner/catalog";
import { GiB, planMemory } from "../src";

describe("the architecture catalog", () => {
  for (const s of CATALOG) {
    it(`${s.name}: computed parameters match the published ${s.publishedParamsB}B within 2%`, () => {
      const b = totalParams(s) / 1e9;
      expect(Math.abs(b - s.publishedParamsB) / s.publishedParamsB).toBeLessThan(0.02);
    });
  }
  it("sizes a layout from bits per weight", () => {
    const ds = CATALOG.find((s) => s.id === "deepseek-v3")!;
    const l = approximateLayout(ds, QUANTS["Q4_K_M"] as number);
    expect(l.approximate).toBe(true);
    expect(l.layers.filter((x) => x.experts.length > 0)).toHaveLength(58);
    expect(l.totalBytes / 2 ** 30).toBeGreaterThan(370);
    expect(l.totalBytes / 2 ** 30).toBeLessThan(400);
  });

  describe("glm-5-744b (stress-test profile)", () => {
    const spec = CATALOG.find((c) => c.id === "glm-5-744b")!;
    const layout = approximateLayout(spec, QUANTS["Q4_K_M"] as number);
    it("has 78 layers of 256 routed experts with top-8 routing, and says it is not a published configuration", () => {
      expect([spec.layers, spec.experts, spec.expertsUsed, spec.denseLayers]).toEqual([78, 256, 8, 0]);
      expect(spec.source).toMatch(/not the vendor/);
      expect(layout.layers.filter((l) => l.experts.length === 256)).toHaveLength(78);
    });
    it("computes to about 9.9 GiB of dense weights and 419 GiB of routed experts", () => {
      const experts = layout.layers.reduce((n, l) => n + l.experts.reduce((m, e) => m + e.reduce((k, r) => k + r.length, 0), 0), 0);
      expect(experts / GiB).toBeGreaterThan(416);
      expect(experts / GiB).toBeLessThan(422);
      const dense = layout.totalBytes - experts;
      expect(dense / GiB).toBeGreaterThan(9.5);
      expect(dense / GiB).toBeLessThan(10.3);
    });
    it("plans a 2-layer dense window at 8 and 16 GiB, resident dense at 25 GiB, and stays inside the RAM ceiling each time", () => {
      const p8 = planMemory(layout, { hostRamBytes: 8 * GiB }), p16 = planMemory(layout, { hostRamBytes: 16 * GiB }), p25 = planMemory(layout, { hostRamBytes: 25 * GiB });
      expect(p8.strategy).toBe("dense-ring-experts-streamed");
      expect(p16.strategy).toBe("dense-ring-experts-streamed");
      expect(p25.strategy).toBe("dense-resident-experts-streamed");
      for (const [p, ram] of [[p8, 8], [p16, 16], [p25, 25]] as const) {
        expect(p.fits).toBe(true);
        expect(p.usedBytes).toBeLessThanOrEqual(p.ceilingBytes);
        expect(p.ceilingBytes).toBeLessThan(ram * GiB);
        expect(p.expertCache.totalExperts).toBe(19968);
        expect(p.expertCache.slotsPerToken).toBe(624);
      }
      expect(p8.expertCache.slots).toBeLessThan(p16.expertCache.slots);
    });
  });
});
