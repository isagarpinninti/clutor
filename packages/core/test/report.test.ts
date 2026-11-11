import { describe, expect, it } from "vitest";
import { GiB, QUANTS, approximateLayout, buildReport, CATALOG, planMemory, suitability } from "../src";

const spec = (id: string) => CATALOG.find((c) => c.id === id)!;
const planFor = (id: string, ramGiB: number, disk = 3, measured = false) => {
  const layout = approximateLayout(spec(id), QUANTS["Q4_K_M"]!);
  return { layout, plan: planMemory(layout, { hostRamBytes: ramGiB * GiB, diskBytesPerSec: disk * GiB, diskMeasured: measured }) };
};

describe("plan report", () => {
  it("labels a streamed MoE model: estimated speed, theoretical ceiling, never measured", () => {
    const { layout, plan } = planFor("qwen3-30b-a3b", 16);
    const r = buildReport(layout, plan);
    expect(r.verdict.mode).toBe("disk streaming");
    expect(r.speed?.low.basis).toBe("estimated");
    expect(r.diskCeilingTps?.basis).toBe("theoretical");
    expect(r.diskCeilingTps!.value).toBeLessThan(r.speed!.high.value * 4);
    expect(r.confidence.level).not.toBe("high");
    expect(r.suitability).toHaveLength(4);
  });
  it("uses a measured speed when given, and says so", () => {
    const { layout, plan } = planFor("qwen3-30b-a3b", 16, 3, true);
    const r = buildReport(layout, plan, { measuredTps: 4.7 });
    expect(r.speed?.low.basis).toBe("measured");
    expect(r.speed!.low.value).toBeCloseTo(4.7 * 0.7);
    expect(r.confidence.level).toBe("high");
  });
  it("gives no speed when the model does not fit", () => {
    const { layout, plan } = planFor("qwen3-30b-a3b", 4);
    const r = buildReport(layout, plan);
    if (!plan.fits) { expect(r.speed).toBeUndefined(); expect(r.verdict.canRun).toBe(false); expect(r.suitability).toEqual([]); }
  });
  it("rates tasks by speed: slow is fine for background work but not for chat", () => {
    const t = Object.fromEntries(suitability(0.3, 0.5).map((x) => [x.task, x.rating]));
    expect(t["chat"]).toBe("poor");
    expect(t["background"]).toBe("good");
    const fast = Object.fromEntries(suitability(30, 40).map((x) => [x.task, x.rating]));
    expect(fast["chat"]).toBe("good");
  });
});
