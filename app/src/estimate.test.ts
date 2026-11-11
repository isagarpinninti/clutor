import { describe, expect, it } from "vitest";
import { contract, estimates, estimateSpeed, fmtDuration, fmtRange, remainingFromPlan } from "./estimate";
import type { ModelCheck } from "./plan";

const GiB = 2 ** 30;
const base = (over: Partial<ModelCheck["plan"]> & { strategy: ModelCheck["plan"]["strategy"] }, check: Partial<ModelCheck> = {}): ModelCheck => ({
  name: "m", arch: "x", fileBytes: 17 * GiB, layers: 48, experts: 128, expertsUsed: 8, verdict: "streams", headline: "", speed: "", notes: [],
  plan: { fits: true, ceilingBytes: 12 * GiB, usedBytes: 12 * GiB, components: [{ id: "resident", label: "r", bytes: 1.5 * GiB }], shortfallBytes: 0,
    expertCache: { bytes: 4 * GiB, slots: 1400, expertBytes: 2.7e6, totalExperts: 6144, fractionOfExperts: 0.25, slotsPerToken: 384 },
    perToken: { denseStreamBytes: 0, activeExpertBytes: 1.0e9, routedLayers: 48, expertsUsed: 8 }, disk: { bytesPerSec: 3 * GiB, measured: false }, speed: [], warnings: [], approximate: false, ...over } as ModelCheck["plan"],
  ...check,
});

describe("speed estimate", () => {
  it("is close to the measured 4.7 tokens per second for a 30B MoE with a quarter of its experts cached", () => {
    const e = estimateSpeed(base({ strategy: "dense-resident-experts-streamed" }))!;
    expect(e.source).toBe("estimated");
    const mid = Math.sqrt(e.lowTps * e.highTps);
    expect(mid).toBeGreaterThan(3); expect(mid).toBeLessThan(6);
  });
  it("is faster when the whole model is in memory and slower than a floor warns", () => {
    const resident = estimateSpeed(base({ strategy: "fully-resident", expertCache: { bytes: 0, slots: 0, expertBytes: 2.7e6, totalExperts: 6144, fractionOfExperts: 1, slotsPerToken: 384 } }, { fileBytes: 4.7 * GiB, experts: 0, verdict: "fits" }))!;
    expect(resident.highTps).toBeGreaterThan(8);
    const crawl = estimateSpeed(base({ strategy: "dense-ring-experts-streamed", perToken: { denseStreamBytes: 40 * GiB, activeExpertBytes: 0, routedLayers: 0, expertsUsed: 0 } }, { experts: 0, fileBytes: 40 * GiB, verdict: "tight" }))!;
    expect(crawl.lowTps).toBeLessThan(0.5);
    expect(crawl.caution).toMatch(/Very slow/);
  });
  it("uses a measured speed when there is one, and gives nothing for a model that cannot load", () => {
    expect(estimateSpeed(base({ strategy: "dense-resident-experts-streamed" }), { measuredTps: 2 })?.source).toBe("measured");
    expect(estimateSpeed(base({ strategy: "does-not-fit" }, { verdict: "too-big" }))).toBeUndefined();
  });
  it("turns a speed into times for a chat answer and a small coding task", () => {
    const e = estimates(base({ strategy: "dense-resident-experts-streamed" }), { measuredTps: 2 })!;
    expect(e.lines).toHaveLength(2);
    expect(e.lines[1]!.time).toMatch(/min|h/);
  });
});

describe("durations", () => {
  it("writes them the way a person would", () => {
    expect(fmtDuration(40)).toBe("40 s");
    expect(fmtDuration(600)).toBe("10 min");
    expect(fmtDuration(4800)).toBe("80 min");
    expect(fmtDuration(8400)).toBe("2 h 20 min");
    expect(fmtDuration(20 * 3600)).toBe("about 20 h");
    expect(fmtRange(720, 1200)).toBe("12 to 20 min");
  });
  it("estimates the time left from the plan", () => {
    expect(remainingFromPlan(120, 1, 4)).toBe(360);
    expect(remainingFromPlan(120, 0, 4)).toBeUndefined();
    expect(remainingFromPlan(120, 4, 4)).toBeUndefined();
  });
});

describe("performance contract", () => {
  it("says a 30B MoE with a quarter of its experts cached is runnable but not fast, and what it is for", () => {
    const c = base({ strategy: "dense-resident-experts-streamed" });
    const ct = contract(c, estimates(c));
    expect(ct.loadable && ct.executable).toBe(true);
    expect(ct.fast).toBe(false);
    expect(ct.use.length).toBeGreaterThan(10);
  });
  it("says a small model fully in memory is fast", () => {
    const c = base({ strategy: "fully-resident", expertCache: { bytes: 0, slots: 0, expertBytes: 2.7e6, totalExperts: 6144, fractionOfExperts: 1, slotsPerToken: 384 } }, { fileBytes: 1 * GiB, experts: 0, verdict: "fits" });
    const ct = contract(c, estimates(c));
    expect(ct.fast).toBe(true);
  });
  it("says a model that is too big is neither runnable nor fast, and that it can still be loaded only if the verdict allows", () => {
    const c = base({ strategy: "fully-resident" }, { verdict: "too-big" });
    const ct = contract(c, estimates(c));
    expect(ct).toMatchObject({ loadable: false, executable: false, fast: false });
  });
});
