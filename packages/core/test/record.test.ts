import { describe, expect, it } from "vitest";
import { emptyResults, validateBenchRecord, type BenchRecord } from "../src";
import { detectCapabilities } from "../src/node";

const good = (): BenchRecord => ({
  schema: 1, clutorVersion: "0.0.0", createdAt: "2026-01-01T00:00:00.000Z",
  machine: { os: "macOS", osRelease: "1", arch: "arm64", cpu: "x", cores: 8, ramBytes: 1, cpuFeatures: [], accelerators: [] },
  model: { name: "m", arch: "a", quantization: null, fileBytes: 1, experts: 0, expertsUsed: 0, synthetic: true },
  runtime: { engine: null, contextTokens: null, cacheBytes: null, cachePolicy: null, prompt: null, tokens: null },
  results: { ...emptyResults(), diskReadBytesPerSec: { value: 3e9, basis: "measured", runs: 96 } }, notes: [],
});

describe("bench record", () => {
  it("accepts a well-formed record with unmeasured fields as null", () => expect(validateBenchRecord(good())).toEqual([]));
  it("rejects wrong schema, bad basis, negative or non-finite values, zero runs", () => {
    const r = good();
    r.results.loadSeconds = { value: -1, basis: "guess" as never, runs: 0 };
    r.results.cacheHitRate = { value: Infinity, basis: "measured", runs: 1 };
    const e = validateBenchRecord({ ...r, schema: 2 });
    expect(e.join("\n")).toMatch(/schema/);
    expect(e.join("\n")).toMatch(/loadSeconds.value/);
    expect(e.join("\n")).toMatch(/loadSeconds.basis/);
    expect(e.join("\n")).toMatch(/loadSeconds.runs/);
    expect(e.join("\n")).toMatch(/cacheHitRate.value/);
  });
  it("never reports an accelerator as used by Clutor", () => {
    for (const a of detectCapabilities().accelerators) expect(a.usedByClutor).toBe(false);
  });
});
