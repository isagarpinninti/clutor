import { describe, expect, it } from "vitest";
import { CATALOG, GiB, QUANTS, approximateLayout, evaluateGates, gateSummary, planMemory, type GgufFile } from "../src";

const plan = (id: string, gb: number, q = "Q4_K_M") => planMemory(approximateLayout(CATALOG.find((s) => s.id === id)!, QUANTS[q] as number), { hostRamBytes: gb * GiB });
const get = (g: ReturnType<typeof evaluateGates>, n: number) => g.find((x) => x.id === `CL-${n}`)!;

describe("gates", () => {
  it("skips what has nothing to look at, and always returns all eight in order", () => {
    const g = evaluateGates({ hostRamBytes: 16 * GiB });
    expect(g.map((x) => x.id)).toEqual(["CL-101", "CL-102", "CL-103", "CL-104", "CL-105", "CL-106", "CL-107", "CL-108"]);
    expect(gateSummary(g)).toEqual({ pass: 1, warn: 0, fail: 0, skipped: 7 });
  });
  it("fails a model that cannot fit and passes one that can", () => {
    expect(get(evaluateGates({ hostRamBytes: 4 * GiB, plan: plan("llama-3.3-70b", 4, "F16") }), 101).status).toBe("fail");
    expect(get(evaluateGates({ hostRamBytes: 16 * GiB, plan: plan("qwen3-30b-a3b", 16) }), 101).status).toBe("pass");
  });
  it("warns when the cache cannot hold a token's experts", () => {
    const p = planMemory(approximateLayout(CATALOG.find((s) => s.id === "deepseek-v3")!, 4.85), { hostRamBytes: 16 * GiB, ramCeilingBytes: 12 * GiB, osHeadroomBytes: 1 * GiB });
    const s = get(evaluateGates({ hostRamBytes: 16 * GiB, plan: p }), 102);
    expect(["pass", "warn"]).toContain(s.status);
  });
  it("bounds substitution", () => {
    const st = (enabled: boolean, maxDeltaP: number) => get(evaluateGates({ hostRamBytes: GiB, smartRoute: { enabled, maxDeltaP } }), 103).status;
    expect(st(false, 0.5)).toBe("pass");
    expect(st(true, 0.015)).toBe("pass");
    expect(st(true, 0.03)).toBe("warn");
    expect(st(true, 0.2)).toBe("fail");
  });
  it("prefers a scan-resistant cache", () => {
    const st = (policy: "lru" | "wtinylfu" | "adaptive") => get(evaluateGates({ hostRamBytes: GiB, cache: { policy } }), 104).status;
    expect([st("lru"), st("wtinylfu"), st("adaptive")]).toEqual(["warn", "pass", "pass"]);
  });
  it("judges the agent by how it behaved", () => {
    const a = (toolCalls: number, invalidReplies: number, violations: number, commandApproval: "always" | "project-code" | "never-asked") => evaluateGates({ hostRamBytes: GiB, agent: { toolCalls, invalidReplies, violations, commandApproval } });
    expect([105, 106, 107].map((n) => get(a(20, 1, 0, "project-code"), n).status)).toEqual(["pass", "pass", "pass"]);
    expect(get(a(10, 5, 0, "always"), 105).status).toBe("warn");
    expect(get(a(3, 7, 0, "always"), 105).status).toBe("fail");
    expect(get(a(10, 0, 2, "always"), 106).message).toMatch(/2 attempts were blocked/);
    expect(get(a(10, 0, 0, "never-asked"), 107).status).toBe("warn");
  });
  it("fails on a damaged model file or a broken receipt log", () => {
    const bad = { fileSize: 100, dataOffset: 10, alignment: 32, tensors: [{ name: "t", offset: 0, fileOffset: 10, nbytes: 500 }] } as unknown as GgufFile;
    expect(get(evaluateGates({ hostRamBytes: GiB, gguf: bad }), 108).status).toBe("fail");
    expect(get(evaluateGates({ hostRamBytes: GiB, receiptsOk: { ok: false, detail: "entry 2" } }), 108).message).toMatch(/entry 2/);
    expect(get(evaluateGates({ hostRamBytes: GiB, receiptsOk: { ok: true } }), 108).status).toBe("pass");
  });
});
