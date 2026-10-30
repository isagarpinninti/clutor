import { describe, expect, it } from "vitest";
import {
  AdaptiveTinyLfuCache, BASELINE, CATALOG, CLUTOR_DEFAULTS, GiB, LruCache, QUANTS, WTinyLfuCache, approximateLayout, generateRoutingTrace, expertSizeBytes, parseRoutingTrace, planMemory, simulateStreaming,
  type ModelLayout, type StreamerConfig, type TokenRouting,
} from "../src";

const spec = CATALOG.find((s) => s.id === "qwen3-30b-a3b")!;
const layout: ModelLayout = approximateLayout(spec, QUANTS["Q4_K_M"] as number);
const moe = layout.layers.filter((l) => l.experts.length).map((l) => l.index);
const eBytes = expertSizeBytes(layout);
const trace = (o: { tokens?: number; locality?: number; seed?: number; skew?: number; topicEvery?: number } = {}): TokenRouting[] =>
  generateRoutingTrace({ moeLayers: moe, experts: spec.experts, used: spec.expertsUsed, tokens: o.tokens ?? 400, seed: o.seed ?? 5, locality: o.locality ?? 0.9, ...(o.skew !== undefined ? { skew: o.skew } : {}), ...(o.topicEvery !== undefined ? { topicEvery: o.topicEvery } : {}) });
const base = (cacheGiB: number, compute = 0.0005) => ({ cacheBytes: cacheGiB * GiB, disk: { bytesPerSec: 3 * GiB, requestLatencySec: 100e-6 }, computePerLayerSec: compute });
const run = (t: TokenRouting[], over: Partial<StreamerConfig> & { cacheGiB?: number } = {}, d: Omit<StreamerConfig, "cacheBytes" | "disk" | "computePerLayerSec"> = CLUTOR_DEFAULTS) =>
  simulateStreaming(layout, t, { ...d, ...base(over.cacheGiB ?? 2), ...over });

describe("routing traces", () => {
  it("are reproducible from a seed, ranked, and softmax-valid", () => {
    const a = trace({ tokens: 20 }), b = trace({ tokens: 20 }), c = trace({ tokens: 20, seed: 6 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    for (const tok of a) for (const l of tok) {
      expect(l.experts).toHaveLength(l.probs.length);
      expect(new Set(l.experts).size).toBe(l.experts.length);
      for (let i = 1; i < l.probs.length; i++) expect(l.probs[i]!).toBeLessThanOrEqual(l.probs[i - 1]!);
      expect(l.probs.reduce((s, x) => s + x, 0)).toBeLessThanOrEqual(1 + 1e-9);
    }
  });
  it("show temporal locality when asked: neighbouring tokens share more experts", () => {
    const overlap = (loc: number): number => {
      const t = trace({ tokens: 300, locality: loc });
      let same = 0, n = 0;
      for (let i = 1; i < t.length; i++) for (let l = 0; l < t[i]!.length; l++) {
        const a = new Set(t[i - 1]![l]!.experts.slice(0, 8));
        for (const e of t[i]![l]!.experts.slice(0, 8)) { n++; if (a.has(e)) same++; }
      }
      return same / n;
    };
    expect(overlap(0.95)).toBeGreaterThan(overlap(0.1) + 0.1);
  });
  it("round-trips through the text format and rejects malformed lines", () => {
    const t = trace({ tokens: 3 });
    const text = t.map((tok) => tok.map((l) => JSON.stringify(l)).join("\n")).join("\n\n");
    expect(parseRoutingTrace(text)).toEqual(t);
    expect(() => parseRoutingTrace('{"layer":1,"experts":[1,2],"probs":[0.5]}')).toThrow();
  });
});

describe("streaming simulation", () => {
  it("accounts for every expert use exactly once, and every byte read to a read request", () => {
    const m = run(trace(), { cacheGiB: 1 });
    expect(m.expertUses).toBe(400 * 48 * 8);
    expect(m.hits + m.prefetchUsed + m.substituted + m.demandMisses).toBe(m.expertUses);
    expect(m.bytesRead).toBe((m.demandMisses + m.prefetchIssued) * eBytes);
    expect(m.requests).toBeGreaterThan(0);
    expect(m.totalSec).toBeGreaterThanOrEqual(m.stallSec);
  });

  it("is deterministic", () => {
    const t = trace({ tokens: 150 });
    expect(run(t)).toEqual(run(t));
  });

  it("with no cache reads every expert every time, and the planner's disk bound is that speed", () => {
    const t = trace({ tokens: 100 });
    const m = run(t, { cacheGiB: 0 }, BASELINE);
    expect(m.hitRate).toBe(0);
    expect(m.bytesPerToken).toBe(48 * 8 * eBytes);
    const plan = planMemory(layout, { hostRamBytes: 16 * GiB, diskBytesPerSec: 3 * GiB, diskMeasured: true });
    const bound = plan.speed[0]!.maxTokensPerSec;
    expect(m.tokensPerSec).toBeLessThanOrEqual(bound * 1.001);
    expect(m.tokensPerSec).toBeGreaterThan(bound * 0.5);
  });

  it("a cache that holds every expert stops reading after warm-up", () => {
    const m = run(trace({ tokens: 600 }), { cacheGiB: 64 });
    expect(m.hitRate).toBeGreaterThan(0.85);
    expect(m.demandMisses + m.prefetchUsed).toBeLessThanOrEqual(layout.expertCount * 48);
  });

  it("more cache means a higher hit rate and a faster model", () => {
    const t = trace({ tokens: 400 });
    let last = -1, lastSpeed = 0;
    for (const g of [0.5, 1, 2, 4, 8]) {
      const m = run(t, { cacheGiB: g });
      expect(m.hitRate).toBeGreaterThanOrEqual(last - 0.01);
      expect(m.tokensPerSec).toBeGreaterThanOrEqual(lastSpeed * 0.97);
      last = m.hitRate;
      lastSpeed = m.tokensPerSec;
    }
  });

  it("the adaptive policy is never far behind LRU and is far ahead when the cache is too small for LRU to work at all", () => {
    const t = trace({ tokens: 700, locality: 0.9 });
    const lru = (g: number) => run(t, { cacheGiB: g, policy: "lru" }, { ...CLUTOR_DEFAULTS, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 } }).hitRate;
    const ada = (g: number) => run(t, { cacheGiB: g }, { ...CLUTOR_DEFAULTS, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 } }).hitRate;
    expect(lru(1)).toBeLessThan(0.02);       // one token's experts (1.4 GB) do not fit: LRU cycles and never hits
    expect(ada(1)).toBeGreaterThan(0.1);      // frequency admission keeps something useful
    for (const g of [2, 4]) expect(ada(g)).toBeGreaterThan(lru(g) - 0.08);
  });

  it("coalescing reads of neighbouring experts cuts requests but not bytes", () => {
    const t = trace({ tokens: 150 });
    const off = run(t, { coalesce: false }), on = run(t, { coalesce: true });
    expect(on.requests).toBeLessThan(off.requests);
    expect(on.bytesRead).toBe(off.bytesRead);
    expect(on.totalSec).toBeLessThanOrEqual(off.totalSec);
  });

  it("only prefetches when it is confident and the disk would otherwise be idle", () => {
    const t = trace({ tokens: 300 });
    // Disk-bound (tiny compute): the idle window is shorter than one expert read, so nothing is prefetched and speed is untouched.
    const bound = run(t, { cacheGiB: 2 }), none = run(t, { cacheGiB: 2 }, { ...CLUTOR_DEFAULTS, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 } });
    expect(bound.prefetchIssued).toBe(0);
    expect(bound.tokensPerSec).toBeCloseTo(none.tokensPerSec, 6);
    // Compute-bound (a slow CPU per layer): there is idle disk time, so prefetch happens, and it must not slow the model down.
    const slow = run(t, { cacheGiB: 2, computePerLayerSec: 0.02 }), slowNone = run(t, { cacheGiB: 2, computePerLayerSec: 0.02 }, { ...CLUTOR_DEFAULTS, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 } });
    expect(slow.prefetchIssued).toBeGreaterThan(0);
    expect(slow.prefetchUsed).toBeGreaterThan(0);
    expect(slow.stallSec).toBeLessThanOrEqual(slowNone.stallSec * 1.02);
    expect(slow.tokensPerSec).toBeGreaterThanOrEqual(slowNone.tokensPerSec * 0.98);
    expect(slow.prefetchUsed + slow.prefetchWasted).toBeLessThanOrEqual(slow.prefetchIssued);
  });

  describe("SMART-ROUTE substitution", () => {
    const sr = (over: { maxDeltaP?: number; maxSubstitutions?: number; candidates?: number } = {}) => ({ ...CLUTOR_DEFAULTS, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 }, smartRoute: { enabled: true, maxSubstitutions: 2, candidates: 12, maxDeltaP: 0.05, ...over } });
    const t = trace({ tokens: 300 });

    it("changes nothing when switched off or when the allowed probability gap is zero", () => {
      const off = run(t, { cacheGiB: 2 });
      const zero = run(t, { cacheGiB: 2 }, sr({ maxDeltaP: 0 }));
      expect(off.substituted).toBe(0);
      expect(zero.substituted).toBe(0);
      expect(zero.exactMassShare).toBe(1);
    });

    it("trades exactness for fewer reads, more of it as the allowed gap grows, and reports how much routing mass was not exact", () => {
      const a = run(t, { cacheGiB: 2 }, sr({ maxDeltaP: 0.015 })), b = run(t, { cacheGiB: 2 }, sr({ maxDeltaP: 0.1 })), none = run(t, { cacheGiB: 2 });
      expect(a.substituted).toBeGreaterThan(0);
      expect(b.substituted).toBeGreaterThanOrEqual(a.substituted);
      expect(b.bytesRead).toBeLessThan(none.bytesRead);
      expect(a.exactMassShare).toBeLessThan(1);
      expect(b.exactMassShare).toBeLessThanOrEqual(a.exactMassShare);
      expect(a.exactShare).toBeCloseTo(1 - a.substituted / a.expertUses, 10);
    });

    it("never substitutes more than J experts per layer per token", () => {
      const one = run(t, { cacheGiB: 4 }, sr({ maxDeltaP: 1, maxSubstitutions: 1 }));
      expect(one.substituted).toBeLessThanOrEqual(300 * 48);
      const zero = run(t, { cacheGiB: 4 }, sr({ maxDeltaP: 1, maxSubstitutions: 0 }));
      expect(zero.substituted).toBe(0);
    });

    it("only considers the best M candidates", () => {
      const narrow = run(t, { cacheGiB: 2 }, sr({ maxDeltaP: 1, candidates: 8 })); // M equal to the experts used: nothing to choose from
      expect(narrow.substituted).toBe(0);
    });
  });

  it("clutor's defaults are at least as fast as the plain-LRU baseline across cache sizes and localities", () => {
    for (const loc of [0.5, 0.9]) {
      const t = trace({ tokens: 500, locality: loc });
      for (const g of [1, 2, 4]) {
        const b = run(t, { cacheGiB: g }, BASELINE), c = run(t, { cacheGiB: g });
        expect(c.tokensPerSec, `loc ${loc} cache ${g}`).toBeGreaterThanOrEqual(b.tokensPerSec * 0.97);
      }
    }
}, 60_000);
});

describe("heatmap snapshot", () => {
  it("reports which experts of a layer are cached and how often each was chosen", () => {
    const t = trace({ tokens: 200 });
    const layer = moe[10]!;
    const m = run(t, { cacheGiB: 2, snapshotLayer: layer });
    expect(m.snapshot).toMatchObject({ layer });
    expect(m.snapshot!.cached).toHaveLength(spec.experts);
    expect(m.snapshot!.uses.reduce((a, b) => a + b, 0)).toBe(200 * spec.expertsUsed);
    // an expert chosen very often is more likely to be cached than one never chosen
    const idx = m.snapshot!.uses.map((u, e) => [u, e] as const).sort((a, b) => b[0] - a[0]);
    const topCached = idx.slice(0, 16).filter(([, e]) => m.snapshot!.cached[e] === 1).length;
    const bottomCached = idx.slice(-16).filter(([, e]) => m.snapshot!.cached[e] === 1).length;
    expect(topCached).toBeGreaterThan(bottomCached);
    expect(run(t).snapshot).toBeUndefined();
  });
});

describe("adaptive W-TinyLFU", () => {
  it("never exceeds capacity and agrees with its own count", () => {
    const c = new AdaptiveTinyLfuCache(500, { expectedKeys: 300, period: 200 });
    let x = 1;
    for (let i = 0; i < 20000; i++) {
      x = (x * 1103515245 + 12345) & 0x7fffffff;
      const k = x % 400;
      if (!c.access(k)) c.admit(k, 5);
      expect(c.usedBytes).toBeLessThanOrEqual(500);
    }
    let n = 0;
    for (let k = 0; k < 400; k++) if (c.peek(k)) n++;
    expect(n).toBe(c.count);
  });

  it("moves its window toward recency when recent entries are reused, and toward frequency when a hot set hides among one-offs", () => {
    const recency = new AdaptiveTinyLfuCache(100, { expectedKeys: 1000, period: 500 });
    for (let round = 0; round < 400; round++) for (let k = 0; k < 80; k++) { const key = (round % 3) * 1000 + k; if (!recency.access(key)) recency.admit(key, 1); }
    const scan = new AdaptiveTinyLfuCache(100, { expectedKeys: 5000, period: 500 });
    let nextScan = 100000;
    for (let i = 0; i < 40000; i++) {
      const key = i % 5 === 0 ? (i / 5) % 40 : nextScan++;
      if (!scan.access(key)) scan.admit(key, 1);
    }
    expect(scan.currentWindowFraction).toBeLessThanOrEqual(0.25);
    expect(recency.currentWindowFraction).toBeGreaterThanOrEqual(0.01);
    const lru = new LruCache(100), fixed = new WTinyLfuCache(100, { expectedKeys: 5000 });
    nextScan = 100000;
    for (let i = 0; i < 40000; i++) {
      const key = i % 5 === 0 ? (i / 5) % 40 : nextScan++;
      if (!lru.access(key)) lru.admit(key, 1);
      if (!fixed.access(key)) fixed.admit(key, 1);
    }
    expect(scan.stats.hits / (scan.stats.hits + scan.stats.misses)).toBeGreaterThan((lru.stats.hits / (lru.stats.hits + lru.stats.misses)) + 0.1);
  });
});
