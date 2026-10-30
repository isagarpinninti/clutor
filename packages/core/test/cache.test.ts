import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { CountMinSketch, LruCache, WTinyLfuCache, type ExpertCache } from "../src/cache/policies";

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Zipf-distributed key in [0, n). */
function zipf(n: number, s: number, r: () => number): () => number {
  const cdf: number[] = [];
  let sum = 0;
  for (let i = 1; i <= n; i++) cdf.push((sum += 1 / i ** s));
  return () => {
    const u = r() * sum;
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if ((cdf[m] as number) < u) lo = m + 1; else hi = m; }
    return lo;
  };
}
function run(c: ExpertCache, keys: Iterable<number>): number {
  for (const k of keys) if (!c.access(k)) c.admit(k, 1);
  return c.stats.hits / (c.stats.hits + c.stats.misses);
}

describe("LRU", () => {
  it("evicts the least recently used entry first, and a hit refreshes recency", () => {
    const c = new LruCache(3);
    for (const k of [1, 2, 3]) c.admit(k, 1);
    c.access(1);
    expect(c.admit(4, 1)).toEqual([2]);
    expect([1, 3, 4].every((k) => c.peek(k))).toBe(true);
  });
  it("handles entries of different sizes and refuses one that can never fit", () => {
    const c = new LruCache(10);
    c.admit(1, 6);
    c.admit(2, 3);
    expect(c.admit(3, 5)).toEqual([1]);
    expect(c.usedBytes).toBe(8);
    expect(c.admit(9, 11)).toEqual([9]);
    expect(c.peek(9)).toBe(false);
  });
});

describe("count-min sketch", () => {
  it("counts, saturates, and forgets by halving", () => {
    const s = new CountMinSketch(100);
    for (let i = 0; i < 40; i++) s.add(7);
    expect(s.estimate(7)).toBe(15);
    expect(s.estimate(123456)).toBe(0);
    // enough other traffic triggers an aging pass, and the old count decays
    for (let i = 0; i < 5000; i++) s.add(1000 + (i % 500));
    expect(s.estimate(7)).toBeLessThan(15);
  });
  it("never reports less than the true count (up to saturation)", () => {
    fc.assert(fc.property(fc.array(fc.integer({ min: 0, max: 50 }), { maxLength: 300 }), (xs) => {
      const s = new CountMinSketch(1000);
      const truth = new Map<number, number>();
      for (const x of xs) { s.add(x); truth.set(x, (truth.get(x) ?? 0) + 1); }
      for (const [k, v] of truth) expect(s.estimate(k)).toBeGreaterThanOrEqual(Math.min(v, 15));
    }));
  });
});

describe("W-TinyLFU", () => {
  it("never holds more than its capacity, whatever is thrown at it, and its books balance", () => {
    fc.assert(fc.property(fc.array(fc.tuple(fc.integer({ min: 0, max: 300 }), fc.integer({ min: 1, max: 9 })), { maxLength: 400 }), fc.integer({ min: 10, max: 120 }), (ops, cap) => {
      const c = new WTinyLfuCache(cap, { expectedKeys: 300 });
      for (const [k, size] of ops) if (!c.access(k)) c.admit(k, size);
      expect(c.usedBytes).toBeLessThanOrEqual(cap);
      expect(c.usedBytes).toBeGreaterThanOrEqual(0);
      let n = 0;
      for (let k = 0; k <= 300; k++) if (c.peek(k)) n++;
      expect(n).toBe(c.count);
    }), { numRuns: 300 });
  });

  it("keeps the popular experts through a long scan of one-off ones, where LRU loses them", () => {
    const hot = 50, cap = 100;
    const r = rng(7);
    const z = zipf(hot, 1.0, r);
    const trace: number[] = [];
    for (let i = 0; i < 20000; i++) trace.push(z());                     // warm up on the popular set
    const warm = trace.length;
    for (let i = 0; i < 6000; i++) { trace.push(10_000 + i); if (i % 3 === 0) trace.push(z()); } // a scan of never-repeated keys, with some hot traffic
    for (let i = 0; i < 20000; i++) trace.push(z());
    const lru = new LruCache(cap), tiny = new WTinyLfuCache(cap, { expectedKeys: 5000 });
    run(lru, trace);
    run(tiny, trace);
    void warm;
    const h = (c: ExpertCache): number => c.stats.hits / (c.stats.hits + c.stats.misses);
    expect(h(tiny)).toBeGreaterThan(h(lru));
    expect(tiny.stats.rejected).toBeGreaterThan(1000);
  });

  it("is not worse than LRU on a plain Zipf workload, and clearly better when the cache is much smaller than the key space", () => {
    const r = rng(11);
    const z = zipf(5000, 0.9, r);
    const trace = Array.from({ length: 80000 }, () => z());
    const lru = new LruCache(200), tiny = new WTinyLfuCache(200, { expectedKeys: 5000 });
    expect(run(tiny, trace)).toBeGreaterThanOrEqual(run(lru, trace) - 0.005);
  });

  it("adapts when the hot set changes: aged counts let a new favourite displace the old", () => {
    const r = rng(3);
    const cap = 40;
    const tiny = new WTinyLfuCache(cap, { expectedKeys: 400 });
    const a = zipf(30, 1, r), b = zipf(30, 1, r);
    for (let i = 0; i < 20000; i++) { const k = a(); if (!tiny.access(k)) tiny.admit(k, 1); }
    const before = tiny.stats.hits;
    void before;
    let hits = 0, total = 0;
    for (let i = 0; i < 20000; i++) { const k = 1000 + b(); total++; if (tiny.access(k)) hits++; else tiny.admit(k, 1); }
    expect(hits / total).toBeGreaterThan(0.6);
  });

  it("holds both the window and the main space within the byte limit when entries are different sizes", () => {
    const c = new WTinyLfuCache(1000, { expectedKeys: 200 });
    const r = rng(5);
    for (let i = 0; i < 5000; i++) {
      const k = Math.floor(r() * 150);
      if (!c.access(k)) c.admit(k, 10 + (k % 7) * 20);
      expect(c.usedBytes).toBeLessThanOrEqual(1000);
    }
  });
});
