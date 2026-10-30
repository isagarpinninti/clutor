/**
 * Cache policies for expert weights. Both implement `ExpertCache`; the engine and the benchmarks run the same code against either,
 * so any difference in hit rate is the policy's.
 *
 * Capacity is in bytes and entries carry their size, so a mixed set of tensor sizes works; experts of one layer are all one size.
 */

export interface CacheStats {
  hits: number;
  misses: number;
  admitted: number;
  /** A candidate that W-TinyLFU refused because the entry it would have evicted was used more often. */
  rejected: number;
  evictions: number;
}

export interface ExpertCache {
  readonly capacityBytes: number;
  readonly usedBytes: number;
  readonly count: number;
  readonly stats: CacheStats;
  /** Records an access of `key` (hit or miss) and returns whether it is cached. A hit updates recency. */
  access(key: number): boolean;
  /** Presence without touching recency or counts. */
  peek(key: number): boolean;
  /** Adds an entry after a miss. Returns the keys evicted to make room (the entry itself if the policy declined it). */
  admit(key: number, sizeBytes: number): number[];
  clear(): void;
}


/**
 * Entries in recency order with O(1) insert, touch, remove and "oldest". (A Map's iteration order would also work, but taking the
 * first entry of a Map after many deletions is slow in V8, which showed up as LRU being slower than the cleverer policies.)
 */
class LruList {
  private readonly nodes = new Map<number, { key: number; size: number; prev: number | null; next: number | null }>();
  private head: number | null = null; // oldest
  private tail: number | null = null; // newest
  get size(): number { return this.nodes.size; }
  has(key: number): boolean { return this.nodes.has(key); }
  get(key: number): number | undefined { return this.nodes.get(key)?.size; }
  private unlink(n: { prev: number | null; next: number | null }): void {
    if (n.prev === null) this.head = n.next; else (this.nodes.get(n.prev) as { next: number | null }).next = n.next;
    if (n.next === null) this.tail = n.prev; else (this.nodes.get(n.next) as { prev: number | null }).prev = n.prev;
  }
  /** Adds the key as the newest entry (moving it there if it exists). */
  set(key: number, size: number): void {
    const old = this.nodes.get(key);
    if (old) this.unlink(old);
    const n = { key, size, prev: this.tail, next: null as number | null };
    if (this.tail !== null) (this.nodes.get(this.tail) as { next: number | null }).next = key; else this.head = key;
    this.tail = key;
    this.nodes.set(key, n);
  }
  delete(key: number): boolean {
    const n = this.nodes.get(key);
    if (!n) return false;
    this.unlink(n);
    this.nodes.delete(key);
    return true;
  }
  /** The oldest entry. */
  first(): [number, number] | undefined {
    if (this.head === null) return undefined;
    const n = this.nodes.get(this.head) as { key: number; size: number };
    return [n.key, n.size];
  }
  clear(): void { this.nodes.clear(); this.head = this.tail = null; }
}

const emptyStats = (): CacheStats => ({ hits: 0, misses: 0, admitted: 0, rejected: 0, evictions: 0 });

/** Plain least-recently-used, the baseline. */
export class LruCache implements ExpertCache {
  readonly stats = emptyStats();
  usedBytes = 0;
  private readonly map = new LruList();
  constructor(readonly capacityBytes: number) {}
  get count(): number { return this.map.size; }
  peek(key: number): boolean { return this.map.has(key); }
  access(key: number): boolean {
    const size = this.map.get(key);
    if (size === undefined) { this.stats.misses++; return false; }
    this.map.set(key, size);
    this.stats.hits++;
    return true;
  }
  admit(key: number, size: number): number[] {
    if (size > this.capacityBytes) return [key];
    const evicted: number[] = [];
    const old = this.map.get(key);
    if (old !== undefined) { this.map.delete(key); this.usedBytes -= old; }
    while (this.usedBytes + size > this.capacityBytes) {
      const [k, s] = this.map.first() as [number, number];
      this.map.delete(k);
      this.usedBytes -= s;
      this.stats.evictions++;
      evicted.push(k);
    }
    this.map.set(key, size);
    this.usedBytes += size;
    this.stats.admitted++;
    return evicted;
  }
  clear(): void { this.map.clear(); this.usedBytes = 0; }
}

/** Approximate access frequencies in a small fixed space, aged by halving (the "TinyLFU" sketch). */
export class CountMinSketch {
  private readonly rows = 4;
  private readonly width: number;
  private table: Uint8Array;
  private additions = 0;
  constructor(expectedItems: number, readonly maxCount = 15) {
    let w = 64;
    while (w < expectedItems * 2) w <<= 1;
    this.width = w;
    this.table = new Uint8Array(this.rows * w);
    this.sample = Math.max(256, expectedItems * 10);
  }
  private readonly sample: number;
  private index(key: number, row: number): number {
    let h = Math.imul(key ^ (row * 0x9e3779b1), 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
    return row * this.width + (h & (this.width - 1));
  }
  add(key: number): void {
    let incremented = false;
    // Conservative update: only the smallest counters grow, which keeps over-counting low.
    let min = 255;
    for (let r = 0; r < this.rows; r++) min = Math.min(min, this.table[this.index(key, r)] as number);
    if (min < this.maxCount) {
      for (let r = 0; r < this.rows; r++) {
        const i = this.index(key, r);
        if (this.table[i] === min) { this.table[i] = min + 1; incremented = true; }
      }
    }
    if (incremented && ++this.additions >= this.sample) this.age();
  }
  estimate(key: number): number {
    let min = 255;
    for (let r = 0; r < this.rows; r++) min = Math.min(min, this.table[this.index(key, r)] as number);
    return min;
  }
  /** Halves every counter, so old popularity fades and a changing workload can displace what was hot. */
  private age(): void {
    for (let i = 0; i < this.table.length; i++) this.table[i] = (this.table[i] as number) >> 1;
    this.additions = Math.floor(this.additions / 2);
  }
}

export interface TinyLfuOptions {
  /** Share of capacity for the admission window (default 1%). */
  windowFraction?: number;
  /** Share of the main space protected from one-hit wonders (default 80%). */
  protectedFraction?: number;
  /** Rough number of distinct keys, to size the sketch. */
  expectedKeys: number;
}

/**
 * W-TinyLFU: a small LRU window takes every new entry; when it overflows, its oldest entry competes with the main space's eviction
 * victim and is admitted only if it has been used more often. Frequently used experts therefore survive a long scan of rarely used
 * ones, where LRU would flush them. The main space is segmented (probation and protected) so an entry must be hit twice to be safe.
 */
export class WTinyLfuCache implements ExpertCache {
  readonly stats = emptyStats();
  usedBytes = 0;
  private readonly sketch: CountMinSketch;
  private readonly window = new LruList();
  private readonly probation = new LruList();
  private readonly prot = new LruList();
  private windowBytes = 0;
  private probationBytes = 0;
  private protBytes = 0;
  /** The entry most recently demoted from the window: the one that has to win its place against the eviction victim. */
  private candidate: number | undefined;
  private windowCap: number;
  private protCap: number;
  private windowFraction: number;
  private readonly protectedFraction: number;

  constructor(readonly capacityBytes: number, opts: TinyLfuOptions) {
    this.windowFraction = opts.windowFraction ?? 0.01;
    this.protectedFraction = opts.protectedFraction ?? 0.8;
    this.windowCap = Math.max(1, Math.floor(capacityBytes * this.windowFraction));
    this.protCap = Math.floor((capacityBytes - this.windowCap) * this.protectedFraction);
    this.sketch = new CountMinSketch(opts.expectedKeys);
  }

  /** The window as a share of capacity right now. */
  get currentWindowFraction(): number { return this.windowFraction; }

  /** Moves the boundary between the recency window and the frequency-protected main space. Entries are not evicted by this alone. */
  setWindowFraction(fraction: number): void {
    this.windowFraction = Math.min(0.9, Math.max(0.01, fraction));
    this.windowCap = Math.max(1, Math.floor(this.capacityBytes * this.windowFraction));
    this.protCap = Math.floor((this.capacityBytes - this.windowCap) * this.protectedFraction);
    while (this.protBytes > this.protCap && this.prot.size > 0) {
      const [k, sz] = this.prot.first() as [number, number];
      this.prot.delete(k);
      this.protBytes -= sz;
      this.probation.set(k, sz);
      this.probationBytes += sz;
    }
  }

  get count(): number { return this.window.size + this.probation.size + this.prot.size; }
  peek(key: number): boolean { return this.window.has(key) || this.probation.has(key) || this.prot.has(key); }

  access(key: number): boolean {
    this.sketch.add(key);
    const w = this.window.get(key);
    if (w !== undefined) { this.window.delete(key); this.window.set(key, w); this.stats.hits++; return true; }
    const p = this.prot.get(key);
    if (p !== undefined) { this.prot.delete(key); this.prot.set(key, p); this.stats.hits++; return true; }
    const pr = this.probation.get(key);
    if (pr !== undefined) {
      this.probation.delete(key);
      this.probationBytes -= pr;
      this.prot.set(key, pr);
      this.protBytes += pr;
      // The protected segment is bounded; its oldest entries drop back to probation, where they can still be evicted.
      while (this.protBytes > this.protCap) {
        const [k, s] = this.prot.first() as [number, number];
        this.prot.delete(k);
        this.protBytes -= s;
        this.probation.set(k, s);
        this.probationBytes += s;
      }
      this.stats.hits++;
      return true;
    }
    this.stats.misses++;
    return false;
  }

  admit(key: number, size: number): number[] {
    if (size > this.capacityBytes) return [key];
    const evicted: number[] = [];
    this.window.set(key, size);
    this.windowBytes += size;
    this.usedBytes += size;
    this.stats.admitted++;
    while (this.windowBytes > this.windowCap && this.window.size > 0) {
      const [cand, cs] = this.window.first() as [number, number];
      this.window.delete(cand);
      this.windowBytes -= cs;
      this.probation.set(cand, cs);
      this.probationBytes += cs;
      this.candidate = cand;
    }
    // Make room in the whole cache: each time the main space is over capacity, the newest probation entry fights the oldest.
    while (this.usedBytes > this.capacityBytes) {
      const victim = this.pickVictim();
      if (victim === undefined) break;
      this.remove(victim);
      this.stats.evictions++;
      evicted.push(victim);
    }
    return evicted;
  }

  /** The probation segment's oldest entry, unless the newest candidate (just demoted from the window) is rarer: then the candidate loses. */
  private pickVictim(): number | undefined {
    const oldest = this.probation.first()?.[0];
    if (oldest === undefined) return this.window.first()?.[0] ?? this.prot.first()?.[0];
    const newest = this.candidate !== undefined && this.probation.has(this.candidate) ? this.candidate : undefined;
    if (newest !== undefined && newest !== oldest) {
      if (this.sketch.estimate(newest) <= this.sketch.estimate(oldest)) { this.stats.rejected++; return newest; }
    }
    return oldest;
  }
  private remove(key: number): void {
    const w = this.window.get(key);
    if (w !== undefined) { this.window.delete(key); this.windowBytes -= w; this.usedBytes -= w; return; }
    const pr = this.probation.get(key);
    if (pr !== undefined) { this.probation.delete(key); this.probationBytes -= pr; this.usedBytes -= pr; return; }
    const p = this.prot.get(key);
    if (p !== undefined) { this.prot.delete(key); this.protBytes -= p; this.usedBytes -= p; }
  }
  clear(): void {
    this.window.clear(); this.probation.clear(); this.prot.clear();
    this.windowBytes = this.probationBytes = this.protBytes = this.usedBytes = 0;
  }
}

/**
 * W-TinyLFU whose window size follows the workload. The best window depends on it: when recent entries are reused (consecutive
 * tokens pick similar experts) a large window, which behaves like LRU, wins; when a few entries are reused a lot among many
 * one-offs a small window, which behaves like LFU, wins. Hill climbing gets stuck on this landscape, so instead several key-only
 * "ghost" caches with different windows replay the same accesses, and the real cache takes the window of whichever ghost has had
 * the best recent hit rate (with some hysteresis, so it does not flap).
 */
export class AdaptiveTinyLfuCache implements ExpertCache {
  private readonly real: WTinyLfuCache;
  private readonly ghosts: Array<{ fraction: number; cache: WTinyLfuCache; hits: number; score: number }>;
  private accesses = 0;
  private typicalSize = 1;
  private readonly period: number;

  constructor(readonly capacityBytes: number, opts: Omit<TinyLfuOptions, "windowFraction"> & { windows?: number[]; period?: number }) {
    const windows = opts.windows ?? [0.01, 0.25, 0.9];
    this.real = new WTinyLfuCache(capacityBytes, { ...opts, windowFraction: windows[0] as number });
    this.ghosts = windows.map((fraction) => ({ fraction, cache: new WTinyLfuCache(capacityBytes, { ...opts, windowFraction: fraction }), hits: 0, score: -1 }));
    this.period = opts.period ?? 4000;
  }
  get stats(): CacheStats { return this.real.stats; }
  get usedBytes(): number { return this.real.usedBytes; }
  get count(): number { return this.real.count; }
  get currentWindowFraction(): number { return this.real.currentWindowFraction; }
  peek(key: number): boolean { return this.real.peek(key); }
  access(key: number): boolean {
    // Each ghost behaves as if it were the cache: it takes every access, and fetches (admits) on its own misses.
    for (const g of this.ghosts) {
      if (g.cache.access(key)) g.hits++;
      else g.cache.admit(key, this.typicalSize);
    }
    const hit = this.real.access(key);
    if (++this.accesses >= this.period) this.decide();
    return hit;
  }
  admit(key: number, size: number): number[] {
    this.typicalSize = size;
    return this.real.admit(key, size);
  }
  private decide(): void {
    for (const g of this.ghosts) {
      const rate = g.hits / this.accesses;
      g.score = g.score < 0 ? rate : 0.7 * g.score + 0.3 * rate;
      g.hits = 0;
    }
    this.accesses = 0;
    const current = this.ghosts.find((g) => g.fraction === this.real.currentWindowFraction) ?? (this.ghosts[0] as (typeof this.ghosts)[number]);
    const best = this.ghosts.reduce((a, b) => (b.score > a.score ? b : a));
    if (best !== current && best.score > current.score + 0.01) this.real.setWindowFraction(best.fraction);
  }
  clear(): void {
    this.real.clear();
    for (const g of this.ghosts) { g.cache.clear(); g.hits = 0; g.score = -1; }
    this.accesses = 0;
  }
}
