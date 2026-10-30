import { AdaptiveTinyLfuCache, LruCache, WTinyLfuCache, type ExpertCache } from "../cache/policies";
import { expertSizeBytes, type ModelLayout } from "../gguf/layout";
import type { TokenRouting } from "./trace";

/**
 * A simulator of out-of-core expert streaming: replays a routing trace against a cache, a prefetcher, optional router-drift-bounded
 * substitution and a pipelined disk, and reports what was read, what was hit and how long the disk made the model wait.
 *
 * What is real: the cache policies, the prefetch predictor, the substitution rule, the request coalescing and every count. What is
 * modelled: time, from the disk speed and per-layer compute time you give it (or measure with `clutor bench`). It does not run a
 * model, so it says nothing about output quality; for substitution it reports the share of routing probability served exactly.
 */

export interface DiskModel {
  bytesPerSec: number;
  /** Fixed cost per read request (seek, syscall, queueing). */
  requestLatencySec: number;
}

export interface StreamerConfig {
  policy: "lru" | "wtinylfu" | "adaptive";
  /** `wtinylfu` only: a fixed share of the cache used as the admission window. */
  windowFraction?: number;
  cacheBytes: number;
  /** Record, for this layer, which experts are cached at the end and how often each was chosen (for a heatmap). */
  snapshotLayer?: number;
  disk: DiskModel;
  /** Compute time per MoE layer per token (attention and the experts' arithmetic). */
  computePerLayerSec: number;
  /** Merge reads of experts that sit next to each other in the file into one request. */
  coalesce: boolean;
  /**
   * Predict the next layer's experts from this layer's choices and read them while this layer computes. A read is issued only if the
   * predictor is at least `minConfidence` sure, and only into disk time that would otherwise be idle during this layer's compute: a
   * prefetch that delays a demand read costs more than it saves.
   */
  lookahead: { enabled: boolean; perLayer: number; minConfidence: number };
  /**
   * Replace a missing expert with a cached one that the router scored almost as high. `maxSubstitutions` per layer per token (J),
   * only among the best `candidates` (M), only when the probability gap is at most `maxDeltaP` (tau).
   */
  smartRoute: { enabled: boolean; maxSubstitutions: number; candidates: number; maxDeltaP: number };
}

export const BASELINE: Omit<StreamerConfig, "cacheBytes" | "disk" | "computePerLayerSec"> = {
  policy: "lru", coalesce: false, lookahead: { enabled: false, perLayer: 0, minConfidence: 1 }, smartRoute: { enabled: false, maxSubstitutions: 0, candidates: 0, maxDeltaP: 0 },
};
export const CLUTOR_DEFAULTS: Omit<StreamerConfig, "cacheBytes" | "disk" | "computePerLayerSec"> = {
  policy: "adaptive", coalesce: true, lookahead: { enabled: true, perLayer: 6, minConfidence: 0.4 }, smartRoute: { enabled: false, maxSubstitutions: 2, candidates: 12, maxDeltaP: 0.015 },
};

export interface StreamMetrics {
  tokens: number;
  /** Expert uses the model asked for (layers x experts used x tokens). */
  expertUses: number;
  hits: number;
  /** Uses served by a different, cached expert (only with substitution). */
  substituted: number;
  /** Uses that had to wait for a read. */
  demandMisses: number;
  /** Uses whose expert had been prefetched (fully or partly hidden). */
  prefetchUsed: number;
  prefetchIssued: number;
  prefetchWasted: number;
  bytesRead: number;
  requests: number;
  /** Seconds the model spent waiting for the disk. */
  stallSec: number;
  totalSec: number;
  /** Modelled generation speed given the disk and compute times above. */
  tokensPerSec: number;
  /** Share of uses served by the exact expert the router chose, from cache or disk. */
  exactShare: number;
  /** Share of router probability mass that went to exactly the chosen experts (1 = no substitution). */
  exactMassShare: number;
  /** Uses whose expert was already in the cache or had been prefetched into it, as a share of all uses. */
  hitRate: number;
  effectiveHitRate: number;
  bytesPerToken: number;
  policy: string;
  /** Present when `snapshotLayer` was set: per expert, 1 if cached at the end of the run; and how many times it was chosen. */
  snapshot?: { layer: number; cached: number[]; uses: number[] };
}

export function simulateStreaming(layout: ModelLayout, trace: TokenRouting[], cfg: StreamerConfig): StreamMetrics {
  const E = layout.expertCount;
  const eBytes = expertSizeBytes(layout);
  const rangesPer = layout.layers.find((l) => l.experts.length > 0)?.experts[0]?.length ?? 1;
  const key = (layer: number, e: number): number => layer * E + e;
  const keys = Math.max(1024, layout.layers.length * E);
  const cache: ExpertCache =
    cfg.policy === "lru" ? new LruCache(cfg.cacheBytes)
    : cfg.policy === "adaptive" ? new AdaptiveTinyLfuCache(cfg.cacheBytes, { expectedKeys: keys })
    : new WTinyLfuCache(cfg.cacheBytes, { expectedKeys: keys, ...(cfg.windowFraction !== undefined ? { windowFraction: cfg.windowFraction } : {}) });

  const m: StreamMetrics = {
    tokens: 0, expertUses: 0, hits: 0, substituted: 0, demandMisses: 0, prefetchUsed: 0, prefetchIssued: 0, prefetchWasted: 0, bytesRead: 0, requests: 0, stallSec: 0, totalSec: 0,
    tokensPerSec: 0, exactShare: 0, exactMassShare: 0, hitRate: 0, effectiveHitRate: 0, bytesPerToken: 0, policy: cfg.policy,
  };

  let now = 0;
  let diskFreeAt = 0;
  /** Experts a prefetch brought in (they occupy cache space) and nothing has used yet: key -> time the data is in memory. */
  const prefetched = new Map<number, number>();
  const snapUses = cfg.snapshotLayer !== undefined ? new Array<number>(E).fill(0) : null;
  let massTotal = 0;
  let massExact = 0;

  // Learned from the trace as it plays: after expert a at layer L, how often expert b followed at the next MoE layer.
  const moeLayerIdx = new Map<number, number>();
  const moeLayers: number[] = (trace[0] ?? []).map((x) => x.layer);
  moeLayers.forEach((l, i) => moeLayerIdx.set(l, i));
  const after = moeLayers.map(() => new Map<number, Map<number, number>>());
  const totals = moeLayers.map(() => new Map<number, number>());
  const popularity = moeLayers.map(() => new Float32Array(E));

  const issue = (keys: number[]): Map<number, number> => {
    // Experts are laid out by (layer, expert); sorting by key makes neighbours adjacent, which is what coalescing exploits.
    const sorted = [...keys].sort((x, y) => x - y);
    const done = new Map<number, number>();
    let i = 0;
    while (i < sorted.length) {
      let j = i;
      while (cfg.coalesce && j + 1 < sorted.length && (sorted[j + 1] as number) === (sorted[j] as number) + 1 && Math.floor((sorted[j + 1] as number) / E) === Math.floor((sorted[i] as number) / E)) j++;
      const run = sorted.slice(i, j + 1);
      const requests = cfg.coalesce ? rangesPer : run.length * rangesPer;
      const bytes = run.length * eBytes;
      const start = Math.max(now, diskFreeAt);
      const finish = start + requests * cfg.disk.requestLatencySec + bytes / cfg.disk.bytesPerSec;
      diskFreeAt = finish;
      m.bytesRead += bytes;
      m.requests += requests;
      for (const k of run) done.set(k, finish);
      i = j + 1;
    }
    return done;
  };
  /** Admits an entry and forgets any prefetch that the admission pushed out (it was read for nothing). */
  const admit = (k: number): void => {
    for (const out of cache.admit(k, eBytes)) {
      if (prefetched.delete(out)) m.prefetchWasted++;
    }
  };

  for (const token of trace) {
    m.tokens++;
    for (let li = 0; li < token.length; li++) {
      const lr = token[li] as TokenRouting[number];
      const used = Math.min(layout.expertsUsed, lr.experts.length);
      const cached: number[] = [];
      const missing: number[] = [];
      let subsLeft = cfg.smartRoute.enabled ? cfg.smartRoute.maxSubstitutions : 0;
      const chosen = new Set<number>();
      for (let r = 0; r < used; r++) chosen.add(lr.experts[r] as number);

      for (let r = 0; r < used; r++) {
        const e = lr.experts[r] as number;
        const k = key(lr.layer, e);
        const p = lr.probs[r] as number;
        m.expertUses++;
        massTotal += p;
        if (snapUses && lr.layer === cfg.snapshotLayer) snapUses[e] = (snapUses[e] as number) + 1;
        if (cache.access(k)) { cached.push(k); massExact += p; continue; }
        // Missing. Try a cached stand-in from the router's next-best choices.
        if (subsLeft > 0) {
          const lim = Math.min(lr.experts.length, cfg.smartRoute.candidates);
          let best = -1;
          for (let c = used; c < lim; c++) {
            const ce = lr.experts[c] as number;
            if (chosen.has(ce) || !cache.peek(key(lr.layer, ce))) continue;
            if (p - (lr.probs[c] as number) <= cfg.smartRoute.maxDeltaP) { best = c; break; }
          }
          if (best >= 0) {
            const ce = lr.experts[best] as number;
            chosen.add(ce);
            cache.access(key(lr.layer, ce));
            cached.push(key(lr.layer, ce));
            subsLeft--;
            m.substituted++;
            continue;
          }
        }
        massExact += p;
        missing.push(k);
      }

      // Wait for what this layer needs. Demand reads queue behind earlier prefetches on the one disk.
      const demandDone = issue(missing);
      let ready = now;
      for (const k of cached) {
        const t = prefetched.get(k);
        if (t !== undefined) { ready = Math.max(ready, t); m.prefetchUsed++; prefetched.delete(k); }
        else m.hits++;
      }
      for (const k of missing) {
        ready = Math.max(ready, demandDone.get(k) as number);
        m.demandMisses++;
        admit(k);
      }
      m.stallSec += ready - now;
      now = ready;

      // Learn how this layer's choices relate to the next layer's, and read ahead while this layer computes.
      const next = token[li + 1];
      if (next) {
        const ni = (moeLayerIdx.get(lr.layer) ?? 0) + 1;
        const nextUsed = Math.min(layout.expertsUsed, next.experts.length);
        const row = after[ni - 1] as Map<number, Map<number, number>>;
        for (let r = 0; r < used; r++) {
          const a = lr.experts[r] as number;
          let cnt = row.get(a);
          if (!cnt) row.set(a, (cnt = new Map()));
          for (let q = 0; q < nextUsed; q++) { const b = next.experts[q] as number; cnt.set(b, (cnt.get(b) ?? 0) + 1); }
          const tot = totals[ni - 1] as Map<number, number>;
          tot.set(a, (tot.get(a) ?? 0) + 1);
        }
        const pop = popularity[ni] as Float32Array;
        for (let q = 0; q < nextUsed; q++) pop[next.experts[q] as number]! += 1;

        if (cfg.lookahead.enabled) {
          // P(expert b is used next | the experts chosen here): 1 - product over chosen a of (1 - count(a,b)/count(a)).
          const tot = totals[ni - 1] as Map<number, number>;
          const miss = new Map<number, number>();
          for (let r = 0; r < used; r++) {
            const a = lr.experts[r] as number;
            const n = tot.get(a) ?? 0;
            if (n === 0) continue;
            for (const [b, c] of row.get(a) ?? []) miss.set(b, (miss.get(b) ?? 1) * (1 - Math.min(1, c / n)));
          }
          const ranked = [...miss.entries()].map(([b, q]) => [b, 1 - q] as const).filter(([, conf]) => conf >= cfg.lookahead.minConfidence).sort((x, y) => y[1] - x[1]);
          // Idle disk time during this layer's compute, in whole experts.
          const idle = Math.max(0, cfg.computePerLayerSec - Math.max(0, diskFreeAt - now));
          const room = Math.floor((idle * cfg.disk.bytesPerSec) / Math.max(1, eBytes));
          const want: number[] = [];
          for (const [b] of ranked) {
            if (want.length >= Math.min(cfg.lookahead.perLayer, room)) break;
            const k = key(next.layer, b);
            if (!cache.peek(k)) want.push(k);
          }
          if (want.length) {
            const done = issue(want);
            for (const [k, t] of done) {
              m.prefetchIssued++;
              admit(k);
              if (cache.peek(k)) prefetched.set(k, t);
              else m.prefetchWasted++; // the policy would not keep it
            }
          }
        }
      }
      now += cfg.computePerLayerSec;
    }
  }
  m.prefetchWasted += prefetched.size;
  m.totalSec = now;
  m.tokensPerSec = m.tokens / Math.max(now, 1e-12);
  m.hitRate = m.expertUses ? (m.hits + m.prefetchUsed) / m.expertUses : 0;
  m.effectiveHitRate = m.expertUses ? (m.hits + m.prefetchUsed + m.substituted) / m.expertUses : 0;
  m.exactShare = m.expertUses ? (m.expertUses - m.substituted) / m.expertUses : 1;
  m.exactMassShare = massTotal ? massExact / massTotal : 1;
  m.bytesPerToken = m.tokens ? m.bytesRead / m.tokens : 0;
  if (snapUses && cfg.snapshotLayer !== undefined) m.snapshot = { layer: cfg.snapshotLayer, cached: Array.from({ length: E }, (_, e) => (cache.peek(key(cfg.snapshotLayer as number, e)) ? 1 : 0)), uses: snapUses };
  return m;
}
