import { expertSizeBytes, type ModelLayout } from "../gguf/layout";

/**
 * The memory planner: given a model and a RAM ceiling, where does every byte go, and what is the fastest the model could possibly
 * run? The second number is a bound from arithmetic (bytes that must come off the disk per token, divided by the disk's speed), not
 * a promise: real speed is lower, and nothing here is a measurement unless a field says so.
 */

export const GiB = 1024 ** 3;
export const MiB = 1024 ** 2;

export type Strategy = "fully-resident" | "dense-resident-experts-streamed" | "dense-ring-experts-streamed" | "does-not-fit";

export interface PlanOptions {
  /** Physical RAM of the machine. */
  hostRamBytes: number;
  /** The most Clutor may use. Default: host RAM minus OS headroom. */
  ramCeilingBytes?: number;
  /** RAM left to the operating system and other programs. Default 1.5 GiB, or 15% of RAM if that is more. */
  osHeadroomBytes?: number;
  contextTokens?: number;
  /** KV-cache element size in bits: 16, 8 or 4. */
  kvBits?: 16 | 8 | 4;
  /** Layers of dense weights kept in the streaming window (double buffering needs 2). */
  ringLayers?: number;
  /** Sequential read speed of the disk holding the model, bytes per second. */
  diskBytesPerSec?: number;
  /** Whether the speed above was measured on this machine. */
  diskMeasured?: boolean;
  /** Fixed memory the runtime itself needs (compute buffers, tokeniser, the app). */
  runtimeOverheadBytes?: number;
  /** Fraction of expert reads served from the cache, if known (for example from a simulation). */
  expectedHitRate?: number;
}

export interface Component {
  id: "resident" | "dense" | "dense-ring" | "kv" | "runtime" | "expert-cache" | "os";
  label: string;
  bytes: number;
}

export interface SpeedBound {
  hitRate: number;
  bytesPerToken: number;
  /** Upper bound on generated tokens per second from disk bandwidth alone. */
  maxTokensPerSec: number;
}

export interface MemoryPlan {
  strategy: Strategy;
  fits: boolean;
  ceilingBytes: number;
  usedBytes: number;
  components: Component[];
  shortfallBytes: number;
  expertCache: { bytes: number; slots: number; expertBytes: number; totalExperts: number; fractionOfExperts: number; slotsPerToken: number };
  perToken: { denseStreamBytes: number; activeExpertBytes: number; routedLayers: number; expertsUsed: number };
  disk: { bytesPerSec: number; measured: boolean };
  speed: SpeedBound[];
  warnings: string[];
  approximate: boolean;
}

export function kvCacheBytes(layout: ModelLayout, contextTokens: number, kvBits: number): number {
  const k = layout.kv;
  return Math.ceil(2 * k.layers * k.kvHeads * k.headDim * contextTokens * (kvBits / 8));
}

/** What the operating system keeps: 22% of physical RAM, but never under 1.8 GiB or over 3.5 GiB. Reserved before the dense window and the expert cache are sized. */
export const defaultHeadroom = (hostRamBytes: number): number => Math.round(Math.min(3.5 * GiB, Math.max(1.8 * GiB, hostRamBytes * 0.22)));

export function planMemory(layout: ModelLayout, o: PlanOptions): MemoryPlan {
  const headroom = o.osHeadroomBytes ?? defaultHeadroom(o.hostRamBytes);
  const ceiling = Math.min(o.ramCeilingBytes ?? o.hostRamBytes - headroom, o.hostRamBytes);
  const ctx = Math.max(1, Math.min(o.contextTokens ?? 8192, layout.kv.maxContext || o.contextTokens || 8192));
  const kv = kvCacheBytes(layout, ctx, o.kvBits ?? 16);
  const runtime = o.runtimeOverheadBytes ?? 384 * MiB;
  const ringLayers = Math.max(1, o.ringLayers ?? 2);
  const disk = { bytesPerSec: o.diskBytesPerSec ?? 3 * GiB, measured: o.diskMeasured ?? false };

  const resident = layout.residentBytes;
  const denseAll = layout.layers.reduce((s, l) => s + l.denseBytes, 0);
  const denseMax = layout.layers.reduce((m, l) => Math.max(m, l.denseBytes), 0);
  const ring = Math.min(denseAll, ringLayers * denseMax);
  const eBytes = expertSizeBytes(layout);
  const routed = layout.layers.filter((l) => l.experts.length > 0);
  const totalExperts = routed.reduce((s, l) => s + l.experts.length, 0);
  const expertsAll = routed.reduce((s, l) => s + l.experts.length * l.expertBytes, 0);
  // The os share is the part of the host that is not Clutor's; the ceiling already excludes it when it is defaulted.
  const base = resident + kv + runtime;
  const warnings: string[] = [];

  let strategy: Strategy;
  let denseBytes = 0;
  let ringBytes = 0;
  if (base + denseAll + expertsAll <= ceiling) {
    strategy = "fully-resident";
    denseBytes = denseAll;
  } else if (base + denseAll + eBytes * Math.max(1, layout.expertsUsed) * 2 <= ceiling) {
    strategy = "dense-resident-experts-streamed";
    denseBytes = denseAll;
  } else if (base + ring <= ceiling) {
    strategy = "dense-ring-experts-streamed";
    ringBytes = ring;
  } else strategy = "does-not-fit";

  const fixed = base + denseBytes + ringBytes;
  const cacheBytes = strategy === "does-not-fit" ? 0 : strategy === "fully-resident" ? expertsAll : Math.max(0, ceiling - fixed);
  const slots = eBytes > 0 ? Math.min(totalExperts, Math.floor(cacheBytes / eBytes)) : 0;

  const components: Component[] = [
    { id: "resident", label: "Embeddings, output head, routers (always in RAM)", bytes: resident },
    ...(denseBytes > 0 ? [{ id: "dense" as const, label: "Dense layer weights (attention, shared experts)", bytes: denseBytes }] : []),
    ...(ringBytes > 0 ? [{ id: "dense-ring" as const, label: `Dense streaming window (${ringLayers} layers)`, bytes: ringBytes }] : []),
    { id: "kv", label: `KV cache (${ctx.toLocaleString("en-US")} tokens, ${o.kvBits ?? 16}-bit)`, bytes: kv },
    { id: "runtime", label: "Double-Buffer DMA (runtime buffers)", bytes: runtime },
    { id: "expert-cache", label: "Expert cache", bytes: strategy === "fully-resident" ? expertsAll : slots * eBytes },
    { id: "os", label: "Left for the operating system", bytes: o.hostRamBytes - ceiling },
  ];
  const used = components.filter((c) => c.id !== "os").reduce((s, c) => s + c.bytes, 0);

  const routedLayers = routed.length;
  const activeExpertBytes = routedLayers * layout.expertsUsed * eBytes;
  const denseStreamBytes = strategy === "dense-ring-experts-streamed" ? denseAll : 0;
  const speed: SpeedBound[] = [0, 0.5, 0.8, 0.95, 0.99]
    .concat(o.expectedHitRate !== undefined ? [o.expectedHitRate] : [])
    .sort((a, b) => a - b)
    .filter((h, i, arr) => arr.indexOf(h) === i)
    .map((hitRate) => {
      const bytesPerToken = strategy === "fully-resident" ? 0 : denseStreamBytes + activeExpertBytes * (1 - hitRate);
      return { hitRate, bytesPerToken, maxTokensPerSec: bytesPerToken === 0 ? Infinity : disk.bytesPerSec / bytesPerToken };
    });

  if (strategy === "does-not-fit") warnings.push(`Even the always-resident weights, the KV cache and the runtime need ${fmtGiB(base + ring)}; the ceiling is ${fmtGiB(ceiling)}.`);
  if (strategy !== "does-not-fit" && strategy !== "fully-resident" && routedLayers > 0 && slots < routedLayers * layout.expertsUsed) {
    warnings.push(`The cache holds ${slots} experts but one token needs ${routedLayers * layout.expertsUsed}: every token will re-read most of its experts, so the cache cannot help much.`);
  }
  if (strategy === "dense-ring-experts-streamed") {
    warnings.push(`The dense weights (${fmtGiB(denseAll)}) do not fit and are re-read from disk for every token. That cost is the same whatever the cache does.`);
  }
  if (strategy !== "fully-resident" && strategy !== "does-not-fit") {
    const best = speed[speed.length - 1] as SpeedBound;
    if (best.maxTokensPerSec < 2) warnings.push(`Even with a ${(best.hitRate * 100).toFixed(0)}% cache hit rate the disk caps generation at ${best.maxTokensPerSec.toFixed(2)} tokens per second.`);
  }
  if (layout.approximate) warnings.push("Sizes are computed from the published architecture, not from a file; a real GGUF will differ by a few percent.");
  if (!disk.measured && strategy !== "fully-resident") warnings.push("Disk speed is an assumption (3 GiB/s). Run `clutor bench` to measure this machine.");
  if (layout.unsizedTensors > 0) warnings.push(`${layout.unsizedTensors} tensors use a quantisation this reader cannot size; totals are too small.`);

  return {
    strategy, fits: strategy !== "does-not-fit", ceilingBytes: ceiling, usedBytes: used, components, shortfallBytes: strategy === "does-not-fit" ? base + ring - ceiling : 0,
    expertCache: { bytes: slots * eBytes, slots, expertBytes: eBytes, totalExperts, fractionOfExperts: totalExperts ? slots / totalExperts : 0, slotsPerToken: routedLayers * layout.expertsUsed },
    perToken: { denseStreamBytes, activeExpertBytes, routedLayers, expertsUsed: layout.expertsUsed },
    disk, speed, warnings, approximate: layout.approximate,
  };
}

export const fmtGiB = (b: number): string => `${(b / GiB).toFixed(b >= 10 * GiB ? 1 : 2)} GiB`;

/** A size in the unit that reads best: bytes, KiB, MiB or GiB. */
export function fmtBytes(b: number): string {
  if (b >= GiB) return fmtGiB(b);
  if (b >= MiB) return `${(b / MiB).toFixed(b >= 100 * MiB ? 0 : 1)} MiB`;
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KiB`;
  return `${b} B`;
}
