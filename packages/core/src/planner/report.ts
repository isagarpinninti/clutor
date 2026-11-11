import type { ModelLayout } from "../gguf/layout";
import { GiB, type MemoryPlan } from "./plan";

/**
 * A plan read the way a person decides: will it run, how fast, is that good enough for what I want to do, and how sure is this?
 *
 * Every number carries a basis, and the three are never mixed:
 *   measured     observed on this machine (only the disk speed can be, here; generation speed is measured by `clutor bench`)
 *   estimated    worked out by the model below from the file and the machine; a guide, not a promise
 *   theoretical  a ceiling from physics (disk bytes per second divided by bytes needed per token), never reached in practice
 *   unknown      cannot be determined from what is given
 */
export type Basis = "measured" | "estimated" | "theoretical" | "unknown";
export interface Labelled<T> { value: T; basis: Basis; note?: string }
export type Rating = "good" | "usable" | "slow" | "poor";
export interface TaskFit { task: "chat" | "coding" | "background" | "long-generation"; rating: Rating; why: string }
export type Confidence = "high" | "medium" | "low";

export interface PlanReport {
  verdict: { canRun: boolean; mode: "all in memory" | "disk streaming" | "does not fit"; headline: string };
  memory: { modelBytes: number; ceilingBytes: number; usedBytes: number };
  /** Generation speed in tokens per second: a cautious and an optimistic figure. Absent when the model does not fit. */
  speed?: { low: Labelled<number>; high: Labelled<number> };
  /** Time to load what must be resident before the first token. */
  startSeconds?: Labelled<number>;
  /** Fastest the disk allows with nothing cached: a ceiling, not a forecast. */
  diskCeilingTps?: Labelled<number>;
  cache: { recommendedBytes: Labelled<number>; minimumBytes: Labelled<number> };
  suitability: TaskFit[];
  confidence: { level: Confidence; reasons: string[] };
  why: string[];
}

/** Memory-read speed assumed when none was measured; an ordinary laptop's order of magnitude, and labelled as an assumption in the output. */
export const ASSUMED_MEMORY_BYTES_PER_SEC = 40e9;

export interface ReportOptions { memBytesPerSec?: number; measuredTps?: number }

/** Tokens per second: time to read the weights a token uses from memory, plus the share of its experts that are not cached from the disk. Hit rate = cached share ^ 0.6, fitted to one measurement. */
function estimateTps(layout: ModelLayout, p: MemoryPlan, memBytesPerSec: number): number {
  const moe = layout.expertCount > 0;
  const expertAll = p.expertCache.expertBytes * p.expertCache.totalExperts;
  const active = p.perToken.activeExpertBytes;
  const memBytes = moe ? Math.max(0, layout.totalBytes - expertAll) + active : layout.totalBytes;
  const f = p.strategy === "fully-resident" ? 1 : Math.min(1, Math.max(0, p.expertCache.fractionOfExperts));
  const hit = moe ? Math.min(0.97, Math.pow(f, 0.6)) : 1;
  const disk = p.strategy === "fully-resident" ? 0 : p.perToken.denseStreamBytes + active * (1 - hit);
  const secs = memBytes / memBytesPerSec + disk / Math.max(1, p.disk.bytesPerSec);
  return secs > 0 ? 1 / secs : 0;
}

const rate = (tps: number, good: number, usable: number, slow: number): Rating => (tps >= good ? "good" : tps >= usable ? "usable" : tps >= slow ? "slow" : "poor");

export function suitability(lowTps: number, highTps: number): TaskFit[] {
  const t = (n: number): string => (n >= 10 ? n.toFixed(0) : n.toFixed(1));
  const at = `${t(lowTps)} to ${t(highTps)} tokens/s`;
  return [
    { task: "chat", rating: rate(lowTps, 15, 5, 2), why: lowTps >= 5 ? `${at}: replies keep up with reading` : `${at}: noticeable waiting on every reply` },
    { task: "coding", rating: rate(highTps, 5, 2, 0.5), why: `${at}: a small task needs about 3,000 generated tokens` },
    { task: "background", rating: rate(highTps, 0.5, 0.1, 0.02), why: `${at}: fine when you are not waiting for it` },
    { task: "long-generation", rating: rate(highTps, 2, 0.5, 0.1), why: `${at}: a long answer is thousands of tokens` },
  ];
}

export function buildReport(layout: ModelLayout, p: MemoryPlan, o: ReportOptions = {}): PlanReport {
  const mem = o.memBytesPerSec ?? ASSUMED_MEMORY_BYTES_PER_SEC;
  const memAssumed = o.memBytesPerSec === undefined;
  const mode = !p.fits ? "does not fit" : p.strategy === "fully-resident" ? "all in memory" : "disk streaming";
  const why: string[] = [];
  const confidenceReasons: string[] = [];

  const report: PlanReport = {
    verdict: { canRun: p.fits, mode, headline: !p.fits ? "Does not fit in memory" : mode === "all in memory" ? "Runs entirely from memory" : "Runs, but needs the disk: unused experts stream from SSD" },
    memory: { modelBytes: layout.totalBytes, ceilingBytes: p.ceilingBytes, usedBytes: p.usedBytes },
    cache: {
      recommendedBytes: { value: p.expertCache.bytes, basis: "estimated", note: "what the plan can give the expert cache within the memory ceiling" },
      minimumBytes: { value: p.perToken.activeExpertBytes, basis: "estimated", note: "one token's experts; below this the same experts are re-read every token" },
    },
    suitability: [],
    confidence: { level: "low", reasons: confidenceReasons },
    why,
  };

  if (!p.fits) {
    why.push(`The model needs more than the ${(p.ceilingBytes / GiB).toFixed(1)} GiB available${p.shortfallBytes > 0 ? ` (short by ${(p.shortfallBytes / GiB).toFixed(1)} GiB)` : ""}.`);
    why.push(layout.expertCount > 0 ? "Even with experts streamed from disk the always-used weights do not fit." : "A dense model must be read in full for every token, so the disk cannot stand in for memory.");
    confidenceReasons.push("There is no speed to estimate because the plan does not fit.");
    return report;
  }

  const base = o.measuredTps && o.measuredTps > 0 ? o.measuredTps : estimateTps(layout, p, mem);
  const measured = !!(o.measuredTps && o.measuredTps > 0);
  report.speed = measured
    ? { low: { value: base * 0.7, basis: "measured" }, high: { value: base * 1.2, basis: "measured" } }
    : { low: { value: base / 1.5, basis: "estimated" }, high: { value: base * 1.3, basis: "estimated" } };
  const loadBytes = p.strategy === "fully-resident" ? layout.totalBytes : (p.components.find((c) => c.id === "resident")?.bytes ?? 0);
  report.startSeconds = { value: loadBytes / Math.max(1, p.disk.bytesPerSec), basis: "estimated", note: p.disk.measured ? "from the measured disk speed, cold file cache" : "from an assumed disk speed, cold file cache" };
  if (p.strategy !== "fully-resident") {
    const coldPerToken = p.perToken.denseStreamBytes + p.perToken.activeExpertBytes;
    report.diskCeilingTps = { value: coldPerToken > 0 ? p.disk.bytesPerSec / coldPerToken : Infinity, basis: "theoretical", note: "nothing cached, disk always busy" };
  }
  report.suitability = suitability(report.speed.low.value, report.speed.high.value);

  if (mode === "all in memory") why.push("The whole model fits in the memory available, so nothing is read from the disk while it runs.");
  else {
    why.push(`The model is ${(layout.totalBytes / GiB).toFixed(1)} GiB and the plan has ${(p.ceilingBytes / GiB).toFixed(1)} GiB, so inactive experts stay on the SSD and are read when the router picks them.`);
    why.push("The cache starts empty: the first replies are slower than the speed shown, which is for a cache that has warmed up. Start time counts only what must be loaded up front.");
    why.push("Technically runnable is not the same as interactively fast: speed depends on how often the cache already holds the experts a token needs.");
  }

  // Confidence: how much of this rests on a measurement, and how much on assumptions.
  let score = 2;
  if (measured) confidenceReasons.push("Speed comes from a measured run.");
  else {
    if (layout.expertCount > 0 && p.strategy !== "fully-resident") { score -= 1; confidenceReasons.push("The cache hit rate is a fit to a single measurement (Qwen3-30B-A3B on an Apple M2), not a model of your workload."); }
    if (memAssumed) confidenceReasons.push(`Memory speed is assumed (${(mem / 1e9).toFixed(0)} GB/s).`);
  }
  if (p.strategy !== "fully-resident") {
    if (!p.disk.measured) { score -= 1; confidenceReasons.push("Disk speed is assumed; `--measure-disk` times the real file."); }
    else confidenceReasons.push("Disk speed was measured on this machine.");
  }
  if (p.approximate) { score -= 1; confidenceReasons.push("Sizes come from the published architecture, not from a file; a real file differs by a few percent."); }
  if (measured) score = Math.max(score, 2) + 1;
  report.confidence.level = score >= 3 ? "high" : score >= 1 ? "medium" : "low";
  return report;
}
