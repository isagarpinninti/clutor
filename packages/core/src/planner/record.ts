import type { Basis } from "./report";

/**
 * One benchmark result in a stable, machine-readable form, so results from different computers and versions can be compared and committed
 * (benchmarks/<machine>/<model>.json). Every measurement states how it was obtained; null means not measured, never zero.
 * Bump `schema` when a field changes meaning.
 */
export interface BenchRecord {
  schema: 1;
  clutorVersion: string;
  createdAt: string;
  machine: { os: string; osRelease: string; arch: string; cpu: string; cores: number; ramBytes: number; cpuFeatures: string[]; accelerators: Array<{ kind: string; name: string; api: string; usedByClutor: boolean }> };
  model: { name: string; arch: string; quantization: string | null; fileBytes: number; experts: number; expertsUsed: number; synthetic: boolean };
  runtime: { engine: string | null; contextTokens: number | null; cacheBytes: number | null; cachePolicy: string | null; prompt: string | null; tokens: number | null };
  results: {
    diskReadBytesPerSec: Measure | null;
    loadSeconds: Measure | null;
    firstTokenSeconds: Measure | null;
    generationTokensPerSec: Measure | null;
    promptTokensPerSec: Measure | null;
    cacheHitRate: Measure | null;
    peakRssBytes: Measure | null;
  };
  notes: string[];
}
export interface Measure { value: number; basis: Basis; runs: number }

export const emptyResults = (): BenchRecord["results"] => ({ diskReadBytesPerSec: null, loadSeconds: null, firstTokenSeconds: null, generationTokensPerSec: null, promptTokensPerSec: null, cacheHitRate: null, peakRssBytes: null });

/** Rejects a record that cannot be compared: wrong schema, missing machine or model, a result that is not a finite non-negative number. */
export function validateBenchRecord(x: unknown): string[] {
  const errs: string[] = [];
  const r = x as Partial<BenchRecord> | null;
  if (!r || typeof r !== "object") return ["not an object"];
  if (r.schema !== 1) errs.push(`schema must be 1, got ${String(r.schema)}`);
  for (const k of ["machine", "model", "runtime", "results"] as const) if (!r[k] || typeof r[k] !== "object") errs.push(`missing ${k}`);
  if (typeof r.createdAt !== "string" || Number.isNaN(Date.parse(r.createdAt))) errs.push("createdAt must be an ISO date");
  for (const [k, v] of Object.entries(r.results ?? {})) {
    if (v === null) continue;
    const m = v as Measure;
    if (typeof m.value !== "number" || !Number.isFinite(m.value) || m.value < 0) errs.push(`results.${k}.value must be a finite number >= 0`);
    if (!["measured", "estimated", "theoretical"].includes(m.basis)) errs.push(`results.${k}.basis must be measured, estimated or theoretical`);
    if (!Number.isInteger(m.runs) || m.runs < 1) errs.push(`results.${k}.runs must be a whole number >= 1`);
  }
  return errs;
}
