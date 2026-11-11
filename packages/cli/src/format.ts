import { fmtGiB, type GateResult, type Labelled, type MemoryPlan, type PlanReport, type StreamMetrics } from "@clutor/core";

export function table(rows: string[][], header?: string[]): string {
  const all = header ? [header, ...rows] : rows;
  const widths = all[0]?.map((_, i) => Math.max(...all.map((r) => (r[i] ?? "").length))) ?? [];
  const line = (r: string[]): string => r.map((c, i) => c.padEnd(widths[i] ?? 0)).join("  ").trimEnd();
  return [...(header ? [line(header), widths.map((w) => "-".repeat(w)).join("  ")] : []), ...rows.map(line)].join("\n");
}

export const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
export const tps = (x: number): string => (x === Infinity ? "no disk limit" : x >= 10 ? x.toFixed(1) : x >= 0.1 ? x.toFixed(2) : x.toFixed(3));

export function formatPlan(name: string, quant: string | undefined, p: MemoryPlan): string {
  const out: string[] = [];
  out.push(`${name}${quant ? ` (${quant})` : ""}: ${p.fits ? p.strategy.replace(/-/g, " ") : "DOES NOT FIT"} within ${fmtGiB(p.ceilingBytes)}`);
  out.push("", table(p.components.filter((c) => c.bytes > 0).map((c) => [fmtGiB(c.bytes), c.label])));
  if (p.fits && p.strategy !== "fully-resident") {
    out.push("", `Expert cache: ${p.expertCache.slots} of ${p.expertCache.totalExperts} experts (${pct(p.expertCache.fractionOfExperts)}); one token uses ${p.expertCache.slotsPerToken}.`);
    out.push(`Per token, from disk: ${fmtGiB(p.perToken.denseStreamBytes)} dense weights + ${fmtGiB(p.perToken.activeExpertBytes)} of active experts (before the cache helps).`);
    out.push("", `Fastest the disk allows (${(p.disk.bytesPerSec / 2 ** 30).toFixed(1)} GiB/s, ${p.disk.measured ? "measured" : "assumed"}):`);
    out.push(table(p.speed.map((s) => [`cache hit rate ${pct(s.hitRate)}`, `at most ${tps(s.maxTokensPerSec)} tokens/s`])));
  }
  for (const w of p.warnings) out.push("", `! ${w}`);
  return out.join("\n");
}

export function formatGates(g: GateResult[]): string {
  const mark = { pass: "ok  ", warn: "warn", fail: "FAIL", skipped: "--  " } as const;
  return g.map((x) => `${mark[x.status]} ${x.id} ${x.name}: ${x.message}`).join("\n");
}

export function formatSimulation(base: StreamMetrics, c: StreamMetrics): string {
  const row = (label: string, f: (m: StreamMetrics) => string): string[] => [label, f(base), f(c)];
  return table([
    row("cache hit rate", (m) => pct(m.hitRate)),
    row("read from disk per token", (m) => `${(m.bytesPerToken / 2 ** 30).toFixed(2)} GiB`),
    row("read requests per token", (m) => (m.requests / Math.max(1, m.tokens)).toFixed(0)),
    row("time waiting for disk", (m) => pct(m.stallSec / Math.max(1e-9, m.totalSec))),
    row("modelled speed", (m) => `${tps(m.tokensPerSec)} tokens/s`),
    row("prefetches used / issued", (m) => `${m.prefetchUsed} / ${m.prefetchIssued}`),
    row("experts replaced by stand-ins", (m) => pct(1 - m.exactShare)),
    row("router probability served exactly", (m) => pct(m.exactMassShare)),
  ], ["", "baseline (LRU)", "clutor"]);
}

const TAG: Record<string, string> = { measured: "MEASURED", estimated: "ESTIMATED", theoretical: "THEORETICAL", unknown: "UNKNOWN" };
const tagOf = (l: Labelled<number>): string => `[${TAG[l.basis]}]`;
const MARK: Record<string, string> = { good: "✓ good", usable: "✓ usable", slow: "⚠ slow", poor: "✗ poor" };
const TASK: Record<string, string> = { chat: "Interactive chat", coding: "Coding agent", background: "Background jobs", "long-generation": "Long generation" };
const secs = (s: number): string => (s < 90 ? `${Math.max(1, Math.round(s))} s` : `${Math.round(s / 60)} min`);

/** The plan as a decision: verdict, speed with its basis, what each kind of use will feel like, and how sure the numbers are. */
export function formatReport(name: string, quant: string | undefined, r: PlanReport, machine: { ram: string; disk: string; cpu: string }): string {
  const head = (t: string): string => `${t}\n${"-".repeat(32)}`;
  const o: string[] = [];
  o.push("CLUTOR PERFORMANCE PLAN", "", "MODEL", `${name}${quant ? ` ${quant}` : ""}`, `${fmtGiB(r.memory.modelBytes)}`, "");
  o.push("YOUR MACHINE", `RAM      ${machine.ram}`, `SSD      ${machine.disk}`, `CPU      ${machine.cpu}`, "");
  o.push(head("VERDICT"), r.verdict.canRun ? "✓ CAN RUN" : "✗ DOES NOT FIT", ...(r.verdict.mode === "disk streaming" ? ["⚠ SSD STREAMING REQUIRED"] : []), r.verdict.headline, "");
  if (r.speed) {
    o.push(head("EXPECTED PERFORMANCE"));
    o.push(`Speed        ${tps(r.speed.low.value)} to ${tps(r.speed.high.value)} tokens/s   [${TAG[r.speed.low.basis]}]`);
    if (r.startSeconds) o.push(`Start        about ${secs(r.startSeconds.value)} before the first token   [${TAG[r.startSeconds.basis]}]`);
    if (r.diskCeilingTps) o.push(`Disk limit   at most ${tps(r.diskCeilingTps.value)} tokens/s with nothing cached   [${TAG[r.diskCeilingTps.basis]}]`);
    o.push("");
    o.push(head("CACHE"), `Recommended  ${fmtGiB(r.cache.recommendedBytes.value)}   ${tagOf(r.cache.recommendedBytes)}`, `Minimum      ${fmtGiB(r.cache.minimumBytes.value)}   ${tagOf(r.cache.minimumBytes)}`, "");
    o.push(head("SUITABILITY"), table(r.suitability.map((t) => [TASK[t.task] ?? t.task, MARK[t.rating] ?? t.rating, t.why])), "");
  }
  o.push(head("CONFIDENCE"), r.confidence.level.toUpperCase(), ...r.confidence.reasons.map((x) => `- ${x}`), "");
  o.push(head("WHY"), ...r.why);
  return o.join("\n");
}
