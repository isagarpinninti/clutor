import type { ModelCheck } from "./plan";

/**
 * How long things take on this computer with this model. Every number here is an estimate worked out from the model file and the machine, and says so; the one measured input is the
 * model's own speed from earlier replies (`measuredTps`), which replaces the estimated speed when there is one.
 *
 * The model: one token costs the time to read the weights it uses from memory, plus (when experts are streamed) the time to read the experts that are not already cached from the disk.
 * How many are cached is guessed from the share of the experts that fit: hit rate = share ^ 0.6, a fit to one measurement (Qwen3-30B-A3B on an Apple M2 with 16 GB: a cache holding a
 * quarter of the experts gave 4.7 tokens per second, half gave about 10), so treat it as a rough guide, not a promise.
 */
export const ASSUMED_MEMORY_BYTES_PER_SEC = 40e9;
/** Prompt processing runs several times faster than generation (measured about 5 to 15 times on the 30B model); 5 is the cautious end. */
export const PREFILL_SPEEDUP = 5;
export interface Typical { label: string; promptTokens: number; outputTokens: number }
export const TYPICAL: Record<"chat" | "task", Typical> = {
  chat: { label: "A chat answer (about 150 words)", promptTokens: 100, outputTokens: 200 },
  task: { label: "A small coding task (a web page in three files)", promptTokens: 2500, outputTokens: 3000 },
};

export interface SpeedEstimate { /** Generation speed, tokens per second: a cautious figure and an optimistic one. */ lowTps: number; highTps: number; source: "measured" | "estimated"; /** Plain-language reason when the speed is poor. */ caution?: string }

export function estimateSpeed(check: ModelCheck, o: { memBytesPerSec?: number; measuredTps?: number } = {}): SpeedEstimate | undefined {
  if (check.verdict === "too-big") return undefined;
  if (o.measuredTps && o.measuredTps > 0) return { lowTps: o.measuredTps * 0.7, highTps: o.measuredTps * 1.2, source: "measured" };
  const p = check.plan;
  const mem = o.memBytesPerSec ?? ASSUMED_MEMORY_BYTES_PER_SEC;
  const moe = check.experts > 0;
  const expertAll = p.expertCache.expertBytes * p.expertCache.totalExperts;
  const activeExpert = p.perToken.activeExpertBytes;
  // Bytes read from memory per token: the whole file for a dense model, otherwise the always-used weights plus the experts that are used.
  const memBytes = moe ? Math.max(0, check.fileBytes - expertAll) + activeExpert : check.fileBytes;
  const f = p.strategy === "fully-resident" ? 1 : Math.min(1, Math.max(0, p.expertCache.fractionOfExperts));
  const hit = moe ? Math.min(0.97, Math.pow(f, 0.6)) : 1;
  const diskBytes = p.strategy === "fully-resident" ? 0 : p.perToken.denseStreamBytes + activeExpert * (1 - hit);
  const secs = memBytes / mem + diskBytes / Math.max(1, p.disk.bytesPerSec);
  const tps = secs > 0 ? 1 / secs : 0;
  const out: SpeedEstimate = { lowTps: tps / 1.5, highTps: tps * 1.3, source: "estimated" };
  if (tps < 0.5) out.caution = "Very slow: under half a token per second. It will work, but a reply can take many minutes.";
  else if (tps < 2) out.caution = "Slow: expect minutes for a short answer and an hour or more for a whole program.";
  return out;
}

/** Seconds for `t` at a generation speed (tokens per second). */
export function seconds(t: Typical, tps: number): number { return t.promptTokens / (tps * PREFILL_SPEEDUP) + t.outputTokens / tps; }

/** "45 s", "12 min", "1 h 20 min", "about 3 h". */
export function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec)) return "unknown";
  if (sec < 90) return `${Math.max(1, Math.round(sec / 5) * 5)} s`;
  const min = sec / 60;
  if (min < 90) return `${Math.round(min)} min`;
  const h = min / 60;
  if (h < 10) { const hh = Math.floor(h), mm = Math.round((h - hh) * 60 / 5) * 5; return mm === 60 ? `${hh + 1} h` : mm ? `${hh} h ${mm} min` : `${hh} h`; }
  return `about ${Math.round(h)} h`;
}

/** A range such as "12 to 20 min" (or one figure when both ends are alike). */
export function fmtRange(fast: number, slow: number): string {
  const a = fmtDuration(fast), b = fmtDuration(slow);
  if (a === b) return a;
  const ua = a.split(" ").slice(-1)[0], ub = b.split(" ").slice(-1)[0];
  return ua === ub ? `${a.replace(/ [^ ]+$/, "")} to ${b}` : `${a} to ${b}`;
}

export interface Estimates { tps: SpeedEstimate; load: string; lines: Array<{ label: string; time: string }> }
export function estimates(check: ModelCheck, o: { memBytesPerSec?: number; measuredTps?: number } = {}): Estimates | undefined {
  const tps = estimateSpeed(check, o);
  if (!tps) return undefined;
  const loadSecs = check.plan.strategy === "fully-resident" ? check.fileBytes / Math.max(1, check.plan.disk.bytesPerSec) : Math.max(3, (check.plan.components.find((c) => c.id === "resident")?.bytes ?? 0) / Math.max(1, check.plan.disk.bytesPerSec));
  const line = (t: Typical): { label: string; time: string } => ({ label: t.label, time: fmtRange(seconds(t, tps.highTps), seconds(t, tps.lowTps)) });
  return { tps, load: fmtDuration(loadSecs), lines: [line(TYPICAL.chat), line(TYPICAL.task)] };
}

/** Time left for a task from the plan: what it has taken so far, per finished step, times the steps remaining. Needs at least one finished step and one still to do. */
export function remainingFromPlan(elapsedSec: number, done: number, total: number): number | undefined {
  if (done < 1 || total <= done || elapsedSec <= 0) return undefined;
  return (elapsedSec / done) * (total - done);
}

/**
 * The three questions in plain words: can it be loaded, can it be run, is it fast enough to talk to, and what it is good for at that speed.
 * "Fast" means the cautious end of the speed estimate is at least 5 tokens per second. These come from the estimate, so they carry its uncertainty.
 */
export interface Contract { loadable: boolean; executable: boolean; fast: boolean; use: string }
export function contract(check: ModelCheck, est: Estimates | undefined): Contract {
  if (check.verdict === "too-big" || !est) return { loadable: check.verdict !== "too-big", executable: false, fast: false, use: "Not on this computer: it does not fit even with the disk helping." };
  const low = est.tps.lowTps, high = est.tps.highTps;
  const fast = low >= 5;
  const use = low >= 15 ? "Chat and coding feel immediate."
    : low >= 5 ? "Comfortable for chat and coding."
    : high >= 2 ? "Usable for chat if you are willing to wait; better for tasks you leave running."
    : high >= 0.5 ? "Background jobs only: a reply takes minutes."
    : "Overnight jobs only: a reply takes many minutes.";
  return { loadable: true, executable: true, fast, use };
}
