import { GiB, fmtGiB, type MemoryPlan } from "./planner/plan";
import { validateBounds, type GgufFile } from "./gguf/parse";

/**
 * The eight checks Clutor runs before and after a task. Each looks at something concrete and says pass, warn, fail or skipped
 * (when there is nothing to look at yet). They report; they do not decide for you.
 */

export type GateStatus = "pass" | "warn" | "fail" | "skipped";
export interface GateResult {
  id: `CL-${number}`;
  name: string;
  status: GateStatus;
  message: string;
}

export interface GateInput {
  hostRamBytes: number;
  plan?: MemoryPlan;
  cache?: { policy: "lru" | "wtinylfu" | "adaptive" };
  smartRoute?: { enabled: boolean; maxDeltaP: number };
  agent?: { toolCalls: number; invalidReplies: number; violations: number; commandApproval: "always" | "project-code" | "never-asked" };
  gguf?: GgufFile;
  receiptsOk?: { ok: boolean; detail?: string };
}

const r = (id: number, name: string, status: GateStatus, message: string): GateResult => ({ id: `CL-${id}` as const, name, status, message });

export function evaluateGates(i: GateInput): GateResult[] {
  const out: GateResult[] = [];

  // 101: the memory ceiling.
  if (!i.plan) out.push(r(101, "Physical RAM ceiling", "skipped", "No model is loaded, so there is no memory plan yet."));
  else if (!i.plan.fits) out.push(r(101, "Physical RAM ceiling", "fail", `This model cannot run within ${fmtGiB(i.plan.ceilingBytes)}: it is short by ${fmtGiB(i.plan.shortfallBytes)}.`));
  else if (i.plan.usedBytes > i.plan.ceilingBytes) out.push(r(101, "Physical RAM ceiling", "fail", `The plan uses ${fmtGiB(i.plan.usedBytes)} but the ceiling is ${fmtGiB(i.plan.ceilingBytes)}.`));
  else if (i.plan.ceilingBytes > i.hostRamBytes - 1 * GiB) out.push(r(101, "Physical RAM ceiling", "warn", `The ceiling (${fmtGiB(i.plan.ceilingBytes)}) leaves under 1 GiB of ${fmtGiB(i.hostRamBytes)} to the operating system; it may swap.`));
  else out.push(r(101, "Physical RAM ceiling", "pass", `Plan uses ${fmtGiB(i.plan.usedBytes)} of a ${fmtGiB(i.plan.ceilingBytes)} ceiling on a ${fmtGiB(i.hostRamBytes)} machine.`));

  // 102: does the streaming window hold a token's worth of work.
  if (!i.plan) out.push(r(102, "Streaming window", "skipped", "No model is loaded."));
  else if (!i.plan.fits) out.push(r(102, "Streaming window", "fail", "There is no room for even the always-resident weights."));
  else if (i.plan.strategy === "fully-resident") out.push(r(102, "Streaming window", "pass", "The whole model is in memory; nothing is streamed."));
  else if (i.plan.expertCache.totalExperts > 0 && i.plan.expertCache.slots < i.plan.expertCache.slotsPerToken) out.push(r(102, "Streaming window", "warn", `The expert cache holds ${i.plan.expertCache.slots} experts but one token uses ${i.plan.expertCache.slotsPerToken}; most reads will be repeated.`));
  else out.push(r(102, "Streaming window", "pass", `${i.plan.strategy.replace(/-/g, " ")}; the cache holds ${i.plan.expertCache.slots} experts (${(i.plan.expertCache.fractionOfExperts * 100).toFixed(1)}% of them).`));

  // 103: substitution may change the output; keep it small.
  if (!i.smartRoute || !i.smartRoute.enabled) out.push(r(103, "Router drift bound", "pass", "Expert substitution is off: every expert the router chose is the one used."));
  else if (i.smartRoute.maxDeltaP > 0.05) out.push(r(103, "Router drift bound", "fail", `Substituting experts that scored up to ${i.smartRoute.maxDeltaP} lower can visibly change the model's output; the limit here is 0.05.`));
  else if (i.smartRoute.maxDeltaP > 0.015) out.push(r(103, "Router drift bound", "warn", `A substitution gap of ${i.smartRoute.maxDeltaP} is above the recommended 0.015.`));
  else out.push(r(103, "Router drift bound", "pass", `Substitution only when the router scored the stand-in within ${i.smartRoute.maxDeltaP}.`));

  // 104: scan resistance.
  if (!i.cache) out.push(r(104, "Cache scan resistance", "skipped", "No cache policy is set."));
  else if (i.cache.policy === "lru") out.push(r(104, "Cache scan resistance", "warn", "Plain LRU can be flushed by a long run of rarely used experts (a topic change); the adaptive policy resists that."));
  else out.push(r(104, "Cache scan resistance", "pass", `${i.cache.policy === "adaptive" ? "Adaptive W-TinyLFU" : "W-TinyLFU"} admits a new expert only if it has been used more than the one it would evict.`));

  // 105: did the model produce well-formed tool calls.
  if (!i.agent || i.agent.toolCalls + i.agent.invalidReplies === 0) out.push(r(105, "Tool-call validity", "skipped", "The agent has not run yet."));
  else {
    const bad = i.agent.invalidReplies / (i.agent.toolCalls + i.agent.invalidReplies);
    const pct = `${i.agent.invalidReplies} of ${i.agent.toolCalls + i.agent.invalidReplies} replies were not valid tool calls`;
    out.push(r(105, "Tool-call validity", bad <= 0.2 ? "pass" : bad <= 0.5 ? "warn" : "fail", bad <= 0.2 ? `${pct}.` : `${pct}: this model is not following the tool format well; try another model or JSON mode.`));
  }

  // 106: the path jail.
  if (!i.agent) out.push(r(106, "Project folder jail", "skipped", "The agent has not run yet."));
  else out.push(i.agent.violations === 0 ? r(106, "Project folder jail", "pass", "The agent stayed inside the project and its limits.") : r(106, "Project folder jail", "warn", `${i.agent.violations} attempt${i.agent.violations === 1 ? " was" : "s were"} blocked (secrets, paths outside the project, protected files, disallowed commands or size limits).`));

  // 107: command approval.
  if (!i.agent) out.push(r(107, "Command safety", "skipped", "The agent has not run yet."));
  else if (i.agent.commandApproval === "never-asked") out.push(r(107, "Command safety", "warn", "Commands that run project code were approved automatically. A planted instruction in a file could make the agent run something harmful."));
  else out.push(r(107, "Command safety", "pass", i.agent.commandApproval === "always" ? "Every command needed your approval." : "Commands that run project code needed your approval; read-only commands ran without asking. No shell is used."));

  // 108: file and log integrity.
  const problems: string[] = [];
  if (i.gguf) { try { validateBounds(i.gguf); } catch (e) { problems.push(e instanceof Error ? e.message : String(e)); } }
  if (i.receiptsOk && !i.receiptsOk.ok) problems.push(`the receipt log is broken${i.receiptsOk.detail ? `: ${i.receiptsOk.detail}` : ""}`);
  if (!i.gguf && !i.receiptsOk) out.push(r(108, "File and log integrity", "skipped", "No model file or receipt log to check."));
  else out.push(problems.length ? r(108, "File and log integrity", "fail", problems.join("; ")) : r(108, "File and log integrity", "pass", [i.gguf ? "Every tensor lies inside the model file." : "", i.receiptsOk ? "The receipt chain is intact." : ""].filter(Boolean).join(" ")));

  return out;
}

export const gateSummary = (g: GateResult[]): { pass: number; warn: number; fail: number; skipped: number } =>
  g.reduce((a, x) => ({ ...a, [x.status]: a[x.status] + 1 }), { pass: 0, warn: 0, fail: 0, skipped: 0 });
