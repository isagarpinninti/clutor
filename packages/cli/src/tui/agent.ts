import { ShadowWorkspace, runAgent, type AgentEvent, type FileDiff, type LlmClient } from "@clutor/core";
import { NodeProject, appendReceipt, applyChanges, lastReceiptTag, loadOrCreateReceiptSigner, scratchRunner, ConflictError } from "@clutor/core/node";
import { agentLlm } from "../agentllm";
import type { ClutorConfig } from "../config";
import type { SessionState } from "../cli";
import { fit, stripAnsi, wrap, type Region, type Style } from "./ansi";
import { MarkdownStream } from "./markdown";
import { formatElapsed, spinnerLine } from "./spinner";

/**
 * The coding agent inside the chat screen: every tool call is shown as it happens (`● Read(file)` and a result line under it), the
 * proposed changes are shown as a coloured diff, and nothing is written until you answer a question with the arrow keys.
 */

export interface AgentHost {
  st: Style;
  cols(): number;
  println(s?: string): void;
  region(): Region;
  /** A list to choose from; resolves to the index, or undefined for escape. */
  choose(title: string, items: Array<{ label: string; hint?: string }>): Promise<number | undefined>;
  /** Calls `onKey` with raw input until the returned function is called (raw mode stays on). */
  listen(onKey: (s: string) => void): () => void;
}

export interface AgentRun {
  task: string;
  dir: string;
  config: ClutorConfig;
  endpoint: { baseUrl: string; model: string; apiKey?: string; local: boolean };
  /** A model client to use instead of the HTTP endpoint: Clutor's own engine. */
  llm?: LlmClient;
  session: SessionState;
  /** Write the changes without asking (still only if the check passed, when there is one). */
  autoApply: boolean;
  contextTokens: number;
}

const clip = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const plural = (n: number, w: string): string => `${n} ${w}${n === 1 ? "" : "s"}`;
const one = (s: string): string => s.split("\n").find((l) => l.trim())?.trim() ?? "";

/** `Read(src/a.ts)`: how a tool call is titled. */
export function callTitle(name: string, a: Record<string, unknown>): string | undefined {
  const s = (k: string): string => String(a[k] ?? "");
  switch (name) {
    case "list_files": return `List(${s("dir") || "."})`;
    case "read_file": return `Read(${s("path")}${a["start_line"] ? `:${s("start_line")}-${s("end_line")}` : ""})`;
    case "search": return `Search(${JSON.stringify(s("pattern"))}${a["dir"] ? ` in ${s("dir")}` : ""})`;
    case "write_file": return `Write(${s("path")})`;
    case "edit_file": return `Update(${s("path")})`;
    case "delete_file": return `Delete(${s("path")})`;
    case "repo_map": return "Map(project)";
    case "run_command": return `Run(${s("command")})`;
    default: return undefined;
  }
}

/** The line under a tool call: what came back, in a few words. */
export function resultSummary(name: string, ok: boolean, output: string): string {
  const lines = output.split("\n").filter((l) => l.length > 0);
  if (!ok) return one(output) || "failed";
  switch (name) {
    case "read_file": {
      const numbered = output.split("\n").filter((l) => /^\s*\d+\|/.test(l));
      // A file that ends in a newline shows one empty last row; it is not a line.
      const n = numbered.length > 0 && /^\s*\d+\| ?$/.test(numbered[numbered.length - 1] as string) ? numbered.length - 1 : numbered.length;
      return `Read ${plural(n, "line")}`;
    }
    case "list_files": return output.startsWith("(no files)") ? "no files" : `${lines.filter((l) => !l.startsWith("...")).length} files`;
    case "search": return lines.length === 0 || output.startsWith("no matches") ? "no matches" : `${lines.length} matches`;
    case "write_file": case "edit_file": case "delete_file": return one(output) || "done";
    case "run_command": return one(output) || "ran";
    default: return one(output) || "ok";
  }
}

/** Colours a unified diff with line numbers; long ones are cut. */
export function renderPatch(st: Style, patch: string, max = 36): string[] {
  const out: string[] = [];
  let oldN = 0, newN = 0, shown = 0, hidden = 0;
  for (const l of patch.split("\n")) {
    const h = /^@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(l);
    if (h) { oldN = Number(h[1]); newN = Number(h[2]); if (out.length) out.push(st.dim("    ⋮")); continue; }
    if (l.startsWith("---") || l.startsWith("+++") || l === "") continue;
    const sign = l[0], text = l.slice(1);
    if (shown >= max) { hidden++; continue; }
    shown++;
    if (sign === "+") { out.push(`${st.dim(String(newN).padStart(5))} ${st.ok(`+ ${text}`)}`); newN++; }
    else if (sign === "-") { out.push(`${st.dim(String(oldN).padStart(5))} ${st.bad(`- ${text}`)}`); oldN++; }
    else { out.push(`${st.dim(String(newN).padStart(5))}   ${st.dim(text)}`); oldN++; newN++; }
  }
  if (hidden) out.push(st.dim(`        … ${hidden} more line${hidden === 1 ? "" : "s"}`));
  return out;
}

/** Adds `enable_thinking: false` to every request, so a thinking model answers each step quickly. */
export function withoutThinking(f: typeof globalThis.fetch = globalThis.fetch): typeof globalThis.fetch {
  return (async (url: Parameters<typeof fetch>[0], init?: RequestInit) => {
    if (typeof init?.body === "string") {
      try { init = { ...init, body: JSON.stringify({ ...(JSON.parse(init.body) as object), chat_template_kwargs: { enable_thinking: false } }) }; } catch { /* not JSON: send as it was */ }
    }
    return f(url, init);
  }) as typeof fetch;
}

export async function runAgentTask(h: AgentHost, o: AgentRun, state: { allowCommands: boolean }): Promise<void> {
  const st = h.st;
  const mark = (glyph: string, paint: (s: string) => string, text: string, hang = 4): void => {
    for (const [i, r] of wrap(paint(text), h.cols() - 2, hang).entries()) h.println(i === 0 ? `${glyph} ${r}` : `  ${r}`);
  };
  const sub = (text: string, paint: (s: string) => string = st.dim): void => { for (const [i, r] of wrap(paint(text), h.cols() - 6, 0).entries()) h.println(i === 0 ? `  ${paint("·")}  ${r}` : `     ${r}`); };

  const project = new NodeProject(o.dir);
  const ws = new ShadowWorkspace(project);
  const key = await loadOrCreateReceiptSigner(o.dir);
  const cfg = { ...o.config, agent: { ...o.config.agent, prefixSnapshots: false } };
  const llm = o.llm ?? agentLlm(cfg, o.endpoint, withoutThinking());
  const verify = o.session.verify ?? cfg.agent.verifyCommand;

  const region = h.region();
  const ac = new AbortController();
  const t0 = Date.now();
  let step = 0, label = "Working", paused = false;
  const paint = (): void => { if (!paused) region.draw([fit(spinnerLine(st, Date.now() - t0, label, [formatElapsed(Date.now() - t0), step ? `step ${step}` : "starting", "esc to stop"]), h.cols() - 1)]); };
  const emit = (rows: () => void): void => { region.clear(); rows(); paint(); };
  let timer = setInterval(paint, 80);
  let stopKeys = h.listen((s) => { if (s === "\x1b" || s === "\x03") ac.abort(); });
  paint();

  const pause = async <T>(f: () => Promise<T>): Promise<T> => {
    paused = true; clearInterval(timer); stopKeys(); region.clear();
    try { return await f(); } finally { paused = false; stopKeys = h.listen((s) => { if (s === "\x1b" || s === "\x03") ac.abort(); }); timer = setInterval(paint, 80); }
  };

  const onEvent = (e: AgentEvent): void => {
    if (e.type === "thought") { step = e.step; label = "Thinking"; const t = one(e.text); if (t) emit(() => mark(st.dim("∴"), (s) => st.dim(st.italic(s)), clip(t, 240), 2)); }
    else if (e.type === "tool_call") {
      step = e.step; label = "Working";
      const title = callTitle(e.name, e.args);
      if (!title) return;
      emit(() => {
        h.println(`${st.accent("●")} ${st.bold(title.replace(/\(.*$/s, ""))}${st.dim(title.slice(title.indexOf("(")))}`);
        if (e.name === "edit_file") {
          const oldL = String(e.args["old_text"] ?? "").split("\n"), newL = String(e.args["new_text"] ?? "").split("\n");
          for (const l of oldL.slice(0, 6)) h.println(`      ${st.bad(`- ${clip(l, h.cols() - 12)}`)}`);
          for (const l of newL.slice(0, 6)) h.println(`      ${st.ok(`+ ${clip(l, h.cols() - 12)}`)}`);
          if (oldL.length > 6 || newL.length > 6) h.println(`      ${st.dim("⋮")}`);
        }
      });
    } else if (e.type === "tool_result") {
      if (e.name === "finish") return;
      const text = resultSummary(e.name, e.ok, e.output);
      emit(() => sub(clip(text, h.cols() - 8), e.ok ? st.dim : st.bad));
    } else if (e.type === "invalid") emit(() => sub(`the model's reply was not a valid action (${clip(e.error, 90)}); asking again`, st.warn));
    else if (e.type === "verify") emit(() => { h.println(`${st.accent("●")} ${st.bold("Verify")}${st.dim(`(${e.command})`)}`); sub(!e.ran ? "not run" : e.ok ? "passed" : `failed${e.attempt > 0 ? ` (after ${plural(e.attempt, "retry")})` : ""}`, !e.ran ? st.warn : e.ok ? st.ok : st.bad); });
  };

  let result;
  try {
    result = await runAgent({
      task: o.task, llm, ws, maxSteps: o.config.agent.maxSteps, contextChars: Math.floor(o.contextTokens * 2.5), signal: ac.signal,
      ...(verify ? { verifyCommand: verify } : {}),
      ...(o.session.notes.length ? { systemExtra: `Earlier in this session (for context only):\n${o.session.notes.slice(-6).map((n) => `- ${n}`).join("\n")}` } : {}),
      onEvent,
      runCommand: scratchRunner(project, ws, { timeoutMs: cfg.agent.commandTimeoutSeconds * 1000 }),
      approve: async ({ command, network }) => {
        if (state.allowCommands || o.session.auto || cfg.agent.autoApproveProjectCommands) return true;
        if (verify !== undefined && command === o.session.verify) return true;
        return pause(async () => {
          const i = await h.choose(`Run \`${command}\`${network ? " (uses the network)" : ""} in a scratch copy of the project?`, [{ label: "Yes" }, { label: "Yes, and don't ask again this session" }, { label: "No", hint: "(the agent continues without it)" }]);
          if (i === 1) state.allowCommands = true;
          return i === 0 || i === 1;
        });
      },
      receipt: { signer: key, previous: await lastReceiptTag(o.dir) },
    });
  } finally { clearInterval(timer); stopKeys(); region.clear(); }

  const tokens = result.usage.completionTokens;
  if (result.receipt) await appendReceipt(o.dir, result.receipt);
  o.session.usage.promptTokens += result.usage.promptTokens;
  o.session.usage.completionTokens += result.usage.completionTokens;
  o.session.notes.push(`"${o.task.slice(0, 120)}" -> ${result.status}: ${result.summary.slice(0, 200)}`);

  h.println();
  if (result.status === "aborted") h.println(`${st.warn("●")} Stopped.`);
  else {
    const md = new MarkdownStream(st, () => h.cols() - 2);
    const rows = [...md.push(`${result.summary}\n`).done, ...md.end().done];
    rows.forEach((r, i) => h.println(i === 0 ? `${result.status === "finished" ? st.accent("●") : st.warn("●")} ${r}` : `  ${r}`));
    if (result.status !== "finished") sub(`ended with status "${result.status}"${result.error ? `: ${result.error}` : ""}`, st.warn);
  }

  const diffs: FileDiff[] = result.diffs;
  for (const d of diffs) {
    h.println();
    h.println(`${st.accent("●")} ${st.bold(d.status === "added" ? "Create" : d.status === "deleted" ? "Delete" : "Update")}${st.dim(`(${d.path})`)}`);
    sub(`${d.additions ? st.ok(`+${plural(d.additions, "line")}`) : ""}${d.additions && d.deletions ? st.dim(", ") : ""}${d.deletions ? st.bad(`-${plural(d.deletions, "line")}`) : ""}`);
    for (const r of renderPatch(st, d.patch)) h.println(fit(`    ${r}`, h.cols() - 1));
  }
  h.println();
  if (result.verification) sub(`check \`${result.verification.command}\`: ${!result.verification.ran ? "not run" : result.verification.ok ? "passed" : "FAILED"}${result.verification.retries ? ` after ${plural(result.verification.retries, "retry")}` : ""}`, result.verification.ok ? st.ok : st.warn);
  sub(`${plural(diffs.length, "file")} changed · ${plural(result.steps, "step")} · ${tokens} tokens · ${formatElapsed(Date.now() - t0)}${result.violations ? ` · ${plural(result.violations, "blocked attempt")}` : ""}`);
  if (diffs.length === 0) return;

  // With a check requested, nothing is written unless it passed on the final state.
  if (verify !== undefined && result.verification && !result.verification.ok) { sub(`nothing was written: the check \`${verify}\` ${result.verification.ran ? "did not pass" : "was not run"}. The diff above is what the agent proposed.`, st.warn); return; }
  let apply = o.autoApply;
  if (!apply) {
    const i = await h.choose(`Apply ${plural(diffs.length, "change")} to ${diffs.length === 1 ? diffs[0]?.path : "the folder"}?`, [{ label: "Yes, write the changes" }, { label: "No, discard them", hint: "(nothing is written)" }]);
    apply = i === 0;
  }
  if (!apply) { sub("nothing was written"); return; }
  try {
    const r = await applyChanges(project, await ws.changeSet());
    o.session.lastCheckpoint = { id: r.id, dir: o.dir };
    sub(`applied ${plural(r.applied.length, "file")}. /undo restores them`, st.ok);
  } catch (e) {
    if (e instanceof ConflictError) { sub(`${stripAnsi(e.message)} Nothing was written.`, st.bad); return; }
    throw e;
  }
}
