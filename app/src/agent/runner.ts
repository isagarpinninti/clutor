import { parseHooks, runAgent, type AgentEvent, type AgentResult } from "@core/agent/loop";
import { ShadowWorkspace } from "@core/agent/workspace";
import type { AgentHost, WebApi } from "../bridge";
import { agentWeb } from "../web";
import { hostProject } from "./fs";
import { engineLlm, type AgentSession } from "./llm";
import type { McpManager } from "./mcp";

export interface CodingTask {
  host: AgentHost;
  root: string;
  session: AgentSession;
  modelName: string;
  task: string;
  mcp?: McpManager;
  /** Web search and page reading for the agent; the user is asked once per run. */
  web?: WebApi;
  /** Asked before a command that runs project code or uses the network, and before an MCP tool that may change something. */
  approve: (info: { command: string; network: boolean }) => Promise<boolean>;
  /** Investigate and propose a plan only: the agent is told not to change files, and whatever it changes is discarded. */
  planOnly?: boolean;
  /** What earlier tasks in this session did (see sessionMemory). */
  history?: string;
  /** Told the generation speed of each reply, so the app can remember how fast this model is here. */
  onSpeed?: (tokensPerSecond: number, tokens: number) => void;
  verifyCommand?: string;
  maxSteps?: number;
  /** The model's context window in tokens; the agent keeps about this much of the conversation (a token is about three characters). */
  contextTokens?: number;
  /** Reports how many tokens the model's context held after each step. */
  onContext?: (tokens: number) => void;
  signal?: AbortSignal;
  onEvent: (e: AgentEvent) => void;
}

/** How long one reply of the model may be. A whole file written into a tool call is long: a cap of 1,024 tokens cut such replies off in the middle, so a quarter of the context is allowed (at most 4,096: a model that rambles would otherwise keep going for many minutes). */
export function replyTokens(contextTokens?: number): number { return Math.max(1024, Math.min((contextTokens ?? 0) >= 65536 ? 8192 : 4096, Math.floor((contextTokens ?? 4096) / 4))); }

/** Runs the agent on a project folder. Nothing in the folder changes: the result carries the proposed changes, which the caller applies or discards. */
export async function runCodingTask(t: CodingTask): Promise<{ result: AgentResult; ws: ShadowWorkspace }> {
  const ws = new ShadowWorkspace(hostProject(t.host, t.root));
  const external = t.mcp?.external();
  // Project instructions: a file the user keeps in the folder for the agent (the first of these that exists), read at the start of every task.
  const hooks = parseHooks(await t.host.read(t.root, ".clutor/hooks.json").catch(() => null));
  let instructions = "", instructionsFile = "";
  for (const name of ["CLUTOR.md", "AGENTS.md", "CLAUDE.md"]) {
    const text = await t.host.read(t.root, name).catch(() => null);
    if (text && text.trim()) { instructions = text.trim().slice(0, 6000); instructionsFile = name; break; }
  }
  const result = await runAgent({
    task: t.task, ws, llm: engineLlm(t.session, t.modelName, { maxTokens: replyTokens(t.contextTokens), ...(t.onContext ? { onContext: t.onContext } : {}), ...(t.onSpeed ? { onSpeed: t.onSpeed } : {}) }),
    ...(hooks ? { hooks } : {}), ...(t.planOnly ? { planOnly: true } : {}), maxSteps: t.maxSteps ?? 20, contextChars: Math.max(12_000, Math.min(600_000, Math.round((t.contextTokens ?? 4096) * 2.5))),
    approve: t.approve,
    runCommand: async (plan, command) => {
      const r = await t.host.run(t.root, (await ws.overlayEntries()).map(([path, content]) => ({ path, content })), plan.steps.map((s) => s.argv), 120_000);
      void command;
      return { exitCode: r.exit_code, output: r.output, timedOut: r.timed_out, durationMs: r.duration_ms };
    },
    ...(external?.available() ? { external } : {}),
    ...(t.web ? { web: agentWeb(t.web, () => t.approve({ command: "Search the web and read pages for this task", network: true })) } : {}),
    systemExtra: [t.planOnly ? "PLAN MODE: do not write, edit or delete any file. Read and search the project, then call finish with a clear step-by-step plan: which files to change and how." : "", instructions ? `Project instructions from ${instructionsFile} (follow them):\n${instructions}` : "", t.history ? `Earlier in this session (the files named are already in the folder; do not make them again):\n${t.history}` : "", external?.available() ? external.describe() : ""].filter(Boolean).join("\n\n"),
    ...(t.verifyCommand ? { verifyCommand: t.verifyCommand } : {}),
    ...(t.signal ? { signal: t.signal } : {}),
    onEvent: t.onEvent,
  });
  return { result, ws };
}
