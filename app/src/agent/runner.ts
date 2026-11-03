import { runAgent, type AgentEvent, type AgentResult } from "@core/agent/loop";
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
  verifyCommand?: string;
  maxSteps?: number;
  signal?: AbortSignal;
  onEvent: (e: AgentEvent) => void;
}

/** Runs the agent on a project folder. Nothing in the folder changes: the result carries the proposed changes, which the caller applies or discards. */
export async function runCodingTask(t: CodingTask): Promise<{ result: AgentResult; ws: ShadowWorkspace }> {
  const ws = new ShadowWorkspace(hostProject(t.host, t.root));
  const external = t.mcp?.external();
  const result = await runAgent({
    task: t.task, ws, llm: engineLlm(t.session, t.modelName),
    maxSteps: t.maxSteps ?? 20, contextChars: 12_000,
    approve: t.approve,
    runCommand: async (plan, command) => {
      const r = await t.host.run(t.root, (await ws.overlayEntries()).map(([path, content]) => ({ path, content })), plan.steps.map((s) => s.argv), 120_000);
      void command;
      return { exitCode: r.exit_code, output: r.output, timedOut: r.timed_out, durationMs: r.duration_ms };
    },
    ...(external?.available() ? { external } : {}),
    ...(t.web ? { web: agentWeb(t.web, () => t.approve({ command: "Search the web and read pages for this task", network: true })) } : {}),
    systemExtra: [t.planOnly ? "PLAN MODE: do not write, edit or delete any file. Read and search the project, then call finish with a clear step-by-step plan: which files to change and how." : "", external?.available() ? external.describe() : ""].filter(Boolean).join("\n\n"),
    ...(t.verifyCommand ? { verifyCommand: t.verifyCommand } : {}),
    ...(t.signal ? { signal: t.signal } : {}),
    onEvent: t.onEvent,
  });
  return { result, ws };
}
