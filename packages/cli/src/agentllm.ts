import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { OpenAiCompatibleClient, toolCallJsonSchema, type LlmClient } from "@clutor/core";
import { SlotSnapshots, withPrefixSnapshots } from "./backend";
import type { ClutorConfig } from "./config";

/** Where llama-server saves KV-cache snapshots. They contain the project map the model read, so they stay in the user's own home folder. */
export const slotDir = (): string => join(homedir(), ".clutor", "slots");

/** llama-server refuses `--slot-save-path` for a folder that does not exist, so make it before starting. */
export function ensureSlotDir(): string {
  const dir = slotDir();
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}

/**
 * The model client the agent uses. Against a local llama-server it (1) constrains decoding to the tool-call JSON Schema, (2) keeps the
 * conversation in one slot so the server reuses its cached prefix between steps, and (3) saves and restores that prefix between runs.
 * Against any other OpenAI-compatible server only (1) is attempted, and dropped automatically if the server refuses it.
 */
export function agentLlm(config: ClutorConfig, ep: { baseUrl: string; model: string; apiKey?: string; local: boolean }, fetchFn?: typeof globalThis.fetch): LlmClient {
  const json = config.agent.toolMode === "json";
  const client = new OpenAiCompatibleClient({
    baseUrl: ep.baseUrl, model: ep.model, ...(ep.apiKey ? { apiKey: ep.apiKey } : {}), toolMode: config.agent.toolMode,
    ...(json && config.agent.constrainToolCalls ? { jsonSchema: toolCallJsonSchema() } : {}),
    ...(ep.local ? { slot: 0 } : {}),
    ...(fetchFn ? { fetch: fetchFn } : {}),
  });
  if (ep.local && config.agent.prefixSnapshots) return withPrefixSnapshots(client, new SlotSnapshots(ep.baseUrl, 0, fetchFn));
  return client;
}
