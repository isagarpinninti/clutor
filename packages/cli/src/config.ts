import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { z } from "zod";

export const CONFIG_FILE = "clutor.config.json";
export const DEFAULT_PORT = 4320;

export const ConfigSchema = z
  .object({
    port: z.number().int().min(0).max(65535).default(DEFAULT_PORT),
    host: z.string().min(1).default("127.0.0.1"),
    /** Folder scanned for .gguf files. */
    models: z.string().min(1).default("models"),
    /** What answers chat requests. `none` still serves planning, inspection, and the agent's read-only tools. */
    backend: z
      .object({
        kind: z.enum(["none", "openai", "llama-server"]).default("none"),
        /** `openai`: the base URL of any OpenAI-compatible server, ending in /v1. */
        baseUrl: z.string().url().optional(),
        model: z.string().optional(),
        /** Name of the environment variable holding an API key (never put the key in this file). */
        apiKeyEnv: z.string().optional(),
        llamaServer: z
          .object({
            /** Path to the llama-server binary; found on PATH if omitted. */
            binary: z.string().optional(),
            /** Path to the .gguf file to load. */
            model: z.string().optional(),
            port: z.number().int().min(1).max(65535).default(8081),
            threads: z.number().int().min(1).max(512).optional(),
            extraArgs: z.array(z.string()).default([]),
          })
          .strict()
          .default({ port: 8081, extraArgs: [] }),
      })
      .strict()
      .default({ kind: "none", llamaServer: { port: 8081, extraArgs: [] } }),
    memory: z
      .object({
        ramCeilingGiB: z.number().min(1).max(4096).optional(),
        osHeadroomGiB: z.number().min(0).max(512).optional(),
        contextTokens: z.number().int().min(256).max(2_000_000).default(8192),
        kvBits: z.union([z.literal(16), z.literal(8), z.literal(4)]).default(16),
        /** Disk read speed in GB/s if you know it; `clutor bench` measures it. */
        diskGBps: z.number().min(0.01).max(100).optional(),
      })
      .strict()
      .default({ contextTokens: 8192, kvBits: 16 }),
    cache: z
      .object({
        policy: z.enum(["lru", "wtinylfu", "adaptive"]).default("adaptive"),
        smartRoute: z.object({ enabled: z.boolean().default(false), maxSubstitutions: z.number().int().min(0).max(8).default(2), candidates: z.number().int().min(1).max(64).default(12), maxDeltaP: z.number().min(0).max(1).default(0.015) }).strict().default({ enabled: false, maxSubstitutions: 2, candidates: 12, maxDeltaP: 0.015 }),
      })
      .strict()
      .default({ policy: "adaptive", smartRoute: { enabled: false, maxSubstitutions: 2, candidates: 12, maxDeltaP: 0.015 } }),
    agent: z
      .object({
        toolMode: z.enum(["json", "native"]).default("json"),
        maxSteps: z.number().int().min(1).max(200).default(30),
        verifyCommand: z.string().max(500).optional(),
        /** Run commands that execute project code without asking. Off by default; see docs/SECURITY.md. */
        autoApproveProjectCommands: z.boolean().default(false),
        commandTimeoutSeconds: z.number().int().min(5).max(3600).default(120),
        /** Ask the model server to constrain each reply to the tool-call JSON Schema (llama.cpp turns it into a grammar). Replies are still validated. */
        constrainToolCalls: z.boolean().default(true),
        /** With llama-server, save the KV cache of the project-map prefix under .clutor/slots and restore it on the next run. */
        prefixSnapshots: z.boolean().default(true),
      })
      .strict()
      .default({ toolMode: "json", maxSteps: 30, autoApproveProjectCommands: false, commandTimeoutSeconds: 120, constrainToolCalls: true, prefixSnapshots: true }),
    /** Web origins allowed to call the server from a browser. The desktop app is always allowed. */
    corsOrigins: z.array(z.string().url()).default([]),
  })
  .strict();

export type ClutorConfig = z.infer<typeof ConfigSchema>;

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

export function loadConfig(dir: string, file?: string): { config: ClutorConfig; path: string | null; root: string } {
  const root = resolve(dir);
  const path = file ? resolve(root, file) : join(root, CONFIG_FILE);
  if (!existsSync(path)) {
    if (file) throw new ConfigError(`${path}: no such file`);
    return { config: ConfigSchema.parse({}), path: null, root };
  }
  let raw: unknown;
  try { raw = JSON.parse(readFileSync(path, "utf8")); } catch (e) { throw new ConfigError(`${path}: ${e instanceof Error ? e.message : "cannot be read"}`); }
  const parsed = ConfigSchema.safeParse(raw);
  if (!parsed.success) throw new ConfigError(parsed.error.issues.map((i) => `${path}: ${i.path.join(".") || "(root)"}: ${i.message}`).join("\n"));
  return { config: parsed.data, path, root };
}
