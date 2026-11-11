import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * A policy file a company can place on a machine to fix a few settings that a user's own configuration cannot loosen.
 * Looked for at the path in CLUTOR_POLICY, else `clutor.policy.json` in the project folder. Unknown keys are an error, so a typo is not silently ignored.
 *
 *   { "requireSandbox": true,    // commands the agent runs are refused when there is no OS sandbox (see osandbox.ts)
 *     "forbidAutoApprove": true  // the configuration key agent.autoApproveProjectCommands and the --yes flag are ignored: the user is always asked }
 *
 * What it is not: protection against the person at the keyboard. Anyone who can edit files on their own computer can delete the policy file or
 * change CLUTOR_POLICY; a company that needs more than a statement of intent must put the file where users cannot write (device management).
 * The desktop app does not read this file yet.
 */
export interface Policy { requireSandbox: boolean; forbidAutoApprove: boolean; source: string | null }

const KEYS = new Set(["requireSandbox", "forbidAutoApprove"]);

export class PolicyError extends Error {}

export function loadPolicy(projectRoot: string, env: NodeJS.ProcessEnv = process.env): Policy {
  const path = env["CLUTOR_POLICY"] || join(projectRoot, "clutor.policy.json");
  if (!existsSync(path)) {
    if (env["CLUTOR_POLICY"]) throw new PolicyError(`${path}: CLUTOR_POLICY names a file that does not exist`);
    return { requireSandbox: false, forbidAutoApprove: false, source: null };
  }
  let raw: unknown;
  try { raw = JSON.parse(readFileSync(path, "utf8")); } catch (e) { throw new PolicyError(`${path}: ${e instanceof Error ? e.message : "cannot be read"}`); }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) throw new PolicyError(`${path}: must be a JSON object`);
  const o = raw as Record<string, unknown>;
  for (const k of Object.keys(o)) if (!KEYS.has(k)) throw new PolicyError(`${path}: unknown key "${k}" (known: ${[...KEYS].join(", ")})`);
  for (const k of KEYS) if (k in o && typeof o[k] !== "boolean") throw new PolicyError(`${path}: "${k}" must be true or false`);
  return { requireSandbox: o["requireSandbox"] === true, forbidAutoApprove: o["forbidAutoApprove"] === true, source: path };
}
