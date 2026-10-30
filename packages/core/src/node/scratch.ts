import { spawn } from "node:child_process";
import { copyFile, mkdir, mkdtemp, rm, symlink, writeFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { CommandPlan, CommandStep } from "../agent/commands";
import type { CommandResult } from "../agent/tools";
import type { ShadowWorkspace } from "../agent/workspace";
import { type NodeProject } from "./project";

/**
 * Runs an approved command in a scratch copy of the project that already contains the agent's changes, so tests can be run against
 * work that has not been applied. The copy is deleted afterwards. The environment is stripped to what a build needs (no tokens, no
 * credentials, a throwaway HOME), and a runaway process is killed at a timeout.
 *
 * This is isolation of the project and of the environment, not of the machine: a program that is allowed to run can still read any
 * file the user can. That is why such commands need approval.
 */

export interface ScratchOptions {
  timeoutMs?: number;
  maxOutputBytes?: number;
  maxCopyBytes?: number;
  /** Environment variable names to pass through from the real environment (default none beyond PATH). */
  passEnv?: string[];
  env?: NodeJS.ProcessEnv;
}

async function materialise(project: NodeProject, ws: ShadowWorkspace, dir: string, maxBytes: number): Promise<void> {
  let total = 0;
  for (const rel of await project.list()) {
    const src = await project.safePath(rel);
    total += (await stat(src)).size;
    if (total > maxBytes) throw new Error(`the project is larger than ${Math.round(maxBytes / 1e6)} MB, too big to copy for a test run`);
    await mkdir(dirname(join(dir, rel)), { recursive: true });
    await copyFile(src, join(dir, rel));
  }
  for (const [rel, text] of await ws.overlayEntries()) {
    const dest = join(dir, rel);
    if (text === null) await rm(dest, { force: true });
    else { await mkdir(dirname(dest), { recursive: true }); await writeFile(dest, text, "utf8"); }
  }
  // Dependencies are linked, not copied: they are large, and the agent cannot change them (node_modules is protected).
  try { await stat(join(project.root, "node_modules")); await symlink(join(project.root, "node_modules"), join(dir, "node_modules"), "dir"); } catch { /* none */ }
}

function cleanEnv(scratch: string, o: ScratchOptions): NodeJS.ProcessEnv {
  const src = o.env ?? process.env;
  const env: NodeJS.ProcessEnv = { PATH: src["PATH"] ?? "", HOME: join(scratch, ".home"), TMPDIR: scratch, CI: "1", NO_COLOR: "1", FORCE_COLOR: "0", LANG: "C.UTF-8", NODE_ENV: "test" };
  if (process.platform === "win32") { for (const k of ["SystemRoot", "ComSpec", "PATHEXT", "TEMP", "TMP"]) if (src[k]) env[k] = src[k]; }
  for (const k of o.passEnv ?? []) if (src[k] !== undefined) env[k] = src[k];
  return env;
}

function runStep(step: CommandStep, cwd: string, env: NodeJS.ProcessEnv, timeoutMs: number, maxOut: number, left: { ms: number }): Promise<{ code: number | null; out: string; timedOut: boolean }> {
  return new Promise((resolve) => {
    const win = process.platform === "win32";
    // On Windows, programs like npm are .cmd files and need a shell; the arguments were already restricted by the command policy,
    // and are refused here if they contain anything cmd.exe would interpret.
    if (win && step.argv.some((a) => /[&|<>^%"!`]/.test(a))) return resolve({ code: null, out: "an argument contains a character that is not allowed on Windows", timedOut: false });
    const child = spawn(step.argv[0] as string, step.argv.slice(1), { cwd, env, shell: win, detached: !win, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    let out = "";
    let size = 0;
    let timedOut = false;
    const take = (b: Buffer): void => { if (size < maxOut) { out += b.toString("utf8").slice(0, maxOut - size); size += b.length; if (size >= maxOut) out += "\n[output cut off]"; } };
    child.stdout?.on("data", take);
    child.stderr?.on("data", take);
    const kill = (): void => { try { if (win) child.kill("SIGKILL"); else process.kill(-(child.pid as number), "SIGKILL"); } catch { /* already gone */ } };
    const timer = setTimeout(() => { timedOut = true; kill(); }, Math.max(1000, left.ms));
    const started = Date.now();
    child.on("error", (e) => { clearTimeout(timer); resolve({ code: null, out: `could not start ${step.argv[0]}: ${e.message}`, timedOut: false }); });
    child.on("close", (code) => { clearTimeout(timer); left.ms -= Date.now() - started; resolve({ code: timedOut ? null : code, out, timedOut }); });
  });
}

export function scratchRunner(project: NodeProject, ws: ShadowWorkspace, o: ScratchOptions = {}): (plan: Extract<CommandPlan, { ok: true }>, command: string) => Promise<CommandResult> {
  return async (plan) => {
    const t0 = Date.now();
    const dir = await mkdtemp(join(tmpdir(), "clutor-run-"));
    try {
      await mkdir(join(dir, ".home"), { recursive: true });
      await materialise(project, ws, dir, o.maxCopyBytes ?? 200_000_000);
      const env = cleanEnv(dir, o);
      const left = { ms: o.timeoutMs ?? 120_000 };
      let output = "";
      for (const step of plan.steps) {
        const r = await runStep(step, dir, env, o.timeoutMs ?? 120_000, o.maxOutputBytes ?? 200_000, left);
        output += (output ? "\n" : "") + r.out;
        if (r.timedOut) return { exitCode: null, output, timedOut: true, durationMs: Date.now() - t0 };
        if (r.code !== 0) return { exitCode: r.code, output, timedOut: false, durationMs: Date.now() - t0 };
      }
      return { exitCode: 0, output, timedOut: false, durationMs: Date.now() - t0 };
    } catch (e) {
      return { exitCode: null, output: e instanceof Error ? e.message : String(e), timedOut: false, durationMs: Date.now() - t0 };
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  };
}
