import { createHash, randomBytes } from "node:crypto";
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { type NodeProject } from "./project";

/**
 * Turning the agent's shadow changes into real files, safely: each file is checked against the version the agent saw (so a change
 * made in your editor meanwhile is never overwritten silently), the originals are saved in a checkpoint first, and every write is a
 * temp file plus rename so a crash never leaves half a file. A checkpoint can be rolled back.
 */

export interface Change { path: string; before: string | null; after: string | null; baseHash: string | null }
export interface AppliedChange { path: string; status: "added" | "modified" | "deleted" }
export interface ApplyResult { id: string; applied: AppliedChange[] }

export class ConflictError extends Error {
  constructor(readonly conflicts: Array<{ path: string; reason: string }>) {
    super(`these files changed on disk since the agent read them: ${conflicts.map((c) => c.path).join(", ")}`);
    this.name = "ConflictError";
  }
}

const sha = (t: string): string => createHash("sha256").update(t).digest("hex");
const CHECKPOINTS = ".clutor/checkpoints";

interface Manifest { id: string; createdAt: string; files: Array<{ path: string; existed: boolean; beforeSha: string | null; afterSha: string | null }> }

async function readIfExists(abs: string): Promise<string | null> {
  try { return await readFile(abs, "utf8"); } catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") return null; throw e; }
}

async function writeAtomic(abs: string, text: string): Promise<void> {
  await mkdir(dirname(abs), { recursive: true });
  let mode: number | undefined;
  try { mode = (await stat(abs)).mode & 0o777; } catch { /* new file */ }
  const tmp = `${abs}.${randomBytes(4).toString("hex")}.clutor-tmp`;
  await writeFile(tmp, text, "utf8");
  if (mode !== undefined) await chmod(tmp, mode);
  await rename(tmp, abs);
}

export async function applyChanges(project: NodeProject, changes: Change[], opts: { force?: boolean } = {}): Promise<ApplyResult> {
  const conflicts: Array<{ path: string; reason: string }> = [];
  const current = new Map<string, string | null>();
  for (const c of changes) {
    const abs = await project.safePath(c.path);
    const now = await readIfExists(abs);
    current.set(c.path, now);
    const nowHash = now === null ? null : sha(now);
    if (nowHash !== c.baseHash) conflicts.push({ path: c.path, reason: now === null ? "it was deleted" : c.baseHash === null ? "it now exists" : "it was edited" });
  }
  if (conflicts.length && !opts.force) throw new ConflictError(conflicts);

  const id = `ckpt_${Date.now().toString(36)}${randomBytes(3).toString("hex")}`;
  const dir = join(project.root, CHECKPOINTS, id);
  const manifest: Manifest = { id, createdAt: new Date().toISOString(), files: [] };
  for (const c of changes) {
    const now = current.get(c.path) ?? null;
    manifest.files.push({ path: c.path, existed: now !== null, beforeSha: now === null ? null : sha(now), afterSha: c.after === null ? null : sha(c.after) });
    if (now !== null) {
      await mkdir(dirname(join(dir, "files", c.path)), { recursive: true });
      await writeFile(join(dir, "files", c.path), now, "utf8");
    }
  }
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2));

  const applied: AppliedChange[] = [];
  for (const c of changes) {
    const abs = await project.safePath(c.path);
    if (c.after === null) { await rm(abs, { force: true }); applied.push({ path: c.path, status: "deleted" }); }
    else { await writeAtomic(abs, c.after); applied.push({ path: c.path, status: (current.get(c.path) ?? null) === null ? "added" : "modified" }); }
  }
  return { id, applied };
}

/** Puts every file of a checkpoint back. Refuses if a file was edited after the apply, unless forced. */
export async function rollbackCheckpoint(project: NodeProject, id: string, opts: { force?: boolean } = {}): Promise<AppliedChange[]> {
  if (!/^ckpt_[a-z0-9]+$/.test(id)) throw new Error("not a checkpoint id");
  const dir = join(project.root, CHECKPOINTS, id);
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as Manifest;
  const conflicts: Array<{ path: string; reason: string }> = [];
  for (const f of manifest.files) {
    const now = await readIfExists(await project.safePath(f.path));
    const h = now === null ? null : sha(now);
    if (h !== f.afterSha) conflicts.push({ path: f.path, reason: "it was edited after the agent's changes were applied" });
  }
  if (conflicts.length && !opts.force) throw new ConflictError(conflicts);
  const restored: AppliedChange[] = [];
  for (const f of manifest.files) {
    const abs = await project.safePath(f.path);
    if (f.existed) { await writeAtomic(abs, await readFile(join(dir, "files", f.path), "utf8")); restored.push({ path: f.path, status: "modified" }); }
    else { await rm(abs, { force: true }); restored.push({ path: f.path, status: "deleted" }); }
  }
  return restored;
}

export async function listCheckpoints(project: NodeProject): Promise<Array<{ id: string; createdAt: string; files: number }>> {
  const { readdir } = await import("node:fs/promises");
  try {
    const ids = (await readdir(join(project.root, CHECKPOINTS))).filter((x) => x.startsWith("ckpt_")).sort().reverse();
    const out = [];
    for (const id of ids) {
      try { const m = JSON.parse(await readFile(join(project.root, CHECKPOINTS, id, "manifest.json"), "utf8")) as Manifest; out.push({ id, createdAt: m.createdAt, files: m.files.length }); } catch { /* skip a broken one */ }
    }
    return out;
  } catch { return []; }
}
