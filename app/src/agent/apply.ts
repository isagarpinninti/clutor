import { sha256Hex } from "@core/agent/workspace";
import type { AgentHost } from "../bridge";

/**
 * Putting the agent's changes into the real folder, and taking them back. Nothing is written unless every file is still as the agent saw it (otherwise you changed it
 * meanwhile and the apply is refused), and each apply keeps a checkpoint with the old contents so it can be undone.
 */

export interface Change { path: string; before: string | null; after: string | null; baseHash: string | null }

interface Manifest { id: string; created: number; title: string; files: Array<{ path: string; before: string | null; afterHash: string | null }> }

const DIR = ".clutor/checkpoints";
const INDEX = `${DIR}/index.json`;

export class ConflictError extends Error {
  constructor(readonly paths: string[]) {
    super(`these files changed since the agent read them: ${paths.join(", ")}`);
    this.name = "ConflictError";
  }
}

const hashOf = async (t: string | null): Promise<string | null> => (t === null ? null : sha256Hex(t));

async function readIndex(host: AgentHost, root: string): Promise<Array<{ id: string; created: number; title: string; undone?: boolean }>> {
  try { const raw = await host.read(root, INDEX); return raw ? JSON.parse(raw) : []; } catch { return []; }
}

export async function listCheckpoints(host: AgentHost, root: string): Promise<Array<{ id: string; created: number; title: string; undone?: boolean }>> {
  return (await readIndex(host, root)).slice().reverse();
}

export async function applyChanges(host: AgentHost, root: string, changes: Change[], title: string, now: () => number = Date.now): Promise<string> {
  const conflicts: string[] = [];
  for (const c of changes) {
    const current = await host.read(root, c.path).catch(() => "");
    if ((await hashOf(current)) !== c.baseHash) conflicts.push(c.path);
  }
  if (conflicts.length) throw new ConflictError(conflicts);

  const id = `ckpt_${now().toString(36)}`;
  const manifest: Manifest = { id, created: now(), title, files: await Promise.all(changes.map(async (c) => ({ path: c.path, before: c.before, afterHash: await hashOf(c.after) }))) };
  // The checkpoint is saved before the first file is touched.
  await host.write(root, `${DIR}/${id}.json`, JSON.stringify(manifest));
  const index = await readIndex(host, root);
  await host.write(root, INDEX, JSON.stringify([...index, { id, created: manifest.created, title }]));
  const done: Change[] = [];
  try {
    for (const c of changes) { await host.write(root, c.path, c.after); done.push(c); }
  } catch (e) {
    for (const c of done.reverse()) await host.write(root, c.path, c.before).catch(() => undefined);
    throw e;
  }
  return id;
}

/** Restores the files of a checkpoint. Refused if a file has been changed since the apply (unless `force`). */
export async function undoCheckpoint(host: AgentHost, root: string, id: string, force = false): Promise<void> {
  const raw = await host.read(root, `${DIR}/${id}.json`);
  if (!raw) throw new Error("that checkpoint no longer exists");
  const m = JSON.parse(raw) as Manifest;
  if (!force) {
    const conflicts: string[] = [];
    for (const f of m.files) {
      const current = await host.read(root, f.path).catch(() => "");
      if ((await hashOf(current)) !== f.afterHash) conflicts.push(f.path);
    }
    if (conflicts.length) throw new ConflictError(conflicts);
  }
  for (const f of m.files) await host.write(root, f.path, f.before);
  const index = await readIndex(host, root);
  await host.write(root, INDEX, JSON.stringify(index.map((i) => (i.id === id ? { ...i, undone: true } : i))));
}
