import { computeUnifiedDiff, type FileDiff } from "./diff";
import { checkRead, checkWrite, JailError } from "./jail";

/**
 * The project as the agent sees it. Reads fall through to the real folder; every write goes to an in-memory overlay (the "shadow"),
 * so nothing touches the disk until the user applies the changes. The overlay also remembers the hash of each file as first read, so
 * applying can detect that someone edited the file in the meantime.
 */

export interface ProjectFs {
  /** Every file path in the project, relative, forward slashes, junk directories excluded. */
  list(): Promise<string[]>;
  /** The file's text, or null if it does not exist. Throws for binary or oversized files. */
  read(path: string): Promise<string | null>;
}

export class WorkspaceError extends Error {
  constructor(message: string, readonly code: "not_found" | "exists" | "no_match" | "ambiguous" | "too_large" | "too_many" | "binary" | "jail") {
    super(message);
    this.name = "WorkspaceError";
  }
}

export interface ShadowLimits {
  maxFileBytes: number;
  maxChangedFiles: number;
  maxTotalChangeBytes: number;
}
export const DEFAULT_LIMITS: ShadowLimits = { maxFileBytes: 1024 * 1024, maxChangedFiles: 200, maxTotalChangeBytes: 8 * 1024 * 1024 };

export async function sha256Hex(text: string): Promise<string> {
  const d = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

export class ShadowWorkspace {
  /** path -> new content, or null for a deletion. */
  private readonly overlay = new Map<string, string | null>();
  /** path -> hash of the file on disk when first read (null if it did not exist). */
  private readonly baseHash = new Map<string, string | null>();
  private listing: string[] | null = null;
  readonly limits: ShadowLimits;
  /** How many times the jail or a limit turned the agent away: shown in the gate report. */
  violations = 0;

  constructor(private readonly base: ProjectFs, limits: Partial<ShadowLimits> = {}) {
    this.limits = { ...DEFAULT_LIMITS, ...limits };
  }

  private async baseRead(path: string): Promise<string | null> {
    const text = await this.base.read(path);
    if (!this.baseHash.has(path)) this.baseHash.set(path, text === null ? null : await sha256Hex(text));
    return text;
  }

  private guard<T>(f: () => T): T {
    try { return f(); } catch (e) {
      if (e instanceof JailError) { this.violations++; throw new WorkspaceError(e.message, "jail"); }
      throw e;
    }
  }

  async list(): Promise<string[]> {
    this.listing ??= await this.base.list();
    const set = new Set(this.listing);
    for (const [p, v] of this.overlay) { if (v === null) set.delete(p); else set.add(p); }
    // Files the agent may not read are not shown to it at all.
    return [...set].filter((p) => { try { checkRead(p); return true; } catch { return false; } }).sort();
  }

  async exists(rawPath: string): Promise<boolean> {
    return (await this.read(rawPath)) !== null;
  }

  /** The current text of a file as the agent should see it (its own edits included), or null. */
  async read(rawPath: string): Promise<string | null> {
    const path = this.guard(() => checkRead(rawPath));
    if (this.overlay.has(path)) return this.overlay.get(path) ?? null;
    return this.baseRead(path);
  }

  async write(rawPath: string, content: string): Promise<{ created: boolean }> {
    const path = this.guard(() => checkWrite(rawPath));
    const bytes = new TextEncoder().encode(content).length;
    if (bytes > this.limits.maxFileBytes) { this.violations++; throw new WorkspaceError(`${path} would be ${bytes} bytes; the limit is ${this.limits.maxFileBytes}`, "too_large"); }
    if (!this.overlay.has(path) && this.overlay.size >= this.limits.maxChangedFiles) { this.violations++; throw new WorkspaceError(`at most ${this.limits.maxChangedFiles} files can change in one task`, "too_many"); }
    let total = bytes;
    for (const [p, v] of this.overlay) if (p !== path && v !== null) total += new TextEncoder().encode(v).length;
    if (total > this.limits.maxTotalChangeBytes) { this.violations++; throw new WorkspaceError("the task has changed too much text in total", "too_large"); }
    const existing = this.overlay.has(path) ? this.overlay.get(path) : await this.baseRead(path);
    this.overlay.set(path, content);
    return { created: existing === null || existing === undefined };
  }

  /** Replaces `oldText` with `newText`. Without `replaceAll`, `oldText` must occur exactly once, so an edit can never land in the wrong place. */
  async edit(rawPath: string, oldText: string, newText: string, replaceAll = false): Promise<{ replacements: number }> {
    if (oldText === "") throw new WorkspaceError("old text must not be empty. edit_file replaces text that is already in the file; to create a new file (or replace a whole file) use write_file with the full content", "no_match");
    const text = await this.read(rawPath);
    if (text === null) throw new WorkspaceError(`${rawPath} does not exist; use write_file to create it`, "not_found");
    const count = text.split(oldText).length - 1;
    if (count === 0) throw new WorkspaceError(`the old text was not found in ${rawPath}; read the file again and copy the text exactly, including whitespace`, "no_match");
    if (count > 1 && !replaceAll) throw new WorkspaceError(`the old text occurs ${count} times in ${rawPath}; include more surrounding lines so it is unique, or set replace_all`, "ambiguous");
    await this.write(rawPath, replaceAll ? text.split(oldText).join(newText) : text.replace(oldText, () => newText));
    return { replacements: replaceAll ? count : 1 };
  }

  async delete(rawPath: string): Promise<void> {
    const path = this.guard(() => checkWrite(rawPath));
    const existing = this.overlay.has(path) ? this.overlay.get(path) : await this.baseRead(path);
    if (existing === null || existing === undefined) throw new WorkspaceError(`${path} does not exist`, "not_found");
    this.overlay.set(path, null);
  }

  /** Forget every change. */
  reset(): void { this.overlay.clear(); }

  get changedPaths(): string[] { return [...this.overlay.keys()].sort(); }
  get size(): number { return this.overlay.size; }

  /** Unified diffs of every change against the files on disk, skipping writes that left a file as it was. */
  async diffs(): Promise<FileDiff[]> {
    const out: FileDiff[] = [];
    for (const path of this.changedPaths) {
      const before = await this.baseRead(path);
      const after = this.overlay.get(path) ?? null;
      const d = computeUnifiedDiff(path, before, after);
      if (d.additions || d.deletions || (before === null) !== (after === null)) out.push(d);
    }
    return out;
  }

  /** What to write to disk: only files that really differ, with the hash the file had when the agent first saw it. */
  async changeSet(): Promise<Array<{ path: string; before: string | null; after: string | null; baseHash: string | null }>> {
    const out: Array<{ path: string; before: string | null; after: string | null; baseHash: string | null }> = [];
    for (const path of this.changedPaths) {
      const before = await this.baseRead(path);
      const after = this.overlay.get(path) ?? null;
      if (before === after) continue;
      out.push({ path, before, after, baseHash: this.baseHash.get(path) ?? null });
    }
    return out;
  }

  /** Every file as it would be after applying the changes (used to build a scratch copy to run tests in). */
  async overlayEntries(): Promise<Array<[string, string | null]>> { return [...this.overlay]; }
}

/** A project held in memory, for tests and for previews. */
export class MemoryProject implements ProjectFs {
  constructor(readonly files: Map<string, string> = new Map()) {}
  static from(obj: Record<string, string>): MemoryProject { return new MemoryProject(new Map(Object.entries(obj))); }
  async list(): Promise<string[]> { return [...this.files.keys()].sort(); }
  async read(path: string): Promise<string | null> { return this.files.get(path) ?? null; }
}
