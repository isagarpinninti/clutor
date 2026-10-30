import { lstat, open, readdir, readFile, realpath } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import type { ProjectFs } from "../agent/workspace";

/** A project folder on disk, as the agent is allowed to see it. */

const SKIP_DIRS = new Set([".git", "node_modules", ".clutor", "dist", "build", "out", ".next", ".turbo", ".cache", "target", "__pycache__", ".venv", "venv", ".idea", ".vscode", "coverage", ".gradle", ".DS_Store"]);

export interface NodeProjectOptions {
  maxFiles?: number;
  maxFileBytes?: number;
  /** Extra directory names to skip. */
  skip?: string[];
}

export class ProjectTooLargeError extends Error {}

/** A tiny .gitignore: whole-name and path-prefix patterns, `*` wildcards, no negation. Enough to skip generated folders. */
export function parseIgnore(text: string): (rel: string, isDir: boolean) => boolean {
  const rules: RegExp[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || line.startsWith("!")) continue;
    const anchored = line.startsWith("/");
    const dirOnly = line.endsWith("/");
    const body = line.replace(/^\//, "").replace(/\/$/, "");
    const re = body.split("*").map((x) => x.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join("[^/]*");
    rules.push(new RegExp(`${anchored || body.includes("/") ? "^" : "(^|/)"}${re}${dirOnly ? "(/|$)" : "(/|$)"}`));
  }
  return (rel) => rules.some((r) => r.test(rel));
}

export class NodeProject implements ProjectFs {
  readonly root: string;
  private realRoot: string | null = null;
  private readonly maxFiles: number;
  private readonly maxFileBytes: number;
  private readonly skip: Set<string>;

  constructor(root: string, o: NodeProjectOptions = {}) {
    this.root = resolve(root);
    this.maxFiles = o.maxFiles ?? 20_000;
    this.maxFileBytes = o.maxFileBytes ?? 1024 * 1024;
    this.skip = new Set([...SKIP_DIRS, ...(o.skip ?? [])]);
  }

  private async rr(): Promise<string> {
    return (this.realRoot ??= await realpath(this.root));
  }

  async list(): Promise<string[]> {
    const out: string[] = [];
    let ignored: (rel: string, isDir: boolean) => boolean = () => false;
    try { ignored = parseIgnore(await readFile(join(this.root, ".gitignore"), "utf8")); } catch { /* no .gitignore */ }
    const walk = async (dir: string): Promise<void> => {
      for (const e of await readdir(dir, { withFileTypes: true })) {
        const abs = join(dir, e.name);
        const rel = relative(this.root, abs).split(sep).join("/");
        if (e.isDirectory()) {
          if (this.skip.has(e.name) || ignored(rel, true)) continue;
          await walk(abs);
        } else if (e.isFile() && !ignored(rel, false)) {
          if (out.length >= this.maxFiles) throw new ProjectTooLargeError(`the folder has more than ${this.maxFiles} files; open a smaller folder`);
          out.push(rel);
        }
        // symlinks are neither listed nor followed
      }
    };
    await walk(this.root);
    return out.sort();
  }

  /** The real path of `rel`, refusing anything that is a link out of the project. */
  async safePath(rel: string): Promise<string> {
    const abs = resolve(this.root, rel);
    const root = await this.rr();
    if (abs !== this.root && !abs.startsWith(this.root + sep)) throw new Error(`${rel} is outside the project`);
    // Resolve the deepest existing ancestor, so a symlinked directory in the middle of the path is caught too.
    let probe = abs;
    for (;;) {
      try {
        const real = await realpath(probe);
        if (real !== root && !real.startsWith(root + sep)) throw new Error(`${rel} resolves outside the project through a link`);
        break;
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
        const parent = resolve(probe, "..");
        if (parent === probe) break;
        probe = parent;
      }
    }
    return abs;
  }

  async read(rel: string): Promise<string | null> {
    const abs = await this.safePath(rel);
    let st;
    try { st = await lstat(abs); } catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT" || (e as NodeJS.ErrnoException).code === "ENOTDIR") return null; throw e; }
    if (st.isSymbolicLink()) throw new Error(`${rel} is a symbolic link`);
    if (!st.isFile()) return null;
    if (st.size > this.maxFileBytes) throw new Error(`${rel} is ${st.size} bytes; files over ${this.maxFileBytes} bytes are not read`);
    const fh = await open(abs, "r");
    try {
      const buf = Buffer.alloc(st.size);
      await fh.read(buf, 0, st.size, 0);
      if (buf.subarray(0, 8192).includes(0)) throw new Error(`${rel} is a binary file`);
      return buf.toString("utf8");
    } finally { await fh.close(); }
  }
}
