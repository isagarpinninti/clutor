import { existsSync, readdirSync, statSync, type Dirent } from "node:fs";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

export interface FoundModel { path: string; name: string; bytes: number }

/** Every .gguf under `dir`, a few levels deep. Multimodal projectors and the later parts of a split model are not models you can chat with. */
export function findGguf(dir: string, depth = 4): FoundModel[] {
  const out: FoundModel[] = [];
  const walk = (d: string, left: number): void => {
    let entries: Dirent[];
    try { entries = readdirSync(d, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (e.name.startsWith(".") || e.name === "node_modules") continue;
      const p = join(d, e.name);
      if (e.isDirectory()) { if (left > 0) walk(p, left - 1); continue; }
      if (!e.isFile() || !e.name.toLowerCase().endsWith(".gguf")) continue;
      if (/^mmproj/i.test(e.name) || laterPart(e.name)) continue;
      try { out.push({ path: p, name: displayName(p), bytes: statSync(p).size }); } catch { /* vanished */ }
    }
  };
  walk(dir, depth);
  return out;
}

function laterPart(name: string): boolean {
  const m = /-(\d+)-of-\d+\.gguf$/i.exec(name);
  return m !== null && Number(m[1]) > 1;
}

export function displayName(path: string): string { return basename(path).replace(/\.gguf$/i, "").replace(/-?\d{5}-of-\d{5}$/, ""); }

/** The clutor checkout this command runs from (…/packages/cli/dist → the repository), or undefined when it is an installed package. */
export function checkoutRoot(): string | undefined {
  try {
    const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
    return existsSync(join(root, "c", "Makefile")) && existsSync(join(root, "packages", "cli")) ? root : undefined;
  } catch { return undefined; }
}

/**
 * Where models are looked for: the configured folder and `models` / `Models` under the folder you are in, ~/.clutor/models, the `models` and
 * `Models` folders of the clutor checkout this command runs from (so it works from any directory), and folders models were chosen from before.
 */
export function modelFolders(root: string, configured: string, remembered: string[] = []): string[] {
  const co = checkoutRoot();
  return [...new Set([resolve(root, configured), resolve(root, "models"), resolve(root, "Models"), join(homedir(), ".clutor", "models"), ...(co ? [join(co, "models"), join(co, "Models")] : []), ...remembered])];
}

export function discoverModels(root: string, configured: string, remembered: string[] = []): FoundModel[] {
  const seen = new Set<string>();
  const out: FoundModel[] = [];
  for (const d of modelFolders(root, configured, remembered)) {
    for (const m of findGguf(d)) {
      let key = m.path;
      try { key = `${statSync(m.path).dev}:${statSync(m.path).ino}`; } catch { /* keep the path */ }
      if (!seen.has(key)) { seen.add(key); out.push(m); }
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

export function gb(bytes: number): string { return `${(bytes / 1e9).toFixed(1)} GB`; }
