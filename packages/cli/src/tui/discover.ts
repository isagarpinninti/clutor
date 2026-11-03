import { readdirSync, statSync, type Dirent } from "node:fs";
import { homedir } from "node:os";
import { basename, join, resolve } from "node:path";

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

/** The folders a first run looks in: the configured models folder, `Models` beside it, and ~/.clutor/models. */
export function modelFolders(root: string, configured: string): string[] {
  return [...new Set([resolve(root, configured), resolve(root, "models"), resolve(root, "Models"), join(homedir(), ".clutor", "models")])];
}

export function discoverModels(root: string, configured: string): FoundModel[] {
  const seen = new Set<string>();
  const out: FoundModel[] = [];
  for (const d of modelFolders(root, configured)) {
    for (const m of findGguf(d)) {
      let key = m.path;
      try { key = `${statSync(m.path).dev}:${statSync(m.path).ino}`; } catch { /* keep the path */ }
      if (!seen.has(key)) { seen.add(key); out.push(m); }
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

export function gb(bytes: number): string { return `${(bytes / 1e9).toFixed(1)} GB`; }
