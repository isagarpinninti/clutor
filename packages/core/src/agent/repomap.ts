import type { ShadowWorkspace } from "./workspace";

/**
 * A compact map of the project for the model's first message: the file tree, and the top-level symbols of source files. Symbols are
 * found with per-language patterns, not a parser, so the map can miss or over-report; it is a guide for choosing what to read, and
 * the agent is told to read a file before editing it.
 */

const SYMBOLS: Array<{ ext: RegExp; patterns: RegExp[] }> = [
  { ext: /\.(ts|tsx|js|jsx|mjs|cjs)$/, patterns: [/^export\s+(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum|const|let)\s+([A-Za-z_$][\w$]*)/, /^(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)/, /^class\s+([A-Za-z_$][\w$]*)/] },
  { ext: /\.py$/, patterns: [/^(?:async\s+)?def\s+([A-Za-z_]\w*)/, /^class\s+([A-Za-z_]\w*)/] },
  { ext: /\.go$/, patterns: [/^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)/, /^type\s+([A-Za-z_]\w*)/] },
  { ext: /\.rs$/, patterns: [/^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait|mod)\s+([A-Za-z_]\w*)/, /^(?:async\s+)?fn\s+([A-Za-z_]\w*)/] },
  { ext: /\.(java|kt|cs|swift)$/, patterns: [/^\s*(?:public|internal|open|final|abstract)?\s*(?:class|interface|enum|struct|object)\s+([A-Za-z_]\w*)/] },
  { ext: /\.(c|h|cc|cpp|hpp)$/, patterns: [/^[A-Za-z_][\w\s*]*\s+\*?([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?$/] },
];

export function extractSymbols(path: string, text: string, limit = 24): string[] {
  const lang = SYMBOLS.find((s) => s.ext.test(path));
  if (!lang) return [];
  const out: string[] = [];
  for (const line of text.split("\n")) {
    for (const p of lang.patterns) {
      const m = p.exec(line);
      if (m?.[1]) { out.push(m[1]); break; }
    }
    if (out.length >= limit) break;
  }
  return [...new Set(out)];
}

const KEY_FILES = /^(readme(\.md)?|package\.json|tsconfig\.json|pyproject\.toml|requirements\.txt|cargo\.toml|go\.mod|makefile|dockerfile)$/i;
const SOURCE = /\.(ts|tsx|js|jsx|mjs|py|go|rs|java|kt|cs|swift|c|h|cc|cpp|hpp)$/;

export async function buildRepoMap(ws: ShadowWorkspace, maxChars = 6000): Promise<string> {
  const files = await ws.list();
  const lines: string[] = [`${files.length} files`];
  const dirs = new Map<string, number>();
  for (const f of files) { const d = f.includes("/") ? f.slice(0, f.indexOf("/")) : "."; dirs.set(d, (dirs.get(d) ?? 0) + 1); }
  lines.push("Top-level: " + [...dirs].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([d, n]) => `${d}${d === "." ? "" : "/"} (${n})`).join(", "));
  const key = files.filter((f) => KEY_FILES.test(f.split("/").pop() ?? "") && f.split("/").length <= 2);
  if (key.length) lines.push("Key files: " + key.join(", "));
  lines.push("");
  const ranked = files.filter((f) => SOURCE.test(f)).sort((a, b) => depth(a) - depth(b) || a.localeCompare(b));
  let used = lines.join("\n").length;
  for (const f of ranked) {
    const text = await ws.read(f).catch(() => null);
    if (text === null) continue;
    const syms = extractSymbols(f, text);
    const line = syms.length ? `${f}: ${syms.join(", ")}` : f;
    if (used + line.length + 1 > maxChars) { lines.push(`... ${ranked.length - lines.length + 5} more source files not shown`); break; }
    lines.push(line);
    used += line.length + 1;
  }
  return lines.join("\n");
}
const depth = (p: string): number => p.split("/").length;
