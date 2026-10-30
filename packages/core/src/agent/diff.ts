/**
 * Line diffs: Myers' O(ND) algorithm, a unified-diff writer, and a patch applier. The applier exists so the writer can be tested
 * by round trip: applying the diff of (a, b) to a must give b, for any a and b.
 */

export type Edit = { kind: "keep" | "add" | "del"; line: string };

export function splitLines(text: string): string[] {
  if (text === "") return [];
  const lines = text.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

export function diffLines(a: string[], b: string[]): Edit[] {
  // Trim the common head and tail first: most edits touch a small region of a large file.
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const mid = myers(a.slice(head, a.length - tail), b.slice(head, b.length - tail));
  return [
    ...a.slice(0, head).map((line): Edit => ({ kind: "keep", line })),
    ...mid,
    ...a.slice(a.length - tail).map((line): Edit => ({ kind: "keep", line })),
  ];
}

function myers(a: string[], b: string[]): Edit[] {
  const n = a.length, m = b.length;
  if (n === 0) return b.map((line) => ({ kind: "add" as const, line }));
  if (m === 0) return a.map((line) => ({ kind: "del" as const, line }));
  const max = n + m;
  const offset = max;
  let v = new Int32Array(2 * max + 2);
  const trace: Int32Array[] = [];
  for (let d = 0; d <= max; d++) {
    trace.push(v.slice());
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && (v[offset + k - 1] as number) < (v[offset + k + 1] as number)) ? (v[offset + k + 1] as number) : (v[offset + k - 1] as number) + 1;
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) { x++; y++; }
      v[offset + k] = x;
      if (x >= n && y >= m) return backtrack(trace, a, b, d, offset);
    }
    v = v.slice();
  }
  throw new Error("unreachable: a diff always exists");
}

function backtrack(trace: Int32Array[], a: string[], b: string[], dEnd: number, offset: number): Edit[] {
  const out: Edit[] = [];
  let x = a.length, y = b.length;
  for (let d = dEnd; d > 0; d--) {
    const v = trace[d] as Int32Array;
    const k = x - y;
    const prevK = k === -d || (k !== d && (v[offset + k - 1] as number) < (v[offset + k + 1] as number)) ? k + 1 : k - 1;
    const prevX = v[offset + prevK] as number;
    const prevY = prevX - prevK;
    while (x > prevX && y > prevY) { x--; y--; out.push({ kind: "keep", line: a[x] as string }); }
    if (x === prevX) { y--; out.push({ kind: "add", line: b[y] as string }); }
    else { x--; out.push({ kind: "del", line: a[x] as string }); }
  }
  while (x > 0 && y > 0) { x--; y--; out.push({ kind: "keep", line: a[x] as string }); }
  return out.reverse();
}

export interface FileDiff {
  path: string;
  status: "added" | "modified" | "deleted";
  additions: number;
  deletions: number;
  /** Unified diff text, empty when nothing changed. */
  patch: string;
}

/** A unified diff of one file with `context` unchanged lines around each change. */
export function computeUnifiedDiff(path: string, before: string | null, after: string | null, context = 3): FileDiff {
  const a = before === null ? [] : splitLines(before);
  const b = after === null ? [] : splitLines(after);
  const edits = diffLines(a, b);
  const additions = edits.filter((e) => e.kind === "add").length;
  const deletions = edits.filter((e) => e.kind === "del").length;
  const status = before === null ? "added" : after === null ? "deleted" : "modified";
  if (additions === 0 && deletions === 0) return { path, status, additions, deletions, patch: "" };

  // Group changes into hunks, merging those whose context overlaps.
  const changeAt: number[] = [];
  edits.forEach((e, i) => { if (e.kind !== "keep") changeAt.push(i); });
  const ranges: Array<[number, number]> = [];
  for (const i of changeAt) {
    const lo = Math.max(0, i - context), hi = Math.min(edits.length, i + context + 1);
    const last = ranges[ranges.length - 1];
    if (last && lo <= last[1]) last[1] = Math.max(last[1], hi);
    else ranges.push([lo, hi]);
  }
  let aLine = 1, bLine = 1;
  const startsA: number[] = [], startsB: number[] = [];
  edits.forEach((e, i) => { startsA[i] = aLine; startsB[i] = bLine; if (e.kind !== "add") aLine++; if (e.kind !== "del") bLine++; });
  const out = [`--- ${before === null ? "/dev/null" : `a/${path}`}`, `+++ ${after === null ? "/dev/null" : `b/${path}`}`];
  for (const [lo, hi] of ranges) {
    const slice = edits.slice(lo, hi);
    const aCount = slice.filter((e) => e.kind !== "add").length, bCount = slice.filter((e) => e.kind !== "del").length;
    const aStart = aCount === 0 ? (startsA[lo] as number) - 1 : (startsA[lo] as number);
    const bStart = bCount === 0 ? (startsB[lo] as number) - 1 : (startsB[lo] as number);
    out.push(`@@ -${aStart},${aCount} +${bStart},${bCount} @@`);
    for (const e of slice) out.push((e.kind === "keep" ? " " : e.kind === "add" ? "+" : "-") + e.line);
  }
  return { path, status, additions, deletions, patch: out.join("\n") + "\n" };
}

/** Applies the hunks of a unified diff (as written by `computeUnifiedDiff`) to `before`. Throws if a hunk does not match. */
export function applyUnifiedDiff(before: string | null, patch: string): string | null {
  const lines = patch.split("\n");
  if (patch === "") return before;
  const src = before === null ? [] : splitLines(before);
  const out: string[] = [];
  let at = 0;
  let i = 0;
  const toDevNull = lines[1]?.startsWith("+++ /dev/null") ?? false;
  while (i < lines.length && !(lines[i] as string).startsWith("@@")) i++;
  for (; i < lines.length; i++) {
    const h = /^@@ -(\d+),(\d+) \+(\d+),(\d+) @@$/.exec(lines[i] as string);
    if (!h) { if ((lines[i] as string) === "") continue; throw new Error(`bad hunk header: ${lines[i]}`); }
    const aStart = Number(h[2]) === 0 ? Number(h[1]) : Number(h[1]) - 1;
    while (at < aStart) out.push(src[at++] as string);
    for (i++; i < lines.length && !(lines[i] as string).startsWith("@@"); i++) {
      const l = lines[i] as string;
      if (l === "" && i === lines.length - 1) break;
      const tag = l[0], text = l.slice(1);
      if (tag === " ") { if (src[at] !== text) throw new Error(`context mismatch at line ${at + 1}`); out.push(text); at++; }
      else if (tag === "-") { if (src[at] !== text) throw new Error(`deletion mismatch at line ${at + 1}`); at++; }
      else if (tag === "+") out.push(text);
      else throw new Error(`bad patch line: ${l}`);
    }
    i--;
  }
  while (at < src.length) out.push(src[at++] as string);
  if (toDevNull && out.length === 0) return null;
  return out.length ? out.join("\n") + "\n" : "";
}
