import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { JailError, applyUnifiedDiff, checkRead, checkWrite, computeUnifiedDiff, diffLines, normalizeRelPath, splitLines } from "../src";

const lines = fc.array(fc.constantFrom("a", "b", "c", "d", "", "function f() {", "}", "  return 1;"), { maxLength: 40 });

describe("line diff", () => {
  it("produces a minimal edit script for a known case", () => {
    const e = diffLines(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"]);
    expect(e.filter((x) => x.kind === "del").map((x) => x.line)).toEqual(["b"]);
    expect(e.filter((x) => x.kind === "add").map((x) => x.line)).toEqual(["x", "e"]);
  });
  it("replays: keeping + deleting gives the old text, keeping + adding gives the new text, for any pair", () => {
    fc.assert(fc.property(lines, lines, (a, b) => {
      const e = diffLines(a, b);
      expect(e.filter((x) => x.kind !== "add").map((x) => x.line)).toEqual(a);
      expect(e.filter((x) => x.kind !== "del").map((x) => x.line)).toEqual(b);
    }), { numRuns: 500 });
  });
  it("is minimal: the number of changes equals n + m - 2 * LCS", () => {
    const lcs = (a: string[], b: string[]): number => {
      const t = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
      for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) t[i]![j] = a[i - 1] === b[j - 1] ? t[i - 1]![j - 1]! + 1 : Math.max(t[i - 1]![j]!, t[i]![j - 1]!);
      return t[a.length]![b.length]!;
    };
    fc.assert(fc.property(lines, lines, (a, b) => {
      const changes = diffLines(a, b).filter((x) => x.kind !== "keep").length;
      expect(changes).toBe(a.length + b.length - 2 * lcs(a, b));
    }), { numRuns: 300 });
  });
});

describe("unified diff", () => {
  it("writes git-style hunks with context", () => {
    const d = computeUnifiedDiff("src/x.ts", "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n", "one\ntwo\nthree\nfour\nFIVE\nsix\nseven\neight\nnine\nten\n");
    expect(d).toMatchObject({ status: "modified", additions: 1, deletions: 1 });
    expect(d.patch).toBe("--- a/src/x.ts\n+++ b/src/x.ts\n@@ -2,7 +2,7 @@\n two\n three\n four\n-five\n+FIVE\n six\n seven\n eight\n");
  });
  it("reports added and deleted files, and nothing for an unchanged one", () => {
    expect(computeUnifiedDiff("n.txt", null, "a\nb\n")).toMatchObject({ status: "added", additions: 2, deletions: 0 });
    expect(computeUnifiedDiff("n.txt", "a\nb\n", null)).toMatchObject({ status: "deleted", additions: 0, deletions: 2 });
    expect(computeUnifiedDiff("n.txt", "a\n", "a\n").patch).toBe("");
  });
  it("round-trips: applying the diff of (a, b) to a gives b, for any texts", () => {
    fc.assert(fc.property(lines, lines, fc.integer({ min: 0, max: 4 }), (a, b, ctx) => {
      const before = a.length ? a.join("\n") + "\n" : "";
      const after = b.length ? b.join("\n") + "\n" : "";
      const d = computeUnifiedDiff("f.txt", before, after, ctx);
      expect(applyUnifiedDiff(before, d.patch)).toBe(after);
    }), { numRuns: 500 });
  });
  it("refuses to apply a patch to the wrong text", () => {
    const d = computeUnifiedDiff("f", "a\nb\nc\n", "a\nX\nc\n");
    expect(() => applyUnifiedDiff("a\nDIFFERENT\nc\n", d.patch)).toThrow(/mismatch/);
  });
  it("keeps file creation and deletion round-trippable", () => {
    expect(applyUnifiedDiff(null, computeUnifiedDiff("n", null, "x\ny\n").patch)).toBe("x\ny\n");
    expect(applyUnifiedDiff("x\ny\n", computeUnifiedDiff("n", "x\ny\n", null).patch)).toBeNull();
  });
  it("splits text into lines the way a diff counts them", () => {
    expect(splitLines("")).toEqual([]);
    expect(splitLines("a\n")).toEqual(["a"]);
    expect(splitLines("a\n\n")).toEqual(["a", ""]);
    expect(splitLines("a")).toEqual(["a"]);
  });
});

describe("path jail", () => {
  it("cleans ordinary paths", () => {
    expect(normalizeRelPath("src/./a/../b.ts")).toBe("src/b.ts");
    expect(normalizeRelPath("a\\b\\c.txt")).toBe("a/b/c.txt");
    expect(normalizeRelPath("./x")).toBe("x");
    expect(normalizeRelPath("a//b")).toBe("a/b");
  });
  it("refuses every way out of the project", () => {
    for (const bad of ["/etc/passwd", "C:\\Windows\\System32", "c:/x", "\\\\server\\share", "//host/x", "~/secrets", "../x", "a/../../x", "..\\x", "a/..", ".", "", "a\0b", "x".repeat(2000)]) {
      expect(() => normalizeRelPath(bad), JSON.stringify(bad.slice(0, 30))).toThrow(JailError);
    }
  });
  it("keeps secrets unreadable and protected places unwritable, but lets ordinary code through", () => {
    for (const p of [".env", ".env.local", "config/.env.production", ".ssh/id_rsa", "a/id_ed25519", "cert.pem", ".aws/credentials", ".netrc", ".clutor/keys/receipt-p256.jwk.json"]) expect(() => checkRead(p), p).toThrow(/secrets/);
    for (const p of [".git/config", ".git/hooks/pre-commit", ".clutor/ledger.jsonl", ".env", "key.pem", "package/.npmrc", "node_modules/x/index.js"]) expect(() => checkWrite(p), p).toThrow(JailError);
    expect(checkRead(".git/config")).toBe(".git/config");
    expect(checkWrite("src/environment.ts")).toBe("src/environment.ts");
    expect(checkWrite(".github/workflows/ci.yml")).toBe(".github/workflows/ci.yml");
    expect(checkRead("src/.envelope.ts")).toBe("src/.envelope.ts");
  });
  it("never lets any generated path resolve outside the root", () => {
    fc.assert(fc.property(fc.array(fc.constantFrom("a", "b", "..", ".", "", "c.txt", "\\", "%2e%2e"), { maxLength: 12 }), (segs) => {
      try {
        const p = normalizeRelPath(segs.join("/"));
        expect(p.split("/").every((s) => s !== ".." && s !== "." && s !== "")).toBe(true);
        expect(p.startsWith("/")).toBe(false);
      } catch (e) {
        expect(e).toBeInstanceOf(JailError);
      }
    }), { numRuns: 1000 });
  });
});
