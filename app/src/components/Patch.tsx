import { useState } from "react";

type Row = { text: string; kind: "add" | "del" | "hunk" | "ctx" };
const classify = (l: string): Row["kind"] => (l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : "ctx");

/**
 * A unified diff showing what changed: the added and removed lines, one line of context around each change, and the hunk markers. A long diff (a whole new file is all added
 * lines) shows its first lines and a button for the rest.
 */
export function Patch({ patch, limit = 40 }: { patch: string; limit?: number }) {
  const [all, setAll] = useState(false);
  const lines = patch.split("\n").filter((l) => !l.startsWith("---") && !l.startsWith("+++") && !l.startsWith("diff ") && !l.startsWith("index "));
  const rows: Row[] = lines.map((text) => ({ text, kind: classify(text) }));
  const near = (i: number): boolean => rows[i - 1]?.kind === "add" || rows[i - 1]?.kind === "del" || rows[i + 1]?.kind === "add" || rows[i + 1]?.kind === "del";
  const kept = rows.filter((r, i) => r.kind !== "ctx" || near(i));
  const shown = all ? kept : kept.slice(0, limit);
  return (
    <>
      <pre className="patch">{shown.map((r, i) => <div key={i} className={r.kind === "ctx" ? "" : r.kind}>{r.text || " "}</div>)}</pre>
      {kept.length > limit ? <button type="button" className="link-btn patch-more" onClick={() => setAll((v) => !v)}>{all ? "Show less" : `Show ${kept.length - limit} more lines`}</button> : null}
    </>
  );
}
