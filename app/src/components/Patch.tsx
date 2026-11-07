/** A unified diff, with added and removed lines coloured. */
export function Patch({ patch }: { patch: string }) {
  return (
    <pre className="patch">{patch.split("\n").map((l, i) => <div key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : ""}>{l || " "}</div>)}</pre>
  );
}
