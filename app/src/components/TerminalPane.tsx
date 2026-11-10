import { useEffect, useRef, useState } from "react";
import { bridge } from "../bridge";
import { uid } from "../store";

const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;
const MAX_LINES = 2000;

/** A plain command line for the project folder. It runs what you type with your own permissions and is not sandboxed; output streams in, and Stop ends the command. */
export function TerminalPane({ root }: { root: string }) {
  const [lines, setLines] = useState<Array<{ text: string; kind: "cmd" | "out" | "exit" }>>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState<string>();
  const [history, setHistory] = useState<string[]>([]);
  const at = useRef(-1);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [lines]);

  const add = (more: Array<{ text: string; kind: "cmd" | "out" | "exit" }>): void => setLines((l) => [...l, ...more].slice(-MAX_LINES));

  const run = (): void => {
    const command = input.trim();
    if (!command || running) return;
    const id = uid();
    setInput(""); at.current = -1; setHistory((h) => [command, ...h.filter((x) => x !== command)].slice(0, 50));
    add([{ text: command, kind: "cmd" }]);
    setRunning(id);
    bridge.host.termRun(id, root, command, (text) => add(text.replace(ANSI, "").split("\n").filter((t, i, a) => !(i === a.length - 1 && t === "")).map((t) => ({ text: t, kind: "out" as const }))),
      (code) => { if (code !== 0) add([{ text: `exit ${code ?? "(stopped)"}`, kind: "exit" }]); setRunning(undefined); })
      .catch((e: unknown) => { add([{ text: e instanceof Error ? e.message : String(e), kind: "exit" }]); setRunning(undefined); });
  };

  return (
    <div className="term">
      <p className="term-note">Runs in <code>{root.split(/[\\/]/).filter(Boolean).pop()}</code> with your permissions. Not sandboxed.</p>
      <div className="term-out" role="log" aria-live="polite">
        {lines.length === 0 ? <span className="empty-hint">Type a command below.</span> : null}
        {lines.map((l, i) => <div key={i} className={l.kind}>{l.kind === "cmd" ? <><i>$</i> {l.text}</> : l.text || " "}</div>)}
        <div ref={end} />
      </div>
      <div className="term-in">
        <i>$</i>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={running ? "running…" : "command"} aria-label="Command" spellCheck={false} autoCapitalize="off" autoCorrect="off"
          onKeyDown={(e) => {
            if (e.key === "Enter") run();
            else if (e.key === "ArrowUp") { e.preventDefault(); at.current = Math.min(history.length - 1, at.current + 1); setInput(history[at.current] ?? input); }
            else if (e.key === "ArrowDown") { e.preventDefault(); at.current = Math.max(-1, at.current - 1); setInput(at.current < 0 ? "" : history[at.current] ?? ""); }
          }} />
        {running ? <button type="button" className="btn small ghost" onClick={() => void bridge.host.termKill(running)}>Stop</button> : null}
      </div>
    </div>
  );
}
