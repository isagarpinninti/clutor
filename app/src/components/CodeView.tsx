import { useEffect, useRef, useState } from "react";
import type { AgentEvent, AgentResult } from "@core/agent/loop";
import type { ShadowWorkspace } from "@core/agent/workspace";
import { bridge } from "../bridge";
import { applyChanges, ConflictError, listCheckpoints, undoCheckpoint } from "../agent/apply";
import type { AgentSession } from "../agent/llm";
import type { McpManager } from "../agent/mcp";
import { runCodingTask } from "../agent/runner";
import { FolderIcon, SendIcon, StopIcon } from "./icons";
import { Spark } from "./Thinking";

interface Props {
  ready: boolean;
  modelName: string;
  session: () => AgentSession | null;
  mcp: McpManager;
  /** Takes the engine from background jobs for the length of a run; the returned function gives it back. */
  hold: () => Promise<() => void>;
  onChooseModel: () => void;
  onMcp: () => void;
}

type Step = { kind: "thought" | "call" | "result" | "invalid" | "note"; text: string; ok?: boolean };
type Phase = "idle" | "running" | "review" | "applied";

const ROOT_KEY = "clutor.project.v1";
const short = (v: unknown): string => { const s = typeof v === "string" ? v : JSON.stringify(v); return s.length > 90 ? `${s.slice(0, 89)}…` : s; };

function toStep(e: AgentEvent): Step | null {
  switch (e.type) {
    case "thought": return { kind: "thought", text: e.text };
    case "tool_call": return { kind: "call", text: `${e.name} ${Object.entries(e.args).filter(([k]) => k !== "content" && k !== "old_text" && k !== "new_text").map(([k, v]) => `${k}=${short(v)}`).join(" ")}`.trim() };
    case "tool_result": return { kind: "result", ok: e.ok, text: e.output.length > 600 ? `${e.output.slice(0, 600)}…` : e.output };
    case "invalid": return { kind: "invalid", text: e.error };
    case "verify": return { kind: "note", text: `check \`${e.command}\`: ${e.ran ? (e.ok ? "passed" : "failed") : "not run"}` };
    default: return null;
  }
}

function Patch({ patch }: { patch: string }) {
  return (
    <pre className="patch">{patch.split("\n").map((l, i) => <div key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : ""}>{l || " "}</div>)}</pre>
  );
}

export function CodeView({ ready, modelName, session, mcp, hold, onChooseModel, onMcp }: Props) {
  const [root, setRoot] = useState<string>(() => { try { return localStorage.getItem(ROOT_KEY) ?? ""; } catch { return ""; } });
  const [task, setTask] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [steps, setSteps] = useState<Step[]>([]);
  const [result, setResult] = useState<AgentResult>();
  const [error, setError] = useState<string>();
  const [approval, setApproval] = useState<{ command: string; network: boolean; answer: (yes: boolean) => void }>();
  const [checkpoint, setCheckpoint] = useState<{ id: string; title: string }>();
  const [open, setOpen] = useState<Set<string>>(new Set());
  const ws = useRef<ShadowWorkspace>();
  const abort = useRef<AbortController>();
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { try { if (root) localStorage.setItem(ROOT_KEY, root); } catch { /* storage unavailable */ } }, [root]);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [steps, phase, approval]);
  useEffect(() => {
    if (!root) return;
    void listCheckpoints(bridge.host, root).then((l) => { const last = l.find((c) => !c.undone); setCheckpoint(last ? { id: last.id, title: last.title } : undefined); }).catch(() => undefined);
  }, [root]);

  const pick = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) { setRoot(p); setPhase("idle"); setSteps([]); setResult(undefined); } };

  const start = async (): Promise<void> => {
    const s = session();
    if (!s || !root || !task.trim() || phase === "running") return;
    setPhase("running"); setSteps([]); setResult(undefined); setError(undefined);
    const ac = new AbortController();
    abort.current = ac;
    const release = await hold();
    try {
      const out = await runCodingTask({
        host: bridge.host, root, session: s, modelName, task: task.trim(), mcp, signal: ac.signal,
        approve: (info) => new Promise<boolean>((resolve) => setApproval({ ...info, answer: (yes) => { setApproval(undefined); resolve(yes); } })),
        onEvent: (e) => { const st = toStep(e); if (st) setSteps((all) => [...all, st]); },
      });
      ws.current = out.ws;
      setResult(out.result);
      setPhase(out.result.diffs.some((d) => d.patch || d.status !== "modified") ? "review" : "idle");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    } finally {
      setApproval((a) => { a?.answer(false); return undefined; });
      release();
    }
  };

  const apply = async (): Promise<void> => {
    if (!ws.current || !result) return;
    try {
      const id = await applyChanges(bridge.host, root, await ws.current.changeSet(), task.trim().slice(0, 80));
      setCheckpoint({ id, title: task.trim().slice(0, 80) });
      setPhase("applied");
      setError(undefined);
    } catch (e) {
      setError(e instanceof ConflictError ? `Nothing was written. ${e.message}. Run the task again to start from the current files.` : e instanceof Error ? e.message : String(e));
    }
  };

  const undo = async (): Promise<void> => {
    if (!checkpoint) return;
    try { await undoCheckpoint(bridge.host, root, checkpoint.id); setCheckpoint(undefined); setPhase("idle"); setError(undefined); }
    catch (e) { setError(e instanceof ConflictError ? `Not undone, because ${e.paths.join(", ")} changed after the apply.` : e instanceof Error ? e.message : String(e)); }
  };

  const running = phase === "running";
  const diffs = result?.diffs ?? [];

  return (
    <>
      <div className="scroll">
        <div className="thread code">
          <div className="project-bar">
            <button type="button" className="row" onClick={() => void pick()}><FolderIcon /><span className="row-main"><b>{root ? root.split(/[\\/]/).filter(Boolean).pop() : "Choose a project folder"}</b><small>{root || "The agent can read this folder and propose changes. Nothing is written until you apply them."}</small></span></button>
            <button type="button" className="btn small ghost" onClick={onMcp}>Tools{mcp.list().filter((c) => c.status === "ready").length ? ` · ${mcp.list().filter((c) => c.status === "ready").length}` : ""}</button>
          </div>

          {steps.length === 0 && phase === "idle" ? (
            <p className="empty-hint">Describe a change. The agent reads the project, edits a private copy, can run your tests there (asking first), and then shows you the diff to apply or discard.{!ready ? " Load a model first." : ""}</p>
          ) : null}

          <ol className="timeline">
            {steps.map((s, i) => (
              <li key={i} className={`step ${s.kind}${s.ok === false ? " bad" : ""}`}>
                {s.kind === "call" ? <code>{s.text}</code> : s.kind === "result" ? <pre>{s.text}</pre> : <span>{s.text}</span>}
              </li>
            ))}
          </ol>

          {running && !approval ? <div className="working"><Spark size={16} /><span>Working…</span></div> : null}

          {approval ? (
            <div className="approval" role="alertdialog" aria-label="Approval needed">
              <b>{approval.network ? "Allow this to run, with network access?" : "Allow this to run?"}</b>
              <code>{approval.command}</code>
              <small>It runs in a scratch copy of the project, with a clean environment and a time limit. It can still read files your user can read.</small>
              <div className="actions"><button type="button" className="btn small" onClick={() => approval.answer(true)}>Allow</button><button type="button" className="btn small ghost" onClick={() => approval.answer(false)}>Refuse</button></div>
            </div>
          ) : null}

          {result && !running ? (
            <div className={`summary ${result.status}`}>
              <b>{result.status === "finished" ? "Done" : result.status === "aborted" ? "Stopped" : result.status === "max_steps" ? "Ran out of steps" : result.status === "stuck" ? "Got stuck" : "Failed"}</b>
              <p>{result.summary}</p>
              <small>{result.steps} steps · {result.toolCalls} tool calls{result.invalidReplies ? ` · ${result.invalidReplies} unusable replies` : ""}</small>
            </div>
          ) : null}

          {diffs.length > 0 && !running ? (
            <div className="changes">
              <h3>{phase === "applied" ? "Applied" : "Proposed changes"} · {diffs.length} file{diffs.length === 1 ? "" : "s"}</h3>
              {diffs.map((d) => (
                <div key={d.path} className="filediff">
                  <button type="button" className="filehead" onClick={() => setOpen((o) => { const n = new Set(o); if (n.has(d.path)) n.delete(d.path); else n.add(d.path); return n; })} aria-expanded={open.has(d.path) || diffs.length === 1}>
                    <span className={`st ${d.status}`}>{d.status}</span><b>{d.path}</b><span className="counts"><i className="add">+{d.additions}</i> <i className="del">−{d.deletions}</i></span>
                  </button>
                  {open.has(d.path) || diffs.length === 1 ? <Patch patch={d.patch} /> : null}
                </div>
              ))}
              {phase === "review" ? <div className="actions"><button type="button" className="btn" onClick={() => void apply()}>Apply to folder</button><button type="button" className="btn ghost" onClick={() => { setPhase("idle"); setResult(undefined); }}>Discard</button></div> : null}
            </div>
          ) : null}

          {checkpoint && !running ? <div className="undo"><span>Last applied: {checkpoint.title}</span><button type="button" className="btn small ghost" onClick={() => void undo()}>Undo</button></div> : null}
          {error ? <div className="notice err" role="alert">{error}</div> : null}
          <div ref={end} />
        </div>
      </div>
      <div className="composer-wrap">
        <div className={`composer${!ready || !root ? " is-disabled" : ""}`}>
          <textarea rows={2} value={task} placeholder={!ready ? "Load a model to start" : !root ? "Choose a project folder first" : "What should change in this project?"} disabled={!ready || !root || running} aria-label="Task"
            onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void start(); } }} />
          <div className="composer-row">
            {!ready ? <button type="button" className="btn small ghost" onClick={onChooseModel}>Choose a model</button> : <span className="hint">Small models make mistakes; every change is shown to you first.</span>}
            <span className="spacer" />
            {running ? <button type="button" className="send stop" onClick={() => abort.current?.abort()} aria-label="Stop"><StopIcon /></button>
              : <button type="button" className="send" onClick={() => void start()} disabled={!ready || !root || !task.trim()} aria-label="Run"><SendIcon /></button>}
          </div>
        </div>
      </div>
    </>
  );
}
