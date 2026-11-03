import { useEffect, useRef, useState, type ReactNode } from "react";
import { bridge } from "../bridge";
import { applyChanges, ConflictError, undoCheckpoint } from "../agent/apply";
import type { AgentSession } from "../agent/llm";
import type { McpManager } from "../agent/mcp";
import { runCodingTask } from "../agent/runner";
import { applyEvent, folderName, lastTurn, reviewable, type CodeSession, type Turn } from "../agent/sessions";
import { titleFrom, uid } from "../store";
import { Markdown } from "./Markdown";
import { ModePicker, type ModeOption } from "./ModePicker";
import { PlusMenu } from "./PlusMenu";
import { SidePanel } from "./SidePanel";
import { Spark } from "./Thinking";
import { ToolBlock } from "./ToolBlock";
import { FolderIcon, SendIcon, StopIcon } from "./icons";

export type Permission = "ask" | "auto" | "plan";
const PERMISSIONS: ModeOption<Permission>[] = [
  { id: "ask", label: "Ask first", hint: "Asks before running anything" },
  { id: "auto", label: "Auto-run checks", hint: "Runs tests without asking; still asks for network and tools" },
  { id: "plan", label: "Plan only", hint: "Reads the project and writes a plan; changes nothing" },
];

interface Props {
  /** The open session, or none. */
  active: CodeSession | undefined;
  onChange: (id: string, f: (s: CodeSession) => CodeSession) => void;
  /** Starts a session on a folder and opens it. */
  onCreate: (root: string) => CodeSession;
  ready: boolean;
  modelName: string;
  engine: () => AgentSession | null;
  mcp: McpManager;
  /** Takes the engine from background jobs for the length of a run; the returned function gives it back. */
  hold: () => Promise<() => void>;
  picker: ReactNode;
  onChooseModel: () => void;
  onMcp: () => void;
  onUsage: (u: { steps: number; tokens: number }) => void;
  panel: boolean;
  onPanel: (open: boolean) => void;
}

const count = (n: number, w: string): string => `${n} ${w}${n === 1 ? "" : "s"}`;

export function CodeView({ active, onChange, onCreate, ready, modelName, engine, mcp, hold, picker, onChooseModel, onMcp, onUsage, panel, onPanel }: Props) {
  const [task, setTask] = useState("");
  const [perm, setPerm] = useState<Permission>(() => { try { const p = localStorage.getItem("clutor.perm"); return p === "auto" || p === "plan" ? p : "ask"; } catch { return "ask"; } });
  useEffect(() => { try { localStorage.setItem("clutor.perm", perm); } catch { /* storage unavailable */ } }, [perm]);
  const [error, setError] = useState<string>();
  const [approval, setApproval] = useState<{ command: string; network: boolean; answer: (yes: boolean) => void }>();
  const [focusFile, setFocusFile] = useState<string>();
  const [applying, setApplying] = useState(false);
  const abort = useRef<AbortController>();
  const end = useRef<HTMLDivElement>(null);
  const last = active ? lastTurn(active) : undefined;
  const running = last?.status === "running";
  const withChanges = active ? [...active.turns].reverse().find((t) => t.diffs?.length) : undefined;

  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [last?.blocks.length, last?.status, approval, active?.id]);
  useEffect(() => { setError(undefined); setApproval(undefined); }, [active?.id]);

  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) onCreate(p); };

  const patchTurn = (sid: string, tid: string, f: (t: Turn) => Turn): void => onChange(sid, (s) => ({ ...s, updated: Date.now(), turns: s.turns.map((t) => (t.id === tid ? f(t) : t)) }));

  const start = async (): Promise<void> => {
    const s = engine();
    const text = task.trim();
    if (!s || !active || !text || running) return;
    const sid = active.id, root = active.root;
    const turn: Turn = { id: uid(), task: text, created: Date.now(), status: "running", blocks: [] };
    onChange(sid, (x) => ({ ...x, title: x.turns.length ? x.title : titleFrom(text), updated: Date.now(), turns: [...x.turns, turn] }));
    setTask(""); setError(undefined);
    const ac = new AbortController();
    abort.current = ac;
    const release = await hold();
    try {
      const out = await runCodingTask({
        host: bridge.host, web: bridge.web, root, session: s, modelName, task: text, mcp, signal: ac.signal, planOnly: perm === "plan",
        approve: (info) => perm === "plan" ? Promise.resolve(false) : perm === "auto" && !info.network ? Promise.resolve(true) : new Promise<boolean>((resolve) => setApproval({ ...info, answer: (yes) => { setApproval(undefined); resolve(yes); } })),
        onEvent: (e) => patchTurn(sid, turn.id, (t) => ({ ...t, blocks: applyEvent(t.blocks, e) })),
      });
      const changes = perm === "plan" ? [] : await out.ws.changeSet();
      const diffs = perm === "plan" ? [] : out.result.diffs.filter((d) => d.patch || d.status !== "modified");
      onUsage({ steps: out.result.steps, tokens: out.result.usage.completionTokens });
      patchTurn(sid, turn.id, (t) => ({
        ...t, status: out.result.status === "aborted" ? "stopped" : out.result.status === "error" ? "failed" : "done", summary: out.result.summary,
        usage: { steps: out.result.steps, tokens: out.result.usage.completionTokens }, ...(out.result.error ? { error: out.result.error } : {}),
        ...(diffs.length ? { changes, diffs } : {}),
      }));
      if (diffs.length) onPanel(true);
    } catch (e) {
      patchTurn(sid, turn.id, (t) => ({ ...t, status: "failed", error: e instanceof Error ? e.message : String(e) }));
    } finally {
      setApproval((a) => { a?.answer(false); return undefined; });
      release();
    }
  };

  const apply = async (): Promise<void> => {
    if (!active || !withChanges?.changes) return;
    setApplying(true);
    try {
      const id = await applyChanges(bridge.host, active.root, withChanges.changes, withChanges.task.slice(0, 80));
      patchTurn(active.id, withChanges.id, (t) => ({ ...t, applied: id }));
      setError(undefined);
    } catch (e) {
      setError(e instanceof ConflictError ? `Nothing was written. ${e.message}. Run the task again to start from the current files.` : e instanceof Error ? e.message : String(e));
    } finally { setApplying(false); }
  };
  const undo = async (): Promise<void> => {
    if (!active || !withChanges?.applied) return;
    try { await undoCheckpoint(bridge.host, active.root, withChanges.applied); patchTurn(active.id, withChanges.id, (t) => { const { applied: _a, ...rest } = t; void _a; return { ...rest, discarded: true }; }); setError(undefined); }
    catch (e) { setError(e instanceof ConflictError ? `Not undone, because ${e.paths.join(", ")} changed after the apply.` : e instanceof Error ? e.message : String(e)); }
  };
  const discard = (): void => { if (active && withChanges) patchTurn(active.id, withChanges.id, (t) => ({ ...t, discarded: true })); };

  if (!active) {
    return (
      <div className="scroll"><div className="welcome code-welcome">
        <div className="folder-mark"><FolderIcon /></div>
        <h2>Work on a project</h2>
        <p>Pick a folder and describe a change. The agent edits a private copy, can run your tests there after asking, and shows you the diff to apply or discard.</p>
        <button type="button" className="btn" onClick={() => void chooseFolder()}>Choose a project folder</button>
      </div></div>
    );
  }

  return (
    <div className={`code-layout${panel ? " with-panel" : ""}`}>
      <div className="code-main">
        <div className="scroll">
          <div className="thread code">
            {active.turns.length === 0 ? <p className="empty-hint">{ready ? `Describe a change in ${folderName(active.root)}.` : "Load a model first, then describe a change."}</p> : null}
            {active.turns.map((t) => (
              <section key={t.id} className="turn">
                <div className="msg user"><div className="prompt">{t.task}</div></div>
                <div className="turn-body">
                  {t.blocks.map((b) => b.kind === "text" ? <p key={b.id} className="turn-text">{b.text}</p>
                    : b.kind === "note" ? <p key={b.id} className={`turn-note${b.bad ? " bad" : ""}`}>{b.text}</p>
                    : <ToolBlock key={b.id} block={b} onOpenFile={(p) => { setFocusFile(p); onPanel(true); }} />)}
                  {t.status === "running" && !approval ? <div className="working"><Spark size={14} /><span>Working…</span></div> : null}
                  {t.summary ? <div className="turn-summary"><Markdown text={t.summary} /></div> : null}
                  {t.status === "interrupted" ? <p className="turn-note bad">The app was closed while this ran.</p> : null}
                  {t.status === "stopped" ? <p className="turn-note">Stopped.</p> : null}
                  {t.status === "failed" ? <p className="turn-note bad">{t.error ?? "The task failed."}</p> : null}
                  {t.diffs?.length ? (
                    <button type="button" className="changes-card" onClick={() => onPanel(true)}>
                      <b>{count(t.diffs.length, "file")} changed</b>
                      <span className="counts"><i className="add">+{t.diffs.reduce((n, d) => n + d.additions, 0)}</i> <i className="del">−{t.diffs.reduce((n, d) => n + d.deletions, 0)}</i></span>
                      <span className="card-state">{t.applied ? "Applied" : t.discarded ? "Discarded" : reviewable(t) ? "Review" : ""}</span>
                    </button>
                  ) : t.lost ? <p className="turn-note">The changes were too large to keep after a restart. Run the task again.</p> : null}
                </div>
              </section>
            ))}
            {approval ? (
              <div className="approval" role="alertdialog" aria-label="Approval needed">
                <b>{approval.network ? "Allow this, with network access?" : "Allow this to run?"}</b>
                <code>{approval.command}</code>
                <small>Commands run in a scratch copy of the project, with a clean environment and a time limit. They can still read files your user can read.</small>
                <div className="actions"><button type="button" className="btn small" onClick={() => approval.answer(true)}>Allow</button><button type="button" className="btn small ghost" onClick={() => approval.answer(false)}>Refuse</button></div>
              </div>
            ) : null}
            {error && !panel ? <div className="notice err" role="alert">{error}</div> : null}
            <div ref={end} />
          </div>
        </div>
        <div className="composer-wrap">
          <div className={`composer${!ready ? " is-disabled" : ""}`}>
            <textarea rows={2} value={task} placeholder={!ready ? "Load a model to start" : `What should change in ${folderName(active.root)}?`} disabled={!ready || running} aria-label="Task"
              onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void start(); } }} />
            <div className="composer-row">
              <PlusMenu items={[
                { id: "folder", label: "Open another folder", hint: "Starts a new session on it", onSelect: () => void chooseFolder() },
                { id: "tools", label: "Tools (MCP servers)", hint: "Extra tools for the agent", onSelect: onMcp },
                { id: "panel", label: "Changes and files panel", checked: panel, onSelect: () => onPanel(!panel) },
              ]} />
              <ModePicker value={perm} options={PERMISSIONS} onChange={setPerm} label="Permissions" />
              <span className="spacer" />
              {picker}
              {!ready ? <button type="button" className="btn small ghost" onClick={onChooseModel}>Choose a model</button> : null}
              {running ? <button type="button" className="round stop" onClick={() => abort.current?.abort()} aria-label="Stop"><StopIcon /></button>
                : <button type="button" className="round send" onClick={() => void start()} disabled={!ready || !task.trim()} aria-label="Run"><SendIcon /></button>}
            </div>
          </div>
          <p className="fineprint">{perm === "plan" ? "Plan only: nothing will be changed." : "Every change is shown to you before it is written."}</p>
        </div>
      </div>
      {panel ? <SidePanel host={bridge.host} root={active.root} turn={withChanges} focusFile={focusFile} onApply={() => void apply()} onDiscard={discard} onUndo={() => void undo()} onClose={() => onPanel(false)} busy={applying || running} error={error} /> : null}
    </div>
  );
}
