import { useEffect, useRef, useState, type ReactNode } from "react";
import { bridge } from "../bridge";
import type { AgentController, Approval, Permission } from "../agent/controller";
import { folderName, lastTurn, reviewable, type CodeSession } from "../agent/sessions";
import { Markdown } from "./Markdown";
import { ModePicker, type ModeOption } from "./ModePicker";
import { PlusMenu } from "./PlusMenu";
import type { OverviewPane } from "./OverviewPane";
import { SidePanel, type PanelTab } from "./SidePanel";
import { Alert } from "./Alert";
import { WorkingLine } from "./ProgressPane";
import { ToolBlock } from "./ToolBlock";
import { CheckIcon, CopyIcon, EditIcon, FolderIcon, RedoIcon, SendIcon, StopIcon } from "./icons";

export type { Permission };
const PERMISSIONS: ModeOption<Permission>[] = [
  { id: "ask", label: "Ask first", hint: "Asks before running anything" },
  { id: "auto", label: "Auto-run checks", hint: "Runs tests without asking; still asks for network and tools" },
  { id: "plan", label: "Plan only", hint: "Reads the project and writes a plan; changes nothing" },
];

interface Props {
  /** The open session, or none. */
  active: CodeSession | undefined;
  /** Starts a session on a folder and opens it. */
  onCreate: (root: string) => CodeSession;
  ready: boolean;
  /** Runs tasks, asks for approvals and applies changes: shared with a paired phone. */
  ctl: AgentController;
  approvals: Approval[];
  picker: ReactNode;
  onChooseModel: () => void;
  onMcp: () => void;
  panel: boolean;
  onPanel: (open: boolean) => void;
  overviewProps: Omit<React.ComponentProps<typeof OverviewPane>, "session" | "onOpenFile">;
  tab: PanelTab;
  onTab: (t: PanelTab) => void;
  perm: Permission;
  onPerm: (p: Permission) => void;
  verbose: boolean;
}

const RedoSmall = () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M19 12a7 7 0 1 1-2.2-5.1" /><path d="M19 4.5v4h-4" /></svg>;
const count = (n: number, w: string): string => `${n} ${w}${n === 1 ? "" : "s"}`;

export function CodeView({ active, onCreate, ready, ctl, approvals, picker, onChooseModel, onMcp, panel, onPanel, overviewProps, tab, onTab, perm, onPerm, verbose }: Props) {
  const [task, setTask] = useState("");
  const [error, setError] = useState<string>();
  const [focusFile, setFocusFile] = useState<string>();
  const [applying, setApplying] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [copied, setCopied] = useState<string | null>(null);
  const copyTask = (id: string, text: string): void => { void navigator.clipboard?.writeText(text).then(() => { setCopied(id); setTimeout(() => setCopied((c) => (c === id ? null : c)), 1500); }); };
  const end = useRef<HTMLDivElement>(null);
  const last = active ? lastTurn(active) : undefined;
  const running = last?.status === "running";
  const withChanges = active ? [...active.turns].reverse().find((t) => t.diffs?.length) : undefined;

  // A task that starts opens the panel on its progress; when it ends with changes to review, the panel moves on to them (unless you have gone elsewhere in it).
  const runningId = last?.status === "running" ? last.id : undefined;
  useEffect(() => { if (runningId) { onTab("progress"); onPanel(true); } }, [runningId]);
  const prevStatus = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (prevStatus.current === "running" && last && last.status !== "running" && last.diffs?.length && tab === "progress") onTab("changes");
    prevStatus.current = last?.status;
  }, [last?.status, last?.id]);
  // Follow the task: a new task, each step, each plan tick and each approval scroll into view, so pressing Enter always shows that something started.
  const planDone = last?.plan?.filter((p) => p.state === "done").length ?? 0;
  useEffect(() => { const t = requestAnimationFrame(() => end.current?.scrollIntoView({ block: "end", behavior: "smooth" })); return () => cancelAnimationFrame(t); }, [active?.turns.length, last?.id, last?.blocks.length, planDone, last?.status, approvals.length, active?.id]);
  useEffect(() => { setError(undefined); }, [active?.id]);

  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) onCreate(p); };

  const pending = approvals.filter((a) => a.sessionId === active?.id);
  const approval = pending[0];

  const start = (): void => {
    const text = task.trim();
    if (!active || !text || running) return;
    setTask(""); setError(undefined);
    void ctl.run(active.id, text, { perm, source: "local" }).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  };
  const retry = (text: string): void => { if (!active || running) return; setError(undefined); void ctl.run(active.id, text, { perm, source: "local" }).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e))); };
  const apply = async (): Promise<void> => {
    if (!active || !withChanges) return;
    setApplying(true);
    const r = await ctl.apply(active.id, withChanges.id);
    setError(r.ok ? undefined : r.error);
    setApplying(false);
  };
  const undo = async (): Promise<void> => {
    if (!active || !withChanges?.applied) return;
    const r = await ctl.undo(active.id, withChanges.id);
    setError(r.ok ? undefined : r.error);
  };
  const discard = (): void => { if (active && withChanges) ctl.discard(active.id, withChanges.id); };

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
                <div className="msg user">
                  {editing === t.id ? (
                    <div className="edit-box">
                      <textarea value={draft} autoFocus rows={Math.min(8, Math.max(2, draft.split("\n").length))} aria-label="Edit your task" onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (draft.trim()) { setEditing(null); retry(draft.trim()); } } if (e.key === "Escape") setEditing(null); }} />
                      <div className="edit-actions"><button type="button" className="btn small ghost" onClick={() => setEditing(null)}>Cancel</button><button type="button" className="btn small" disabled={!draft.trim()} onClick={() => { setEditing(null); retry(draft.trim()); }}>Save and run</button></div>
                    </div>
                  ) : (
                    <>
                      <div className="prompt">{t.task}</div>
                      <div className="msg-actions" role="toolbar" aria-label="Task actions">
                        <button type="button" className="icon-btn tip" data-tip={copied === t.id ? "Copied" : "Copy"} onClick={() => copyTask(t.id, t.task)} aria-label="Copy the task">{copied === t.id ? <CheckIcon /> : <CopyIcon />}</button>
                        <button type="button" className="icon-btn tip" data-tip="Edit" disabled={running || !ready} onClick={() => { setDraft(t.task); setEditing(t.id); }} aria-label="Edit the task"><EditIcon /></button>
                        <button type="button" className="icon-btn tip" data-tip="Redo" disabled={running || !ready} onClick={() => retry(t.task)} aria-label="Run this task again"><RedoIcon /></button>
                      </div>
                    </>
                  )}
                </div>
                <div className="turn-body">
                  {t.blocks.map((b) => b.kind === "text" ? <p key={b.id} className="turn-text">{b.text}</p>
                    : b.kind === "note" ? (b.bad ? <Alert key={b.id} tone="error" title="A step went wrong">{b.text}</Alert> : <p key={b.id} className="turn-hint"><RedoSmall />{b.text}{b.count && b.count > 1 ? <i>×{b.count}</i> : null}</p>)
                    : <ToolBlock key={b.id} block={b} verbose={verbose} onOpenFile={(p) => { setFocusFile(p); onTab("files"); onPanel(true); }} />)}
                  {t.status === "running" && !approval ? <WorkingLine turn={t} onOpen={() => { onTab("progress"); onPanel(true); }} /> : null}
                  {t.summary ? <div className="turn-summary"><Markdown text={t.summary} /></div> : null}
                  {t.status === "done" && !t.planOnly && !t.diffs?.length && !t.lost && !t.blocks.some((b) => b.kind === "tool" && b.name === "run_command") ? <Alert tone="warn" title="Nothing was changed" action={ready && !running ? { label: "Try again", onClick: () => retry(t.task) } : undefined}>No files were created or edited. If you asked for a change, the model did not make it: try again, or choose a more capable model.</Alert> : null}
                  {t.status === "interrupted" ? <Alert tone="warn" title="This task was interrupted" action={ready && !running ? { label: "Try again", onClick: () => retry(t.task) } : undefined}>The app was closed while it ran.</Alert> : null}
                  {t.status === "stopped" ? <p className="turn-note">Stopped.</p> : null}
                  {t.status === "failed" ? <Alert tone="error" title="The task could not finish" action={ready && !running ? { label: "Try again", onClick: () => retry(t.task) } : undefined}>{t.error ?? "The task failed."}</Alert> : null}
                  {t.diffs?.length ? (
                    <button type="button" className="changes-card" onClick={() => { onTab("changes"); onPanel(true); }}>
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
                <b>{approval.network ? "Allow this, with network access?" : "Allow this to run?"}{approval.source === "remote" ? <span className="tag">asked from {approval.device ?? "your phone"}</span> : null}</b>
                <code>{approval.command}</code>
                <small>Commands run in a scratch copy of the project, with a clean environment and a time limit. They can still read files your user can read.</small>
                <div className="actions"><button type="button" className="btn small" onClick={() => ctl.answer(approval.id, true, { source: "local" })}>Allow</button><button type="button" className="btn small ghost" onClick={() => ctl.answer(approval.id, false, { source: "local" })}>Refuse</button></div>
              </div>
            ) : null}
            {error && !panel ? <Alert tone="error" title="That did not work">{error}</Alert> : null}
            <div ref={end} />
          </div>
        </div>
        <div className="composer-wrap">
          <div className={`composer${!ready ? " is-disabled" : ""}`}>
            <textarea rows={2} value={task} placeholder={!ready ? "Load a model to start" : `What should change in ${folderName(active.root)}?`} disabled={!ready || running} aria-label="Task"
              onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); start(); } }} />
            <div className="composer-row">
              <PlusMenu items={[
                { id: "folder", label: "Open another folder", hint: "Starts a new session on it", onSelect: () => void chooseFolder() },
                { id: "tools", label: "Tools (MCP servers)", hint: "Extra tools for the agent", onSelect: onMcp },
                { id: "panel", label: "Changes and files panel", checked: panel, onSelect: () => onPanel(!panel) },
              ]} />
              <ModePicker value={perm} options={PERMISSIONS} onChange={onPerm} label="Permissions" />
              <span className="spacer" />
              {picker}
              {!ready ? <button type="button" className="btn small ghost" onClick={onChooseModel}>Choose a model</button> : null}
              {running ? <button type="button" className="round stop" onClick={() => active && ctl.stop(active.id)} aria-label="Stop"><StopIcon /></button>
                : <button type="button" className="round send" onClick={start} disabled={!ready || !task.trim()} aria-label="Run"><SendIcon /></button>}
            </div>
          </div>
          <p className="fineprint">{perm === "plan" ? "Plan only: nothing will be changed." : "Every change is shown to you before it is written."}</p>
        </div>
      </div>
      <SidePanel host={bridge.host} root={active.root} turn={withChanges} progress={last} focusFile={focusFile} onApply={() => void apply()} onDiscard={discard} onUndo={() => void undo()} onClose={() => onPanel(false)} busy={applying || running} error={error} tab={tab} onTab={onTab} hidden={!panel} overview={{ ...overviewProps, session: active }} />
    </div>
  );
}
