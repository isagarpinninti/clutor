import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { bridge } from "../bridge";
import type { AgentController, Approval, Permission } from "../agent/controller";
import { folderName, groupBlocks, lastTurn, reviewable, type CodeSession } from "../agent/sessions";
import { Markdown } from "./Markdown";
import { ModePicker, type ModeOption } from "./ModePicker";
import { PlusMenu } from "./PlusMenu";
import type { OverviewPane } from "./OverviewPane";
import { SidePanel, type PanelTab } from "./SidePanel";
import { Alert } from "./Alert";
import { WorkingLine } from "./ProgressPane";
import { parseHooks } from "@core/agent/loop";
import { ToolGroup } from "./ToolGroup";
import { Logo } from "./Logo";
import { fmtElapsed } from "./RunProgress";
import { RewindIcon, CheckIcon, CopyIcon, EditIcon, FolderIcon, RedoIcon, SendIcon, StopIcon } from "./icons";

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
  const [queue, setQueue] = useState<string[]>([]);
  const [info, setInfo] = useState<string>();
  const [slashAt, setSlashAt] = useState(0);
  const histPos = useRef(-1);
  const withChanges = active ? [...active.turns].reverse().find((t) => t.diffs?.length) : undefined;

  // A task that starts opens the panel on its progress; when it ends with changes to review, the panel moves on to them (unless you have gone elsewhere in it).
  const runningId = last?.status === "running" ? last.id : undefined;
  useEffect(() => { if (runningId) { onTab("overview"); onPanel(true); } }, [runningId]);
  const prevStatus = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (prevStatus.current === "running" && last && last.status !== "running" && last.diffs?.length && tab === "overview") onTab("changes");
    prevStatus.current = last?.status;
  }, [last?.status, last?.id]);
  // Follow the task: a new task, each step, each plan tick and each approval scroll into view, so pressing Enter always shows that something started.
  const planDone = last?.plan?.filter((p) => p.state === "done").length ?? 0;
  useEffect(() => { const t = requestAnimationFrame(() => end.current?.scrollIntoView({ block: "end", behavior: "smooth" })); return () => cancelAnimationFrame(t); }, [active?.turns.length, last?.id, last?.blocks.length, planDone, last?.status, approvals.length, active?.id]);
  useEffect(() => { setError(undefined); }, [active?.id]);

  const openFile = useCallback((p: string): void => { setFocusFile(p); onTab("files"); onPanel(true); }, [onTab, onPanel]);
  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) onCreate(p); };

  const pending = approvals.filter((a) => a.sessionId === active?.id);
  const approval = pending[0];

  const [files, setFiles] = useState<string[]>([]);
  useEffect(() => { setFiles([]); if (active?.root) void bridge.host.list(active.root).then(setFiles).catch(() => setFiles([])); }, [active?.root, active?.turns.length]);
  /** Files named with @ in the task, read so the agent has them in front of it (each at most 6,000 characters, at most 5 files). */
  const mentioned = async (text: string): Promise<string> => {
    if (!active) return "";
    const names = [...new Set([...text.matchAll(/(?:^|\s)@([\w./@-]+)/g)].map((m) => m[1] as string))].filter((n) => files.includes(n)).slice(0, 5);
    const parts: string[] = [];
    for (const n of names) { const c = await bridge.host.read(active.root, n).catch(() => null); if (c !== null) parts.push(`Contents of ${n}${c.length > 6000 ? " (first 6,000 characters)" : ""}:\n${c.slice(0, 6000)}`); }
    return parts.join("\n\n");
  };
  const exec = (text: string): void => {
    if (!active) return;
    setError(undefined);
    void mentioned(text).then((extra) => ctl.run(active.id, text, { perm, source: "local", ...(extra ? { extra } : {}) })).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  };
  const at = /(?:^|\s)@([\w./@-]*)$/.exec(task);
  const atMatches = at ? files.filter((f) => f.toLowerCase().includes((at[1] ?? "").toLowerCase())).slice(0, 8) : [];
  const pickFile = (f: string): void => { setTask((t) => t.replace(/@([\w./@-]*)$/, `@${f} `)); setSlashAt(0); };
  const rewind = async (id: string): Promise<void> => {
    if (!active) return;
    const r = await ctl.rewind(active.id, id);
    if (r.ok) { setTask(r.text); setInfo("Rewound: the folder is back to how it was before that task, and the task is in the box so you can change it and run it again."); } else setError(r.error);
  };
  const contextUsed = [...(active?.turns ?? [])].reverse().find((t) => t.context)?.context ?? 0;
  const contextMax = overviewProps.contextMax;
  const COMMANDS: Array<{ name: string; hint: string; run: () => void }> = active ? [
    { name: "help", hint: "What you can type here", run: () => setInfo("Type a task and press Enter. While it works you can queue another message (it runs next) or press Esc to stop. Commands: " + "/init, /clear, /compact, /undo, /rewind, /permissions, /hooks, /plan, /ask, /auto, /context, /diff, /model, /tools, /help. Up arrow brings back your earlier tasks. Put instructions for the agent in CLUTOR.md (or AGENTS.md / CLAUDE.md) in the folder.") },
    { name: "init", hint: "Write CLUTOR.md: instructions for this project", run: () => exec("Look at this project (its files, how it is built and run) and write a CLUTOR.md in the folder for future tasks: what the project is, how to run and test it, the folder layout, and the conventions to follow. Keep it under 60 lines and only say what you saw in the files.") },
    { name: "clear", hint: "Start a new session in this folder", run: () => { onCreate(active.root); } },
    { name: "compact", hint: "Make later tasks forget the earlier ones (the files stay)", run: () => { ctl.forget(active.id); setInfo("Compacted: later tasks no longer remember the earlier ones in this session. The files in the folder stay as they are."); } },
    { name: "undo", hint: "Undo the last task's changes in the folder", run: () => { if (withChanges?.applied) void undo(); else setInfo("There is nothing to undo."); } },
    { name: "plan", hint: "Plan only: read the project, change nothing", run: () => { onPerm("plan"); setInfo("Plan only: the next tasks read the project and write a plan, and change nothing."); } },
    { name: "ask", hint: "Ask before running anything", run: () => { onPerm("ask"); setInfo("The agent asks before running a command."); } },
    { name: "auto", hint: "Run checks without asking", run: () => { onPerm("auto"); setInfo("Tests and checks run without asking; the network and tools still ask."); } },
    { name: "rewind", hint: "Go back to before the last task (folder and session)", run: () => { const t = [...(active.turns)].reverse().find((x) => x.status !== "running"); if (t) void rewind(t.id); else setInfo("There is nothing to rewind."); } },
    { name: "context", hint: "How full the model's context is", run: () => setInfo(contextUsed ? `${contextUsed.toLocaleString()} of ${contextMax.toLocaleString()} tokens used (${Math.round((contextUsed / Math.max(1, contextMax)) * 100)}%) after the latest step.` : "No task has run yet in this session.") },
    { name: "permissions", hint: "Commands always allowed in this folder (add /permissions clear to remove them)", run: () => { const r = ctl.rules(active.id); setInfo(r.length ? `Always allowed here: ${r.join(" · ")}. Type /revoke to remove them.` : "No command is always allowed in this folder. Choose \"Always allow this command here\" when asked."); } },
    { name: "revoke", hint: "Remove every always-allowed command in this folder", run: () => { ctl.clearRules(active.id); setInfo("Removed. Commands ask again."); } },
    { name: "hooks", hint: "Commands run around the agent's steps (.clutor/hooks.json)", run: () => { void bridge.host.read(active.root, ".clutor/hooks.json").then((t) => { const h = parseHooks(t); setInfo(h ? `Hooks loaded: ${h.pre?.length ?? 0} before a step, ${h.post?.length ?? 0} after a step, ${h.stop?.length ?? 0} when done. They run in the scratch copy with the same allowed programs as commands, without asking.` : 'No hooks. Create .clutor/hooks.json like {"hooks": {"PostToolUse": [{"matcher": "write_file|edit_file", "command": "npm test"}], "Stop": [{"command": "npm test"}]}}'); }); } },
    { name: "diff", hint: "Open the changes", run: () => { onTab("changes"); onPanel(true); } },
    { name: "model", hint: "Choose a model", run: onChooseModel },
    { name: "tools", hint: "Connect tools (MCP servers)", run: onMcp },
  ] : [];
  const slash = task.startsWith("/") && !task.includes(" ") ? COMMANDS.filter((c) => c.name.startsWith(task.slice(1).toLowerCase())) : [];
  const runCommand = (c: { run: () => void }): void => { setTask(""); setInfo(undefined); c.run(); };
  const start = (): void => {
    const text = task.trim();
    if (!active || !text) return;
    if (text.startsWith("/")) {
      const c = COMMANDS.find((x) => x.name === text.slice(1).split(/\s+/)[0]?.toLowerCase());
      if (c) { runCommand(c); return; }
      setInfo(`There is no command ${text.split(/\s+/)[0]}. Type /help to see them.`); return;
    }
    setTask(""); setInfo(undefined); histPos.current = -1;
    if (running) { setQueue((q) => [...q, text]); return; }   // typed while it works: it runs next
    exec(text);
  };
  // The next queued message starts when the running task ends (not after a stop: a stop means the person wants control back).
  useEffect(() => {
    if (!running && queue.length && last && last.status !== "stopped" && last.status !== "failed") { const [next, ...rest] = queue; setQueue(rest); if (next) exec(next); }
    if (last?.status === "stopped" && queue.length) setQueue([]);
  }, [running, last?.status]);
  useEffect(() => { setQueue([]); setInfo(undefined); }, [active?.id]);
  useEffect(() => {
    if (!running || !active) return;
    const on = (e: KeyboardEvent): void => { if (e.key === "Escape" && !editing) { ctl.stop(active.id); } };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [running, active?.id, editing]);
  const recall = (dir: -1 | 1): void => {
    const tasks = (active?.turns ?? []).map((t) => t.task).reverse();
    if (!tasks.length) return;
    const n = Math.max(-1, Math.min(tasks.length - 1, histPos.current + (dir === -1 ? 1 : -1)));
    histPos.current = n;
    setTask(n < 0 ? "" : (tasks[n] as string));
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
                        <button type="button" className="icon-btn tip" data-tip="Rewind to before this" disabled={running} onClick={() => void rewind(t.id)} aria-label="Rewind the folder and session to before this task"><RewindIcon /></button>
                      </div>
                    </>
                  )}
                </div>
                <div className="turn-body">
                  {groupBlocks(t.blocks).map((b) => b.kind === "text" ? <p key={b.id} className="turn-text">{b.text}</p>
                    : b.kind === "note" ? (b.bad ? <Alert key={b.id} tone="error" title="A step went wrong">{b.text}</Alert> : <p key={b.id} className="turn-hint"><RedoSmall />{b.text}{b.count && b.count > 1 ? <i>×{b.count}</i> : null}</p>)
                    : b.kind === "group" ? <ToolGroup key={b.id} tools={b.tools} verbose={verbose} onOpenFile={openFile} />
                    : null)}
                  {t.status === "running" && !approval ? <WorkingLine turn={t} onOpen={() => { onTab("overview"); onPanel(true); }} /> : null}
                  {t.ended && t.status !== "running" && t.blocks.length > 0 ? <p className="worked"><Logo size={14} />{t.status === "stopped" ? "Stopped after" : "Worked for"} {fmtElapsed(Math.max(1, Math.round((t.ended - t.created) / 1000)))}{t.usage ? ` · ${t.usage.steps} step${t.usage.steps === 1 ? "" : "s"} · ${t.usage.tokens.toLocaleString()} tokens` : ""}</p> : null}
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
                <div className="actions"><button type="button" className="btn small" onClick={() => ctl.answer(approval.id, true, { source: "local" })}>Allow once</button>{!approval.network && approval.source === "local" ? <><button type="button" className="btn small ghost" onClick={() => { ctl.allowSession(approval.sessionId); ctl.answer(approval.id, true, { source: "local" }); }}>Allow for this session</button><button type="button" className="btn small ghost" onClick={() => { ctl.allowAlways(approval.sessionId, approval.command); ctl.answer(approval.id, true, { source: "local" }); }}>Always allow this command here</button></> : null}<button type="button" className="btn small ghost" onClick={() => ctl.answer(approval.id, false, { source: "local" })}>Refuse</button></div>
              </div>
            ) : null}
            {error && !panel ? <Alert tone="error" title="That did not work">{error}</Alert> : null}
            <div ref={end} />
          </div>
        </div>
        <div className="composer-wrap">
          {info ? <div className="notice info-note" role="status"><span>{info}</span><button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => setInfo(undefined)}>×</button></div> : null}
          {queue.length ? <ul className="queued" aria-label="Queued messages">{queue.map((q, i) => <li key={i}><span>Queued</span><em>{q}</em><button type="button" className="icon-btn" aria-label="Remove from the queue" onClick={() => setQueue((all) => all.filter((_, j) => j !== i))}>×</button></li>)}</ul> : null}
          {atMatches.length ? <ul className="slash-menu" role="listbox" aria-label="Files">{atMatches.map((f, i) => <li key={f} role="option" aria-selected={i === slashAt} className={i === slashAt ? "on" : ""} onMouseEnter={() => setSlashAt(i)} onMouseDown={(e) => { e.preventDefault(); pickFile(f); }}><b>@</b><span>{f}</span></li>)}</ul> : null}
          {slash.length ? <ul className="slash-menu" role="listbox" aria-label="Commands">{slash.map((c, i) => <li key={c.name} role="option" aria-selected={i === slashAt} className={i === slashAt ? "on" : ""} onMouseEnter={() => setSlashAt(i)} onMouseDown={(e) => { e.preventDefault(); runCommand(c); }}><b>/{c.name}</b><span>{c.hint}</span></li>)}</ul> : null}
          <div className={`composer${!ready ? " is-disabled" : ""}`}>
            <textarea rows={2} value={task} placeholder={!ready ? "Load a model to start" : running ? "Type to queue the next message… (Esc stops the task)" : `What should change in ${folderName(active.root)}? (/ for commands, @ for files)`} disabled={!ready} aria-label="Task"
              onChange={(e) => { setTask(e.target.value); setSlashAt(0); }}
              onKeyDown={(e) => {
                if (e.nativeEvent.isComposing) return;
                if (atMatches.length) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setSlashAt((i) => (i + 1) % atMatches.length); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setSlashAt((i) => (i - 1 + atMatches.length) % atMatches.length); return; }
                  if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) { e.preventDefault(); pickFile(atMatches[slashAt] ?? atMatches[0] as string); return; }
                  if (e.key === "Escape") { setTask((t) => t.replace(/@([\w./@-]*)$/, "")); return; }
                }
                if (slash.length) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setSlashAt((i) => (i + 1) % slash.length); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setSlashAt((i) => (i - 1 + slash.length) % slash.length); return; }
                  if (e.key === "Tab") { e.preventDefault(); setTask(`/${(slash[slashAt] ?? slash[0])?.name ?? ""}`); return; }
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const c = slash[slashAt] ?? slash[0]; if (c) runCommand(c); return; }
                  if (e.key === "Escape") { setTask(""); return; }
                }
                if (e.key === "ArrowUp" && (task === "" || histPos.current >= 0) && e.currentTarget.selectionStart === 0) { e.preventDefault(); recall(-1); return; }
                if (e.key === "ArrowDown" && histPos.current >= 0) { e.preventDefault(); recall(1); return; }
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); start(); }
              }} />
            <div className="composer-row">
              <PlusMenu items={[
                { id: "folder", label: "Open another folder", hint: "Starts a new session on it", onSelect: () => void chooseFolder() },
                { id: "tools", label: "Tools (MCP servers)", hint: "Extra tools for the agent", onSelect: onMcp },
                { id: "panel", label: "Changes and files panel", checked: panel, onSelect: () => onPanel(!panel) },
              ]} />
              <ModePicker value={perm} options={PERMISSIONS} onChange={onPerm} label="Permissions" />
              <span className="spacer" />
              {contextUsed > 0 ? <span className={`ctx-meter${contextUsed / Math.max(1, contextMax) > 0.8 ? " hot" : ""}`} title={`${contextUsed.toLocaleString()} of ${contextMax.toLocaleString()} tokens`}>{Math.round((contextUsed / Math.max(1, contextMax)) * 100)}% context</span> : null}
              {picker}
              {!ready ? <button type="button" className="btn small ghost" onClick={onChooseModel}>Choose a model</button> : null}
              {running && !task.trim() ? <button type="button" className="round stop" onClick={() => active && ctl.stop(active.id)} aria-label="Stop (Esc)"><StopIcon /></button>
                : <button type="button" className="round send" onClick={start} disabled={!ready || !task.trim()} aria-label={running ? "Queue" : "Run"}><SendIcon /></button>}
            </div>
          </div>
          <p className="fineprint">{perm === "plan" ? "Plan only: nothing will be changed." : "Enter to send · Shift+Enter for a new line · files are written to the folder when the task ends, and Undo restores it."}</p>
        </div>
      </div>
      <SidePanel host={bridge.host} root={active.root} turn={withChanges} progress={last} focusFile={focusFile} onApply={() => void apply()} onDiscard={discard} onUndo={() => void undo()} onClose={() => onPanel(false)} busy={applying || running} error={error} tab={tab} onTab={onTab} hidden={!panel} overview={{ ...overviewProps, session: active }} />
    </div>
  );
}
