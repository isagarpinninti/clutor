import { applyTheme } from "./theme";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge, type ModelInfo, type SystemInfo } from "./bridge";
import { streamReply, toChatMessages } from "./chat";
import { BackIcon, ChatIcon, ChevronIcon, CodeIcon, ForwardIcon, DiffIcon, FilesIcon, GlobeIcon, LaptopIcon, MoreIcon, SidebarIcon, TerminalIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Thinking } from "./components/Thinking";
import { Composer } from "./components/Composer";
import { JobsDialog } from "./components/JobsDialog";
import type { JobAction } from "./components/JobStrip";
import { MessageView } from "./components/MessageView";
import { Onboarding } from "./components/Onboarding";
import { PlusMenu } from "./components/PlusMenu";
import { CodeView } from "./components/CodeView";
import type { PanelTab } from "./components/SidePanel";
import { McpDialog } from "./components/McpDialog";
import { loadServers, McpManager, saveServers, type McpServerConfig } from "./agent/mcp";
import { ModelPicker } from "./components/ModelPicker";
import { UsageDialog } from "./components/UsageDialog";
import { fitAttachments, type Attachment } from "./attachments";
import { gather, needsWeb, withWebContext, type Gathered } from "./web";
import { chatUsage, contextPercent } from "./usage";
import { ModelDialog } from "./components/ModelDialog";
import { SettingsDialog } from "./components/SettingsDialog";
import { AgentController, type Approval } from "./agent/controller";
import { RemoteHandler } from "./remote/handler";
import { loadSessions, newSession, saveSessions, type CodeSession } from "./agent/sessions";
import { Sidebar } from "./components/Sidebar";
import type { EngineSession } from "@engine/session";
import { useEngine } from "./engine";
import { isActive, JobQueue, localJobStore, uid as jobUid, type Job } from "./jobs";
import { modelForMode, loadConversations, loadSettings, saveConversations, saveSettings, titleFrom, uid, type Conversation, type Msg, type Settings } from "./store";

const newConversation = (): Conversation => ({ id: uid(), title: "New chat", updated: Date.now(), messages: [] });

const SUGGESTIONS = [
  { label: "Explain something", hint: "How a mixture-of-experts model works", text: "Explain how a mixture-of-experts model works, in simple terms." },
  { label: "Write code", hint: "A function, a script, a fix", text: "Write a function that " },
  { label: "Summarise", hint: "Long text in five bullets", text: "Summarise the following text in five bullet points:\n\n" },
  { label: "Look something up", hint: "With the web, if Browse is on", text: "What are the latest developments in " },
];
const greeting = (name?: string): string => { const h = new Date().getHours(); const base = h < 5 ? "Burning the midnight oil" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening"; return name ? `${base}, ${name}` : h < 5 ? `${base}?` : base; };


export function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [conversations, setConversations] = useState<Conversation[]>(() => { const c = loadConversations(); return c.length ? c : [newConversation()]; });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? "");
  const [system, setSystem] = useState<SystemInfo>();
  const [dialog, setDialog] = useState<"models" | "settings" | "jobs" | "mcp" | "usage" | null>(null);
  const folders = useMemo(() => (settings.modelsDir ? [settings.modelsDir] : []), [settings.modelsDir]);
  const [lastModel, setLastModel] = useState<ModelInfo>();
  const [mode, setMode] = useState<"chat" | "code">(() => { try { return localStorage.getItem("clutor.mode") === "code" ? "code" : "chat"; } catch { return "chat"; } });
  const [servers, setServers] = useState<McpServerConfig[]>(loadServers);
  const [, bump] = useState(0);
  const mcp = useMemo(() => new McpManager(bridge.host, () => bump((n) => n + 1)), []);
  useEffect(() => { saveServers(servers); mcp.sync(servers); }, [servers, mcp]);
  useEffect(() => { try { localStorage.setItem("clutor.mode", mode); } catch { /* storage unavailable */ } }, [mode]);
  const [codeSessions, setCodeSessionsRaw] = useState<CodeSession[]>(loadSessions);
  // The sessions are also kept in a ref that is updated at once, so code that just changed a session (a running task, a phone) reads the new value, not last render's.
  const sessionsRef = useRef(codeSessions);
  const setCodeSessions = useCallback((u: CodeSession[] | ((all: CodeSession[]) => CodeSession[])): void => { const next = typeof u === "function" ? u(sessionsRef.current) : u; sessionsRef.current = next; setCodeSessionsRaw(next); }, []);
  const [codeId, setCodeId] = useState<string>(() => { try { return localStorage.getItem("clutor.codeSession") ?? ""; } catch { return ""; } });
  const [panel, setPanel] = useState(false);
  const [panelTab, setPanelTab] = useState<PanelTab>("overview");
  const [renaming, setRenaming] = useState(false);
  const [settingsAt, setSettingsAt] = useState<"usage" | "connectors">("usage");
  const togglePanel = (t: PanelTab): void => { if (panel && panelTab === t) setPanel(false); else { setPanelTab(t); setPanel(true); } };
  useEffect(() => { saveSessions(codeSessions); }, [codeSessions]);
  useEffect(() => { try { localStorage.setItem("clutor.codeSession", codeId); } catch { /* storage unavailable */ } }, [codeId]);
  const codeActive = codeSessions.find((s) => s.id === codeId);
  const lastAgentContext = [...(codeActive?.turns ?? [])].reverse().find((t) => t.context)?.context;
  const changeSession = useCallback((id: string, f: (s: CodeSession) => CodeSession) => setCodeSessions((all) => all.map((s) => (s.id === id ? f(s) : s))), []);
  /** Adds a session for a folder without opening it on this screen (a phone's task must not take over what you are looking at). */
  const addSession = useCallback((root: string): CodeSession => { const s = newSession(root); setCodeSessions((all) => [s, ...all]); return s; }, [setCodeSessions]);
  const createSession = useCallback((root: string): CodeSession => { const s = addSession(root); setCodeId(s.id); setSidebar(false); return s; }, [addSession]);
  const newCodeSession = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) createSession(p); };
  const deleteSession = (id: string): void => { setCodeSessions((all) => all.filter((s) => s.id !== id)); if (id === codeId) setCodeId(""); };
  useEffect(() => { try { (document.documentElement.style as CSSStyleDeclaration & { zoom: string }).zoom = ({ small: "0.92", medium: "1", large: "1.1" } as const)[settings.textSize]; } catch { /* unsupported */ } }, [settings.textSize]);
  useEffect(() => applyTheme(settings.theme ?? "dark"), [settings.theme]);
  const mac = bridge.native && /Mac/i.test(navigator.userAgent);
  // Back and forward move between the places you opened (a chat or a coding session), like a browser.
  const hist = useRef<{ stack: string[]; i: number; skip: boolean }>({ stack: [], i: -1, skip: false });
  const [, navBump] = useState(0);
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem("clutor.sidebar") === "off"; } catch { return false; } });
  useEffect(() => { try { localStorage.setItem("clutor.sidebar", collapsed ? "off" : "on"); } catch { /* storage unavailable */ } }, [collapsed]);
  const [agentUsage, setAgentUsage] = useState({ runs: 0, steps: 0, tokens: 0 });
  const [notice, setNotice] = useState<string>();
  const [sidebar, setSidebar] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const { state, session, load } = useEngine();

  // ---- background jobs: the queue lives for the life of the app; its changes are written into the conversation message each job answers ----
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const syncJob = useRef<(j: Job) => void>(() => undefined);
  const queue = useMemo(() => new JobQueue(localJobStore, {
    onUpdate: (j) => syncJob.current(j),
    onFinish: (j) => {
      if (!settingsRef.current.notifyJobs) return;
      if (j.status === "done" && (!document.hasFocus() || j.elapsedMs > 20_000)) void bridge.notify("Your answer is ready", j.title);
      else if (j.status === "failed") void bridge.notify("A background job failed", `${j.title}: ${j.error ?? "unknown error"}`);
    },
  }), []);
  const [jobList, setJobList] = useState<Job[]>(() => queue.list());
  useEffect(() => queue.subscribe(() => setJobList(queue.list())), [queue]);
  const jobByMessage = useMemo(() => new Map(jobList.map((j) => [j.messageId, j])), [jobList]);
  const activeJobs = jobList.filter(isActive).length;

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? conversations[0] as Conversation, [conversations, activeId]);

  useEffect(() => { saveSettings(settings); }, [settings]);
  useEffect(() => { saveConversations(conversations); }, [conversations]);

  // First start: learn about this machine, and load the model used last time.
  useEffect(() => {
    void bridge.system().then(setSystem);
    // The model is not loaded until you ask: pinning several GB at launch stalls the whole computer.
    void bridge.listModels(folders).then((models) => {
      const last = models.find((m) => m.path === settings.model?.path) ?? (models.length === 1 ? models[0] : undefined);
      if (last) setLastModel(last);
      else if (models.length > 1) setDialog("models");
    });
  }, []);

  useEffect(() => { const el = scroller.current; if (el && stick.current) el.scrollTop = el.scrollHeight; }, [active?.messages]);

  const update = useCallback((id: string, f: (c: Conversation) => Conversation) => setConversations((all) => all.map((c) => (c.id === id ? f(c) : c))), []);

  syncJob.current = (j) => update(j.conversationId, (c) => ({
    ...c,
    messages: c.messages.map((m) => (m.id !== j.messageId ? m : {
      ...m, jobId: j.id, content: j.content,
      ...(j.reasoning ? { reasoning: j.reasoning } : {}),
      ...(j.status === "done" ? { stats: { tokens: j.tokens, ms: j.elapsedMs, ...(j.tps ? { tps: j.tps } : {}) } } : {}),
      ...(j.status === "failed" && j.error ? { error: j.error } : {}),
      ...(j.status === "cancelled" ? { interrupted: true } : {}),
    })),
  }));
  // The queue runs on whichever model is loaded; it waits when there is none.
  useEffect(() => { queue.attach(state.status === "ready" ? session.current : null); }, [queue, state.status, state.model?.path, state.loadMs, session]);
  // After a restart the saved jobs bring their partial answers back into their conversations.
  useEffect(() => { for (const j of queue.list()) syncJob.current(j); }, [queue]);

  const flash = (text: string): void => { setNotice(text); setTimeout(() => setNotice((n) => (n === text ? undefined : n)), 6000); };

  const sendBackground = (text: string, attached: Attachment[] = []): void => {
    if (!active) return;
    setPrefill(undefined);
    const convId = active.id;
    const user: Msg = { id: uid(), role: "user", content: text, ...fileFields(attached) };
    const jobId = jobUid();
    const reply: Msg = { id: uid(), role: "assistant", content: "", jobId };
    const history = [...active.messages, user];
    update(convId, (c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), updated: Date.now(), messages: [...history, reply] }));
    stick.current = true;
    queue.enqueue({ id: jobId, title: titleFrom(text), conversationId: convId, messageId: reply.id, messages: toChatMessages(history, settings), thinking: settings.thinking, temperature: settings.temperature, maxTokens: settings.jobTokens });
  };
  const jobAction = (id: string, a: JobAction): void => { if (a === "pause") queue.pause(id); else if (a === "resume") queue.resume(id); else if (a === "cancel") queue.cancel(id); else queue.retry(id); };

  /** The model this mode should run: the shared one, or the one chosen for this mode when they differ. */
  const modelFor = (m: "chat" | "code"): ModelInfo | undefined => modelForMode(settings, lastModel, m);
  const wanted = modelFor(mode);
  /** Which mode a choice in the model dialog is for (the dialog can be opened from Settings for either). */
  const [pickFor, setPickFor] = useState<"chat" | "code" | null>(null);
  const pickModel = (m: ModelInfo): void => {
    const target = pickFor ?? mode;
    setPickFor(null);
    // By default one choice serves both modes; with "same model" off it only changes the mode you are in.
    if (settings.sameModel === false) setSettings((s) => ({ ...s, model: s.model ?? m, models: { ...s.models, [target]: m } }));
    else { setSettings((s) => { const { models: _m, ...rest } = s; void _m; return { ...rest, model: m }; }); setLastModel(m); }
    if (target === mode) void load(m, settings, system?.ramBytes);
  };
  /** The engine, loading the remembered model first if nothing is loaded yet (so a restart needs no choosing, and launching stays instant). */
  const ensureModel = async (): Promise<EngineSession | null> => {
    if (!wanted) return session.current;
    if (session.current && state.model?.path === wanted.path) return session.current;
    return load(wanted, settings, system?.ramBytes);
  };

  // Tasks, approvals and apply/undo live in one controller, used by this screen and by a paired phone. It reads current settings through a ref, so it is built once.
  const live = useRef({ ensure: ensureModel, maxSteps: settings.maxSteps, contextTokens: settings.ctx, modelName: "model", notify: settings.notifyAgent });
  live.current = { ensure: ensureModel, maxSteps: settings.maxSteps, contextTokens: state.ready?.nCtx ?? settings.ctx, modelName: wanted?.name ?? state.model?.name ?? "model", notify: settings.notifyAgent };
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const ctl = useMemo(() => new AgentController({
    host: bridge.host, web: bridge.web,
    getSession: (id) => sessionsRef.current.find((s) => s.id === id),
    change: (id, f) => setCodeSessions((all) => all.map((s) => (s.id === id ? f(s) : s))),
    engine: () => live.current.ensure(), hold: () => queue.holdForChat(), mcp: () => mcp,
    config: () => ({ maxSteps: live.current.maxSteps, contextTokens: live.current.contextTokens, modelName: live.current.modelName }),
    onUsage: (u) => setAgentUsage((x) => ({ runs: x.runs + 1, steps: x.steps + u.steps, tokens: x.tokens + u.tokens })),
    onFinish: (s, t) => { if (live.current.notify && !document.hasFocus()) void bridge.notify(t.status === "done" ? "Task finished" : "Task stopped", (t.summary ?? s.title).slice(0, 120)); },
    onApprovals: setApprovals,
    log: (e) => { if (e.device) void bridge.remote.log(JSON.stringify({ at: Date.now(), device: e.device, what: e.what })); },
    id: uid, titleFrom,
  }), [queue, mcp, setCodeSessions]);

  // Phones: what a paired phone sends is checked and carried out here, and each phone is kept up to date with what it is watching.
  const remoteCfg = useRef({ projectsRoot: settings.remote?.projectsRoot, allowedRoots: settings.remote?.allowedRoots ?? [], model: undefined as { name: string; loaded: boolean } | undefined });
  remoteCfg.current = { projectsRoot: settings.remote?.projectsRoot, allowedRoots: settings.remote?.allowedRoots ?? [], model: wanted ? { name: wanted.name, loaded: state.status === "ready" && state.model?.path === wanted.path } : undefined };
  const remoteHandler = useMemo(() => new RemoteHandler({
    ctl, host: bridge.host, sessions: () => sessionsRef.current, createSession: addSession,
    settings: () => ({ projectsRoot: remoteCfg.current.projectsRoot, allowedRoots: remoteCfg.current.allowedRoots }), model: () => remoteCfg.current.model,
    send: (client, text) => { void bridge.remote.send(client, text); },
    log: (device, what) => { void bridge.remote.log(JSON.stringify({ at: Date.now(), device, what })); },
  }), [ctl, addSession]);
  useEffect(() => {
    let dead = false;
    let off: Array<() => void> = [];
    void (async () => {
      const subs = [await bridge.remote.onOpen((c, d) => remoteHandler.onOpen(c, d)), await bridge.remote.onMessage((c, t) => void remoteHandler.onMessage(c, t)), await bridge.remote.onClose((c) => remoteHandler.onClose(c))];
      if (dead) subs.forEach((f) => f()); else off = subs;
    })();
    return () => { dead = true; off.forEach((f) => f()); };
  }, [remoteHandler]);
  const pushTimer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => { clearTimeout(pushTimer.current); pushTimer.current = setTimeout(() => remoteHandler.onChange(), 150); }, [codeSessions, approvals, remoteHandler]);

  /** The files as stored on a message, cut to fit the model's context. */
  const fileFields = (attached: Attachment[]): Pick<Msg, "files"> => {
    if (!attached.length) return {};
    const fit = fitAttachments(attached, state.ready?.nCtx ?? settings.ctx);
    if (fit.some((f) => f.truncated)) flash("The files are long, so they were cut to fit the model's context.");
    return { files: fit.map(({ name, bytes, text, truncated }) => ({ name, bytes, text, truncated })) };
  };

  const send = async (text: string, attached: Attachment[] = []): Promise<void> => {
    if (!active || (!session.current && !wanted)) return;
    setPrefill(undefined);
    const convId = active.id;
    const user: Msg = { id: uid(), role: "user", content: text, ...fileFields(attached) };
    const reply: Msg = { id: uid(), role: "assistant", content: "" };
    const history = [...active.messages, user];
    update(convId, (c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), updated: Date.now(), messages: [...history, reply] }));
    stick.current = true;
    const ac = new AbortController();
    abort.current = ac;
    setBusy(true);
    if (!session.current || (wanted && state.model?.path !== wanted.path)) flash(`Loading ${wanted?.name ?? "your model"} first…`);
    const s = await ensureModel();
    if (!s) { update(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === reply.id ? { ...m, error: "The model could not be loaded." } : m)) })); abort.current = null; setBusy(false); return; }
    // Browse: look the question up first (this only uses the network, so a background job keeps running meanwhile).
    let web: Gathered | undefined;
    const setWeb = (w: NonNullable<Msg["web"]>, sources?: Gathered["sources"]): void => update(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === reply.id ? { ...m, web: w, ...(sources ? { sources } : {}) } : m)) }));
    if (settings.browse && needsWeb(text)) {
      setWeb({ query: text.slice(0, 120), state: "searching" });
      try {
        web = await gather(bridge.web, text);
        if (!ac.signal.aborted) setWeb({ query: text.slice(0, 120), state: "done" }, web.sources);
      } catch (e) { setWeb({ query: text.slice(0, 120), state: "failed", error: e instanceof Error ? e.message : String(e) }); }
    }
    // A background job on the engine steps aside for this message, and carries on afterwards.
    const jobWasRunning = queue.running !== undefined;
    const release = await queue.holdForChat();
    if (jobWasRunning) flash("A background job paused while you chat; it carries on afterwards.");
    await streamReply(s, history, settings, ac.signal, (u) =>
      update(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === reply.id ? { ...m, content: u.content, ...(u.reasoning ? { reasoning: u.reasoning } : {}), ...(u.stats ? { stats: u.stats } : {}), ...(u.error ? { error: u.error } : {}), ...(u.interrupted ? { interrupted: true } : {}) } : m)) })), web ? (m) => withWebContext(m, web as Gathered) : undefined);
    abort.current = null;
    release();
    setBusy(false);
  };

  const stop = (): void => abort.current?.abort();
  const onNew = (): void => { const c = newConversation(); setConversations((all) => [c, ...all]); setActiveId(c.id); setSidebar(false); };
  const onDelete = (id: string): void => {
    for (const j of queue.list()) if (j.conversationId === id) queue.remove(j.id);
    setConversations((all) => { const rest = all.filter((c) => c.id !== id); return rest.length ? rest : [newConversation()]; });
    if (id === activeId) setActiveId((conversations.find((c) => c.id !== id) ?? { id: "" }).id);
  };
  useEffect(() => { if (!conversations.some((c) => c.id === activeId) && conversations[0]) setActiveId(conversations[0].id); }, [conversations, activeId]);

  const ready = state.status === "ready";
  const note = state.ready?.locked ? "pinned" : state.unpinned ? "not pinned" : undefined;
  const loadedIsWanted = !!wanted && state.model?.path === wanted.path;
  const picker = <ModelPicker current={wanted} folders={folders} same={settings.sameModel !== false} onSame={(v) => setSettings((s) => ({ ...s, sameModel: v, ...(v ? {} : { models: { chat: s.model, code: s.model } }) }))} status={loadedIsWanted ? state.status : state.status === "loading" ? "loading" : "idle"} note={note} onPick={pickModel} onMore={() => setDialog("models")} />;
  const usage = chatUsage(active?.messages ?? [], state.ready?.nCtx ?? settings.ctx);
  const pct = contextPercent(usage);
  const here = `${mode}|${mode === "chat" ? activeId : codeId}`;
  useEffect(() => {
    const h = hist.current;
    if (h.skip) { h.skip = false; return; }
    if (h.stack[h.i] === here) return;
    h.stack = [...h.stack.slice(0, h.i + 1), here].slice(-50);
    h.i = h.stack.length - 1;
    navBump((n) => n + 1);
  }, [here]);
  const go = (d: number): void => {
    const h = hist.current, j = h.i + d;
    if (j < 0 || j >= h.stack.length) return;
    const target = h.stack[j] as string;
    h.i = j;
    if (target !== here) h.skip = true;
    const [m, id] = target.split("|") as [string, string];
    setMode(m === "code" ? "code" : "chat");
    if (m === "code") setCodeId(id); else setActiveId(id);
    navBump((n) => n + 1);
  };
  const canBack = hist.current.i > 0, canFwd = hist.current.i < hist.current.stack.length - 1;
  const homeScreen = !!active && active.messages.length === 0;
  const haveModel = ready || !!wanted;
  const [prefill, setPrefill] = useState<{ text: string; n: number }>();
  const composer = (home: boolean) => (
    <Composer busy={busy} disabled={!ready && !wanted} placeholder={ready || wanted ? "Message Clutor" : "Choose a model to start"} thinking={settings.thinking} onThinking={(v) => setSettings((s) => ({ ...s, thinking: v }))} browse={settings.browse} onBrowse={(v) => { setSettings((s) => ({ ...s, browse: v })); if (v) flash("Browse is on: your message goes to a search service (DuckDuckGo) and the pages are fetched from the internet."); }} onSend={(t, f) => void send(t, f)} onBackground={sendBackground} notice={notice} onStop={stop} picker={picker} home={home} prefill={prefill} />
  );
  const lastAssistant = active?.messages[active.messages.length - 1];

  return (
    <div className={`app${collapsed ? " nav-collapsed" : ""}${mac ? " mac" : ""}`}>
      <header className="titlebar" data-tauri-drag-region>
        <div className="tb-left" data-tauri-drag-region>
          <button type="button" className="icon-btn tip" data-tip={collapsed ? "Show sidebar" : "Hide sidebar"} onClick={() => { if (window.matchMedia("(max-width: 820px)").matches) setSidebar((s) => !s); else setCollapsed((c) => !c); }} aria-label="Toggle sidebar"><SidebarIcon /></button>
          <button type="button" className="icon-btn tip" data-tip="Back" onClick={() => go(-1)} disabled={!canBack} aria-label="Back"><BackIcon /></button>
          <button type="button" className="icon-btn tip" data-tip="Forward" onClick={() => go(1)} disabled={!canFwd} aria-label="Forward"><ForwardIcon /></button>
          <div className="mode-icons" role="tablist" aria-label="Mode">
            <button type="button" role="tab" aria-selected={mode === "chat"} className={`mode-icon tip${mode === "chat" ? " on" : ""}`} data-tip="Chat" onClick={() => setMode("chat")}><ChatIcon /></button>
            <button type="button" role="tab" aria-selected={mode === "code"} className={`mode-icon tip${mode === "code" ? " on" : ""}`} data-tip="Code" onClick={() => setMode("code")}><CodeIcon /></button>
          </div>
        </div>
        <div className="topbar" data-tauri-drag-region>
          {renaming && (mode === "chat" ? active : codeActive) ? (
            <input className="title-input" autoFocus defaultValue={mode === "chat" ? active?.title : codeActive?.title} aria-label="Session name"
              onBlur={(e) => { const t = e.target.value.trim(); if (t) { if (mode === "chat" && active) update(active.id, (c) => ({ ...c, title: t })); else if (codeActive) changeSession(codeActive.id, (s) => ({ ...s, title: t })); } setRenaming(false); }}
              onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setRenaming(false); }} />
          ) : (
            <div className="title-wrap">
              <h1 className="title">{mode === "chat" ? active?.title : codeActive?.title ?? "Code"}</h1>
              {(mode === "chat" ? active : codeActive) ? <PlusMenu label="Session" className="icon-btn caret-btn" icon={<ChevronIcon />} items={[
                { id: "rename", label: "Rename", onSelect: () => setRenaming(true) },
                { id: "delete", label: "Delete", onSelect: () => (mode === "chat" ? (active && onDelete(active.id)) : (codeActive && deleteSession(codeActive.id))) },
              ]} /> : null}
            </div>
          )}
          <span className="spacer" />
          {mode === "chat" && ready ? (
            <button type="button" className="ring-btn tip" data-tip={`Context ${pct}% full`} onClick={() => setDialog("usage")} aria-label={`Context ${pct}% full. Open usage`}>
              <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" className="ring-bg" /><circle cx="10" cy="10" r="7.5" className="ring-fg" strokeDasharray={`${(pct / 100) * 47.1} 47.1`} transform="rotate(-90 10 10)" /></svg>
              <span>{pct}%</span>
            </button>
          ) : null}
          {mode === "code" && codeActive ? (
            <div className="tool-tabs" role="toolbar" aria-label="Panels">
              {([["overview", "Session", <LaptopIcon key="o" />], ["terminal", "Terminal", <TerminalIcon key="t" />], ["changes", "Changes", <DiffIcon key="c" />], ["browser", "Browser", <GlobeIcon key="b" />], ["files", "Files", <FilesIcon key="f" />]] as Array<[PanelTab, string, JSX.Element]>).map(([id, label, icon]) => (
                <button type="button" key={id} className={`tool-tab tip${panel && panelTab === id ? " on" : ""}`} data-tip={label} onClick={() => togglePanel(id)} aria-pressed={panel && panelTab === id} aria-label={label}>{icon}</button>
              ))}
            </div>
          ) : null}
          <PlusMenu label="More" className="icon-btn" align="right" icon={<MoreIcon />} items={[
            { id: "settings", label: "Settings", onSelect: () => { setSettingsAt("usage"); setDialog("settings"); } },
            { id: "usage", label: "Usage", onSelect: () => setDialog("usage") },
            { id: "tools", label: "Connectors", hint: "Tool servers for the agent", onSelect: () => { setSettingsAt("connectors"); setDialog("settings"); } },
          ]} />
        </div>
      </header>
      <div className="shell">
      <Sidebar userName={settings.userName} ramLabel={system ? `Local · ${Math.round(system.ramBytes / 2 ** 30)} GB` : "Local"} conversations={conversations} activeId={active?.id ?? ""} open={sidebar} onSelect={(id) => { setActiveId(id); setSidebar(false); }} onNew={onNew} onDelete={onDelete} onSettings={() => { setSettingsAt("usage"); setDialog("settings"); setSidebar(false); }} mode={mode} codeSessions={codeSessions} codeId={codeId} onSelectCode={(id) => { setCodeId(id); setSidebar(false); }} onNewCode={() => void newCodeSession()} onDeleteCode={deleteSession} onJobs={() => { setDialog("jobs"); setSidebar(false); }} onUsage={() => { setDialog("usage"); setSidebar(false); }} collapsed={collapsed} activeJobs={activeJobs} onClose={() => setSidebar(false)} />
      <main className="main">
        {mode === "code" ? <CodeView active={codeActive} onCreate={createSession} ready={ready || !!wanted} ctl={ctl} approvals={approvals} onChooseModel={() => setDialog("models")} picker={picker} onMcp={() => { setSettingsAt("connectors"); setDialog("settings"); }} overviewProps={{ system, modelName: state.model?.name ?? lastModel?.name ?? "No model", pinned: note, contextMax: state.ready?.nCtx ?? settings.ctx, permission: ({ ask: "Ask first", auto: "Auto-run checks", plan: "Plan only" } as const)[settings.permission], mcp, onManageTools: () => { setSettingsAt("connectors"); setDialog("settings"); } }} panel={panel} onPanel={setPanel} tab={panelTab} onTab={setPanelTab} perm={settings.permission} onPerm={(p) => setSettings((s) => ({ ...s, permission: p }))} verbose={settings.transcript === "verbose"} /> : <>
        <div className="scroll" ref={scroller} onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
          {active && active.messages.length > 0 ? (
            <div className="thread">
              {active.messages.map((m) => <MessageView key={m.id} msg={m} streaming={busy && m.id === lastAssistant?.id && !m.jobId} showReasoning={settings.showReasoning} job={jobByMessage.get(m.id)} position={m.jobId ? queue.position(m.jobId) : 0} onJobAction={jobAction} />)}
            </div>
          ) : (
            <div className="home">
              <div className="mark"><Logo size={64} state={state.status === "loading" || busy ? "thinking" : "idle"} /></div>
              {state.status === "loading" ? <div className="loading-wait"><Thinking label="Loading your model" orb={false} /></div> : <><h2>{greeting(settings.userName)}</h2><p className="tagline">What would you like to think through?</p></>}
              {composer(true)}
              {state.status === "error" ? <p className="home-note err">{state.error}</p> : null}
              {!haveModel && state.status !== "loading" ? (
                <div className="home-note"><p>Choose a model to start. Clutor runs it on your own hardware, with its own engine.</p><button type="button" className="btn" onClick={() => setDialog("models")}>Choose a model</button></div>
              ) : state.status === "loading" ? <p className="home-note">Mapping the weights into memory. A large model can take a while the first time.</p> : (
                <div className="suggest">
                  {settings.suggestions ? SUGGESTIONS.map((s) => <button type="button" key={s.label} className="suggestion" onClick={() => setPrefill((p) => ({ text: s.text, n: (p?.n ?? 0) + 1 }))}><b>{s.label}</b><span>{s.hint}</span></button>) : null}
                </div>
              )}
            </div>
          )}
        </div>
        {homeScreen ? null : composer(false)}
        </>}
      </main>
      </div>
      {settings.onboarded ? null : <Onboarding onDone={(name) => setSettings((s) => ({ ...s, onboarded: true, ...(name ? { userName: name } : {}) }))} />}
      {dialog === "models" ? <ModelDialog contextTokens={settings.ctx} folders={folders} onFolder={(p) => setSettings((s) => ({ ...s, modelsDir: p }))} system={system} current={state.model?.path} onPick={pickModel} onClose={() => setDialog(null)} /> : null}
      {dialog === "jobs" ? <JobsDialog jobs={jobList} conversations={conversations} position={(id) => queue.position(id)} onAction={jobAction} onOpen={(id) => setActiveId(id)} onRemove={(id) => queue.remove(id)} onClose={() => setDialog(null)} /> : null}
      {dialog === "usage" ? <UsageDialog state={state} ramBytes={system?.ramBytes} usage={usage} agent={agentUsage} {...(lastAgentContext ? { agentContext: { used: lastAgentContext, max: state.ready?.nCtx ?? settings.ctx } } : {})} jobs={{ active: activeJobs, done: jobList.filter((j) => j.status === "done").length }} onClose={() => setDialog(null)} /> : null}
      {dialog === "mcp" ? <McpDialog servers={servers} manager={mcp} onChange={setServers} onClose={() => setDialog(null)} /> : null}
      {dialog === "settings" ? <SettingsDialog start={settingsAt} settings={settings} onChange={setSettings} system={system} modelName={wanted?.name ?? state.model?.name} servers={servers} manager={mcp} onServers={setServers} usage={{ state, ramBytes: system?.ramBytes, usage, agent: agentUsage, ...(lastAgentContext ? { agentContext: { used: lastAgentContext, max: state.ready?.nCtx ?? settings.ctx } } : {}), jobs: { active: activeJobs, done: jobList.filter((j) => j.status === "done").length } }} engine={state.ready} onModels={(which) => { setPickFor(which ?? null); setDialog("models"); }} onDeleteChats={() => { for (const j of queue.list()) queue.remove(j.id); const c = newConversation(); setConversations([c]); setActiveId(c.id); }} onDeleteSessions={() => { setCodeSessions([]); setCodeId(""); }} onClose={() => setDialog(null)} /> : null}
    </div>
  );
}
