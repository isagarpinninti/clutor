import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge, type ModelInfo, type SystemInfo } from "./bridge";
import { streamReply, toChatMessages } from "./chat";
import { DiffIcon, FilesIcon, GearIcon, GlobeIcon, SidebarIcon, TerminalIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Thinking } from "./components/Thinking";
import { Composer } from "./components/Composer";
import { JobsDialog } from "./components/JobsDialog";
import type { JobAction } from "./components/JobStrip";
import { MessageView } from "./components/MessageView";
import { CodeView } from "./components/CodeView";
import type { PanelTab } from "./components/SidePanel";
import { McpDialog } from "./components/McpDialog";
import { loadServers, McpManager, saveServers, type McpServerConfig } from "./agent/mcp";
import { ModelPicker } from "./components/ModelPicker";
import { UsageDialog } from "./components/UsageDialog";
import { gather, needsWeb, withWebContext, type Gathered } from "./web";
import { chatUsage, contextPercent } from "./usage";
import { ModelDialog } from "./components/ModelDialog";
import { SettingsDialog } from "./components/SettingsDialog";
import { folderName, loadSessions, newSession, saveSessions, type CodeSession } from "./agent/sessions";
import { Sidebar } from "./components/Sidebar";
import type { EngineSession } from "@engine/session";
import { useEngine } from "./engine";
import { isActive, JobQueue, localJobStore, uid as jobUid, type Job } from "./jobs";
import { loadConversations, loadSettings, saveConversations, saveSettings, titleFrom, uid, type Conversation, type Msg, type Settings } from "./store";

const newConversation = (): Conversation => ({ id: uid(), title: "New chat", updated: Date.now(), messages: [] });

const SUGGESTIONS = [
  { label: "Explain something", text: "Explain how a mixture-of-experts model works, in simple terms." },
  { label: "Write code", text: "Write a function that " },
  { label: "Summarise", text: "Summarise the following text in five bullet points:\n\n" },
  { label: "Look something up", text: "What are the latest developments in " },
];
const greeting = (): string => { const h = new Date().getHours(); return h < 5 ? "Burning the midnight oil?" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening"; };

const FolderMini = () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3.5 7.5c0-1.1.9-2 2-2h3.2c.5 0 1 .2 1.4.6l1 1.1c.3.3.8.5 1.3.5h5.1c1.1 0 2 .9 2 2v7.8c0 1.1-.9 2-2 2h-12c-1.1 0-2-.9-2-2z" /></svg>;

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
  const [codeSessions, setCodeSessions] = useState<CodeSession[]>(loadSessions);
  const [codeId, setCodeId] = useState<string>(() => { try { return localStorage.getItem("clutor.codeSession") ?? ""; } catch { return ""; } });
  const [panel, setPanel] = useState(false);
  const [panelTab, setPanelTab] = useState<PanelTab>("changes");
  const togglePanel = (t: PanelTab): void => { if (panel && panelTab === t) setPanel(false); else { setPanelTab(t); setPanel(true); } };
  useEffect(() => { saveSessions(codeSessions); }, [codeSessions]);
  useEffect(() => { try { localStorage.setItem("clutor.codeSession", codeId); } catch { /* storage unavailable */ } }, [codeId]);
  const codeActive = codeSessions.find((s) => s.id === codeId);
  const changeSession = useCallback((id: string, f: (s: CodeSession) => CodeSession) => setCodeSessions((all) => all.map((s) => (s.id === id ? f(s) : s))), []);
  const createSession = useCallback((root: string): CodeSession => { const s = newSession(root); setCodeSessions((all) => [s, ...all]); setCodeId(s.id); setSidebar(false); return s; }, []);
  const newCodeSession = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) createSession(p); };
  const deleteSession = (id: string): void => { setCodeSessions((all) => all.filter((s) => s.id !== id)); if (id === codeId) setCodeId(""); };
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
  const syncJob = useRef<(j: Job) => void>(() => undefined);
  const queue = useMemo(() => new JobQueue(localJobStore, {
    onUpdate: (j) => syncJob.current(j),
    onFinish: (j) => {
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

  const sendBackground = (text: string): void => {
    if (!active) return;
    const convId = active.id;
    const user: Msg = { id: uid(), role: "user", content: text };
    const jobId = jobUid();
    const reply: Msg = { id: uid(), role: "assistant", content: "", jobId };
    const history = [...active.messages, user];
    update(convId, (c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), updated: Date.now(), messages: [...history, reply] }));
    stick.current = true;
    queue.enqueue({ id: jobId, title: titleFrom(text), conversationId: convId, messageId: reply.id, messages: toChatMessages(history, settings), thinking: settings.thinking, temperature: settings.temperature, maxTokens: settings.jobTokens });
  };
  const jobAction = (id: string, a: JobAction): void => { if (a === "pause") queue.pause(id); else if (a === "resume") queue.resume(id); else if (a === "cancel") queue.cancel(id); else queue.retry(id); };

  const pickModel = (m: ModelInfo): void => { setSettings((s) => ({ ...s, model: m })); setLastModel(m); void load(m, settings, system?.ramBytes); };
  /** The engine, loading the remembered model first if nothing is loaded yet (so a restart needs no choosing, and launching stays instant). */
  const ensureModel = async (): Promise<EngineSession | null> => {
    if (session.current) return session.current;
    if (!lastModel) return null;
    return load(lastModel, settings, system?.ramBytes);
  };

  const send = async (text: string): Promise<void> => {
    if (!active || (!session.current && !lastModel)) return;
    const convId = active.id;
    const user: Msg = { id: uid(), role: "user", content: text };
    const reply: Msg = { id: uid(), role: "assistant", content: "" };
    const history = [...active.messages, user];
    update(convId, (c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), updated: Date.now(), messages: [...history, reply] }));
    stick.current = true;
    const ac = new AbortController();
    abort.current = ac;
    setBusy(true);
    if (!session.current) flash("Loading your model first…");
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
  const picker = <ModelPicker current={state.model ?? lastModel} folders={folders} status={state.status} note={note} onPick={pickModel} onMore={() => setDialog("models")} />;
  const usage = chatUsage(active?.messages ?? [], state.ready?.nCtx ?? settings.ctx);
  const pct = contextPercent(usage);
  const homeScreen = !!active && active.messages.length === 0 && (ready || !!lastModel) && state.status !== "loading";
  const [prefill, setPrefill] = useState<{ text: string; n: number }>();
  const composer = (home: boolean) => (
    <Composer busy={busy} disabled={!ready && !lastModel} placeholder={ready || lastModel ? "Message Clutor" : "Choose a model to start"} thinking={settings.thinking} onThinking={(v) => setSettings((s) => ({ ...s, thinking: v }))} browse={settings.browse} onBrowse={(v) => { setSettings((s) => ({ ...s, browse: v })); if (v) flash("Browse is on: your message goes to a search service (DuckDuckGo) and the pages are fetched from the internet."); }} onSend={(t) => void send(t)} onBackground={sendBackground} notice={notice} onStop={stop} picker={picker} home={home} prefill={prefill} />
  );
  const lastAssistant = active?.messages[active.messages.length - 1];

  return (
    <div className={`app${collapsed ? " nav-collapsed" : ""}`}>
      <Sidebar conversations={conversations} activeId={active?.id ?? ""} open={sidebar} onSelect={(id) => { setActiveId(id); setSidebar(false); }} onNew={onNew} onDelete={onDelete} onSettings={() => { setDialog("settings"); setSidebar(false); }} mode={mode} codeSessions={codeSessions} codeId={codeId} onSelectCode={(id) => { setCodeId(id); setSidebar(false); }} onNewCode={() => void newCodeSession()} onDeleteCode={deleteSession} onJobs={() => { setDialog("jobs"); setSidebar(false); }} onUsage={() => { setDialog("usage"); setSidebar(false); }} onCollapse={() => setCollapsed(true)} collapsed={collapsed} activeJobs={activeJobs} onClose={() => setSidebar(false)} />
      <main className="main">
        <header className="topbar">
          <button type="button" className={`icon-btn side-open${collapsed ? " always" : ""}`} onClick={() => { setSidebar(true); setCollapsed(false); }} aria-label="Show sidebar" title="Show sidebar"><SidebarIcon /></button>
          <div className="seg" role="tablist" aria-label="Mode">
            <button type="button" role="tab" aria-selected={mode === "chat"} className={mode === "chat" ? "on" : ""} onClick={() => setMode("chat")}>Chat</button>
            <button type="button" role="tab" aria-selected={mode === "code"} className={mode === "code" ? "on" : ""} onClick={() => setMode("code")}>Code</button>
          </div>
          <h1 className="title">{mode === "chat" ? active?.title : codeActive?.title ?? "Code"}</h1>
          {mode === "code" && codeActive ? <span className="project-chip" title={codeActive.root}><FolderMini />{folderName(codeActive.root)}</span> : null}
          {mode === "chat" && ready ? (
            <button type="button" className="ring-btn" onClick={() => setDialog("usage")} title={`Context ${pct}% full`} aria-label={`Context ${pct}% full. Open usage`}>
              <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" className="ring-bg" /><circle cx="10" cy="10" r="7.5" className="ring-fg" strokeDasharray={`${(pct / 100) * 47.1} 47.1`} transform="rotate(-90 10 10)" /></svg>
              <span>{pct}%</span>
            </button>
          ) : null}
          {mode === "code" && codeActive ? (
            <div className="tool-tabs" role="toolbar" aria-label="Panels">
              {([["terminal", "Terminal", <TerminalIcon key="t" />], ["browser", "Browser", <GlobeIcon key="b" />], ["changes", "Changes", <DiffIcon key="c" />], ["files", "Files", <FilesIcon key="f" />]] as Array<[PanelTab, string, JSX.Element]>).map(([id, label, icon]) => (
                <button type="button" key={id} className={`tool-tab${panel && panelTab === id ? " on" : ""}`} onClick={() => togglePanel(id)} aria-pressed={panel && panelTab === id} title={label}>{icon}<span>{label}</span></button>
              ))}
            </div>
          ) : null}
          <button type="button" className="icon-btn" onClick={() => setDialog("settings")} aria-label="Settings" title="Settings"><GearIcon /></button>
        </header>
        {mode === "code" ? <CodeView active={codeActive} onChange={changeSession} onCreate={createSession} ready={ready || !!lastModel} modelName={state.model?.name ?? lastModel?.name ?? "model"} engine={ensureModel} mcp={mcp} hold={() => queue.holdForChat()} onChooseModel={() => setDialog("models")} picker={picker} onMcp={() => setDialog("mcp")} onUsage={(u) => setAgentUsage((x) => ({ runs: x.runs + 1, steps: x.steps + u.steps, tokens: x.tokens + u.tokens }))} panel={panel} onPanel={setPanel} tab={panelTab} onTab={setPanelTab} perm={settings.permission} onPerm={(p) => setSettings((s) => ({ ...s, permission: p }))} maxSteps={settings.maxSteps} /> : <>
        <div className="scroll" ref={scroller} onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
          {active && active.messages.length > 0 ? (
            <div className="thread">
              {active.messages.map((m) => <MessageView key={m.id} msg={m} streaming={busy && m.id === lastAssistant?.id && !m.jobId} showReasoning={settings.showReasoning} job={jobByMessage.get(m.id)} position={m.jobId ? queue.position(m.jobId) : 0} onJobAction={jobAction} />)}
            </div>
          ) : (
            homeScreen ? (
              <div className="home">
                <div className="mark"><Logo size={48} /></div>
                <h2>{greeting()}</h2>
                {composer(true)}
                <div className="suggest">
                  {SUGGESTIONS.map((s) => <button type="button" key={s.label} className="suggestion" onClick={() => setPrefill((p) => ({ text: s.text, n: (p?.n ?? 0) + 1 }))}>{s.label}</button>)}
                </div>
              </div>
            ) : (
            <div className="welcome">
              <div className="mark"><Logo size={64} /></div>
              {state.status === "loading" ? <div className="loading-wait"><Thinking label="Loading your model" /></div> : <h2>{ready ? "What can I help with?" : "Welcome to Clutor"}</h2>}
              <p>{state.status === "loading" ? "Mapping the weights into memory. A large model can take a while the first time." : ready ? "Everything stays on this device." : state.status === "error" ? state.error : "Pick a model to start. Clutor runs it on your own hardware, with its own engine."}</p>
              {!ready && state.status !== "loading" && lastModel ? <button type="button" className="btn" onClick={() => pickModel(lastModel)}>Load {lastModel.name}</button> : null}
              {!ready && state.status !== "loading" ? <button type="button" className={lastModel ? "btn ghost" : "btn"} onClick={() => setDialog("models")}>Choose a model</button> : null}
            </div>
            )
          )}
        </div>
        {homeScreen ? null : composer(false)}
        </>}
      </main>
      {dialog === "models" ? <ModelDialog folders={folders} onFolder={(p) => setSettings((s) => ({ ...s, modelsDir: p }))} system={system} current={state.model?.path} onPick={pickModel} onClose={() => setDialog(null)} /> : null}
      {dialog === "jobs" ? <JobsDialog jobs={jobList} conversations={conversations} position={(id) => queue.position(id)} onAction={jobAction} onOpen={(id) => setActiveId(id)} onRemove={(id) => queue.remove(id)} onClose={() => setDialog(null)} /> : null}
      {dialog === "usage" ? <UsageDialog state={state} ramBytes={system?.ramBytes} usage={usage} agent={agentUsage} jobs={{ active: activeJobs, done: jobList.filter((j) => j.status === "done").length }} onClose={() => setDialog(null)} /> : null}
      {dialog === "mcp" ? <McpDialog servers={servers} manager={mcp} onChange={setServers} onClose={() => setDialog(null)} /> : null}
      {dialog === "settings" ? <SettingsDialog settings={settings} onChange={setSettings} system={system} modelName={state.model?.name ?? lastModel?.name} servers={servers} manager={mcp} onServers={setServers} onModels={() => setDialog("models")} onClose={() => setDialog(null)} /> : null}
    </div>
  );
}
