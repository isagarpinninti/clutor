import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge, type ModelInfo, type SystemInfo } from "./bridge";
import { streamReply, toChatMessages } from "./chat";
import { ChipIcon, MenuIcon, PlusIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Spark, Thinking } from "./components/Thinking";
import { Composer } from "./components/Composer";
import { JobsDialog } from "./components/JobsDialog";
import type { JobAction } from "./components/JobStrip";
import { MessageView } from "./components/MessageView";
import { CodeView } from "./components/CodeView";
import { McpDialog } from "./components/McpDialog";
import { loadServers, McpManager, saveServers, type McpServerConfig } from "./agent/mcp";
import { ModelDialog } from "./components/ModelDialog";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { useEngine } from "./engine";
import { isActive, JobQueue, localJobStore, uid as jobUid, type Job } from "./jobs";
import { loadConversations, loadSettings, saveConversations, saveSettings, titleFrom, uid, type Conversation, type Msg, type Settings } from "./store";

const newConversation = (): Conversation => ({ id: uid(), title: "New chat", updated: Date.now(), messages: [] });

export function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [conversations, setConversations] = useState<Conversation[]>(() => { const c = loadConversations(); return c.length ? c : [newConversation()]; });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? "");
  const [system, setSystem] = useState<SystemInfo>();
  const [dialog, setDialog] = useState<"models" | "settings" | "jobs" | "mcp" | null>(null);
  const [lastModel, setLastModel] = useState<ModelInfo>();
  const [mode, setMode] = useState<"chat" | "code">(() => { try { return localStorage.getItem("clutor.mode") === "code" ? "code" : "chat"; } catch { return "chat"; } });
  const [servers, setServers] = useState<McpServerConfig[]>(loadServers);
  const [, bump] = useState(0);
  const mcp = useMemo(() => new McpManager(bridge.host, () => bump((n) => n + 1)), []);
  useEffect(() => { saveServers(servers); mcp.sync(servers); }, [servers, mcp]);
  useEffect(() => { try { localStorage.setItem("clutor.mode", mode); } catch { /* storage unavailable */ } }, [mode]);
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
    void bridge.listModels().then((models) => {
      const last = models.find((m) => m.path === settings.lastModel) ?? (models.length === 1 ? models[0] : undefined);
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

  const pickModel = (m: ModelInfo): void => { setSettings((s) => ({ ...s, lastModel: m.path })); void load(m, settings, system?.ramBytes); };

  const send = async (text: string): Promise<void> => {
    const s = session.current;
    if (!s || !active) return;
    const convId = active.id;
    const user: Msg = { id: uid(), role: "user", content: text };
    const reply: Msg = { id: uid(), role: "assistant", content: "" };
    const history = [...active.messages, user];
    update(convId, (c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), updated: Date.now(), messages: [...history, reply] }));
    stick.current = true;
    const ac = new AbortController();
    abort.current = ac;
    setBusy(true);
    // A background job on the engine steps aside for this message, and carries on afterwards.
    const jobWasRunning = queue.running !== undefined;
    const release = await queue.holdForChat();
    if (jobWasRunning) flash("A background job paused while you chat; it carries on afterwards.");
    await streamReply(s, history, settings, ac.signal, (u) =>
      update(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === reply.id ? { ...m, content: u.content, ...(u.reasoning ? { reasoning: u.reasoning } : {}), ...(u.stats ? { stats: u.stats } : {}), ...(u.error ? { error: u.error } : {}), ...(u.interrupted ? { interrupted: true } : {}) } : m)) })));
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
  const badge = state.status === "loading" ? "Loading model…" : ready ? `${state.model?.name} · ${state.ready?.locked ? "pinned in memory" : state.unpinned ? "not pinned: large model" : "Clutor engine"}` : state.status === "error" ? "Model failed to load" : "Choose a model";
  const lastAssistant = active?.messages[active.messages.length - 1];

  return (
    <div className="app">
      <Sidebar conversations={conversations} activeId={active?.id ?? ""} open={sidebar} onSelect={(id) => { setActiveId(id); setSidebar(false); }} onNew={onNew} onDelete={onDelete} onSettings={() => { setDialog("settings"); setSidebar(false); }} onJobs={() => { setDialog("jobs"); setSidebar(false); }} activeJobs={activeJobs} onClose={() => setSidebar(false)} />
      <main className="main">
        <header className="topbar">
          <button type="button" className="icon-btn menu" onClick={() => setSidebar(true)} aria-label="Open conversations"><MenuIcon /></button>
          <div className="seg" role="tablist" aria-label="Mode">
            <button type="button" role="tab" aria-selected={mode === "chat"} className={mode === "chat" ? "on" : ""} onClick={() => setMode("chat")}>Chat</button>
            <button type="button" role="tab" aria-selected={mode === "code"} className={mode === "code" ? "on" : ""} onClick={() => setMode("code")}>Code</button>
          </div>
          <h1 className="title">{mode === "chat" ? active?.title : ""}</h1>
          <button type="button" className={`model-pill ${state.status}`} onClick={() => setDialog("models")}>{state.status === "loading" ? <Spark size={16} /> : <ChipIcon />}<span>{badge}</span></button>
          <button type="button" className="icon-btn" onClick={onNew} aria-label="New chat"><PlusIcon /></button>
        </header>
        {mode === "code" ? <CodeView ready={ready} modelName={state.model?.name ?? "model"} session={() => session.current} mcp={mcp} hold={() => queue.holdForChat()} onChooseModel={() => setDialog("models")} onMcp={() => setDialog("mcp")} /> : <>
        <div className="scroll" ref={scroller} onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
          {active && active.messages.length > 0 ? (
            <div className="thread">
              {active.messages.map((m) => <MessageView key={m.id} msg={m} streaming={busy && m.id === lastAssistant?.id && !m.jobId} showReasoning={settings.showReasoning} job={jobByMessage.get(m.id)} position={m.jobId ? queue.position(m.jobId) : 0} onJobAction={jobAction} />)}
            </div>
          ) : (
            <div className="welcome">
              <div className="mark"><Logo size={64} /></div>
              {state.status === "loading" ? <div className="loading-wait"><Thinking label="Loading your model" /></div> : <h2>{ready ? "What can I help with?" : "Welcome to Clutor"}</h2>}
              <p>{state.status === "loading" ? "Mapping the weights into memory. A large model can take a while the first time." : ready ? "Everything stays on this device." : state.status === "error" ? state.error : "Pick a model to start. Clutor runs it on your own hardware, with its own engine."}</p>
              {!ready && state.status !== "loading" && lastModel ? <button type="button" className="btn" onClick={() => pickModel(lastModel)}>Load {lastModel.name}</button> : null}
              {!ready && state.status !== "loading" ? <button type="button" className={lastModel ? "btn ghost" : "btn"} onClick={() => setDialog("models")}>Choose a model</button> : null}
            </div>
          )}
        </div>
        <Composer busy={busy} disabled={!ready} placeholder={ready ? "Message Clutor" : "Choose a model to start"} thinking={settings.thinking} onThinking={(v) => setSettings((s) => ({ ...s, thinking: v }))} onSend={(t) => void send(t)} onBackground={sendBackground} notice={notice} onStop={stop} />
        </>}
      </main>
      {dialog === "models" ? <ModelDialog system={system} current={state.model?.path} onPick={pickModel} onClose={() => setDialog(null)} /> : null}
      {dialog === "jobs" ? <JobsDialog jobs={jobList} conversations={conversations} position={(id) => queue.position(id)} onAction={jobAction} onOpen={(id) => setActiveId(id)} onRemove={(id) => queue.remove(id)} onClose={() => setDialog(null)} /> : null}
      {dialog === "mcp" ? <McpDialog servers={servers} manager={mcp} onChange={setServers} onClose={() => setDialog(null)} /> : null}
      {dialog === "settings" ? <SettingsDialog settings={settings} onChange={setSettings} onClose={() => setDialog(null)} /> : null}
    </div>
  );
}
