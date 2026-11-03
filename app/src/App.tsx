import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge, type ModelInfo, type SystemInfo } from "./bridge";
import { streamReply } from "./chat";
import { ChipIcon, MenuIcon, PlusIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Composer } from "./components/Composer";
import { MessageView } from "./components/MessageView";
import { ModelDialog } from "./components/ModelDialog";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { useEngine } from "./engine";
import { loadConversations, loadSettings, saveConversations, saveSettings, titleFrom, uid, type Conversation, type Msg, type Settings } from "./store";

const newConversation = (): Conversation => ({ id: uid(), title: "New chat", updated: Date.now(), messages: [] });

export function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [conversations, setConversations] = useState<Conversation[]>(() => { const c = loadConversations(); return c.length ? c : [newConversation()]; });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? "");
  const [system, setSystem] = useState<SystemInfo>();
  const [dialog, setDialog] = useState<"models" | "settings" | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const { state, session, load } = useEngine();

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? conversations[0] as Conversation, [conversations, activeId]);

  useEffect(() => { saveSettings(settings); }, [settings]);
  useEffect(() => { saveConversations(conversations); }, [conversations]);
  useEffect(() => {
    const dark = matchMedia("(prefers-color-scheme: dark)");
    const apply = (): void => { document.documentElement.dataset["theme"] = settings.theme === "system" ? (dark.matches ? "dark" : "light") : settings.theme; };
    apply(); dark.addEventListener("change", apply);
    return () => dark.removeEventListener("change", apply);
  }, [settings.theme]);

  // First start: learn about this machine, and load the model used last time.
  useEffect(() => {
    void bridge.system().then(setSystem);
    void bridge.listModels().then((models) => {
      const last = models.find((m) => m.path === settings.lastModel) ?? (models.length === 1 ? models[0] : undefined);
      if (last) void load(last, settings);
      else if (models.length > 1) setDialog("models");
    });
  }, []);

  useEffect(() => { const el = scroller.current; if (el && stick.current) el.scrollTop = el.scrollHeight; }, [active?.messages]);

  const update = useCallback((id: string, f: (c: Conversation) => Conversation) => setConversations((all) => all.map((c) => (c.id === id ? f(c) : c))), []);

  const pickModel = (m: ModelInfo): void => { setSettings((s) => ({ ...s, lastModel: m.path })); void load(m, settings); };

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
    await streamReply(s, history, settings, ac.signal, (u) =>
      update(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === reply.id ? { ...m, content: u.content, ...(u.reasoning ? { reasoning: u.reasoning } : {}), ...(u.stats ? { stats: u.stats } : {}), ...(u.error ? { error: u.error } : {}), ...(u.interrupted ? { interrupted: true } : {}) } : m)) })));
    abort.current = null;
    setBusy(false);
  };

  const stop = (): void => abort.current?.abort();
  const onNew = (): void => { const c = newConversation(); setConversations((all) => [c, ...all]); setActiveId(c.id); setSidebar(false); };
  const onDelete = (id: string): void => {
    setConversations((all) => { const rest = all.filter((c) => c.id !== id); return rest.length ? rest : [newConversation()]; });
    if (id === activeId) setActiveId((conversations.find((c) => c.id !== id) ?? { id: "" }).id);
  };
  useEffect(() => { if (!conversations.some((c) => c.id === activeId) && conversations[0]) setActiveId(conversations[0].id); }, [conversations, activeId]);

  const ready = state.status === "ready";
  const badge = state.status === "loading" ? "Loading model…" : ready ? `${state.model?.name} · ${state.ready?.locked ? "pinned in memory" : "Clutor engine"}` : state.status === "error" ? "Model failed to load" : "Choose a model";
  const lastAssistant = active?.messages[active.messages.length - 1];

  return (
    <div className="app">
      <Sidebar conversations={conversations} activeId={active?.id ?? ""} open={sidebar} onSelect={(id) => { setActiveId(id); setSidebar(false); }} onNew={onNew} onDelete={onDelete} onSettings={() => { setDialog("settings"); setSidebar(false); }} onClose={() => setSidebar(false)} />
      <main className="main">
        <header className="topbar">
          <button type="button" className="icon-btn menu" onClick={() => setSidebar(true)} aria-label="Open conversations"><MenuIcon /></button>
          <h1 className="title">{active?.title}</h1>
          <button type="button" className={`model-pill ${state.status}`} onClick={() => setDialog("models")}><ChipIcon /><span>{badge}</span></button>
          <button type="button" className="icon-btn" onClick={onNew} aria-label="New chat"><PlusIcon /></button>
        </header>
        <div className="scroll" ref={scroller} onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
          {active && active.messages.length > 0 ? (
            <div className="thread">
              {active.messages.map((m) => <MessageView key={m.id} msg={m} streaming={busy && m.id === lastAssistant?.id} showReasoning={settings.showReasoning} />)}
            </div>
          ) : (
            <div className="welcome">
              <div className="mark"><Logo size={64} /></div>
              <h2>{ready ? "What can I help with?" : state.status === "loading" ? "Loading your model…" : "Welcome to Clutor"}</h2>
              <p>{ready ? "Everything stays on this device." : state.status === "error" ? state.error : "Pick a model to start. Clutor runs it on your own hardware, with its own engine."}</p>
              {!ready && state.status !== "loading" ? <button type="button" className="btn" onClick={() => setDialog("models")}>Choose a model</button> : null}
            </div>
          )}
        </div>
        <Composer busy={busy} disabled={!ready} placeholder={ready ? "Message Clutor" : "Choose a model to start"} thinking={settings.thinking} onThinking={(v) => setSettings((s) => ({ ...s, thinking: v }))} onSend={(t) => void send(t)} onStop={stop} />
      </main>
      {dialog === "models" ? <ModelDialog system={system} current={state.model?.path} onPick={pickModel} onClose={() => setDialog(null)} /> : null}
      {dialog === "settings" ? <SettingsDialog settings={settings} onChange={setSettings} onClose={() => setDialog(null)} /> : null}
    </div>
  );
}
