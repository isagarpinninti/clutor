import { DialogHeader } from "./DialogHeader";
import { useMemo, useState, type ReactNode } from "react";
import type { NativeReady } from "@engine/session";
import { bridge, type SystemInfo } from "../bridge";
import { Logo } from "./Logo";
import type { McpManager, McpServerConfig } from "../agent/mcp";
import { DEFAULT_SYSTEM, DEFAULT_SETTINGS, fmtBytes, type Settings } from "../store";
import { McpPanel } from "./McpDialog";
import { RemotePanel } from "./RemotePanel";
import { UsagePanel, type UsageData } from "./UsageDialog";
import { BellIcon, GearIcon, ChatIcon, ChevronIcon, ChipIcon, CodeIcon, GaugeIcon, GlobeIcon, LaptopIcon, PaletteIcon, InfoIcon, PhoneIcon, PlugIcon, SearchIcon, ShieldIcon, UserIcon } from "./icons";

export type Section = "remote" | "profile" | "usage" | "desktop" | "appearance" | "notifications" | "privacy" | "models" | "chat" | "web" | "connectors" | "code" | "desktop" | "about";

interface Props {
  settings: Settings;
  onChange: (s: Settings) => void;
  system: SystemInfo | undefined;
  modelName: string | undefined;
  servers: McpServerConfig[];
  manager: McpManager;
  onServers: (s: McpServerConfig[]) => void;
  usage: UsageData;
  engine: NativeReady | undefined;
  onModels: (which?: "chat" | "code") => void;
  onDeleteChats: () => void;
  onDeleteSessions: () => void;
  onClose: () => void;
  start?: Section;
}

const GROUPS: Array<Array<{ id: Section; label: string; icon: ReactNode; words: string }>> = [
  [
    { id: "profile", label: "Profile", icon: <UserIcon />, words: "name greeting account" },
    { id: "usage", label: "Usage", icon: <GaugeIcon />, words: "context tokens speed memory" },
  ],
  [
    { id: "appearance", label: "Appearance", icon: <PaletteIcon />, words: "theme dark text size font" },
    { id: "notifications", label: "Notifications", icon: <BellIcon />, words: "alerts finished job task" },
    { id: "privacy", label: "Privacy", icon: <ShieldIcon />, words: "delete data local history" },
  ],
  [
    { id: "models", label: "Models", icon: <ChipIcon />, words: "folder gguf context memory pin load" },
    { id: "chat", label: "Chat", icon: <ChatIcon />, words: "creativity temperature instructions thinking suggestions background" },
    { id: "web", label: "Web", icon: <GlobeIcon />, words: "browse search duckduckgo internet" },
    { id: "connectors", label: "Connectors", icon: <PlugIcon />, words: "mcp tools servers" },
    { id: "remote", label: "Remote", icon: <PhoneIcon />, words: "phone pair qr wifi control awake projects" },
    { id: "code", label: "Code", icon: <CodeIcon />, words: "agent permissions steps transcript verbose" },
  ],
  [
    { id: "desktop", label: "Desktop app", icon: <LaptopIcon />, words: "reset defaults restore" },
    { id: "about", label: "About", icon: <InfoIcon />, words: "version updated engine licence github" },
  ],
];

const CTX_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
const gb = (b: number): string => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${Math.round(b / 2 ** 20)} MB`);
/** What a context length costs: the attention cache grows with the conversation, so the figure is the most it can reach, and a model cannot go past the length it was trained for. */
function ctxHint(ctx: number, engine: NativeReady | undefined): string {
  const base = "How much the model can keep in mind. Applies the next time a model loads.";
  if (!engine?.kvBytesPerToken) return `${base} Longer windows use more memory as the conversation grows.`;
  const full = engine.kvBytesPerToken * ctx;
  const limit = engine.nCtxTrain && ctx > engine.nCtxTrain ? ` This model was trained for ${Math.round(engine.nCtxTrain / 1024)}K, so it will use that.` : "";
  return `${base} For the loaded model a full ${Math.round(ctx / 1024)}K window can take up to ${gb(full)} for its attention cache (it fills as you chat).${limit}`;
}
function Details({ rows }: { rows: Array<[string, string]> }) {
  return <dl className="details">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>;
}
function Row({ title, hint, children, tag }: { title: string; hint?: ReactNode; children?: ReactNode; tag?: string }) {
  return (
    <div className="srow">
      <div className="srow-text"><b>{title}{tag ? <span className="tag">{tag}</span> : null}</b>{hint ? <small>{hint}</small> : null}</div>
      <div className="srow-ctl">{children}</div>
    </div>
  );
}
const Sub = ({ children }: { children: ReactNode }) => <h4 className="ssub">{children}</h4>;
function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button type="button" role="switch" aria-checked={on} aria-label={label} className={`switch${on ? " on" : ""}`} onClick={() => onChange(!on)}><i /></button>;
}
function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: Array<{ id: T; label: string }>; onChange: (v: T) => void; label: string }) {
  return <div className="segmented" role="radiogroup" aria-label={label}>{options.map((o) => <button type="button" role="radio" aria-checked={o.id === value} key={o.id} className={o.id === value ? "on" : ""} onClick={() => onChange(o.id)}>{o.label}</button>)}</div>;
}
function Select<T extends string | number>({ value, options, onChange, label }: { value: T; options: Array<{ id: T; label: string }>; onChange: (v: T) => void; label: string }) {
  return (
    <span className="select"><select aria-label={label} value={String(value)} onChange={(e) => { const o = options.find((x) => String(x.id) === e.target.value); if (o) onChange(o.id); }}>{options.map((o) => <option key={String(o.id)} value={String(o.id)}>{o.label}</option>)}</select><ChevronIcon /></span>
  );
}
/** A destructive button that asks once more before it acts. */
function Confirm({ label, onConfirm }: { label: string; onConfirm: () => void }) {
  const [sure, setSure] = useState(false);
  return sure
    ? <><button type="button" className="btn small danger" onClick={() => { setSure(false); onConfirm(); }}>Yes, delete</button><button type="button" className="btn small ghost" onClick={() => setSure(false)}>Cancel</button></>
    : <button type="button" className="btn small danger" onClick={() => setSure(true)}>{label}</button>;
}

export function SettingsDialog({ settings, onChange, system, modelName, servers, manager, onServers, usage, engine, onModels, onDeleteChats, onDeleteSessions, onClose, start = "usage" }: Props) {
  const [sec, setSec] = useState<Section>(start);
  const [q, setQ] = useState("");
  const set = <K extends keyof Settings>(k: K, v: Settings[K]): void => onChange({ ...settings, [k]: v });
  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) set("modelsDir", p); };
  const clearFolder = (): void => { const { modelsDir: _d, ...rest } = settings; void _d; onChange(rest as Settings); };
  const needle = q.trim().toLowerCase();
  const groups = useMemo(() => GROUPS.map((g) => g.filter((i) => !needle || `${i.label} ${i.words}`.toLowerCase().includes(needle))).filter((g) => g.length), [needle]);
  const current = GROUPS.flat().find((i) => i.id === sec);

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog settings" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Settings">
        <DialogHeader settings icon={<GearIcon />} title="Settings" subtitle="Preferences for this app" onClose={onClose} />
        <div className="sbody">
          <nav className="snav" aria-label="Settings sections">
            <label className="ssearch"><SearchIcon /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" aria-label="Search settings" /></label>
            {groups.map((g, gi) => (
              <div className="sgroup" key={gi}>
                {g.map((n) => <button type="button" key={n.id} className={n.id === sec ? "on" : ""} aria-current={n.id === sec} onClick={() => setSec(n.id)}>{n.icon}<span>{n.label}</span></button>)}
              </div>
            ))}
            {groups.length === 0 ? <p className="empty-hint">No matching settings.</p> : null}
          </nav>
          <div className="spane">
            <h3 className="stitle">{current?.label}</h3>

            {sec === "profile" ? (
              <>
                <Row title="Your name" hint="Used in greetings and shown in the sidebar. It stays on this device.">
                  <input className="text-input" value={settings.userName ?? ""} maxLength={40} placeholder="Your name" aria-label="Your name" onChange={(e) => { const { userName: _u, ...rest } = settings; void _u; onChange(e.target.value ? { ...rest, userName: e.target.value } : (rest as Settings)); }} />
                </Row>
              </>
            ) : null}

            {sec === "usage" ? <UsagePanel {...usage} /> : null}

            {sec === "appearance" ? (
              <>
                <Row title="Theme" hint="Follow the system, or always dark or light."><Segmented label="Theme" value={settings.theme ?? "dark"} options={[{ id: "dark", label: "Dark" }, { id: "light", label: "Light" }, { id: "system", label: "System" }]} onChange={(v) => set("theme", v)} /></Row>
                <Row title="Text size" hint="Scales the whole interface."><Segmented label="Text size" value={settings.textSize} options={[{ id: "small", label: "Small" }, { id: "medium", label: "Medium" }, { id: "large", label: "Large" }]} onChange={(v) => set("textSize", v)} /></Row>
                <Row title="Suggestions on the chat home screen" hint="A few starter prompts under the message box."><Switch on={settings.suggestions} onChange={(v) => set("suggestions", v)} label="Suggestions" /></Row>
              </>
            ) : null}

            {sec === "notifications" ? (
              <>
                <Row title="Background jobs" hint="Tell me when an answer is ready or a job fails, if the window is not in front or it took a while."><Switch on={settings.notifyJobs} onChange={(v) => set("notifyJobs", v)} label="Background job notifications" /></Row>
                <Row title="Agent tasks" hint="Tell me when a task finishes while the window is not in front."><Switch on={settings.notifyAgent} onChange={(v) => set("notifyAgent", v)} label="Agent task notifications" /></Row>
              </>
            ) : null}

            {sec === "privacy" ? (
              <>
                <Row title="Everything stays on this device" hint="Conversations, sessions and settings are stored locally. Nothing is sent anywhere unless Browse is on or a tool you added uses the network."><span /></Row>
                <Sub>Your data</Sub>
                <Row title="Delete all conversations" hint="Removes every chat and its background jobs from this device."><Confirm label="Delete…" onConfirm={onDeleteChats} /></Row>
                <Row title="Delete all coding sessions" hint="Removes the session list and transcripts. Files in your projects, and the checkpoints kept inside them, are not touched."><Confirm label="Delete…" onConfirm={onDeleteSessions} /></Row>
              </>
            ) : null}

            {sec === "models" ? (
              <>
                <Row title="Models folder" hint={settings.modelsDir ?? `Not chosen. Looking in the app's own folder${system ? ` (${system.modelsDir})` : ""}.`}>
                  <button type="button" className="btn small ghost" onClick={() => void chooseFolder()}>{settings.modelsDir ? "Change…" : "Choose…"}</button>
                  {settings.modelsDir ? <button type="button" className="btn small ghost" onClick={clearFolder}>Clear</button> : null}
                </Row>
                <Row title="Same model for Chat and Code" hint="Choosing a model once applies it to both. Turn this off to give each its own, which loads when you switch between them."><Switch on={settings.sameModel !== false} onChange={(v) => onChange({ ...settings, sameModel: v, ...(v ? {} : { models: { chat: settings.models?.chat ?? settings.model, code: settings.models?.code ?? settings.model } }) })} label="Same model for Chat and Code" /></Row>
                {settings.sameModel === false ? (
                  <>
                    <Row title="Chat model" hint={settings.models?.chat ? `${settings.models.chat.name} · ${fmtBytes(settings.models.chat.bytes)}` : "None chosen yet."}><button type="button" className="btn small ghost" onClick={() => { onClose(); onModels("chat"); }}>Change…</button></Row>
                    <Row title="Code model" hint={settings.models?.code ? `${settings.models.code.name} · ${fmtBytes(settings.models.code.bytes)}` : "None chosen yet."}><button type="button" className="btn small ghost" onClick={() => { onClose(); onModels("code"); }}>Change…</button></Row>
                  </>
                ) : (
                  <Row title="Model" hint={settings.model ? `${settings.model.name} · ${fmtBytes(settings.model.bytes)}. Used for Chat and Code; selected again each time you open the app and loaded when you first use it.` : "None chosen yet."}>
                    <button type="button" className="btn small ghost" onClick={() => { onClose(); onModels(); }}>{modelName ? "Change…" : "Choose…"}</button>
                  </Row>
                )}
                <Row title="Context window" hint={ctxHint(settings.ctx, engine)}><Select label="Context window" value={settings.ctx} options={CTX_OPTIONS.map((n) => ({ id: n, label: `${n >= 1024 ? `${Math.round(n / 1024)}K` : n} tokens` }))} onChange={(v) => set("ctx", v)} /></Row>
                <Row title="Compact attention cache" hint="Stores the conversation memory as 8-bit numbers: about half the memory for long chats, a few percent slower. Takes effect when the model loads next."><Switch on={settings.kv8 === true} onChange={(v) => set("kv8", v)} label="Compact attention cache" /></Row>
                <Row title="Keep the model in memory" hint="Much faster when other apps are open. Models over 40% of your RAM are never pinned."><Switch on={settings.lock} onChange={(v) => set("lock", v)} label="Keep the model in memory" /></Row>
              </>
            ) : null}

            {sec === "chat" ? (
              <>
                <Row title="Creativity" hint={`${settings.temperature.toFixed(1)}. Lower is more precise, higher more varied.`}><input type="range" min={0} max={1.5} step={0.1} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} aria-label="Creativity" /></Row>
                <Row title="Think first" hint="Let thinking models reason before answering. Slower, often better."><Switch on={settings.thinking} onChange={(v) => set("thinking", v)} label="Think first" /></Row>
                <Row title="Show the thought process" hint="Open the reasoning by default."><Switch on={settings.showReasoning} onChange={(v) => set("showReasoning", v)} label="Show the thought process" /></Row>
                <Row title="Longest background answer"><Select label="Longest background answer" value={settings.jobTokens} options={[512, 1024, 2048, 4096].map((n) => ({ id: n, label: `${n.toLocaleString()} tokens` }))} onChange={(v) => set("jobTokens", v)} /></Row>
                <div className="sblock"><b>Instructions</b><textarea rows={5} value={settings.system} onChange={(e) => set("system", e.target.value)} /><button type="button" className="link" onClick={() => set("system", DEFAULT_SYSTEM)}>Reset to default</button></div>
              </>
            ) : null}

            {sec === "web" ? (
              <>
                <Row title="Browse the web in chat" tag="Sends your question out" hint="Look questions up first and answer with sources. Greetings and one-word replies are not searched."><Switch on={settings.browse} onChange={(v) => set("browse", v)} label="Browse the web in chat" /></Row>
                <Row title="Search service" hint="Your question is sent to DuckDuckGo and the pages it finds are fetched from the internet. Addresses on your own network are never opened."><span className="badge">DuckDuckGo</span></Row>
                <Row title="The agent" hint="In Code mode the agent can search and read pages too, after asking once per task."><span /></Row>
              </>
            ) : null}

            {sec === "remote" ? <RemotePanel settings={settings} onChange={onChange} /> : null}

            {sec === "connectors" ? <div className="stools"><McpPanel servers={servers} manager={manager} onChange={onServers} /></div> : null}

            {sec === "code" ? (
              <>
                <Row title="Permissions" hint="What the agent may do without asking."><Select label="Permissions" value={settings.permission} options={[{ id: "ask", label: "Ask first" }, { id: "auto", label: "Auto-run checks" }, { id: "plan", label: "Plan only" }]} onChange={(v) => set("permission", v)} /></Row>
                <Row title="Default transcript view" hint="How much detail the agent's steps show."><Segmented label="Transcript view" value={settings.transcript} options={[{ id: "normal", label: "Normal" }, { id: "verbose", label: "Verbose" }]} onChange={(v) => set("transcript", v)} /></Row>
                <Row title="Most steps per task" hint="The agent stops after this many tool calls."><Select label="Most steps" value={settings.maxSteps} options={[10, 20, 40, 80].map((n) => ({ id: n, label: String(n) }))} onChange={(v) => set("maxSteps", v)} /></Row>
                <Sub>Changes</Sub>
                <Row title="Changes are always shown first" hint="The agent works on a private copy. Nothing reaches your folder until you apply it, and every apply can be undone."><span /></Row>
              </>
            ) : null}

            {sec === "desktop" ? (
              <>
                <Row title="Restore default settings" hint="Your conversations, sessions and the model folder are kept."><button type="button" className="btn small ghost" onClick={() => onChange({ ...DEFAULT_SETTINGS, ...(settings.model ? { model: settings.model } : {}), ...(settings.modelsDir ? { modelsDir: settings.modelsDir } : {}) })}>Restore</button></Row>
                <Row title="Models folder" hint={system?.modelsDir ?? "Unknown"}><button type="button" className="btn small ghost" onClick={() => void bridge.revealModels()}>Open</button></Row>
              </>
            ) : null}

            {sec === "about" ? (
              <>
                <div className="about-hero">
                  <Logo size={64} />
                  <div><b>Clutor</b><small>Run large language models on your own machine, with honest numbers.</small><span className="badge">Version {__APP_VERSION__}</span></div>
                </div>
                <Sub>App</Sub>
                <Details rows={[["Version", __APP_VERSION__], ["Last updated", new Date(`${__APP_BUILT__}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })], ["Platform", bridge.native ? "Desktop app" : "Browser preview (no engine)"]]} />
                <Sub>This computer</Sub>
                <Details rows={system ? [["System", `${system.os} ${system.arch}`], ["Memory", fmtBytes(system.ramBytes)], ["Cores", String(system.cores)], ["Models folder", settings.modelsDir ?? system.modelsDir]] : [["System", "Unknown"]]} />
                <Sub>Engine</Sub>
                <Details rows={engine ? [["Model", engine.name], ["Architecture", engine.arch], ["Layers", String(engine.nLayer)], ["Vocabulary", engine.nVocab.toLocaleString()], ["Context", `${engine.nCtx.toLocaleString()} tokens`], ["Threads", String(engine.threads)], ["Memory", engine.locked ? "pinned" : "not pinned"], ["Experts", engine.streaming ? "streamed from disk" : "not streaming"]] : [["Model", "none loaded"]]} />
                <Sub>Licence and links</Sub>
                <Row title="MIT licence" hint="Free to use, change and share."><button type="button" className="btn small ghost" onClick={() => void bridge.web.openExternal("https://github.com/isagarpinninti/clutor")}>Source on GitHub</button></Row>
                <Row title="Report a problem" hint="Include the version shown above."><button type="button" className="btn small ghost" onClick={() => void bridge.web.openExternal("https://github.com/isagarpinninti/clutor/issues")}>Open issues</button></Row>
                <Row title="Privacy" hint="Everything runs and stays on this device. Browse and tool servers are the only things that use the network, and only when you turn them on."><span /></Row>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
