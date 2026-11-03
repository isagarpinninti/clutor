import { useState } from "react";
import { bridge, type SystemInfo } from "../bridge";
import type { McpManager, McpServerConfig } from "../agent/mcp";
import { DEFAULT_SYSTEM, fmtBytes, type Settings } from "../store";
import { McpPanel } from "./McpDialog";
import { CloseIcon } from "./icons";

interface Props {
  settings: Settings;
  onChange: (s: Settings) => void;
  system: SystemInfo | undefined;
  modelName: string | undefined;
  servers: McpServerConfig[];
  manager: McpManager;
  onServers: (s: McpServerConfig[]) => void;
  onModels: () => void;
  onClose: () => void;
  start?: Section;
}

export type Section = "general" | "models" | "chat" | "agent" | "web" | "tools" | "about";
const NAV: Array<{ id: Section; label: string }> = [
  { id: "general", label: "General" }, { id: "models", label: "Models" }, { id: "chat", label: "Chat" },
  { id: "agent", label: "Agent" }, { id: "web", label: "Web" }, { id: "tools", label: "Tools" }, { id: "about", label: "About" },
];

function Row({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return <div className="srow"><div className="srow-text"><b>{title}</b>{hint ? <small>{hint}</small> : null}</div><div className="srow-ctl">{children}</div></div>;
}
function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button type="button" role="switch" aria-checked={on} aria-label={label} className={`switch${on ? " on" : ""}`} onClick={() => onChange(!on)}><i /></button>;
}

export function SettingsDialog({ settings, onChange, system, modelName, servers, manager, onServers, onModels, onClose, start = "general" }: Props) {
  const [sec, setSec] = useState<Section>(start);
  const set = <K extends keyof Settings>(k: K, v: Settings[K]): void => onChange({ ...settings, [k]: v });
  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) set("modelsDir", p); };
  const clearFolder = (): void => { const { modelsDir: _d, ...rest } = settings; void _d; onChange(rest as Settings); };

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog settings" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Settings">
        <nav className="snav" aria-label="Settings sections">
          <h2>Settings</h2>
          {NAV.map((n) => <button type="button" key={n.id} className={n.id === sec ? "on" : ""} aria-current={n.id === sec} onClick={() => setSec(n.id)}>{n.label}</button>)}
        </nav>
        <div className="spane">
          <button type="button" className="icon-btn sclose" onClick={onClose} aria-label="Close"><CloseIcon /></button>
          <h3 className="stitle">{NAV.find((n) => n.id === sec)?.label}</h3>

          {sec === "general" ? (
            <>
              <Row title="Appearance" hint="Clutor is dark only."><span className="badge">Dark</span></Row>
              <Row title="Everything stays on this device" hint="Conversations, sessions and settings are stored locally. Nothing is sent anywhere unless you turn on Browse or run a tool that uses the network."><span /></Row>
              <Row title="Keep the model in memory" hint="Much faster when other apps are open. Models over 40% of your RAM are never pinned."><Switch on={settings.lock} onChange={(v) => set("lock", v)} label="Keep the model in memory" /></Row>
            </>
          ) : null}

          {sec === "models" ? (
            <>
              <Row title="Models folder" hint={settings.modelsDir ?? `Not chosen. Looking in the app's own folder${system ? ` (${system.modelsDir})` : ""}.`}>
                <button type="button" className="btn small ghost" onClick={() => void chooseFolder()}>{settings.modelsDir ? "Change…" : "Choose…"}</button>
                {settings.modelsDir ? <button type="button" className="btn small ghost" onClick={clearFolder}>Clear</button> : null}
              </Row>
              <Row title="Model" hint={settings.model ? `${settings.model.name} · ${fmtBytes(settings.model.bytes)}. It is selected again each time you open the app and loads when you first use it.` : "None chosen yet."}>
                <button type="button" className="btn small ghost" onClick={() => { onClose(); onModels(); }}>{modelName ? "Change…" : "Choose…"}</button>
              </Row>
              <Row title="Conversation length" hint="How much the model can keep in mind. Applies the next time a model loads.">
                <select value={settings.ctx} onChange={(e) => set("ctx", Number(e.target.value))}>{[2048, 4096, 8192, 16384].map((n) => <option key={n} value={n}>{n.toLocaleString()} tokens</option>)}</select>
              </Row>
            </>
          ) : null}

          {sec === "chat" ? (
            <>
              <Row title="Creativity" hint={`${settings.temperature.toFixed(1)}. Lower is more precise, higher more varied.`}><input type="range" min={0} max={1.5} step={0.1} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} /></Row>
              <Row title="Think first" hint="Let thinking models reason before answering. Slower, often better."><Switch on={settings.thinking} onChange={(v) => set("thinking", v)} label="Think first" /></Row>
              <Row title="Show the thought process" hint="Open the reasoning by default."><Switch on={settings.showReasoning} onChange={(v) => set("showReasoning", v)} label="Show the thought process" /></Row>
              <Row title="Longest background answer"><select value={settings.jobTokens} onChange={(e) => set("jobTokens", Number(e.target.value))}>{[512, 1024, 2048, 4096].map((n) => <option key={n} value={n}>{n.toLocaleString()} tokens</option>)}</select></Row>
              <div className="sblock"><b>Instructions</b><textarea rows={5} value={settings.system} onChange={(e) => set("system", e.target.value)} /><button type="button" className="link" onClick={() => set("system", DEFAULT_SYSTEM)}>Reset to default</button></div>
            </>
          ) : null}

          {sec === "agent" ? (
            <>
              <Row title="Permissions" hint="What the agent may do without asking.">
                <select value={settings.permission} onChange={(e) => set("permission", e.target.value as Settings["permission"])}>
                  <option value="ask">Ask first</option><option value="auto">Auto-run checks</option><option value="plan">Plan only</option>
                </select>
              </Row>
              <Row title="Most steps per task" hint="The agent stops after this many tool calls."><select value={settings.maxSteps} onChange={(e) => set("maxSteps", Number(e.target.value))}>{[10, 20, 40, 80].map((n) => <option key={n} value={n}>{n}</option>)}</select></Row>
              <Row title="Changes are always shown first" hint="The agent works on a private copy. Nothing reaches your folder until you apply it, and every apply can be undone."><span /></Row>
            </>
          ) : null}

          {sec === "web" ? (
            <>
              <Row title="Browse the web in chat" hint="Look questions up first and answer with sources. Greetings and one-word replies are not searched."><Switch on={settings.browse} onChange={(v) => set("browse", v)} label="Browse the web in chat" /></Row>
              <Row title="Search service" hint="Your question is sent to DuckDuckGo, and the pages it finds are fetched from the internet. Addresses on your own network are never opened."><span className="badge">DuckDuckGo</span></Row>
              <Row title="The agent" hint="In Code mode the agent can search and read pages too, after asking once per task."><span /></Row>
            </>
          ) : null}

          {sec === "tools" ? <div className="stools"><McpPanel servers={servers} manager={manager} onChange={onServers} /></div> : null}

          {sec === "about" ? (
            <>
              <Row title="Clutor" hint="Run large language models on your own machine, with honest numbers."><span className="badge">0.4.0</span></Row>
              <Row title="This computer" hint={system ? `${system.os} ${system.arch} · ${fmtBytes(system.ramBytes)} memory · ${system.cores} cores` : "Unknown"}><span /></Row>
              <Row title="Source and licence" hint="MIT licensed."><button type="button" className="btn small ghost" onClick={() => void bridge.web.openExternal("https://github.com/isagarpinninti/clutor")}>Open on GitHub</button></Row>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
