import { DEFAULT_SYSTEM, type Settings } from "../store";
import { CloseIcon } from "./icons";

interface Props { settings: Settings; onChange: (s: Settings) => void; onTools: () => void; onClose: () => void }

export function SettingsDialog({ settings, onChange, onTools, onClose }: Props) {
  const set = <K extends keyof Settings>(k: K, v: Settings[K]): void => onChange({ ...settings, [k]: v });
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Settings">
        <header><h2>Settings</h2><button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon /></button></header>
        <div className="form sections">
          <section className="panel">
            <h3>Model and memory</h3>
            <label className="field">Conversation length<small>Applies the next time a model loads.</small>
              <select value={settings.ctx} onChange={(e) => set("ctx", Number(e.target.value))}>
                {[2048, 4096, 8192, 16384].map((n) => <option key={n} value={n}>{n.toLocaleString()} tokens</option>)}
              </select>
            </label>
            <label className="check"><input type="checkbox" checked={settings.lock} onChange={(e) => set("lock", e.target.checked)} />
              <span>Keep the model in memory<small>Much faster when other apps are open. Models over 40% of your RAM are never pinned.</small></span>
            </label>
          </section>
          <section className="panel">
            <h3>Replies</h3>
            <label className="field">Creativity <output>{settings.temperature.toFixed(1)}</output>
              <input type="range" min={0} max={1.5} step={0.1} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
            </label>
            <label className="field">Instructions
              <textarea rows={4} value={settings.system} onChange={(e) => set("system", e.target.value)} />
              <button type="button" className="link" onClick={() => set("system", DEFAULT_SYSTEM)}>Reset to default</button>
            </label>
            <label className="field">Longest background answer
              <select value={settings.jobTokens} onChange={(e) => set("jobTokens", Number(e.target.value))}>
                {[512, 1024, 2048, 4096].map((n) => <option key={n} value={n}>{n.toLocaleString()} tokens</option>)}
              </select>
            </label>
            <label className="check"><input type="checkbox" checked={settings.thinking} onChange={(e) => set("thinking", e.target.checked)} />
              <span>Let thinking models think first<small>Slower, often better. Also on the Think button.</small></span>
            </label>
            <label className="check"><input type="checkbox" checked={settings.showReasoning} onChange={(e) => set("showReasoning", e.target.checked)} />
              <span>Show the thought process by default</span>
            </label>
          </section>
          <section className="panel">
            <h3>Agent and tools</h3>
            <button type="button" className="row" onClick={() => { onClose(); onTools(); }}><span className="row-main"><b>Tools (MCP servers)</b><small>Give the agent extra tools, such as search or a database.</small></span><span className="badge">Manage</span></button>
          </section>
        </div>
      </div>
    </div>
  );
}
