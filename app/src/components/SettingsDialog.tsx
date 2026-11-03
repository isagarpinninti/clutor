import { DEFAULT_SYSTEM, type Settings } from "../store";

interface Props { settings: Settings; onChange: (s: Settings) => void; onClose: () => void }

export function SettingsDialog({ settings, onChange, onClose }: Props) {
  const set = <K extends keyof Settings>(k: K, v: Settings[K]): void => onChange({ ...settings, [k]: v });
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Settings">
        <header><h2>Settings</h2><button type="button" className="icon-btn" onClick={onClose} aria-label="Close">✕</button></header>
        <div className="form">
          <label>Appearance
            <select value={settings.theme} onChange={(e) => set("theme", e.target.value as Settings["theme"])}>
              <option value="system">Match the system</option><option value="light">Light</option><option value="dark">Dark</option>
            </select>
          </label>
          <label>Creativity <output>{settings.temperature.toFixed(1)}</output>
            <input type="range" min={0} max={1.5} step={0.1} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
          </label>
          <label>Instructions
            <textarea rows={4} value={settings.system} onChange={(e) => set("system", e.target.value)} />
            <button type="button" className="link" onClick={() => set("system", DEFAULT_SYSTEM)}>Reset</button>
          </label>
          <label>Conversation length (applies when a model is loaded)
            <select value={settings.ctx} onChange={(e) => set("ctx", Number(e.target.value))}>
              {[2048, 4096, 8192, 16384].map((n) => <option key={n} value={n}>{n.toLocaleString()} tokens</option>)}
            </select>
          </label>
          <label className="check"><input type="checkbox" checked={settings.lock} onChange={(e) => set("lock", e.target.checked)} />
            <span>Keep the model in memory<small>Much faster when other apps are open; uses that much RAM.</small></span>
          </label>
          <label className="check"><input type="checkbox" checked={settings.thinking} onChange={(e) => set("thinking", e.target.checked)} />
            <span>Let thinking models think first<small>Slower, often better. Also on the Think button.</small></span>
          </label>
          <label className="check"><input type="checkbox" checked={settings.showReasoning} onChange={(e) => set("showReasoning", e.target.checked)} />
            <span>Show the thought process by default</span>
          </label>
        </div>
      </div>
    </div>
  );
}
