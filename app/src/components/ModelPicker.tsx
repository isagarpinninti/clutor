import { useEffect, useRef, useState } from "react";
import { bridge, type ModelInfo } from "../bridge";
import { fmtBytes } from "../store";
import { loadProviders } from "../cloud/providers";
import { cloudModel } from "./CloudPane";
import { Spark } from "./Thinking";

interface Props {
  current?: ModelInfo | undefined;
  status: "idle" | "loading" | "ready" | "error";
  /** Short note about how the loaded model runs, e.g. "pinned in memory". */
  note?: string | undefined;
  folders?: string[];
  /** Chat and Code share one model (the default). */
  same?: boolean;
  onSame?: (v: boolean) => void;
  /** Clutor picks the model for each task from the folder. */
  auto?: boolean;
  /** Why the automatic pick chose the model it did. */
  autoReason?: string;
  onAuto?: () => void;
  onPick: (m: ModelInfo) => void;
  onMore: () => void;
}

/** The model chooser that sits under the message box: the loaded model's name, and a small menu of the models on this device. */
export function ModelPicker({ current, status, note, folders, same, onSame, auto, autoReason, onAuto, onPick, onMore }: Props) {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => { if (open) void bridge.listModels(folders).then(setModels); }, [open, folders]);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent): void => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent): void => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", away); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", esc); };
  }, [open]);

  const label = status === "loading" ? "Loading…" : current ? (auto ? `Auto · ${current.name}` : current.name) : "Choose a model";
  return (
    <div className="picker" ref={box}>
      <button type="button" className={`picker-btn ${status}`} onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}>
        {status === "loading" ? <Spark size={12} /> : null}
        <span className="picker-name">{label}</span>
        {status === "ready" && note ? <small>{note}</small> : status === "idle" && current ? <small>not loaded</small> : null}
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="m2 4 3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open ? (
        <div className="pmenu" role="listbox" aria-label="Models">
          {onAuto ? <button type="button" role="menuitemcheckbox" aria-checked={!!auto} className={`menu-item two auto${auto ? " on" : ""}`} onClick={() => { setOpen(false); onAuto(); }} title={autoReason}><span className="mi-text"><b>Auto</b><small>{auto && autoReason ? autoReason : "the best model for each task, from your models folder"}</small></span></button> : null}
          {models.length === 0 && loadProviders().length === 0 ? <p className="menu-empty">No models on this device yet.</p> : null}
          {models.map((m) => (
            <button type="button" role="option" aria-selected={m.path === current?.path} key={m.path} className={`menu-item${m.path === current?.path ? " on" : ""}`} onClick={() => { setOpen(false); if (m.path !== current?.path) onPick(m); }}>
              <span>{m.name}</span><small>{fmtBytes(m.bytes)}</small>
            </button>
          ))}
          {loadProviders().filter((p) => p.confirmed).flatMap((p) => p.models.map((m) => cloudModel(p, m))).map((m) => (
            <button type="button" role="option" aria-selected={m.path === current?.path} key={m.path} className={`menu-item${m.path === current?.path ? " on" : ""}`} onClick={() => { setOpen(false); if (m.path !== current?.path) onPick(m); }}>
              <span>{m.name}</span><small>{m.cloud} · cloud</small>
            </button>
          ))}
          {onSame ? <button type="button" role="menuitemcheckbox" aria-checked={same ?? true} className="menu-item same" onClick={() => onSame(!(same ?? true))}><span>Same model for Chat and Code</span><span className={`mini-check${(same ?? true) ? " on" : ""}`} aria-hidden="true" /></button> : null}
          <button type="button" className="menu-item more" onClick={() => { setOpen(false); onMore(); }}>Get more models…</button>
        </div>
      ) : null}
    </div>
  );
}
