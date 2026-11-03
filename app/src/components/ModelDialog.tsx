import { useEffect, useState } from "react";
import { bridge, type CatalogItem, type DownloadProgress, type ModelInfo, type SystemInfo } from "../bridge";
import { fmtBytes } from "../store";
import { ChipIcon, DownloadIcon, FolderIcon } from "./icons";

interface Props {
  system: SystemInfo | undefined;
  current: string | undefined;
  onPick: (m: ModelInfo) => void;
  onClose: () => void;
}

export function ModelDialog({ system, current, onPick, onClose }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [tab, setTab] = useState<"local" | "get">("local");

  const refresh = (): void => { void bridge.listModels().then(setModels); };
  useEffect(() => { refresh(); void bridge.catalog().then(setCatalog); }, []);

  const get = (item: CatalogItem): void => {
    setProgress((p) => ({ ...p, [item.id]: { id: item.id, done: 0, total: item.bytes, state: "downloading" } }));
    bridge.download(item.id, (p) => setProgress((all) => ({ ...all, [item.id]: p })))
      .then(refresh)
      .catch((e: unknown) => setProgress((all) => ({ ...all, [item.id]: { id: item.id, done: 0, total: item.bytes, state: "error", error: e instanceof Error ? e.message : String(e) } })));
  };

  const fit = (bytes: number): string => {
    if (!system) return "";
    const r = bytes / system.ramBytes;
    return r <= 0.6 ? "fits in memory" : r <= 0.8 ? "tight: close other apps" : "larger than memory: only mixture-of-experts models can stream from disk";
  };

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Models">
        <header><h2>Models</h2><button type="button" className="icon-btn" onClick={onClose} aria-label="Close">✕</button></header>
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "local"} className={tab === "local" ? "on" : ""} onClick={() => setTab("local")}>On this device</button>
          <button type="button" role="tab" aria-selected={tab === "get"} className={tab === "get" ? "on" : ""} onClick={() => setTab("get")}>Get a model</button>
        </div>
        {tab === "local" ? (
          <div className="list">
            {models.length === 0 ? <p className="empty-hint">No models yet. Open “Get a model”, or put a .gguf file in the models folder.</p> : null}
            {models.map((m) => (
              <button type="button" key={m.path} className={`row${m.path === current ? " active" : ""}`} onClick={() => { onPick(m); onClose(); }}>
                <ChipIcon />
                <span className="row-main"><b>{m.name}</b><small>{fmtBytes(m.bytes)} · {fit(m.bytes)}</small></span>
                {m.path === current ? <span className="badge">Loaded</span> : null}
              </button>
            ))}
            <button type="button" className="row ghost" onClick={() => void bridge.revealModels()}><FolderIcon /><span className="row-main"><b>Open models folder</b><small>{system?.modelsDir ?? ""}</small></span></button>
          </div>
        ) : (
          <div className="list">
            {catalog.map((c) => {
              const p = progress[c.id];
              const pct = p && p.total ? Math.round((p.done / p.total) * 100) : 0;
              return (
                <div key={c.id} className="row static">
                  <DownloadIcon />
                  <span className="row-main"><b>{c.name}</b><small>{fmtBytes(c.bytes)} · {c.description}</small>
                    {p && p.state !== "done" && p.state !== "error" ? <span className="bar"><i style={{ width: `${pct}%` }} /></span> : null}
                    {p?.state === "error" ? <small className="err">{p.error}</small> : null}
                  </span>
                  {p?.state === "done" ? <span className="badge">Downloaded</span> : p?.state === "downloading" || p?.state === "verifying" ? <span className="badge">{p.state === "verifying" ? "Checking…" : `${pct}%`}</span> : <button type="button" className="btn small" onClick={() => get(c)}>Download</button>}
                </div>
              );
            })}
            <p className="fineprint">Downloads are checked against the publisher's SHA-256 before use.</p>
          </div>
        )}
      </div>
    </div>
  );
}
