import { DialogHeader } from "./DialogHeader";
import { ModelCheckCard } from "./ModelCheckCard";
import { useEffect, useState } from "react";
import { bridge, type CatalogItem, type DownloadProgress, type ModelInfo, type SystemInfo } from "../bridge";
import { fmtBytes } from "../store";
import { ChipIcon, DownloadIcon, FolderIcon } from "./icons";
import { CloudPane } from "./CloudPane";

interface Props {
  /** The context window the check assumes (Settings > Models). */
  contextTokens: number;
  folders: string[];
  onFolder: (path: string) => void;
  system: SystemInfo | undefined;
  current: string | undefined;
  onPick: (m: ModelInfo) => void;
  onClose: () => void;
}

export function ModelDialog({ contextTokens, folders, onFolder, system, current, onPick, onClose }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [tab, setTab] = useState<"local" | "get" | "cloud">("local");
  const [checking, setChecking] = useState<string>();

  const refresh = (): void => { void bridge.listModels(folders).then(setModels); };
  useEffect(() => { refresh(); }, [folders]);
  useEffect(() => { void bridge.catalog().then(setCatalog); }, []);
  const chooseFolder = async (): Promise<void> => { const p = await bridge.host.pickFolder(); if (p) onFolder(p); };

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
        <DialogHeader icon={<ChipIcon />} title="Models" subtitle="Choose what runs on this device" onClose={onClose} />
        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "local"} className={tab === "local" ? "on" : ""} onClick={() => setTab("local")}>On this device</button>
          <button type="button" role="tab" aria-selected={tab === "get"} className={tab === "get" ? "on" : ""} onClick={() => setTab("get")}>Get a model</button>
          <button type="button" role="tab" aria-selected={tab === "cloud"} className={tab === "cloud" ? "on" : ""} onClick={() => setTab("cloud")}>Cloud</button>
        </div>
        {tab === "local" ? (
          <div className="list">
            {models.length === 0 ? <p className="empty-hint">No models yet. Open “Get a model”, or put a .gguf file in the models folder.</p> : null}
            {models.map((m) => (
              <div key={m.path} className="model-item">
                <div className={`row split${m.path === current ? " active" : ""}`}>
                  <button type="button" className="row-pick" onClick={() => { onPick(m); onClose(); }}>
                    <ChipIcon />
                    <span className="row-main"><b>{m.name}</b><small>{fmtBytes(m.bytes)} · {fit(m.bytes)}</small></span>
                    {m.path === current ? <span className="badge">Loaded</span> : null}
                  </button>
                  <button type="button" className="btn small ghost" aria-expanded={checking === m.path} onClick={() => setChecking((c) => (c === m.path ? undefined : m.path))}>Will it run?</button>
                </div>
                {checking === m.path && system ? <ModelCheckCard model={m} ramBytes={system.ramBytes} contextTokens={contextTokens} /> : null}
              </div>
            ))}
            <button type="button" className="row ghost" onClick={() => void chooseFolder()}><FolderIcon /><span className="row-main"><b>{folders[0] ? "Change models folder" : "Choose a models folder"}</b><small>{folders[0] ?? "Pick the folder where your .gguf files are"}</small></span></button>
            <button type="button" className="row ghost" onClick={() => void bridge.revealModels()}><FolderIcon /><span className="row-main"><b>Open models folder</b><small>{system?.modelsDir ?? ""}</small></span></button>
          </div>
        ) : tab === "cloud" ? (
          <CloudPane current={current} onPick={onPick} onClose={onClose} />
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
