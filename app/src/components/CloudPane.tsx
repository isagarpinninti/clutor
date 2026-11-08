import { useEffect, useState } from "react";
import { bridge, type ModelInfo } from "../bridge";
import { CloudSession } from "../cloud/session";
import { cloudPath, loadProviders, PRESETS, saveProviders, type CloudProvider } from "../cloud/providers";
import { ChipIcon } from "./icons";

/** A cloud model as the rest of the app sees a model: no file, so no size. */
export const cloudModel = (p: CloudProvider, model: string): ModelInfo => ({ path: cloudPath(p.id, model), name: model, bytes: 0, cloud: p.name });

interface Draft { presetId: string; id: string; name: string; kind: CloudProvider["kind"]; baseUrl: string; auth: CloudProvider["auth"]; models: string; ctx: number; key: string; understood: boolean }
const fromPreset = (id: string): Draft => { const p = PRESETS.find((x) => x.id === id) ?? PRESETS[0]!; return { presetId: p.id, id: p.id, name: p.name, kind: p.kind, baseUrl: p.baseUrl, auth: p.auth, models: p.models.join(", "), ctx: p.ctx, key: "", understood: false }; };

/** Setting up cloud providers (a key, an address, the models to offer) and choosing one of their models. Keys go to the app's native side and are never shown again. */
export function CloudPane({ current, onPick, onClose }: { current: string | undefined; onPick: (m: ModelInfo) => void; onClose: () => void }) {
  const [providers, setProviders] = useState<CloudProvider[]>(loadProviders);
  const [hasKey, setHasKey] = useState<Record<string, boolean>>({});
  const [draft, setDraft] = useState<Draft>();
  const [error, setError] = useState<string>();
  const [test, setTest] = useState<Record<string, string>>({});
  useEffect(() => { void Promise.all(providers.map(async (p) => [p.id, await bridge.cloud.keyHas(p.id)] as const)).then((r) => setHasKey(Object.fromEntries(r))); }, [providers]);
  const commit = (list: CloudProvider[]): void => { setProviders(list); saveProviders(list); };

  const save = async (): Promise<void> => {
    if (!draft) return;
    setError(undefined);
    const base = draft.baseUrl.trim();
    if (!/^https?:\/\/.+/.test(base)) { setError("The address must start with http:// or https://"); return; }
    if (!draft.understood) { setError("Tick the box to confirm that this provider will receive your messages."); return; }
    try {
      if (draft.auth !== "none" && draft.key.trim()) await bridge.cloud.keySet(draft.id, draft.key.trim());
      else if (draft.auth !== "none" && !hasKey[draft.id]) { setError("Paste the API key from your provider."); return; }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); return; }
    const p: CloudProvider = { id: draft.id, name: draft.name.trim() || draft.id, kind: draft.kind, baseUrl: base, auth: draft.auth, models: draft.models.split(/[,\n]/).map((m) => m.trim()).filter(Boolean), ctx: Math.max(4096, Math.round(draft.ctx) || 32000), confirmed: true };
    commit([...providers.filter((x) => x.id !== p.id), p]);
    setDraft(undefined);
  };
  const remove = async (p: CloudProvider): Promise<void> => { await bridge.cloud.keyClear(p.id).catch(() => undefined); commit(providers.filter((x) => x.id !== p.id)); };
  const runTest = async (p: CloudProvider, model: string): Promise<void> => {
    setTest((t) => ({ ...t, [p.id]: "Testing…" }));
    try {
      let text = "";
      for await (const ev of new CloudSession(p, model).stream([{ role: "user", content: "Reply with the single word: ready" }], { max: 20, temp: 0 })) if (ev.type === "content") text += ev.text;
      setTest((t) => ({ ...t, [p.id]: text.trim() ? `Works: “${text.trim().slice(0, 60)}”` : "Connected, but the model sent no text." }));
    } catch (e) { setTest((t) => ({ ...t, [p.id]: e instanceof Error ? e.message : String(e) })); }
  };

  if (draft) {
    const set = <K extends keyof Draft>(k: K, v: Draft[K]): void => setDraft((d) => (d ? { ...d, [k]: v } : d));
    const editing = providers.some((p) => p.id === draft.id);
    return (
      <div className="list cloud-form">
        {!editing ? (
          <label className="field"><span>Provider</span>
            <select value={draft.presetId} onChange={(e) => setDraft(fromPreset(e.target.value))} aria-label="Provider">{PRESETS.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
          </label>
        ) : <p className="empty-hint">Editing {draft.name}</p>}
        <label className="field"><span>Name</span><input value={draft.name} onChange={(e) => set("name", e.target.value)} /></label>
        <label className="field"><span>Address</span><input value={draft.baseUrl} onChange={(e) => set("baseUrl", e.target.value)} spellCheck={false} /></label>
        {draft.auth !== "none" ? (
          <label className="field"><span>API key {hasKey[draft.id] ? "(saved; leave empty to keep it)" : ""}</span><input type="password" autoComplete="off" value={draft.key} onChange={(e) => set("key", e.target.value)} placeholder={hasKey[draft.id] ? "••••••••" : `from ${PRESETS.find((p) => p.id === draft.presetId)?.keyHint ?? "your provider"}`} /></label>
        ) : null}
        <label className="field"><span>Models to offer (separated by commas; any name your provider accepts)</span><textarea rows={2} value={draft.models} onChange={(e) => set("models", e.target.value)} spellCheck={false} /></label>
        <label className="field"><span>Context window (tokens)</span><input type="number" min={4096} value={draft.ctx} onChange={(e) => set("ctx", Number(e.target.value))} /></label>
        <label className="check"><input type="checkbox" checked={draft.understood} onChange={(e) => set("understood", e.target.checked)} /> <span>I understand that with this provider my messages, and in coding mode the files the agent reads, are sent to {draft.name || "it"}{draft.auth === "none" ? " (an address you chose: check it is yours)" : ""}. Local models never send anything.</span></label>
        {error ? <p className="err" role="alert">{error}</p> : null}
        <div className="actions"><button type="button" className="btn small" onClick={() => void save()}>Save</button><button type="button" className="btn small ghost" onClick={() => { setDraft(undefined); setError(undefined); }}>Cancel</button></div>
      </div>
    );
  }

  return (
    <div className="list">
      <p className="fineprint">Cloud models run at a provider you set up, so your messages leave this computer. The API key is kept by the app and never shown again. Use a local model when the text must stay private.</p>
      {providers.length === 0 ? <p className="empty-hint">No provider yet. Add one to use its models.</p> : null}
      {providers.map((p) => (
        <div key={p.id} className="model-item">
          <div className="row static"><ChipIcon /><span className="row-main"><b>{p.name}</b><small>{p.baseUrl} · {p.auth === "none" ? "no key needed" : hasKey[p.id] ? "key saved" : "no key"}</small></span>
            <button type="button" className="btn small ghost" onClick={() => setDraft({ ...fromPreset(PRESETS.find((x) => x.id === p.id)?.id ?? "custom"), id: p.id, name: p.name, kind: p.kind, baseUrl: p.baseUrl, auth: p.auth, models: p.models.join(", "), ctx: p.ctx, key: "", understood: true, presetId: p.id })}>Edit</button>
            <button type="button" className="btn small ghost" onClick={() => void remove(p)}>Remove</button></div>
          {p.models.length === 0 ? <p className="empty-hint">Edit this provider and add the model names to use.</p> : null}
          {p.models.map((m) => (
            <div key={m} className={`row split${current === cloudPath(p.id, m) ? " active" : ""}`}>
              <button type="button" className="row-pick" onClick={() => { onPick(cloudModel(p, m)); onClose(); }}><span className="row-main"><b>{m}</b><small>{p.name} · {Math.round(p.ctx / 1000)}K context</small></span>{current === cloudPath(p.id, m) ? <span className="badge">Loaded</span> : null}</button>
              <button type="button" className="btn small ghost" onClick={() => void runTest(p, m)}>Test</button>
            </div>
          ))}
          {test[p.id] ? <p className="check-note">{test[p.id]}</p> : null}
        </div>
      ))}
      <button type="button" className="row ghost" onClick={() => setDraft(fromPreset("anthropic"))}><span className="row-main"><b>Add a provider…</b><small>Anthropic, OpenAI, Gemini, OpenRouter, DeepSeek, Groq, Mistral, Ollama and any OpenAI-compatible address</small></span></button>
    </div>
  );
}
