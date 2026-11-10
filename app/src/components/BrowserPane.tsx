import { useState } from "react";
import { bridge, type SearchHit, type WebPage } from "../bridge";
import { SearchIcon } from "./icons";

type View = { kind: "hits"; query: string; hits: SearchHit[] } | { kind: "page"; page: WebPage } | { kind: "error"; text: string } | undefined;

const looksLikeAddress = (s: string): boolean => /^https?:\/\//i.test(s) || /^[a-z0-9-]+(\.[a-z0-9-]+)+(\/\S*)?$/i.test(s);

/** A reader for the web: search, or open an address, and read the page as text. Pages are fetched by the app (addresses on your own network are refused), not shown in a live browser. */
export function BrowserPane() {
  const [q, setQ] = useState("");
  const [view, setView] = useState<View>();
  const [back, setBack] = useState<View[]>([]);
  const [busy, setBusy] = useState(false);

  const go = async (text: string, push = true): Promise<void> => {
    const t = text.trim();
    if (!t) return;
    setBusy(true);
    if (push && view) setBack((b) => [...b, view].slice(-20));
    try {
      if (looksLikeAddress(t)) setView({ kind: "page", page: await bridge.web.fetch(/^https?:\/\//i.test(t) ? t : `https://${t}`) });
      else setView({ kind: "hits", query: t, hits: await bridge.web.search(t) });
    } catch (e) { setView({ kind: "error", text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(false); }
  };

  return (
    <div className="browser">
      <form className="addr" onSubmit={(e) => { e.preventDefault(); void go(q); }}>
        <button type="button" className="icon-btn" disabled={back.length === 0} onClick={() => { const b = back[back.length - 1]; setBack((x) => x.slice(0, -1)); setView(b); }} aria-label="Back">‹</button>
        <label className="search inpanel"><SearchIcon /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the web or enter an address" aria-label="Search or address" /></label>
      </form>
      <div className="browser-body">
        {busy ? <p className="empty-hint">Loading…</p> : null}
        {!busy && !view ? <p className="empty-hint">Search the web, or open a page to read it here as text.</p> : null}
        {!busy && view?.kind === "error" ? <p className="notice err">{view.text}</p> : null}
        {!busy && view?.kind === "hits" ? view.hits.map((h) => (
          <button type="button" key={h.url} className="hit" onClick={() => { setQ(h.url); void go(h.url); }}>
            <b>{h.title}</b><span className="hit-url">{h.url}</span><small>{h.snippet}</small>
          </button>
        )) : null}
        {!busy && view?.kind === "page" ? (
          <article className="reader">
            <header><b>{view.page.title || view.page.url}</b><button type="button" className="link" onClick={() => void bridge.web.openExternal(view.page.url)}>Open in your browser</button></header>
            <small className="hit-url">{view.page.url}{view.page.truncated ? " · cut off by the size limit" : ""}</small>
            <pre>{view.page.text}</pre>
          </article>
        ) : null}
      </div>
    </div>
  );
}
