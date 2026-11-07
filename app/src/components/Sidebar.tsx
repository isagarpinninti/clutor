import { useEffect, useRef, useState } from "react";
import { groupByProject, sessionDot, type CodeSession } from "../agent/sessions";
import type { Conversation } from "../store";
import { Logo } from "./Logo";
import { ChevronIcon, ClockIcon, GaugeIcon, GearIcon, PlusIcon, SearchIcon, TrashIcon } from "./icons";

interface Props {
  conversations: Conversation[];
  activeId: string;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  mode: "chat" | "code";
  codeSessions: CodeSession[];
  codeId: string;
  onSelectCode: (id: string) => void;
  onNewCode: () => void;
  onDeleteCode: (id: string) => void;
  onDelete: (id: string) => void;
  onSettings: () => void;
  onJobs: () => void;
  onUsage: () => void;
  collapsed: boolean;
  activeJobs: number;
  onClose: () => void;
  ramLabel: string;
  userName?: string | undefined;
}

function group(conversations: Conversation[]): Array<[string, Conversation[]]> {
  const now = new Date(), day = 86_400_000;
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const buckets: Array<[string, Conversation[]]> = [["Today", []], ["Yesterday", []], ["Previous 7 days", []], ["Older", []]];
  for (const c of [...conversations].sort((a, b) => b.updated - a.updated)) {
    const i = c.updated >= start ? 0 : c.updated >= start - day ? 1 : c.updated >= start - 7 * day ? 2 : 3;
    (buckets[i] as [string, Conversation[]])[1].push(c);
  }
  return buckets.filter(([, list]) => list.length > 0);
}

/** A heading that folds its list away. */
function Heading({ label, title, folded, onToggle }: { label: string; title?: string; folded: boolean; onToggle: () => void }) {
  return <button type="button" className="group-head" onClick={onToggle} aria-expanded={!folded} title={title}><span>{label}</span><span className={`gh-caret${folded ? " folded" : ""}`}><ChevronIcon /></span></button>;
}

export function Sidebar(p: Props) {
  const [q, setQ] = useState("");
  const [folded, setFolded] = useState<Set<string>>(new Set());
  const searchBox = useRef<HTMLInputElement>(null);
  const [searching, setSearching] = useState(false);
  useEffect(() => {
    const k = (e: KeyboardEvent): void => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setSearching(true); setTimeout(() => { searchBox.current?.focus(); searchBox.current?.select(); }, 0); } };
    document.addEventListener("keydown", k);
    return () => document.removeEventListener("keydown", k);
  }, []);
  const toggle = (k: string): void => setFolded((f) => { const n = new Set(f); if (n.has(k)) n.delete(k); else n.add(k); return n; });
  const needle = q.trim().toLowerCase();
  const sessionsShown = needle ? p.codeSessions.filter((c) => c.title.toLowerCase().includes(needle) || c.root.toLowerCase().includes(needle)) : p.codeSessions;
  const shown = needle ? p.conversations.filter((c) => c.title.toLowerCase().includes(needle)) : p.conversations;

  return (
    <>
      <div className={`scrim${p.open ? " show" : ""}`} onClick={p.onClose} aria-hidden="true" />
      <aside className={`sidebar${p.open ? " open" : ""}${p.collapsed ? " collapsed" : ""}`} aria-label="Sidebar">
        <div className="side-brand"><Logo size={26} /><span>Clutor</span></div>
        <div className="side-nav">
          <button type="button" className="nav-item primary" onClick={p.mode === "code" ? p.onNewCode : p.onNew}><PlusIcon /><span>{p.mode === "code" ? "New session" : "New chat"}</span></button>
          {searching || q ? (
            <label className="nav-item searchrow"><SearchIcon /><input ref={searchBox} value={q} onChange={(e) => setQ(e.target.value)} onBlur={() => { if (!q) setSearching(false); }} placeholder="Search" aria-label="Search" /></label>
          ) : <button type="button" className="nav-item" onClick={() => { setSearching(true); setTimeout(() => searchBox.current?.focus(), 0); }}><SearchIcon /><span>Search</span><kbd>⌘K</kbd></button>}
          <button type="button" className="nav-item" onClick={p.onJobs}><ClockIcon /><span>Background jobs</span>{p.activeJobs > 0 ? <span className="count">{p.activeJobs}</span> : null}</button>
        </div>

        {p.mode === "code" ? (
          <nav className="history">
            {sessionsShown.length === 0 ? <p className="empty-hint">{q ? "Nothing matches." : "Sessions on your projects appear here."}</p> : null}
            {groupByProject(sessionsShown).map((g) => (
              <section key={g.root}>
                <Heading label={g.name} title={g.root} folded={folded.has(g.root)} onToggle={() => toggle(g.root)} />
                {folded.has(g.root) ? null : g.sessions.map((c) => (
                  <div key={c.id} className={`conv${c.id === p.codeId ? " active" : ""}`}>
                    <button type="button" className="conv-main" onClick={() => p.onSelectCode(c.id)}>{["running", "review", "failed"].includes(sessionDot(c)) ? <i className={`dot ${sessionDot(c)}`} aria-label={sessionDot(c)} /> : null}<span>{c.title}</span></button>
                    <button type="button" className="icon-btn conv-del tip" data-tip="Delete" onClick={() => p.onDeleteCode(c.id)} aria-label={`Delete ${c.title}`}><TrashIcon /></button>
                  </div>
                ))}
              </section>
            ))}
          </nav>
        ) : (
          <nav className="history">
            {shown.length === 0 ? <p className="empty-hint">{q ? "Nothing matches." : "Your conversations stay on this device."}</p> : null}
            {group(shown).map(([label, list]) => (
              <section key={label}>
                <Heading label={label} folded={folded.has(label)} onToggle={() => toggle(label)} />
                {folded.has(label) ? null : list.map((c) => (
                  <div key={c.id} className={`conv${c.id === p.activeId ? " active" : ""}`}>
                    <button type="button" className="conv-main" onClick={() => p.onSelect(c.id)}><span>{c.title}</span></button>
                    <button type="button" className="icon-btn conv-del tip" data-tip="Delete" onClick={() => p.onDelete(c.id)} aria-label={`Delete ${c.title}`}><TrashIcon /></button>
                  </div>
                ))}
              </section>
            ))}
          </nav>
        )}

        <div className="side-foot">
          <div className="account"><span className="avatar" aria-hidden="true">{(p.userName?.trim()[0] ?? "C").toUpperCase()}</span><span className="acc-text"><b>{p.userName || "Clutor"}</b><small>{p.ramLabel}</small></span></div>
          <button type="button" className="icon-btn tip" data-tip="Usage" onClick={p.onUsage} aria-label="Usage"><GaugeIcon /></button>
          <button type="button" className="icon-btn tip" data-tip="Settings" onClick={p.onSettings} aria-label="Settings"><GearIcon /></button>
        </div>
      </aside>
    </>
  );
}
