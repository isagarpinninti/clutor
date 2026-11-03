import type { Conversation } from "../store";
import { groupByProject, sessionDot, type CodeSession } from "../agent/sessions";
import { useEffect, useRef, useState } from "react";
import { ClockIcon, GaugeIcon, GearIcon, PlusIcon, SearchIcon, SidebarIcon, TrashIcon } from "./icons";
import { Logo } from "./Logo";

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
  onCollapse: () => void;
  collapsed: boolean;
  activeJobs: number;
  onClose: () => void;
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

export function Sidebar({ conversations, activeId, open, onSelect, onNew, mode, codeSessions, codeId, onSelectCode, onNewCode, onDeleteCode, onDelete, onSettings, onJobs, onUsage, onCollapse, collapsed, activeJobs, onClose }: Props) {
  const [q, setQ] = useState("");
  const searchBox = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const k = (e: KeyboardEvent): void => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); searchBox.current?.focus(); searchBox.current?.select(); } };
    document.addEventListener("keydown", k);
    return () => document.removeEventListener("keydown", k);
  }, []);
  const needle = q.trim().toLowerCase();
  const sessionsShown = needle ? codeSessions.filter((c) => c.title.toLowerCase().includes(needle) || c.root.toLowerCase().includes(needle)) : codeSessions;
  const shown = q.trim() ? conversations.filter((c) => c.title.toLowerCase().includes(q.trim().toLowerCase())) : conversations;
  return (
    <>
      <div className={`scrim${open ? " show" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar${open ? " open" : ""}${collapsed ? " collapsed" : ""}`} aria-label="Conversations">
        <div className="side-top">
          <div className="brand"><Logo size={22} /><span>Clutor</span></div>
          <button type="button" className="icon-btn" onClick={onCollapse} aria-label="Hide sidebar" title="Hide sidebar"><SidebarIcon /></button>
        </div>
        <div className="side-nav">
          <button type="button" className="side-btn strong" onClick={mode === "code" ? onNewCode : onNew}><PlusIcon /> {mode === "code" ? "New session" : "New chat"}</button>
          <label className="search"><SearchIcon /><input ref={searchBox} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search  ⌘K" aria-label="Search conversations" /></label>
        </div>
        {mode === "code" ? (
          <nav className="history">
            {sessionsShown.length === 0 ? <p className="empty-hint">{q ? "Nothing matches." : "Sessions on your projects appear here."}</p> : null}
            {groupByProject(sessionsShown).map((g) => (
              <section key={g.root}>
                <h3 title={g.root}>{g.name}</h3>
                {g.sessions.map((c) => (
                  <div key={c.id} className={`conv${c.id === codeId ? " active" : ""}`}>
                    <button type="button" className="conv-main" onClick={() => onSelectCode(c.id)}><i className={`dot ${sessionDot(c)}`} aria-label={sessionDot(c)} />{c.title}</button>
                    <button type="button" className="icon-btn conv-del" onClick={() => onDeleteCode(c.id)} aria-label={`Delete ${c.title}`}><TrashIcon /></button>
                  </div>
                ))}
              </section>
            ))}
          </nav>
        ) : null}
        <nav className="history" hidden={mode === "code"}>
          {shown.length === 0 ? <p className="empty-hint">{q ? "Nothing matches." : "Your conversations stay on this device."}</p> : null}
          {group(shown).map(([label, list]) => (
            <section key={label}>
              <h3>{label}</h3>
              {list.map((c) => (
                <div key={c.id} className={`conv${c.id === activeId ? " active" : ""}`}>
                  <button type="button" className="conv-main" onClick={() => onSelect(c.id)}>{c.title}</button>
                  <button type="button" className="icon-btn conv-del" onClick={() => onDelete(c.id)} aria-label={`Delete ${c.title}`}><TrashIcon /></button>
                </div>
              ))}
            </section>
          ))}
        </nav>
        <div className="side-foot">
          <button type="button" className="side-btn" onClick={onJobs}><ClockIcon /> Background jobs{activeJobs > 0 ? <span className="count">{activeJobs}</span> : null}</button>
          <button type="button" className="side-btn" onClick={onUsage}><GaugeIcon /> Usage</button>
          <button type="button" className="side-btn" onClick={onSettings}><GearIcon /> Settings</button>
        </div>
      </aside>
    </>
  );
}
