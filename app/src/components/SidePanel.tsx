import { useEffect, useMemo, useState } from "react";
import type { AgentHost } from "../bridge";
import type { Turn } from "../agent/sessions";
import { reviewable } from "../agent/sessions";
import { Patch } from "./Patch";
import { CloseIcon, SearchIcon } from "./icons";

interface Props {
  host: AgentHost;
  root: string;
  /** The turn whose changes are shown (the latest that has any). */
  turn: Turn | undefined;
  focusFile: string | undefined;
  onApply: () => void;
  onDiscard: () => void;
  onUndo: () => void;
  onClose: () => void;
  busy: boolean;
  error: string | undefined;
}

type Tab = "changes" | "files";

/** The panel beside the conversation: the changes waiting for review, and the project's files. */
export function SidePanel({ host, root, turn, focusFile, onApply, onDiscard, onUndo, onClose, busy, error }: Props) {
  const [tab, setTab] = useState<Tab>("changes");
  const [files, setFiles] = useState<string[]>([]);
  const [q, setQ] = useState("");
  const [viewing, setViewing] = useState<{ path: string; text: string | null; error?: string }>();
  const diffs = turn?.diffs ?? [];
  const review = !!turn && reviewable(turn);

  useEffect(() => { if (root) void host.list(root).then(setFiles).catch(() => setFiles([])); }, [host, root, turn?.applied]);
  const open = (path: string): void => {
    setTab("files");
    void host.read(root, path).then((text) => setViewing({ path, text }), (e: unknown) => setViewing({ path, text: null, error: e instanceof Error ? e.message : String(e) }));
  };
  useEffect(() => { if (focusFile) open(focusFile); }, [focusFile]);
  const shown = useMemo(() => (q.trim() ? files.filter((f) => f.toLowerCase().includes(q.trim().toLowerCase())) : files).slice(0, 400), [files, q]);

  return (
    <aside className="side-panel" aria-label="Changes and files">
      <header>
        <div className="seg" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "changes"} className={tab === "changes" ? "on" : ""} onClick={() => setTab("changes")}>Changes{diffs.length ? ` · ${diffs.length}` : ""}</button>
          <button type="button" role="tab" aria-selected={tab === "files"} className={tab === "files" ? "on" : ""} onClick={() => setTab("files")}>Files</button>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close panel"><CloseIcon /></button>
      </header>
      <div className="panel-body">
        {tab === "changes" ? (
          diffs.length === 0 ? <p className="empty-hint">{turn?.lost ? "The changes from this task were too large to keep after a restart. Run it again." : "No changes yet. When the agent edits files they appear here for review."}</p> : (
            <>
              {diffs.map((d) => (
                <details key={d.path} className="filediff" open={diffs.length <= 3}>
                  <summary className="filehead"><span className={`st ${d.status}`}>{d.status}</span><b>{d.path}</b><span className="counts"><i className="add">+{d.additions}</i> <i className="del">−{d.deletions}</i></span></summary>
                  {d.patch ? <Patch patch={d.patch} /> : <p className="empty-hint">No text changes.</p>}
                </details>
              ))}
            </>
          )
        ) : (
          <>
            <label className="search inpanel"><SearchIcon /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter files" aria-label="Filter files" /></label>
            {viewing ? (
              <div className="viewer">
                <div className="viewer-head"><b>{viewing.path}</b><button type="button" className="link" onClick={() => setViewing(undefined)}>Back to list</button></div>
                {viewing.text === null ? <p className="empty-hint">{viewing.error ?? "This file does not exist."}</p> : <pre className="code-view">{viewing.text.split("\n").map((l, i) => <div key={i}><i>{i + 1}</i>{l || " "}</div>)}</pre>}
              </div>
            ) : (
              <ul className="file-list">{shown.map((f) => <li key={f}><button type="button" onClick={() => open(f)}>{f}</button></li>)}{shown.length === 0 ? <li className="empty-hint">No files.</li> : null}</ul>
            )}
          </>
        )}
      </div>
      {error ? <div className="notice err" role="alert">{error}</div> : null}
      {tab === "changes" && turn && diffs.length > 0 ? (
        <footer>
          {review ? <><button type="button" className="btn small" onClick={onApply} disabled={busy}>Apply to folder</button><button type="button" className="btn small ghost" onClick={onDiscard} disabled={busy}>Discard</button></>
            : turn.applied ? <><span className="applied-note">Applied to the folder</span><button type="button" className="btn small ghost" onClick={onUndo}>Undo</button></>
            : turn.discarded ? <span className="applied-note">Discarded</span> : null}
        </footer>
      ) : null}
    </aside>
  );
}
