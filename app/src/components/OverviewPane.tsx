import { useState, type ReactNode } from "react";
import type { McpManager } from "../agent/mcp";
import type { CodeSession } from "../agent/sessions";
import type { SystemInfo } from "../bridge";
import { fmtBytes } from "../store";
import { ChevronIcon, FolderIcon, GaugeIcon } from "./icons";

interface Props {
  session: CodeSession;
  system: SystemInfo | undefined;
  modelName: string;
  pinned: string | undefined;
  contextMax: number;
  permission: string;
  mcp: McpManager;
  onManageTools: () => void;
  onOpenFile: (path: string) => void;
}

const parent = (root: string): string => { const parts = root.split(/[\\/]/).filter(Boolean); return parts.length > 1 ? `…/${parts[parts.length - 2]}` : root; };
const name = (root: string): string => root.split(/[\\/]/).filter(Boolean).pop() ?? root;

function Fold({ title, right, children }: { title: string; right?: ReactNode; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="ov-sec">
      <button type="button" className="ov-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className={`ov-caret${open ? " open" : ""}`}><ChevronIcon /></span><b>{title}</b><span className="ov-right">{right}</span>
      </button>
      {open ? <div className="ov-body">{children}</div> : null}
    </section>
  );
}

/** The session at a glance: where it runs, how full the model's context is, which tool servers are connected, and the files it changed. */
export function OverviewPane({ session, system, modelName, pinned, contextMax, permission, mcp, onManageTools, onOpenFile }: Props) {
  const last = [...session.turns].reverse().find((t) => t.context);
  const used = last?.context ?? 0;
  const pct = contextMax > 0 ? Math.min(100, Math.round((used / contextMax) * 100)) : 0;
  const servers = mcp.list();
  const broken = servers.filter((c) => c.status === "error").length;
  const ready = servers.filter((c) => c.status === "ready").length;
  const outputs = [...new Map(session.turns.flatMap((t) => t.diffs ?? []).map((d) => [d.path, d])).values()];

  return (
    <div className="overview">
      <div className="ov-group">
        <h4>Environment</h4>
        <div className="ov-row"><GaugeIcon /><span>This computer</span><small>{system ? `${fmtBytes(system.ramBytes)} · ${system.cores} cores` : ""}</small></div>
        <div className="ov-row" title={session.root}><FolderIcon /><span>{name(session.root)}</span><small>{parent(session.root)}</small></div>
        <div className="ov-row plain"><span className="ov-dim">Model</span><span>{modelName}</span>{pinned ? <small>{pinned}</small> : null}</div>
        <div className="ov-row plain"><span className="ov-dim">Permissions</span><span>{permission}</span></div>
      </div>

      <Fold title="Context" right={<><span>{pct}%</span><span className="ov-bar"><i style={{ width: `${pct}%` }} /></span></>}>
        <p>{used ? `${used.toLocaleString()} of ${contextMax.toLocaleString()} tokens in the model's context after the latest step.` : "No task has run yet in this session."}</p>
        <p className="ov-dim">Each task starts with a fresh context. When it fills up the agent has less room for files, so keep tasks focused.</p>
      </Fold>

      <Fold title="Connectors" right={<span className={broken ? "bad" : ""}>{broken ? `${broken} to reconnect` : ready ? `${ready} connected` : "none"}</span>}>
        {servers.length === 0 ? <p className="ov-dim">No tool servers yet.</p> : servers.map((c) => (
          <div key={c.config.id} className="ov-row plain">
            <i className={`dot ${c.status === "ready" ? "done" : c.status === "error" ? "failed" : c.status === "starting" ? "running" : "idle"}`} />
            <span>{c.config.id}</span>
            <small className={c.status === "error" ? "bad" : ""}>{c.status === "ready" ? `${c.tools.length} tools` : c.status === "error" ? c.error : c.status}</small>
            {c.status === "error" ? <button type="button" className="link" onClick={() => void mcp.restart(c.config.id, session.root)}>Reconnect</button> : null}
          </div>
        ))}
        <button type="button" className="link" onClick={onManageTools}>Manage connectors…</button>
      </Fold>

      <div className="ov-group">
        <h4>Outputs</h4>
        {outputs.length === 0 ? <p className="ov-dim">Files the agent changes will be listed here.</p> : outputs.map((d) => (
          <button type="button" key={d.path} className="ov-row file" onClick={() => onOpenFile(d.path)}><span className="ov-file">{d.path}</span><small><i className="add">+{d.additions}</i> <i className="del">−{d.deletions}</i></small></button>
        ))}
      </div>
    </div>
  );
}
