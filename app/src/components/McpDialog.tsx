import { PlugIcon } from "./icons";
import { DialogHeader } from "./DialogHeader";
import { useState } from "react";
import { serverName, splitCommandLine, type McpManager, type McpServerConfig } from "../agent/mcp";

interface Props { servers: McpServerConfig[]; manager: McpManager; onChange: (s: McpServerConfig[]) => void; onClose: () => void }

export function McpPanel({ servers, manager, onChange }: Omit<Props, "onClose">) {
  const [line, setLine] = useState("");
  const [name, setName] = useState("");
  const [env, setEnv] = useState("");
  const status = (id: string) => manager.list().find((c) => c.config.id === id);

  const add = (): void => {
    const parts = splitCommandLine(line);
    if (parts.length === 0) return;
    const id = serverName(name || parts.filter((p) => !p.startsWith("-"))[1] || parts[0] as string);
    const vars: Record<string, string> = {};
    for (const l of env.split("\n")) { const i = l.indexOf("="); if (i > 0) vars[l.slice(0, i).trim()] = l.slice(i + 1).trim(); }
    onChange([...servers.filter((s) => s.id !== id), { id, command: parts[0] as string, args: parts.slice(1), env: vars, enabled: true }]);
    setLine(""); setName(""); setEnv("");
  };

  return (
    <>
        <p className="empty-hint">An MCP server is a program that gives the agent extra tools, such as web search or a database. It runs on this computer with your permissions: add only servers you trust. The agent asks before calling any tool the server does not mark read-only.</p>
        <div className="list">
          {servers.map((s) => {
            const c = status(s.id);
            return (
              <div key={s.id} className="row static">
                <span className="row-main"><b>{s.id}</b><small>{s.command} {s.args.join(" ")}</small>
                  <small className={c?.status === "error" ? "err" : ""}>{!s.enabled ? "off" : c?.status === "ready" ? `connected · ${c.tools.length} tool${c.tools.length === 1 ? "" : "s"}` : c?.status === "error" ? c.error : "starting…"}</small></span>
                <button type="button" className="btn small ghost" onClick={() => onChange(servers.map((x) => (x.id === s.id ? { ...x, enabled: !x.enabled } : x)))}>{s.enabled ? "Turn off" : "Turn on"}</button>
                <button type="button" className="btn small ghost" onClick={() => onChange(servers.filter((x) => x.id !== s.id))}>Remove</button>
              </div>
            );
          })}
          {servers.length === 0 ? <p className="empty-hint">No servers yet.</p> : null}
        </div>
        <div className="form">
          <label>Command<input value={line} onChange={(e) => setLine(e.target.value)} placeholder="npx -y @modelcontextprotocol/server-filesystem /path/to/folder" /></label>
          <label>Name (optional)<input value={name} onChange={(e) => setName(e.target.value)} placeholder="files" /></label>
          <label>Environment (optional, one NAME=value per line)<textarea rows={2} value={env} onChange={(e) => setEnv(e.target.value)} /></label>
          <button type="button" className="btn" onClick={add} disabled={!line.trim()}>Add server</button>
        </div>
    </>
  );
}

export function McpDialog({ servers, manager, onChange, onClose }: Props) {
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Tools">
        <DialogHeader icon={<PlugIcon />} title="Connectors" subtitle="Tool servers the agent can use" onClose={onClose} />
        <McpPanel servers={servers} manager={manager} onChange={onChange} />
      </div>
    </div>
  );
}
