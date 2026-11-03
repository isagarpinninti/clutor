import { useState } from "react";
import type { Msg } from "../store";
import { Markdown } from "./Markdown";
import { CheckIcon, CloseIcon, CopyIcon } from "./icons";
import { Logo } from "./Logo";
import { Spark, Thinking } from "./Thinking";
import { bridge } from "../bridge";
import type { Job } from "../jobs";
import { JobStrip, type JobAction } from "./JobStrip";

export function MessageView({ msg, streaming, showReasoning, job, position, onJobAction }: { msg: Msg; streaming: boolean; showReasoning: boolean; job?: Job | undefined; position?: number; onJobAction?: (id: string, a: JobAction) => void }) {
  const [copied, setCopied] = useState(false);
  if (msg.role === "user") return <div className="msg user"><div className="who">You</div><div className="prompt">{msg.content}</div></div>;
  const s = msg.stats;
  const copy = (): void => { void navigator.clipboard?.writeText(msg.content).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); };
  return (
    <div className="msg assistant">
      <div className="who"><Logo size={22} /><span>Clutor</span></div>
      <div className="body">
        {msg.reasoning ? (
          <details className="reasoning" open={showReasoning}>
            <summary>{streaming && !msg.content ? "Thinking…" : "Thought process"}</summary>
            <div className="reasoning-text">{msg.reasoning}</div>
          </details>
        ) : null}
        {job && onJobAction ? <JobStrip job={job} position={position ?? 0} onAction={onJobAction} /> : null}
        {msg.content ? <Markdown text={msg.content} /> : (streaming || job?.status === "running") && !msg.reasoning ? <Thinking /> : null}
        {msg.web ? (
          <div className={`websearch ${msg.web.state}`} role="status">
            {msg.web.state === "searching" ? <Spark size={12} /> : msg.web.state === "done" ? <CheckIcon /> : <CloseIcon />}
            <span>{msg.web.state === "searching" ? `Searching the web for “${msg.web.query}”…` : msg.web.state === "done" ? `Searched the web${msg.sources?.length ? ` · read ${msg.sources.length} source${msg.sources.length === 1 ? "" : "s"}` : " · nothing useful found"}` : `Could not search the web${msg.web.error ? ` (${msg.web.error})` : ""}; answered without it`}</span>
          </div>
        ) : null}
        {msg.sources?.length ? (
          <div className="sources" aria-label="Sources">
            {msg.sources.map((s, i) => <button type="button" key={s.url} className="source" title={s.url} onClick={() => void bridge.web.openExternal(s.url)}><i>{i + 1}</i><span>{s.title || s.url}</span></button>)}
          </div>
        ) : null}
        {msg.error ? <div className="error">{msg.error}</div> : null}
        {!streaming && (s || msg.interrupted) ? (
          <div className="meta">
            {msg.interrupted ? <span>Stopped</span> : null}
            {s ? <span>{s.tokens} tokens</span> : null}
            {s?.tps ? <span>{s.tps.toFixed(1)} tok/s</span> : null}
            {s?.promptTokens ? <span>context {s.promptTokens}</span> : null}
            {s?.cache ? <span>cache {Math.round(s.cache.hitRate * 100)}% hit · {s.cache.gbRead.toFixed(2)} GB read</span> : null}
            {msg.content ? <button type="button" className="icon-btn" onClick={copy} title="Copy reply" aria-label="Copy reply">{copied ? <span className="copied">Copied</span> : <CopyIcon />}</button> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
