import { useState } from "react";
import type { Msg } from "../store";
import { Markdown } from "./Markdown";
import { CopyIcon } from "./icons";
import { Logo } from "./Logo";

export function MessageView({ msg, streaming, showReasoning }: { msg: Msg; streaming: boolean; showReasoning: boolean }) {
  const [copied, setCopied] = useState(false);
  if (msg.role === "user") return <div className="msg user"><div className="who">You</div><div className="prompt">{msg.content}</div></div>;
  const s = msg.stats;
  const copy = (): void => { void navigator.clipboard?.writeText(msg.content).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); };
  return (
    <div className="msg assistant">
      <div className="who"><Logo size={18} />Clutor</div>
      <div className="body">
        {msg.reasoning ? (
          <details className="reasoning" open={showReasoning}>
            <summary>{streaming && !msg.content ? "Thinking…" : "Thought process"}</summary>
            <div className="reasoning-text">{msg.reasoning}</div>
          </details>
        ) : null}
        {msg.content ? <Markdown text={msg.content} /> : streaming && !msg.reasoning ? <div className="typing" aria-label="Thinking"><i /><i /><i /></div> : null}
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
