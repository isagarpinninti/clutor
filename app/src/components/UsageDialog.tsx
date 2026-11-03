import type { EngineState } from "../engine";
import { fmtBytes } from "../store";
import { contextPercent, type ChatUsage } from "../usage";
import { CloseIcon } from "./icons";

interface Props {
  state: EngineState;
  ramBytes: number | undefined;
  usage: ChatUsage;
  agent: { runs: number; steps: number; tokens: number };
  jobs: { active: number; done: number };
  onClose: () => void;
}

const n = (v: number): string => v.toLocaleString();

export function UsageDialog({ state, ramBytes, usage, agent, jobs, onClose }: Props) {
  const pct = contextPercent(usage);
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Usage">
        <header><h2>Usage</h2><button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon /></button></header>
        <p className="lead">Everything runs on this device, so there is no cost and no quota. These are the numbers the engine measured.</p>

        <section className="panel">
          <h3>This conversation</h3>
          <div className="meter" role="img" aria-label={`Context ${pct}% full`}><i style={{ width: `${pct}%` }} /></div>
          <div className="kv"><span>Context</span><b>{n(usage.contextUsed)} of {n(usage.contextMax)} tokens ({pct}%)</b></div>
          <div className="kv"><span>Replies</span><b>{n(usage.replies)}</b></div>
          <div className="kv"><span>Tokens written</span><b>{n(usage.tokens)}</b></div>
          <div className="kv"><span>Speed (measured)</span><b>{usage.tps ? `${usage.tps.toFixed(1)} tokens/s` : "no reply yet"}</b></div>
        </section>

        <section className="panel">
          <h3>Agent (this session)</h3>
          <div className="kv"><span>Runs</span><b>{n(agent.runs)}</b></div>
          <div className="kv"><span>Steps</span><b>{n(agent.steps)}</b></div>
          <div className="kv"><span>Tokens written</span><b>{n(agent.tokens)}</b></div>
        </section>

        <section className="panel">
          <h3>Model and memory</h3>
          <div className="kv"><span>Model</span><b>{state.model?.name ?? "none loaded"}</b></div>
          {state.model ? <div className="kv"><span>File size</span><b>{fmtBytes(state.model.bytes)}{ramBytes ? ` · ${Math.round((state.model.bytes / ramBytes) * 100)}% of this computer's memory` : ""}</b></div> : null}
          {state.ready ? <div className="kv"><span>In memory</span><b>{state.ready.locked ? "pinned" : state.unpinned ? "not pinned (large model)" : "not pinned"}</b></div> : null}
          {state.loadMs ? <div className="kv"><span>Load time</span><b>{(state.loadMs / 1000).toFixed(1)} s</b></div> : null}
          <div className="kv"><span>Background jobs</span><b>{jobs.active} active · {jobs.done} finished</b></div>
        </section>
      </div>
    </div>
  );
}
