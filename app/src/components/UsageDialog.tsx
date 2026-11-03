import { GaugeIcon } from "./icons";
import { DialogHeader } from "./DialogHeader";
import type { EngineState } from "../engine";
import { fmtBytes } from "../store";
import { contextPercent, type ChatUsage } from "../usage";

export interface UsageData {
  state: EngineState;
  ramBytes: number | undefined;
  usage: ChatUsage;
  agent: { runs: number; steps: number; tokens: number };
  jobs: { active: number; done: number };
  /** The model's context after the agent's latest step, when a coding task has run. */
  agentContext?: { used: number; max: number };
}
interface Props extends UsageData { onClose: () => void }

const n = (v: number): string => v.toLocaleString();

function Stat({ label, value, unit, hint }: { label: string; value: string; unit?: string; hint?: string }) {
  return <div className="stat"><small>{label}</small><b>{value}{unit ? <span>{unit}</span> : null}</b>{hint ? <em>{hint}</em> : null}</div>;
}
function Table({ rows }: { rows: Array<[string, string]> }) {
  return <dl className="details">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>;
}

export function UsagePanel({ state, ramBytes, usage, agent, jobs, agentContext }: UsageData) {
  const agentPct = agentContext && agentContext.max > 0 ? Math.min(100, Math.round((agentContext.used / agentContext.max) * 100)) : 0;
  const pct = contextPercent(usage);
  const tone = pct >= 90 ? "high" : pct >= 70 ? "mid" : "";
  const share = state.model && ramBytes ? Math.round((state.model.bytes / ramBytes) * 100) : undefined;
  return (
    <div className="usage">
      <p className="lead">Everything runs on this device, so there is no cost and no quota. These are the numbers the engine measured.</p>

      <section className={`ctx-card ${tone}`}>
        <div className="ctx-top"><div><small>Context window</small><b>{pct}%<span> used</span></b></div><div className="ctx-nums">{n(usage.contextUsed)} <span>of {n(usage.contextMax)} tokens</span></div></div>
        <div className="meter big" role="img" aria-label={`Context ${pct}% full`}><i style={{ width: `${Math.max(pct, usage.contextUsed ? 2 : 0)}%` }} /></div>
        <p>{pct >= 90 ? "Nearly full: start a new chat so the model keeps room to answer." : pct >= 70 ? "Getting full. Long conversations make the model slower and forgetful." : "Plenty of room left in this conversation."}</p>
      </section>

      {agentContext ? (
        <section className="ctx-card slim">
          <div className="ctx-top"><div><small>Agent context (latest task)</small><b>{agentPct}%<span> used</span></b></div><div className="ctx-nums">{n(agentContext.used)} <span>of {n(agentContext.max)} tokens</span></div></div>
          <div className="meter big" role="img" aria-label={`Agent context ${agentPct}% full`}><i style={{ width: `${Math.max(agentPct, agentContext.used ? 2 : 0)}%` }} /></div>
        </section>
      ) : null}

      <div className="stats">
        <Stat label="Replies" value={n(usage.replies)} />
        <Stat label="Tokens written" value={n(usage.tokens)} />
        <Stat label="Speed" value={usage.tps ? usage.tps.toFixed(1) : "-"} unit={usage.tps ? " tok/s" : ""} hint={usage.tps ? "measured" : "no reply yet"} />
        <Stat label="Agent runs" value={n(agent.runs)} hint={agent.runs ? `${n(agent.steps)} steps` : undefined} />
      </div>

      <h4 className="ssub">Model and memory</h4>
      <Table rows={[
        ["Model", state.model?.name ?? "none loaded"],
        ...(state.model ? [["File size", `${fmtBytes(state.model.bytes)}${share !== undefined ? ` · ${share}% of memory` : ""}`] as [string, string]] : []),
        ...(state.ready ? [["In memory", state.ready.locked ? "pinned" : state.unpinned ? "not pinned (large model)" : "not pinned"] as [string, string]] : []),
        ...(state.loadMs ? [["Load time", `${(state.loadMs / 1000).toFixed(1)} s`] as [string, string]] : []),
      ]} />

      <h4 className="ssub">This session</h4>
      <Table rows={[["Agent tokens written", n(agent.tokens)], ["Background jobs", `${jobs.active} active · ${jobs.done} finished`]]} />
    </div>
  );
}

export function UsageDialog({ onClose, ...data }: Props) {
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog wide" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Usage">
        <DialogHeader icon={<GaugeIcon />} title="Usage" subtitle="What this conversation and the model are using" onClose={onClose} />
        <div className="dbody"><UsagePanel {...data} /></div>
      </div>
    </div>
  );
}
