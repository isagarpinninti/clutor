import { useEffect, useState } from "react";
import type { Turn } from "../agent/sessions";
import { currentStage, type Stage } from "../agent/sessions";
import { Logo } from "./Logo";
import { CheckIcon } from "./icons";

/** 12s, 1m 05s, 1h 02m: how long something has been going. */
export function fmtElapsed(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** Seconds since `key` last changed (the stage, or the step of the plan, changed), ticking while the task runs. */
function useSince(key: string, running: boolean): number {
  const [since, setSince] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { setSince(Date.now()); setNow(Date.now()); }, [key]);
  useEffect(() => { if (!running) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [running]);
  return Math.floor((now - since) / 1000);
}

/** Seconds since the task started. */
function useTotal(from: number, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { if (!running) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [running]);
  return Math.floor((now - from) / 1000);
}

const STAGES: Array<[Stage, string]> = [["planning", "Plan"], ["exploring", "Explore"], ["writing", "Write"], ["checking", "Check"]];

/**
 * Where a coding task is. When the agent wrote a plan, the plan is shown as a checklist that ticks off as it works; when it did not, the stages (plan, explore, write, check)
 * are worked out from what the agent is actually doing, with a line saying what it is doing right now.
 */
export function RunProgress({ turn }: { turn: Turn }) {
  const running = turn.status === "running";
  const plan = turn.plan;
  const activeStep = plan?.find((s) => s.state === "active")?.text ?? "";
  const stageNow = currentStage(turn);
  const inStep = useSince(plan?.length ? `plan:${activeStep}` : `stage:${stageNow.stage}`, running);
  const total = useTotal(turn.created, running);
  if (plan?.length) {
    const done = plan.filter((s) => s.state === "done").length;
    return (
      <div className="plan" role="group" aria-label="Plan">
        <div className="plan-head"><span>Plan</span><small>{done} of {plan.length}{running ? ` · ${fmtElapsed(total)}` : done === plan.length ? " · complete" : ""}</small></div>
        <ol>
          {plan.map((s, i) => (
            <li key={i} className={s.state === "active" ? (running ? "active" : "todo") : s.state}>
              <span className="plan-mark" aria-hidden="true">{s.state === "done" ? <CheckIcon /> : s.state === "active" && running ? <Logo size={16} state="thinking" /> : null}</span>
              <span className="plan-text">{s.text}</span>
              {s.state === "active" && running ? <small className="plan-time">{fmtElapsed(inStep)}</small> : null}
            </li>
          ))}
        </ol>
      </div>
    );
  }
  if (!running) return null;
  const { stage, label, seen } = stageNow;
  const at = STAGES.findIndex(([k]) => k === stage);
  return (
    <div className="stages" role="status" aria-live="polite">
      <ol className="stage-row">
        {STAGES.map(([k, name], i) => {
          const state = k === stage ? "active" : seen.includes(k) || i < at ? "done" : "todo";
          return <li key={k} className={state}><span className="stage-dot" aria-hidden="true">{state === "done" ? <CheckIcon /> : null}</span>{name}</li>;
        })}
      </ol>
      <div className="stage-now"><Logo size={18} state="thinking" /><span className="shimmer">{label}…</span><small className="stage-time" title="Time in this stage">{fmtElapsed(inStep)}</small><small className="stage-total" title="Time since the task started">total {fmtElapsed(total)}</small></div>
    </div>
  );
}
