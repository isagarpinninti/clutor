import { useEffect, useState } from "react";
import type { PlanStep } from "../agent/sessions";
import { Logo } from "./Logo";
import { CheckIcon } from "./icons";

/** 12s, 1m 05s, 1h 02m: how long something has been going. */
export function fmtElapsed(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** Seconds since `key` last changed (the step, or what the model is doing, changed), ticking while `running`. */
export function useSince(key: string, running: boolean): number {
  const [since, setSince] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { setSince(Date.now()); setNow(Date.now()); }, [key]);
  useEffect(() => { if (!running) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [running]);
  return Math.floor((now - since) / 1000);
}

/** Seconds since `from`, ticking while `running`. */
export function useTotal(from: number, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { if (!running) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [running]);
  return Math.floor((now - from) / 1000);
}

/** The agent's plan as a checklist that ticks off as it works; the step being worked on shows how long it has been on it. */
export function PlanList({ plan, running, stepSeconds }: { plan: PlanStep[]; running: boolean; stepSeconds: number }) {
  const done = plan.filter((s) => s.state === "done").length;
  return (
    <div className="plan" role="group" aria-label="Plan">
      <div className="plan-head"><span>Plan</span><small>{done} of {plan.length}{!running && done === plan.length ? " · complete" : ""}</small></div>
      <ol>
        {plan.map((s, i) => (
          <li key={i} className={s.state === "active" ? (running ? "active" : "todo") : s.state}>
            <span className="plan-mark" aria-hidden="true">{s.state === "done" ? <CheckIcon /> : s.state === "active" && running ? <Logo size={16} state="thinking" /> : null}</span>
            <span className="plan-text">{s.text}</span>
            {s.state === "active" && running ? <small className="plan-time">{fmtElapsed(stepSeconds)}</small> : null}
          </li>
        ))}
      </ol>
    </div>
  );
}
