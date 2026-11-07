import type { Turn } from "../agent/sessions";
import { currentStage, type Stage } from "../agent/sessions";
import { Logo } from "./Logo";
import { CheckIcon } from "./icons";

const STAGES: Array<[Stage, string]> = [["planning", "Plan"], ["exploring", "Explore"], ["writing", "Write"], ["checking", "Check"]];

/**
 * Where a coding task is. When the agent wrote a plan, the plan is shown as a checklist that ticks off as it works; when it did not, the stages (plan, explore, write, check)
 * are worked out from what the agent is actually doing, with a line saying what it is doing right now.
 */
export function RunProgress({ turn }: { turn: Turn }) {
  const running = turn.status === "running";
  const plan = turn.plan;
  if (plan?.length) {
    const done = plan.filter((s) => s.state === "done").length;
    return (
      <div className="plan" role="group" aria-label="Plan">
        <div className="plan-head"><span>Plan</span><small>{done} of {plan.length}{running ? "" : done === plan.length ? " · complete" : ""}</small></div>
        <ol>
          {plan.map((s, i) => (
            <li key={i} className={s.state === "active" ? (running ? "active" : "todo") : s.state}>
              <span className="plan-mark" aria-hidden="true">{s.state === "done" ? <CheckIcon /> : s.state === "active" && running ? <Logo size={16} state="thinking" /> : null}</span>
              <span className="plan-text">{s.text}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }
  if (!running) return null;
  const { stage, label, seen } = currentStage(turn);
  const at = STAGES.findIndex(([k]) => k === stage);
  return (
    <div className="stages" role="status" aria-live="polite">
      <ol className="stage-row">
        {STAGES.map(([k, name], i) => {
          const state = k === stage ? "active" : seen.includes(k) || i < at ? "done" : "todo";
          return <li key={k} className={state}><span className="stage-dot" aria-hidden="true">{state === "done" ? <CheckIcon /> : null}</span>{name}</li>;
        })}
      </ol>
      <div className="stage-now"><Logo size={18} state="thinking" /><span className="shimmer">{label}…</span></div>
    </div>
  );
}
