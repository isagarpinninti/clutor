import { useEffect, useRef, useSyncExternalStore } from "react";
import { currentStage, type Turn } from "../agent/sessions";
import { liveStore, parseLive } from "../agent/live";
import { Logo } from "./Logo";
import { useVerb } from "./Thinking";
import { PlanList, fmtElapsed, useSince, useTotal } from "./RunProgress";
import { Patch } from "./Patch";
import { CheckIcon, CloseIcon } from "./icons";

const count = (patch: string): { add: number; del: number } => { let add = 0, del = 0; for (const l of patch.split("\n")) { if (l.startsWith("+") && !l.startsWith("+++")) add++; else if (l.startsWith("-") && !l.startsWith("---")) del++; } return { add, del }; };

/** The reply being written for this task, re-read when it changes (it changes several times a second, so it lives outside the app's state). */
export function useLiveReply(taskId: string | undefined): string | undefined {
  return useSyncExternalStore(liveStore.subscribe, () => (taskId ? liveStore.get(taskId) : undefined), () => undefined);
}

const kb = (n: number): string => (n < 1000 ? `${n} characters` : `${(n / 1000).toFixed(1)}K characters`);

/** One line saying what the agent is doing, with how long it has been at it. Used in the conversation, where the detail would be too much. */
export function WorkingLine({ turn, onOpen }: { turn: Turn; onOpen?: () => void }) {
  const reply = useLiveReply(turn.id);
  const live = reply ? parseLive(reply) : undefined;
  const stage = currentStage(turn);
  const idle = !(live && live.label !== "Thinking") && stage.label === "Planning the change";
  const verb = useVerb(true);
  const label = live && live.label !== "Thinking" ? live.label : idle ? verb : stage.label;
  const total = useTotal(turn.created, true);
  const tokens = live ? Math.round(live.chars / 4) : 0;
  return (
    <div className="working" role="status" aria-live="polite">
      <Logo size={18} state="thinking" />
      <span className="shimmer" key={label}>{label}…</span>
      <small className="stage-time">{fmtElapsed(total)}{tokens > 0 ? ` · ↓ ${tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : tokens} tokens` : ""}</small>
      {onOpen ? <button type="button" className="link-btn" onClick={onOpen}>Progress</button> : null}
    </div>
  );
}

/**
 * The agent's progress, in the panel beside the conversation: the plan as it ticks off, what it is doing right now (and the file it is writing, as it is written), how long
 * each part has taken, and the steps so far.
 */
export function ProgressPane({ turn, onOpenFile }: { turn: Turn | undefined; onOpenFile: (path: string) => void }) {
  const running = turn?.status === "running";
  const reply = useLiveReply(turn?.id);
  const live = reply ? parseLive(reply) : undefined;
  const stage = turn ? currentStage(turn) : undefined;
  const plan = (live?.plan?.length ? live.plan : turn?.plan) ?? [];
  const activeStep = plan.find((s) => s.state === "active")?.text ?? "";
  const nowLabel = live && live.label !== "Thinking" ? live.label : stage?.label ?? "Thinking";
  const inStep = useSince(plan.length ? `plan:${activeStep}` : `now:${nowLabel}`, !!running);
  const total = useTotal(turn?.created ?? Date.now(), !!running);
  const code = useRef<HTMLPreElement>(null);
  const preview = live?.tool === "write_file" || live?.tool === "edit_file" ? live.text : undefined;
  useEffect(() => { const el = code.current; if (el) el.scrollTop = el.scrollHeight; }, [preview]);

  if (!turn) return <p className="empty-hint pane-empty">Run a task and its plan and progress appear here.</p>;
  const tools = turn.blocks.filter((b): b is Extract<Turn["blocks"][number], { kind: "tool" }> => b.kind === "tool");
  const took = turn.ended ? Math.max(0, Math.round((turn.ended - turn.created) / 1000)) : total;

  return (
    <div className="progress-pane">
      <div className="pp-head">
        <p className="pp-task">{turn.task}</p>
        <div className="pp-status">
          {running ? <><Logo size={18} state="thinking" /><span>Working</span></>
            : turn.status === "done" ? <><span className="pp-ok"><CheckIcon /></span><span>Finished</span></>
            : turn.status === "failed" ? <><span className="pp-bad"><CloseIcon /></span><span>Could not finish</span></>
            : turn.status === "stopped" ? <span>Stopped</span> : <span>Interrupted</span>}
          <small>{fmtElapsed(took)}</small>
        </div>
      </div>

      {plan.length ? <PlanList plan={plan} running={!!running} stepSeconds={inStep} /> : null}

      {running ? (
        <section className="pp-now" aria-label="Right now">
          <div className="pp-now-head"><Logo size={16} state="thinking" /><b>{nowLabel}</b><small>{fmtElapsed(inStep)}</small>{live ? <small className="pp-chars">{kb(live.chars)}</small> : null}</div>
          {live?.prose && !preview ? <p className="pp-prose">{live.prose.slice(-600)}</p> : null}
          {preview !== undefined ? <pre ref={code} className="pp-code" aria-label="The file as it is written">{preview.split("\n").slice(-8).join("\n") || " "}</pre> : null}
          {!live ? <p className="pp-prose">Reading the task and deciding what to do first…</p> : null}
        </section>
      ) : null}

      {tools.length ? (
        <section className="pp-steps" aria-label="Steps">
          <h4>Activity</h4>
          <ul>
            {tools.map((b) => {
              const c = b.patch ? count(b.patch) : undefined;
              const detail = b.name === "run_command" ? b.output : b.patch ? undefined : b.status === "error" ? b.output : undefined;
              return (
                <li key={b.id} className={b.status}>
                  <div className="pp-line">
                    <span className="pp-dot" aria-hidden="true">{b.status === "ok" ? <CheckIcon /> : b.status === "error" ? <CloseIcon /> : <Logo size={12} state="thinking" />}</span>
                    <span className="pp-label">{b.label}</span>
                    {c ? <span className="counts"><i className="add">+{c.add}</i> <i className="del">−{c.del}</i></span> : null}
                    {b.path ? <button type="button" className="link-btn" onClick={() => onOpenFile(b.path as string)}>open</button> : null}
                  </div>
                  {b.patch ? <details className="pp-detail"><summary>Changes</summary><Patch patch={b.patch} /></details> : null}
                  {detail ? <details className="pp-detail" open={b.name === "run_command" && b.status === "error"}><summary>{b.name === "run_command" ? "Output" : "Details"}</summary><pre className="pp-out">{detail.slice(0, 2000)}</pre></details> : null}
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
