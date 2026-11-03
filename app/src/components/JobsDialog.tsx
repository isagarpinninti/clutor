import { isActive, type Job } from "../jobs";
import type { Conversation } from "../store";
import { fmtDuration, type JobAction } from "./JobStrip";

interface Props {
  jobs: Job[];
  conversations: Conversation[];
  position: (id: string) => number;
  onAction: (id: string, a: JobAction) => void;
  onOpen: (conversationId: string) => void;
  onRemove: (id: string) => void;
  onClose: () => void;
}

const LABEL: Record<Job["status"], string> = { queued: "Queued", running: "Running", paused: "Paused", done: "Done", failed: "Failed", cancelled: "Cancelled" };

export function JobsDialog({ jobs, conversations, position, onAction, onOpen, onRemove, onClose }: Props) {
  // Running first, then paused, then the queue in the order it will run; finished jobs last, newest first.
  const rank: Record<Job["status"], number> = { running: 0, paused: 1, queued: 2, done: 3, failed: 3, cancelled: 3 };
  const shown = [...jobs].sort((a, b) => rank[a.status] - rank[b.status] || (isActive(a) ? a.created - b.created : b.created - a.created));
  const exists = (id: string): boolean => conversations.some((c) => c.id === id);
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Background jobs">
        <header><h2>Background jobs</h2><button type="button" className="icon-btn" onClick={onClose} aria-label="Close">✕</button></header>
        <p className="fineprint left">A job is a question the model answers while you do something else. One runs at a time; the rest wait their turn. Jobs are kept if you close the app.</p>
        <div className="list">
          {shown.length === 0 ? <p className="empty-hint">No jobs yet. Use the clock button next to Send to run a question in the background.</p> : null}
          {shown.map((j) => (
            <div key={j.id} className={`row static job ${j.status}`}>
              <span className="jdot" aria-hidden="true" />
              <span className="row-main">
                <b>{j.title}</b>
                <small>
                  {LABEL[j.status]}{j.status === "queued" ? ` · position ${position(j.id)}` : ""} · {j.tokens} tokens
                  {j.elapsedMs > 0 ? ` · ${fmtDuration(j.elapsedMs)}` : ""}{j.tps ? ` · ${j.tps.toFixed(1)} tok/s` : ""}
                </small>
                {j.error ? <small className="err">{j.error}</small> : null}
                {j.status === "running" ? <span className="bar"><i style={{ width: `${Math.min(100, Math.round((j.tokens / Math.max(1, j.maxTokens)) * 100))}%` }} /></span> : null}
              </span>
              <span className="job-actions">
                {exists(j.conversationId) ? <button type="button" className="link-btn" onClick={() => { onOpen(j.conversationId); onClose(); }}>Open</button> : null}
                {j.status === "running" ? <button type="button" className="link-btn" onClick={() => onAction(j.id, "pause")}>Pause</button> : null}
                {j.status === "paused" ? <button type="button" className="link-btn" onClick={() => onAction(j.id, "resume")}>Resume</button> : null}
                {j.status === "failed" || j.status === "cancelled" ? <button type="button" className="link-btn" onClick={() => onAction(j.id, "retry")}>Retry</button> : null}
                {isActive(j) ? <button type="button" className="link-btn" onClick={() => onAction(j.id, "cancel")}>Cancel</button> : <button type="button" className="link-btn" onClick={() => onRemove(j.id)}>Remove</button>}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
