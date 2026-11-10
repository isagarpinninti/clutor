import { etaMs, type Job } from "../jobs";

export type JobAction = "pause" | "resume" | "cancel" | "retry";

export function fmtDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min${s % 60 >= 30 ? " 30 s" : ""}`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}

const WHY: Record<NonNullable<Job["pausedBy"]>, string> = { user: "by you", interrupted: "the app was closed or the model changed", chat: "while you chatted" };

/** The live status of a background job, shown with the reply it is writing: where it is in line, how far it got, how fast, and what you can do. */
export function JobStrip({ job, position, onAction }: { job: Job; position: number; onAction: (id: string, a: JobAction) => void }) {
  if (job.status === "done") return null;
  const eta = etaMs(job);
  const pct = Math.min(100, Math.round((job.tokens / Math.max(1, job.maxTokens)) * 100));
  const btn = (a: JobAction, label: string) => <button type="button" className="link-btn" onClick={() => onAction(job.id, a)}>{label}</button>;
  return (
    <div className={`jobstrip ${job.status}`} role="status">
      <div className="jobline">
        <span className="jdot" aria-hidden="true" />
        <b>
          {job.status === "queued" ? `Queued · ${position > 1 ? `${position - 1} ahead` : "next"}` : job.status === "running" ? "Running in the background" : job.status === "paused" ? "Paused" : job.status === "failed" ? "Failed" : "Cancelled"}
        </b>
        <span className="jdetail">
          {job.status === "running" ? `${job.tokens} tokens${job.tps ? ` · ${job.tps.toFixed(1)} tok/s` : ""}${eta !== undefined ? ` · up to ${fmtDuration(eta)} left` : ""}` : null}
          {job.status === "paused" ? `${job.tokens} tokens kept${job.pausedBy ? ` · ${WHY[job.pausedBy]}` : ""}` : null}
          {job.status === "failed" ? job.error : null}
        </span>
        <span className="spacer" />
        {job.status === "running" ? btn("pause", "Pause") : null}
        {job.status === "queued" ? btn("pause", "Hold") : null}
        {job.status === "paused" ? btn("resume", "Resume") : null}
        {job.status === "failed" || job.status === "cancelled" ? btn("retry", "Try again") : null}
        {job.status !== "failed" && job.status !== "cancelled" ? btn("cancel", "Cancel") : null}
      </div>
      {job.status === "running" ? <div className="bar"><i style={{ width: `${pct}%` }} /></div> : null}
    </div>
  );
}
